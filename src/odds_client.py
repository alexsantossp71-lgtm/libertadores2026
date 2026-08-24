"""
Cliente de odds 1X2 da Bzzoiro Sports Data (https://sports.bzzoiro.com).

Responsabilidades:
  * Buscar odds decimais 1X2 de uma partida usando a chave ``BSD_API``;
  * Mapear os confrontos da Libertadores 2026 para os ``event_id`` da Bzzoiro
    (por nome dos times e data da partida);
  * Salvar as respostas em ``data/raw/odds_<event_id>.json`` com cache
    (evita chamadas repetidas);
  * Processar para colunas canônicas:
      ``odd_mandante``, ``odd_empate``, ``odd_visitante``,
      ``prob_mandante_impl``, ``prob_empate_impl``, ``prob_visitante_impl``
    (probabilidades implícitas = 1/odd, normalizadas para remover a margem
    da casa de apostas).

Autenticação (https://sports.bzzoiro.com/docs):
    Header ``Authorization: Token SUA_CHAVE``.

Comportamento da API gratuita:
    A chave gratuita retorna apenas o **consenso** multi-bookmaker (uma linha
    por evento × mercado × resultado, com ``bookmaker_slug == "consensus"``).
    Chaves pagas retornam uma linha por casa de apostas — nesse caso o cliente
    calcula a média (e registra as casas consultadas).

Fallback:
    Sem chave, sem conexão ou sem eventos correspondentes, o cliente usa a
    base de exemplo ``data/examples/odds_libertadores_2026.csv`` e documenta
    a cobertura (quais jogos têm odds e quais não têm).
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv é opcional
    pass

from src.generate_example_data import (  # noqa: E402
    EXAMPLE_ODDS_PATH,
    EXAMPLE_PARTIDAS_PATH,
    ODDS_COLUMNS,
    generate_odds,
)

# --------------------------------------------------------------------------- #
# Configurações
# --------------------------------------------------------------------------- #
BASE_URL = "https://sports.bzzoiro.com/api/v2"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "libertadores_odds.csv"
MAPPING_PATH = RAW_DIR / "bzzoiro_event_mapping.json"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = 6 * 3600  # reutiliza cache de odds por até 6h
MAX_RETRIES = 3
TIMEOUT = 30
MAX_PAGES = 20  # paginação máxima ao listar eventos


class BzzoiroAuthError(Exception):
    """Chave BSD_API inválida ou sem permissão (401/403)."""


class BzzoiroRateLimitError(Exception):
    """Rate limit (429) não recuperável."""


def normalize_team_name(name: str) -> str:
    """
    Normaliza nomes de times para comparação fuzzy:
    minúsculas, sem acentos, sem pontuação e espaços colapsados.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def implied_probabilities(
    odd_home: float, odd_draw: float, odd_away: float
) -> Tuple[float, float, float, float]:
    """
    Probabilidades implícitas (1/odd) normalizadas para somar 1.

    Returns
    -------
    (p_home, p_draw, p_away, margem) — margem (overround) = soma bruta - 1.
    """
    raw = [1.0 / float(odd_home), 1.0 / float(odd_draw), 1.0 / float(odd_away)]
    margem = sum(raw) - 1.0
    total = sum(raw) or 1.0
    return raw[0] / total, raw[1] / total, raw[2] / total, margem


# --------------------------------------------------------------------------- #
# Cliente
# --------------------------------------------------------------------------- #
class BzzoiroOddsClient:
    """Cliente de odds da Bzzoiro Sports Data."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BSD_API")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {self.api_key}",
                "User-Agent": "libertadores2026-pipeline/1.0",
                "Accept": "application/json",
            }
        )

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------ #
    # Rede
    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """GET com retries, tratamento de rate limit (429) e conexão."""
        url = f"{BASE_URL}{path}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=TIMEOUT)
            except requests.RequestException as exc:
                print(f"  ⚠️  Erro de conexão (tentativa {attempt}/{MAX_RETRIES}): {exc}")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(2 * attempt)
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code in (401, 403):
                raise BzzoiroAuthError(
                    "Chave BSD_API inválida ou sem permissão. "
                    "Verifique a secret no GitHub ou o arquivo .env."
                )
            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                print(
                    f"  ⚠️  Rate limit (429). Aguardando {retry_after}s "
                    f"(tentativa {attempt}/{MAX_RETRIES})..."
                )
                if attempt == MAX_RETRIES:
                    raise BzzoiroRateLimitError("Rate limit da Bzzoiro excedido.")
                time.sleep(retry_after)
                continue
            if response.status_code == 404:
                return None
            print(f"  ⚠️  HTTP {response.status_code} em {url} — continuando...")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(2 * attempt)

        return None

    # ------------------------------------------------------------------ #
    # Endpoints da Bzzoiro
    # ------------------------------------------------------------------ #
    def list_leagues(self) -> List[Dict[str, Any]]:
        """Lista ligas disponíveis (para localizar a Libertadores)."""
        data = self._get("/leagues/", {"limit": 200})
        if data is None:
            return []
        return _as_results(data)

    def find_libertadores_league(self) -> Optional[Dict[str, Any]]:
        """Procura a liga 'Copa Libertadores' na lista de ligas."""
        for league in self.list_leagues():
            nome = " ".join(str(league.get(k, "")) for k in
                            ("name", "title", "country", "sport"))
            if "libertadores" in normalize_team_name(nome):
                print(f"  ✅ Liga encontrada: {league.get('name')} (id={league.get('id')})")
                return league
        print("  ⚠️  Liga 'Libertadores' não encontrada na Bzzoiro.")
        return None

    def list_events(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        league_id: Optional[int] = None,
        team_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lista eventos (partidas) com filtros opcionais, paginando."""
        params: Dict[str, Any] = {"limit": 200}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if league_id:
            params["league_id"] = league_id
        if team_name:
            params["team"] = normalize_team_name(team_name)

        events: List[Dict[str, Any]] = []
        url = "/events/"
        for _ in range(MAX_PAGES):
            data = self._get(url, params if url.startswith("/events") else None)
            if data is None:
                break
            results = _as_results(data)
            events.extend(results)
            next_url = data.get("next") if isinstance(data, dict) else None
            if not next_url:
                break
            url = next_url
        return events

    def get_event_odds(self, event_id: int, market: str = "1x2") -> Optional[Dict[str, Any]]:
        """
        Busca odds de um evento. Usa o atalho por partida
        (``/events/{id}/odds/``) e, como fallback, o feed ``/odds/`` filtrado.

        Returns
        -------
        dict com ``{odd_mandante, odd_empate, odd_visitante, bookmakers}``
        ou ``None`` se não houver odds para o mercado.
        """
        # Cache em disco: data/raw/odds_<event_id>.json
        cache_path = RAW_DIR / f"odds_{event_id}.json"
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                try:
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        cached = json.load(fh)
                    return _odds_from_payload(cached)
                except (json.JSONDecodeError, OSError):
                    pass

        payload = self._get(f"/events/{event_id}/odds/")
        if payload is None:
            payload = self._get(
                "/odds/", {"event_id": event_id, "market": market, "limit": 200}
            )

        if payload is None:
            return None

        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

        return _odds_from_payload(payload)

    # ------------------------------------------------------------------ #
    # Mapeamento Libertadores -> eventos Bzzoiro
    # ------------------------------------------------------------------ #
    def map_matches(
        self, partidas: pd.DataFrame, events: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Mapeia cada partida da Libertadores para o evento correspondente.

        Critério: par de times (nomes normalizados, qualquer ordem) + data
        com tolerância de ±2 dias. O mapeamento é persistido em
        ``data/raw/bzzoiro_event_mapping.json``.
        """
        mapping: Dict[int, Dict[str, Any]] = {}
        event_pool = list(events)

        for _, match in partidas.iterrows():
            home_n = normalize_team_name(match["mandante"])
            away_n = normalize_team_name(match["visitante"])
            match_date = str(match.get("data", ""))[:10]
            best, best_score = None, 0

            for event in event_pool:
                ev_home = normalize_team_name(
                    event.get("home_team") or event.get("home") or ""
                )
                ev_away = normalize_team_name(
                    event.get("away_team") or event.get("away") or ""
                )
                ev_date = str(
                    event.get("start_at") or event.get("start_time")
                    or event.get("date") or ""
                )[:10]

                if not ev_home or not ev_away:
                    continue

                score = 0
                if {ev_home, ev_away} == {home_n, away_n}:
                    score += 4  # par exato de times
                elif ev_home in (home_n, away_n) or ev_away in (home_n, away_n):
                    score += 1  # apenas um time em comum

                if score >= 4:
                    score += 1
                if match_date and ev_date and abs(
                    _days_between(match_date, ev_date)
                ) <= 2:
                    score += 2  # data próxima
                elif match_date and ev_date:
                    score -= 1

                if score > best_score:
                    best, best_score = event, score

            if best is not None and best_score >= 5:
                mapping[int(match["partida_id"])] = {
                    "event_id": best.get("id") or best.get("event_id"),
                    "bzzoiro_home": best.get("home_team") or best.get("home"),
                    "bzzoiro_away": best.get("away_team") or best.get("away"),
                    "score": best_score,
                }

        try:
            with open(MAPPING_PATH, "w", encoding="utf-8") as fh:
                json.dump(mapping, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

        print(
            f"  🔗 Mapeamento: {len(mapping)}/{len(partidas)} partidas "
            f"encontradas na Bzzoiro."
        )
        return mapping

    # ------------------------------------------------------------------ #
    # Orquestração
    # ------------------------------------------------------------------ #
    def fetch_odds_for_matches(
        self, partidas: pd.DataFrame, events: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Busca odds para todas as partidas e retorna o DataFrame processado."""
        if events is None:
            datas = sorted({str(d)[:10] for d in partidas["data"] if d})
            date_from, date_to = (datas[0], datas[-1]) if datas else (None, None)
            league = self.find_libertadores_league()
            events = self.list_events(
                date_from=date_from,
                date_to=date_to,
                league_id=league.get("id") if league else None,
            )
            print(f"  📅 {len(events)} eventos Bzzoiro no período.")

        mapping = self.map_matches(partidas, events)
        rows: List[Dict[str, Any]] = []
        for _, match in partidas.iterrows():
            pid = int(match["partida_id"])
            mapped = mapping.get(pid)
            if mapped is None:
                continue  # sem odds disponíveis para este jogo
            odds = self.get_event_odds(mapped["event_id"])
            if odds is None:
                continue
            prob_h, prob_d, prob_a, margem = implied_probabilities(
                odds["odd_mandante"], odds["odd_empate"], odds["odd_visitante"]
            )
            rows.append({
                "partida_id": pid,
                "data": match.get("data", ""),
                "fase": match.get("fase", ""),
                "rodada": match.get("rodada", ""),
                "mandante": match["mandante"],
                "visitante": match["visitante"],
                "odd_mandante": odds["odd_mandante"],
                "odd_empate": odds["odd_empate"],
                "odd_visitante": odds["odd_visitante"],
                "prob_mandante_impl": round(prob_h, 4),
                "prob_empate_impl": round(prob_d, 4),
                "prob_visitante_impl": round(prob_a, 4),
                "margem": round(margem, 4),
                "bookmaker": ", ".join(odds.get("bookmakers", [])) or "consenso",
                "gols_mandante": match.get("gols_mandante"),
                "gols_visitante": match.get("gols_visitante"),
                "resultado": match.get("resultado"),
                "fonte": "bzzoiro",
            })

        return pd.DataFrame(rows, columns=ODDS_COLUMNS)

    def run(self, partidas: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Coleta odds da Bzzoiro (ou fallback de exemplo) e salva o CSV."""
        print("=" * 60)
        print("🎰 BZZOIRO SPORTS DATA — Odds 1X2 das partidas")
        print("=" * 60)

        if partidas is None:
            partidas = _load_partidas_default()

        df = pd.DataFrame(columns=ODDS_COLUMNS)

        if not self.has_key:
            print("  ℹ️  BSD_API não configurada — usando odds de exemplo.")
        else:
            try:
                df = self.fetch_odds_for_matches(partidas)
                if df.empty:
                    print("  ⚠️  Nenhuma odd encontrada para as partidas da "
                          "Libertadores 2026 na Bzzoiro.")
            except (BzzoiroAuthError, BzzoiroRateLimitError) as exc:
                print(f"  ❌ {exc}")
                df = pd.DataFrame(columns=ODDS_COLUMNS)

        if df.empty:
            df = _load_example_odds(partidas)

        cobertura = len(df) / len(partidas) if len(partidas) else 0.0
        print(f"  📊 Cobertura de odds: {len(df)}/{len(partidas)} partidas "
              f"({cobertura:.0%}).")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"  💾 Odds salvas em {PROCESSED_PATH}")
        return df


# --------------------------------------------------------------------------- #
# Helpers de parsing/fallback
# --------------------------------------------------------------------------- #
def _as_results(data: Any) -> List[Dict[str, Any]]:
    """Normaliza a resposta paginada da Bzzoiro (``{results: [...]}``)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return results
        return [data] if data else []
    return []


def _odds_from_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """
    Converte o payload de odds (feed ou atalho por partida) em
    ``{odd_mandante, odd_empate, odd_visitante, bookmakers}``.

    Com chave gratuita a Bzzoiro retorna apenas o consenso; com chave paga,
    retorna uma linha por casa — calculamos a média simples das casas.
    """
    rows = _as_results(payload)
    if not rows:
        return None

    outcome_map = {"HOME": "odd_mandante", "DRAW": "odd_empate", "AWAY": "odd_visitante"}
    collected: Dict[str, List[float]] = {"odd_mandante": [], "odd_empate": [], "odd_visitante": []}
    bookmakers: List[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        outcome = str(row.get("outcome", "")).upper()
        if outcome not in outcome_map:
            continue
        odds_value = row.get("decimal_odds") or row.get("odds") or row.get("value")
        try:
            odds_float = float(odds_value)
        except (TypeError, ValueError):
            continue
        if odds_float <= 1.0:
            continue
        collected[outcome_map[outcome]].append(odds_float)
        bookmaker = row.get("bookmaker_name") or row.get("bookmaker_slug")
        if bookmaker and bookmaker not in bookmakers:
            bookmakers.append(str(bookmaker))

    if not all(collected.values()):
        return None

    return {
        "odd_mandante": round(float(np_mean(collected["odd_mandante"])), 2),
        "odd_empate": round(float(np_mean(collected["odd_empate"])), 2),
        "odd_visitante": round(float(np_mean(collected["odd_visitante"])), 2),
        "bookmakers": bookmakers,
    }


def np_mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _parse_retry_after(value: Optional[str]) -> int:
    try:
        return max(5, int(float(value)))
    except (TypeError, ValueError):
        return 60


def _days_between(date_a: str, date_b: str) -> int:
    """Diferença absoluta em dias entre duas datas ISO 'YYYY-MM-DD'."""
    try:
        import datetime as dt

        a = dt.date.fromisoformat(date_a)
        b = dt.date.fromisoformat(date_b)
        return abs((a - b).days)
    except ValueError:
        return 999


def _load_partidas_default() -> pd.DataFrame:
    """Prefere o CSV processado de estatísticas; senão, os dados de exemplo."""
    from src.api_futebol_client import PROCESSED_PATH as STATS_PATH

    if STATS_PATH.exists():
        return pd.read_csv(STATS_PATH)
    return _load_example_partidas()


def _load_example_partidas() -> pd.DataFrame:
    if EXAMPLE_PARTIDAS_PATH.exists():
        return pd.read_csv(EXAMPLE_PARTIDAS_PATH)
    from src.generate_example_data import generate_partidas

    return generate_partidas()


def _load_example_odds(partidas: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Carrega (ou gera) as odds de exemplo e as alinha às partidas."""
    if EXAMPLE_ODDS_PATH.exists():
        return pd.read_csv(EXAMPLE_ODDS_PATH)
    partidas = partidas if partidas is not None else _load_example_partidas()
    return generate_odds(partidas)


if __name__ == "__main__":
    client = BzzoiroOddsClient()
    client.run()
