"""
Cliente de odds 1X2 da The Odds API (https://the-odds-api.com).

Responsabilidades:
  * Descobrir o sport key da Copa Libertadores (``list_sports``);
  * Baixar as odds h2h dos próximos jogos (``/v4/sports/{sport_key}/odds``);
  * Agregar as odds médias entre casas de apostas, calcular probabilidades
    implícitas (normalizadas) e margem de mercado;
  * Mapear os nomes dos times da API para os nomes canônicos do projeto
    (via ``normalize_team_name`` + ``_CANONICAL`` de ``src.real_data``);
  * Persistir em ``data/processed/libertadores_odds.csv`` com o schema
    canônico compartilhado com o pipeline.

Cache: ``data/raw/oddsapi_sports.json`` e ``data/raw/oddsapi_upcoming.json``
com TTL de 6h (mesmo estilo do cliente Bzzoiro).

Autenticação: variável ``ODDSAPI_API`` no .env da raiz do workspace.
Header de uso: ``x-requests-used`` / ``x-requests-remaining``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from src.odds_client import (  # noqa: E402
    CACHE_TTL_SECONDS,
    PROCESSED_DIR,
    PROCESSED_PATH,
    RAW_DIR,
    TIMEOUT,
    implied_probabilities,
    normalize_team_name,
)
from src.generate_example_data import ODDS_COLUMNS  # noqa: E402

BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "soccer_conmebol_copa_libertadores"
SPORTS_CACHE = RAW_DIR / "oddsapi_sports.json"
UPCOMING_CACHE = RAW_DIR / "oddsapi_upcoming.json"

REGIONS = "eu,uk,us"


class OddsApiAuthError(Exception):
    """Chave ODDSAPI_API inválida (401)."""


class OddsApiUsageError(Exception):
    """Cota da API esgotada ou outro erro de uso (422/429…)."""


# --------------------------------------------------------------------------- #
# Helpers de nome / parsing
# --------------------------------------------------------------------------- #
def _variantes(nome: str) -> List[str]:
    """Gera variações normalizadas para lookup no _CANONICAL."""
    norm = normalize_team_name(nome)
    out = [norm]
    # "fluminense rj" -> já coberto; remove sufixos duplos / gens
    tokens = norm.split()
    while tokens:
        out.append(" ".join(tokens))
        tokens = tokens[:-1]
    return out


def map_team_name(api_name: str) -> str:
    """
    Mapeia o nome vindo da The Odds API (em inglês / com sufixo regional)
    para o nome canônico do projeto. Fallback: o nome original.
    """
    try:
        from src.real_data import _CANONICAL
    except Exception:  # pragma: no cover - defensivo
        return api_name
    for cand in _variantes(api_name):
        if cand in _CANONICAL:
            return _CANONICAL[cand]
    return api_name


def parse_h2h_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Agrega o mercado h2h de um evento da The Odds API:
    média simples das odds entre casas para cada resultado.

    Returns
    -------
    dict com ``odd_mandante``, ``odd_empate``, ``odd_visitante``,
    ``n_bookmakers``, ou None se faltar qualquer resultado.
    """
    homes: List[float] = []
    draws: List[float] = []
    aways: List[float] = []
    home_team = str(event.get("home_team") or "")
    away_team = str(event.get("away_team") or "")
    if not home_team or not away_team:
        return None

    for bookmaker in event.get("bookmakers") or []:
        for market in (bookmaker or {}).get("markets") or []:
            if market.get("key") != "h2h":
                continue
            h = d = a = None
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "")
                try:
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                if price <= 1.0:
                    continue
                if name == home_team:
                    h = price
                elif name == away_team:
                    a = price
                elif normalize_team_name(name) in ("draw", "empate"):
                    d = price
            if h and d and a:
                homes.append(h)
                draws.append(d)
                aways.append(a)

    if not homes:
        return None
    return {
        "odd_mandante": round(sum(homes) / len(homes), 2),
        "odd_empate": round(sum(draws) / len(draws), 2),
        "odd_visitante": round(sum(aways) / len(aways), 2),
        "n_bookmakers": len(homes),
    }


def event_to_row(
    event: Dict[str, Any],
    partida_id: int,
    data: str,
    fase: str,
    rodada: str,
    gols_mandante: Any = "",
    gols_visitante: Any = "",
    resultado: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Converte um evento h2h da The Odds API numa linha do schema do projeto.
    Retorna None quando o evento não tem odds completas.
    """
    parsed = parse_h2h_event(event)
    if parsed is None:
        return None
    prob_h, prob_d, prob_a, margem = implied_probabilities(
        parsed["odd_mandante"], parsed["odd_empate"], parsed["odd_visitante"]
    )
    return {
        "partida_id": int(partida_id),
        "data": data,
        "fase": fase,
        "rodada": rodada,
        "mandante": map_team_name(str(event.get("home_team") or "")),
        "visitante": map_team_name(str(event.get("away_team") or "")),
        "odd_mandante": parsed["odd_mandante"],
        "odd_empate": parsed["odd_empate"],
        "odd_visitante": parsed["odd_visitante"],
        "prob_mandante_impl": round(prob_h, 4),
        "prob_empate_impl": round(prob_d, 4),
        "prob_visitante_impl": round(prob_a, 4),
        "margem": round(margem, 4),
        "bookmaker": f"The Odds API (média {parsed['n_bookmakers']} casas)",
        "gols_mandante": gols_mandante,
        "gols_visitante": gols_visitante,
        "resultado": resultado,
        "fonte": "the-odds-api",
    }


# --------------------------------------------------------------------------- #
# Cliente
# --------------------------------------------------------------------------- #
class OddsApiClient:
    """Cliente de odds da The Odds API (v4)."""

    def __init__(self, api_key: Optional[str] = None):
        import os

        self.api_key = api_key or os.getenv("ODDSAPI_API")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "libertadores2026-pipeline/1.0", "Accept": "application/json"}
        )
        self.requests_used: Optional[str] = None
        self.requests_remaining: Optional[str] = None

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------ #
    # Rede
    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{BASE_URL}{path}"
        params = dict(params or {})
        params["apiKey"] = self.api_key
        response = self.session.get(url, params=params, timeout=TIMEOUT)
        self.requests_used = response.headers.get("x-requests-used")
        self.requests_remaining = response.headers.get("x-requests-remaining")
        if response.status_code == 401:
            raise OddsApiAuthError(
                "Chave ODDSAPI_API inválida ou expirada. "
                "Verifique o .env na raiz do workspace."
            )
        if response.status_code == 422:
            detail = response.text[:200]
            raise OddsApiUsageError(f"Uso incorreto/limite esgotado: {detail}")
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _read_cache(path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age >= CACHE_TTL_SECONDS:
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _write_cache(path: Path, payload: Any) -> None:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def list_sports(self, force: bool = False) -> List[Dict[str, Any]]:
        if not force:
            cached = self._read_cache(SPORTS_CACHE)
            if cached is not None:
                return cached
        data = self._get("/sports/")
        self._write_cache(SPORTS_CACHE, data)
        return data

    def find_libertadores_sport(self) -> Optional[Dict[str, Any]]:
        for sport in self.list_sports():
            titulo = " ".join(str(sport.get(k, "")) for k in ("key", "title", "group"))
            if "libertadores" in normalize_team_name(titulo):
                return sport
        return None

    def fetch_upcoming_odds(
        self,
        sport_key: str = DEFAULT_SPORT_KEY,
        regions: str = REGIONS,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Baixa as odds h2h dos próximos jogos (cache de 6h)."""
        if not force:
            cached = self._read_cache(UPCOMING_CACHE)
            if cached is not None:
                return cached
        data = self._get(
            f"/sports/{sport_key}/odds/",
            {
                "regions": regions,
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        self._write_cache(UPCOMING_CACHE, data)
        return data

    # ------------------------------------------------------------------ #
    # Orquestração
    # ------------------------------------------------------------------ #
    def build_odds_dataframe(
        self,
        events: List[Dict[str, Any]],
        fase: str = "Quartas de Final",
        rodada: str = "Ida",
        base_partida_id: int = 5001,
    ) -> pd.DataFrame:
        """
        Converte os eventos da API em linhas no schema canônico.

        ``partida_id`` é atribuído de forma determinística (ordenado por
        data de início), começando em ``base_partida_id`` — os próximos
        jogos ainda não têm os IDs do calendário oficial.
        """
        events_sorted = sorted(events, key=lambda e: str(e.get("commence_time") or ""))
        rows: List[Dict[str, Any]] = []
        pid = base_partida_id
        for event in events_sorted:
            commence = str(event.get("commence_time") or "")
            data = commence[:10] if commence else ""
            row = event_to_row(event, partida_id=pid, data=data, fase=fase, rodada=rodada)
            if row is None:
                continue
            rows.append(row)
            pid += 1
        return pd.DataFrame(rows, columns=ODDS_COLUMNS)

    def run(self, force: bool = False) -> pd.DataFrame:
        """Coleta odds reais da The Odds API e salva o CSV processado."""
        print("=" * 60)
        print("🎰 THE ODDS API — Odds 1X2 (Copa Libertadores 2026)")
        print("=" * 60)

        if not self.has_key:
            print("  ℹ️  ODDSAPI_API não configurada — nada a fazer.")
            return pd.DataFrame(columns=ODDS_COLUMNS)

        sport_key = DEFAULT_SPORT_KEY
        if self._read_cache(SPORTS_CACHE) is None:
            sport = self.find_libertadores_sport()
            if sport:
                sport_key = str(sport.get("key") or sport_key)
                print(f"  ✅ Sport key da Libertadores: {sport_key}")
            else:
                print("  ⚠️  Sport key da Libertadores não listada — usando o padrão.")

        events = self.fetch_upcoming_odds(sport_key=sport_key, force=force)
        print(f"  📅 {len(events)} eventos com odds h2h coletados.")

        df = self.build_odds_dataframe(events)
        if df.empty:
            print("  ⚠️  Nenhuma linha de odds montada a partir dos eventos.")
            return df

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"  💾 Odds salvas em {PROCESSED_PATH} ({len(df)} jogos)")
        if self.requests_used is not None:
            print(
                f"  📊 Créditos: used={self.requests_used} "
                f"remaining={self.requests_remaining}"
            )
        return df


if __name__ == "__main__":
    OddsApiClient().run()
