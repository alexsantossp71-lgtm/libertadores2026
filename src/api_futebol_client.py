"""
Cliente da API Futebol (https://api-futebol.com.br) para a coleta de
estatísticas detalhadas das partidas da Libertadores 2026.

Fonte dos dados (por partida):
  * Árbitro principal (nome e nacionalidade, quando disponíveis);
  * Faltas cometidas por time;
  * Cartões amarelos e vermelhos por time;
  * Posse de bola (percentual por time);
  * Passes certos e errados por time;
  * Finalizações (totais, no gol e para fora);
  * Escanteios, impedimentos e defesas do goleiro.

Autenticação
------------
A chave é lida de ``os.getenv("API_FUTEBOL_KEY")`` (ou do arquivo ``.env`` via
``python-dotenv``). Configure a secret ``API_FUTEBOL_KEY`` no GitHub para o
workflow de atualização automática.

Rate limit
----------
A API Futebol impõe o limite de **10 requisições/minuto**. O cliente espera
``time.sleep(6)`` entre chamadas e usa cache em disco
(``data/raw/partidas_estatisticas_<id>.json``) para não repetir chamadas.

Fallback
--------
Sem chave de API (ou diante de falhas de conexão/rate limit persistentes), o
cliente usa a base de exemplo de ``data/examples/`` — assim o pipeline, o
dashboard e o notebook continuam funcionando offline.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Carga opcional do .env (não falha se o pacote não estiver instalado)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv é opcional
    pass

from src.generate_example_data import (  # noqa: E402
    EXAMPLE_PARTIDAS_PATH,
    PARTIDAS_COLUMNS,
    generate_partidas,
)

# --------------------------------------------------------------------------- #
# Configurações
# --------------------------------------------------------------------------- #
BASE_URL = "https://api.api-futebol.com.br/v1"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "libertadores_estatisticas_detalhadas.csv"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_PER_MINUTE = 10
SLEEP_BETWEEN_CALLS = 6  # 60s / 10 req = 6s por requisição
MAX_RETRIES = 3
TIMEOUT = 30

LIBERTADORES_KEYWORDS = ("libertadores", "conmebol")

# --------------------------------------------------------------------------- #
# Erros
# --------------------------------------------------------------------------- #
class ApiFutebolError(Exception):
    """Erro base do cliente da API Futebol."""


class ApiFutebolAuthError(ApiFutebolError):
    """Chave de API inválida ou sem autorização (401/403)."""


class ApiFutebolRateLimitError(ApiFutebolError):
    """Limite de requisições excedido (429) e não recuperável."""


# --------------------------------------------------------------------------- #
# Cliente
# --------------------------------------------------------------------------- #
class ApiFutebolClient:
    """Cliente para estatísticas detalhadas de partidas da API Futebol."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("API_FUTEBOL_KEY")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "libertadores2026-pipeline/1.0",
                "Accept": "application/json",
            }
        )
        self.last_call_ts: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Propriedades e utilidades de rede
    # ------------------------------------------------------------------ #
    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def _wait_rate_limit(self) -> None:
        """Garante o ritmo máximo de 10 req/min (6s entre chamadas)."""
        if self.last_call_ts is not None:
            elapsed = time.monotonic() - self.last_call_ts
            wait = SLEEP_BETWEEN_CALLS - elapsed
            if wait > 0:
                time.sleep(wait)
        self.last_call_ts = time.monotonic()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """GET com retries, backoff e tratamento de rate limit/conexão."""
        url = f"{BASE_URL}{path}"
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_rate_limit()
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
                raise ApiFutebolAuthError(
                    "Chave API_FUTEBOL_KEY inválida ou sem permissão. "
                    "Verifique a secret no GitHub ou o arquivo .env."
                )
            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                print(
                    f"  ⚠️  Rate limit (429). Aguardando {retry_after}s "
                    f"(tentativa {attempt}/{MAX_RETRIES})..."
                )
                if attempt == MAX_RETRIES:
                    raise ApiFutebolRateLimitError(
                        "Limite de requisições da API Futebol excedido."
                    )
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
    # Endpoints
    # ------------------------------------------------------------------ #
    def get_campeonatos(self) -> Optional[List[Dict[str, Any]]]:
        """Lista os campeonatos disponíveis na API."""
        return self._get("/campeonatos")

    def find_libertadores(self) -> Optional[Dict[str, Any]]:
        """Procura o campeonato 'Libertadores' na lista de campeonatos."""
        campeonatos = self.get_campeonatos()
        if not campeonatos:
            return None
        for camp in campeonatos:
            nome = f"{camp.get('nome', '')} {camp.get('nome_popular', '')}".lower()
            if any(k in nome for k in LIBERTADORES_KEYWORDS):
                print(f"  ✅ Campeonato encontrado: {camp.get('nome')} (id={camp.get('campeonato_id')})")
                return camp
        print("  ⚠️  Campeonato 'Libertadores' não encontrado na API Futebol.")
        return None

    def get_partidas(self, campeonato_id: int) -> List[Dict[str, Any]]:
        """
        Lista as partidas do campeonato (resumo, sem estatísticas).

        A resposta é indexada por fase e rodada::

            {"campeonato": {...}, "partidas": {fase_slug: {rodada_slug: [...]}}}

        Retorna uma lista achatada com ``partida_id``, times, status e data.
        """
        data = self._get(f"/campeonatos/{campeonato_id}/partidas")
        if not data or not isinstance(data.get("partidas"), dict):
            return []

        flat: List[Dict[str, Any]] = []
        for fase_slug, rodadas in data["partidas"].items():
            if not isinstance(rodadas, dict):
                continue
            for rodada_slug, jogos in rodadas.items():
                if not isinstance(jogos, list):
                    continue
                for jogo in jogos:
                    mandante = jogo.get("time_mandante") or {}
                    visitante = jogo.get("time_visitante") or {}
                    flat.append({
                        "partida_id": jogo.get("partida_id"),
                        "fase_slug": fase_slug,
                        "rodada_slug": rodada_slug,
                        "status": jogo.get("status"),
                        "data_realizacao_iso": jogo.get("data_realizacao_iso"),
                        "mandante": mandante.get("nome_popular")
                        or mandante.get("nome")
                        or jogo.get("mandante"),
                        "visitante": visitante.get("nome_popular")
                        or visitante.get("nome")
                        or jogo.get("visitante"),
                    })
        return flat

    def get_partida(self, partida_id: int, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Busca o detalhe de uma partida (estatísticas, cartões, gols).

        Usa cache em ``data/raw/partidas_estatisticas_<id>.json`` para evitar
        chamadas repetidas (o arquivo só é recarregado com ``use_cache=False``).
        """
        cache_path = RAW_DIR / f"partidas_estatisticas_{partida_id}.json"
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass  # cache corrompido -> refaz a chamada

        detail = self._get(f"/partidas/{partida_id}")
        if detail is None:
            return None

        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(detail, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"  ⚠️  Não foi possível salvar o cache {cache_path}: {exc}")

        return detail

    # ------------------------------------------------------------------ #
    # Parsing do JSON -> linha canônica
    # ------------------------------------------------------------------ #
    def parse_partida(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converte o JSON de detalhe da API em uma linha do esquema canônico.

        A API não garante todos os campos para todas as partidas/campeonatos:
        campos ausentes viram ``NaN`` (o pipeline continua normalmente).
        """
        row: Dict[str, Any] = {
            "partida_id": detail.get("partida_id"),
            "fase": _first_text(detail.get("fase"), "nome") or "",
            "rodada": _first_text(detail.get("rodada"), "nome", "rodada")
            or detail.get("rodada") or "",
            "grupo": detail.get("grupo") or "",
            "mandante": _first_text(detail.get("time_mandante"), "nome_popular", "nome")
            or _first_text(detail.get("mandante"), "nome_popular", "nome"),
            "visitante": _first_text(detail.get("time_visitante"), "nome_popular", "nome")
            or _first_text(detail.get("visitante"), "nome_popular", "nome"),
        }

        # ---- Data ----
        data = detail.get("data_realizacao_iso") or detail.get("data_realizacao")
        row["data"] = str(data)[:10] if data else ""

        # ---- Placar / gols ----
        gols_m, gols_v = _parse_placar(detail)
        row["gols_mandante"] = gols_m
        row["gols_visitante"] = gols_v
        if gols_m is not None and gols_v is not None:
            row["resultado"] = (
                "mandante" if gols_m > gols_v
                else "visitante" if gols_m < gols_v
                else "empate"
            )
        else:
            row["resultado"] = None

        # ---- Árbitro (campo não garantido pela API — defensivo) ----
        arbitro = (
            detail.get("arbitro")
            or detail.get("arbitragem")
            or detail.get("juiz")
            or detail.get("referee")
        )
        row["arbitro"] = _extract_arbitro_nome(arbitro)
        row["arbitro_pais"] = _extract_arbitro_pais(arbitro)

        # ---- Estatísticas ----
        stats = detail.get("estatisticas") or {}
        stats_m = _side_stats(stats, "mandante")
        stats_v = _side_stats(stats, "visitante")

        row.update({
            "posse_mandante": _num(stats_m.get("posse")),
            "posse_visitante": _num(stats_v.get("posse")),
            "faltas_mandante": _num(stats_m.get("faltas")),
            "faltas_visitante": _num(stats_v.get("faltas")),
            "escanteios_mandante": _num(stats_m.get("escanteios")),
            "escanteios_visitante": _num(stats_v.get("escanteios")),
            "passes_certos_mandante": _num(stats_m.get("passes_certos")),
            "passes_certos_visitante": _num(stats_v.get("passes_certos")),
            "passes_errados_mandante": _num(stats_m.get("passes_errados")),
            "passes_errados_visitante": _num(stats_v.get("passes_errados")),
            "finalizacoes_mandante": _num(stats_m.get("finalizacoes")),
            "finalizacoes_visitante": _num(stats_v.get("finalizacoes")),
            "finalizacoes_no_gol_mandante": _num(stats_m.get("finalizacoes_no_gol")),
            "finalizacoes_no_gol_visitante": _num(stats_v.get("finalizacoes_no_gol")),
            "finalizacoes_fora_mandante": _num(stats_m.get("finalizacoes_fora")),
            "finalizacoes_fora_visitante": _num(stats_v.get("finalizacoes_fora")),
            "impedimentos_mandante": _num(stats_m.get("impedimentos")),
            "impedimentos_visitante": _num(stats_v.get("impedimentos")),
            "defesas_mandante": _num(stats_m.get("defesas")),
            "defesas_visitante": _num(stats_v.get("defesas")),
        })

        # ---- Cartões (bloco `cartoes` ou estatísticas) ----
        cartoes = detail.get("cartoes") or {}
        amarelos_m, amarelos_v = _cartoes_count(cartoes, "amarelo")
        vermelhos_m, vermelhos_v = _cartoes_count(cartoes, "vermelho")
        if amarelos_m is None:
            amarelos_m = _num(stats_m.get("cartoes_amarelos"))
        if amarelos_v is None:
            amarelos_v = _num(stats_v.get("cartoes_amarelos"))
        if vermelhos_m is None:
            vermelhos_m = _num(stats_m.get("cartoes_vermelhos"))
        if vermelhos_v is None:
            vermelhos_v = _num(stats_v.get("cartoes_vermelhos"))

        row.update({
            "cartoes_amarelos_mandante": amarelos_m,
            "cartoes_amarelos_visitante": amarelos_v,
            "cartoes_vermelhos_mandante": vermelhos_m,
            "cartoes_vermelhos_visitante": vermelhos_v,
        })

        row["fonte"] = "api-futebol"
        return row

    # ------------------------------------------------------------------ #
    # Coleta completa
    # ------------------------------------------------------------------ #
    def collect_estatisticas(
        self, campeonato_id: Optional[int] = None, only_finished: bool = True
    ) -> pd.DataFrame:
        """Coleta estatísticas de todas as partidas do campeonato."""
        if campeonato_id is None:
            libertadores = self.find_libertadores()
            if libertadores is None:
                print("  ❌ Nenhum campeonato da Libertadores disponível na API.")
                return pd.DataFrame(columns=PARTIDAS_COLUMNS)
            campeonato_id = libertadores.get("campeonato_id")

        partidas = self.get_partidas(campeonato_id)
        if not partidas:
            print("  ❌ Nenhuma partida encontrada para o campeonato.")
            return pd.DataFrame(columns=PARTIDAS_COLUMNS)

        print(f"  🔎 {len(partidas)} partidas listadas pela API.")
        rows: List[Dict[str, Any]] = []
        total = len(partidas)
        for idx, partida in enumerate(partidas, start=1):
            pid = partida.get("partida_id")
            status = partida.get("status")
            if only_finished and status not in ("finalizado", "encerrado", None):
                continue

            print(f"  [{idx}/{total}] Partida {pid} "
                  f"({partida.get('mandante')} x {partida.get('visitante')})...")
            detail = self.get_partida(pid)
            if detail is None:
                print("     ⚠️  Sem dados disponíveis — marcada como NaN/ignorada.")
                rows.append({
                    **{col: None for col in PARTIDAS_COLUMNS},
                    "partida_id": pid,
                    "mandante": partida.get("mandante"),
                    "visitante": partida.get("visitante"),
                    "fase": partida.get("fase_slug", ""),
                    "rodada": partida.get("rodada_slug", ""),
                    "fonte": "api-futebol",
                })
                continue

            row = self.parse_partida(detail)
            rows.append(row)

        df = pd.DataFrame(rows, columns=PARTIDAS_COLUMNS)
        # Mantém apenas partidas com placar definido OU estatísticas
        if not df.empty and "gols_mandante" in df:
            df = df.dropna(subset=["gols_mandante", "gols_visitante"], how="all")
        return df

    # ------------------------------------------------------------------ #
    # Orquestração (com fallback para dados de exemplo)
    # ------------------------------------------------------------------ #
    def run(self, campeonato_id: Optional[int] = None) -> pd.DataFrame:
        """Coleta da API (ou fallback de exemplo) e salva o CSV processado."""
        print("=" * 60)
        print("📥 API FUTEBOL — Estatísticas detalhadas das partidas")
        print("=" * 60)

        df = pd.DataFrame(columns=PARTIDAS_COLUMNS)

        if not self.has_key:
            print("  ℹ️  API_FUTEBOL_KEY não configurada — usando dados de exemplo.")
        else:
            try:
                df = self.collect_estatisticas(campeonato_id)
            except (ApiFutebolAuthError, ApiFutebolRateLimitError) as exc:
                print(f"  ❌ {exc}")
                print("  ℹ️  Usando dados de exemplo como fallback.")

        if df.empty:
            df = self._load_example_partidas()

        df = df[PARTIDAS_COLUMNS]
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"  💾 {len(df)} partidas salvas em {PROCESSED_PATH}")
        return df

    @staticmethod
    def _load_example_partidas() -> pd.DataFrame:
        """Carrega (ou gera) a base de exemplo de partidas."""
        if EXAMPLE_PARTIDAS_PATH.exists():
            df = pd.read_csv(EXAMPLE_PARTIDAS_PATH)
        else:
            df = generate_partidas()
        print(
            f"  ℹ️  Base de exemplo: {len(df)} partidas "
            f"(fonte='{df['fonte'].iloc[0] if not df.empty else 'exemplo'}')."
        )
        return df


# --------------------------------------------------------------------------- #
# Funções auxiliares de parsing
# --------------------------------------------------------------------------- #
def _parse_retry_after(value: Optional[str]) -> int:
    try:
        return max(60, int(float(value)))
    except (TypeError, ValueError):
        return 60


def _first_text(obj: Any, *keys: str) -> Optional[str]:
    """Extrai o primeiro texto de um objeto/dict, checando várias chaves."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj or None
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if value:
                return str(value)
        return None
    return None


def _num(value: Any) -> Optional[float]:
    """Converte valores como '55%', '1.234' ou 55 para float (ou None)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+([.,]\d+)?", str(value))
    if not match:
        return None
    return float(match.group().replace(",", "."))


def _side_stats(stats: Any, lado: str) -> Dict[str, Any]:
    """
    Extrai o dicionário de estatísticas de um dos lados.

    A API pode retornar ``{"mandante": {...}, "visitante": {...}}`` ou uma
    lista de pares ``{nome/estatistica: valor}``. Aqui normalizamos tudo para
    um dicionário com chaves canônicas (em português, com aliases).
    """
    if isinstance(stats, dict):
        side = stats.get(lado) or stats.get(lado.capitalize()) or {}
        if isinstance(side, dict):
            return _alias_stats(side)
        return {}

    if isinstance(stats, list):
        out: Dict[str, Any] = {}
        for entry in stats:
            if not isinstance(entry, dict):
                continue
            entry_lado = entry.get("lado") or entry.get("time")
            if entry_lado is not None and str(entry_lado).lower() != lado:
                continue
            nome = entry.get("nome") or entry.get("estatistica") or ""
            valor = entry.get("valor") or entry.get("total")
            out.setdefault(str(nome).lower(), valor)
        return _alias_stats(out)

    return {}


_STAT_ALIASES = {
    "posse": ("posse_de_bola", "posse", "posse_%", "posse_de_bola_%"),
    "faltas": ("faltas", "faltas_cometidas", "faltas_%"),
    "escanteios": ("escanteios", "escanteios_cobrados", "escanteios_%"),
    "passes_certos": ("passes_certos", "passes_completos", "passes"),
    "passes_errados": ("passes_errados", "passes_incompletos", "passes_errados_%"),
    "finalizacoes": ("finalizacao", "finalizacoes", "chutes", "finalizacoes_totais"),
    "finalizacoes_no_gol": (
        "finalizacao_no_gol", "finalizacoes_no_gol", "chutes_no_gol",
        "finalizacoes_no_alvo", "chutes_no_alvo",
    ),
    "finalizacoes_fora": (
        "finalizacao_fora", "finalizacoes_fora", "chutes_fora",
        "finalizacoes_para_fora",
    ),
    "impedimentos": ("impedimentos", "impedimentos_%"),
    "defesas": ("defesas", "defesas_do_goleiro", "defesas_dificeis"),
    "cartoes_amarelos": ("cartoes_amarelos", "cartao_amarelo"),
    "cartoes_vermelhos": ("cartoes_vermelhos", "cartao_vermelho"),
}


def _alias_stats(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Mapeia chaves comuns da API para os nomes canônicos."""
    def _norm(key: Any) -> str:
        return (
            str(key).lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "_pct")
        )

    lower = {_norm(k): v for k, v in raw.items()}
    out: Dict[str, Any] = {}
    for canonical, aliases in _STAT_ALIASES.items():
        for alias in aliases:
            alias_norm = _norm(alias)
            if alias_norm in lower:
                out[canonical] = lower[alias_norm]
                break
    return out


def _cartoes_count(cartoes: Any, cor: str) -> Tuple[Optional[int], Optional[int]]:
    """Conta cartões de uma cor para cada lado a partir do bloco `cartoes`."""
    if not isinstance(cartoes, dict):
        return None, None
    bloco = cartoes.get(cor) or {}
    if not isinstance(bloco, dict):
        return None, None
    mandante = bloco.get("mandante")
    visitante = bloco.get("visitante")
    count = lambda x: len(x) if isinstance(x, list) else (_num(x) if x is not None else None)  # noqa: E731
    return count(mandante), count(visitante)


def _extract_arbitro_nome(arbitro: Any) -> Optional[str]:
    if arbitro is None:
        return None
    if isinstance(arbitro, str):
        return arbitro or None
    if isinstance(arbitro, dict):
        return _first_text(arbitro, "nome", "nome_popular", "name", "arbitro")
    if isinstance(arbitro, list) and arbitro:
        return _extract_arbitro_nome(arbitro[0])
    return None


def _extract_arbitro_pais(arbitro: Any) -> Optional[str]:
    if not isinstance(arbitro, dict):
        return None
    return _first_text(arbitro, "pais", "nacionalidade", "country", "nationality")


def _parse_placar(detail: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Extrai os gols de cada lado (placar numérico ou string '2x1').

    Strings com disputa de pênaltis (ex.: ``"Atlético-MG (4)2x2(3) Palmeiras"``)
    têm a parte entre parênteses removida antes do parsing.
    """
    gols_m = detail.get("placar_mandante")
    gols_v = detail.get("placar_visitante")
    if gols_m is None and isinstance(detail.get("placar"), str):
        placar_limpo = re.sub(r"\(\d+\)", "", detail["placar"])
        match = re.search(r"(\d+)\s*[xX\-:]\s*(\d+)", placar_limpo)
        if match:
            return int(match.group(1)), int(match.group(2))
    return _int_or_none(gols_m), _int_or_none(gols_v)


def _int_or_none(value: Any) -> Optional[int]:
    num = _num(value)
    return int(num) if num is not None else None


if __name__ == "__main__":
    client = ApiFutebolClient()
    client.run()
