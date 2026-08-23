"""
Raspagem da FBref para estatísticas de elencos e jogadores da Libertadores.

A FBref (Sports-Reference) publica, para a Copa Libertadores (comps/14),
páginas de **Standard**, **Shooting**, **Playing Time**, **Miscellaneous**
e **Goalkeeping**. Não há (ainda) passing/defense/xG avançados nesta
competição — o scraper só pede o que existe e documenta a lacuna.

Uso::

    python src/fbref_scraper.py scrape --season 2026
    python src/fbref_scraper.py scrape --season 2026 --from-cache
    python src/fbref_scraper.py parse --html tests/fixtures/fbref/stats_sample.html

Política de rede
----------------
Sports-Reference pede no máximo ~20 req/min. O cliente espera
``FBREF_SLEEP`` segundos (padrão 3.5) entre GET, identifica o User-Agent
do projeto e cacheia o HTML em ``data/raw/fbref/``.

Fallback (nada é inventado)
---------------------------
1. HTML em cache (``data/raw/fbref/``);
2. Snapshot versionado em ``data/historical/fbref/`` (raspagem auditável);
3. Sem cache, sem snapshot e sem rede → erro explícito
   (ou base de exemplo se ``ALLOW_EXAMPLE_DATA=1``).
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "fbref"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
HIST_DIR = ROOT_DIR / "data" / "historical" / "fbref"
EXAMPLES_DIR = ROOT_DIR / "data" / "examples"

JOGADORES_CSV = PROCESSED_DIR / "fbref_jogadores.csv"
ELENCOS_CSV = PROCESSED_DIR / "fbref_elencos.csv"
PARTIDAS_CSV = PROCESSED_DIR / "fbref_partidas.csv"
SQLITE_PATH = PROCESSED_DIR / "fbref_libertadores.sqlite"

COMP_ID = 14
COMP_SLUG = "Copa-Libertadores"
BASE_URL = "https://fbref.com"
DEFAULT_SLEEP = float(os.getenv("FBREF_SLEEP", "3.5"))
TIMEOUT = 30
MAX_RETRIES = 3
USER_AGENT = (
    "libertadores2026-fbref/1.0 "
    "(+https://github.com/alexsantossp71-lgtm/libertadores2026; "
    "pesquisa acadêmica / portfólio; respeita rate-limit)"
)

# Páginas realmente publicadas para comps/14 (conferido em 2026-08-22).
# passing / defense / possession / gca NÃO existem nesta competição.
PAGES: Dict[str, Dict[str, str]] = {
    "stats": {
        "path": "stats",
        "slug": f"{COMP_SLUG}-Stats",
        "player_table": "stats_standard",
        "squad_for": "stats_squads_standard_for",
        "squad_against": "stats_squads_standard_against",
    },
    "shooting": {
        "path": "shooting",
        "slug": f"{COMP_SLUG}-Stats",
        "player_table": "stats_shooting",
        "squad_for": "stats_squads_shooting_for",
        "squad_against": "stats_squads_shooting_against",
    },
    "playingtime": {
        "path": "playingtime",
        "slug": f"{COMP_SLUG}-Stats",
        "player_table": "stats_playing_time",
        "squad_for": "stats_squads_playing_time_for",
        "squad_against": "stats_squads_playing_time_against",
    },
    "misc": {
        "path": "misc",
        "slug": f"{COMP_SLUG}-Stats",
        "player_table": "stats_misc",
        "squad_for": "stats_squads_misc_for",
        "squad_against": "stats_squads_misc_against",
    },
    "keepers": {
        "path": "keepers",
        "slug": f"{COMP_SLUG}-Stats",
        "player_table": "stats_keeper",
        "squad_for": "stats_squads_keeper_for",
        "squad_against": "stats_squads_keeper_against",
    },
    "schedule": {
        "path": "schedule",
        "slug": f"{COMP_SLUG}-Scores-and-Fixtures",
        "player_table": "",
        "squad_for": "",
        "squad_against": "",
    },
}

DEFAULT_PAGES = ("stats", "shooting", "misc", "playingtime", "keepers", "schedule")

PLAYER_ID_RE = re.compile(r"/en/players/([0-9a-f]+)/", re.I)
SQUAD_ID_RE = re.compile(r"/en/squads/([0-9a-f]+)/", re.I)
NATION_RE = re.compile(r"\b([A-Z]{3})\b")

# data-stat da FBref → coluna canônica em português.
STAT_MAP: Dict[str, str] = {
    "player": "jogador",
    "nationality": "nacao",
    "position": "posicao",
    "team": "time",
    "squad": "time",
    "age": "idade",
    "born": "ano_nasc",
    "games": "jogos",
    "games_starts": "titular",
    "minutes": "minutos",
    "minutes_90s": "noventas",
    "minutes_per_game": "minutos_por_jogo",
    "minutes_pct": "minutos_pct",
    "games_complete": "jogos_completos",
    "games_subs": "entradas",
    "unused_subs": "banco_nao_usado",
    "points_per_game": "pontos_por_jogo",
    "on_goals_for": "gols_time_em_campo",
    "on_goals_against": "gols_sofridos_em_campo",
    "plus_minus": "saldo_em_campo",
    "plus_minus_per90": "saldo_em_campo_90",
    "goals": "gols",
    "assists": "assistencias",
    "goals_assists": "gols_mais_assistencias",
    "goals_pens": "gols_sem_penalti",
    "pens_made": "penaltis",
    "pens_att": "penaltis_tentados",
    "cards_yellow": "cartoes_amarelos",
    "cards_red": "cartoes_vermelhos",
    "cards_yellow_red": "dois_amarelos",
    "goals_per90": "gols_90",
    "assists_per90": "assistencias_90",
    "goals_assists_per90": "gols_mais_assistencias_90",
    "goals_pens_per90": "gols_sem_penalti_90",
    "xg": "xg",
    "npxg": "xg_sem_penalti",
    "xg_assist": "xa",
    "shots": "finalizacoes",
    "shots_on_target": "finalizacoes_no_gol",
    "shots_on_target_pct": "pct_no_gol",
    "shots_per90": "finalizacoes_90",
    "shots_on_target_per90": "finalizacoes_no_gol_90",
    "goals_per_shot": "gols_por_finalizacao",
    "goals_per_shot_on_target": "gols_por_finalizacao_no_gol",
    "average_shot_distance": "distancia_media_chute",
    "shots_free_kicks": "faltas_cobradas",
    "fouls": "faltas",
    "fouled": "faltas_sofridas",
    "offsides": "impedimentos",
    "crosses": "cruzamentos",
    "interceptions": "interceptacoes",
    "tackles_won": "desarmes_ganhos",
    "pens_won": "penaltis_ganhos",
    "pens_conceded": "penaltis_sofridos",
    "own_goals": "gols_contra",
    "ball_recoveries": "recuperacoes",
    "aerials_won": "aereos_ganhos",
    "aerials_lost": "aereos_perdidos",
    "aerials_won_pct": "pct_aereos",
    "gk_games": "jogos_gk",
    "gk_games_starts": "titular_gk",
    "gk_minutes": "minutos_gk",
    "gk_goals_against": "gols_sofridos_gk",
    "gk_goals_against_per90": "gols_sofridos_gk_90",
    "gk_shots_on_target_against": "chutes_no_gol_sofridos",
    "gk_saves": "defesas",
    "gk_save_pct": "pct_defesas",
    "gk_clean_sheets": "jogos_sem_sofrer",
    "gk_clean_sheets_pct": "pct_jogos_sem_sofrer",
    "gk_pens_att": "penaltis_enfrentados",
    "gk_pens_allowed": "penaltis_sofridos_gk",
    "gk_pens_saved": "penaltis_defendidos",
    "possession": "posse",
    "avg_age": "idade_media",
    "players_used": "n_jogadores",
    # schedule
    "date": "data",
    "start_time": "horario",
    "round": "rodada",
    "dayofweek": "dia_semana",
    "venue": "local",
    "result": "resultado_fbref",
    "goals_for": "gols_mandante",
    "goals_against": "gols_visitante",
    "opponent": "visitante",
    "home_team": "mandante",
    "away_team": "visitante",
    "score": "placar",
    "attendance": "publico",
    "referee": "arbitro",
    "xg_for": "xg_mandante",
    "xg_against": "xg_visitante",
    "notes": "observacao",
}

# IDs de elenco FBref → nome canônico do dashboard (src/real_data.py).
SQUAD_ID_CANONICO: Dict[str, str] = {
    "639950ae": "Flamengo",
    "abdce579": "Palmeiras",
    "bf4acd28": "Corinthians",
    "84d9701c": "Fluminense",
    "03ff5eeb": "Cruzeiro",
    "289e8847": "Mirassol",
    "df734df9": "Estudiantes",
    "3cbfa767": "Platense",
    "8a9d5afa": "Independiente Rivadavia",
    "87a920fa": "Rosario Central",
    "795ca75e": "Boca Juniors",
    "11b6dba8": "Lanús",
    "1284d3f9": "LDU",
    "990519b8": "Independiente del Valle",
    "8c71aef1": "Barcelona SC",
    "9c4c0cc1": "Tolima",
    "70068101": "Medellín",
    "b281fa3b": "Junior",
    "a7854d10": "Santa Fe",
    "85c3a70f": "Coquimbo Unido",
    "3e3fbf36": "Universidad Católica",
    "e4cd6f9a": "Cerro Porteño",
    "ae107695": "Libertad",
    "26ebba72": "Nacional",
    "e2d73ee6": "Peñarol",
    "8d727f54": "Always Ready",
    "e69cb5b6": "Bolívar",
    "d4f8af71": "Cusco FC",
    "8917b8a9": "Sporting Cristal",
    "e4108102": "Universitario",
    "0f8ef17f": "Deportivo La Guaira",
    "d254c5db": "Universidad Central (VEN)",
}

_NAME_ALIASES: Dict[str, str] = {
    "estudiantes lp": "Estudiantes",
    "estudiantes-lp": "Estudiantes",
    "ind rivadavia": "Independiente Rivadavia",
    "independiente rivadavia": "Independiente Rivadavia",
    "ldu quito": "LDU",
    "u catolica": "Universidad Católica",
    "universidad catolica": "Universidad Católica",
    "ucv": "Universidad Central (VEN)",
    "dep la guaira": "Deportivo La Guaira",
    "deportivo la guaira": "Deportivo La Guaira",
    "ind medellin": "Medellín",
    "independiente medellin": "Medellín",
    "deportes tolima": "Tolima",
    "independiente": "Independiente del Valle",  # FBref encurta o IDV
    "barcelona": "Barcelona SC",
}


# --------------------------------------------------------------------------- #
# Erros
# --------------------------------------------------------------------------- #
class FBrefError(Exception):
    """Erro base do scraper FBref."""


class FBrefRateLimitError(FBrefError):
    """HTTP 429 persistente."""


class FBrefUnavailableError(FBrefError):
    """Rede/SSL/HTML indisponível e sem snapshot para cair."""


# --------------------------------------------------------------------------- #
# Parsing de HTML
# --------------------------------------------------------------------------- #
def unwrap_commented_tables(soup: BeautifulSoup) -> BeautifulSoup:
    """A FBref esconde tabelas grandes em comentários HTML — desembrulha."""
    for comment in list(soup.find_all(string=lambda t: isinstance(t, Comment))):
        text = str(comment)
        if "<table" not in text.lower():
            continue
        fragment = BeautifulSoup(f"<div>{text}</div>", "html.parser")
        wrapper = fragment.div
        if wrapper is None:
            continue
        comment.replace_with(wrapper)
    return soup


def _cell_href(cell) -> Optional[str]:
    link = cell.find("a") if hasattr(cell, "find") else None
    if link and link.get("href"):
        return link["href"]
    return None


def _cell_text(cell) -> str:
    return " ".join(cell.get_text(" ", strip=True).split())


def _to_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text in {"", "-", "–", "—"}:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return value


def parse_stats_table(table) -> List[Dict[str, Any]]:
    """Converte uma ``<table>`` FBref (com ``data-stat``) em lista de dicts."""
    header_cells = []
    thead = table.find("thead")
    if thead:
        header_rows = thead.find_all("tr")
        for row in reversed(header_rows):
            cells = row.find_all(["th", "td"])
            if any(c.get("data-stat") for c in cells):
                header_cells = cells
                break
    if not header_cells:
        first = table.find("tr")
        header_cells = first.find_all(["th", "td"]) if first else []

    keys: List[str] = []
    for cell in header_cells:
        stat = (cell.get("data-stat") or "").strip()
        keys.append(stat or _cell_text(cell).lower().replace(" ", "_"))

    rows: List[Dict[str, Any]] = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr", recursive=False) or tbody.find_all("tr"):
        classes = tr.get("class") or []
        if any(c in {"thead", "over_header", "spacer", "hidden"} for c in classes):
            continue
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        record: Dict[str, Any] = {}
        for idx, cell in enumerate(cells):
            stat = (cell.get("data-stat") or "").strip()
            if not stat and idx < len(keys):
                stat = keys[idx]
            if not stat or stat in {"ranker", "matches", "header"}:
                continue
            text = _cell_text(cell)
            href = _cell_href(cell)
            col = STAT_MAP.get(stat, stat)
            if stat in {"player", "team", "squad", "home_team", "away_team", "opponent"}:
                record[col] = text
                if href and PLAYER_ID_RE.search(href):
                    record["jogador_id"] = PLAYER_ID_RE.search(href).group(1)
                if href and SQUAD_ID_RE.search(href):
                    # time da linha (elencos) ou time do jogador
                    if stat in {"team", "squad"}:
                        record["time_id"] = SQUAD_ID_RE.search(href).group(1)
                    elif stat == "home_team":
                        record["mandante_id"] = SQUAD_ID_RE.search(href).group(1)
                    elif stat in {"away_team", "opponent"}:
                        record["visitante_id"] = SQUAD_ID_RE.search(href).group(1)
                if stat == "nationality":
                    m = NATION_RE.search(text)
                    record["nacao"] = m.group(1) if m else text
            elif stat == "nationality":
                m = NATION_RE.search(text)
                record[col] = m.group(1) if m else text
            else:
                record[col] = _to_number(text)
        if record.get("jogador") == "Player" or record.get("time") == "Squad":
            continue
        if record:
            rows.append(record)
    return rows


def extract_tables(html: str) -> Dict[str, List[Dict[str, Any]]]:
    """Devolve ``{table_id: [rows...]}`` a partir do HTML bruto da FBref."""
    soup = BeautifulSoup(html, "html.parser")
    unwrap_commented_tables(soup)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for table in soup.find_all("table"):
        tid = (table.get("id") or "").strip()
        if not tid:
            caption = table.find("caption")
            tid = (caption.get_text(" ", strip=True) if caption else "") or f"table_{len(out)}"
        rows = parse_stats_table(table)
        if rows:
            out[tid] = rows
    return out


def page_url(page: str, season: Optional[int] = None, current_season: int = 2026) -> str:
    """Monta a URL da FBref para uma página/temporada."""
    meta = PAGES[page]
    if season is None or int(season) == int(current_season):
        return f"{BASE_URL}/en/comps/{COMP_ID}/{meta['path']}/{meta['slug']}"
    year = int(season)
    prefix = f"{year}-" if page != "schedule" else f"{year}-"
    slug = meta["slug"]
    if page == "schedule":
        return (
            f"{BASE_URL}/en/comps/{COMP_ID}/{year}/schedule/"
            f"{year}-{COMP_SLUG}-Scores-and-Fixtures"
        )
    return f"{BASE_URL}/en/comps/{COMP_ID}/{year}/{meta['path']}/{prefix}{slug}"


def canonical_squad(name: Optional[str], squad_id: Optional[str] = None) -> str:
    """Nome curto alinhado ao dashboard (``real_data.short_name``)."""
    if squad_id and squad_id in SQUAD_ID_CANONICO:
        return SQUAD_ID_CANONICO[squad_id]
    if not name:
        return ""
    cleaned = re.sub(r"^(vs\.?\s+)", "", str(name), flags=re.I).strip()
    key = (
        cleaned.lower()
        .replace("–", "-")
        .replace("—", "-")
    )
    key = re.sub(r"[^a-z0-9 ]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    try:
        from real_data import short_name

        return short_name(cleaned)
    except Exception:
        return cleaned


def _is_against_table(table_id: str, rows: Sequence[Dict[str, Any]]) -> bool:
    if "against" in table_id:
        return True
    if rows and str(rows[0].get("time") or "").lower().startswith("vs"):
        return True
    return False


def _annotate(rows: List[Dict[str, Any]], temporada: int, fonte_pagina: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["temporada"] = temporada
        item["fonte"] = "fbref"
        item["pagina"] = fonte_pagina
        time_id = item.get("time_id")
        if item.get("time"):
            item["time_fbref"] = item["time"]
            item["time"] = canonical_squad(item["time"], time_id)
            item["time_canonico"] = item["time"]
        if item.get("mandante"):
            item["mandante"] = canonical_squad(item["mandante"], item.get("mandante_id"))
        if item.get("visitante"):
            item["visitante"] = canonical_squad(item["visitante"], item.get("visitante_id"))
        out.append(item)
    return out


def merge_on_keys(
    frames: Sequence[pd.DataFrame],
    keys: Sequence[str],
) -> pd.DataFrame:
    """Outer-join de várias tabelas da FBref no mesmo grão."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame()
    merged = usable[0]
    for extra in usable[1:]:
        overlap = [c for c in extra.columns if c in merged.columns and c not in keys]
        extra = extra.drop(columns=overlap, errors="ignore")
        merged = merged.merge(extra, on=list(keys), how="outer")
    return merged


def split_squad_tables(
    tables: Dict[str, List[Dict[str, Any]]],
    temporada: int,
    page: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa jogadores / elencos (for) / elencos (against) / partidas."""
    players: List[Dict[str, Any]] = []
    squads_for: List[Dict[str, Any]] = []
    squads_against: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []

    for tid, rows in tables.items():
        if not rows:
            continue
        low = tid.lower()
        if low.startswith("sched") or "scores" in low:
            matches.extend(_annotate(rows, temporada, page))
            continue
        sample = rows[0]
        is_player = "jogador" in sample or "jogador_id" in sample
        if is_player and "time" in sample:
            players.extend(_annotate(rows, temporada, page))
            continue
        if "time" in sample or "n_jogadores" in sample or "posse" in sample:
            tagged = _annotate(rows, temporada, page)
            if _is_against_table(tid, rows):
                squads_against.extend(tagged)
            else:
                squads_for.extend(tagged)

    return (
        pd.DataFrame(players),
        pd.DataFrame(squads_for),
        pd.DataFrame(squads_against),
        pd.DataFrame(matches),
    )


def prefix_against(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia métricas do bloco 'Opponent Stats' com sufixo ``_sofrido``."""
    if df.empty:
        return df
    keep = {
        "temporada", "time", "time_id", "time_canonico", "time_fbref",
        "fonte", "pagina", "nacao", "idade", "idade_media", "n_jogadores",
        "jogos", "noventas", "minutos", "titular",
    }
    rename = {c: f"{c}_sofrido" for c in df.columns if c not in keep}
    return df.rename(columns=rename)


# --------------------------------------------------------------------------- #
# Persistência
# --------------------------------------------------------------------------- #
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS elencos (
    temporada INTEGER NOT NULL,
    time_id TEXT NOT NULL,
    time TEXT,
    time_canonico TEXT,
    n_jogadores REAL,
    idade_media REAL,
    posse REAL,
    jogos REAL,
    minutos REAL,
    noventas REAL,
    gols REAL,
    assistencias REAL,
    gols_sofrido REAL,
    finalizacoes REAL,
    finalizacoes_no_gol REAL,
    desarmes_ganhos REAL,
    interceptacoes REAL,
    faltas REAL,
    cartoes_amarelos REAL,
    cartoes_vermelhos REAL,
    fonte TEXT,
    PRIMARY KEY (temporada, time_id)
);

CREATE TABLE IF NOT EXISTS jogadores (
    temporada INTEGER NOT NULL,
    jogador_id TEXT NOT NULL,
    time_id TEXT,
    jogador TEXT,
    nacao TEXT,
    posicao TEXT,
    time TEXT,
    jogos REAL,
    titular REAL,
    minutos REAL,
    noventas REAL,
    gols REAL,
    assistencias REAL,
    finalizacoes REAL,
    finalizacoes_no_gol REAL,
    desarmes_ganhos REAL,
    interceptacoes REAL,
    faltas REAL,
    cartoes_amarelos REAL,
    cartoes_vermelhos REAL,
    fonte TEXT,
    PRIMARY KEY (temporada, jogador_id, time_id)
);

CREATE TABLE IF NOT EXISTS partidas (
    temporada INTEGER NOT NULL,
    data TEXT,
    rodada TEXT,
    mandante TEXT,
    visitante TEXT,
    gols_mandante REAL,
    gols_visitante REAL,
    placar TEXT,
    arbitro TEXT,
    local TEXT,
    fonte TEXT
);
"""


def save_sqlite(
    elencos: pd.DataFrame,
    jogadores: pd.DataFrame,
    partidas: pd.DataFrame,
    path: Path = SQLITE_PATH,
) -> Path:
    """Grava o cruzamento temporada → time → jogador em SQLite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SQLITE_SCHEMA)
        if not elencos.empty:
            cols = [c for c in _table_columns(conn, "elencos") if c in elencos.columns]
            elencos[cols].to_sql("elencos", conn, if_exists="replace", index=False)
        if not jogadores.empty:
            cols = [c for c in _table_columns(conn, "jogadores") if c in jogadores.columns]
            # recreate with whatever columns we have
            jogadores.to_sql("jogadores", conn, if_exists="replace", index=False)
        if not partidas.empty:
            partidas.to_sql("partidas", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jogadores_time "
            "ON jogadores (temporada, time_id)"
        )
    return path


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def save_csvs(
    elencos: pd.DataFrame,
    jogadores: pd.DataFrame,
    partidas: pd.DataFrame,
    out_dir: Path = PROCESSED_DIR,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elencos": out_dir / "fbref_elencos.csv",
        "jogadores": out_dir / "fbref_jogadores.csv",
        "partidas": out_dir / "fbref_partidas.csv",
    }
    if not elencos.empty:
        elencos.to_csv(paths["elencos"], index=False)
    if not jogadores.empty:
        jogadores.to_csv(paths["jogadores"], index=False)
    if not partidas.empty:
        partidas.to_csv(paths["partidas"], index=False)
    return paths


# --------------------------------------------------------------------------- #
# Cliente HTTP
# --------------------------------------------------------------------------- #
class FBrefClient:
    """Cliente educado para as páginas da Libertadores na FBref."""

    def __init__(
        self,
        sleep: float = DEFAULT_SLEEP,
        session: Optional[requests.Session] = None,
        cache_dir: Path = RAW_DIR,
        current_season: int = 2026,
    ):
        self.sleep = sleep
        self.cache_dir = Path(cache_dir)
        self.current_season = current_season
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8,pt-BR;q=0.7",
            }
        )
        self.last_call_ts: Optional[float] = None

    def _wait(self) -> None:
        if self.last_call_ts is None:
            return
        elapsed = time.monotonic() - self.last_call_ts
        wait = self.sleep - elapsed
        if wait > 0:
            time.sleep(wait)

    def cache_path(self, page: str, season: int) -> Path:
        return self.cache_dir / f"{season}_{page}.html"

    def fetch_html(
        self,
        page: str,
        season: int,
        use_cache: bool = True,
        refresh: bool = False,
    ) -> str:
        """Baixa (ou lê do cache) o HTML de uma página."""
        path = self.cache_path(page, season)
        if use_cache and not refresh and path.exists():
            return path.read_text(encoding="utf-8", errors="replace")

        url = page_url(page, season, self.current_season)
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait()
            try:
                resp = self.session.get(url, timeout=TIMEOUT)
                self.last_call_ts = time.monotonic()
            except requests.RequestException as exc:
                last_exc = exc
                print(f"  ⚠️  FBref rede ({page}, tentativa {attempt}/{MAX_RETRIES}): {exc}")
                # TLS/SSL fechado pelo provedor não se recupera com retry.
                if isinstance(exc, requests.exceptions.SSLError) or "SSL" in str(exc):
                    raise FBrefUnavailableError(f"TLS/SSL ao falar com a FBref: {exc}") from exc
                time.sleep(2 * attempt)
                continue

            if resp.status_code == 200 and resp.text:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(resp.text, encoding="utf-8")
                return resp.text
            if resp.status_code == 429:
                retry = int(resp.headers.get("Retry-After") or 60)
                print(f"  ⚠️  FBref 429 — aguardando {retry}s…")
                time.sleep(retry)
                last_exc = FBrefRateLimitError("429")
                continue
            last_exc = FBrefError(f"HTTP {resp.status_code} em {url}")
            print(f"  ⚠️  {last_exc}")
            time.sleep(2 * attempt)

        raise FBrefUnavailableError(
            f"Não foi possível baixar {url}: {last_exc}"
        )

    def parse_page(self, html: str, page: str, season: int) -> Dict[str, pd.DataFrame]:
        tables = extract_tables(html)
        players, squads_for, squads_against, matches = split_squad_tables(
            tables, season, page
        )
        return {
            "jogadores": players,
            "elencos": squads_for,
            "elencos_sofrido": prefix_against(squads_against),
            "partidas": matches,
        }

    def scrape(
        self,
        season: int = 2026,
        pages: Sequence[str] = DEFAULT_PAGES,
        use_cache: bool = True,
        refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Raspa as páginas pedidas e devolve dataframes mesclados."""
        unknown = [p for p in pages if p not in PAGES]
        if unknown:
            raise ValueError(f"Páginas desconhecidas: {unknown}. Válidas: {list(PAGES)}")

        jogadores_parts: List[pd.DataFrame] = []
        elencos_parts: List[pd.DataFrame] = []
        sofrido_parts: List[pd.DataFrame] = []
        partidas_parts: List[pd.DataFrame] = []

        for page in pages:
            print(f"  • FBref {page} ({season})…")
            html = self.fetch_html(page, season, use_cache=use_cache, refresh=refresh)
            parsed = self.parse_page(html, page, season)
            if not parsed["jogadores"].empty:
                jogadores_parts.append(parsed["jogadores"])
            if not parsed["elencos"].empty:
                elencos_parts.append(parsed["elencos"])
            if not parsed["elencos_sofrido"].empty:
                sofrido_parts.append(parsed["elencos_sofrido"])
            if not parsed["partidas"].empty:
                partidas_parts.append(parsed["partidas"])

        jogadores = merge_on_keys(
            jogadores_parts, keys=["temporada", "jogador_id", "time_id"]
        )
        elencos = merge_on_keys(elencos_parts, keys=["temporada", "time_id"])
        sofrido = merge_on_keys(sofrido_parts, keys=["temporada", "time_id"])
        if not elencos.empty and not sofrido.empty:
            elencos = merge_on_keys([elencos, sofrido], keys=["temporada", "time_id"])
        partidas = (
            pd.concat(partidas_parts, ignore_index=True) if partidas_parts else pd.DataFrame()
        )
        if not partidas.empty:
            partidas = partidas.drop_duplicates(
                subset=[c for c in ("temporada", "data", "mandante", "visitante") if c in partidas],
            )

        return {"jogadores": jogadores, "elencos": elencos, "partidas": partidas}

    def load_snapshot(self, season: int = 2026) -> Dict[str, pd.DataFrame]:
        """Carrega o snapshot versionado em ``data/historical/fbref/``."""
        elencos_path = HIST_DIR / f"elencos_{season}.csv"
        jogadores_path = HIST_DIR / f"jogadores_{season}.csv"
        partidas_path = HIST_DIR / f"partidas_{season}.csv"
        if not elencos_path.exists() and not jogadores_path.exists():
            raise FileNotFoundError(
                f"Snapshot FBref {season} não encontrado em {HIST_DIR}."
            )
        elencos = pd.read_csv(elencos_path) if elencos_path.exists() else pd.DataFrame()
        jogadores = pd.read_csv(jogadores_path) if jogadores_path.exists() else pd.DataFrame()
        partidas = pd.read_csv(partidas_path) if partidas_path.exists() else pd.DataFrame()
        return {"elencos": elencos, "jogadores": jogadores, "partidas": partidas}

    def _load_example(self) -> Dict[str, pd.DataFrame]:
        path = EXAMPLES_DIR / "fbref_elencos.csv"
        if path.exists():
            return {
                "elencos": pd.read_csv(path),
                "jogadores": pd.DataFrame(),
                "partidas": pd.DataFrame(),
            }
        return {"elencos": pd.DataFrame(), "jogadores": pd.DataFrame(), "partidas": pd.DataFrame()}

    @staticmethod
    def _permitir_exemplo() -> bool:
        return os.getenv("ALLOW_EXAMPLE_DATA", "").lower() in ("1", "true", "yes")

    def run(
        self,
        season: int = 2026,
        pages: Sequence[str] = DEFAULT_PAGES,
        use_cache: bool = True,
        refresh: bool = False,
        persist: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Orquestra raspagem + fallback + gravação CSV/SQLite."""
        print("=" * 60)
        print(f"📥 FBREF — elencos e jogadores da Libertadores {season}")
        print("=" * 60)
        print(
            "  ℹ️  Páginas disponíveis: stats, shooting, playingtime, misc, keepers, schedule."
        )
        print("  ℹ️  passing/defense/xG avançados NÃO são publicados para comps/14.")

        data: Optional[Dict[str, pd.DataFrame]] = None
        try:
            data = self.scrape(
                season=season, pages=pages, use_cache=use_cache, refresh=refresh
            )
            if data["elencos"].empty and data["jogadores"].empty:
                raise FBrefUnavailableError("HTML parseado sem tabelas úteis.")
            print(
                f"  ✅ Raspagem: {len(data['elencos'])} elencos, "
                f"{len(data['jogadores'])} jogadores, "
                f"{len(data['partidas'])} partidas."
            )
        except (FBrefError, requests.RequestException, OSError) as exc:
            print(f"  ⚠️  Raspagem ao vivo indisponível ({exc}).")
            try:
                data = self.load_snapshot(season)
                print(
                    f"  ✅ Snapshot histórico: {len(data['elencos'])} elencos, "
                    f"{len(data['jogadores'])} jogadores."
                )
            except FileNotFoundError as snap_exc:
                if self._permitir_exemplo():
                    data = self._load_example()
                    print("  ℹ️  ALLOW_EXAMPLE_DATA=1 — usando base de exemplo.")
                else:
                    raise FBrefUnavailableError(
                        "FBref inacessível e sem snapshot versionado. "
                        "Rode a raspagem quando a rede estiver disponível "
                        "ou use o snapshot em data/historical/fbref/."
                    ) from snap_exc

        assert data is not None
        if persist:
            paths = save_csvs(data["elencos"], data["jogadores"], data["partidas"])
            db = save_sqlite(data["elencos"], data["jogadores"], data["partidas"])
            print(f"  💾 CSV: {paths['elencos']}")
            print(f"  💾 SQLite: {db}")
        return data


def load_elencos(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or ELENCOS_CSV
    if path.exists():
        return pd.read_csv(path)
    snap = HIST_DIR / "elencos_2026.csv"
    if snap.exists():
        return pd.read_csv(snap)
    raise FileNotFoundError(
        "Elencos FBref não encontrados. Rode `python src/fbref_scraper.py scrape`."
    )


def load_jogadores(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or JOGADORES_CSV
    if path.exists():
        return pd.read_csv(path)
    snap = HIST_DIR / "jogadores_2026.csv"
    if snap.exists():
        return pd.read_csv(snap)
    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Raspagem FBref — Libertadores")
    sub = parser.add_subparsers(dest="command")

    scrape_p = sub.add_parser("scrape", help="Baixa e persiste as tabelas")
    scrape_p.add_argument("--season", type=int, default=2026)
    scrape_p.add_argument(
        "--pages",
        default=",".join(DEFAULT_PAGES),
        help="Lista separada por vírgula (stats,shooting,misc,…)",
    )
    scrape_p.add_argument("--from-cache", action="store_true")
    scrape_p.add_argument("--refresh", action="store_true")
    scrape_p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)

    parse_p = sub.add_parser("parse", help="Parseia um HTML local (teste/debug)")
    parse_p.add_argument("--html", required=True)
    parse_p.add_argument("--page", default="stats")
    parse_p.add_argument("--season", type=int, default=2026)

    args = parser.parse_args(argv)
    if args.command == "parse":
        html = Path(args.html).read_text(encoding="utf-8")
        tables = extract_tables(html)
        print(f"{len(tables)} tabelas: {list(tables)}")
        for tid, rows in tables.items():
            print(f"  - {tid}: {len(rows)} linhas, cols={list(rows[0]) if rows else []}")
        return 0

    if args.command in (None, "scrape"):
        pages = tuple(p.strip() for p in getattr(args, "pages", ",".join(DEFAULT_PAGES)).split(",") if p.strip())
        client = FBrefClient(sleep=getattr(args, "sleep", DEFAULT_SLEEP))
        client.run(
            season=getattr(args, "season", 2026),
            pages=pages,
            use_cache=True,
            refresh=getattr(args, "refresh", False) and not getattr(args, "from_cache", False),
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
