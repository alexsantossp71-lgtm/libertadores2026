"""
Raspagem de relatórios de partida FBref (comps/14, temporada 2026) para
estatísticas de arbitragem da página "Arbitragem" do dashboard.

Substitui a API-Futebol (plano grátis não cobre a Libertadores 2026) pela
FBref como fonte de faltas, cartões, posse, finalizações, escanteios,
impedimentos e defesas por partida.

Uso::

    python src/fbref_partidas.py build              # usa cache em data/raw/fbref
    python src/fbref_partidas.py build --refresh    # rebaixa schedule + reports

Política de rede: reutiliza ``FBrefClient`` de ``src/fbref_scraper.py``
(3,5 s entre requisições, User-Agent identificado, cache em
``data/raw/fbref/``). A FBref protege comps/14 com desafio Cloudflare para
acesso direto via ``requests``; quando isso ocorre, rode a coleta com o
navegador (Playwright) que preenche o cache — ver README interno no docstring
de ``fetch_missing``. Nada é sintetizado: estatísticas que a FBref não
publica para esta competição ficam como NA.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "fbref"
MATCHES_DIR = RAW_DIR / "matches"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
HISTORICAL_SCHEDULE = ROOT_DIR / "data" / "historical" / "partidas_libertadores.csv"
GRUPOS_CSV = ROOT_DIR / "data" / "raw" / "grupos_libertadores_2026.csv"
OUTPUT_CSV = PROCESSED_DIR / "libertadores_estatisticas_detalhadas.csv"

SCHEDULE_HTML = RAW_DIR / "2026_schedule.html"
MATCH_LINKS_TXT = RAW_DIR / "match_links_2026.txt"

CSV_COLUMNS: List[str] = [
    "partida_id", "data", "fase", "rodada", "grupo", "mandante", "visitante",
    "gols_mandante", "gols_visitante", "resultado", "arbitro", "arbitro_pais",
    "faltas_mandante", "faltas_visitante",
    "cartoes_amarelos_mandante", "cartoes_amarelos_visitante",
    "cartoes_vermelhos_mandante", "cartoes_vermelhos_visitante",
    "posse_mandante", "posse_visitante",
    "passes_certos_mandante", "passes_certos_visitante",
    "passes_errados_mandante", "passes_errados_visitante",
    "finalizacoes_mandante", "finalizacoes_visitante",
    "finalizacoes_no_gol_mandante", "finalizacoes_no_gol_visitante",
    "finalizacoes_fora_mandante", "finalizacoes_fora_visitante",
    "escanteios_mandante", "escanteios_visitante",
    "impedimentos_mandante", "impedimentos_visitante",
    "defesas_mandante", "defesas_visitante",
    "fonte",
]

MATCH_ID_RE = re.compile(r"/matches/([0-9a-f]{8})/")
SQUAD_ID_RE = re.compile(r"/squads/([0-9a-f]+)/")
SCORE_RE = re.compile(r"(\d+)\s*[–-]\s*(\d+)")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
OF_RE = re.compile(r"(\d+)\s+of\s+(\d+)")
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}

# rótulo da coluna "round" da tabela schedule -> (fase canônica do projeto)
FASE_MAP = {
    "group stage": "Fase de Grupos",
    "round of 16": "Oitavas de Final",
    "quarter-finals": "Quartas de Final",
    "semi-finals": "Semifinais",
    "final": "Final",
}

# squad_id FBref -> nome do calendário do projeto, quando a FBref encurta o
# nome de forma diferente do _CANONICAL ("Medellín", "Nacional").
SQUAD_ID_PROJETO: Dict[str, str] = {
    "70068101": "Independiente Medellín",
    "26ebba72": "Nacional (URU)",
}


class FBrefParseError(Exception):
    """HTML de relatório fora do formato esperado (provável bloqueio)."""


# --------------------------------------------------------------------------- #
# Helpers de parsing
# --------------------------------------------------------------------------- #
def _text(node) -> str:
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    m = re.search(r"-?\d+", str(value))
    return int(m.group(0)) if m else None


def _parse_fbref_date(text: str) -> Optional[str]:
    """'Tuesday April 7, 2026' -> '2026-04-07'."""
    m = re.search(r"([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})", text or "")
    if not m or m.group(1) not in MONTHS:
        return None
    return f"{int(m.group(3)):04d}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"


def parse_match_report(html: str, match_id: Optional[str] = None) -> Dict[str, Any]:
    """Extrai do relatório FBref os campos de arbitragem/estatísticas."""
    if "Just a moment" in html and "team_stats" not in html:
        raise FBrefParseError("HTML é desafio Cloudflare, não relatório.")

    soup = BeautifulSoup(html, "html.parser")
    row: Dict[str, Any] = {"match_id": match_id, "fonte": "fbref"}

    # --- scorebox: times, placar ------------------------------------------------
    scorebox = soup.select_one(".scorebox")
    if not scorebox:
        raise FBrefParseError("sem .scorebox")
    teams = []
    for block in scorebox.find_all("div", recursive=False):
        strong = block.find("strong")
        link = strong.find("a") if strong else None
        score = block.select_one(".score")
        if link and score:
            teams.append((link.get_text(strip=True), score.get_text(strip=True),
                          SQUAD_ID_RE.search(link.get("href", "") or "")))
    if len(teams) < 2:
        # fallback: dois primeiros strong>a com .score vizinho
        raise FBrefParseError("scorebox sem dois times")
    (home, home_score, home_id), (away, away_score, away_id) = teams[0], teams[1]
    row.update({
        "mandante_fbref": home,
        "visitante_fbref": away,
        "mandante_id": home_id.group(1) if home_id else None,
        "visitante_id": away_id.group(1) if away_id else None,
        "mandante": _canonical(home, home_id.group(1) if home_id else None),
        "visitante": _canonical(away, away_id.group(1) if away_id else None),
        "gols_mandante": _parse_int(home_score),
        "gols_visitante": _parse_int(away_score),
    })
    if row["gols_mandante"] is not None and row["gols_visitante"] is not None:
        h, a = row["gols_mandante"], row["gols_visitante"]
        row["resultado"] = "mandante" if h > a else "visitante" if a > h else "empate"

    # --- scorebox_meta: data + árbitro -----------------------------------------
    meta = soup.select_one(".scorebox_meta")
    if meta:
        for strong in meta.find_all("strong"):
            d = _parse_fbref_date(strong.get_text())
            if d:
                row["data"] = d
                break
        for a in meta.find_all("a"):
            tail = a.next_sibling or ""
            if "(Referee)" in str(tail):
                row["arbitro"] = a.get_text(" ", strip=True)
                break
        if "arbitro" not in row:
            m = re.search(r"Officials\s*:?\s*([^\u00b7(]+?)\s*\(Referee\)", _text(meta))
            if m:
                row["arbitro"] = m.group(1).strip()

    # --- bloco Team Stats -------------------------------------------------------
    team_stats = soup.select_one("#team_stats")
    if team_stats:
        rows = team_stats.find_all("tr")
        for idx, tr in enumerate(rows):
            th = tr.find("th")
            label = _text(th)
            if not label:
                continue
            next_row = rows[idx + 1] if idx + 1 < len(rows) else None
            tds = next_row.find_all("td") if next_row else []
            if len(tds) < 2:
                continue
            home_td, away_td = tds[0], tds[1]
            low = label.lower()
            if low == "possession":
                home_m = PCT_RE.search(_text(home_td))
                away_m = PCT_RE.search(_text(away_td))
                if home_m:
                    row["posse_mandante"] = float(home_m.group(1))
                if away_m:
                    row["posse_visitante"] = float(away_m.group(1))
            elif low == "shots on target":
                for side, td in (("mandante", home_td), ("visitante", away_td)):
                    m = OF_RE.search(_text(td))
                    if m:
                        on_target, total = int(m.group(1)), int(m.group(2))
                        row[f"finalizacoes_no_gol_{side}"] = on_target
                        row[f"finalizacoes_{side}"] = total
                        row[f"finalizacoes_fora_{side}"] = total - on_target
            elif low == "saves":
                for side, td in (("mandante", home_td), ("visitante", away_td)):
                    m = OF_RE.search(_text(td))
                    if m:
                        row[f"defesas_{side}"] = int(m.group(1))
            elif low == "cards":
                for side, td in (("mandante", home_td), ("visitante", away_td)):
                    yellow = red = 0
                    for icon in td.select(".card_icon, [class$=_card]"):
                        classes = " ".join(icon.get("class") or [])
                        if "yellow_red" in classes:
                            red += 1
                        elif "yellow_card" in classes:
                            yellow += 1
                        elif "red_card" in classes:
                            red += 1
                    row[f"cartoes_amarelos_{side}"] = yellow
                    row[f"cartoes_vermelhos_{side}"] = red

    # --- team_stats_extra: faltas / escanteios / impedimentos -------------------
    extra = soup.select_one("#team_stats_extra")
    if extra:
        for block in extra.find_all("div", recursive=False):
            cells = [c for c in block.find_all("div", recursive=False)
                     if "th" not in (c.get("class") or [])]
            for i in range(0, len(cells) - 2, 3):
                value_h, label, value_a = cells[i], cells[i + 1], cells[i + 2]
                label_text = _text(label).lower()
                mapa = {
                    "fouls": "faltas",
                    "corners": "escanteios",
                    "offsides": "impedimentos",
                }
                dest = mapa.get(label_text)
                if dest:
                    home_v = _parse_int(_text(value_h))
                    away_v = _parse_int(_text(value_a))
                    if home_v is not None:
                        row[f"{dest}_mandante"] = home_v
                    if away_v is not None:
                        row[f"{dest}_visitante"] = away_v

    return row


def parse_schedule_html(html: str) -> Dict[str, Dict[str, Any]]:
    """Tabelas sched_* (todas as fases) -> {match_id: {data, round, arbitro, ...}}."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", id=lambda v: v and v.startswith("sched"))
    out: Dict[str, Dict[str, Any]] = {}
    for table in tables:
        for tr in table.select("tbody tr"):
            _parse_schedule_row(tr, out)
    return out


def _parse_schedule_row(tr, out: Dict[str, Dict[str, Any]]) -> None:
    cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"])}
    if "score" not in cells:
        return
    link = None
    for stat in ("match_report", "score"):
        a = cells.get(stat).find("a") if cells.get(stat) else None
        if a and MATCH_ID_RE.search(a.get("href", "")):
            link = MATCH_ID_RE.search(a.get("href", ""))
            break
    if not link:
        return

    def txt(stat: str) -> str:
        c = cells.get(stat)
        return _text(c) if c else ""

    row: Dict[str, Any] = {
        "match_id": link.group(1),
        "data": txt("date"),
        "fase_fbref": txt("round"),
        "gameweek": _parse_int(txt("gameweek")),
        "arbitro_schedule": txt("referee") or None,
        "local": txt("venue") or None,
    }
    score_m = SCORE_RE.search(txt("score"))
    if score_m:
        row["gols_mandante_sched"] = int(score_m.group(1))
        row["gols_visitante_sched"] = int(score_m.group(2))
    for side, stat in (("mandante", "home_team"), ("visitante", "away_team")):
        c = cells.get(stat)
        if c is None:
            continue
        a = c.find("a")
        if a:
            idm = SQUAD_ID_RE.search(a.get("href", ""))
            row[f"{side}_id_sched"] = idm.group(1) if idm else None
            row[f"{side}_sched"] = a.get_text(strip=True)
    existing = out.get(link.group(1))
    if existing is None:
        out[link.group(1)] = row
    else:
        for key, value in row.items():
            if value not in (None, ""):
                existing[key] = value


# --------------------------------------------------------------------------- #
# Montagem do DataFrame final
# --------------------------------------------------------------------------- #
def _grupo_lookup() -> Dict[str, str]:
    """{time_canônico: 'Grupo X'} a partir do sorteio em data/raw."""
    if not GRUPOS_CSV.exists():
        return {}
    df = pd.read_csv(GRUPOS_CSV)
    from src.fbref_scraper import canonical_squad

    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        time = canonical_squad(str(r["Time"]))
        out[time] = f"Grupo {r['Grupo']}"
    return out


def _canonical(name: Optional[str], squad_id: Optional[str]) -> str:
    if squad_id and squad_id in SQUAD_ID_PROJETO:
        return SQUAD_ID_PROJETO[squad_id]
    from src.fbref_scraper import canonical_squad

    return canonical_squad(name, squad_id)


def build_rows(reports: Sequence[Dict[str, Any]],
               schedule: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    """Converte relatórios parseados no schema de 37 colunas do dashboard."""
    schedule = schedule or {}
    grupos = _grupo_lookup()

    # desduplica por match_id
    seen: Dict[str, Dict[str, Any]] = {}
    for rep in reports:
        mid = rep.get("match_id")
        if not mid:
            continue
        seen[mid] = rep
    ordered = sorted(
        seen.values(),
        key=lambda r: (schedule.get(r["match_id"], {}).get("data") or r.get("data") or "",
                       r["match_id"]),
    )

    rows: List[Dict[str, Any]] = []
    for idx, rep in enumerate(ordered, start=1001):
        mid = rep["match_id"]
        sched = schedule.get(mid, {})
        fase_fbref = (sched.get("fase_fbref") or "").strip().lower()
        fase = FASE_MAP.get(fase_fbref, "Preliminar" if fase_fbref else None)
        gameweek = sched.get("gameweek")
        rodada = f"{gameweek}ª Rodada" if (fase == "Fase de Grupos" and gameweek) else None

        mandante = rep.get("mandante") or _canonical(
            rep.get("mandante_fbref"), rep.get("mandante_id")
            or sched.get("mandante_id_sched"))
        visitante = rep.get("visitante") or _canonical(
            rep.get("visitante_fbref"), rep.get("visitante_id")
            or sched.get("visitante_id_sched"))
        grupo = None
        if fase == "Fase de Grupos":
            g = grupos.get(mandante)
            if g and grupos.get(visitante) == g:
                grupo = g

        row: Dict[str, Any] = {c: None for c in CSV_COLUMNS}
        row.update({
            "partida_id": idx,
            "data": sched.get("data") or rep.get("data"),
            "fase": fase,
            "rodada": rodada,
            "grupo": grupo,
            "mandante": mandante,
            "visitante": visitante,
            "gols_mandante": rep.get("gols_mandante"),
            "gols_visitante": rep.get("gols_visitante"),
            "resultado": rep.get("resultado"),
            "arbitro": rep.get("arbitro") or sched.get("arbitro_schedule"),
            "arbitro_pais": None,  # FBref não publica nacionalidade do árbitro
            "fonte": "fbref",
        })
        for col in CSV_COLUMNS:
            if col.endswith("_mandante") or col.endswith("_visitante"):
                if col in rep and col not in row:
                    row[col] = rep[col]
                elif rep.get(col) is not None:
                    row[col] = rep[col]
        for stat in ("faltas", "cartoes_amarelos", "cartoes_vermelhos", "posse",
                     "finalizacoes", "finalizacoes_no_gol", "finalizacoes_fora",
                     "escanteios", "impedimentos", "defesas"):
            for side in ("mandante", "visitante"):
                key = f"{stat}_{side}"
                if rep.get(key) is not None:
                    row[key] = rep[key]
        rows.append(row)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    # a página de arbitragem precisa de faltas numéricas (polyfit); partidas
    # cuja FBref não publicou fouls são deixadas de fora do CSV final.
    incompletas = df["faltas_mandante"].isna() | df["faltas_visitante"].isna()
    if incompletas.any():
        drop = df.loc[incompletas, ["data", "mandante", "visitante"]]
        print(f"  Sem bloco de faltas na FBref (excluídas): {len(drop)}")
        for _, r in drop.iterrows():
            print(f"    - {r['data']} {r['mandante']} x {r['visitante']}")
        df = df.loc[~incompletas].reset_index(drop=True)
    return df


def load_cached_reports(matches_dir: Path = MATCHES_DIR) -> List[Dict[str, Any]]:
    """Parseia todos os relatórios em cache (data/raw/fbref/matches/*.html)."""
    reports: List[Dict[str, Any]] = []
    for path in sorted(matches_dir.glob("*.html")):
        mid = path.stem
        try:
            reports.append(parse_match_report(
                path.read_text(encoding="utf-8"), match_id=mid))
        except FBrefParseError as exc:
            print(f"  WARNING: {mid}: {exc}")
    return reports


def coverage_report(df: pd.DataFrame) -> None:
    """Cobertura das estatísticas vs. calendário 2026 versionado."""
    if not HISTORICAL_SCHEDULE.exists():
        return
    sched = pd.read_csv(HISTORICAL_SCHEDULE)
    sched26 = sched[sched["temporada"] == 2026]
    jogados = sched26[
        sched26["gols_mandante"].notna() & sched26["gols_visitante"].notna()
    ]
    pares_csv = {
        (str(r["data"]), str(r["nome_curto_mandante"]), str(r["nome_curto_visitante"]))
        for _, r in jogados.iterrows()
    }
    cobertos = sum(
        1 for _, r in df.iterrows()
        if (str(r["data"]), str(r["mandante"]), str(r["visitante"])) in pares_csv
        or (str(r["data"]), r["mandante"], r["visitante"]) in pares_csv
    )
    print(f"  Cobertura FBref: {len(df)} partidas com relatório; "
          f"{cobertos} batem com o calendário 2026 ({len(pares_csv)} jogadas registradas).")
    nan_cols = [c for c in df.columns
                if df[c].isna().all() and c not in ("grupo", "rodada", "fase")]
    if nan_cols:
        print(f"  Colunas sem cobertura FBref (NA): {nan_cols}")
    ref = df["arbitro"].notna().sum()
    print(f"  Árbitros identificados: {ref}/{len(df)}")


def build(save: bool = True) -> pd.DataFrame:
    """Pipeline completo: cache FBref -> CSV de estatísticas detalhadas."""
    if not SCHEDULE_HTML.exists():
        raise FileNotFoundError(
            f"Schedule 2026 não encontrado em {SCHEDULE_HTML}. "
            "Baixe https://fbref.com/en/comps/14/schedule/Copa-Libertadores-Scores-and-Fixtures "
            "(navegador) para o cache."
        )
    schedule = parse_schedule_html(SCHEDULE_HTML.read_text(encoding="utf-8"))
    print(f"  Schedule FBref: {len(schedule)} partidas com relatório linkado.")
    reports = load_cached_reports()
    print(f"  Relatórios em cache: {len(reports)}")
    df = build_rows(reports, schedule)
    df = df.sort_values(["data", "partida_id"]).reset_index(drop=True)
    df["partida_id"] = range(1001, 1001 + len(df))
    coverage_report(df)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  [OK] {OUTPUT_CSV} ({len(df)} linhas, {len(df.columns)} colunas)")
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Estatísticas de partida via FBref")
    parser.add_argument("command", nargs="?", default="build",
                        choices=["build", "parse-report", "parse-schedule"])
    parser.add_argument("--html", help="HTML local para os subcomandos parse-*")
    args = parser.parse_args(argv)

    if args.command == "parse-report":
        html = Path(args.html).read_text(encoding="utf-8")
        row = parse_match_report(html, match_id=Path(args.html).stem)
        print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "parse-schedule":
        html = Path(args.html).read_text(encoding="utf-8")
        sched = parse_schedule_html(html)
        print(f"{len(sched)} partidas com relatório")
        return 0

    build(save=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
