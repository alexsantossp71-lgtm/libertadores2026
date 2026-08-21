"""
Dados reais da Copa Libertadores (2012–2026) — carga, parsing e tabelas.

Fontes
------
* ``data/historical/openfootball/``: arquivos ``YYYY_copal.txt`` do projeto
  openfootball/south-america (domínio público) com todas as partidas de cada
  edição — classificatórias, fase de grupos e mata-mata.
* ``data/historical/espn/``: JSONs da API pública da ESPN usados para
  complementar placares mais recentes do que a última atualização do
  openfootball (ex.: playoffs de agosto/2026).

Este módulo **não gera nenhum dado simulado**: se um arquivo não existir,
ele falha com erro explícito.

Uso::

    python src/real_data.py build    # constrói data/historical/partidas_libertadores.csv

Principais funções
------------------
* :func:`load_partidas` — dataset completo de partidas (todas as edições);
* :func:`standings` — tabela de grupos **derivada das partidas** (garante
  ``sum(GP) == sum(GC)`` por construção);
* :func:`build_app_tables` — gera os três CSVs consumidos pelo dashboard
  (grupos, oitavas/playoffs e confrontos das quartas) a partir de dados reais.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
HIST_DIR = ROOT_DIR / "data" / "historical"
OPENFOOTBALL_DIR = HIST_DIR / "openfootball"
ESPN_DIR = HIST_DIR / "espn"
DATASET_PATH = HIST_DIR / "partidas_libertadores.csv"

# --------------------------------------------------------------------------- #
# Normalização de nomes de times
# --------------------------------------------------------------------------- #
# Prefixos formais frequentes na nomenclatura da openfootball/ESPN.
_PREFIXES = (
    "Club Atletico ", "CA ", "CR ", "SC ", "SE ", "EC ", "CSD ", "CD ", "CF ",
    "CS ", "CAR ", "FR ", "FC ", "Club ", "Deportivo ", "CD ",
)

# Apelidos canônicos usados pelo dashboard (nome curto, sem acento não importa).
_CANONICAL: Dict[str, str] = {
    "flamengo": "Flamengo",
    "flamengo rj": "Flamengo",
    "palmeiras": "Palmeiras",
    "corinthians": "Corinthians",
    "corinthians paulista": "Corinthians",
    "fluminense": "Fluminense",
    "fluminense fc": "Fluminense",
    "fluminense rj": "Fluminense",
    "mirassol": "Mirassol",
    "mirassol fc": "Mirassol",
    "cruzeiro": "Cruzeiro",
    "cruzeiro ec": "Cruzeiro",
    "palmeiras": "Palmeiras",
    "botafogo fr": "Botafogo",
    "gremio": "Grêmio",
    "internacional": "Internacional",
    "vasco da gama": "Vasco",
    "mirassol": "Mirassol",
    "bahia": "Bahia",
    "sao paulo": "São Paulo",
    "atletico mineiro": "Atlético Mineiro",
    "atletico paranaense": "Athletico-PR",
    "athletico paranaense": "Athletico-PR",
    "boca juniors": "Boca Juniors",
    "river plate": "River Plate",
    "racing club": "Racing",
    "racing": "Racing",
    "independiente": "Independiente",
    "velez sarsfield": "Vélez Sarsfield",
    "estudiantes de la plata": "Estudiantes",
    "estudiantes": "Estudiantes",
    "platense": "Platense",
    "rosario central": "Rosario Central",
    "talleres": "Talleres",
    "san lorenzo": "San Lorenzo",
    "lanus": "Lanús",
    "argentinos juniors": "Argentinos Juniors",
    "independiente rivadavia": "Independiente Rivadavia",
    "defensor sporting": "Defensor Sporting",
    "defensor sc": "Defensor Sporting",
    "nacional": "Nacional",
    "penarol": "Peñarol",
    "penarol de montevideo": "Peñarol",
    "cerro porteno": "Cerro Porteño",
    "olimpia": "Olimpia",
    "olimpia asuncion": "Olimpia",
    "libertad": "Libertad",
    "libertad asuncion": "Libertad",
    "guarani": "Guaraní",
    "nacional asuncion": "Nacional (PAR)",
    "bolivar": "Bolívar",
    "the strongest": "The Strongest",
    "club the strongest": "The Strongest",
    "always ready": "Always Ready",
    "blooming": "Blooming",
    "universitario de deportes": "Universitario",
    "universitario": "Universitario",
    "alianza lima": "Alianza Lima",
    "club alianza lima": "Alianza Lima",
    "melgar": "Melgar",
    "fbc melgar": "Melgar",
    "cristal": "Sporting Cristal",
    "sporting cristal": "Sporting Cristal",
    "cienciano": "Cienciano",
    "cusco": "Cusco FC",
    "cscyd cusco fc": "Cusco FC",
    "universidad catolica": "Universidad Católica",
    "universidad catolica de chile": "Universidad Católica",
    "colo colo": "Colo-Colo",
    "csd colo colo": "Colo-Colo",
    "u de chile": "Universidad de Chile",
    "universidad de chile": "Universidad de Chile",
    "coquimbo unido": "Coquimbo Unido",
    "nublense": "Ñublense",
    "cd nublense": "Ñublense",
    "huachipato": "Huachipato",
    "palestino": "Palestino",
    "iquique": "Iquique",
    "cd iquique": "Iquique",
    "audax italiano": "Audax Italiano",
    "national": "Nacional (URU)",
    "ldu de quito": "LDU",
    "ldu quito": "LDU",
    "liga de quito": "LDU",
    "ldu": "LDU",
    "independiente del valle": "Independiente del Valle",
    "barcelona": "Barcelona SC",
    "barcelona sc": "Barcelona SC",
    "emelec": "Emelec",
    "el nacional": "El Nacional",
    "cscyd el nacional": "El Nacional",
    "liga mc": "Liga de Quito",
    "deportivo cuenca": "Deportivo Cuenca",
    "atletico nacional": "Atlético Nacional",
    "america de cali": "América de Cali",
    "america de cali sadr": "América de Cali",
    "junior": "Junior",
    "junior fc": "Junior",
    "milionarios": "Millonarios",
    "millonarios": "Millonarios",
    "santa fe": "Santa Fe",
    "independiente santa fe": "Santa Fe",
    "tolima": "Tolima",
    "cd tolima": "Tolima",
    "deportes tolima": "Tolima",
    "once caldas": "Once Caldas",
    "bucaramanga": "Bucaramanga",
    "ca bucaramanga": "Bucaramanga",
    "medellin": "Medellín",
    "atletico nacional sadr": "Atlético Nacional",
    "carabobo": "Carabobo",
    "carabobo fc": "Carabobo",
    "monagas": "Monagas",
    "monagas sc": "Monagas",
    "deportivo tachira": "Deportivo Táchira",
    "deportivo tachira fc": "Deportivo Táchira",
    "universidad central de venezuela": "Universidad Central (VEN)",
    "universidad central": "Universidad Central (VEN)",
    "portuguesa": "Portuguesa",
    "zamora": "Zamora",
    "boston river": "Boston River",
    "ca boston river": "Boston River",
    "liverpool montevideo": "Liverpool (URU)",
    "liverpool uru": "Liverpool (URU)",
    "progreso": "Progreso",
    "montevideo city": "Montevideo City",
    "danubio": "Danubio",
    "san antonio bulo bulo": "San Antonio Bulo Bulo",
    "always ready la paz": "Always Ready",
    "real potosi": "Real Potosí",
    "sport boys": "Sport Boys",
    "guabira": "Guabirá",
    "olimpia asuncion par": "Olimpia",
    "sol de america": "Sol de América",
    "general caballero": "General Caballero",
    "sportivo luqueno": "Sportivo Luqueño",
    "trinidense": "Sportivo Trinidense",
    "sportivo trinidense": "Sportivo Trinidense",
    "atletico grau": "Atlético Grau",
    "atletico grau sadr": "Atlético Grau",
    "adl tarma": "ADL Tarma",
    "asociacion deportiva tarma": "ADL Tarma",
    "sport huancayo": "Sport Huancayo",
    "canton cristal": "Sporting Cristal",
    "deportivo garcilaso": "Deportivo Garcilaso",
    "cs emelec": "Emelec",
    "cs emelec sadr": "Emelec",
    "imbabura": "Imbabura",
    "sd aucas": "Aucas",
    "aucas": "Aucas",
    "mushuc runa": "Mushuc Runa",
    "cumbaya": "Cumbayá",
    "delfin": "Delfín",
    "delfin sc": "Delfín",
    " Tecnico Universitario": "Técnico Universitario",
    "tecnico universitario": "Técnico Universitario",
    "oyotas": "Oyotas",
    "olmedo": "Olmedo",
    "macara": "Macará",
    "club atletico progreso": "Progreso",
    "juventud": "Juventud",
    "juventud de las piedras": "Juventud (URU)",
    "plaza colon": "Plaza Colonia",
    "plaza colonia": "Plaza Colonia",
    "cerro": "Cerro (URU)",
    "ca cerro": "Cerro (URU)",
    "fenix": "Fénix",
    "sud america": "Sud América",
    "rentistas": "Rentistas",
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_name(name: str) -> str:
    """Chave de comparação: minúscula, sem acento, sem pontuação."""
    text = strip_accents(str(name)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def short_name(name: str) -> str:
    """Nome curto amigável para o dashboard (ex.: ``CR Flamengo`` → ``Flamengo``)."""
    key = normalize_name(name)
    if key in _CANONICAL:
        return _CANONICAL[key]
    stripped = key
    for prefix in _PREFIXES:
        p = normalize_name(prefix)
        if p and key.startswith(p + " "):  # limite de palavra: "CA " ≠ "CAR"
            stripped = key[len(p) + 1:].strip()
            break
    if stripped in _CANONICAL:
        return _CANONICAL[stripped]
    return name.strip()


# --------------------------------------------------------------------------- #
# Parser dos arquivos openfootball
# --------------------------------------------------------------------------- #
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_STAGE_RE = re.compile(r"^▪\s*(?P<stage>.+?)\s*(?:,\s*(?P<round>.+))?$")
_DATE_RE = re.compile(
    r"^(?:[A-Z][a-z]{2})\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?$"
)
_MATCH_RE = re.compile(
    r"^(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)\s*(.*)$"
)


def _parse_tail(tail: str) -> Optional[dict]:
    """
    Interpreta o placar de uma linha de partida do openfootball.

    Variantes reais encontradas nos arquivos (2012–2026)::

        2-0 (0-0)                          → placar + intervalo
        1-2                                → só o placar
        4-3 pen. (2-1, 0-1)                → pênaltis + (placar, intervalo)
        4-3 pen. (2-1)                     → pênaltis + placar
        2-1 a.e.t. (2-1, 1-1)              → prorrogação + (placar, intervalo)
        4-5 pen. 2-1 a.e.t. (2-1, 0-1)     → pênaltis + placar repetido + a.e.t.
        10-9 pen. 0-0 a.e.t. (0-0)         → pênaltis (2 dígitos)

    Retorna dict com ``gols``, ``intervalo``, ``penaltis`` e ``prorrogacao``,
    ou ``None`` se o formato não for reconhecido.
    """
    s = tail.strip()
    if not s:
        return {"gols": None, "intervalo": None, "penaltis": None, "prorrogacao": False}

    pens = ft = ht = None
    prorrogacao = False

    m = re.match(r"^(\d+)-(\d+)\s+pen\.\s+", s)
    if m:
        pens = (int(m.group(1)), int(m.group(2)))
        s = s[m.end():]

    # placar repetido antes de "a.e.t." (redundante, confirma o de dentro)
    m = re.match(r"^(\d+)-(\d+)\s+a\.e\.t\.\s+", s)
    if m:
        ft = (int(m.group(1)), int(m.group(2)))
        prorrogacao = True
        s = s[m.end():]

    m = re.match(r"^a\.e\.t\.\s+", s)
    if m:
        prorrogacao = True
        s = s[m.end():]

    m = re.match(r"^(\d+)-(\d+)(?=\s*\(|$)", s)
    if m:
        ft = (int(m.group(1)), int(m.group(2)))
        s = s[m.end():].strip()

    if s.startswith("(") and s.endswith(")"):
        inner = re.findall(r"(\d+)-(\d+)", s)
        if not inner:
            return None
        pairs = [(int(a), int(b)) for a, b in inner]
        if ft is None:
            ft = pairs[0]
        if len(pairs) > 1:
            ht = pairs[1]
        elif ft is not None and len(pairs) == 1 and pens is None and not prorrogacao:
            ht = pairs[0]  # formato clássico "2-0 (1-0)": par único = intervalo
        s = ""
    elif s:
        return None

    return {
        "gols": ft,
        "intervalo": ht,
        "penaltis": pens,
        "prorrogacao": prorrogacao,
    }


def _normalize_stage(stage: str, round_name: str) -> Tuple[str, str]:
    """
    Padroniza os nomes de fase usados pelo openfootball ao longo dos anos.

    2012–2015 usam nomes soltos/alemães (``Gruppe 1``, ``Round 1``,
    ``Quarterfinals``…); 2017+ usam ``Qualifying/Group/Finals, <rodada>``.
    Saída canônica: fase ∈ {Qualifying, Group, Playoffs, Finals}.
    """
    s = stage.strip()
    if s.startswith("Gruppe") or s.startswith("Group"):
        return "Group", round_name
    if s.startswith("Qualifying") or re.fullmatch(r"Round [1-4]", s) or s.startswith("Preliminar"):
        return "Qualifying", round_name or s
    if s.startswith("Playoffs") or s == "Round of 16":
        return "Playoffs", round_name or s
    if s in ("Quarterfinals", "Semifinals", "Final") or s.startswith("Finals"):
        return "Finals", round_name or s
    return s, round_name


def parse_openfootball_file(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Converte um arquivo ``YYYY_copal.txt`` em DataFrame de partidas."""
    warnings: List[str] = []
    rows: List[dict] = []

    temporada = int(path.name[:4])
    stage, round_name = "", ""
    year: Optional[int] = None
    current_date: Optional[str] = None

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\r").strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue

        m = _STAGE_RE.match(line)
        if m:
            stage, round_name = _normalize_stage(m.group("stage"), (m.group("round") or "").strip())
            continue

        m = _DATE_RE.match(line)
        if m:
            mon = _MONTHS.get(m.group(1).lower())
            if mon is None:
                warnings.append(f"{path.name}:{lineno}: mês inválido '{m.group(1)}'")
                continue
            if m.group(3):
                year = int(m.group(3))
            elif year is None:
                warnings.append(f"{path.name}:{lineno}: data sem ano e sem ano anterior")
                continue
            current_date = f"{year:04d}-{mon:02d}-{int(m.group(2)):02d}"
            continue

        if "N.N." in line:
            continue  # confronto ainda não definido (placeholder da fonte)

        m = _MATCH_RE.match(line)
        if not m:
            warnings.append(f"{path.name}:{lineno}: linha não reconhecida: {line[:60]}")
            continue

        home, home_country, away, away_country, tail = (
            m.group(2), m.group(3), m.group(4), m.group(5), m.group(6).strip()
        )

        # partidas adiadas/canceladas sem placar não entram no dataset
        if tail in ("[postponed]", "[cancelled]"):
            continue

        row = {
            "temporada": temporada,
            "fase": stage,
            "rodada": round_name,
            "data": current_date,
            "mandante": home.strip(),
            "pais_mandante": home_country,
            "visitante": away.strip(),
            "pais_visitante": away_country,
            "gols_mandante": None,
            "gols_visitante": None,
            "gols_intervalo_mandante": None,
            "gols_intervalo_visitante": None,
            "penaltis_mandante": None,
            "penaltis_visitante": None,
            "vencedor_penaltis": "",
            "prorrogacao": False,
            "observacao": "",
            "fonte": "openfootball",
        }

        # anotações como "[awarded]" (placar atribuído por W.O.)
        tail_note = ""
        note_m = re.search(r"\[[a-z]+\]$", tail)
        if note_m:
            tail_note = note_m.group(0)
            tail = tail[: note_m.start()].strip()
            row["observacao"] = tail_note.strip("[]")

        if tail:
            parsed = _parse_tail(tail)
            if parsed is None or parsed["gols"] is None:
                warnings.append(f"{path.name}:{lineno}: placar não parseado: '{tail}'")
                continue
            gols, ht, pens = parsed["gols"], parsed["intervalo"], parsed["penaltis"]
            row["gols_mandante"], row["gols_visitante"] = gols
            if ht is not None:
                row["gols_intervalo_mandante"], row["gols_intervalo_visitante"] = ht
            if pens is not None:
                row["penaltis_mandante"], row["penaltis_visitante"] = pens
            row["prorrogacao"] = parsed["prorrogacao"]
            if tail_note:
                row["observacao"] = tail_note.strip("[]")

        if row["penaltis_mandante"] is not None:
            if row["penaltis_mandante"] > row["penaltis_visitante"]:
                row["vencedor_penaltis"] = row["mandante"]
            elif row["penaltis_visitante"] > row["penaltis_mandante"]:
                row["vencedor_penaltis"] = row["visitante"]

        if row["gols_mandante"] is not None:
            gm, gv = row["gols_mandante"], row["gols_visitante"]
            row["resultado"] = (
                "mandante" if gm > gv else "visitante" if gv > gm else "empate"
            )
        else:
            row["resultado"] = ""

        if row["data"] is None:
            warnings.append(f"{path.name}:{lineno}: partida sem data ({home} x {away})")
        rows.append(row)

    return pd.DataFrame(rows), warnings


# --------------------------------------------------------------------------- #
# Enriquecimentos: grupos, pernas e nomes curtos
# --------------------------------------------------------------------------- #
def _assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Infere o grupo (A–H) de cada partida da fase de grupos por conectividade.

    Times que se enfrentam na fase de grupos pertencem ao mesmo grupo; a
    componente conexa do grafo de confrontos é exatamente o grupo.
    """
    out = df.copy()
    out["grupo"] = ""
    mask = out["fase"].eq("Group") & out["gols_mandante"].notna()
    if not mask.any():
        return out

    sub = out[mask].copy()
    sub["_team_home"] = sub["temporada"].astype(str) + "|" + sub["mandante"].map(normalize_name)
    sub["_team_away"] = sub["temporada"].astype(str) + "|" + sub["visitante"].map(normalize_name)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, r in sub.iterrows():
        union(r["_team_home"], r["_team_away"])

    components: Dict[str, List[str]] = {}
    for team in parent:
        components.setdefault(find(team), []).append(team)

    # Ordena grupos pela primeira aparição no calendário (dentro da temporada)
    first_seen: Dict[str, str] = {}
    for _, r in sub.sort_values("data").iterrows():
        for t in (r["_team_home"], r["_team_away"]):
            first_seen.setdefault(t, str(r["data"]))
    ordered_roots = sorted(
        components, key=lambda root: min(first_seen.get(t, "9999") for t in components[root])
    )

    # Letras A, B, C… reiniciam em cada temporada
    root_to_letter: Dict[str, str] = {}
    season_counter: Dict[int, int] = {}
    for root in ordered_roots:
        temporada = int(root.split("|", 1)[0])
        i = season_counter.get(temporada, 0)
        root_to_letter[root] = chr(ord("A") + i)
        season_counter[temporada] = i + 1
    team_to_group = {
        t: root_to_letter[root] for root, members in components.items() for t in members
    }
    out.loc[mask, "grupo"] = sub.apply(
        lambda r: team_to_group[r["_team_home"]], axis=1
    )
    return out


def _assign_legs(df: pd.DataFrame) -> pd.DataFrame:
    """Marca ida (1) e volta (2) nos confrontos de mata-mata/classificatórias."""
    out = df.copy()
    out["perna"] = 1
    knockout = out["fase"].isin(["Qualifying", "Playoffs", "Finals", "Round of 16"])
    for (_, _, _), grp in out[knockout].groupby(["temporada", "fase", "rodada"]):
        seen: Dict[frozenset, int] = {}
        for idx in grp.sort_values("data").index:
            key = frozenset(
                (normalize_name(out.at[idx, "mandante"]), normalize_name(out.at[idx, "visitante"]))
            )
            seen[key] = seen.get(key, 0) + 1
            out.at[idx, "perna"] = seen[key]
    return out


def load_partidas(path: Optional[Path] = None) -> pd.DataFrame:
    """Carrega o dataset consolidado de partidas reais (todas as edições)."""
    path = path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode `python src/real_data.py build` primeiro."
        )
    return pd.read_csv(path)


def standings(partidas: pd.DataFrame, temporada: int) -> pd.DataFrame:
    """Tabela da fase de grupos **derivada das partidas** (fonte única de verdade).

    Garante por construção que ``sum(GP) == sum(GC)`` — impossível de violar.
    """
    g = partidas[
        (partidas["temporada"] == temporada)
        & (partidas["fase"] == "Group")
        & partidas["gols_mandante"].notna()
    ].copy()
    if g.empty:
        return pd.DataFrame()

    stats = []
    teams = set(g["mandante"]) | set(g["visitante"])
    for team in teams:
        casa = g[g["mandante"] == team]
        fora = g[g["visitante"] == team]
        gp = int(casa["gols_mandante"].sum() + fora["gols_visitante"].sum())
        gc = int(casa["gols_visitante"].sum() + fora["gols_mandante"].sum())
        v = int((casa["gols_mandante"] > casa["gols_visitante"]).sum()
                + (fora["gols_visitante"] > fora["gols_mandante"]).sum())
        e = int((casa["gols_mandante"] == casa["gols_visitante"]).sum()
                + (fora["gols_visitante"] == fora["gols_mandante"]).sum())
        jogos = int(len(casa) + len(fora))
        d = jogos - v - e
        stats.append({
            "Time": short_name(team),
            "Grupo": g[g["mandante"] == team]["grupo"].iloc[0] if len(g[g["mandante"] == team]) else g[g["visitante"] == team]["grupo"].iloc[0],
            "Pais": g[g["mandante"] == team]["pais_mandante"].iloc[0],
            "Pts": 3 * v + e,
            "J": jogos,
            "V": v,
            "E": e,
            "D": d,
            "GP": gp,
            "GC": gc,
            "SG": gp - gc,
        })
    return pd.DataFrame(stats).sort_values(
        ["Pts", "V", "SG", "GP"], ascending=False, ignore_index=True
    )


def knockout_results(partidas: pd.DataFrame, temporada: int, fase: str) -> pd.DataFrame:
    """Resultados agregados de um mata-mata (ida e volta + pênaltis).

    Confrontos com pernas ainda não disputadas aparecem com ``Vencedor`` vazio
    (decisão em andamento) — nunca declara vencedor antes da hora.
    """
    k = partidas[
        (partidas["temporada"] == temporada)
        & (partidas["fase"] == fase)
    ].copy()
    ties: Dict[frozenset, dict] = {}
    for _, r in k.sort_values("data").iterrows():
        key = _pair(r["mandante"], r["visitante"])
        t = ties.setdefault(
            key, {"ida": None, "volta": None, "jogos": 0, "faltam": 0,
                  "time1": r["mandante"], "time2": r["visitante"],
                  "pais1": r["pais_mandante"], "pais2": r["pais_visitante"],
                  "penaltis": "", "agg1": 0, "agg2": 0}
        )
        t["jogos"] += 1
        if pd.isna(r["gols_mandante"]):
            t["faltam"] += 1
            label = "a disputar"
            if t["ida"] is None:
                t["ida"] = label
            else:
                t["volta"] = label
            continue
        played = f"{int(r['gols_mandante'])}x{int(r['gols_visitante'])} ({short_name(r['mandante'])} em casa)"
        if r["mandante"] == t["time1"]:
            t["agg1"] += int(r["gols_mandante"])
            t["agg2"] += int(r["gols_visitante"])
        else:
            t["agg1"] += int(r["gols_visitante"])
            t["agg2"] += int(r["gols_mandante"])
        if t["ida"] is None:
            t["ida"] = played
        else:
            t["volta"] = played
        if r["vencedor_penaltis"]:
            t["penaltis"] = r["vencedor_penaltis"]

    rows = []
    for t in ties.values():
        penaltis = t["penaltis"] if isinstance(t["penaltis"], str) else ""
        if t["faltam"] > 0:
            winner = ""  # confronto em andamento
        elif t["agg1"] > t["agg2"]:
            winner = t["time1"]
        elif t["agg2"] > t["agg1"]:
            winner = t["time2"]
        else:
            winner = penaltis
        rows.append({
            "Time1": short_name(t["time1"]), "Pais1": t["pais1"],
            "Time2": short_name(t["time2"]), "Pais2": t["pais2"],
            "Ida": t["ida"], "Volta": t["volta"] if t["volta"] else "—",
            "Agregado": f"{t['agg1']}-{t['agg2']}",
            "Vencedor": short_name(winner) if winner else "(em andamento)",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Suplementos: placares reais mais novos que o openfootball (FBref/ESPN)
# --------------------------------------------------------------------------- #
SUPPLEMENT_PATH = HIST_DIR / "supplementos_2026.csv"


def _pair(team_a: str, team_b: str) -> frozenset:
    return frozenset((normalize_name(team_a), normalize_name(team_b)))


def apply_supplements(partidas: pd.DataFrame, path: Optional[Path] = None) -> pd.DataFrame:
    """
    Aplica correções/placares reais documentados em CSV de suplemento.

    Ações:

    * ``score`` — preenche o placar de uma partida já listada sem placar
      (casa pelos nomes dos times, fase e data ±1 dia);
    * ``substituir`` — remove o confronto (par de times, fase) da fonte
      original e insere as linhas do suplemento (ex.: partida remarcada).

    O CSV é versionado no repositório como evidência auditável.
    """
    path = path or SUPPLEMENT_PATH
    if not path.exists():
        return partidas

    sup = pd.read_csv(path)
    out = partidas.copy()
    out["_data_dt"] = pd.to_datetime(out["data"], errors="coerce")

    def _base_row(row: pd.Series) -> dict:
        gm = row.get("gols_mandante")
        gv = row.get("gols_visitante")
        base = {
            "temporada": 2026,
            "fase": row["fase"],
            "rodada": row["rodada"],
            "data": row["data"],
            "mandante": row["mandante"],
            "pais_mandante": row["pais_mandante"],
            "visitante": row["visitante"],
            "pais_visitante": row["pais_visitante"],
            "gols_mandante": None if pd.isna(gm) else int(gm),
            "gols_visitante": None if pd.isna(gv) else int(gv),
            "resultado": "",
            "gols_intervalo_mandante": None,
            "gols_intervalo_visitante": None,
            "penaltis_mandante": None if pd.isna(row.get("penaltis_mandante")) else int(row["penaltis_mandante"]),
            "penaltis_visitante": None if pd.isna(row.get("penaltis_visitante")) else int(row["penaltis_visitante"]),
            "vencedor_penaltis": "",
            "prorrogacao": bool(row.get("prorrogacao", False)),
            "observacao": "",
            "fonte": row["fonte"],
        }
        pm, pv = base["penaltis_mandante"], base["penaltis_visitante"]
        if pm is not None and pv is not None:
            base["vencedor_penaltis"] = base["mandante"] if pm > pv else base["visitante"]
        if base["gols_mandante"] is not None:
            gm2, gv2 = base["gols_mandante"], base["gols_visitante"]
            base["resultado"] = "mandante" if gm2 > gv2 else "visitante" if gv2 > gm2 else "empate"
        return base

    def _pair_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
        return df.apply(
            lambda r: _pair(r["mandante"], r["visitante"]) == _pair(row["mandante"], row["visitante"]),
            axis=1,
        ) & df["fase"].eq(row["fase"]) & df["temporada"].eq(2026)

    # 1) Substituições: apaga TODOS os confrontos afetados de uma vez, depois insere
    sub_rows = sup[sup["acao"] == "substituir"]
    if not sub_rows.empty:
        drop_mask = pd.Series(False, index=out.index)
        for _, row in sub_rows.iterrows():
            drop_mask |= _pair_mask(out, row)
        out = out[~drop_mask].copy()
        out = pd.concat(
            [out, pd.DataFrame([_base_row(r) for _, r in sub_rows.iterrows()])],
            ignore_index=True,
        )

    # 2) Placares: preenche partidas existentes sem placar
    for _, row in sup[sup["acao"] == "score"].iterrows():
        base = _base_row(row)
        sup_dt = pd.to_datetime(row["data"])
        pair_mask = _pair_mask(out, row)
        near_date = (out["_data_dt"] - sup_dt).abs() <= pd.Timedelta(days=1)
        target = out[pair_mask & near_date & out["gols_mandante"].isna()]
        if target.empty:
            target = out[pair_mask & out["gols_mandante"].isna()]
        if target.empty:
            print(f"⚠️  suplemento sem correspondência: {row['mandante']} x {row['visitante']} ({row['data']})")
            continue
        idx = target.index[0]
        for col in ("gols_mandante", "gols_visitante", "resultado",
                    "penaltis_mandante", "penaltis_visitante", "vencedor_penaltis",
                    "prorrogacao", "fonte"):
            out.at[idx, col] = base[col]

    return out.drop(columns=["_data_dt"], errors="ignore")



# --------------------------------------------------------------------------- #
# Integração com a ESPN (placares recentes + confrontos futuros)
# --------------------------------------------------------------------------- #
def load_espn_events(json_path: Path) -> List[dict]:
    """Extrai eventos de um JSON de scoreboard salvo em data/historical/espn/."""
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    events = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        home = away = None
        for c in comp["competitors"]:
            entry = {
                "team": c["team"]["displayName"],
                "score": int(c.get("score", 0)) if c.get("score") not in (None, "") else None,
            }
            if c.get("homeAway") == "home":
                home = entry
            else:
                away = entry
        if not home or not away:
            continue
        events.append({
            "event_id": ev["id"],
            "data": ev["date"][:10],
            "fase": (ev.get("season") or {}).get("slug", ""),
            "concluida": comp["status"]["type"]["completed"],
            "mandante": home["team"],
            "visitante": away["team"],
            "gols_mandante": home["score"],
            "gols_visitante": away["score"],
        })
    return events


def merge_espn_into_partidas(partidas: pd.DataFrame, espn_dir: Optional[Path] = None) -> pd.DataFrame:
    """Complementa partidas sem placar com os resultados da ESPN."""
    espn_dir = espn_dir or ESPN_DIR
    if not espn_dir.exists():
        return partidas

    out = partidas.copy()
    for json_path in sorted(espn_dir.glob("scoreboard_*.json")):
        for ev in load_espn_events(json_path):
            if not ev["concluida"] or ev["gols_mandante"] is None:
                continue
            key_home, key_away = normalize_name(ev["mandante"]), normalize_name(ev["visitante"])
            mask = (
                out["gols_mandante"].isna()
                & out["mandante"].map(normalize_name).eq(key_home)
                & out["visitante"].map(normalize_name).eq(key_away)
                & out["data"].eq(ev["data"])
            )
            if mask.any():
                idx = out.index[mask][0]
                out.at[idx, "gols_mandante"] = ev["gols_mandante"]
                out.at[idx, "gols_visitante"] = ev["gols_visitante"]
                gm, gv = ev["gols_mandante"], ev["gols_visitante"]
                out.at[idx, "resultado"] = "mandante" if gm > gv else "visitante" if gv > gm else "empate"
                out.at[idx, "fonte"] = "espn"
    return out


def load_espn_fixtures(espn_dir: Optional[Path] = None) -> pd.DataFrame:
    """Confrontos futuros (sem placar) salvos da ESPN — ex.: quartas de final."""
    espn_dir = espn_dir or ESPN_DIR
    rows = []
    if not espn_dir.exists():
        return pd.DataFrame(rows)
    for json_path in sorted(espn_dir.glob("scoreboard_*.json")):
        for ev in load_espn_events(json_path):
            if ev["concluida"]:
                continue
            rows.append({
                "data": ev["data"], "fase": ev["fase"],
                "mandante": ev["mandante"], "visitante": ev["visitante"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Dataset consolidado
# --------------------------------------------------------------------------- #
def build_dataset(output: Optional[Path] = None) -> pd.DataFrame:
    """Lê todos os arquivos openfootball + complementos ESPN e salva o CSV final."""
    files = sorted(OPENFOOTBALL_DIR.glob("*_copal.txt"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo openfootball em {OPENFOOTBALL_DIR}. "
            "Dados reais são obrigatórios — nada é simulado."
        )

    all_warnings: List[str] = []
    frames = []
    for f in files:
        df, warns = parse_openfootball_file(f)
        frames.append(df)
        all_warnings.extend(warns)

    partidas = pd.concat(frames, ignore_index=True)
    partidas = apply_supplements(partidas)
    partidas = merge_espn_into_partidas(partidas)
    partidas = _assign_legs(_assign_groups(partidas))
    partidas["nome_curto_mandante"] = partidas["mandante"].map(short_name)
    partidas["nome_curto_visitante"] = partidas["visitante"].map(short_name)

    cols = [
        "temporada", "fase", "rodada", "grupo", "perna", "data",
        "mandante", "pais_mandante", "nome_curto_mandante",
        "visitante", "pais_visitante", "nome_curto_visitante",
        "gols_mandante", "gols_visitante", "resultado",
        "gols_intervalo_mandante", "gols_intervalo_visitante",
        "penaltis_mandante", "penaltis_visitante", "vencedor_penaltis",
        "prorrogacao", "observacao",
        "fonte",
    ]
    partidas = partidas[cols]

    output = output or DATASET_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    partidas.to_csv(output, index=False)

    played = partidas[partidas["gols_mandante"].notna()]
    print(f"✅ {len(partidas)} partidas ({len(played)} com placar) → {output}")
    print(f"   Temporadas: {partidas['temporada'].min()}–{partidas['temporada'].max()}")
    if all_warnings:
        print(f"⚠️  {len(all_warnings)} avisos de parsing (ver stdout):")
        for w in all_warnings[:10]:
            print(f"   - {w}")
    return partidas


def validate(partidas: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Validações de integridade — retorna (erros, avisos).

    A invariante fundamental de futebol vale por *conjunto de partidas*:
    na tabela derivada, ``sum(GP) == sum(GC)`` (todo gol marcado por um time
    é sofrido pelo adversário). Somar gols de mandantes vs. visitantes por
    fase NÃO precisa bater (vantagem de campo existe de verdade).
    """
    errors: List[str] = []
    qa_warnings: List[str] = []
    played = partidas[partidas["gols_mandante"].notna()]

    # 1. Placares plausíveis
    if (played["gols_mandante"] < 0).any() or (played["gols_visitante"] < 0).any():
        errors.append("existem placares negativos")

    # 2. Partidas duplicadas
    dup = played.duplicated(subset=["temporada", "fase", "data", "mandante", "visitante"])
    if dup.any():
        d = played[dup]
        errors.append(
            f"{len(d)} partidas duplicadas, ex.: "
            + "; ".join(
                f"{r.temporada} {r.fase} {r.mandante}x{r.visitante} {r.data}"
                for r in d.head(3).itertuples()
            )
        )

    # 3. Tabelas de grupos consistentes (a invariante de verdade)
    for temporada in sorted(played["temporada"].unique()):
        tab = standings(partidas, int(temporada))
        if tab.empty:
            continue
        if int(tab["GP"].sum()) != int(tab["GC"].sum()):
            errors.append(
                f"temporada {temporada}: sum(GP)={tab['GP'].sum()} != "
                f"sum(GC)={tab['GC'].sum()} na tabela de grupos"
            )
        if not ((tab["V"] + tab["E"] + tab["D"]) == tab["J"]).all():
            errors.append(f"temporada {temporada}: V+E+D != J")
        if int(tab["J"].sum()) % 2 != 0:
            errors.append(f"temporada {temporada}: soma de J ímpar na fase de grupos")

    # 4. Mata-mata: nenhum confronto com mais de 2 pernas; vencedor definido
    for (temporada, fase, rodada), grp in played[
        played["fase"].isin(["Qualifying", "Playoffs", "Finals"])
    ].groupby(["temporada", "fase", "rodada"]):
        if (grp["perna"] > 2).any():
            errors.append(f"{temporada} {fase}/{rodada}: confronto com mais de 2 pernas")
        # confrontos empatados no agregado precisam de vencedor em pênaltis
        for _, tie in grp.groupby(
            grp.apply(lambda r: frozenset((normalize_name(r["mandante"]), normalize_name(r["visitante"]))), axis=1)
        ):
            if len(tie) == 1:
                continue  # final em jogo único ou ida ainda sem volta
            t1, t2 = tie.iloc[0]["mandante"], tie.iloc[0]["visitante"]
            agg1 = int(tie.iloc[0]["gols_mandante"]) + int(tie.iloc[1]["gols_visitante"])
            agg2 = int(tie.iloc[0]["gols_visitante"]) + int(tie.iloc[1]["gols_mandante"])
            if agg1 == agg2 and not tie["vencedor_penaltis"].any():
                qa_warnings.append(
                    f"{temporada} {fase}/{rodada}: {t1} x {t2} — agregado empatado e "
                    f"disputa de pênaltis não registrada na fonte (vencedor indefinido)"
                )

    return errors, qa_warnings


# --------------------------------------------------------------------------- #
# Tabelas consumidas pelo dashboard (app.py / preprocessing.load_data)
# --------------------------------------------------------------------------- #
RAW_OUT_DIR = ROOT_DIR / "data" / "raw"
QUARTAS_PATH = HIST_DIR / "confrontos_quartas_2026.csv"


def build_app_tables(temporada: int = 2026) -> Dict[str, pd.DataFrame]:
    """
    Gera os três CSVs que o dashboard consome, 100% a partir de dados reais:

    * ``data/raw/grupos_libertadores_2026.csv`` — tabela de grupos derivada
      das partidas (invariante ``sum(GP) == sum(GC)`` garantida);
    * ``data/raw/oitavas_resultados.csv`` — resultados agregados das oitavas;
    * ``data/raw/confrontos_quartas.csv`` — confrontos reais das quartas.
    """
    partidas = load_partidas()

    grupos = standings(partidas, temporada)
    if grupos.empty:
        raise ValueError(f"Sem partidas da fase de grupos para {temporada} — dados reais obrigatórios.")

    oitavas_raw = knockout_results(partidas, temporada, "Playoffs")
    oitavas = pd.DataFrame({
        "Fase": "Oitavas",
        "Time1": oitavas_raw["Time1"],
        "Time2": oitavas_raw["Time2"],
        "Ida": oitavas_raw["Ida"],
        "Volta": oitavas_raw["Volta"],
        "Agregado": oitavas_raw["Agregado"],
        "Classificado": oitavas_raw["Vencedor"],
    })

    if not QUARTAS_PATH.exists():
        raise FileNotFoundError(f"{QUARTAS_PATH} não encontrado (confrontos reais das quartas).")
    quartas = pd.read_csv(QUARTAS_PATH)

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    grupos.to_csv(RAW_OUT_DIR / "grupos_libertadores_2026.csv", index=False)
    oitavas.to_csv(RAW_OUT_DIR / "oitavas_resultados.csv", index=False)
    quartas.to_csv(RAW_OUT_DIR / "confrontos_quartas.csv", index=False)
    return {"grupos": grupos, "oitavas": oitavas, "quartas": quartas}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dados reais da Libertadores")
    parser.add_argument("command", choices=["build", "validate", "resumo", "tabelas"])
    args = parser.parse_args(argv)

    if args.command == "tabelas":
        tabelas = build_app_tables()
        print("✅ Tabelas do dashboard regeneradas a partir de dados reais:")
        print(f"   • grupos: {len(tabelas['grupos'])} times")
        print(f"   • oitavas: {len(tabelas['oitavas'])} confrontos")
        print(f"   • quartas: {len(tabelas['quartas'])} confrontos")
        return 0

    if args.command == "build":
        partidas = build_dataset()
        errs, warns = validate(partidas)
        for w in warns:
            print(f"⚠️  {w}")
        if errs:
            print("\n❌ VALIDAÇÃO FALHOU:")
            for e in errs:
                print(f"   - {e}")
            return 1
        print("✅ Validação de integridade: OK (gols, tabelas e jogos consistentes)")
        return 0

    partidas = load_partidas()
    if args.command == "validate":
        errs, warns = validate(partidas)
        for w in warns:
            print(f"⚠️  {w}")
        for e in errs:
            print(f"❌ {e}")
        print("✅ sem erros" if not errs else f"{len(errs)} erros")
        return 1 if errs else 0

    if args.command == "resumo":
        resumo = (
            partidas.groupby("temporada")
            .agg(partidas=("data", "count"), com_placar=("gols_mandante", "count"))
            .reset_index()
        )
        print(resumo.to_string(index=False))
        return 0


if __name__ == "__main__":
    sys.exit(main())
