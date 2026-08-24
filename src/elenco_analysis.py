"""
Análise de elencos para o modelo de placares.

Fontes (todas reais, persistidas em CSV)
---------------------------------------
* FBref squad (``data/historical/fbref/elencos_2026.csv`` ou raspagem):
  gols, assistências, finalizações, desarmes ganhos, interceptações,
  faltas, cartões, nº de jogadores usados.
* FBref jogadores (quando a raspagem completa rodou): recorte dos 3
  atacantes e da linha defensiva titular + criador de jogo.
* openfootball (``data/historical/partidas_libertadores.csv``): forma
  recente (últimos N jogos) — química de resultado, não de escalação.

O que NÃO inventamos
--------------------
A FBref da Libertadores não publica passing networks nem XI titular por
jogo. A **química** aqui é o proxy honesto ``11 / n_jogadores``: times
que repetem o elenco usam menos atletas distintos.

Uso::

    python src/elenco_analysis.py          # grava CSVs processados
    python src/elenco_analysis.py confrontos
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fbref_features import (
    confronto_indices,
    indices_completos,
    indices_jogadores,
)
from fbref_scraper import load_elencos, load_jogadores

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
HIST_PARTIDAS = ROOT_DIR / "data" / "historical" / "partidas_libertadores.csv"

ELENCOS_ANALISE_CSV = PROCESSED_DIR / "analise_elencos.csv"
JOGADORES_ANALISE_CSV = PROCESSED_DIR / "analise_jogadores.csv"
FORMA_CSV = PROCESSED_DIR / "forma_recente.csv"
CONFRONTOS_CSV = PROCESSED_DIR / "analise_confrontos_elenco.csv"
INFLUENCIA_CSV = PROCESSED_DIR / "tabela_influencia_jogadores.csv"


@dataclass(frozen=True)
class ElencoWeights:
    """Pesos do ajuste de Poisson. Soma dos |efeitos| fica limitada pelo clip."""

    ofensivo: float = 0.22
    pressao: float = 0.18
    quimica: float = 0.08
    disciplina: float = 0.06
    forma: float = 0.20
    clip_lo: float = 0.70
    clip_hi: float = 1.35


DEFAULT_WEIGHTS = ElencoWeights()

ELENCO_FEATURE_COLS = (
    "indice_forca_ofensiva",
    "indice_pressao_defensiva",
    "indice_disciplina",
    "quimica_elenco",
    "risco_suspensao",
    "indice_ataque_titulares",
    "indice_defesa_titulares",
    "forma_gols_pro_90",
    "forma_gols_contra_90",
    "forma_aproveitamento",
)


# --------------------------------------------------------------------------- #
# Blocos da análise
# --------------------------------------------------------------------------- #
def quimica_e_disciplina(elencos: pd.DataFrame) -> pd.DataFrame:
    """Química (proxy de repetição de elenco) e risco de suspensão."""
    df = elencos.copy()
    n_jog = pd.to_numeric(df.get("n_jogadores"), errors="coerce")
    n90 = pd.to_numeric(df.get("noventas", df.get("jogos")), errors="coerce").replace(0, np.nan)

    df["quimica_elenco"] = (11.0 / n_jog).replace([np.inf, -np.inf], np.nan)
    df["rotatividade"] = (n_jog / 11.0).replace([np.inf, -np.inf], np.nan)

    amarelos = pd.to_numeric(df.get("cartoes_amarelos"), errors="coerce")
    vermelhos = pd.to_numeric(df.get("cartoes_vermelhos"), errors="coerce")
    df["risco_suspensao"] = (amarelos.fillna(0) + 3.0 * vermelhos.fillna(0)) / n90
    return df


def forma_recente(
    partidas: Optional[pd.DataFrame] = None,
    temporada: int = 2026,
    n: int = 5,
) -> pd.DataFrame:
    """Últimos ``n`` jogos com placar de cada time (fonte: openfootball)."""
    if partidas is None:
        if not HIST_PARTIDAS.exists():
            return pd.DataFrame()
        partidas = pd.read_csv(HIST_PARTIDAS)

    df = partidas.copy()
    if "temporada" in df.columns:
        df = df[df["temporada"] == temporada]
    df = df[df["gols_mandante"].notna() & df["gols_visitante"].notna()]
    if df.empty:
        return pd.DataFrame()

    home_col = "nome_curto_mandante" if "nome_curto_mandante" in df.columns else "mandante"
    away_col = "nome_curto_visitante" if "nome_curto_visitante" in df.columns else "visitante"
    df = df.sort_values("data")

    rows = []
    for side, gf, ga, name_col in (
        ("mandante", "gols_mandante", "gols_visitante", home_col),
        ("visitante", "gols_visitante", "gols_mandante", away_col),
    ):
        chunk = df[[name_col, "data", gf, ga]].rename(
            columns={name_col: "time", gf: "gf", ga: "ga"}
        )
        chunk["pts"] = np.where(chunk["gf"] > chunk["ga"], 3, np.where(chunk["gf"] == chunk["ga"], 1, 0))
        rows.append(chunk)
    long = pd.concat(rows, ignore_index=True)

    out = []
    for time, grp in long.groupby("time"):
        last = grp.tail(n)
        jogos = len(last)
        out.append({
            "time": time,
            "forma_jogos": jogos,
            "forma_gols_pro": float(last["gf"].sum()),
            "forma_gols_contra": float(last["ga"].sum()),
            "forma_pontos": int(last["pts"].sum()),
            "forma_gols_pro_90": float(last["gf"].mean()) if jogos else np.nan,
            "forma_gols_contra_90": float(last["ga"].mean()) if jogos else np.nan,
            "forma_aproveitamento": float(last["pts"].sum() / (3 * jogos)) if jogos else np.nan,
        })
    return pd.DataFrame(out)


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - lo) / (hi - lo)


def analisar_elencos(
    elencos: Optional[pd.DataFrame] = None,
    jogadores: Optional[pd.DataFrame] = None,
    partidas: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Tabela única por time: índices FBref + química + forma recente."""
    if elencos is None:
        elencos = load_elencos()
    if jogadores is None:
        jogadores = load_jogadores()

    base = indices_completos(elencos, jogadores)
    base = quimica_e_disciplina(base)

    forma = forma_recente(partidas)
    if not forma.empty:
        base = base.merge(forma, on="time", how="left")

    for col, invert in (
        ("quimica_elenco", False),
        ("risco_suspensao", True),
        ("rotatividade", True),
    ):
        if col not in base.columns:
            continue
        norm = _minmax(base[col])
        base[f"{col}_norm"] = (1.0 - norm) if invert else norm

    # Score composto do elenco (0–1): o que o modelo “enxerga” do plantel
    off_n = base.get("indice_forca_ofensiva_norm", pd.Series(0.5, index=base.index))
    press_n = base.get("indice_pressao_defensiva_norm", pd.Series(0.5, index=base.index))
    chem_n = base.get("quimica_elenco_norm", pd.Series(0.5, index=base.index))
    disc_n = base.get("indice_disciplina_norm", pd.Series(0.5, index=base.index))
    base["score_elenco"] = (
        0.40 * off_n.fillna(0.5)
        + 0.30 * press_n.fillna(0.5)
        + 0.15 * chem_n.fillna(0.5)
        + 0.15 * disc_n.fillna(0.5)
    )
    return base


def analisar_jogadores(jogadores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Papéis no elenco: poder de fogo, linha defensiva, criador."""
    if jogadores is None:
        jogadores = load_jogadores()
    if jogadores is None or jogadores.empty:
        return pd.DataFrame()
    return indices_jogadores(jogadores)


def tabela_influencia(jogadores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Participação de cada jogador nos G+A do time (quando a tabela existe)."""
    if jogadores is None:
        jogadores = load_jogadores()
    if jogadores is None or jogadores.empty:
        return pd.DataFrame()
    df = jogadores.copy()
    df["ga"] = (
        pd.to_numeric(df.get("gols"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("assistencias"), errors="coerce").fillna(0)
    )
    time_col = "time" if "time" in df.columns else "time_canonico"
    totais = df.groupby(time_col)["ga"].transform("sum").replace(0, np.nan)
    df["share_ga"] = df["ga"] / totais
    cols = [c for c in (time_col, "jogador", "posicao", "gols", "assistencias", "ga", "share_ga", "noventas") if c in df.columns]
    return df[cols].sort_values([time_col, "ga"], ascending=[True, False]).reset_index(drop=True)


def lookup_elenco(elencos: pd.DataFrame, time: str) -> Optional[pd.Series]:
    hit = elencos[elencos["time"] == time] if "time" in elencos.columns else pd.DataFrame()
    if hit.empty and "time_canonico" in elencos.columns:
        hit = elencos[elencos["time_canonico"] == time]
    if hit.empty:
        return None
    return hit.iloc[0]


def _centered(value: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return 2.0 * float(value) - 1.0


def multiplicadores_time(
    row: pd.Series,
    weights: ElencoWeights = DEFAULT_WEIGHTS,
) -> Tuple[float, float]:
    """
    Multiplicadores (ataque, defesa) centrados em 1.0.

    Defesa no Poisson é taxa de gols sofridos: pressão alta *reduz* o valor.
    """
    off = _centered(row.get("indice_forca_ofensiva_norm"))
    press = _centered(row.get("indice_pressao_defensiva_norm"))
    chem = _centered(row.get("quimica_elenco_norm"))
    disc = _centered(row.get("indice_disciplina_norm"))

    att = 1.0 + weights.ofensivo * off + weights.quimica * chem
    # disciplina baixa (sujo) → sofre um pouco mais; pressão alta → sofre menos
    defe = 1.0 - weights.pressao * press + weights.disciplina * (-disc)
    att = float(np.clip(att, weights.clip_lo, weights.clip_hi))
    defe = float(np.clip(defe, weights.clip_lo, weights.clip_hi))
    return att, defe


def aplicar_elenco_ao_poisson(
    model,
    elencos: pd.DataFrame,
    forma: Optional[pd.DataFrame] = None,
    weights: ElencoWeights = DEFAULT_WEIGHTS,
) -> Dict[str, Tuple[float, float]]:
    """Ajusta ``model.attack`` / ``model.defense`` com elenco + forma recente."""
    if not getattr(model, "is_fitted", False):
        raise RuntimeError("Poisson precisa estar ajustado antes do elenco.")

    if not hasattr(model, "_attack_base"):
        model._attack_base = dict(model.attack)
        model._defense_base = dict(model.defense)
    else:
        model.attack = dict(model._attack_base)
        model.defense = dict(model._defense_base)

    forma_map: Dict[str, pd.Series] = {}
    if forma is not None and not forma.empty:
        forma_map = {r["time"]: r for _, r in forma.iterrows()}

    applied: Dict[str, Tuple[float, float]] = {}
    w_forma = weights.forma
    for team in list(model.teams):
        row = lookup_elenco(elencos, team)
        if row is None:
            continue

        # Forma recente mistura a taxa observada nos últimos jogos
        fr = forma_map.get(team)
        if fr is not None and w_forma > 0:
            gf90 = fr.get("forma_gols_pro_90")
            ga90 = fr.get("forma_gols_contra_90")
            if pd.notna(gf90):
                model.attack[team] = (1 - w_forma) * model.attack[team] + w_forma * float(gf90)
            if pd.notna(ga90):
                model.defense[team] = (1 - w_forma) * model.defense[team] + w_forma * float(ga90)

        att_m, def_m = multiplicadores_time(row, weights)
        model.attack[team] = max(0.05, model.attack[team] * att_m)
        model.defense[team] = max(0.05, model.defense[team] * def_m)
        applied[team] = (att_m, def_m)

    model.elenco_applied = True
    model.elenco_multipliers = applied
    return applied


def features_confronto_elenco(
    elencos: pd.DataFrame,
    mandante: str,
    visitante: str,
) -> pd.DataFrame:
    """Diferenças de elenco para um confronto (mandante − visitante)."""
    cols = [c for c in ELENCO_FEATURE_COLS if c in elencos.columns]
    extras = [c for c in ("score_elenco", "indice_forca_ofensiva_norm",
                          "indice_pressao_defensiva_norm", "quimica_elenco_norm")
              if c in elencos.columns]
    return confronto_indices(elencos, mandante, visitante, colunas=tuple(cols + extras))


def prever_confronto_com_elenco(
    model,
    mandante: str,
    visitante: str,
    elencos: pd.DataFrame,
) -> Dict[str, object]:
    """Probabilidades 1X2 com e sem ajuste de elenco (para auditoria)."""
    if not model.is_fitted:
        raise RuntimeError("Modelo de Poisson não ajustado.")

    # Snapshot do estado atual (já ajustado ou não)
    current = model.match_probabilities(mandante, visitante)

    base_att = getattr(model, "_attack_base", None)
    out = {
        "mandante": mandante,
        "visitante": visitante,
        "prob_mandante": current["p_home"],
        "prob_empate": current["p_draw"],
        "prob_visitante": current["p_away"],
        "xg_mandante": current["expected_goals_home"],
        "xg_visitante": current["expected_goals_away"],
        "placar_previsto": f"{current['most_likely_score'][0]}x{current['most_likely_score'][1]}",
        "elenco_aplicado": bool(getattr(model, "elenco_applied", False)),
    }

    a = lookup_elenco(elencos, mandante)
    b = lookup_elenco(elencos, visitante)
    if a is not None and b is not None:
        out["indice_ofensivo_mandante"] = float(a.get("indice_forca_ofensiva") or 0)
        out["indice_ofensivo_visitante"] = float(b.get("indice_forca_ofensiva") or 0)
        out["indice_pressao_mandante"] = float(a.get("indice_pressao_defensiva") or 0)
        out["indice_pressao_visitante"] = float(b.get("indice_pressao_defensiva") or 0)
        out["quimica_mandante"] = float(a.get("quimica_elenco") or 0)
        out["quimica_visitante"] = float(b.get("quimica_elenco") or 0)
        out["risco_suspensao_mandante"] = float(a.get("risco_suspensao") or 0)
        out["risco_suspensao_visitante"] = float(b.get("risco_suspensao") or 0)
        out["score_elenco_mandante"] = float(a.get("score_elenco") or 0)
        out["score_elenco_visitante"] = float(b.get("score_elenco") or 0)
        out["nota_elenco"] = _nota_confronto(a, b)

    if base_att is not None:
        # Recalcula o cenário “só fase de grupos” sem destruir o ajuste
        att, defe = dict(model.attack), dict(model.defense)
        model.attack = dict(model._attack_base)
        model.defense = dict(model._defense_base)
        raw = model.match_probabilities(mandante, visitante)
        model.attack, model.defense = att, defe
        out["prob_mandante_base"] = raw["p_home"]
        out["prob_empate_base"] = raw["p_draw"]
        out["prob_visitante_base"] = raw["p_away"]
        out["xg_mandante_base"] = raw["expected_goals_home"]
        out["xg_visitante_base"] = raw["expected_goals_away"]
        out["delta_xg_mandante"] = out["xg_mandante"] - raw["expected_goals_home"]
        out["delta_xg_visitante"] = out["xg_visitante"] - raw["expected_goals_away"]
    return out


def _nota_confronto(a: pd.Series, b: pd.Series) -> str:
    bits = []
    if float(a.get("indice_forca_ofensiva") or 0) > float(b.get("indice_forca_ofensiva") or 0):
        bits.append(f"{a['time']} chega com mais poder de fogo (chutes+gols+assistências /90)")
    else:
        bits.append(f"{b['time']} tem o ataque FBref mais produtivo")
    if float(a.get("indice_pressao_defensiva") or 0) > float(b.get("indice_pressao_defensiva") or 0):
        bits.append(f"{a['time']} pressiona mais (desarmes+interceptações)")
    if float(a.get("quimica_elenco") or 0) > float(b.get("quimica_elenco") or 0):
        bits.append(f"{a['time']} repete mais o elenco (menos jogadores usados)")
    risco_a = float(a.get("risco_suspensao") or 0)
    risco_b = float(b.get("risco_suspensao") or 0)
    if max(risco_a, risco_b) > 2.0:
        sujo = a["time"] if risco_a > risco_b else b["time"]
        bits.append(f"{sujo} chega mais exposto a suspensão (cartões/90)")
    return "; ".join(bits)


def persistir_analise(
    elencos: pd.DataFrame,
    jogadores: Optional[pd.DataFrame] = None,
    forma: Optional[pd.DataFrame] = None,
    influencia: Optional[pd.DataFrame] = None,
    confrontos: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    elencos.to_csv(ELENCOS_ANALISE_CSV, index=False)
    paths["elencos"] = ELENCOS_ANALISE_CSV
    if jogadores is not None and not jogadores.empty:
        jogadores.to_csv(JOGADORES_ANALISE_CSV, index=False)
        paths["jogadores"] = JOGADORES_ANALISE_CSV
    if forma is not None and not forma.empty:
        forma.to_csv(FORMA_CSV, index=False)
        paths["forma"] = FORMA_CSV
    if influencia is not None and not influencia.empty:
        influencia.to_csv(INFLUENCIA_CSV, index=False)
        paths["influencia"] = INFLUENCIA_CSV
    if confrontos is not None and not confrontos.empty:
        confrontos.to_csv(CONFRONTOS_CSV, index=False)
        paths["confrontos"] = CONFRONTOS_CSV
    return paths


def run(persist: bool = True) -> Dict[str, pd.DataFrame]:
    """Gera a análise completa e grava CSVs."""
    print("=" * 60)
    print("👕 ANÁLISE DE ELENCOS — FBref + forma recente")
    print("=" * 60)
    elencos = analisar_elencos()
    jogadores = analisar_jogadores()
    forma = forma_recente()
    influencia = tabela_influencia()
    print(f"  • {len(elencos)} elencos com índices")
    if not jogadores.empty:
        print(f"  • {len(jogadores)} times com recorte de jogadores")
    if not forma.empty:
        print(f"  • forma recente de {len(forma)} times")
    if persist:
        paths = persistir_analise(elencos, jogadores, forma, influencia)
        for key, path in paths.items():
            print(f"  💾 {key}: {path}")
    return {
        "elencos": elencos,
        "jogadores": jogadores,
        "forma": forma,
        "influencia": influencia,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Análise de elencos Libertadores")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "confrontos"])
    args = parser.parse_args(argv)
    data = run(persist=True)
    if args.command == "confrontos":
        print("Use `python src/predict.py` para confrontos das quartas com elenco.")
    print(data["elencos"][["time", "indice_forca_ofensiva", "indice_pressao_defensiva",
                           "quimica_elenco", "score_elenco"]]
          .sort_values("score_elenco", ascending=False)
          .head(8)
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
