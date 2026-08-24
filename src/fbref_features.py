"""
Índices de elenco derivados das estatísticas FBref.

Não joga o dado bruto no modelo: agrega e transforma, no espírito de
feature engineering para placares.

* **Índice de Força Ofensiva** — 0.4·gols/90 + 0.3·assistências/90
  + 0.2·finalizações no gol/90 + 0.1·finalizações/90.
* **Índice de Pressão Defensiva** — (desarmes ganhos + interceptações) / 90.
* **Índice de Disciplina** — (cartões + faltas) / 90 (menor = mais limpo).

Quando há tabela de jogadores, os índices ofensivos/defensivos também
são recalculados pelos 3 principais atacantes e pela linha defensiva
titular — o recorte que realmente entra em campo.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.fbref_scraper import load_elencos, load_jogadores


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _per90(numer: pd.Series, noventas: pd.Series) -> pd.Series:
    denom = pd.to_numeric(noventas, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numer, errors="coerce") / denom


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def indices_elencos(elencos: pd.DataFrame) -> pd.DataFrame:
    """Índices agregados a partir da tabela de times da FBref."""
    df = elencos.copy()
    n90 = df["noventas"] if "noventas" in df.columns else df.get("jogos")

    gols90 = _per90(_col(df, "gols"), n90)
    ast90 = _per90(_col(df, "assistencias"), n90)
    if "finalizacoes_no_gol_90" in df.columns and df["finalizacoes_no_gol_90"].notna().any():
        sot90 = _col(df, "finalizacoes_no_gol_90")
    else:
        sot90 = _per90(_col(df, "finalizacoes_no_gol"), n90)
    if "finalizacoes_90" in df.columns and df["finalizacoes_90"].notna().any():
        sh90 = _col(df, "finalizacoes_90")
    else:
        sh90 = _per90(_col(df, "finalizacoes"), n90)

    df["indice_forca_ofensiva"] = (
        0.4 * gols90.fillna(0)
        + 0.3 * ast90.fillna(0)
        + 0.2 * sot90.fillna(0)
        + 0.1 * sh90.fillna(0)
    )

    tkl = _col(df, "desarmes_ganhos").fillna(0)
    inter = _col(df, "interceptacoes").fillna(0)
    rec = _col(df, "recuperacoes").fillna(0)
    df["indice_pressao_defensiva"] = _per90(tkl + inter + rec, n90).fillna(0)

    cards = _col(df, "cartoes_amarelos").fillna(0) + _col(df, "cartoes_vermelhos").fillna(0)
    faltas = _col(df, "faltas").fillna(0)
    df["indice_disciplina"] = _per90(cards + faltas, n90).fillna(0)

    if "gols_sofrido" in df.columns:
        df["gols_sofridos_90"] = _per90(_col(df, "gols_sofrido"), n90)

    df["indice_forca_ofensiva_norm"] = _minmax(df["indice_forca_ofensiva"])
    df["indice_pressao_defensiva_norm"] = _minmax(df["indice_pressao_defensiva"])
    # disciplina: inverter (time mais limpo → 1)
    df["indice_disciplina_norm"] = 1.0 - _minmax(df["indice_disciplina"])
    return df


def _pos_group(posicao: str) -> str:
    text = str(posicao or "").upper()
    if "GK" in text:
        return "GK"
    if text.startswith("DF") or text == "DF":
        return "DF"
    if "FW" in text:
        return "FW"
    if "MF" in text:
        return "MF"
    return "OT"


def indices_jogadores(
    jogadores: pd.DataFrame,
    n_atacantes: int = 3,
    n_defensores: int = 4,
) -> pd.DataFrame:
    """
    Recorta o elenco pelos atores que realmente pesam no placar.

    * Ataque: top ``n_atacantes`` em gols+assistências (preferência FW/MF).
    * Defesa: top ``n_defensores`` em desarmes+interceptações (preferência DF).
    * Criador: meia com mais assistências do time.
    """
    if jogadores.empty:
        return pd.DataFrame(columns=["time", "indice_ataque_titulares", "indice_defesa_titulares"])

    df = jogadores.copy()
    df["grupo_pos"] = df.get("posicao", "").map(_pos_group) if "posicao" in df.columns else "OT"
    n90 = df["noventas"] if "noventas" in df.columns else pd.Series(1.0, index=df.index)
    df["ga"] = _col(df, "gols").fillna(0) + _col(df, "assistencias").fillna(0)
    df["def_raw"] = _col(df, "desarmes_ganhos").fillna(0) + _col(df, "interceptacoes").fillna(0)
    df["ga90"] = _per90(df["ga"], n90).fillna(0)
    df["def90"] = _per90(df["def_raw"], n90).fillna(0)

    rows = []
    time_col = "time" if "time" in df.columns else "time_canonico"
    for time, grp in df.groupby(time_col):
        atac = grp[grp["grupo_pos"].isin(["FW", "MF"])].nlargest(n_atacantes, "ga")
        if atac.empty:
            atac = grp.nlargest(n_atacantes, "ga")
        defe = grp[grp["grupo_pos"].eq("DF")].nlargest(n_defensores, "def_raw")
        if defe.empty:
            defe = grp.nlargest(n_defensores, "def_raw")
        criadores = grp[grp["grupo_pos"].eq("MF")].nlargest(1, "assistencias") if "assistencias" in grp else grp.iloc[0:0]
        rows.append({
            "time": time,
            "indice_ataque_titulares": float(atac["ga90"].mean()) if len(atac) else 0.0,
            "indice_defesa_titulares": float(defe["def90"].mean()) if len(defe) else 0.0,
            "criador": (criadores.iloc[0]["jogador"] if len(criadores) and "jogador" in criadores else None),
            "criador_assistencias": (
                float(criadores.iloc[0]["assistencias"]) if len(criadores) and "assistencias" in criadores else None
            ),
            "n_jogadores_amostra": int(len(grp)),
        })
    return pd.DataFrame(rows)


def indices_completos(
    elencos: Optional[pd.DataFrame] = None,
    jogadores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Junta índices de elenco (FBref squad) com recorte por jogador, se houver."""
    elencos = load_elencos() if elencos is None else elencos
    if jogadores is None:
        try:
            jogadores = load_jogadores()
        except FileNotFoundError:
            jogadores = pd.DataFrame()

    base = indices_elencos(elencos)
    if jogadores is None or jogadores.empty:
        return base
    recorte = indices_jogadores(jogadores)
    if recorte.empty:
        return base
    return base.merge(recorte, on="time", how="left")


def confronto_indices(
    indices: pd.DataFrame,
    mandante: str,
    visitante: str,
    colunas: Sequence[str] = (
        "indice_forca_ofensiva",
        "indice_pressao_defensiva",
        "indice_disciplina",
    ),
) -> pd.DataFrame:
    """Diferença de índices para um confronto (mandante − visitante)."""
    def _row(nome: str) -> pd.Series:
        hit = indices[indices["time"] == nome]
        if hit.empty and "time_canonico" in indices.columns:
            hit = indices[indices["time_canonico"] == nome]
        if hit.empty:
            raise KeyError(f"Time não encontrado nos índices FBref: {nome}")
        return hit.iloc[0]

    a, b = _row(mandante), _row(visitante)
    out = {
        "mandante": mandante,
        "visitante": visitante,
    }
    for col in colunas:
        if col not in indices.columns:
            continue
        out[f"{col}_mandante"] = a[col]
        out[f"{col}_visitante"] = b[col]
        out[f"diff_{col}"] = a[col] - b[col]
    return pd.DataFrame([out])
