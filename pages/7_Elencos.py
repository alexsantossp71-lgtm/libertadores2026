"""
👕 Análise de Elencos — Libertadores 2026

Poder de fogo, pressão defensiva, química (rotação) e risco de suspensão
a partir da FBref + forma recente do openfootball. Os mesmos índices
ajustam o Poisson das previsões.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.elenco_analysis import (
    analisar_elencos,
    features_confronto_elenco,
    forma_recente,
    persistir_analise,
    tabela_influencia,
)

st.set_page_config(page_title="👕 Análise de Elencos", page_icon="👕", layout="wide")

st.title("👕 Análise de Elencos")
st.caption(
    "Índices reais da FBref (finalizações, desarmes, interceptações, cartões, "
    "nº de jogadores) + forma dos últimos 5 jogos. Química = 11 / jogadores usados "
    "— a Libertadores na FBref não publica o XI titular por partida."
)

try:
    elencos = analisar_elencos()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

forma = forma_recente()
influencia = tabela_influencia()
persistir_analise(elencos, forma=forma, influencia=influencia)

# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
top_off = elencos.sort_values("indice_forca_ofensiva", ascending=False).iloc[0]
top_def = elencos.sort_values("indice_pressao_defensiva", ascending=False).iloc[0]
top_chem = elencos.sort_values("quimica_elenco", ascending=False).iloc[0]
top_score = elencos.sort_values("score_elenco", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔥 Poder de fogo", top_off["time"], f"{top_off['indice_forca_ofensiva']:.2f}")
c2.metric("🛡️ Pressão defensiva", top_def["time"], f"{top_def['indice_pressao_defensiva']:.1f}/90")
c3.metric("🤝 Química (menos rotação)", top_chem["time"], f"{top_chem['quimica_elenco']:.2f}")
c4.metric("⭐ Score de elenco", top_score["time"], f"{top_score['score_elenco']:.2f}")

# --------------------------------------------------------------------------- #
# Scatter
# --------------------------------------------------------------------------- #
st.subheader("Poder de fogo × pressão defensiva")
fig = px.scatter(
    elencos,
    x="indice_pressao_defensiva",
    y="indice_forca_ofensiva",
    text="time",
    size="quimica_elenco",
    color="score_elenco",
    color_continuous_scale="Tealgrn",
    labels={
        "indice_pressao_defensiva": "Pressão defensiva (desarmes+interceptações / 90)",
        "indice_forca_ofensiva": "Poder de fogo (gols+assist+SoT+chutes / 90)",
        "score_elenco": "Score",
    },
    title="Bolha = química do elenco (menos jogadores usados → maior)",
)
fig.update_traces(textposition="top center", textfont_size=10)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Tabela
# --------------------------------------------------------------------------- #
st.subheader("Tabela de elencos")
cols = [
    c for c in (
        "time", "jogos", "gols", "assistencias", "finalizacoes_no_gol",
        "desarmes_ganhos", "interceptacoes", "n_jogadores",
        "indice_forca_ofensiva", "indice_pressao_defensiva",
        "quimica_elenco", "risco_suspensao", "forma_aproveitamento",
        "score_elenco",
    ) if c in elencos.columns
]
st.dataframe(
    elencos[cols].sort_values("score_elenco", ascending=False),
    width="stretch",
    hide_index=True,
    column_config={
        "indice_forca_ofensiva": st.column_config.NumberColumn("Poder de fogo", format="%.2f"),
        "indice_pressao_defensiva": st.column_config.NumberColumn("Pressão /90", format="%.1f"),
        "quimica_elenco": st.column_config.ProgressColumn("Química", format="%.2f", min_value=0.0, max_value=1.0),
        "risco_suspensao": st.column_config.NumberColumn("Risco suspensão", format="%.2f"),
        "forma_aproveitamento": st.column_config.ProgressColumn("Forma (últ. 5)", format="%.0f%%", min_value=0.0, max_value=1.0),
        "score_elenco": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0.0, max_value=1.0),
    },
)

st.download_button(
    "⬇️ Baixar análise de elencos (CSV)",
    data=elencos.to_csv(index=False).encode("utf-8"),
    file_name="analise_elencos_libertadores_2026.csv",
    mime="text/csv",
)

# --------------------------------------------------------------------------- #
# Confronto
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Comparar dois elencos (como o modelo enxerga o confronto)")
times = sorted(elencos["time"].dropna().unique())
col_a, col_b = st.columns(2)
default_a = times.index("Flamengo") if "Flamengo" in times else 0
default_b = times.index("Palmeiras") if "Palmeiras" in times else min(1, len(times) - 1)
a = col_a.selectbox("Mandante", times, index=default_a)
b = col_b.selectbox("Visitante", [t for t in times if t != a], index=0)

cmp = features_confronto_elenco(elencos, a, b)
st.dataframe(cmp, width="stretch", hide_index=True)

radar_cols = [
    c for c in (
        "indice_forca_ofensiva", "indice_pressao_defensiva",
        "quimica_elenco", "indice_disciplina",
    ) if c in elencos.columns
]
if radar_cols:
    ra = elencos.loc[elencos["time"] == a, radar_cols].iloc[0]
    rb = elencos.loc[elencos["time"] == b, radar_cols].iloc[0]
    radar = pd.DataFrame({
        "índice": radar_cols + [radar_cols[0]],
        a: list(ra.values) + [ra.values[0]],
        b: list(rb.values) + [rb.values[0]],
    })
    fig_r = px.line_polar(radar.melt(id_vars="índice", var_name="time", value_name="valor"),
                          r="valor", theta="índice", color="time", line_close=True)
    fig_r.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_r, width="stretch")

if not influencia.empty:
    st.divider()
    st.subheader("Influência individual (G+A / G+A do time)")
    st.caption("Aparece quando a raspagem completa de jogadores rodou.")
    st.dataframe(influencia.head(40), width="stretch", hide_index=True)

st.divider()
st.caption(
    "Fontes: FBref comps/14 (snapshot em data/historical/fbref/) e "
    "openfootball (data/historical/partidas_libertadores.csv). "
    "O Poisson das previsões usa estes índices como multiplicadores de ataque/defesa."
)
