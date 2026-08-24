"""
📊 Odds e Probabilidades de Mercado — Libertadores 2026

Página do dashboard que compara as previsões do modelo (Poisson) com as
probabilidades implícitas das odds 1X2 (Bzzoiro Sports Data):

  * Comparação lado a lado por partida;
  * Divergências modelo × mercado (quem acertou quando discordaram?);
  * Combinação "inteligente" (usa a odd quando o mercado diverge fortemente
    do modelo) e seu impacto em acurácia e Brier Score.

Fonte dos dados: ``data/processed/libertadores_odds.csv`` (gerado pelo
pipeline via Bzzoiro, com fallback para ``data/examples/``).
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import load_model_market
from src.preprocessing import Preprocessor

st.set_page_config(
    page_title="📊 Odds e Probabilidades de Mercado",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Odds e Probabilidades de Mercado")
st.caption(
    "Previsão do modelo (Poisson) vs. probabilidades implícitas das odds 1X2 "
    "— acurácia, Brier Score e divergências."
)

pre = Preprocessor()
cmp = load_model_market()  # modelo + odds + resultado por partida

if cmp.empty:
    st.error("Nenhum dado de odds disponível. Execute o pipeline "
             "(`python src/pipeline.py`) ou verifique `data/examples/`.")
    st.stop()

COLS_MODELO = ("prob_mandante_modelo", "prob_empate_modelo", "prob_visitante_modelo")
COLS_MERCADO = ("prob_mandante_impl", "prob_empate_impl", "prob_visitante_impl")
COLS_COMB = ("prob_mandante_combinada", "prob_empate_combinada", "prob_visitante_combinada")

# --------------------------------------------------------------------------- #
# Combinação modelo + mercado (com seletor de limiar de divergência)
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## ⚙️ Combinação modelo × mercado")
threshold = st.sidebar.slider(
    "Limiar de divergência (mercado)",
    min_value=0.02,
    max_value=0.25,
    value=0.08,
    step=0.01,
    help=(
        "Se |P(modelo) − P(mercado)| ultrapassar este valor em alguma classe, "
        "a combinação usa a probabilidade implícita das odds (o mercado é "
        "tratado como portador de informação extra — lesões, escalações etc.); "
        "caso contrário, usa o modelo."
    ),
)

combinada = pre.combined_probabilities(cmp, threshold=threshold)

ev_modelo = pre.evaluate_probabilities(cmp, COLS_MODELO)
ev_mercado = pre.evaluate_probabilities(cmp, COLS_MERCADO)
ev_combinada = pre.evaluate_probabilities(combinada, COLS_COMB)

# --------------------------------------------------------------------------- #
# Métricas de performance
# --------------------------------------------------------------------------- #
st.subheader("🎯 Acurácia: modelo × mercado × combinação")

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Acurácia — modelo",
    f"{ev_modelo['acuracia']:.1%}",
    help="Fração de jogos em que a classe mais provável do modelo acertou.",
)
m2.metric(
    "Acurácia — odds",
    f"{ev_mercado['acuracia']:.1%}",
    delta=f"{ev_mercado['acuracia'] - ev_modelo['acuracia']:+.1%} vs modelo",
    help="Fração de jogos em que a classe mais provável das odds acertou.",
)
m3.metric(
    "Acurácia — combinada",
    f"{ev_combinada['acuracia']:.1%}",
    delta=f"{ev_combinada['acuracia'] - ev_mercado['acuracia']:+.1%} vs odds",
    help="Combinação inteligente modelo + odds (limiar definido na barra lateral).",
)
m4.metric(
    "Brier Score — combinada",
    f"{ev_combinada['brier_score']:.3f}",
    delta=f"{ev_combinada['brier_score'] - ev_mercado['brier_score']:+.3f} vs odds",
    delta_color="inverse",
    help="Menor é melhor. Modelo: %.3f · Odds: %.3f."
    % (ev_modelo["brier_score"], ev_mercado["brier_score"]),
)

st.caption(
    f"Jogos avaliados: {ev_modelo['n']} · O mercado divergiu do modelo em "
    f"{int(combinada['usou_mercado'].sum())} jogos (|ΔP| > {threshold:.0%})."
)

# --------------------------------------------------------------------------- #
# Indicador: quando divergiram, quem acertou?
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("🔎 Quando o modelo e o mercado divergiram, quem acertou?")

div_max = cmp.copy()
div_max["divergencia_max"] = np.maximum.reduce(
    [
        (cmp[COLS_MODELO[0]] - cmp[COLS_MERCADO[0]]).abs(),
        (cmp[COLS_MODELO[1]] - cmp[COLS_MERCADO[1]]).abs(),
        (cmp[COLS_MODELO[2]] - cmp[COLS_MERCADO[2]]).abs(),
    ]
)
divergiram = div_max[div_max["divergencia_max"] > threshold].copy()

if divergiram.empty:
    st.info("Nenhuma divergência acima do limiar selecionado.")
else:
    acertos_modelo = (
        (divergiram[list(COLS_MODELO)].to_numpy().argmax(axis=1)
         == divergiram["resultado"].map({"mandante": 0, "empate": 1, "visitante": 2}).to_numpy()).mean()
    )
    acertos_mercado = (
        (divergiram[list(COLS_MERCADO)].to_numpy().argmax(axis=1)
         == divergiram["resultado"].map({"mandante": 0, "empate": 1, "visitante": 2}).to_numpy()).mean()
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Divergências", len(divergiram))
    d2.metric("Acertos do mercado nesses jogos", f"{acertos_mercado:.0%}")
    d3.metric("Acertos do modelo nesses jogos", f"{acertos_modelo:.0%}")

    if acertos_mercado > acertos_modelo:
        st.success(
            "Nas divergências fortes, o mercado acertou mais — evidência de "
            "informação extra (lesões, escalações) embutida nas odds. É por "
            "isso que a combinação usa o mercado nesses casos."
        )
    else:
        st.info(
            "Nas divergências fortes, o modelo acertou mais — as odds podem "
            "estar refletindo overreaction do público (viés de favorito)."
        )

# --------------------------------------------------------------------------- #
# Comparação lado a lado (top divergências)
# --------------------------------------------------------------------------- #
st.subheader("⚖️ Lado a lado — modelo vs. odds (maiores divergências)")

lado_a_lado = divergiram.sort_values("divergencia_max", ascending=False).head(12)
cols_lado = [
    "mandante", "visitante", "resultado",
    *COLS_MODELO, *COLS_MERCADO, "divergencia_max",
]
lado_a_lado = lado_a_lado[cols_lado].rename(
    columns={
        "mandante": "Mandante",
        "visitante": "Visitante",
        "resultado": "Resultado",
        "prob_mandante_modelo": "P(mand.) modelo",
        "prob_empate_modelo": "P(empate) modelo",
        "prob_visitante_modelo": "P(visit.) modelo",
        "prob_mandante_impl": "P(mand.) odds",
        "prob_empate_impl": "P(empate) odds",
        "prob_visitante_impl": "P(visit.) odds",
        "divergencia_max": "Divergência",
    }
)
st.dataframe(
    lado_a_lado,
    width="stretch",
    hide_index=True,
    column_config={
        col: st.column_config.ProgressColumn(format="%.0f%%", min_value=0.0, max_value=1.0)
        for col in lado_a_lado.columns
        if col.startswith("P(")
    }
    | {
        "Divergência": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0.0, max_value=float(divergiram["divergencia_max"].max())
        ),
    },
)

# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
st.divider()
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("🎲 Dispersão: probabilidade do modelo × odds")
    dispersao = cmp.copy()
    dispersao["Classe"] = "Mandante"
    fig = px.scatter(
        dispersao,
        x="prob_mandante_modelo",
        y="prob_mandante_impl",
        hover_name="mandante",
        hover_data={"visitante": True, "resultado": True},
        labels={
            "prob_mandante_modelo": "P(vitória do mandante) — modelo",
            "prob_mandante_impl": "P(vitória do mandante) — odds",
        },
        title="Cada ponto é uma partida (classe: vitória do mandante)",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Igualdade (45°)",
            line=dict(color="#8a8f98", dash="dash"),
        )
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col_g2:
    st.subheader("📉 Diferença entre modelo e odds por partida")
    barras = cmp.copy()
    barras["divergencia_mandante"] = (
        barras["prob_mandante_modelo"] - barras["prob_mandante_impl"]
    )
    barras = barras.sort_values("divergencia_mandante")
    barras["Partida"] = (
        barras["mandante"] + " × " + barras["visitante"]
    )
    fig2 = px.bar(
        barras,
        x="divergencia_mandante",
        y="Partida",
        orientation="h",
        color="divergencia_mandante",
        color_continuous_scale=["#d9534f", "#e8f2ec", "#12a05c"],
        labels={
            "divergencia_mandante": "P(modelo) − P(odds) na vitória do mandante",
        },
        title="Valores positivos: o modelo confia mais no mandante que o mercado",
    )
    fig2.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10),
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------------------------------- #
# Tabela completa
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("📋 Todas as partidas: odds, probabilidades e resultados")

tabela_completa = cmp[
    [
        "data", "fase", "mandante", "visitante",
        "odd_mandante", "odd_empate", "odd_visitante",
        *COLS_MERCADO, *COLS_MODELO,
        "gols_mandante", "gols_visitante", "resultado",
    ]
].sort_values("data").rename(
    columns={
        "data": "Data",
        "fase": "Fase",
        "mandante": "Mandante",
        "visitante": "Visitante",
        "odd_mandante": "Odd mand.",
        "odd_empate": "Odd empate",
        "odd_visitante": "Odd visit.",
        "prob_mandante_impl": "Impl. mand.",
        "prob_empate_impl": "Impl. empate",
        "prob_visitante_impl": "Impl. visit.",
        "prob_mandante_modelo": "Modelo mand.",
        "prob_empate_modelo": "Modelo empate",
        "prob_visitante_modelo": "Modelo visit.",
        "gols_mandante": "Gols mand.",
        "gols_visitante": "Gols visit.",
        "resultado": "Resultado",
    }
)

st.dataframe(
    tabela_completa,
    width="stretch",
    hide_index=True,
    column_config={
        "Odd mand.": st.column_config.NumberColumn(format="%.2f"),
        "Odd empate": st.column_config.NumberColumn(format="%.2f"),
        "Odd visit.": st.column_config.NumberColumn(format="%.2f"),
        **{
            col: st.column_config.ProgressColumn(format="%.0f%%", min_value=0.0, max_value=1.0)
            for col in ["Impl. mand.", "Impl. empate", "Impl. visit.",
                        "Modelo mand.", "Modelo empate", "Modelo visit."]
        },
    },
)

csv = tabela_completa.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Baixar tabela completa (CSV)",
    data=csv,
    file_name="libertadores_2026_modelo_vs_odds.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Fonte das odds: Bzzoiro Sports Data (consenso multi-bookmaker, odds "
    "decimais). Probabilidades implícitas = 1/odd, normalizadas para remover a "
    "margem da casa. Sem chave de API, são usados os dados de exemplo de "
    "`data/examples/` (ver README)."
)
