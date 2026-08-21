"""
🟨 Análise de Arbitragem e Estatísticas — Libertadores 2026

Página do dashboard que explora a influência da arbitragem (faltas, cartões)
e das estatísticas detalhadas (posse, passes, finalizações) nos resultados da
fase de grupos e das oitavas de final.

Fonte dos dados: ``data/processed/libertadores_estatisticas_detalhadas.csv``
(gerado pelo pipeline via API Futebol, com fallback para ``data/examples/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard_utils import load_estatisticas, load_referee_summary  # noqa: E402
from preprocessing import Preprocessor  # noqa: E402

st.set_page_config(
    page_title="🟨 Análise de Arbitragem e Estatísticas",
    page_icon="🟨",
    layout="wide",
)

st.title("🟨 Análise de Arbitragem e Estatísticas")
st.caption(
    "Faltas, cartões, posse, passes e finalizações por partida — e como o "
    "estilo de cada árbitro se relaciona com os resultados."
)

# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
pre = Preprocessor()
partidas = load_estatisticas()          # com colunas derivadas (totais)
resumo = load_referee_summary()          # médias por árbitro
rigor = pre.add_rigor_groups(partidas)   # tercis de rigor por média de faltas

ARBITROS = sorted(resumo["arbitro"].tolist())
PAISES = sorted(resumo["arbitro_pais"].dropna().unique().tolist())

# --------------------------------------------------------------------------- #
# Métricas gerais
# --------------------------------------------------------------------------- #
media_faltas_liga = partidas["total_faltas"].mean()
media_cartoes_liga = partidas["total_cartoes"].mean()
media_gols_liga = partidas["total_gols"].mean()
mais_rigoroso = resumo.iloc[0]["arbitro"]
mais_permissivo = resumo.iloc[-1]["arbitro"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Partidas analisadas", len(partidas))
c2.metric("Média de faltas/jogo", f"{media_faltas_liga:.1f}")
c3.metric("Média de cartões/jogo", f"{media_cartoes_liga:.2f}")
c4.metric("Média de gols/jogo", f"{media_gols_liga:.2f}")

st.info(
    f"**Árbitro mais rigoroso:** {mais_rigoroso} "
    f"({resumo.iloc[0]['media_faltas']:.1f} faltas/jogo) · "
    f"**mais permissivo:** {mais_permissivo} "
    f"({resumo.iloc[-1]['media_faltas']:.1f} faltas/jogo)"
)

# --------------------------------------------------------------------------- #
# 1. Perfil do árbitro
# --------------------------------------------------------------------------- #
st.subheader("👤 Perfil do árbitro")

col_sel, col_perfil = st.columns([1, 2.6])
arbitro_selecionado = col_sel.selectbox("Escolha o árbitro", ARBITROS)

perfil = resumo[resumo["arbitro"] == arbitro_selecionado].iloc[0]
jogos_do_arbitro = rigor[rigor["arbitro"] == arbitro_selecionado]

with col_perfil:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jogos apitados", int(perfil["jogos"]))
    m2.metric("Faltas/jogo", f"{perfil['media_faltas']:.1f}")
    m3.metric("Cartões/jogo", f"{perfil['media_cartoes']:.2f}")
    m4.metric("Gols/jogo", f"{perfil['media_gols']:.2f}")

    detalhes = (
        f"🇦🇷 País: **{perfil['arbitro_pais']}** · "
        f"🟨 Amarelos: {perfil['media_cartoes_amarelos']:.2f}/jogo · "
        f"🟥 Vermelhos: {perfil['media_cartoes_vermelhos']:.2f}/jogo · "
        f"🎯 Finalizações: {perfil['media_finalizacoes']:.1f}/jogo · "
        f"🅿️ Posse média do mandante: {perfil['media_posse_mandante']:.1f}%"
    )
    st.markdown(detalhes)
    st.progress(
        float(perfil["media_faltas"] / max(resumo["media_faltas"].max(), 1)),
        text=f"Rigor (faltas): {perfil['media_faltas']:.1f} de "
             f"{resumo['media_faltas'].max():.1f} (máx.)",
    )

    resultados = jogos_do_arbitro["resultado"].value_counts()
    st.caption(
        "Resultados dos jogos apitados: "
        + " · ".join(f"{k}: {v}" for k, v in resultados.items())
    )

# --------------------------------------------------------------------------- #
# 2. Gráficos principais
# --------------------------------------------------------------------------- #
st.divider()
col_a, col_b = st.columns([1.2, 1])

with col_a:
    st.subheader("📊 Top árbitros por média de faltas")
    top10 = resumo.head(10).sort_values("media_faltas")
    fig = px.bar(
        top10,
        x="media_faltas",
        y="arbitro",
        orientation="h",
        color="media_faltas",
        color_continuous_scale=["#12a05c", "#e2b13c", "#d9534f"],
        labels={"media_faltas": "Faltas/jogo (média)", "arbitro": ""},
        title="Média de faltas por jogo (top 10 mais rigorosos)",
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10),
                      coloraxis_showscale=False)
    fig.add_vline(
        x=media_faltas_liga, line_dash="dot", line_color="#8a8f98",
        annotation_text=f"Média da competição ({media_faltas_liga:.1f})",
    )
    st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("🟨 Distribuição de cartões por árbitro")
    pais_filtro = st.selectbox(
        "Filtrar por país do árbitro", ["Todos"] + PAISES, key="pais_boxplot"
    )
    df_box = partidas if pais_filtro == "Todos" else partidas[
        partidas["arbitro_pais"] == pais_filtro
    ]
    fig2 = px.box(
        df_box,
        x="arbitro",
        y="total_cartoes",
        color="arbitro",
        points="all",
        labels={"total_cartoes": "Cartões no jogo (amarelos + vermelhos)", "arbitro": ""},
        title="Cartões por jogo, por árbitro",
    )
    fig2.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10),
                       showlegend=False)
    st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------------------------------- #
# 3. Dispersão faltas x gols
# --------------------------------------------------------------------------- #
st.subheader("⚽ Faltas × Gols por partida")
color_map = {
    "mandante": "Vitória do mandante",
    "empate": "Empate",
    "visitante": "Vitória do visitante",
}
df_scatter = partidas.copy()
df_scatter["Resultado"] = df_scatter["resultado"].map(color_map)

fig3 = px.scatter(
    df_scatter,
    x="total_faltas",
    y="total_gols",
    color="Resultado",
    size="total_cartoes",
    hover_name="mandante",
    hover_data={
        "visitante": True,
        "arbitro": True,
        "total_faltas": True,
        "total_gols": True,
        "total_cartoes": True,
    },
    color_discrete_map={
        "Vitória do mandante": "#12a05c",
        "Empate": "#8a8f98",
        "Vitória do visitante": "#d9534f",
    },
    labels={
        "total_faltas": "Faltas na partida (total)",
        "total_gols": "Gols na partida (total)",
    },
    title="Jogos com mais faltas tendem a ter menos gols",
)
trend = np.polyfit(df_scatter["total_faltas"], df_scatter["total_gols"], 1)
xs = np.linspace(
    df_scatter["total_faltas"].min(), df_scatter["total_faltas"].max(), 50
)
fig3.add_trace(
    go.Scatter(
        x=xs, y=trend[0] * xs + trend[1], mode="lines", name="Tendência linear",
        line=dict(color="#e2b13c", dash="dash", width=2),
    )
)
fig3.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig3, width="stretch")

# --------------------------------------------------------------------------- #
# 4. Radar — comparação entre dois árbitros
# --------------------------------------------------------------------------- #
st.subheader("🕸️ Radar — comparação do perfil de dois árbitros")
col_r1, col_r2 = st.columns(2)
arbitro_a = col_r1.selectbox("Árbitro A", ARBITROS, key="radar_a")
arbitro_b = col_r2.selectbox(
    "Árbitro B", [a for a in ARBITROS if a != arbitro_a], key="radar_b"
)

DIMENSOES = [
    ("Faltas", "media_faltas"),
    ("Cartões", "media_cartoes"),
    ("Passes certos", "media_passes_certos"),
    ("Posse (mandante)", "media_posse_mandante"),
    ("Finalizações", "media_finalizacoes"),
    ("Impedimentos", "media_impedimentos"),
]

# Normaliza 0-1 por dimensão (para comparabilidade no radar)
fig4 = go.Figure()
for arbitro, cor in [(arbitro_a, "#12a05c"), (arbitro_b, "#d9534f")]:
    linha = resumo[resumo["arbitro"] == arbitro].iloc[0]
    valores = []
    for _, col in DIMENSOES:
        minimo, maximo = resumo[col].min(), resumo[col].max()
        valor = (linha[col] - minimo) / (maximo - minimo) if maximo > minimo else 0.5
        valores.append(round(float(valor), 3))
    fig4.add_trace(
        go.Scatterpolar(
            r=valores + valores[:1],
            theta=[d[0] for d in DIMENSOES] + [DIMENSOES[0][0]],
            fill="toself",
            name=arbitro,
            line_color=cor,
        )
    )
fig4.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=460,
    margin=dict(l=60, r=60, t=40, b=40),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig4, width="stretch")
st.caption("Valores normalizados (0–1) por dimensão para permitir comparação direta.")

# --------------------------------------------------------------------------- #
# 5. Testes estatísticos
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("🧪 Testes estatísticos")

grupos_rigor = [
    g["total_gols"].dropna() for _, g in rigor.groupby("grupo_rigor")
]
f_stat, p_anova = stats.f_oneway(*grupos_rigor)

r_faltas, p_faltas = stats.pearsonr(rigor["total_faltas"], rigor["total_gols"])
r_cartoes, p_cartoes = stats.pearsonr(rigor["total_cartoes"], rigor["total_gols"])
r_posse, p_posse = stats.pearsonr(rigor["posse_mandante"], rigor["total_gols"])

t1, t2, t3 = st.columns(3)
with t1:
    st.metric(
        "ANOVA: gols × grupo de rigor",
        f"F = {f_stat:.2f}",
        delta=f"p = {p_anova:.4f}",
        delta_color="inverse" if p_anova < 0.05 else "normal",
    )
    st.caption(
        f"Médias de gols por grupo — "
        + " · ".join(
            f"{g}: {rigor[rigor['grupo_rigor'] == g]['total_gols'].mean():.2f}"
            for g in rigor["grupo_rigor"].dropna().unique()
        )
    )
with t2:
    st.metric(
        "Correlação faltas × gols",
        f"r = {r_faltas:.3f}",
        delta=f"p = {p_faltas:.4f}",
        delta_color="inverse" if p_faltas < 0.05 else "normal",
    )
with t3:
    st.metric(
        "Correlação cartões × gols",
        f"r = {r_cartoes:.3f}",
        delta=f"p = {p_cartoes:.4f}",
        delta_color="inverse" if p_cartoes < 0.05 else "normal",
    )

st.caption(
    f"Correlação posse do mandante × gols: r = {r_posse:.3f} (p = {p_posse:.4f}). "
    "Significância adotada: α = 0,05."
)

if p_anova < 0.05 and r_faltas < 0:
    st.success(
        "**Conclusão:** há diferença significativa de gols entre os grupos de "
        "rigor da arbitragem (ANOVA, p < 0,05) e correlação negativa "
        "significativa entre faltas e gols — indício de que árbitros mais "
        "permissivos estão associados a jogos com mais gols."
    )
else:
    st.warning(
        "Nesta amostra, os testes não atingiram significância a 5% — o efeito "
        "da arbitragem pode exigir uma amostra maior para ser detectado."
    )

# --------------------------------------------------------------------------- #
# 6. Tabela interativa
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("📋 Estatísticas por árbitro (tabela interativa)")
st.caption("Clique nos cabeçalhos das colunas para ordenar.")

colunas_tabela = {
    "arbitro": "Árbitro",
    "arbitro_pais": "País",
    "jogos": "Jogos",
    "media_faltas": "Faltas/jogo",
    "media_cartoes": "Cartões/jogo",
    "media_gols": "Gols/jogo",
    "media_posse_mandante": "Posse mand. (%)",
    "media_passes_certos": "Passes certos/jogo",
    "media_finalizacoes": "Finalizações/jogo",
    "media_escanteios": "Escanteios/jogo",
    "media_impedimentos": "Impedimentos/jogo",
}
tabela = resumo[list(colunas_tabela)].rename(columns=colunas_tabela)
st.dataframe(
    tabela,
    width="stretch",
    hide_index=True,
    column_config={
        "Faltas/jogo": st.column_config.NumberColumn(format="%.1f"),
        "Cartões/jogo": st.column_config.NumberColumn(format="%.2f"),
        "Gols/jogo": st.column_config.NumberColumn(format="%.2f"),
        "Posse mand. (%)": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.divider()
st.caption(
    "Fonte: API Futebol (api-futebol.com.br) — dados da fase de grupos e "
    "oitavas da Libertadores 2026. Sem chave de API, são usados os dados de "
    "exemplo de `data/examples/` (ver README)."
)
