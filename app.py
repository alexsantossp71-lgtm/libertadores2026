"""
⚽ Libertadores 2026 — Dashboard Interativo (Streamlit)

Aplicativo web que expõe o modelo de Regressão de Poisson do projeto
(``src/poisson.py``) em uma interface visual:

  * Visão geral da fase de grupos e das forças de ataque/defesa estimadas;
  * Previsões das quartas de final (probabilidades 1X2, gols esperados, placar);
  * Simulador de confronto livre entre quaisquer dois times, com matriz de
    placares, mercados derivados (over/under, ambos marcam) e placares prováveis;
  * Simulação de Monte Carlo do mata-mata (quartas → título).

Execução::

    streamlit run app.py

Os dados são 100% reais (openfootball 2012–2026 + suplementos ESPN/FBref),
versionados em ``data/historical/`` e materializados em ``data/raw/*.csv``
pelo ``src/real_data.py``. Nenhum dado simulado é usado.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuração de paths / imports do projeto
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poisson import PoissonScoreModel  # noqa: E402
from preprocessing import Preprocessor  # noqa: E402
from scraper import LibertadoresScraper  # noqa: E402

REQUIRED_FILES = [
    "grupos_libertadores_2026.csv",
    "oitavas_resultados.csv",
    "confrontos_quartas.csv",
]

FLAGS = {
    "BRA": "🇧🇷",
    "ARG": "🇦🇷",
    "ECU": "🇪🇨",
    "CHI": "🇨🇱",
    "COL": "🇨🇴",
    "URU": "🇺🇾",
    "PAR": "🇵🇾",
    "PER": "🇵🇪",
    "BOL": "🇧🇴",
    "VEN": "🇻🇪",
}

# --------------------------------------------------------------------------- #
# Configuração da página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Libertadores 2026 — Previsões",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
      div[data-testid="stMetric"] {
          background: linear-gradient(160deg, #10261c 0%, #0d1f17 100%);
          border: 1px solid #1f4d38;
          border-radius: 12px;
          padding: 14px 16px;
      }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .hero {
          background: linear-gradient(110deg, #04361f 0%, #0a6b3d 55%, #12a05c 100%);
          border-radius: 16px;
          padding: 26px 30px;
          margin-bottom: 22px;
          color: #ffffff;
      }
      .hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; letter-spacing: -0.5px; }
      .hero p  { margin: 0; opacity: 0.88; font-size: 1rem; }
      .matchcard {
          border: 1px solid #234a37;
          border-radius: 14px;
          padding: 16px 18px;
          margin-bottom: 12px;
          background: rgba(16, 38, 28, 0.45);
      }
      .matchcard .teams { font-size: 1.12rem; font-weight: 700; margin-bottom: 2px; }
      .matchcard .meta  { font-size: 0.82rem; opacity: 0.7; }
      .pill {
          display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 0.75rem; font-weight: 600; background: #12a05c; color: #042413;
      }
      .disclaimer { font-size: 0.82rem; opacity: 0.75; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Carga de dados e modelo (com cache)
# --------------------------------------------------------------------------- #
def _ensure_raw_data() -> None:
    """Garante que os CSVs brutos existam; se faltar algum, roda o scraper."""
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        LibertadoresScraper().run()


@st.cache_data(show_spinner="Carregando dados da Libertadores 2026...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega grupos (com features), oitavas e confrontos das quartas."""
    _ensure_raw_data()
    pre = Preprocessor()
    grupos, oitavas, quartas = pre.load_data()
    grupos_features = pre.create_features(grupos)
    return grupos_features, oitavas, quartas


@st.cache_resource(show_spinner="Ajustando modelo de Poisson...")
def fit_model(
    grupos_features: pd.DataFrame, home_advantage: float, max_goals: int
) -> PoissonScoreModel:
    """Ajusta o modelo de Poisson com os parâmetros escolhidos."""
    model = PoissonScoreModel(home_advantage=home_advantage, max_goals=max_goals)
    model.fit(grupos_features)
    return model


def flag(pais: str) -> str:
    return FLAGS.get(str(pais).upper(), "🏳️")


def team_country(grupos: pd.DataFrame, time: str) -> str:
    row = grupos.loc[grupos["Time"] == time, "Pais"]
    return str(row.iloc[0]) if len(row) else ""


def prob_bar(p_home: float, p_draw: float, p_away: float, home: str, away: str) -> go.Figure:
    """Barra horizontal empilhada com as probabilidades 1X2."""
    fig = go.Figure()
    for label, value, color in [
        (home, p_home, "#12a05c"),
        ("Empate", p_draw, "#8a8f98"),
        (away, p_away, "#d9534f"),
    ]:
        fig.add_bar(
            y=["1X2"],
            x=[value * 100],
            name=label,
            orientation="h",
            marker_color=color,
            text=[f"{label}<br>{value:.1%}"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        height=110,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def derived_markets(matrix: np.ndarray) -> Dict[str, float]:
    """Mercados derivados da matriz de placares."""
    n = matrix.shape[0]
    idx_home = np.arange(n).reshape(-1, 1)
    idx_away = np.arange(n).reshape(1, -1)
    total = idx_home + idx_away

    both_score = matrix[1:, 1:].sum()
    over25 = matrix[total > 2.5].sum()
    over15 = matrix[total > 1.5].sum()
    clean_sheet_home = matrix[:, 0].sum()
    clean_sheet_away = matrix[0, :].sum()

    return {
        "Ambos marcam": float(both_score),
        "Mais de 1.5 gols": float(over15),
        "Mais de 2.5 gols": float(over25),
        "Menos de 2.5 gols": float(1 - over25),
        "Mandante não sofre gol": float(clean_sheet_home),
        "Visitante não sofre gol": float(clean_sheet_away),
    }


def top_scores(matrix: np.ndarray, n: int = 6) -> pd.DataFrame:
    """Top-N placares mais prováveis."""
    flat = matrix.flatten()
    order = np.argsort(flat)[::-1][:n]
    rows = []
    for k in order:
        i, j = np.unravel_index(k, matrix.shape)
        rows.append({"Placar": f"{i} x {j}", "Probabilidade": float(flat[k])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Simulação de mata-mata (Monte Carlo)
# --------------------------------------------------------------------------- #
def simulate_tie(
    model: PoissonScoreModel, team_a: str, team_b: str, rng: np.random.Generator
) -> str:
    """Simula um confronto de ida e volta (A manda o jogo de ida)."""
    lam_a1, lam_b1 = model.expected_goals(team_a, team_b)  # ida: A em casa
    lam_b2, lam_a2 = model.expected_goals(team_b, team_a)  # volta: B em casa

    gols_a = rng.poisson(lam_a1) + rng.poisson(lam_a2)
    gols_b = rng.poisson(lam_b1) + rng.poisson(lam_b2)

    if gols_a > gols_b:
        return team_a
    if gols_b > gols_a:
        return team_b
    # Empate agregado -> pênaltis (moeda justa)
    return team_a if rng.random() < 0.5 else team_b


@st.cache_data(show_spinner="Simulando o mata-mata...")
def monte_carlo_bracket(
    pairs: List[Tuple[str, str]],
    n_sims: int,
    home_advantage: float,
    max_goals: int,
    grupos_features: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Simula quartas → semis → final N vezes e retorna probabilidades por fase."""
    model = PoissonScoreModel(home_advantage=home_advantage, max_goals=max_goals)
    model.fit(grupos_features)

    rng = np.random.default_rng(seed)
    teams = sorted({t for pair in pairs for t in pair})
    semi = {t: 0 for t in teams}
    final = {t: 0 for t in teams}
    champ = {t: 0 for t in teams}

    for _ in range(n_sims):
        qf_winners = [simulate_tie(model, a, b, rng) for a, b in pairs]
        for w in qf_winners:
            semi[w] += 1

        sf_winners = []
        for k in range(0, len(qf_winners) - 1, 2):
            sf_winners.append(simulate_tie(model, qf_winners[k], qf_winners[k + 1], rng))
        for w in sf_winners:
            final[w] += 1

        if len(sf_winners) >= 2:
            champion = simulate_tie(model, sf_winners[0], sf_winners[1], rng)
        elif sf_winners:
            champion = sf_winners[0]
        else:
            continue
        champ[champion] += 1

    df = pd.DataFrame(
        {
            "Time": teams,
            "Semifinal": [semi[t] / n_sims for t in teams],
            "Final": [final[t] / n_sims for t in teams],
            "Título": [champ[t] / n_sims for t in teams],
        }
    ).sort_values("Título", ascending=False, ignore_index=True)
    return df


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📑 Páginas")
    st.page_link("app.py", label="🏠 Início — Previsões", icon="⚽")
    st.page_link(
        "pages/5_Arbitragem.py",
        label="🟨 Arbitragem e Estatísticas",
        icon="🟨",
    )
    st.page_link(
        "pages/6_Odds.py",
        label="📊 Odds e Probabilidades de Mercado",
        icon="📊",
    )
    st.divider()

    st.markdown("## ⚙️ Parâmetros do modelo")

    home_advantage = st.slider(
        "Vantagem de mando de campo",
        min_value=1.00,
        max_value=1.50,
        value=1.15,
        step=0.01,
        help=(
            "Multiplicador aplicado aos gols esperados do mandante. "
            "O fator do visitante é 2 - vantagem, preservando o total de gols."
        ),
    )
    max_goals = st.slider(
        "Máximo de gols na matriz",
        min_value=5,
        max_value=12,
        value=10,
        step=1,
        help="Truncamento da cauda da distribuição de Poisson.",
    )

    st.divider()
    st.markdown("## 🗂️ Dados")
    if st.button("🔄 Recarregar dados reais", width="stretch"):
        LibertadoresScraper().run()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Dados atualizados!")

    st.caption(
        "Fonte: dados reais — openfootball/south-america (2012–2026) "
        "+ ESPN/FBref (`src/real_data.py`). Nada simulado."
    )

    st.divider()
    st.caption("Modelo: Poisson (Maher 1982 / Dixon & Coles 1997), forma multiplicativa.")


# --------------------------------------------------------------------------- #
# Dados + modelo
# --------------------------------------------------------------------------- #
grupos, oitavas, quartas = load_data()
model = fit_model(grupos, home_advantage, max_goals)

st.markdown(
    """
    <div class="hero">
      <h1>⚽ Copa Libertadores 2026 — Previsões</h1>
      <p>Dashboard interativo do modelo de Regressão de Poisson · probabilidades 1X2,
      gols esperados, placares e simulação do mata-mata.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Times analisados", len(model.teams))
c2.metric("Média de gols/jogo", f"{model.league_avg:.2f}")
c3.metric("Vantagem de mando", f"{model.home_advantage:.2f}×")
c4.metric("Confrontos nas quartas", len(quartas))

# --------------------------------------------------------------------------- #
# Métricas resumidas das novas análises (arbitragem e odds)
# --------------------------------------------------------------------------- #
try:
    from dashboard_utils import load_referee_summary, load_model_market
    from preprocessing import Preprocessor as _Preprocessor

    _resumo_arbitros = load_referee_summary()
    _model_market = load_model_market()

    if not _resumo_arbitros.empty:
        _arbitro_top = _resumo_arbitros.iloc[0]
        _arbitro_nome = _arbitro_top["arbitro"]
        _arbitro_faltas = _arbitro_top["media_faltas"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "🟨 Árbitro com mais faltas",
            _arbitro_nome,
            help=f"Média de {_arbitro_faltas:.1f} faltas por jogo.",
        )
        m2.metric(
            "🟥 Cartões/jogo (média)",
            f"{_resumo_arbitros['media_cartoes'].mean():.2f}",
            help="Cartões amarelos + vermelhos por partida.",
        )

        if not _model_market.empty:
            _pre = _Preprocessor()
            _ev_odds = _pre.evaluate_probabilities(
                _model_market,
                ("prob_mandante_impl", "prob_empate_impl", "prob_visitante_impl"),
            )
            _ev_modelo = _pre.evaluate_probabilities(
                _model_market,
                (
                    "prob_mandante_modelo",
                    "prob_empate_modelo",
                    "prob_visitante_modelo",
                ),
            )
            m3.metric(
                "📊 Acertos das odds",
                f"{_ev_odds['acuracia']:.0%}",
                delta=f"{_ev_odds['acuracia'] - _ev_modelo['acuracia']:+.0%} vs modelo",
                help=(
                    "Fração de jogos em que a classe mais provável das odds "
                    "acertou o resultado."
                ),
            )
            m4.metric(
                "🎯 Acertos do modelo (Poisson)",
                f"{_ev_modelo['acuracia']:.0%}",
                help=(
                    "Fração de jogos em que a classe mais provável do modelo "
                    "acertou o resultado."
                ),
            )
        else:
            m3.metric("📊 Acertos das odds", "—")
            m4.metric("🎯 Acertos do modelo", "—")
except FileNotFoundError:
    st.info(
        "As análises de arbitragem e odds exigem chaves de API reais "
        "(API_FUTEBOL_KEY / BSD_API — ver `.env.example`). "
        "Sem chaves, nenhuma base simulada é usada."
    )
except Exception as exc:  # análises opcionais: não quebram o dashboard
    with st.expander("⚠️ Análises opcionais indisponíveis (detalhes)"):
        st.write(exc)

tab_geral, tab_quartas, tab_sim, tab_mc, tab_sobre = st.tabs(
    ["📊 Visão geral", "🔮 Quartas de final", "⚔️ Simulador", "🎲 Monte Carlo", "ℹ️ Sobre"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — Visão geral
# --------------------------------------------------------------------------- #
with tab_geral:
    st.subheader("Fase de grupos")

    tabela = grupos.copy()
    tabela.insert(0, "🏳️", tabela["Pais"].map(flag))
    cols = [
        "🏳️", "Time", "Pais", "Pts", "J", "V", "E", "D", "GP", "GC", "SG",
        "Aproveitamento", "Media_Gols_Marcados", "Media_Gols_Sofridos", "Score_Forca",
    ]
    tabela = tabela[cols].sort_values("Pts", ascending=False, ignore_index=True)

    st.dataframe(
        tabela,
        width="stretch",
        hide_index=True,
        column_config={
            "Aproveitamento": st.column_config.ProgressColumn(
                "Aproveitamento", format="%.0f%%", min_value=0.0, max_value=1.0
            ),
            "Media_Gols_Marcados": st.column_config.NumberColumn(
                "Gols marcados/jogo", format="%.2f"
            ),
            "Media_Gols_Sofridos": st.column_config.NumberColumn(
                "Gols sofridos/jogo", format="%.2f"
            ),
            "Score_Forca": st.column_config.NumberColumn("Score de força", format="%.1f"),
        },
    )

    st.subheader("Forças estimadas pelo modelo")
    col_a, col_b = st.columns([1.15, 1])

    forcas = model.strengths().merge(grupos[["Time", "Pais"]], on="Time", how="left")

    with col_a:
        fig = px.scatter(
            forcas,
            x="Defesa",
            y="Ataque",
            text="Time",
            color="Pais",
            size=[16] * len(forcas),
            labels={"Ataque": "Ataque (gols marcados/jogo)", "Defesa": "Defesa (gols sofridos/jogo)"},
            title="Ataque × Defesa — quadrante superior esquerdo é o melhor",
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.add_vline(x=forcas["Defesa"].mean(), line_dash="dot", line_color="#7d848c")
        fig.add_hline(y=forcas["Ataque"].mean(), line_dash="dot", line_color="#7d848c")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    with col_b:
        ranking = forcas.assign(Saldo=forcas["Ataque"] - forcas["Defesa"]).sort_values(
            "Saldo", ascending=True
        )
        fig2 = px.bar(
            ranking,
            x="Saldo",
            y="Time",
            orientation="h",
            color="Saldo",
            color_continuous_scale=["#d9534f", "#8a8f98", "#12a05c"],
            title="Saldo por jogo (ataque − defesa)",
        )
        fig2.update_layout(
            height=470, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False
        )
        st.plotly_chart(fig2, width="stretch")

    with st.expander("📋 Resultados das oitavas de final"):
        st.dataframe(oitavas, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 2 — Quartas de final
# --------------------------------------------------------------------------- #
with tab_quartas:
    st.subheader("Previsões das quartas de final")

    linhas = []
    for _, row in quartas.iterrows():
        mandante, visitante = row["Mandante"], row["Visitante"]
        if mandante not in model.teams or visitante not in model.teams:
            st.warning(f"Confronto ignorado: {mandante} x {visitante} (time fora da base).")
            continue
        p = model.match_probabilities(mandante, visitante)
        favorito = max(
            [(mandante, p["p_home"]), ("Empate", p["p_draw"]), (visitante, p["p_away"])],
            key=lambda t: t[1],
        )[0]
        linhas.append(
            {
                "Confronto": row["Confronto"],
                "Mandante": mandante,
                "Visitante": visitante,
                "Pais_Mandante": row.get("Pais_Mandante", ""),
                "Pais_Visitante": row.get("Pais_Visitante", ""),
                "Data_Ida": row.get("Data_Ida", ""),
                "Data_Volta": row.get("Data_Volta", ""),
                "Prob_Mandante": p["p_home"],
                "Prob_Empate": p["p_draw"],
                "Prob_Visitante": p["p_away"],
                "xG_Mandante": p["expected_goals_home"],
                "xG_Visitante": p["expected_goals_away"],
                "Placar_Previsto": f"{p['most_likely_score'][0]} x {p['most_likely_score'][1]}",
                "Favorito": favorito,
            }
        )

    preds = pd.DataFrame(linhas)

    if preds.empty:
        st.info("Nenhum confronto disponível para previsão.")
    else:
        cards = st.columns(2)
        for i, row in preds.iterrows():
            with cards[i % 2]:
                st.markdown(
                    f"""
                    <div class="matchcard">
                      <div class="teams">{flag(row['Pais_Mandante'])} {row['Mandante']}
                          <span style="opacity:.6">x</span>
                          {row['Visitante']} {flag(row['Pais_Visitante'])}</div>
                      <div class="meta">{row['Confronto']} · Ida {row['Data_Ida']} ·
                          Volta {row['Data_Volta']} ·
                          <span class="pill">Favorito: {row['Favorito']}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    prob_bar(
                        row["Prob_Mandante"],
                        row["Prob_Empate"],
                        row["Prob_Visitante"],
                        row["Mandante"],
                        row["Visitante"],
                    ),
                    width="stretch",
                    key=f"bar_{row['Confronto']}",
                )
                m1, m2 = st.columns(2)
                m1.metric(
                    "Gols esperados",
                    f"{row['xG_Mandante']:.2f} x {row['xG_Visitante']:.2f}",
                )
                m2.metric("Placar mais provável", row["Placar_Previsto"])

        st.divider()
        st.markdown("#### Tabela consolidada")
        st.dataframe(
            preds.drop(columns=["Pais_Mandante", "Pais_Visitante"]),
            width="stretch",
            hide_index=True,
            column_config={
                "Prob_Mandante": st.column_config.ProgressColumn(
                    "P(mandante)", format="%.1f%%", min_value=0.0, max_value=1.0
                ),
                "Prob_Empate": st.column_config.ProgressColumn(
                    "P(empate)", format="%.1f%%", min_value=0.0, max_value=1.0
                ),
                "Prob_Visitante": st.column_config.ProgressColumn(
                    "P(visitante)", format="%.1f%%", min_value=0.0, max_value=1.0
                ),
                "xG_Mandante": st.column_config.NumberColumn("xG mandante", format="%.2f"),
                "xG_Visitante": st.column_config.NumberColumn("xG visitante", format="%.2f"),
            },
        )

        csv = preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar previsões (CSV)",
            data=csv,
            file_name="quartas_previsao.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Tab 3 — Simulador de confronto
# --------------------------------------------------------------------------- #
with tab_sim:
    st.subheader("Simulador de confronto")

    times = model.teams
    col1, col2 = st.columns(2)
    mandante = col1.selectbox("🏠 Mandante", times, index=times.index("Flamengo") if "Flamengo" in times else 0)
    visitante_opts = [t for t in times if t != mandante]
    visitante = col2.selectbox("✈️ Visitante", visitante_opts, index=0)

    probs = model.match_probabilities(mandante, visitante)
    matrix = model.score_probability_matrix(mandante, visitante)

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Vitória {mandante}", f"{probs['p_home']:.1%}")
    k2.metric("Empate", f"{probs['p_draw']:.1%}")
    k3.metric(f"Vitória {visitante}", f"{probs['p_away']:.1%}")

    st.plotly_chart(
        prob_bar(probs["p_home"], probs["p_draw"], probs["p_away"], mandante, visitante),
        width="stretch",
        key="bar_simulador",
    )

    g1, g2 = st.columns([1.3, 1])

    with g1:
        show = min(6, matrix.shape[0] - 1)
        sub = matrix[: show + 1, : show + 1]
        heat = px.imshow(
            sub * 100,
            labels=dict(x=f"Gols {visitante}", y=f"Gols {mandante}", color="Prob. (%)"),
            x=list(range(show + 1)),
            y=list(range(show + 1)),
            color_continuous_scale="Greens",
            text_auto=".1f",
            aspect="auto",
            title="Matriz de probabilidade de placares (%)",
        )
        heat.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(heat, width="stretch")

    with g2:
        st.markdown("#### Gols esperados")
        e1, e2 = st.columns(2)
        e1.metric(mandante, f"{probs['expected_goals_home']:.2f}")
        e2.metric(visitante, f"{probs['expected_goals_away']:.2f}")

        st.markdown("#### Placares mais prováveis")
        tops = top_scores(matrix, 6)
        st.dataframe(
            tops,
            width="stretch",
            hide_index=True,
            column_config={
                "Probabilidade": st.column_config.ProgressColumn(
                    "Probabilidade", format="%.1f%%", min_value=0.0, max_value=float(tops["Probabilidade"].max())
                )
            },
        )

        st.markdown("#### Mercados derivados")
        mercados = derived_markets(matrix)
        st.dataframe(
            pd.DataFrame({"Mercado": mercados.keys(), "Probabilidade": mercados.values()}),
            width="stretch",
            hide_index=True,
            column_config={
                "Probabilidade": st.column_config.ProgressColumn(
                    "Probabilidade", format="%.1f%%", min_value=0.0, max_value=1.0
                )
            },
        )


# --------------------------------------------------------------------------- #
# Tab 4 — Monte Carlo
# --------------------------------------------------------------------------- #
with tab_mc:
    st.subheader("Simulação do mata-mata (Monte Carlo)")
    st.caption(
        "Cada confronto é disputado em ida e volta (com mando alternado); "
        "empate no agregado é decidido por pênaltis (50/50). "
        "Chaveamento: QF1×QF2 → SF1 e QF3×QF4 → SF2."
    )

    pares = [
        (r["Mandante"], r["Visitante"])
        for _, r in quartas.iterrows()
        if r["Mandante"] in model.teams and r["Visitante"] in model.teams
    ]

    if len(pares) < 2:
        st.info("São necessários ao menos 2 confrontos válidos para simular o mata-mata.")
    else:
        n_sims = st.select_slider(
            "Número de simulações", options=[1000, 5000, 10000, 25000, 50000], value=10000
        )

        resultado = monte_carlo_bracket(
            pares, int(n_sims), home_advantage, max_goals, grupos
        )

        top = resultado.iloc[0]
        st.success(
            f"🏆 Favorito ao título: **{top['Time']}** "
            f"({top['Título']:.1%} em {n_sims:,} simulações)".replace(",", ".")
        )

        col1, col2 = st.columns([1.25, 1])

        with col1:
            longo = resultado.melt(
                id_vars="Time",
                value_vars=["Semifinal", "Final", "Título"],
                var_name="Fase",
                value_name="Probabilidade",
            )
            fig = px.bar(
                longo,
                x="Probabilidade",
                y="Time",
                color="Fase",
                orientation="h",
                barmode="group",
                color_discrete_sequence=["#8a8f98", "#3d8f6a", "#12a05c"],
                title="Probabilidade de avanço por fase",
            )
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.dataframe(
                resultado,
                width="stretch",
                hide_index=True,
                column_config={
                    "Semifinal": st.column_config.ProgressColumn(
                        "Semifinal", format="%.1f%%", min_value=0.0, max_value=1.0
                    ),
                    "Final": st.column_config.ProgressColumn(
                        "Final", format="%.1f%%", min_value=0.0, max_value=1.0
                    ),
                    "Título": st.column_config.ProgressColumn(
                        "Título", format="%.1f%%", min_value=0.0, max_value=1.0
                    ),
                },
            )
            st.download_button(
                "⬇️ Baixar simulação (CSV)",
                data=resultado.to_csv(index=False).encode("utf-8"),
                file_name="simulacao_mata_mata.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------- #
# Tab 5 — Sobre
# --------------------------------------------------------------------------- #
with tab_sobre:
    st.subheader("Sobre o modelo")
    st.markdown(
        r"""
O número de gols de cada time é modelado por uma distribuição de Poisson cuja
taxa depende da força de ataque do time, da fragilidade defensiva do adversário
e do mando de campo:

$$\lambda_{casa} = \frac{ataque_{casa} \times defesa_{fora} \times vantagem_{casa}}{média_{liga}}$$

$$\lambda_{fora} = \frac{ataque_{fora} \times defesa_{casa} \times fator_{fora}}{média_{liga}}$$

A probabilidade de um placar $(i, j)$ é o produto das marginais
$P(i;\lambda_{casa}) \cdot P(j;\lambda_{fora})$, e as probabilidades 1X2 saem da
soma dos triângulos e da diagonal da matriz de placares.

**Referências:** Maher (1982); Dixon & Coles (1997).
        """
    )

    st.markdown("#### Parâmetros atuais")
    st.json(
        {
            "home_advantage": model.home_advantage,
            "away_factor": round(model.away_factor, 4),
            "max_goals": model.max_goals,
            "league_avg_gols_por_jogo": round(model.league_avg, 4),
            "times": len(model.teams),
        }
    )

    st.markdown("#### Estrutura do projeto")
    st.code(
        "src/real_data.py      → dados reais 2012–2026 (openfootball + ESPN/FBref)\n"
        "src/scraper.py        → materializa as tabelas do dashboard a partir deles\n"
        "src/preprocessing.py  → engenharia de features\n"
        "src/poisson.py        → modelo de Poisson (usado por este app)\n"
        "src/model.py          → XGBoost + Poisson\n"
        "src/predict.py        → geração de previsões em CSV\n"
        "src/pipeline.py       → pipeline completo em linha de comando\n"
        "app.py                → este dashboard Streamlit",
        language="text",
    )

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ <b>Aviso:</b> as previsões são estatísticas e baseadas em dados <b>reais</b>
        da fase de grupos 2026 (32 times · openfootball + ESPN/FBref). Lesões,
        suspensões, calendário, decisões táticas e mercado de transferências
        não são considerados. Uso educacional — não é recomendação de aposta.
        </div>
        """,
        unsafe_allow_html=True,
    )
