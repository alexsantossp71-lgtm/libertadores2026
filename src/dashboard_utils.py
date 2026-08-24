"""
Utilitários compartilhados do dashboard Streamlit.

Centraliza o carregamento (com cache) dos dados do projeto — fase de grupos,
estatísticas detalhadas das partidas e odds — e o ajuste do modelo de Poisson,
para que ``app.py`` e as páginas ``pages/*`` usem exatamente a mesma lógica.

Todas as funções têm fallback para ``data/examples/``, então o dashboard
funciona mesmo sem as chaves de API ou sem rodar o pipeline antes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from src.poisson import PoissonScoreModel
from src.preprocessing import Preprocessor
from src.scraper import LibertadoresScraper

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"

REQUIRED_FILES = [
    "grupos_libertadores_2026.csv",
    "oitavas_resultados.csv",
    "confrontos_quartas.csv",
]

FLAGS = {
    "BRA": "🇧🇷", "ARG": "🇦🇷", "ECU": "🇪🇨", "CHI": "🇨🇱", "COL": "🇨🇴",
    "URU": "🇺🇾", "PAR": "🇵🇾", "PER": "🇵🇪", "BOL": "🇧🇴", "VEN": "🇻🇪",
}


def flag(pais: str) -> str:
    return FLAGS.get(str(pais).upper(), "🏳️")


def _ensure_raw_data() -> None:
    """Garante que os CSVs brutos existam; se faltar algum, roda o scraper."""
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        LibertadoresScraper().run()


@st.cache_data(show_spinner="Carregando dados da Libertadores 2026...")
def load_grupos_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega grupos (com features), oitavas e confrontos das quartas."""
    _ensure_raw_data()
    pre = Preprocessor()
    grupos, oitavas, quartas = pre.load_data()
    grupos_features = pre.create_features(grupos)
    return grupos_features, oitavas, quartas


@st.cache_resource(show_spinner="Ajustando modelo de Poisson...")
def fit_model(home_advantage: float = 1.15, max_goals: int = 10) -> PoissonScoreModel:
    """Ajusta o modelo de Poisson com os dados da fase de grupos."""
    grupos_features, _, _ = load_grupos_features()
    model = PoissonScoreModel(home_advantage=home_advantage, max_goals=max_goals)
    model.fit(grupos_features)
    try:
        from src.elenco_analysis import analisar_elencos, aplicar_elenco_ao_poisson, forma_recente

        aplicar_elenco_ao_poisson(model, analisar_elencos(), forma=forma_recente())
    except FileNotFoundError:
        pass
    return model


@st.cache_data(show_spinner="Carregando estatísticas detalhadas...")
def load_estatisticas() -> pd.DataFrame:
    """Carrega estatísticas detalhadas com colunas derivadas (totais)."""
    pre = Preprocessor()
    df = pre.load_estatisticas()
    return pre.process_estatisticas(df)


@st.cache_data(show_spinner="Carregando odds de mercado...")
def load_odds() -> pd.DataFrame:
    """Carrega odds processadas (probabilidades implícitas normalizadas)."""
    return Preprocessor().process_odds(Preprocessor().load_odds())


@st.cache_data(show_spinner="Carregando perfil dos árbitros...")
def load_referee_summary() -> pd.DataFrame:
    """Resumo descritivo por árbitro."""
    return Preprocessor().referee_summary(Preprocessor().load_estatisticas())


@st.cache_data(show_spinner="Comparando modelo x mercado...")
def load_model_market() -> pd.DataFrame:
    """Junta probabilidades do modelo e implícitas das odds por partida."""
    model = fit_model()
    return Preprocessor().model_vs_market(model, Preprocessor().load_odds())
