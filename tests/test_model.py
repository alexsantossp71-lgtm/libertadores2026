"""Testes de integração entre o modelo e o pipeline de previsão."""

import pandas as pd
import pytest

from model import LibertadoresModel
from preprocessing import Preprocessor


@pytest.fixture
def grupos_df():
    """Dados da fase de grupos com os times usados nas quartas."""
    return pd.DataFrame(
        {
            "Time": ["Time_Forte", "Time_Fraco"],
            "Pais": ["BRA", "ARG"],
            "Pts": [16, 4],
            "J": [6, 6],
            "V": [5, 1],
            "E": [1, 1],
            "D": [0, 4],
            "GP": [15, 3],
            "GC": [2, 14],
            "SG": [13, -11],
        }
    )


def test_fit_poisson_and_predict_match(grupos_df):
    preprocessor = Preprocessor()
    features = preprocessor.create_features(grupos_df)

    model = LibertadoresModel()
    model.fit_poisson(features)

    match = preprocessor.create_match_features(features, "Time_Forte", "Time_Fraco")
    probs = model.predict_match_poisson(match)

    assert probs["prob_vitoria_mandante"] > probs["prob_derrota_mandante"]
    assert probs["resultado_previsto"] == "Mandante"
    total = (
        probs["prob_vitoria_mandante"]
        + probs["prob_empate"]
        + probs["prob_derrota_mandante"]
    )
    assert total == pytest.approx(1.0)


def test_predict_score_returns_non_negative_ints(grupos_df):
    preprocessor = Preprocessor()
    features = preprocessor.create_features(grupos_df)

    model = LibertadoresModel()
    model.fit_poisson(features)

    match = preprocessor.create_match_features(features, "Time_Forte", "Time_Fraco")
    gols_casa, gols_fora = model.predict_score(match)

    assert isinstance(gols_casa, int) and isinstance(gols_fora, int)
    assert gols_casa >= 0 and gols_fora >= 0


def test_predict_without_fit_raises(grupos_df):
    preprocessor = Preprocessor()
    features = preprocessor.create_features(grupos_df)

    model = LibertadoresModel()  # Poisson ainda não ajustado
    match = preprocessor.create_match_features(features, "Time_Forte", "Time_Fraco")

    with pytest.raises(RuntimeError):
        model.predict_match_poisson(match)
