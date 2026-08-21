"""Testes do modelo de Regressão de Poisson (src/poisson.py)."""

import numpy as np
import pandas as pd
import pytest

from poisson import PoissonScoreModel, REQUIRED_COLUMNS


@pytest.fixture
def grupos_df():
    """Tabela agregada sintética da fase de grupos (formato esperado)."""
    return pd.DataFrame(
        {
            "Time": [
                "Time_Forte", "Time_Medio", "Time_Fraco", "Time_Equilibrado",
            ],
            "J": [6, 6, 6, 6],
            "GP": [18, 10, 3, 9],
            "GC": [3, 8, 15, 9],
        }
    )


@pytest.fixture
def fitted(grupos_df):
    return PoissonScoreModel().fit(grupos_df)


def test_fit_builds_strengths_and_league_average(grupos_df):
    model = PoissonScoreModel().fit(grupos_df)

    assert model.is_fitted is True
    assert set(model.teams) == set(grupos_df["Time"])
    # Ataque = GP / J ; Defesa = GC / J
    assert model.attack["Time_Forte"] == pytest.approx(18 / 6)
    assert model.defense["Time_Fraco"] == pytest.approx(15 / 6)
    # Média da liga = total de gols marcados / total de jogos
    expected_avg = grupos_df["GP"].sum() / grupos_df["J"].sum()
    assert model.league_avg == pytest.approx(expected_avg)


def test_fit_requires_mandatory_columns(grupos_df):
    df = grupos_df.drop(columns=["GC"])
    with pytest.raises(ValueError):
        PoissonScoreModel().fit(df)


def test_fit_ignores_teams_without_games(grupos_df):
    df = grupos_df.copy()
    df.loc[df["Time"] == "Time_Fraco", "J"] = 0
    model = PoissonScoreModel().fit(df)
    assert "Time_Fraco" not in model.teams


def test_fit_raises_on_zero_league_average(grupos_df):
    df = grupos_df.copy()
    df["GP"] = 0
    with pytest.raises(ValueError):
        PoissonScoreModel().fit(df)


def test_expected_goals_balanced_match_preserves_total():
    # Para um time exatamente na média da liga (ataque == defesa == média),
    # o total esperado de gols de um confronto entre dois times idênticos deve
    # ser igual a 2 * média da liga (independente da vantagem de mando).
    df = pd.DataFrame({"Time": ["Time_X"], "J": [6], "GP": [12], "GC": [12]})
    model = PoissonScoreModel().fit(df)  # league_avg = att = def = 2.0
    lam_home, lam_away = model.expected_goals("Time_X", "Time_X")
    assert (lam_home + lam_away) == pytest.approx(2 * model.league_avg, rel=1e-9)


def test_home_advantage_increases_home_goals(grupos_df):
    model_high = PoissonScoreModel(home_advantage=1.30).fit(grupos_df)
    model_low = PoissonScoreModel(home_advantage=1.05).fit(grupos_df)
    home_goals_high = model_high.expected_goals("Time_Forte", "Time_Fraco")[0]
    home_goals_low = model_low.expected_goals("Time_Forte", "Time_Fraco")[0]
    assert home_goals_high > home_goals_low


def test_stronger_team_has_higher_win_probability(fitted):
    probs = fitted.match_probabilities("Time_Forte", "Time_Fraco")
    assert probs["p_home"] > probs["p_away"]


def test_match_probabilities_sum_to_one(fitted):
    probs = fitted.match_probabilities("Time_Forte", "Time_Medio")
    assert probs["p_home"] + probs["p_draw"] + probs["p_away"] == pytest.approx(1.0)


def test_score_matrix_is_probability_distribution(fitted):
    matrix = fitted.score_probability_matrix("Time_Medio", "Time_Fraco", max_goals=15)
    assert matrix.shape == (16, 16)
    assert np.all(matrix >= 0)
    # Com max_goals alto, a cauda truncada é desprezível
    assert matrix.sum() == pytest.approx(1.0, abs=1e-6)


def test_most_likely_score_is_argmax_of_matrix(fitted):
    matrix = fitted.score_probability_matrix("Time_Medio", "Time_Fraco", max_goals=10)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    probs = fitted.match_probabilities("Time_Medio", "Time_Fraco", max_goals=10)
    assert probs["most_likely_score"] == (int(i), int(j))


def test_unknown_team_raises_key_error(fitted):
    with pytest.raises(KeyError):
        fitted.expected_goals("Time_Inexistente", "Time_Forte")


def test_predictions_before_fit_raise_runtime_error():
    model = PoissonScoreModel()
    with pytest.raises(RuntimeError):
        model.expected_goals("A", "B")


def test_strengths_dataframe_is_sorted_by_attack(grupos_df):
    model = PoissonScoreModel().fit(grupos_df)
    strengths = model.strengths()
    assert list(strengths.columns) == ["Time", "Ataque", "Defesa"]
    assert strengths["Ataque"].is_monotonic_decreasing
    assert strengths.iloc[0]["Time"] == "Time_Forte"


def test_repr_contains_status(grupos_df):
    model = PoissonScoreModel().fit(grupos_df)
    assert "ajustado" in repr(model)
    assert "Time_Forte" not in repr(model)  # não deve vazar nomes de times
