"""Testes do pré-processamento e engenharia de features."""

import pandas as pd
import pytest

from preprocessing import Preprocessor


@pytest.fixture
def grupos_df():
    return pd.DataFrame(
        {
            "Time": ["Time_A", "Time_B"],
            "Pais": ["BRA", "ARG"],
            "Pts": [12, 6],
            "J": [6, 6],
            "V": [4, 2],
            "E": [0, 0],
            "D": [2, 4],
            "GP": [10, 4],
            "GC": [4, 10],
            "SG": [6, -6],
        }
    )


def test_create_features_adds_expected_columns(grupos_df):
    preprocessor = Preprocessor()
    df = preprocessor.create_features(grupos_df)

    expected = {
        "Aproveitamento",
        "Media_Gols_Marcados",
        "Media_Gols_Sofridos",
        "Razao_Gols",
        "Vitorias_Seq",
        "Pais_Cod",
        "Score_Forca",
    }
    assert expected.issubset(set(df.columns))


def test_create_features_computes_aproveitamento(grupos_df):
    preprocessor = Preprocessor()
    df = preprocessor.create_features(grupos_df)
    row_a = df[df["Time"] == "Time_A"].iloc[0]
    assert row_a["Aproveitamento"] == pytest.approx(12 / (6 * 3))
    assert row_a["Media_Gols_Marcados"] == pytest.approx(10 / 6)


def test_pais_cod_mapping(grupos_df):
    preprocessor = Preprocessor()
    df = preprocessor.create_features(grupos_df)
    pais_cod = dict(zip(df["Time"], df["Pais_Cod"]))
    assert pais_cod["Time_A"] == 3  # BRA
    assert pais_cod["Time_B"] == 2  # ARG


def test_create_match_features_diffs(grupos_df):
    preprocessor = Preprocessor()
    features = preprocessor.create_features(grupos_df)
    match = preprocessor.create_match_features(features, "Time_A", "Time_B")

    assert match["Time_Mandante"].values[0] == "Time_A"
    assert match["Time_Visitante"].values[0] == "Time_B"
    assert match["Diff_Pts"].values[0] == pytest.approx(12 - 6)
    assert match["Diff_SG"].values[0] == pytest.approx(6 - (-6))
    assert match["Mesmo_Pais"].values[0] == 0
