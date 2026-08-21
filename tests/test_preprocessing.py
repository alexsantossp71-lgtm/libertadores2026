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


# --------------------------------------------------------------------------- #
# Novas funcionalidades: estatísticas detalhadas, odds e arbitragem
# --------------------------------------------------------------------------- #
def _partidas_exemplo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "partida_id": [1, 2, 3, 4],
            "data": ["2026-04-14"] * 4,
            "fase": ["Fase de Grupos"] * 4,
            "rodada": ["1ª Rodada"] * 4,
            "grupo": ["Grupo A"] * 4,
            "mandante": ["Time_A", "Time_B", "Time_A", "Time_B"],
            "visitante": ["Time_B", "Time_A", "Time_B", "Time_A"],
            "gols_mandante": [2, 1, 0, 1],
            "gols_visitante": [0, 1, 2, 1],
            "resultado": ["mandante", "empate", "visitante", "empate"],
            "arbitro": ["Árbitro X", "Árbitro Y", "Árbitro X", "Árbitro Y"],
            "arbitro_pais": ["Brasil", "Argentina", "Brasil", "Argentina"],
            "faltas_mandante": [10, 8, 12, 9],
            "faltas_visitante": [8, 10, 9, 9],
            "cartoes_amarelos_mandante": [2, 1, 3, 1],
            "cartoes_amarelos_visitante": [1, 2, 1, 2],
            "cartoes_vermelhos_mandante": [0, 0, 1, 0],
            "cartoes_vermelhos_visitante": [0, 1, 0, 0],
            "posse_mandante": [55, 60, 45, 50],
            "posse_visitante": [45, 40, 55, 50],
            "passes_certos_mandante": [400, 420, 380, 410],
            "passes_certos_visitante": [380, 400, 420, 390],
            "passes_errados_mandante": [80, 75, 90, 85],
            "passes_errados_visitante": [85, 80, 75, 90],
            "finalizacoes_mandante": [14, 12, 10, 13],
            "finalizacoes_visitante": [10, 11, 14, 12],
            "finalizacoes_no_gol_mandante": [6, 5, 4, 6],
            "finalizacoes_no_gol_visitante": [4, 5, 7, 5],
            "finalizacoes_fora_mandante": [8, 7, 6, 7],
            "finalizacoes_fora_visitante": [6, 6, 7, 7],
            "escanteios_mandante": [5, 4, 3, 6],
            "escanteios_visitante": [3, 4, 6, 4],
            "impedimentos_mandante": [1, 2, 0, 1],
            "impedimentos_visitante": [2, 1, 2, 2],
            "defesas_mandante": [3, 4, 5, 4],
            "defesas_visitante": [4, 3, 2, 5],
            "fonte": ["exemplo"] * 4,
        }
    )


def test_process_estatisticas_totals():
    preprocessor = Preprocessor()
    df = preprocessor.process_estatisticas(_partidas_exemplo())
    assert df["total_faltas"].iloc[0] == 18
    assert df["total_cartoes"].iloc[0] == 3
    assert df["total_gols"].iloc[0] == 2
    assert df["aproveitamento_mandante"].iloc[2] == 0.0  # derrota do mandante


def test_referee_summary():
    preprocessor = Preprocessor()
    resumo = preprocessor.referee_summary(_partidas_exemplo())
    assert set(resumo["arbitro"]) == {"Árbitro X", "Árbitro Y"}
    assert "media_faltas" in resumo.columns
    assert "media_gols" in resumo.columns
    # Árbitro X tem mais faltas que Y (22/2 vs 18/2... usar médias)
    media_x = resumo.loc[resumo["arbitro"] == "Árbitro X", "media_faltas"].iloc[0]
    assert media_x > 10


def test_add_rigor_groups():
    preprocessor = Preprocessor()
    df = preprocessor.add_rigor_groups(_partidas_exemplo())
    assert "grupo_rigor" in df.columns
    assert df["grupo_rigor"].notna().all()


def test_evaluate_probabilities_accuracy_and_brier():
    preprocessor = Preprocessor()
    df = pd.DataFrame(
        {
            "resultado": ["mandante", "visitante"],
            "p_mandante": [0.6, 0.3],
            "p_empate": [0.3, 0.3],
            "p_visitante": [0.1, 0.4],
        }
    )
    ev = preprocessor.evaluate_probabilities(
        df, ("p_mandante", "p_empate", "p_visitante")
    )
    assert ev["acuracia"] == 1.0  # ambas as previsões corretas
    # Brier multiclasse: (1-0.6)^2+(0-0.3)^2+(0-0.1)^2 = 0.26 e
    # (0-0.3)^2+(0-0.3)^2+(1-0.4)^2 = 0.54 -> média 0.40
    assert ev["brier_score"] == pytest.approx(0.40)
    assert ev["n"] == 2


def test_combined_probabilities_rule():
    preprocessor = Preprocessor()
    df = pd.DataFrame(
        {
            "prob_mandante_modelo": [0.5, 0.5],
            "prob_empate_modelo": [0.3, 0.3],
            "prob_visitante_modelo": [0.2, 0.2],
            "prob_mandante_impl": [0.7, 0.55],
            "prob_empate_impl": [0.2, 0.25],
            "prob_visitante_impl": [0.1, 0.20],
        }
    )
    out = preprocessor.combined_probabilities(df, threshold=0.08)
    # 1ª linha: divergência 0.2 > 0.08 -> usa mercado
    assert out["prob_mandante_combinada"].iloc[0] == pytest.approx(0.7)
    # 2ª linha: divergência 0.05 <= 0.08 -> usa modelo
    assert out["prob_mandante_combinada"].iloc[1] == pytest.approx(0.5)
    assert list(out["usou_mercado"]) == [True, False]


def test_load_estatisticas_raises_without_flag(monkeypatch, tmp_path):
    """Sem ALLOW_EXAMPLE_DATA a base sintética é recusada (só dados reais)."""
    import sys
    from pathlib import Path

    monkeypatch.delenv("ALLOW_EXAMPLE_DATA", raising=False)
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import api_futebol_client
    import odds_client

    monkeypatch.setattr(api_futebol_client, "PROCESSED_PATH", tmp_path / "nao_existe.csv")
    monkeypatch.setattr(odds_client, "PROCESSED_PATH", tmp_path / "nao_existe.csv")

    preprocessor = Preprocessor()
    with pytest.raises(FileNotFoundError, match="ALLOW_EXAMPLE_DATA"):
        preprocessor.load_estatisticas()
    with pytest.raises(FileNotFoundError, match="ALLOW_EXAMPLE_DATA"):
        preprocessor.load_odds()


def test_load_estatisticas_and_odds_fallback(monkeypatch, tmp_path):
    """Com ALLOW_EXAMPLE_DATA=1 os loaders funcionam offline (bases de exemplo)."""
    import sys
    from pathlib import Path

    monkeypatch.setenv("ALLOW_EXAMPLE_DATA", "1")
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import api_futebol_client
    import odds_client

    monkeypatch.setattr(api_futebol_client, "PROCESSED_PATH", tmp_path / "nao_existe.csv")
    monkeypatch.setattr(odds_client, "PROCESSED_PATH", tmp_path / "nao_existe.csv")

    preprocessor = Preprocessor()
    partidas = preprocessor.load_estatisticas()
    odds = preprocessor.load_odds()
    assert len(partidas) > 0
    assert {"mandante", "visitante", "arbitro"}.issubset(partidas.columns)
    assert {"odd_mandante", "odd_empate", "odd_visitante"}.issubset(odds.columns)


def test_model_vs_market_joins_probabilities():
    preprocessor = Preprocessor()
    grupos = pd.DataFrame(
        {
            "Time": ["Time_A", "Time_B"],
            "Pais": ["BRA", "ARG"],
            "Pts": [12, 6], "J": [6, 6], "V": [4, 2], "E": [0, 0], "D": [2, 4],
            "GP": [10, 4], "GC": [4, 10], "SG": [6, -6],
        }
    )
    from poisson import PoissonScoreModel

    model = PoissonScoreModel()
    model.fit(preprocessor.create_features(grupos))

    odds = pd.DataFrame(
        {
            "partida_id": [1],
            "mandante": ["Time_A"],
            "visitante": ["Time_B"],
            "odd_mandante": [2.0],
            "odd_empate": [3.2],
            "odd_visitante": [4.5],
            "resultado": ["mandante"],
        }
    )
    out = preprocessor.model_vs_market(model, odds)
    assert len(out) == 1
    assert "prob_mandante_modelo" in out.columns
    assert "prob_mandante_impl" in out.columns
    assert "divergencia_mandante" in out.columns
    assert out["prob_mandante_impl"].iloc[0] > 0.4  # favorito claro
