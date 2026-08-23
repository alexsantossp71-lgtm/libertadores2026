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


@pytest.fixture
def historia_csv(tmp_path):
    """CSV sintético determinístico com as 3 classes representadas no treino.

    ``Forte`` vence ``Fraco`` em todas as partidas dele (mandante e visitante);
    partidas entre ``OutroA``/``OutroB`` terminam empatadas para fornecer a
    classe ``empate`` (alvo 1) e garantir classes contíguas no split temporal.
    O 1º jogo tem Forte visitante (Elo já diverge antes de qualquer partida em
    que Forte seja mandante) e o último tem Forte mandante.
    """
    m, v, r = "mandante", "visitante", "empate"
    specs = [
        ("Fraco", "Forte", 0, 3, v),
        ("Forte", "Fraco", 3, 0, m),
        ("OutroA", "OutroB", 1, 1, r),
        ("Forte", "Fraco", 3, 0, m),
        ("Fraco", "Forte", 0, 3, v),
        ("OutroA", "OutroB", 1, 1, r),
        ("Forte", "Fraco", 3, 0, m),
        ("Fraco", "Forte", 0, 3, v),
        ("OutroA", "OutroB", 1, 1, r),
        ("Forte", "Fraco", 3, 0, m),
        ("Fraco", "Forte", 0, 3, v),
        ("Forte", "Fraco", 3, 0, m),
        ("Fraco", "Forte", 0, 3, v),
        ("OutroA", "OutroB", 1, 1, r),
        ("Forte", "Fraco", 3, 0, m),
        ("Fraco", "Forte", 0, 3, v),
        ("Forte", "Fraco", 3, 0, m),
        ("Fraco", "Forte", 0, 3, v),
        ("Forte", "Fraco", 3, 0, m),
        ("Forte", "Fraco", 3, 0, m),
    ]
    rows = []
    for i, (mand, vis, gols_m, gols_v, res) in enumerate(specs):
        rows.append(
            {
                "data": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "mandante": mand,
                "visitante": vis,
                "pais_mandante": "BRA" if mand == "Forte" else "ECU",
                "pais_visitante": "BRA" if vis == "Forte" else "COL",
                "fase": "Group Stage",
                "gols_mandante": gols_m,
                "gols_visitante": gols_v,
                "resultado": res,
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "hist.csv"
    df.to_csv(path, index=False)
    return path


def test_classifier_learns_from_real_history(historia_csv):
    """Sanidade do classificador XGBoost reescrito (features causais).

    Com ``Forte`` vencendo sempre, as features devem refletir essa vantagem
    (``Diff_Elo > 0`` e ``Ataque_M`` acima da média inicial da liga de 1.25)
    e ``train()`` deve rodar com split temporal sem quebrar.
    """
    model = LibertadoresModel()
    X, y = model.prepare_training_data(csv_path=str(historia_csv))

    assert X.shape == (20, 10)
    assert set(y).issubset({0, 1, 2})
    assert len(model.feature_names) == 10

    df = pd.read_csv(historia_csv, parse_dates=["data"])
    for i, m in df.iterrows():
        if m["mandante"] == "Forte":
            assert X[i, 0] > 0, "Diff_Elo deveria ser positivo após vitórias do Forte"
            assert X[i, 1] > 1.25, "Ataque_M do Forte deveria superar a média inicial da liga"

    results = model.train(X, y, test_size=0.25)
    assert "accuracy" in results
    assert "log_loss" in results
    assert model.is_trained is True
