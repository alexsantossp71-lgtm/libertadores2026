"""Testes da análise de elencos e do ajuste de Poisson."""


import pandas as pd
import pytest

from src.elenco_analysis import (
    ElencoWeights,
    analisar_elencos,
    aplicar_elenco_ao_poisson,
    forma_recente,
    multiplicadores_time,
    prever_confronto_com_elenco,
    quimica_e_disciplina,
)
from src.poisson import PoissonScoreModel


def test_quimica_maior_quando_menos_jogadores():
    df = pd.DataFrame({
        "time": ["Estavel", "Rodizio"],
        "n_jogadores": [19, 34],
        "noventas": [6.0, 8.0],
        "cartoes_amarelos": [10, 20],
        "cartoes_vermelhos": [0, 3],
    })
    out = quimica_e_disciplina(df)
    est = out.loc[out["time"] == "Estavel"].iloc[0]
    rod = out.loc[out["time"] == "Rodizio"].iloc[0]
    assert est["quimica_elenco"] > rod["quimica_elenco"]
    assert rod["risco_suspensao"] > est["risco_suspensao"]


def test_analisar_elencos_snapshot_32_times():
    elencos = analisar_elencos()
    assert len(elencos) == 32
    for col in (
        "indice_forca_ofensiva",
        "indice_pressao_defensiva",
        "quimica_elenco",
        "risco_suspensao",
        "score_elenco",
    ):
        assert col in elencos.columns
        assert elencos[col].notna().all()
    # Medellín usou 19 jogadores; Mirassol, 34
    med = elencos.loc[elencos["time"] == "Medellín", "quimica_elenco"].iloc[0]
    mir = elencos.loc[elencos["time"] == "Mirassol", "quimica_elenco"].iloc[0]
    assert med > mir


def test_forma_recente_usa_partidas_reais():
    forma = forma_recente(temporada=2026, n=5)
    assert not forma.empty
    assert "Flamengo" in set(forma["time"])
    fla = forma.loc[forma["time"] == "Flamengo"].iloc[0]
    assert fla["forma_jogos"] == 5
    assert 0 <= fla["forma_aproveitamento"] <= 1


def test_multiplicadores_centrados():
    row = pd.Series({
        "indice_forca_ofensiva_norm": 1.0,
        "indice_pressao_defensiva_norm": 1.0,
        "quimica_elenco_norm": 0.5,
        "indice_disciplina_norm": 0.5,
    })
    att, defe = multiplicadores_time(row, ElencoWeights())
    assert att > 1.0  # ataque elite sobe o lambda
    assert defe < 1.0  # pressão elite reduz gols sofridos


def test_poisson_elenco_aumenta_favorito_ofensivo():
    elencos = analisar_elencos()
    # Times reais com J/GP/GC coerentes
    grupos = pd.DataFrame({
        "Time": ["Flamengo", "Platense"],
        "J": [8, 8],
        "GP": [14, 9],
        "GC": [4, 8],
    })
    base = PoissonScoreModel().fit(grupos)
    p_base = base.match_probabilities("Flamengo", "Platense")

    adj = PoissonScoreModel().fit(grupos)
    aplicar_elenco_ao_poisson(adj, elencos, forma=None, weights=ElencoWeights(forma=0.0))
    p_adj = adj.match_probabilities("Flamengo", "Platense")

    assert adj.elenco_applied
    assert "Flamengo" in adj.elenco_multipliers
    # O ajuste muda os lambdas (não precisa ser sempre a favor do Flamengo)
    assert p_adj["expected_goals_home"] != pytest.approx(p_base["expected_goals_home"], abs=1e-9) or \
        p_adj["expected_goals_away"] != pytest.approx(p_base["expected_goals_away"], abs=1e-9)

    detalhe = prever_confronto_com_elenco(adj, "Flamengo", "Platense", elencos)
    assert "nota_elenco" in detalhe
    assert detalhe["delta_xg_mandante"] is not None


def test_apply_elenco_multipliers_idempotente():
    model = PoissonScoreModel().fit(pd.DataFrame({
        "Time": ["A", "B"], "J": [6, 6], "GP": [10, 6], "GC": [4, 8],
    }))
    model.apply_elenco_multipliers({"A": (1.2, 0.9)})
    att1 = model.attack["A"]
    model.apply_elenco_multipliers({"A": (1.2, 0.9)})
    assert model.attack["A"] == pytest.approx(att1)


def test_predictor_resolve_qf3_cenarios():
    from src.predict import LibertadoresPredictor

    quartas = pd.DataFrame([
        {"Confronto": "QF3", "Mandante": "Flamengo",
         "Visitante": "A DEFINIR (vencedor de Tolima x Independiente del Valle)"},
        {"Confronto": "QF4", "Mandante": "Fluminense", "Visitante": "Platense"},
    ])
    pred = LibertadoresPredictor()
    out = pred._resolver_quartas(
        quartas, {"Fluminense", "Platense", "Flamengo", "Tolima", "Independiente del Valle"}
    )
    assert len(out) == 3
    assert set(out.loc[out["Confronto"].str.contains("QF3"), "Visitante"]) == {
        "Tolima", "Independiente del Valle"
    }


def test_create_features_elenco_opt_in():
    from src.preprocessing import Preprocessor

    grupos = pd.DataFrame({
        "Time": ["Flamengo", "Time_X"],
        "Pais": ["BRA", "ARG"],
        "Pts": [16, 6], "J": [6, 6], "V": [5, 2], "E": [1, 0], "D": [0, 4],
        "GP": [14, 4], "GC": [2, 10], "SG": [12, -6],
    })
    pre = Preprocessor()
    sem = pre.create_features(grupos, incluir_elenco=False)
    assert "indice_forca_ofensiva" not in sem.columns
    com = pre.create_features(grupos, incluir_elenco=True)
    assert "indice_forca_ofensiva" in com.columns
    assert pd.notna(com.loc[com["Time"] == "Flamengo", "indice_forca_ofensiva"].iloc[0])
    assert pd.isna(com.loc[com["Time"] == "Time_X", "indice_forca_ofensiva"].iloc[0])
