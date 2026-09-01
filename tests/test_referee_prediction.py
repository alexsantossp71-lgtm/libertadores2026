"""Testes do módulo de previsão de árbitros (src/referee_prediction.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.referee_prediction import (
    TEAM_MATCH_WEIGHT,
    arbitros_mais_provaveis,
)


def _fixture_df() -> pd.DataFrame:
    """Pequeno dataset determinístico com histórico de times e fases."""
    linhas = [
        # árbitro A: 3 jogos, 2 envolvendo o Flamengo, 1 de mata-mata
        ("Arbitro A", "BRA", "Flamengo", "Palmeiras", "Fase de Grupos"),
        ("Arbitro A", "BRA", "River Plate", "Flamengo", "Oitavas de Final"),
        ("Arbitro A", "BRA", "São Paulo", "Cerro", "Fase de Grupos"),
        # árbitro B: 2 jogos de mata-mata, sem envolver os times-alvo
        ("Arbitro B", "ARG", "Grêmio", "Tolima", "Oitavas de Final"),
        ("Arbitro B", "ARG", "Peñarol", "Nacional", "Preliminar"),
        # árbitro C: 1 jogo de grupos
        ("Arbitro C", np.nan, "Atlético Mineiro", "Libertad", "Fase de Grupos"),
        # linha sem árbitro — deve ser ignorada
        (np.nan, np.nan, "Flamengo", "Palmeiras", "Fase de Grupos"),
    ]
    return pd.DataFrame(
        linhas,
        columns=["arbitro", "arbitro_pais", "mandante", "visitante", "fase"],
    )


def test_top2_probabilidades_validas():
    df = _fixture_df()
    res = arbitros_mais_provaveis(df, "Flamengo", "Palmeiras")
    assert len(res) == 2
    probs = [p for _, _, p in res]
    # probabilidades do subconjunto top-2: cada uma em (0, 1), soma <= 1
    assert all(0.0 < p <= 1.0 for p in probs)
    assert sum(probs) <= 1.0 + 1e-9
    # ordenadas desc
    assert probs[0] >= probs[1]


def test_historico_do_time_tem_boost():
    df = _fixture_df()
    res = arbitros_mais_provaveis(df, "Flamengo", "Palmeiras")
    # Arbitro A apitou jogos do Flamengo → deve ser o 1º
    assert res[0][0] == "Arbitro A"

    # efeito numérico do bônus: sem histórico de time, A teria prob menor
    # times sem nenhuma partida apitada pelo mesmo árbitro do histórico
    res_sem = arbitros_mais_provaveis(df, "Cerro", "Libertad")
    p_a_sem = next(p for n, _, p in res_sem if n == "Arbitro A")
    assert res[0][2] > p_a_sem


def test_fallback_distribuicao_global_sem_historico():
    df = _fixture_df()
    res = arbitros_mais_provaveis(df, "Time Inédito X", "Time Inédito Y")
    assert len(res) == 2
    # sem bônus de time nem de fase → ordenação = contagem global (A, B)
    assert [n for n, _, _ in res] == ["Arbitro A", "Arbitro B"]


def test_bonus_mata_mata():
    df = _fixture_df()
    # B tem só histórico de mata-mata; com bônus deve superar C claramente
    res = arbitros_mais_provaveis(
        df, "Nacional", "Libertad", fase="Quartas de Final"
    )
    p_b = next(p for n, _, p in res if n == "Arbitro B")
    # contagem global de B = 1 real + sem times; com boost > C (1.0)
    assert p_b > 0.2
    # determinismo: repetir dá o mesmo resultado
    assert arbitros_mais_provaveis(df, "Nacional", "Libertad") == arbitros_mais_provaveis(
        df, "Nacional", "Libertad"
    )


def test_pais_nan_tratado():
    df = _fixture_df()
    res = arbitros_mais_provaveis(df, "Atlético Mineiro", "Libertad", top_n=3)
    pais_c = next(pais for n, pais, _ in res if n == "Arbitro C")
    assert pais_c is None  # NaN → None, sem quebrar


def test_peso_do_time_documentado_e_efetivo():
    df = _fixture_df()
    assert TEAM_MATCH_WEIGHT > 0
    res = arbitros_mais_provaveis(df, "Flamengo", "Palmeiras", top_n=3)
    probs = {n: p for n, _, p in res}
    assert probs["Arbitro A"] > probs["Arbitro B"]


def test_df_vazio_ou_invalido():
    vazio = pd.DataFrame(
        columns=["arbitro", "arbitro_pais", "mandante", "visitante", "fase"]
    )
    assert arbitros_mais_provaveis(vazio, "A", "B") == []
    assert arbitros_mais_provaveis(None, "A", "B") == []
