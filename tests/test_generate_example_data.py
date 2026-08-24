"""Testes do gerador de dados de exemplo (determinismo e esquema)."""

import numpy as np
import pandas as pd

from generate_example_data import (
    ODDS_COLUMNS,
    PARTIDAS_COLUMNS,
    generate_odds,
    generate_partidas,
    run,
)


def test_generation_is_deterministic():
    df1 = generate_partidas()
    df2 = generate_partidas()
    pd.testing.assert_frame_equal(df1, df2)

    odds1 = generate_odds(df1)
    odds2 = generate_odds(df2)
    pd.testing.assert_frame_equal(odds1, odds2)


def test_partidas_schema_and_groups():
    df = generate_partidas()
    assert list(df.columns) == PARTIDAS_COLUMNS
    assert len(df) == 48  # 36 jogos de grupos + 12 de oitavas (ida/volta)

    grupos = df[df["fase"] == "Fase de Grupos"]
    jogos_por_time = pd.concat(
        [grupos["mandante"], grupos["visitante"]]
    ).value_counts()
    assert (jogos_por_time == 6).all()  # cada time joga 6 vezes na fase de grupos

    assert df["resultado"].isin(["mandante", "empate", "visitante"]).all()
    assert df[["gols_mandante", "gols_visitante"]].notna().all().all()


def test_partidas_have_all_referees_and_stats():
    df = generate_partidas()
    assert df["arbitro"].nunique() >= 6
    stat_cols = [c for c in df.columns if c.startswith(("faltas", "cartoes", "posse", "passes"))]
    assert df[stat_cols].notna().all().all()


def test_odds_schema_and_valid_ranges():
    partidas = generate_partidas()
    odds = generate_odds(partidas)
    assert list(odds.columns) == ODDS_COLUMNS
    assert len(odds) == len(partidas)

    # Odds decimais plausíveis
    assert (odds[["odd_mandante", "odd_empate", "odd_visitante"]] > 1.0).all().all()

    # Probabilidades implícitas normalizadas somam ~1
    soma = odds[["prob_mandante_impl", "prob_empate_impl", "prob_visitante_impl"]].sum(axis=1)
    assert np.allclose(soma, 1.0, atol=1e-2)

    # Margem positiva
    assert (odds["margem"] > 0).all()


def test_run_saves_files(tmp_path):
    partidas, odds = run(output_dir=tmp_path)
    assert (tmp_path / "partidas_libertadores_2026.csv").exists()
    assert (tmp_path / "odds_libertadores_2026.csv").exists()
    assert len(partidas) == len(odds)
