"""Testes do scraper FBref (parse de HTML local, fallback, índices)."""

from pathlib import Path

import pandas as pd
import pytest

from src.fbref_features import confronto_indices, indices_elencos, indices_jogadores
from src.fbref_scraper import (
    FBrefClient,
    FBrefUnavailableError,
    canonical_squad,
    extract_tables,
    page_url,
    save_sqlite,
    split_squad_tables,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fbref" / "stats_sample.html"


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_extract_tables_unwraps_comments_and_ids(fixture_html):
    tables = extract_tables(fixture_html)
    assert "stats_squads_standard_for" in tables
    assert "stats_squads_standard_against" in tables
    # tabela de jogadores vinha comentada
    assert "stats_standard" in tables

    squads = tables["stats_squads_standard_for"]
    assert {r["time"] for r in squads} == {"Flamengo", "Palmeiras"}
    fla = next(r for r in squads if r["time"] == "Flamengo")
    assert fla["time_id"] == "639950ae"
    assert fla["gols"] == 14
    assert fla["posse"] == 56.7

    players = tables["stats_standard"]
    assert len(players) == 3
    bruno = next(r for r in players if r["jogador"] == "Bruno Henrique")
    assert bruno["jogador_id"] == "aa111aaa"
    assert bruno["nacao"] == "BRA"
    assert bruno["gols"] == 4
    assert bruno["finalizacoes_no_gol"] == 8


def test_split_annotates_against_suffix(fixture_html):
    tables = extract_tables(fixture_html)
    players, squads_for, squads_against, _matches = split_squad_tables(
        tables, temporada=2026, page="stats"
    )
    assert set(squads_for["time"]) == {"Flamengo", "Palmeiras"}
    assert "gols" in squads_for.columns
    from src.fbref_scraper import prefix_against

    against = prefix_against(squads_against)
    assert "gols_sofrido" in against.columns
    fla = against[against["time"] == "Flamengo"].iloc[0]
    assert fla["gols_sofrido"] == 4


def test_canonical_squad_uses_id_and_aliases():
    assert canonical_squad("Independiente", "990519b8") == "Independiente del Valle"
    assert canonical_squad("Estudiantes–LP", "df734df9") == "Estudiantes"
    assert canonical_squad("LDU Quito", None) == "LDU"
    assert canonical_squad("Ind. Rivadavia", None) == "Independiente Rivadavia"


def test_page_url_current_vs_historical():
    current = page_url("stats", 2026, current_season=2026)
    assert current.endswith("/comps/14/stats/Copa-Libertadores-Stats")
    old = page_url("shooting", 2025, current_season=2026)
    assert "/2025/shooting/2025-Copa-Libertadores-Stats" in old
    sched = page_url("schedule", 2024, current_season=2026)
    assert "/2024/schedule/2024-Copa-Libertadores-Scores-and-Fixtures" in sched


def test_client_parse_page_from_fixture(fixture_html):
    client = FBrefClient(sleep=0)
    parsed = client.parse_page(fixture_html, "stats", 2026)
    assert len(parsed["elencos"]) == 2
    assert len(parsed["jogadores"]) == 3
    assert parsed["jogadores"]["jogador_id"].notna().all()


def test_run_falls_back_to_snapshot(monkeypatch, tmp_path):
    client = FBrefClient(sleep=0, cache_dir=tmp_path / "cache")

    def boom(*_a, **_k):
        raise FBrefUnavailableError("ssl bloqueado")

    monkeypatch.setattr(client, "scrape", boom)
    data = client.run(season=2026, persist=False)
    assert len(data["elencos"]) == 32
    assert "Flamengo" in set(data["elencos"]["time"])
    assert data["elencos"]["fonte"].eq("fbref").all()


def test_run_without_snapshot_or_network_raises(monkeypatch, tmp_path):
    client = FBrefClient(sleep=0, cache_dir=tmp_path / "cache")
    monkeypatch.setattr(client, "scrape", lambda **_k: (_ for _ in ()).throw(FBrefUnavailableError("x")))
    monkeypatch.setattr(client, "load_snapshot", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.delenv("ALLOW_EXAMPLE_DATA", raising=False)
    with pytest.raises(FBrefUnavailableError):
        client.run(season=2099, persist=False)


def test_save_sqlite_roundtrip(tmp_path, fixture_html):
    client = FBrefClient(sleep=0)
    parsed = client.parse_page(fixture_html, "stats", 2026)
    db = tmp_path / "fbref.sqlite"
    save_sqlite(parsed["elencos"], parsed["jogadores"], parsed["partidas"], path=db)
    import sqlite3

    with sqlite3.connect(db) as conn:
        n_times = conn.execute("SELECT COUNT(*) FROM elencos").fetchone()[0]
        n_jogs = conn.execute("SELECT COUNT(*) FROM jogadores").fetchone()[0]
    assert n_times == 2
    assert n_jogs == 3


def test_indices_elencos_snapshot():
    elencos = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "historical" / "fbref" / "elencos_2026.csv"
    )
    idx = indices_elencos(elencos)
    assert len(idx) == 32
    assert idx["indice_forca_ofensiva"].notna().all()
    # Rivadavia lidera em gols (15) e G+A (26) — deve estar no topo ofensivo
    top = idx.sort_values("indice_forca_ofensiva", ascending=False).iloc[0]["time"]
    assert top in {"Independiente Rivadavia", "Flamengo", "Independiente del Valle", "Palmeiras"}
    # Flamengo sofreu 4 em 8 jogos e desarma bem — pressão não é zero
    fla = idx[idx["time"] == "Flamengo"].iloc[0]
    assert fla["indice_pressao_defensiva"] > 0
    assert fla["gols_sofridos_90"] == pytest.approx(0.5)


def test_indices_jogadores_recorte():
    df = pd.DataFrame([
        {"time": "Flamengo", "jogador": "Bruno Henrique", "posicao": "FW",
         "noventas": 6.0, "gols": 4, "assistencias": 2, "desarmes_ganhos": 2, "interceptacoes": 1},
        {"time": "Flamengo", "jogador": "Arrascaeta", "posicao": "MF",
         "noventas": 5.0, "gols": 1, "assistencias": 3, "desarmes_ganhos": 4, "interceptacoes": 2},
        {"time": "Flamengo", "jogador": "Léo Pereira", "posicao": "DF",
         "noventas": 8.0, "gols": 0, "assistencias": 0, "desarmes_ganhos": 12, "interceptacoes": 10},
        {"time": "Flamengo", "jogador": "Rossi", "posicao": "GK",
         "noventas": 8.0, "gols": 0, "assistencias": 0, "desarmes_ganhos": 0, "interceptacoes": 0},
    ])
    recorte = indices_jogadores(df)
    row = recorte.iloc[0]
    assert row["time"] == "Flamengo"
    assert row["criador"] == "Arrascaeta"
    assert row["indice_ataque_titulares"] > 0
    assert row["indice_defesa_titulares"] > 0


def test_confronto_indices():
    elencos = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "historical" / "fbref" / "elencos_2026.csv"
    )
    idx = indices_elencos(elencos)
    confronto = confronto_indices(idx, "Flamengo", "Platense")
    assert confronto.loc[0, "mandante"] == "Flamengo"
    assert "diff_indice_forca_ofensiva" in confronto.columns


def test_fetch_html_uses_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    html_path = cache / "2026_stats.html"
    html_path.write_text("<html><table id='x'></table></html>", encoding="utf-8")
    client = FBrefClient(sleep=0, cache_dir=cache)

    def fail_get(*_a, **_k):
        raise AssertionError("não deveria ir à rede com cache")

    client.session.get = fail_get  # type: ignore[method-assign]
    text = client.fetch_html("stats", 2026, use_cache=True)
    assert "table" in text
