"""Testes do cliente de odds da Bzzoiro (parsing e fallback offline)."""


import pytest

from src.odds_client import (
    BzzoiroOddsClient,
    _odds_from_payload,
    implied_probabilities,
    normalize_team_name,
)


def test_normalize_team_name():
    assert normalize_team_name("  Clube Atlético Mineiro  ") == "clube atletico mineiro"
    assert normalize_team_name("São Paulo FC") == "sao paulo fc"
    assert normalize_team_name(None) == ""


def test_implied_probabilities_sums_to_one():
    p_h, p_d, p_a, margem = implied_probabilities(2.0, 3.2, 4.5)
    assert p_h + p_d + p_a == pytest.approx(1.0)
    assert margem > 0  # overround positivo


def test_implied_probabilities_margin():
    # Sem margem (odds justas), overround ~ 0
    _, _, _, margem = implied_probabilities(2.0, 3.0, 6.0)
    assert margem == pytest.approx(0.0, abs=1e-9)


def test_odds_from_payload_consensus():
    payload = {
        "results": [
            {"outcome": "HOME", "decimal_odds": 2.10, "bookmaker_slug": "consensus"},
            {"outcome": "DRAW", "decimal_odds": 3.20, "bookmaker_slug": "consensus"},
            {"outcome": "AWAY", "decimal_odds": 4.00, "bookmaker_slug": "consensus"},
        ]
    }
    odds = _odds_from_payload(payload)
    assert odds["odd_mandante"] == 2.10
    assert odds["odd_empate"] == 3.20
    assert odds["odd_visitante"] == 4.00


def test_odds_from_payload_averages_bookmakers():
    payload = [
        {"outcome": "HOME", "decimal_odds": 2.00, "bookmaker_name": "Bet365"},
        {"outcome": "HOME", "decimal_odds": 2.40, "bookmaker_name": "Pinnacle"},
        {"outcome": "DRAW", "decimal_odds": 3.20, "bookmaker_name": "Bet365"},
        {"outcome": "AWAY", "decimal_odds": 4.00, "bookmaker_name": "Bet365"},
    ]
    odds = _odds_from_payload(payload)
    assert odds["odd_mandante"] == pytest.approx(2.20)
    assert "Bet365" in odds["bookmakers"] and "Pinnacle" in odds["bookmakers"]


def test_odds_from_payload_incomplete_returns_none():
    payload = [{"outcome": "HOME", "decimal_odds": 2.00}]
    assert _odds_from_payload(payload) is None
    assert _odds_from_payload([]) is None


def test_has_key_uses_environment(monkeypatch):
    monkeypatch.delenv("BSD_API", raising=False)
    assert not BzzoiroOddsClient(api_key=None).has_key
    monkeypatch.setenv("BSD_API", "chave-teste")
    assert BzzoiroOddsClient(api_key=None).has_key


def test_run_offline_uses_example_odds(monkeypatch, tmp_path):
    monkeypatch.delenv("BSD_API", raising=False)
    monkeypatch.setattr(
        "src.odds_client.PROCESSED_PATH", tmp_path / "libertadores_odds.csv"
    )
    client = BzzoiroOddsClient(api_key=None)
    df = client.run()
    required = {
        "odd_mandante", "odd_empate", "odd_visitante",
        "prob_mandante_impl", "prob_empate_impl", "prob_visitante_impl",
    }
    assert required.issubset(df.columns)
    assert len(df) > 0
    assert (tmp_path / "libertadores_odds.csv").exists()
