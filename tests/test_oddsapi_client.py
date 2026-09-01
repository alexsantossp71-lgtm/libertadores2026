"""Testes do cliente de odds da The Odds API (src/oddsapi_client.py)."""

from __future__ import annotations

import pytest

from src.oddsapi_client import (
    event_to_row,
    map_team_name,
    parse_h2h_event,
)


@pytest.fixture()
def h2h_event() -> dict:
    """Evento h2h com duas casas de apostas (payload típico da v4)."""
    return {
        "id": "abc123",
        "sport_key": "soccer_conmebol_copa_libertadores",
        "commence_time": "2026-09-09T22:00:00Z",
        "home_team": "Palmeiras-SP",
        "away_team": "LDU Quito",
        "bookmakers": [
            {
                "key": "bet365",
                "title": "Bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Palmeiras-SP", "price": 1.90},
                            {"name": "LDU Quito", "price": 4.20},
                            {"name": "Draw", "price": 3.30},
                        ],
                    }
                ],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Palmeiras-SP", "price": 2.10},
                            {"name": "LDU Quito", "price": 4.00},
                            {"name": "Draw", "price": 3.50},
                        ],
                    }
                ],
            },
        ],
    }


def test_parse_h2h_event_media_das_casas(h2h_event):
    parsed = parse_h2h_event(h2h_event)
    assert parsed is not None
    # média simples das duas casas
    assert parsed["odd_mandante"] == pytest.approx(2.00)
    assert parsed["odd_empate"] == pytest.approx(3.40)
    assert parsed["odd_visitante"] == pytest.approx(4.10)
    assert parsed["n_bookmakers"] == 2


def test_parse_h2h_event_sem_bookmakers():
    evento = {
        "home_team": "Flamengo-RJ",
        "away_team": "Independiente del Valle",
        "bookmakers": [],
    }
    assert parse_h2h_event(evento) is None


def test_parse_h2h_event_sem_draw():
    evento = {
        "home_team": "Platense",
        "away_team": "Fluminense-RJ",
        "bookmakers": [
            {
                "key": "bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Platense", "price": 5.0},
                            {"name": "Fluminense-RJ", "price": 1.7},
                        ],
                    }
                ],
            }
        ],
    }
    assert parse_h2h_event(evento) is None


def test_event_to_row_mapeia_nomes_e_fonte(h2h_event):
    row = event_to_row(h2h_event, partida_id=9001, data="2026-09-09",
                       fase="Quartas de Final", rodada="Ida")
    assert row["mandante"] == "Palmeiras"
    assert row["visitante"] == "LDU"
    assert row["fonte"] == "the-odds-api"
    assert row["bookmaker"] == "The Odds API (média 2 casas)"
    # probabilidades implícitas somam 1
    soma = (
        row["prob_mandante_impl"] + row["prob_empate_impl"] + row["prob_visitante_impl"]
    )
    assert soma == pytest.approx(1.0, abs=1e-4)
    assert row["margem"] > 0
    # partida futura: sem gols/resultado
    assert row["gols_mandante"] in ("", None) or row["gols_mandante"] != row["gols_mandante"]
    assert row["resultado"] in ("", None)


def test_map_team_name_sufixos_e_variantes():
    assert map_team_name("Fluminense-RJ") == "Fluminense"
    assert map_team_name("Corinthians-SP") == "Corinthians"
    assert map_team_name("Independiente del Valle") == "Independiente del Valle"
    assert map_team_name("Platense") == "Platense"
    # desconhecido: retorna o nome original
    assert map_team_name("Time Desconhecido FC") == "Time Desconhecido FC"
