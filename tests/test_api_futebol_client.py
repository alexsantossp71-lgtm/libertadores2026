"""Testes do cliente da API Futebol (parsing e fallback offline)."""

import json
import os

import pandas as pd
import pytest

from api_futebol_client import (
    ApiFutebolClient,
    _alias_stats,
    _extract_arbitro_nome,
    _extract_arbitro_pais,
    _num,
    _parse_placar,
    _side_stats,
)


def test_num_parses_percent_and_locale_numbers():
    assert _num("55%") == 55.0
    assert _num("1.234") == 1.234
    assert _num("1,5") == 1.5
    assert _num(None) is None
    assert _num(42) == 42.0


def test_alias_stats_maps_known_keys():
    raw = {"Posse de bola": "58%", "faltas": 12, "Passes Certos": 400}
    out = _alias_stats(raw)
    assert out["posse"] == "58%"
    assert out["faltas"] == 12
    assert out["passes_certos"] == 400


def test_side_stats_dict_form():
    stats = {
        "mandante": {"posse_de_bola": "60%", "faltas": 10},
        "visitante": {"posse_de_bola": "40%", "faltas": 14},
    }
    assert _side_stats(stats, "mandante")["posse"] == "60%"
    assert _side_stats(stats, "visitante")["faltas"] == 14


def test_side_stats_list_form():
    stats = [
        {"nome": "posse_de_bola", "valor": "55%", "time": "mandante"},
        {"nome": "faltas", "valor": 8, "time": "visitante"},
    ]
    mandante = _side_stats(stats, "mandante")
    visitante = _side_stats(stats, "visitante")
    assert mandante["posse"] == "55%"
    assert visitante["faltas"] == 8


def test_parse_placar_numeric_and_string():
    assert _parse_placar({"placar_mandante": 2, "placar_visitante": 1}) == (2, 1)
    assert _parse_placar({"placar": "Atlético-MG (4)2x2(3) Palmeiras"}) == (2, 2)
    assert _parse_placar({}) == (None, None)


def test_arbitro_extraction():
    assert _extract_arbitro_nome("Wilmar Roldán") == "Wilmar Roldán"
    assert _extract_arbitro_nome({"nome": "Raphael Claus", "pais": "Brasil"}) == "Raphael Claus"
    assert _extract_arbitro_nome(None) is None
    assert _extract_arbitro_pais({"nacionalidade": "Uruguai"}) == "Uruguai"


def test_parse_partida_full():
    detail = {
        "partida_id": 999,
        "fase": {"nome": "Fase de Grupos"},
        "rodada": {"nome": "1ª Rodada"},
        "grupo": "Grupo A",
        "time_mandante": {"nome_popular": "Flamengo"},
        "time_visitante": {"nome_popular": "Cruzeiro"},
        "placar_mandante": 2,
        "placar_visitante": 1,
        "data_realizacao_iso": "2026-04-14T21:00:00-03:00",
        "arbitro": {"nome": "Facundo Tello", "nacionalidade": "Argentina"},
        "estatisticas": {
            "mandante": {
                "posse_de_bola": "58%", "faltas": 12, "escanteios": 5,
                "passes_certos": 420, "passes_errados": 80,
                "finalizacao": 14, "finalizacao_no_gol": 6,
                "finalizacao_fora": 8, "impedimentos": 2, "defesas": 3,
            },
            "visitante": {
                "posse_de_bola": "42%", "faltas": 16, "escanteios": 3,
                "passes_certos": 310, "passes_errados": 95,
                "finalizacao": 9, "finalizacao_no_gol": 4,
                "finalizacao_fora": 5, "impedimentos": 1, "defesas": 4,
            },
        },
        "cartoes": {
            "amarelo": {"mandante": [{}, {}], "visitante": [{}]},
            "vermelho": {"mandante": [], "visitante": [{}]},
        },
    }
    client = ApiFutebolClient(api_key=None)
    row = client.parse_partida(detail)

    assert row["mandante"] == "Flamengo"
    assert row["gols_mandante"] == 2
    assert row["resultado"] == "mandante"
    assert row["arbitro"] == "Facundo Tello"
    assert row["arbitro_pais"] == "Argentina"
    assert row["posse_mandante"] == 58.0
    assert row["faltas_visitante"] == 16.0
    assert row["cartoes_amarelos_mandante"] == 2
    assert row["cartoes_vermelhos_visitante"] == 1
    assert row["passes_certos_mandante"] == 420.0
    assert row["finalizacoes_visitante"] == 9.0
    assert row["fonte"] == "api-futebol"


def test_parse_partida_without_stats_gives_nan():
    detail = {
        "partida_id": 1,
        "time_mandante": {"nome_popular": "Time A"},
        "time_visitante": {"nome_popular": "Time B"},
        "placar_mandante": 0,
        "placar_visitante": 0,
    }
    client = ApiFutebolClient(api_key=None)
    row = client.parse_partida(detail)
    assert row["faltas_mandante"] is None
    assert row["arbitro"] is None
    assert row["resultado"] == "empate"


def test_has_key_uses_environment(monkeypatch):
    monkeypatch.delenv("API_FUTEBOL_KEY", raising=False)
    assert not ApiFutebolClient(api_key=None).has_key
    monkeypatch.setenv("API_FUTEBOL_KEY", "chave-teste")
    assert ApiFutebolClient(api_key=None).has_key


def test_run_offline_uses_example_data(monkeypatch, tmp_path):
    monkeypatch.delenv("API_FUTEBOL_KEY", raising=False)
    monkeypatch.setattr(
        "api_futebol_client.PROCESSED_PATH", tmp_path / "estatisticas.csv"
    )
    client = ApiFutebolClient(api_key=None)
    df = client.run()
    assert len(df) > 0
    assert {"mandante", "visitante", "arbitro", "faltas_mandante"}.issubset(df.columns)
    assert (tmp_path / "estatisticas.csv").exists()
