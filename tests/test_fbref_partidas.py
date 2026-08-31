"""Testes do parser de relatórios de partida FBref (src/fbref_partidas.py)."""

from pathlib import Path


from src.fbref_partidas import (
    parse_match_report,
    parse_schedule_html,
    build_rows,
    CSV_COLUMNS,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "fbref" / "match_report_sample.html"


def test_parse_match_report_basic_fields():
    html = FIXTURE.read_text(encoding="utf-8")
    row = parse_match_report(html, match_id="3e3678ab")

    assert row["mandante"] == "Deportivo La Guaira"
    assert row["visitante"] == "Fluminense"
    assert row["mandante_id"] == "0f8ef17f"
    assert row["visitante_id"] == "84d9701c"
    assert row["gols_mandante"] == 0
    assert row["gols_visitante"] == 0
    assert row["resultado"] == "empate"
    assert row["arbitro"] == "José Javier Burgos"
    assert row["data"] == "2026-04-07"


def test_parse_match_report_team_stats():
    html = FIXTURE.read_text(encoding="utf-8")
    row = parse_match_report(html, match_id="3e3678ab")

    assert row["posse_mandante"] == 36.0
    assert row["posse_visitante"] == 64.0
    # Shots on Target row: home "1 of 6", away "7 of 18"
    assert row["finalizacoes_no_gol_mandante"] == 1
    assert row["finalizacoes_mandante"] == 6
    assert row["finalizacoes_no_gol_visitante"] == 7
    assert row["finalizacoes_visitante"] == 18
    assert row["finalizacoes_fora_mandante"] == 5
    assert row["finalizacoes_fora_visitante"] == 11
    # Saves row: home "7 of 7", away "1 of 1"
    assert row["defesas_mandante"] == 7
    assert row["defesas_visitante"] == 1


def test_parse_match_report_cards_and_extra():
    html = FIXTURE.read_text(encoding="utf-8")
    row = parse_match_report(html, match_id="3e3678ab")

    # Home: 1 yellow + 1 yellow-red (conta como vermelho); Away: 1 yellow
    assert row["cartoes_amarelos_mandante"] == 1
    assert row["cartoes_vermelhos_mandante"] == 1
    assert row["cartoes_amarelos_visitante"] == 1
    assert row["cartoes_vermelhos_visitante"] == 0
    # team_stats_extra
    assert row["faltas_mandante"] == 10
    assert row["faltas_visitante"] == 9
    assert row["escanteios_mandante"] == 3
    assert row["escanteios_visitante"] == 8
    assert row["impedimentos_mandante"] == 5
    assert row["impedimentos_visitante"] == 0


SCHEDULE_SNIPPET = """
<table id="sched_2026_14_1">
 <thead><tr><th data-stat="round"></th><th data-stat="gameweek"></th>
 <th data-stat="date"></th><th data-stat="home_team"></th><th data-stat="score"></th>
 <th data-stat="away_team"></th><th data-stat="referee"></th><th data-stat="match_report"></th></tr></thead>
 <tbody>
  <tr>
   <td data-stat="round">Group stage</td>
   <td data-stat="gameweek">1</td>
   <td data-stat="date">2026-04-07</td>
   <td data-stat="home_team"><a href="/en/squads/0f8ef17f/Deportivo-La-Guaira-Stats">Dep. La Guaira</a></td>
   <td data-stat="score"><a href="/en/matches/3e3678ab/Deportivo-La-Guaira-Fluminense-April-7-2026">0–0</a></td>
   <td data-stat="away_team"><a href="/en/squads/84d9701c/Fluminense-Stats">Fluminense</a></td>
   <td data-stat="referee">José Javier Burgos</td>
   <td data-stat="match_report"><a href="/en/matches/3e3678ab/Deportivo-La-Guaira-Fluminense-April-7-2026">Match Report</a></td>
  </tr>
 </tbody>
</table>
"""


def test_parse_schedule_html():
    sched = parse_schedule_html(SCHEDULE_SNIPPET)
    assert "3e3678ab" in sched
    row = sched["3e3678ab"]
    assert row["data"] == "2026-04-07"
    assert row["fase_fbref"] == "Group stage"
    assert row["gameweek"] == 1
    assert row["arbitro_schedule"] == "José Javier Burgos"
    assert row["mandante_id_sched"] == "0f8ef17f"
    assert row["visitante_id_sched"] == "84d9701c"
    assert row["gols_mandante_sched"] == 0


def test_build_rows_schema_and_fonte(tmp_path):
    html = FIXTURE.read_text(encoding="utf-8")
    report = parse_match_report(html, match_id="3e3678ab")
    df = build_rows([report], schedule=None)
    assert list(df.columns) == CSV_COLUMNS
    assert df.loc[0, "fonte"] == "fbref"
    # campos que a FBref não publica para esta comp ficam como NA
    assert df["passes_certos_mandante"].isna().all()
    assert df["arbitro_pais"].isna().all()
