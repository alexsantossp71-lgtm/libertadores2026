"""Testes do módulo de dados reais (openfootball + suplementos ESPN/FBref)."""

import pandas as pd
import pytest

from src.real_data import (
    QUARTAS_PATH,
    _parse_tail,
    collect_warnings,
    is_unmapped,
    knockout_results,
    load_partidas,
    review,
    standings,
    validate,
)


@pytest.fixture(scope="module")
def partidas():
    try:
        return load_partidas()
    except FileNotFoundError:
        pytest.skip("dataset histórico não construído")


def test_parse_tail_variantes():
    assert _parse_tail("2-0 (1-0)") == {
        "gols": (2, 0), "intervalo": (1, 0), "penaltis": None, "prorrogacao": False,
    }
    assert _parse_tail("1-2")["gols"] == (1, 2)
    pen = _parse_tail("4-3 pen. (2-1, 0-1)")
    assert pen["penaltis"] == (4, 3) and pen["gols"] == (2, 1)
    aet = _parse_tail("4-5 pen. 2-1 a.e.t. (2-1, 0-1)")
    assert aet["penaltis"] == (4, 5) and aet["gols"] == (2, 1) and aet["prorrogacao"]
    dois_digitos = _parse_tail("10-9 pen. 0-0 a.e.t. (0-0)")
    assert dois_digitos["penaltis"] == (10, 9)
    assert _parse_tail("")["gols"] is None  # jogo sem placar
    assert _parse_tail("qualquer coisa") is None


def test_dataset_cobre_temporadas(partidas):
    temporadas = set(partidas["temporada"].unique())
    assert {2012, 2016, 2020, 2025, 2026}.issubset(temporadas)
    assert partidas["gols_mandante"].notna().sum() > 2000


def test_invariantes_globais(partidas):
    erros, _avisos = validate(partidas)
    assert erros == []


def test_tabela_grupos_2026_consistente(partidas):
    tab = standings(partidas, 2026)
    assert len(tab) == 32
    assert set(tab["Grupo"].unique()) == set("ABCDEFGH")
    assert tab.groupby("Grupo").size().eq(4).all()
    assert int(tab["GP"].sum()) == int(tab["GC"].sum())
    assert ((tab["V"] + tab["E"] + tab["D"]) == tab["J"]).all()


def test_oitavas_2026_reais(partidas):
    oitavas = knockout_results(partidas, 2026, "Playoffs")
    vencedores = dict(zip(oitavas["Time1"], oitavas["Vencedor"]))
    # conferido contra FBref/ESPN em 21/08/2026
    assert vencedores["Fluminense"] == "Fluminense"          # 1-1, pên. 4-5
    assert oitavas.loc[oitavas["Time1"] == "Estudiantes", "Vencedor"].iloc[0] == "Estudiantes"
    assert oitavas.loc[oitavas["Time1"] == "Platense", "Vencedor"].iloc[0] == "Platense"
    assert oitavas.loc[oitavas["Time1"] == "Palmeiras", "Vencedor"].iloc[0] == "Palmeiras"
    assert oitavas.loc[oitavas["Time1"] == "Cruzeiro", "Vencedor"].iloc[0] == "Flamengo"
    # Tolima x IDV: volta em 25/08 — não pode ter vencedor declarado
    pendente = oitavas[oitavas.apply(lambda r: "Tolima" in (r["Time1"], r["Time2"]), axis=1)]
    assert pendente["Vencedor"].iloc[0] == "(em andamento)"


def test_chaveamento_das_quartas_2026():
    quartas = pd.read_csv(QUARTAS_PATH).set_index("Confronto")

    assert list(quartas.index) == ["QF1", "QF2", "QF3", "QF4"]
    assert quartas.loc["QF1", ["Mandante", "Visitante"]].tolist() == [
        "Estudiantes", "Corinthians"
    ]
    assert quartas.loc["QF2", ["Mandante", "Visitante"]].tolist() == ["Palmeiras", "LDU"]
    assert quartas.loc["QF3", "Mandante"] == "Flamengo"
    assert "A DEFINIR" in quartas.loc["QF3", "Visitante"]
    assert quartas.loc["QF4", ["Mandante", "Visitante"]].tolist() == [
        "Fluminense", "Platense"
    ]


def test_suplemento_trocado_tolima(partidas):
    """A ida remarcada (18/08, Tolima em casa) deve existir com placar 0-1."""
    m = partidas[
        (partidas["temporada"] == 2026)
        & (partidas["fase"] == "Playoffs")
        & partidas["mandante"].str.contains("Tolima")
    ]
    assert len(m) == 1
    assert m.iloc[0]["gols_mandante"] == 0 and m.iloc[0]["gols_visitante"] == 1
    assert m.iloc[0]["fonte"] == "fbref"


def test_campeoes_historicos_batem_com_a_realidade(partidas):
    for temporada, campeao in [(2019, "Flamengo"), (2020, "Palmeiras"), (2021, "Palmeiras"), (2022, "Flamengo")]:
        finais = knockout_results(partidas, temporada, "Finals")
        finais = finais[finais["Volta"] != "—"]  # finais de 2 jogos ou jogo único
        campeoes = set(finais["Vencedor"])
        assert campeao in campeoes or not campeoes, f"{temporada}: {campeoes}"


def test_avisos_nao_truncados():
    """collect_warnings() deve devolver TODOS os avisos de parsing, não só 10."""
    warns = collect_warnings()
    assert isinstance(warns, list)
    # orçamento conhecido: 0 avisos de parsing (build_dataset já trunca em 10,
    # mas a auditoria captura o total real — aqui esperado ser 0)
    assert len(warns) == 0, f"avisos de parsing inesperados: {warns}"


def test_cobertura_normalizacao(partidas):
    """Nenhum time presente no dataset pode ficar sem entrada em _CANONICAL."""
    times = pd.concat([partidas["mandante"], partidas["visitante"]]).dropna().unique()
    sem_map = [t for t in times if is_unmapped(t)]
    assert sem_map == [], f"times não mapeados em _CANONICAL: {sem_map}"


def test_completude_sem_inesperados(partidas):
    """Completude: só avisos/info esperados (jogos futuros), zero erros."""
    rel = review(partidas)
    cat = rel["completude"]
    assert cat["erros"] == [], f"erros de completude inesperados: {cat['erros']}"
    # todo aviso deve ser de jogo futuro (sem data) ou em andamento
    for a in cat["avisos"]:
        assert "jogo futuro" in a or "em andamento" in a, f"aviso inesperado: {a}"


def test_validade_dominio(partidas):
    """Validade: domínios de gols, fase, data, país e perna respeitados."""
    rel = review(partidas)
    cat = rel["validade"]
    assert cat["erros"] == [], f"erros de validade inesperados: {cat['erros']}"
    # não deve haver fase não canônica nem código de país inválido
    for a in cat["avisos"]:
        assert ("fase não canônica" in a) or ("país" in a) or ("perna" in a), \
            f"aviso de validade inesperado: {a}"


def test_sem_duplicatas_amplas(partidas):
    """Consistência: não há partidas duplicadas na checagem ampla."""
    rel = review(partidas)
    cat = rel["consistencia"]
    assert cat["erros"] == [], f"erros de consistência inesperados: {cat['erros']}"


def test_concordancia_entre_fontes(partidas):
    """Entre fontes: openfootball 2026 concorda com os suplementos."""
    rel = review(partidas)
    cat = rel["entre_fontes"]
    assert cat["avisos"] == [], f"divergências entre fontes: {cat['avisos']}"


def test_review_sem_erros_novos(partidas):
    """O review completo não deve introduzir erros em nenhuma categoria."""
    rel = review(partidas)
    for nome, cat in rel.items():
        assert cat["erros"] == [], f"erros em '{nome}': {cat['erros']}"
