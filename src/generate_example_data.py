"""
Gerador de dados de exemplo (determinístico) para o projeto Libertadores 2026.

Este módulo cria duas bases de exemplo usadas quando as APIs reais não estão
disponíveis (sem chave, sem internet ou erro de rate limit):

  * ``data/examples/partidas_libertadores_2026.csv``
      Partidas da fase de grupos e oitavas com estatísticas detalhadas
      (faltas, cartões, posse, passes, finalizações, escanteios, impedimentos,
      defesas) e o árbitro responsável.
  * ``data/examples/odds_libertadores_2026.csv``
      Odds decimais 1X2 (estilo consenso multi-bookmaker) com as
      probabilidades implícitas derivadas e o resultado de cada partida.

Os dados são gerados de forma **determinística** (semente fixa) e coerente com
o modelo de Poisson do projeto (``src/poisson.py``): as partidas são simuladas
com os mesmos lambdas que o modelo estimaria a partir da tabela da fase de
grupos, e as odds são geradas a partir dessas probabilidades com uma margem de
casa de apostas e ruído de mercado.

Uso::

    python src/generate_example_data.py

Os arquivos são salvos em ``data/examples/`` (versionados no Git) e servem de
fallback para o pipeline, para o dashboard e para o notebook de análise.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

# --------------------------------------------------------------------------- #
# Caminhos e constantes
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT_DIR / "data" / "examples"
EXAMPLE_PARTIDAS_PATH = EXAMPLES_DIR / "partidas_libertadores_2026.csv"
EXAMPLE_ODDS_PATH = EXAMPLES_DIR / "odds_libertadores_2026.csv"

SEED = 2026
MAX_GOALS = 10
HOME_ADVANTAGE = 1.15
AWAY_FACTOR = 2.0 - HOME_ADVANTAGE

# Esquema canônico das partidas (usado pelos clientes e pelo pré-processamento)
PARTIDAS_COLUMNS = [
    "partida_id", "data", "fase", "rodada", "grupo",
    "mandante", "visitante", "gols_mandante", "gols_visitante", "resultado",
    "arbitro", "arbitro_pais",
    "faltas_mandante", "faltas_visitante",
    "cartoes_amarelos_mandante", "cartoes_amarelos_visitante",
    "cartoes_vermelhos_mandante", "cartoes_vermelhos_visitante",
    "posse_mandante", "posse_visitante",
    "passes_certos_mandante", "passes_certos_visitante",
    "passes_errados_mandante", "passes_errados_visitante",
    "finalizacoes_mandante", "finalizacoes_visitante",
    "finalizacoes_no_gol_mandante", "finalizacoes_no_gol_visitante",
    "finalizacoes_fora_mandante", "finalizacoes_fora_visitante",
    "escanteios_mandante", "escanteios_visitante",
    "impedimentos_mandante", "impedimentos_visitante",
    "defesas_mandante", "defesas_visitante",
    "fonte",
]

ODDS_COLUMNS = [
    "partida_id", "data", "fase", "rodada",
    "mandante", "visitante",
    "odd_mandante", "odd_empate", "odd_visitante",
    "prob_mandante_impl", "prob_empate_impl", "prob_visitante_impl",
    "margem", "bookmaker",
    "gols_mandante", "gols_visitante", "resultado",
    "fonte",
]

# --------------------------------------------------------------------------- #
# Times (tabela da fase de grupos — mesma base do scraper de exemplo)
# --------------------------------------------------------------------------- #
# Time, País, Pts, J, V, E, D, GP, GC
TIMES_GRUPOS: List[Tuple[str, str, int, int, int, int, int, int, int]] = [
    ("Flamengo", "BRA", 16, 6, 5, 1, 0, 14, 2),
    ("Estudiantes", "ARG", 9, 6, 2, 3, 1, 6, 5),
    ("Coquimbo Unido", "CHI", 10, 6, 3, 1, 2, 8, 6),
    ("Deportes Tolima", "COL", 8, 6, 2, 2, 2, 7, 6),
    ("Independiente Rivadavia", "ARG", 12, 6, 4, 0, 2, 15, 8),
    ("Palmeiras", "BRA", 13, 6, 4, 1, 1, 12, 5),
    ("Corinthians", "BRA", 11, 6, 3, 2, 1, 9, 5),
    ("Fluminense", "BRA", 14, 6, 4, 2, 0, 10, 4),
    ("LDU", "ECU", 11, 6, 3, 2, 1, 7, 5),
    ("Platense", "ARG", 10, 6, 3, 1, 2, 8, 6),
    ("Independiente del Valle", "ECU", 9, 6, 2, 3, 1, 6, 5),
    ("Cruzeiro", "BRA", 10, 6, 3, 1, 2, 9, 7),
]

GRUPOS: Dict[str, List[str]] = {
    "Grupo A": [
        "Flamengo", "Palmeiras", "Estudiantes",
        "Coquimbo Unido", "Deportes Tolima", "Independiente Rivadavia",
    ],
    "Grupo B": [
        "Fluminense", "Corinthians", "Cruzeiro",
        "LDU", "Platense", "Independiente del Valle",
    ],
}

# Oitavas: (mandante ida, visitante ida, gols ida, gols volta)
OITAVAS_FIXTURES: List[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = [
    ("Flamengo", "Cruzeiro", (2, 1), (1, 1)),
    ("Palmeiras", "Coquimbo Unido", (3, 0), (2, 1)),
    ("Corinthians", "Independiente Rivadavia", (1, 0), (2, 1)),
    ("Fluminense", "Deportes Tolima", (2, 1), (1, 0)),
    ("Estudiantes", "Platense", (1, 0), (2, 1)),
    ("LDU", "Independiente del Valle", (2, 1), (1, 0)),
]

# --------------------------------------------------------------------------- #
# Árbitros de exemplo — (nome, país, faltas base por jogo, fator de cartões,
# efeito sobre o total de gols). Efeito negativo = árbitro rigoroso.
#
# NOTA: os efeitos sobre gols são deliberadamente amplificados (em relação ao
# que se observaria em dados reais) para que as análises estatísticas do
# notebook/dashboard ilustrem o padrão "árbitro permissivo → mais gols" com
# significância em uma amostra pequena. Em dados reais, o efeito é mais sutil
# e exige amostras maiores para ser detectado.
# --------------------------------------------------------------------------- #
REFEREES: List[Tuple[str, str, float, float, float]] = [
    ("Wilmar Roldán", "Colômbia", 13.5, 0.82, 1.20),
    ("Raphael Claus", "Brasil", 17.0, 1.00, 0.16),
    ("Facundo Tello", "Argentina", 24.5, 1.30, -1.10),
    ("Esteban Ostojich", "Uruguai", 19.0, 1.05, -0.12),
    ("Piero Maza", "Chile", 23.0, 1.22, -1.20),
    ("Andrés Matonte", "Uruguai", 18.0, 1.00, 0.20),
    ("Kevin Ortega", "Peru", 15.0, 0.88, 0.90),
    ("Jesús Valenzuela", "Venezuela", 21.0, 1.12, -0.60),
]

# Tendência de faltas por time (agressividade) — exemplo ilustrativo
FALTAS_TIME: Dict[str, float] = {
    "Flamengo": -1.5, "Estudiantes": 1.5, "Coquimbo Unido": 0.5,
    "Deportes Tolima": 2.0, "Independiente Rivadavia": 1.0, "Palmeiras": -0.5,
    "Corinthians": 2.5, "Fluminense": -1.0, "LDU": 0.0,
    "Platense": 1.5, "Independiente del Valle": -0.5, "Cruzeiro": 0.5,
}

RODADAS_GRUPOS: List[str] = [
    "1ª Rodada", "2ª Rodada", "3ª Rodada",
    "4ª Rodada", "5ª Rodada", "6ª Rodada",
]

# Datas das rodadas (exemplo)
RODADAS_DATAS: List[Tuple[str, str]] = [
    ("2026-04-14", "2026-04-15"),
    ("2026-04-21", "2026-04-22"),
    ("2026-04-28", "2026-04-29"),
    ("2026-05-05", "2026-05-06"),
    ("2026-05-12", "2026-05-13"),
    ("2026-05-19", "2026-05-20"),
]


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _time_stats(time: str) -> Tuple[float, float]:
    """Retorna (ataque, defesa) = (GP/J, GC/J) de um time."""
    for row in TIMES_GRUPOS:
        if row[0] == time:
            _, _, _, j, _, _, _, gp, gc = row
            return gp / j, gc / j
    raise KeyError(f"Time não encontrado na tabela: {time}")


def _league_avg() -> float:
    total_gp = sum(row[7] for row in TIMES_GRUPOS)
    total_j = sum(row[3] for row in TIMES_GRUPOS)
    return total_gp / total_j


def _expected_goals(home: str, away: str) -> Tuple[float, float]:
    """Mesma fórmula do PoissonScoreModel.expected_goals."""
    attack_h, defense_h = _time_stats(home)
    attack_a, defense_a = _time_stats(away)
    league_avg = _league_avg()
    lam_home = attack_h * defense_a * HOME_ADVANTAGE / league_avg
    lam_away = attack_a * defense_h * AWAY_FACTOR / league_avg
    return lam_home, lam_away


def _match_probs(home: str, away: str) -> Tuple[float, float, float]:
    """Probabilidades 1X2 (truncadas em MAX_GOALS), como no modelo."""
    lam_home, lam_away = _expected_goals(home, away)
    goals = np.arange(MAX_GOALS + 1)
    matrix = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    p_home = float(np.tril(matrix, k=-1).sum())
    p_away = float(np.triu(matrix, k=1).sum())
    p_draw = float(np.trace(matrix))
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def _round_robin_6(teams: List[str], rng: np.random.Generator) -> List[Tuple[int, str, str]]:
    """
    Gera 6 rodadas de confrontos para um grupo de 6 times (método do círculo,
    5 rodadas + rodada espelhada), cada time com 6 jogos. Retorna
    (rodada_index, mandante, visitante).
    """
    fixtures: List[Tuple[int, str, str]] = []
    n = len(teams)
    rotation = teams[:-1]
    fixed = teams[-1]

    for r in range(n - 1):  # 5 rodadas
        pairs: List[Tuple[str, str]] = []
        for i in range(n // 2):
            if i == 0:
                # o time fixo joga contra o primeiro da rotação
                left, right = rotation[0], fixed
            else:
                left, right = rotation[i], rotation[len(rotation) - i]
            # alternância de mando
            if (r + i) % 2 == 0:
                pairs.append((left, right))
            else:
                pairs.append((right, left))
        for home, away in pairs:
            fixtures.append((r, home, away))
        rotation = [rotation[-1]] + rotation[:-1]

    # 6ª rodada: espelho da 1ª (inverte mando)
    first_round = [f for f in fixtures if f[0] == 0]
    for _, home, away in first_round:
        fixtures.append((5, away, home))

    # Embaralha a ordem dentro da rodada (determinístico)
    ordered: List[Tuple[int, str, str]] = []
    for r in range(6):
        round_fixtures = [f for f in fixtures if f[0] == r]
        idx = list(range(len(round_fixtures)))
        rng.shuffle(idx)
        ordered.extend([round_fixtures[i] for i in idx])
    return ordered


# --------------------------------------------------------------------------- #
# Simulação das partidas
# --------------------------------------------------------------------------- #
def _simulate_match_stats(
    home: str,
    away: str,
    gols_home: int,
    gols_away: int,
    referee: Tuple[str, str, float, float, float],
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Gera estatísticas plausíveis e coerentes para uma partida."""
    _, _, faltas_base, fator_cartoes, _ = referee

    # Faltas: estilo do árbitro + tendência dos times + ruído
    faltas_h = int(np.clip(
        round(rng.normal(faltas_base / 2 + FALTAS_TIME[home], 2.2)), 4, 18))
    faltas_a = int(np.clip(
        round(rng.normal(faltas_base / 2 + FALTAS_TIME[away], 2.2)), 4, 18))

    # Cartões: correlacionados com as faltas e com o rigor do árbitro
    amarelos_h = int(np.clip(rng.poisson(max(0.3, faltas_h * 0.11 * fator_cartoes)), 0, 6))
    amarelos_a = int(np.clip(rng.poisson(max(0.3, faltas_a * 0.11 * fator_cartoes)), 0, 6))
    vermelhos_h = int(rng.random() < 0.030 * fator_cartoes)
    vermelhos_a = int(rng.random() < 0.030 * fator_cartoes)

    # Posse de bola (mandante em geral tem leve vantagem)
    posse_h = float(np.clip(rng.normal(54, 7), 38, 68))
    posse_a = 100.0 - posse_h

    # Passes (total ~ proporcional à posse)
    total_passes = float(np.clip(rng.normal(830, 80), 560, 1060))
    passes_h = total_passes * posse_h / 100.0
    passes_a = total_passes - passes_h
    pct_certos = float(np.clip(rng.normal(0.82, 0.025), 0.74, 0.90))
    certos_h = int(round(passes_h * pct_certos))
    certos_a = int(round(passes_a * pct_certos))
    errados_h = int(round(passes_h)) - certos_h
    errados_a = int(round(passes_a)) - certos_a

    # Finalizações: no gol >= gols marcados (coerência)
    no_gol_h = gols_home + int(rng.poisson(2.2))
    no_gol_a = gols_away + int(rng.poisson(2.2))
    fora_h = int(rng.poisson(3.1))
    fora_a = int(rng.poisson(3.1))
    finalizacoes_h = no_gol_h + fora_h
    finalizacoes_a = no_gol_a + fora_a

    # Defesas do goleiro = finalizações no gol do adversário - gols sofridos
    defesas_h = max(0, no_gol_a - gols_away)
    defesas_a = max(0, no_gol_h - gols_home)

    escanteios_h = int(rng.poisson(4.6))
    escanteios_a = int(rng.poisson(4.1))
    impedimentos_h = int(rng.poisson(1.3))
    impedimentos_a = int(rng.poisson(1.1))

    return {
        "faltas_mandante": faltas_h, "faltas_visitante": faltas_a,
        "cartoes_amarelos_mandante": amarelos_h,
        "cartoes_amarelos_visitante": amarelos_a,
        "cartoes_vermelhos_mandante": vermelhos_h,
        "cartoes_vermelhos_visitante": vermelhos_a,
        "posse_mandante": round(posse_h, 1), "posse_visitante": round(posse_a, 1),
        "passes_certos_mandante": certos_h, "passes_certos_visitante": certos_a,
        "passes_errados_mandante": errados_h,
        "passes_errados_visitante": errados_a,
        "finalizacoes_mandante": finalizacoes_h,
        "finalizacoes_visitante": finalizacoes_a,
        "finalizacoes_no_gol_mandante": no_gol_h,
        "finalizacoes_no_gol_visitante": no_gol_a,
        "finalizacoes_fora_mandante": fora_h,
        "finalizacoes_fora_visitante": fora_a,
        "escanteios_mandante": escanteios_h,
        "escanteios_visitante": escanteios_a,
        "impedimentos_mandante": impedimentos_h,
        "impedimentos_visitante": impedimentos_a,
        "defesas_mandante": defesas_h,
        "defesas_visitante": defesas_a,
    }


def _resultado(gols_home: int, gols_away: int) -> str:
    if gols_home > gols_away:
        return "mandante"
    if gols_home < gols_away:
        return "visitante"
    return "empate"


def generate_partidas(rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Gera a base de partidas de exemplo (fase de grupos + oitavas)."""
    rng = rng or np.random.default_rng(SEED)

    rows: List[Dict[str, object]] = []
    partida_id = 1001

    # ---- Fase de grupos ----
    for grupo, teams in GRUPOS.items():
        fixtures = _round_robin_6(teams, rng)
        for rodada_idx, home, away in fixtures:
            lam_h, lam_a = _expected_goals(home, away)
            referee = REFEREES[partida_id % len(REFEREES)]
            efeito = referee[4]
            # Árbitros permissivos (efeito > 0) -> mais gols; rigorosos -> menos
            gols_h = int(rng.poisson(max(0.1, lam_h + efeito / 2)))
            gols_a = int(rng.poisson(max(0.1, lam_a + efeito / 2)))
            stats = _simulate_match_stats(home, away, gols_h, gols_a, referee, rng)

            dia = RODADAS_DATAS[rodada_idx][partida_id % 2]
            rows.append({
                "partida_id": partida_id,
                "data": dia,
                "fase": "Fase de Grupos",
                "rodada": RODADAS_GRUPOS[rodada_idx],
                "grupo": grupo,
                "mandante": home,
                "visitante": away,
                "gols_mandante": gols_h,
                "gols_visitante": gols_a,
                "resultado": _resultado(gols_h, gols_a),
                "arbitro": referee[0],
                "arbitro_pais": referee[1],
                **stats,
                "fonte": "exemplo",
            })
            partida_id += 1

    # ---- Oitavas de final ----
    for idx, (home, away, placar_ida, placar_volta) in enumerate(OITAVAS_FIXTURES):
        for leg, (gols_time1, gols_time2) in enumerate([placar_ida, placar_volta]):
            referee = REFEREES[(partida_id + idx) % len(REFEREES)]
            if leg == 0:
                mandante, visitante = home, away
                g_h, g_a = gols_time1, gols_time2
                data = "2026-07-28" if idx % 2 == 0 else "2026-07-29"
            else:
                # Na volta, o mando se inverte (time2 em casa)
                mandante, visitante = away, home
                g_h, g_a = gols_time2, gols_time1
                data = "2026-08-04" if idx % 2 == 0 else "2026-08-05"

            stats = _simulate_match_stats(
                mandante, visitante, g_h, g_a, referee, rng)

            rows.append({
                "partida_id": partida_id,
                "data": data,
                "fase": "Oitavas de Final",
                "rodada": "Ida" if leg == 0 else "Volta",
                "grupo": "",
                "mandante": mandante,
                "visitante": visitante,
                "gols_mandante": g_h,
                "gols_visitante": g_a,
                "resultado": _resultado(g_h, g_a),
                "arbitro": referee[0],
                "arbitro_pais": referee[1],
                **stats,
                "fonte": "exemplo",
            })
            partida_id += 1

    df = pd.DataFrame(rows)
    return df[PARTIDAS_COLUMNS]


# --------------------------------------------------------------------------- #
# Odds de exemplo
# --------------------------------------------------------------------------- #
def _implied_probabilities(
    odd_home: float, odd_draw: float, odd_away: float
) -> Tuple[float, float, float, float]:
    """Probabilidades implícitas (1/odd) normalizadas; retorna também a margem."""
    raw = [1.0 / odd_home, 1.0 / odd_draw, 1.0 / odd_away]
    margem = sum(raw) - 1.0
    norm = [p / sum(raw) for p in raw]
    return norm[0], norm[1], norm[2], margem


def generate_odds(
    partidas: pd.DataFrame, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Gera odds 1X2 de exemplo coerentes com as probabilidades do modelo.

    O mercado simulado parte das probabilidades do modelo (Poisson) com ruído
    e três componentes adicionais:

      * **viés de favorito** (longshot-favourite bias): o mercado reforça o
        favorito em todos os jogos;
      * **informação extra** (~30% dos jogos): o mercado "sabe de algo"
        (lesões, escalações, viagens) e ajusta as probabilidades na direção do
        resultado real;
      * **armadilhas de overreaction** (~15% dos jogos): o mercado infla o
        favorito do modelo mesmo quando ele não vence (comportamento de
        público).

    Isso reproduz, de forma controlada, a assimetria de informação entre um
    modelo puramente estatístico e o mercado de apostas — e permite demonstrar
    por que a combinação modelo + odds supera as duas fontes isoladas.
    """
    rng = rng or np.random.default_rng(SEED + 1)

    idx_resultado = {"mandante": 0, "empate": 1, "visitante": 2}
    rows: List[Dict[str, object]] = []
    for _, match in partidas.iterrows():
        home, away = match["mandante"], match["visitante"]
        p_home, p_draw, p_away = _match_probs(home, away)

        # Viés de favorito + ruído de mercado
        market = np.array([p_home, p_draw, p_away])
        favorito = int(np.argmax(market))
        market[favorito] += 0.04
        market[[k for k in range(3) if k != favorito]] -= 0.02
        market = np.clip(market + rng.normal(0.0, 0.04, 3), 0.02, None)

        # Informação extra / armadilha de overreaction
        sorteio = rng.random()
        if sorteio < 0.30:
            shift = rng.uniform(0.14, 0.24)
            j = idx_resultado[match["resultado"]]
            market[j] += shift
            market[[k for k in range(3) if k != j]] -= shift / 2
            market = np.clip(market, 0.02, None)
        elif sorteio < 0.45:
            shift = rng.uniform(0.14, 0.24)
            j = int(np.argmax(market))
            market[j] += shift
            market[[k for k in range(3) if k != j]] -= shift / 2
            market = np.clip(market, 0.02, None)

        market = market / market.sum()

        # Margem da casa de apostas (overround) e arredondamento das odds
        # (casas de apostas nunca cotam abaixo de 1.01)
        margem = float(rng.uniform(0.045, 0.075))
        odds = [1.0 / (p * (1.0 + margem)) for p in market]
        odds = [max(1.01, float(np.round(o * 100) / 100)) for o in odds]

        prob_h_impl, prob_d_impl, prob_a_impl, margem_final = _implied_probabilities(
            *odds
        )

        bookmaker = rng.choice(
            ["Consenso Bzzoiro", "Consenso Bzzoiro", "Consenso Bzzoiro", "Bet365"],
        )

        rows.append({
            "partida_id": int(match["partida_id"]),
            "data": match["data"],
            "fase": match["fase"],
            "rodada": match["rodada"],
            "mandante": home,
            "visitante": away,
            "odd_mandante": odds[0],
            "odd_empate": odds[1],
            "odd_visitante": odds[2],
            "prob_mandante_impl": round(prob_h_impl, 4),
            "prob_empate_impl": round(prob_d_impl, 4),
            "prob_visitante_impl": round(prob_a_impl, 4),
            "margem": round(margem_final, 4),
            "bookmaker": bookmaker,
            "gols_mandante": int(match["gols_mandante"]),
            "gols_visitante": int(match["gols_visitante"]),
            "resultado": match["resultado"],
            "fonte": "exemplo",
        })

    df = pd.DataFrame(rows)
    return df[ODDS_COLUMNS]


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def run(output_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gera e salva as bases de exemplo. Retorna (partidas, odds)."""
    output_dir = output_dir or EXAMPLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    partidas = generate_partidas(rng)
    odds = generate_odds(partidas)

    partidas_path = output_dir / "partidas_libertadores_2026.csv"
    odds_path = output_dir / "odds_libertadores_2026.csv"
    partidas.to_csv(partidas_path, index=False)
    odds.to_csv(odds_path, index=False)

    print(f"✅ {len(partidas)} partidas de exemplo salvas em {partidas_path}")
    print(f"✅ {len(odds)} odds de exemplo salvas em {odds_path}")
    return partidas, odds


if __name__ == "__main__":
    run()
