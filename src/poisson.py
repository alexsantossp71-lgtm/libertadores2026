"""
Modelo de Regressão de Poisson para previsão de placares - Libertadores 2026.

O modelo parte da premissa clássica de que o número de gols de cada time em uma
partida segue uma distribuição de Poisson cuja taxa (lambda) depende da força de
ataque do time e da fragilidade defensiva do adversário, ajustada pelo mando de
campo:

    lambda_casa = ataque_casa * defesa_fora * vantagem_casa / media_liga
    lambda_fora = ataque_fora * defesa_casa * fator_fora     / media_liga

Onde:
    - ``ataque_i`` é a média de gols marcados por jogo do time ``i``;
    - ``defesa_j`` é a média de gols sofridos por jogo do time ``j``;
    - ``media_liga`` é a média de gols por jogo (por time) da competição;
    - ``vantagem_casa`` e ``fator_fora`` modelam o efeito de jogar em casa.

Com os lambdas estimados, a probabilidade de um placar (i x j) é o produto das
probabilidades marginais (Poissons independentes):

    P(casa = i, fora = j) = Poisson(i; lambda_casa) * Poisson(j; lambda_fora)

A partir da matriz de placares derivam-se as probabilidades 1X2 (vitória do
mandante, empate e vitória do visitante) e o placar mais provável.

Referência conceitual: Maher (1982) e Dixon & Coles (1997), na forma
multiplicativa simplificada aplicável a estatísticas agregadas (fase de grupos).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Colunas mínimas esperadas nos dados agregados da fase de grupos
REQUIRED_COLUMNS = {"Time", "J", "GP", "GC"}


class PoissonScoreModel:
    """Modelo de Poisson para previsão de placares e probabilidades 1X2.

    Parameters
    ----------
    home_advantage : float
        Fator multiplicativo de gols esperados para o time mandante.
    away_factor : float
        Fator multiplicativo de gols esperados para o time visitante.
        Por padrão ``away_factor = 2 - home_advantage``, preservando o total
        esperado de gols da partida para um confronto equilibrado.
    max_goals : int
        Número máximo de gols considerado na matriz de placares (truncamento da
        cauda da distribuição de Poisson).
    """

    def __init__(
        self,
        home_advantage: float = 1.15,
        away_factor: Optional[float] = None,
        max_goals: int = 10,
    ):
        self.home_advantage = float(home_advantage)
        self.away_factor = (
            float(away_factor)
            if away_factor is not None
            else 2.0 - self.home_advantage
        )
        self.max_goals = int(max_goals)

        self.attack: Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.league_avg: float = 0.0
        self.teams: List[str] = []
        self.is_fitted: bool = False
        self.elenco_applied: bool = False
        self.elenco_multipliers: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    # Ajuste do modelo
    # ------------------------------------------------------------------ #
    def fit(self, grupos_df: pd.DataFrame) -> "PoissonScoreModel":
        """Estima forças de ataque/defesa a partir dos dados da fase de grupos.

        Parameters
        ----------
        grupos_df : pd.DataFrame
            Tabela agregada da fase de grupos. Deve conter ao menos as colunas
            ``Time``, ``J`` (jogos), ``GP`` (gols pró) e ``GC`` (gols contra).

        Returns
        -------
        PoissonScoreModel
            A própria instância, para encadeamento de chamadas.
        """
        if not isinstance(grupos_df, pd.DataFrame):
            raise TypeError("grupos_df deve ser um pandas.DataFrame.")

        missing = REQUIRED_COLUMNS - set(grupos_df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

        df = grupos_df.copy()
        # Remove times sem jogos (evita divisão por zero)
        df = df[df["J"] > 0]

        if df.empty:
            raise ValueError("Nenhum time com jogos registrados (coluna 'J').")

        total_jogos = df["J"].sum()
        total_gols = df["GP"].sum()

        self.league_avg = float(total_gols / total_jogos)
        if self.league_avg <= 0:
            raise ValueError(
                "Média de gols por jogo igual a zero; impossível ajustar o modelo."
            )

        self.attack = {
            row["Time"]: float(row["GP"] / row["J"]) for _, row in df.iterrows()
        }
        self.defense = {
            row["Time"]: float(row["GC"] / row["J"]) for _, row in df.iterrows()
        }
        self.teams = sorted(self.attack.keys())
        self.is_fitted = True
        self.elenco_applied = False
        self.elenco_multipliers = {}
        self._attack_base = dict(self.attack)
        self._defense_base = dict(self.defense)

        return self

    def fit_enhanced(self, grupos_df: pd.DataFrame, knockout_weight: float = 0.15) -> "PoissonScoreModel":
        """Ajusta o modelo com ajuste baseado em desempenho no mata-mata.

        Parameters
        ----------
        grupos_df : pd.DataFrame
            Tabela agregada da fase de grupos com colunas opcionais
            ``chegou_oitavas`` (0/1) e ``chegou_quartas`` (0/1).
        knockout_weight : float
            Peso do ajuste baseado no mata-mata (0 a 1). Default 0.15.

        Returns
        -------
        PoissonScoreModel
            A própria instância.
        """
        # Primeiro, ajuste base
        self.fit(grupos_df)

        # Ajuste baseado em desempenho no mata-mata
        if "chegou_oitavas" in grupos_df.columns or "chegou_quartas" in grupos_df.columns:
            df = grupos_df.copy()
            df = df[df["J"] > 0]

            # Fator de ajuste baseado no progresso no mata-mata
            # Times que chegaram às oitavas: +5% ataque, -5% defesa
            # Times que chegaram às quartas: +10% ataque, -10% defesa
            # (adicional ao bônus de oitavas)
            for _, row in df.iterrows():
                team = row["Time"]
                if team not in self.attack:
                    continue

                mult_attack = 1.0
                mult_defense = 1.0

                if row.get("chegou_oitavas", 0) == 1:
                    mult_attack *= (1 + 0.05 * knockout_weight)
                    mult_defense *= (1 - 0.05 * knockout_weight)

                if row.get("chegou_quartas", 0) == 1:
                    mult_attack *= (1 + 0.10 * knockout_weight)
                    mult_defense *= (1 - 0.10 * knockout_weight)

                self.attack[team] *= mult_attack
                self.defense[team] *= mult_defense

            # Recalcular médias da liga após ajustes
            total_attack = sum(self.attack.values())
            total_defense = sum(self.defense.values())
            n_teams = len(self.teams)
            self.league_avg = (total_attack + total_defense) / (2 * n_teams)

            # Atualizar bases para idempotência
            self._attack_base = dict(self.attack)
            self._defense_base = dict(self.defense)

        return self

    def apply_elenco_multipliers(
        self, multipliers: Dict[str, Tuple[float, float]]
    ) -> "PoissonScoreModel":
        """Aplica multiplicadores de elenco sobre as forças da fase de grupos.

        ``multipliers[time] = (mult_ataque, mult_defesa)``. Defesa é taxa de
        gols sofridos: valor < 1 significa sofrer menos. Sempre parte das
        forças-base do ``fit`` (idempotente).
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")
        base_att = getattr(self, "_attack_base", self.attack)
        base_def = getattr(self, "_defense_base", self.defense)
        self.attack = dict(base_att)
        self.defense = dict(base_def)
        applied: Dict[str, Tuple[float, float]] = {}
        for team, pair in multipliers.items():
            if team not in self.attack:
                continue
            att_m, def_m = float(pair[0]), float(pair[1])
            self.attack[team] = max(0.05, self.attack[team] * att_m)
            self.defense[team] = max(0.05, self.defense[team] * def_m)
            applied[team] = (att_m, def_m)
        self.elenco_applied = bool(applied)
        self.elenco_multipliers = applied
        return self

    # ------------------------------------------------------------------ #
    # Previsões
    # ------------------------------------------------------------------ #
    def _check_team(self, team: str, param: str) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Modelo não ajustado. Execute fit() antes de fazer previsões."
            )
        if team not in getattr(self, param):
            raise KeyError(
                f"Time '{team}' não encontrado no modelo. "
                f"Times conhecidos: {self.teams}"
            )

    def expected_goals(
        self, home: str, away: str, neutral: bool = False
    ) -> Tuple[float, float]:
        """Retorna os gols esperados (lambdas) para um confronto.

        Parameters
        ----------
        home : str
            Nome do time mandante.
        away : str
            Nome do time visitante.
        neutral : bool
            Se ``True``, desliga a vantagem de campo (fatores 1.0 para os
            dois lados) — usado em finais em campo neutro.

        Returns
        -------
        Tuple[float, float]
            ``(lambda_casa, lambda_fora)``.
        """
        self._check_team(home, "attack")
        self._check_team(away, "defense")

        home_factor = 1.0 if neutral else self.home_advantage
        away_factor = 1.0 if neutral else self.away_factor

        lam_home = (
            self.attack[home]
            * self.defense[away]
            * home_factor
            / self.league_avg
        )
        lam_away = (
            self.attack[away]
            * self.defense[home]
            * away_factor
            / self.league_avg
        )
        return lam_home, lam_away

    def score_probability_matrix(
        self, home: str, away: str, max_goals: Optional[int] = None,
        neutral: bool = False,
    ) -> np.ndarray:
        """Matriz de probabilidade de placares.

        O elemento ``matrix[i, j]`` é a probabilidade de o placar ser ``i x j``
        (mandante x visitante).

        Returns
        -------
        np.ndarray
            Matriz ``(max_goals + 1) x (max_goals + 1)``.
        """
        max_goals = max_goals or self.max_goals
        lam_home, lam_away = self.expected_goals(home, away, neutral=neutral)

        goals = np.arange(max_goals + 1)
        prob_home = poisson.pmf(goals, lam_home)
        prob_away = poisson.pmf(goals, lam_away)

        return np.outer(prob_home, prob_away)

    def match_probabilities(
        self, home: str, away: str, max_goals: Optional[int] = None,
        neutral: bool = False,
    ) -> Dict[str, float]:
        """Probabilidades 1X2 e placar mais provável para um confronto.

        Com ``neutral=True`` a vantagem de campo é desligada (final em campo
        neutro); ``p_home``/``p_away`` passam a se referir aos dois lados
        sem mando.
        """
        max_goals = max_goals or self.max_goals
        matrix = self.score_probability_matrix(home, away, max_goals, neutral=neutral)

        # Mandante vence quando marca mais gols que o visitante.
        # Na matriz ``matrix[i, j]``, ``i`` é o placar do mandante e ``j`` o do
        # visitante; logo, vitória do mandante corresponde ao triângulo inferior
        # estrito (i > j).
        p_home = float(np.tril(matrix, k=-1).sum())
        # Vitória do visitante: triângulo superior estrito (j > i).
        p_away = float(np.triu(matrix, k=1).sum())
        # Empate: diagonal principal (i == j).
        p_draw = float(np.trace(matrix))

        # Normaliza pelo truncamento da cauda da Poisson em max_goals
        total = p_home + p_draw + p_away
        if total > 0:
            p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        lam_home, lam_away = self.expected_goals(home, away)

        return {
            "home": home,
            "away": away,
            "expected_goals_home": lam_home,
            "expected_goals_away": lam_away,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "most_likely_score": (int(i), int(j)),
        }

    def tie_probabilities(
        self, team_a: str, team_b: str, max_goals: Optional[int] = None
    ) -> Dict[str, float]:
        """Probabilidades de um confronto eliminatório em ida e volta.

        ``team_a`` manda o jogo de ida; ``team_b`` manda a volta. A
        probabilidade de avançar soma os placares agregados das duas pernas
        (matrizes de Poisson convoluídas); empate no agregado é decidido por
        pênaltis e modelado como 50/50 — informação que o modelo não tem.

        Returns
        -------
        dict
            ``p_advance_a``, ``p_advance_b``, ``p_aggregate_draw`` (pênaltis),
            ``agg_mais_provavel`` (tupla de gols agregados), ``lambda_ida``
            e ``lambda_volta`` (gols esperados por perna).
        """
        self._check_team(team_a, "attack")
        self._check_team(team_b, "attack")

        m_ida = self.score_probability_matrix(team_a, team_b, max_goals)
        m_volta = self.score_probability_matrix(team_b, team_a, max_goals)

        # Enumeração conjunta das duas pernas: A marca i (ida) + l (volta);
        # B marca j (ida) + k (volta).
        n = m_ida.shape[0]
        idx = np.arange(n)
        ga = np.add.outer(idx, np.zeros(n))[:, :, None, None] + np.add.outer(np.zeros(n), idx)[None, None, :, :]
        gb = np.add.outer(np.zeros(n), idx)[:, :, None, None] + np.add.outer(idx, np.zeros(n))[None, None, :, :]

        joint = m_ida[:, :, None, None] * m_volta[None, None, :, :]
        flat = joint.ravel()
        total = flat.sum()
        a_flat, b_flat = ga.ravel(), gb.ravel()

        p_a = float(flat[a_flat > b_flat].sum() / total)
        p_b = float(flat[a_flat < b_flat].sum() / total)
        p_pen = float(flat[a_flat == b_flat].sum() / total)

        combos: Dict[Tuple[int, int], float] = {}
        for x, y, p in zip(a_flat, b_flat, flat):
            key = (int(x), int(y))
            combos[key] = combos.get(key, 0.0) + float(p)
        agg_ml = max(combos, key=combos.get)

        lam_a1, lam_b1 = self.expected_goals(team_a, team_b)
        lam_b2, lam_a2 = self.expected_goals(team_b, team_a)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "p_advance_a": p_a + 0.5 * p_pen,
            "p_advance_b": p_b + 0.5 * p_pen,
            "p_aggregate_draw": p_pen,
            "agg_mais_provavel": agg_ml,
            "lambda_ida": (lam_a1, lam_b1),
            "lambda_volta": (lam_a2, lam_b2),
        }

    def cup_tie_probabilities(
        self, team_a: str, team_b: str, max_goals: Optional[int] = None
    ) -> Dict[str, float]:
        """Probabilidades de vencer um jogo único em campo neutro (final).

        Vitória em 90 minutos pela matriz de Poisson neutra; empate vai para
        prorrogação + pênaltis, modelado como 50/50.
        """
        probs = self.match_probabilities(team_a, team_b, max_goals, neutral=True)
        return {
            "p_win_a": probs["p_home"] + 0.5 * probs["p_draw"],
            "p_win_b": probs["p_away"] + 0.5 * probs["p_draw"],
            "p_draw_90min": probs["p_draw"],
            "placar_mais_provavel": probs["most_likely_score"],
            "xg": (probs["expected_goals_home"], probs["expected_goals_away"]),
        }

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def strengths(self) -> pd.DataFrame:
        """DataFrame com forças de ataque e defesa de cada time."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não ajustado. Execute fit() primeiro.")

        return pd.DataFrame(
            {
                "Time": self.teams,
                "Ataque": [self.attack[t] for t in self.teams],
                "Defesa": [self.defense[t] for t in self.teams],
            }
        ).sort_values("Ataque", ascending=False)

    def __repr__(self) -> str:
        status = "ajustado" if self.is_fitted else "não ajustado"
        return (
            f"PoissonScoreModel(status={status}, times={len(self.teams)}, "
            f"home_advantage={self.home_advantage}, "
            f"away_factor={self.away_factor})"
        )
