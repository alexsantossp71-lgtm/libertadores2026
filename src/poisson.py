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

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        """Retorna os gols esperados (lambdas) para um confronto.

        Parameters
        ----------
        home : str
            Nome do time mandante.
        away : str
            Nome do time visitante.

        Returns
        -------
        Tuple[float, float]
            ``(lambda_casa, lambda_fora)``.
        """
        self._check_team(home, "attack")
        self._check_team(away, "defense")

        lam_home = (
            self.attack[home]
            * self.defense[away]
            * self.home_advantage
            / self.league_avg
        )
        lam_away = (
            self.attack[away]
            * self.defense[home]
            * self.away_factor
            / self.league_avg
        )
        return lam_home, lam_away

    def score_probability_matrix(
        self, home: str, away: str, max_goals: Optional[int] = None
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
        lam_home, lam_away = self.expected_goals(home, away)

        goals = np.arange(max_goals + 1)
        prob_home = poisson.pmf(goals, lam_home)
        prob_away = poisson.pmf(goals, lam_away)

        return np.outer(prob_home, prob_away)

    def match_probabilities(
        self, home: str, away: str, max_goals: Optional[int] = None
    ) -> Dict[str, float]:
        """Probabilidades 1X2 e placar mais provável para um confronto.

        Returns
        -------
        dict
            Chaves: ``p_home``, ``p_draw``, ``p_away``, ``expected_goals_home``,
            ``expected_goals_away`` e ``most_likely_score`` (tupla ``(i, j)``).
        """
        max_goals = max_goals or self.max_goals
        matrix = self.score_probability_matrix(home, away, max_goals)

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
