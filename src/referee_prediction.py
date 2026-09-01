"""
Previsão de árbitros prováveis para partidas da Libertadores 2026.

Modelo determinístico, simples e defensável, baseado apenas nos dados reais de
2026 (``data/processed/libertadores_estatisticas_detalhadas.csv`` — as colunas
``arbitro``/``arbitro_pais`` só existem nessa temporada).

Para uma partida (mandante A × visitante B), cada árbitro ``r`` recebe um
escore em "pseudo-contagens":

1. **Prior global** — número de partidas apitadas por ``r`` em 2026
   (frequência bruta; a divisão pelo total é irrelevante após normalizar).
2. **Familiaridade com os times** — ``+TEAM_MATCH_WEIGHT`` (0.5) para cada
   partida de 2026 em que ``r`` apitou A ou B (mandante ou visitante).
3. **Fator de fase** — se a partida é de mata-mata (``fase`` contém 'Oitavas',
   'Quartas', 'Semi' ou 'Final'), árbitros com experiência em fases de
   mata-mata em 2026 (incluindo 'Preliminar') recebem multiplicador
   ``1 + KNOCKOUT_BONUS`` (0.25).

Os escores são normalizados para somar 1 (distribuição sobre TODOS os árbitros)
e a função retorna os ``top_n`` mais prováveis. Times sem histórico de árbitro
em 2026 caem naturalmente no prior global (com o fator de fase, se aplicável).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

TEAM_MATCH_WEIGHT = 0.5
"""Peso (pseudo-contagens) por partida de 2026 em que o árbitro apitou A ou B."""

KNOCKOUT_BONUS = 0.25
"""Multiplicador (1 + bônus) para árbitros com experiência em mata-mata."""

FASES_MATA_MATA = ("Oitavas", "Quartas", "Semi", "Final")
FASES_EXPERIENCIA_MATA_MATA = FASES_MATA_MATA + ("Preliminar",)

# (nome, pais, probabilidade)
ArbitroProvavel = Tuple[str, Optional[str], float]


def _eh_mata_mata(fase: object) -> bool:
    if fase is None or (isinstance(fase, float) and pd.isna(fase)):
        return False
    texto = str(fase)
    return any(k in texto for k in FASES_MATA_MATA)


def arbitros_mais_provaveis(
    df: pd.DataFrame,
    mandante: str,
    visitante: str,
    fase: Optional[str] = None,
    top_n: int = 2,
) -> List[ArbitroProvavel]:
    """Retorna os ``top_n`` árbitros mais prováveis para a partida.

    Parâmetros
    ----------
    df:
        DataFrame com colunas ``arbitro``, ``arbitro_pais``, ``mandante``,
        ``visitante`` e ``fase`` (partidas de 2026).
    mandante, visitante:
        Nomes dos times exatamente como aparecem nas colunas do ``df``.
    fase:
        Fase da partida futura (ex.: ``"Quartas de Final"``). Se for mata-mata,
        aplica o bônus de experiência em mata-mata.
    top_n:
        Quantidade de árbitros retornados (padrão: 2).

    Retorno
    -------
    Lista de tuplas ``(nome, pais, probabilidade)`` ordenada por probabilidade
    decrescente. ``pais`` pode ser ``None`` quando ``arbitro_pais`` é NaN na
    fonte (dados fbref). As probabilidades são fatias de uma distribuição
    normalizada sobre todos os árbitros de 2026. Retorna lista vazia se o
    ``df`` não tiver árbitros.
    """
    colunas = {"arbitro", "mandante", "visitante", "fase"}
    if df is None or df.empty or not colunas.issubset(df.columns):
        return []

    dados = df.dropna(subset=["arbitro"])
    if dados.empty:
        return []

    # 1) prior global (contagens brutas)
    escores = dados["arbitro"].value_counts().astype(float)

    # 2) familiaridade com os times
    mask_times = (
        dados["mandante"].isin([mandante, visitante])
        | dados["visitante"].isin([mandante, visitante])
    )
    contagens_times = dados.loc[mask_times, "arbitro"].value_counts().astype(float)
    escores = escores.add(TEAM_MATCH_WEIGHT * contagens_times, fill_value=0.0)

    # 3) fator de fase (bônus de experiência em mata-mata)
    if _eh_mata_mata(fase):
        experientes = dados.loc[
            dados["fase"].astype(str).str.contains(
                "|".join(FASES_EXPERIENCIA_MATA_MATA), regex=True
            ),
            "arbitro",
        ].unique()
        boost = pd.Series(1.0, index=escores.index)
        boost.loc[boost.index.isin(experientes)] = 1.0 + KNOCKOUT_BONUS
        escores = escores * boost

    probabilidades = escores / escores.sum()
    top = probabilidades.sort_values(ascending=False).head(top_n)

    paises = (
        dados.dropna(subset=["arbitro_pais"])
        .drop_duplicates(subset=["arbitro"])
        .set_index("arbitro")["arbitro_pais"]
        .to_dict()
        if "arbitro_pais" in dados.columns
        else {}
    )

    return [
        (str(nome), (str(paises[nome]) if nome in paises else None), float(prob))
        for nome, prob in top.items()
    ]
