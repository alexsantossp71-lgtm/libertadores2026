"""
Pré-processamento e engenharia de features para o modelo preditivo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# Configurações
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class Preprocessor:
    """Pré-processador de dados para o modelo de previsão."""
    
    def __init__(self):
        self.feature_names = []
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega dados brutos."""
        grupos = pd.read_csv(RAW_DIR / "grupos_libertadores_2026.csv")
        oitavas = pd.read_csv(RAW_DIR / "oitavas_resultados.csv")
        quartas = pd.read_csv(RAW_DIR / "confrontos_quartas.csv")
        return grupos, oitavas, quartas
    
    def create_features(self, df_grupos: pd.DataFrame) -> pd.DataFrame:
        """Cria features a partir dos dados da fase de grupos."""
        df = df_grupos.copy()
        
        # Features básicas
        df['Aproveitamento'] = df['Pts'] / (df['J'] * 3)  # % de pontos possíveis
        df['Media_Gols_Marcados'] = df['GP'] / df['J']
        df['Media_Gols_Sofridos'] = df['GC'] / df['J']
        df['Razao_Gols'] = df['GP'] / (df['GC'] + 1)  # +1 para evitar divisão por zero
        df['Vitorias_Seq'] = df['V']  # Placeholder para vitórias consecutivas
        
        # Features categóricas codificadas
        pais_map = {'BRA': 3, 'ARG': 2, 'ECU': 1, 'CHI': 1, 'COL': 1}
        df['Pais_Cod'] = df['Pais'].map(pais_map)
        
        # Score composto (força do time)
        df['Score_Forca'] = (
            df['Pts'] * 0.4 + 
            df['SG'] * 0.3 + 
            df['GP'] * 0.2 + 
            df['V'] * 10 * 0.1
        )
        
        self.feature_names = [
            'Pts', 'J', 'V', 'E', 'D', 'GP', 'GC', 'SG',
            'Aproveitamento', 'Media_Gols_Marcados', 'Media_Gols_Sofridos',
            'Razao_Gols', 'Pais_Cod', 'Score_Forca'
        ]
        
        return df
    
    def create_match_features(
        self, 
        df_grupos: pd.DataFrame, 
        mandante: str, 
        visitante: str
    ) -> pd.DataFrame:
        """Cria features para um confronto específico."""
        df = df_grupos.copy()
        
        # Busca dados dos times
        time_mandante = df[df['Time'] == mandante].iloc[0]
        time_visitante = df[df['Time'] == visitante].iloc[0]
        
        # Cria features do confronto
        match_features = pd.DataFrame({
            'Time_Mandante': [mandante],
            'Time_Visitante': [visitante],
            
            # Diferenças de estatísticas
            'Diff_Pts': [time_mandante['Pts'] - time_visitante['Pts']],
            'Diff_GP': [time_mandante['GP'] - time_visitante['GP']],
            'Diff_GC': [time_mandante['GC'] - time_visitante['GC']],
            'Diff_SG': [time_mandante['SG'] - time_visitante['SG']],
            'Diff_Aproveitamento': [
                time_mandante['Aproveitamento'] - time_visitante['Aproveitamento']
            ],
            'Diff_Media_Gols': [
                time_mandante['Media_Gols_Marcados'] - time_visitante['Media_Gols_Marcados']
            ],
            'Diff_Score_Forca': [
                time_mandante['Score_Forca'] - time_visitante['Score_Forca']
            ],
            
            # Razão entre estatísticas
            'Razao_Pts': [time_mandante['Pts'] / (time_visitante['Pts'] + 1)],
            'Razao_GP': [time_mandante['GP'] / (time_visitante['GP'] + 1)],
            'Razao_Score': [time_mandante['Score_Forca'] / (time_visitante['Score_Forca'] + 1)],
            
            # País (fator casa pode ser relevante)
            'Pais_Mandante_Cod': [time_mandante['Pais_Cod']],
            'Pais_Visitante_Cod': [time_visitante['Pais_Cod']],
            'Mesmo_Pais': [1 if time_mandante['Pais'] == time_visitante['Pais'] else 0],
        })
        
        return match_features
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados para treinamento."""
        # Remove colunas não-numéricas
        df_numeric = df.select_dtypes(include=[np.number])
        
        # Preenche valores ausentes
        df_numeric = df_numeric.fillna(0)
        
        return df_numeric
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Salva dados processados."""
        filepath = PROCESSED_DIR / filename
        df.to_csv(filepath, index=False)
        print(f"Dados processados salvos em: {filepath}")
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Executa pipeline completo de pré-processamento."""
        print("=" * 50)
        print("Iniciando pré-processamento")
        print("=" * 50)
        
        # Carrega dados
        grupos, oitavas, quartas = self.load_data()
        
        # Cria features
        grupos_features = self.create_features(grupos)
        
        # Salva dados processados
        self.save_processed_data(grupos_features, "features_libertadores.csv")
        
        print("=" * 50)
        print("Pré-processamento concluído!")
        print("=" * 50)
        
        return grupos_features, oitavas, quartas

    # ------------------------------------------------------------------ #
    # Novas fontes: estatísticas detalhadas e odds (arbitragem/mercado)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _permitir_exemplo() -> bool:
        """Fallback para base sintética só com opt-in explícito."""
        import os

        return os.getenv("ALLOW_EXAMPLE_DATA", "").lower() in ("1", "true", "yes")

    def load_estatisticas(self) -> pd.DataFrame:
        """
        Carrega as estatísticas detalhadas das partidas (API Futebol).

        Ordem de prioridade:
          1. ``data/processed/libertadores_estatisticas_detalhadas.csv``
             (gerado pelo pipeline a partir da API Futebol);
          2. base de exemplo — **apenas** se ``ALLOW_EXAMPLE_DATA=1``
             (desligada por padrão: o projeto usa somente dados reais).
        """
        from api_futebol_client import PROCESSED_PATH as STATS_PATH

        if STATS_PATH.exists():
            return pd.read_csv(STATS_PATH)

        if not self._permitir_exemplo():
            raise FileNotFoundError(
                "Estatísticas detalhadas reais indisponíveis: configure "
                "API_FUTEBOL_KEY (ver .env.example) e rode o pipeline. "
                "Para usar a base sintética em desenvolvimento: ALLOW_EXAMPLE_DATA=1."
            )
        from generate_example_data import EXAMPLE_PARTIDAS_PATH, generate_partidas

        if EXAMPLE_PARTIDAS_PATH.exists():
            return pd.read_csv(EXAMPLE_PARTIDAS_PATH)
        return generate_partidas()

    def load_odds(self) -> pd.DataFrame:
        """
        Carrega as odds processadas das partidas (Bzzoiro Sports Data).

        Ordem de prioridade:
          1. ``data/processed/libertadores_odds.csv`` (gerado pelo pipeline);
          2. base de exemplo — **apenas** se ``ALLOW_EXAMPLE_DATA=1``.
        """
        from odds_client import PROCESSED_PATH as ODDS_PATH

        if ODDS_PATH.exists():
            return pd.read_csv(ODDS_PATH)

        if not self._permitir_exemplo():
            raise FileNotFoundError(
                "Odds reais indisponíveis: configure BSD_API "
                "(ver .env.example) e rode o pipeline. "
                "Para usar a base sintética em desenvolvimento: ALLOW_EXAMPLE_DATA=1."
            )
        from generate_example_data import EXAMPLE_ODDS_PATH, generate_odds

        if EXAMPLE_ODDS_PATH.exists():
            return pd.read_csv(EXAMPLE_ODDS_PATH)
        return generate_odds(self.load_estatisticas())

    def process_estatisticas(self, df_partidas: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas derivadas (totais por partida) às estatísticas."""
        df = df_partidas.copy()

        df["total_faltas"] = df["faltas_mandante"] + df["faltas_visitante"]
        df["total_cartoes_amarelos"] = (
            df["cartoes_amarelos_mandante"] + df["cartoes_amarelos_visitante"]
        )
        df["total_cartoes_vermelhos"] = (
            df["cartoes_vermelhos_mandante"] + df["cartoes_vermelhos_visitante"]
        )
        df["total_cartoes"] = df["total_cartoes_amarelos"] + df["total_cartoes_vermelhos"]
        df["total_gols"] = df["gols_mandante"] + df["gols_visitante"]
        df["total_passes_certos"] = (
            df["passes_certos_mandante"] + df["passes_certos_visitante"]
        )
        df["total_passes_errados"] = (
            df["passes_errados_mandante"] + df["passes_errados_visitante"]
        )
        df["total_finalizacoes"] = (
            df["finalizacoes_mandante"] + df["finalizacoes_visitante"]
        )
        df["total_finalizacoes_no_gol"] = (
            df["finalizacoes_no_gol_mandante"] + df["finalizacoes_no_gol_visitante"]
        )
        df["total_escanteios"] = df["escanteios_mandante"] + df["escanteios_visitante"]
        df["total_impedimentos"] = (
            df["impedimentos_mandante"] + df["impedimentos_visitante"]
        )
        df["total_defesas"] = df["defesas_mandante"] + df["defesas_visitante"]

        # Estatísticas agregadas do mandante (mando de campo)
        df["aproveitamento_mandante"] = np.where(
            df["resultado"] == "mandante", 1.0,
            np.where(df["resultado"] == "empate", 1 / 3, 0.0),
        )
        return df

    def referee_summary(self, df_partidas: pd.DataFrame) -> pd.DataFrame:
        """Resumo descritivo por árbitro (médias por jogo)."""
        df = self.process_estatisticas(df_partidas)

        summary = (
            df.groupby("arbitro")
            .agg(
                arbitro_pais=("arbitro_pais", "first"),
                jogos=("partida_id", "count"),
                media_faltas=("total_faltas", "mean"),
                media_cartoes_amarelos=("total_cartoes_amarelos", "mean"),
                media_cartoes_vermelhos=("total_cartoes_vermelhos", "mean"),
                media_cartoes=("total_cartoes", "mean"),
                media_gols=("total_gols", "mean"),
                media_posse_mandante=("posse_mandante", "mean"),
                media_passes_certos=("total_passes_certos", "mean"),
                media_passes_errados=("total_passes_errados", "mean"),
                media_finalizacoes=("total_finalizacoes", "mean"),
                media_escanteios=("total_escanteios", "mean"),
                media_impedimentos=("total_impedimentos", "mean"),
                media_defesas=("total_defesas", "mean"),
            )
            .reset_index()
        )
        # Arredonda as médias para melhor legibilidade
        for col in summary.columns:
            if col not in ("arbitro", "arbitro_pais", "jogos"):
                summary[col] = summary[col].round(2)
        return summary.sort_values("media_faltas", ascending=False).reset_index(drop=True)

    def add_rigor_groups(
        self,
        df_partidas: pd.DataFrame,
        n_groups: int = 3,
        labels: Tuple[str, str, str] = ("Permissivo", "Moderado", "Rigoroso"),
    ) -> pd.DataFrame:
        """Classifica partidas em tercis de rigor pela média de faltas do árbitro."""
        df = self.process_estatisticas(df_partidas)
        media_por_arbitro = df.groupby("arbitro")["total_faltas"].transform("mean")
        try:
            df["grupo_rigor"] = pd.qcut(
                media_por_arbitro, q=n_groups, labels=list(labels), duplicates="drop"
            ).astype(str)
        except ValueError:
            # Poucos árbitros distintos — cai para 2 grupos
            df["grupo_rigor"] = pd.qcut(
                media_por_arbitro, q=2, labels=["Menos rigoroso", "Mais rigoroso"],
                duplicates="drop",
            ).astype(str)
        return df

    def process_odds(self, df_odds: pd.DataFrame) -> pd.DataFrame:
        """
        Processa odds 1X2: garante as colunas de odds e recalcula as
        probabilidades implícitas normalizadas (margem removida).
        """
        df = df_odds.copy()
        required = {"odd_mandante", "odd_empate", "odd_visitante"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas de odds ausentes: {sorted(missing)}")

        probs = np.array([
            implied_probabilities(row["odd_mandante"], row["odd_empate"], row["odd_visitante"])
            for _, row in df.iterrows()
        ])
        df["prob_mandante_impl"] = probs[:, 0]
        df["prob_empate_impl"] = probs[:, 1]
        df["prob_visitante_impl"] = probs[:, 2]
        df["margem"] = probs[:, 3]
        return df

    def model_vs_market(self, model, df_odds: pd.DataFrame) -> pd.DataFrame:
        """
        Junta as probabilidades 1X2 do modelo (Poisson) e as implícitas das
        odds para cada partida, com a divergência entre as duas visões.
        """
        df = self.process_odds(df_odds)
        linhas = []
        for _, row in df.iterrows():
            try:
                probs = model.match_probabilities(row["mandante"], row["visitante"])
            except (KeyError, RuntimeError):
                continue  # time fora da base do modelo
            linhas.append({
                **row.to_dict(),
                "prob_mandante_modelo": probs["p_home"],
                "prob_empate_modelo": probs["p_draw"],
                "prob_visitante_modelo": probs["p_away"],
                "xG_mandante": probs["expected_goals_home"],
                "xG_visitante": probs["expected_goals_away"],
            })
        if not linhas:
            return pd.DataFrame()
        out = pd.DataFrame(linhas)
        out["divergencia_mandante"] = (
            out["prob_mandante_modelo"] - out["prob_mandante_impl"]
        )
        out["divergencia_abs"] = out["divergencia_mandante"].abs()
        return out

    def combined_probabilities(
        self, df: pd.DataFrame, threshold: float = 0.08
    ) -> pd.DataFrame:
        """
        Combinação "inteligente" modelo × mercado:

        * quando o mercado diverge **fortemente** do modelo em alguma classe
          (|p_modelo - p_mercado| > ``threshold``), assume-se que o mercado
          tem informação extra (lesões, escalações etc.) e usa-se a odd;
        * caso contrário, usa-se a probabilidade do modelo.

        Retorna o DataFrame com as colunas ``prob_mandante_combinada``,
        ``prob_empate_combinada``, ``prob_visitante_combinada`` e a flag
        ``usou_mercado``.
        """
        cols_modelo = ("prob_mandante_modelo", "prob_empate_modelo", "prob_visitante_modelo")
        cols_mercado = ("prob_mandante_impl", "prob_empate_impl", "prob_visitante_impl")
        missing = set(cols_modelo + cols_mercado) - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes para a combinação: {sorted(missing)}")

        div = pd.DataFrame({
            cols_modelo[0]: (df[cols_modelo[0]] - df[cols_mercado[0]]).abs(),
            cols_modelo[1]: (df[cols_modelo[1]] - df[cols_mercado[1]]).abs(),
            cols_modelo[2]: (df[cols_modelo[2]] - df[cols_mercado[2]]).abs(),
        })
        usar_mercado = div.max(axis=1) > threshold

        out = df.copy()
        out["usou_mercado"] = usar_mercado
        out["prob_mandante_combinada"] = np.where(
            usar_mercado, df["prob_mandante_impl"], df["prob_mandante_modelo"]
        )
        out["prob_empate_combinada"] = np.where(
            usar_mercado, df["prob_empate_impl"], df["prob_empate_modelo"]
        )
        out["prob_visitante_combinada"] = np.where(
            usar_mercado, df["prob_visitante_impl"], df["prob_visitante_modelo"]
        )
        return out

    def evaluate_probabilities(
        self, df: pd.DataFrame, prob_cols: Tuple[str, str, str], resultado_col: str = "resultado"
    ) -> Dict[str, float]:
        """
        Avalia probabilidades 1X2 contra o resultado real.

        Returns
        -------
        dict com ``acuracia`` (fração em que a classe mais provável acertou),
        ``brier_score`` (Brier multiclasse) e ``n`` (número de jogos avaliados).
        """
        df_valid = df.dropna(subset=[resultado_col, *prob_cols]).copy()
        df_valid = df_valid[df_valid[resultado_col].isin(("mandante", "empate", "visitante"))]

        if df_valid.empty:
            return {"acuracia": float("nan"), "brier_score": float("nan"), "n": 0}

        # As colunas de probabilidade devem estar na ordem
        # (mandante, empate, visitante).
        classes = {"mandante": 0, "empate": 1, "visitante": 2}
        probs = df_valid[list(prob_cols)].to_numpy()
        reais = df_valid[resultado_col].map(classes).to_numpy()

        previstos = probs.argmax(axis=1)
        acuracia = float((previstos == reais).mean())

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(reais)), reais] = 1.0
        brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())

        return {"acuracia": acuracia, "brier_score": brier, "n": int(len(df_valid))}


def implied_probabilities(
    odd_mandante: float, odd_empate: float, odd_visitante: float
) -> Tuple[float, float, float, float]:
    """
    Probabilidades implícitas 1X2 (1/odd) normalizadas para somar 1.

    Returns
    -------
    (p_mandante, p_empate, p_visitante, margem) — margem = overround - 1.
    """
    raw = [1.0 / float(odd_mandante), 1.0 / float(odd_empate), 1.0 / float(odd_visitante)]
    margem = sum(raw) - 1.0
    total = sum(raw) or 1.0
    return raw[0] / total, raw[1] / total, raw[2] / total, margem


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
