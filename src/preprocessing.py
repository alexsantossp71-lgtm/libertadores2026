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


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.run()
