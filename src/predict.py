"""
Script para geração de previsões das Quartas de Final
"""

import pandas as pd
from pathlib import Path
from typing import Dict

from preprocessing import Preprocessor
from model import LibertadoresModel

# Configurações
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LibertadoresPredictor:
    """Gerador de previsões para a Libertadores 2026."""
    
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = LibertadoresModel()
    
    def generate_quartas_predictions(self) -> pd.DataFrame:
        """Gera previsões para as quartas de final usando o modelo de Poisson."""
        print("=" * 50)
        print("Gerando Previsões - Quartas de Final")
        print("=" * 50)
        
        # Carrega dados
        grupos, _, quartas = self.preprocessor.load_data()
        grupos_features = self.preprocessor.create_features(grupos)
        
        # Ajusta o modelo de Poisson com os dados da fase de grupos
        print("Ajustando modelo de Poisson com os dados da fase de grupos...")
        self.model.fit_poisson(grupos_features)
        
        # Gera features para cada confronto
        confrontos = []
        
        for _, row in quartas.iterrows():
            mandante = row['Mandante']
            visitante = row['Visitante']
            
            # Cria features do confronto
            match_features = self.preprocessor.create_match_features(
                grupos_features, mandante, visitante
            )
            
            # Faz previsão
            result_probs = self.model.predict_match_poisson(match_features)
            score = self.model.predict_score(match_features)
            
            confrontos.append({
                'Confronto': row['Confronto'],
                'Mandante': mandante,
                'Visitante': visitante,
                'Pais_Mandante': row['Pais_Mandante'],
                'Pais_Visitante': row['Pais_Visitante'],
                'Data_Ida': row['Data_Ida'],
                'Data_Volta': row['Data_Volta'],
                'Prob_Mandante': result_probs['prob_vitoria_mandante'],
                'Prob_Empate': result_probs['prob_empate'],
                'Prob_Visitante': result_probs['prob_derrota_mandante'],
                'Gols_Esperados_Mandante': result_probs['gols_esperados_mandante'],
                'Gols_Esperados_Visitante': result_probs['gols_esperados_visitante'],
                'Placar_Previsto': f"{score[0]}x{score[1]}",
                'Favorito': result_probs['resultado_previsto']
            })
        
        df_predictions = pd.DataFrame(confrontos)
        
        return df_predictions
    
    def save_predictions(self, df: pd.DataFrame, format: str = 'csv'):
        """Salva previsões em arquivo."""
        if format == 'csv':
            filepath = OUTPUT_DIR / "quartas_previsao.csv"
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            filepath = OUTPUT_DIR / "quartas_previsao.xlsx"
            df.to_excel(filepath, index=False)
        
        print(f"Previsões salvas em: {filepath}")
    
    def print_predictions(self, df: pd.DataFrame):
        """Imprime previsões formatadas."""
        print("\n" + "=" * 80)
        print("⚽ PREVISÕES - QUARTAS DE FINAL DA LIBERTADORES 2026")
        print("=" * 80)
        
        for _, row in df.iterrows():
            print(f"\n{row['Confronto']}: {row['Mandante']} ({row['Pais_Mandante']}) x ({row['Pais_Visitante']}) {row['Visitante']}")
            print(f"   📅 Ida: {row['Data_Ida']} | Volta: {row['Data_Volta']}")
            print(f"   📊 Probabilidades:")
            print(f"      {row['Mandante']}: {row['Prob_Mandante']:.1%}")
            print(f"      Empate: {row['Prob_Empate']:.1%}")
            print(f"      {row['Visitante']}: {row['Prob_Visitante']:.1%}")
            print(f"   🥅 Gols esperados (Poisson): "
                  f"{row['Gols_Esperados_Mandante']:.2f} x "
                  f"{row['Gols_Esperados_Visitante']:.2f}")
            print(f"   🔮 Placar Previsto: {row['Placar_Previsto']}")
            print(f"   ⭐ Favorito: {row['Favorito']}")
        
        print("\n" + "=" * 80)
        print("⚠️  Aviso: Previsões baseadas em dados estatísticos.")
        print("   Fatores como lesões, suspensões e decisões táticas ")
        print("   não são considerados no modelo atual.")
        print("=" * 80)
    
    def run(self):
        """Executa pipeline completo de previsões."""
        # Gera previsões
        predictions = self.generate_quartas_predictions()
        
        # Imprime e salva
        self.print_predictions(predictions)
        self.save_predictions(predictions)
        
        return predictions


if __name__ == "__main__":
    predictor = LibertadoresPredictor()
    predictor.run()
