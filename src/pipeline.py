"""
Pipeline principal do projeto Libertadores 2026
Executa todo o fluxo: scraping -> preprocessing -> modelagem -> previsões
"""

import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

from scraper import LibertadoresScraper
from preprocessing import Preprocessor
from model import LibertadoresModel
from predict import LibertadoresPredictor


class LibertadoresPipeline:
    """Pipeline completo para previsão da Libertadores 2026."""
    
    def __init__(self):
        self.scraper = LibertadoresScraper()
        self.preprocessor = Preprocessor()
        self.model = LibertadoresModel()
        self.predictor = LibertadoresPredictor()
    
    def run(self, skip_scraping: bool = False, skip_training: bool = False):
        """
        Executa pipeline completo.
        
        Args:
            skip_scraping: Se True, pula a coleta de dados
            skip_training: Se True, pula o treinamento do modelo
        """
        print("\n" + "=" * 70)
        print("⚽ COPA LIBERTADORES 2026 - PIPELINE DE PREVISÃO ⚽")
        print("=" * 70 + "\n")
        
        # Etapa 1: Scraping (coleta de dados)
        if not skip_scraping:
            print("[1/4] 📥 Coletando dados...")
            self.scraper.run()
            print()
        else:
            print("[1/4] ⏭️  Pulando coleta de dados...\n")
        
        # Etapa 2: Pré-processamento
        print("[2/4] 🔧 Processando dados...")
        grupos_features, oitavas, quartas = self.preprocessor.run()
        print()
        
        # Etapa 3: Ajuste do modelo de Poisson
        if not skip_training:
            print("[3/4] 📐 Ajustando modelo de Poisson...")
            self.model.fit_poisson(grupos_features)
            print(f"   • Times analisados: {len(self.model.poisson.teams)}")
            print(f"   • Média de gols/jogo (liga): {self.model.poisson.league_avg:.2f}")
            print()
        else:
            print("[3/4] ⏭️  Pulando ajuste do modelo...\n")
        
        # Etapa 4: Geração de previsões
        print("[4/4] 🔮 Gerando previsões...")
        predictions = self.predictor.run()
        
        print("\n" + "=" * 70)
        print("✅ Pipeline concluído com sucesso!")
        print("=" * 70)
        print("\n📁 Arquivos gerados:")
        print("   • data/raw/ - Dados brutos coletados")
        print("   • data/processed/ - Dados processados com features")
        print("   • models/ - Modelo treinado")
        print("   • outputs/quartas_previsao.csv - Previsões das quartas")
        print("\n📓 Notebooks disponíveis:")
        print("   • notebooks/01_eda_libertadores.ipynb - Análise exploratória")
        print("   • notebooks/02_feature_engineering.ipynb - Engenharia de features")
        print("\n" + "=" * 70)
        
        return predictions


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline Libertadores 2026 - Previsão de Resultados"
    )
    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Pula a etapa de coleta de dados'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Pula a etapa de treinamento do modelo'
    )
    
    args = parser.parse_args()
    
    pipeline = LibertadoresPipeline()
    pipeline.run(
        skip_scraping=args.skip_scraping,
        skip_training=args.skip_training
    )


if __name__ == "__main__":
    main()
