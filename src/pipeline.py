"""
Pipeline principal do projeto Libertadores 2026
Executa todo o fluxo: coleta -> pré-processamento -> modelagem -> previsões

Etapas:
  1. Scraping (tabelas da fase de grupos, oitavas e quartas);
  2. Estatísticas detalhadas (API Futebol — faltas, cartões, posse, passes,
     finalizações, escanteios, impedimentos, defesas e árbitro);
  3. Odds 1X2 (Bzzoiro Sports Data — com probabilidades implícitas);
  4. Pré-processamento (features + processamento dos novos dados);
  5. Ajuste do modelo de Poisson;
  6. Geração de previsões.

Fallback: sem as chaves ``API_FUTEBOL_KEY``/``BSD_API`` (ou diante de falhas
de API), as etapas 2 e 3 usam as bases de exemplo de ``data/examples/``, de
modo que o pipeline roda do início ao fim sem erros.
"""

import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

from scraper import LibertadoresScraper
from preprocessing import Preprocessor
from model import LibertadoresModel
from predict import LibertadoresPredictor
from api_futebol_client import ApiFutebolClient
from odds_client import BzzoiroOddsClient


class LibertadoresPipeline:
    """Pipeline completo para previsão da Libertadores 2026."""

    def __init__(self):
        self.scraper = LibertadoresScraper()
        self.preprocessor = Preprocessor()
        self.model = LibertadoresModel()
        self.predictor = LibertadoresPredictor()
        self.futebol_client = ApiFutebolClient()
        self.odds_client = BzzoiroOddsClient()

    def run(
        self,
        skip_scraping: bool = False,
        skip_training: bool = False,
        skip_stats: bool = False,
        skip_odds: bool = False,
    ):
        """
        Executa pipeline completo.

        Args:
            skip_scraping: Se True, pula a coleta de dados
            skip_training: Se True, pula o treinamento do modelo
            skip_stats: Se True, pula a coleta de estatísticas (API Futebol)
            skip_odds: Se True, pula a coleta de odds (Bzzoiro)
        """
        print("\n" + "=" * 70)
        print("⚽ COPA LIBERTADORES 2026 - PIPELINE DE PREVISÃO ⚽")
        print("=" * 70 + "\n")

        # Etapa 1: Scraping (coleta de dados)
        if not skip_scraping:
            print("[1/6] 📥 Coletando dados...")
            self.scraper.run()
            print()
        else:
            print("[1/6] ⏭️  Pulando coleta de dados...\n")

        # Etapa 2: Estatísticas detalhadas (API Futebol)
        if not skip_stats:
            print("[2/6] 🧾 Coletando estatísticas detalhadas (API Futebol)...")
            stats_df = self.futebol_client.run()
            print()
        else:
            print("[2/6] ⏭️  Pulando coleta de estatísticas...\n")
            stats_df = self.preprocessor.load_estatisticas()

        # Etapa 3: Odds (Bzzoiro)
        if not skip_odds:
            print("[3/6] 🎰 Coletando odds 1X2 (Bzzoiro Sports Data)...")
            odds_df = self.odds_client.run(partidas=self.preprocessor.load_estatisticas())
            print()
        else:
            print("[3/6] ⏭️  Pulando coleta de odds...\n")
            odds_df = self.preprocessor.load_odds()

        # Etapa 4: Pré-processamento
        print("[4/6] 🔧 Processando dados...")
        grupos_features, oitavas, quartas = self.preprocessor.run()
        print()

        # Etapa 5: Ajuste do modelo de Poisson
        if not skip_training:
            print("[5/6] 📐 Ajustando modelo de Poisson...")
            self.model.fit_poisson(grupos_features)
            print(f"   • Times analisados: {len(self.model.poisson.teams)}")
            print(f"   • Média de gols/jogo (liga): {self.model.poisson.league_avg:.2f}")
            print()
        else:
            print("[5/6] ⏭️  Pulando ajuste do modelo...\n")

        # Etapa 6: Geração de previsões
        print("[6/6] 🔮 Gerando previsões...")
        predictions = self.predictor.run()

        # Resumo final
        print("\n" + "=" * 70)
        print("✅ Pipeline concluído com sucesso!")
        print("=" * 70)
        print("\n📁 Arquivos gerados:")
        print("   • data/raw/ - Dados brutos coletados (incl. estatísticas e odds)")
        print("   • data/processed/ - Dados processados com features")
        print("   • data/processed/libertadores_estatisticas_detalhadas.csv")
        print("   • data/processed/libertadores_odds.csv")
        print("   • models/ - Modelo treinado")
        print("   • outputs/quartas_previsao.csv - Previsões das quartas")
        if not stats_df.empty:
            print(f"\n🧾 Estatísticas: {len(stats_df)} partidas com dados detalhados")
        if not odds_df.empty:
            print(f"🎰 Odds: {len(odds_df)} partidas com odds 1X2")
        print("\n📓 Notebooks disponíveis:")
        print("   • notebooks/01_eda_libertadores.ipynb - Análise exploratória")
        print("   • notebooks/02_feature_engineering.ipynb - Engenharia de features")
        print("   • notebooks/05_analise_arbitragem_odds.ipynb - Arbitragem e odds")
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
        help='Pula a etapa de coleta de dados (scraper)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Pula a etapa de treinamento do modelo'
    )
    parser.add_argument(
        '--skip-stats',
        action='store_true',
        help='Pula a coleta de estatísticas da API Futebol'
    )
    parser.add_argument(
        '--skip-odds',
        action='store_true',
        help='Pula a coleta de odds da Bzzoiro'
    )

    args = parser.parse_args()

    pipeline = LibertadoresPipeline()
    pipeline.run(
        skip_scraping=args.skip_scraping,
        skip_training=args.skip_training,
        skip_stats=args.skip_stats,
        skip_odds=args.skip_odds,
    )


if __name__ == "__main__":
    main()
