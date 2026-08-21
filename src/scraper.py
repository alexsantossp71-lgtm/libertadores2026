"""
Coleta de dados REAIS da Copa Libertadores (2012–2026).

Este módulo não gera mais dados fictícios de exemplo. Ele monta as tabelas
consumidas pelo dashboard a partir do dataset histórico versionado em
``data/historical/partidas_libertadores.csv`` (openfootball + suplementos
ESPN/FBref), produzido por ``src/real_data.py``::

    python src/real_data.py build     # reconstrói o dataset histórico
    python src/real_data.py tabelas   # regenera data/raw/*.csv deste módulo

Se o dataset ainda não existir, o build roda automaticamente (fontes em
``data/historical/openfootball/`` — nenhum dado simulado é criado).
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent

if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from real_data import build_app_tables, build_dataset, load_partidas  # noqa: E402


class LibertadoresScraper:
    """Carrega dados reais e materializa as tabelas do dashboard.

    Mantém a interface ``run()`` usada por ``app.py`` e ``pipeline.py``,
    mas agora tudo vem de fontes reais versionadas no repositório.
    """

    def run(self):
        try:
            load_partidas()
        except FileNotFoundError:
            print("Dataset histórico não encontrado — construindo a partir das fontes…")
            build_dataset()
        tabelas = build_app_tables()
        print(
            "✅ Dados reais carregados: "
            f"{len(tabelas['grupos'])} times na fase de grupos, "
            f"{len(tabelas['oitavas'])} confrontos de oitavas, "
            f"{len(tabelas['quartas'])} confrontos de quartas."
        )
        return tabelas["grupos"], tabelas["oitavas"], tabelas["quartas"]


if __name__ == "__main__":
    scraper = LibertadoresScraper()
    scraper.run()
