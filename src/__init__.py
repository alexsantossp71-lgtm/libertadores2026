"""
Libertadores 2026 - Módulo de Previsão de Resultados
"""

__version__ = "1.0.0"
__author__ = "Alex Santos"

from .scraper import LibertadoresScraper
from .preprocessing import Preprocessor
from .model import LibertadoresModel
from .predict import LibertadoresPredictor
from .fbref_scraper import FBrefClient

__all__ = [
    'LibertadoresScraper',
    'Preprocessor',
    'LibertadoresModel',
    'LibertadoresPredictor',
    'FBrefClient',
]
