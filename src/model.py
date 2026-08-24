"""
Modelo preditivo para Libertadores 2026

O classificador XGBoost (1X2) é treinado com dados históricos REAIS
(``data/historical/partidas_libertadores.csv``, edições 2012–2026), com
features causais — extraídas apenas do estado ANTERIOR a cada partida
(Elo, médias móveis de gols com shrinkage, mando/país/fase) — e avaliado
com split temporal honesto: os últimos 20% das partidas (em ordem de data)
formam o holdout out-of-sample. Sem dados sintéticos, sem vazamento
(nenhum scale/CV aleatório sobre série temporal).

A Regressão de Poisson (``poisson.py``) prevê placares e probabilidades
1X2 a partir de gols esperados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
import xgboost as xgb

from src.poisson import PoissonScoreModel

# Configurações
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class LibertadoresModel:
    """Modelo preditivo para resultados da Libertadores.

    Combina duas abordagens:
      * Classificador XGBoost (1X2) treinado com o histórico real
        2012–2026 (features causais extraídas do estado anterior a cada
        partida; holdout temporal = últimos 20% das partidas por data);
      * Modelo de Regressão de Poisson para previsão de placares e
        probabilidades 1X2 a partir de gols esperados (ver ``poisson.py``).
    """

    def __init__(self):
        self.classifier = None
        # Árvores não precisam de scaling; campo mantido apenas para
        # compatibilidade com pickles antigos (load_model usa data.get).
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        self.poisson = PoissonScoreModel()
    
    def prepare_training_data(
        self,
        csv_path=None,
        window: int = 10,
        shrink: float = 5.0,
        elo_k: float = 32.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constrói o dataset a partir do histórico real, sem vazamento.

        Itera as partidas em ordem de data; as features são extraídas do
        estado ANTERIOR a cada jogo e só então o estado é atualizado.
        Alvo: ``{"visitante": 0, "empate": 1, "mandante": 2}``.
        """
        if csv_path is None:
            csv_path = DATA_DIR / "historical" / "partidas_libertadores.csv"

        df = pd.read_csv(csv_path, parse_dates=["data"])
        df = df.dropna(subset=["gols_mandante", "gols_visitante"])
        df = df.sort_values("data").reset_index(drop=True)

        elo: Dict[str, float] = {}
        gols_pro: Dict[str, List[float]] = {}
        gols_contra: Dict[str, List[float]] = {}
        mu = 1.25
        total_gols = 0.0
        total_partidas = 0

        target_map = {"visitante": 0, "empate": 1, "mandante": 2}
        score_mandante = {"mandante": 1.0, "empate": 0.5, "visitante": 0.0}

        def media_movel(hist: List[float]) -> float:
            n = len(hist)
            return (sum(hist[-window:]) + shrink * mu) / (n + shrink) if n else mu

        rows = []
        targets = []

        for _, m in df.iterrows():
            mand, vis = m["mandante"], m["visitante"]
            gm, gv = float(m["gols_mandante"]), float(m["gols_visitante"])

            elo_m = elo.get(mand, 1500.0)
            elo_v = elo.get(vis, 1500.0)
            ataq_m = media_movel(gols_pro.get(mand, []))
            def_m = media_movel(gols_contra.get(mand, []))
            ataq_v = media_movel(gols_pro.get(vis, []))
            def_v = media_movel(gols_contra.get(vis, []))

            rows.append({
                "Diff_Elo": elo_m - elo_v,
                "Ataque_M": ataq_m,
                "Defesa_M": def_m,
                "Ataque_V": ataq_v,
                "Defesa_V": def_v,
                "Jogos_M": len(gols_pro.get(mand, [])),
                "Jogos_V": len(gols_pro.get(vis, [])),
                "Mesmo_Pais": 1 if m["pais_mandante"] == m["pais_visitante"] else 0,
                "Mata_Mata": 1 if m["fase"] == "Playoffs" else 0,
                "Diff_Ataque": ataq_m - ataq_v,
            })
            targets.append(target_map[m["resultado"]])

            esperado_m = 1.0 / (1.0 + 10.0 ** ((elo_v - elo_m) / 400.0))
            delta = elo_k * (score_mandante[m["resultado"]] - esperado_m)
            elo[mand] = elo_m + delta
            elo[vis] = elo_v - delta
            gols_pro.setdefault(mand, []).append(gm)
            gols_contra.setdefault(mand, []).append(gv)
            gols_pro.setdefault(vis, []).append(gv)
            gols_contra.setdefault(vis, []).append(gm)
            total_gols += gm + gv
            total_partidas += 1
            mu = total_gols / (2.0 * total_partidas)

        X = pd.DataFrame(rows)
        self.feature_names = X.columns.tolist()

        return X.values, np.array(targets)

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Treina o XGBoost com split temporal: os últimos ``test_size``%
        das partidas (X já ordenado por data) formam o holdout de teste."""
        print("=" * 50)
        print("Treinando modelo (split temporal)...")
        print("=" * 50)

        cut = int(len(X) * (1 - test_size))
        X_train, X_test = X[:cut], X[cut:]
        y_train, y_test = y[:cut], y[cut:]

        self.classifier = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, self.classifier.predict_proba(X_test), labels=[0, 1, 2])

        results = {
            "accuracy": accuracy,
            "log_loss": ll,
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        self.is_trained = True

        print(f"Acurácia (holdout temporal): {accuracy:.2%}")
        print(f"Log-loss (holdout temporal): {ll:.4f}")
        print("=" * 50)

        return results
    
    def predict_match(
        self, 
        match_features: pd.DataFrame
    ) -> Dict[str, float]:
        """Prevê resultado de um confronto."""
        if not self.is_trained:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        # Prepara features
        X = match_features[self.feature_names].values

        # Prediz probabilidades
        proba = self.classifier.predict_proba(X)[0]
        
        # Mapeia probabilidades
        results = {
            'prob_derrota_mandante': proba[0],  # Vitória visitante
            'prob_empate': proba[1],
            'prob_vitoria_mandante': proba[2],
            'resultado_previsto': ['Visitante', 'Empate', 'Mandante'][np.argmax(proba)]
        }
        
        return results
    
    def fit_poisson(
        self,
        grupos_df: pd.DataFrame,
        aplicar_elenco: bool = True,
    ) -> PoissonScoreModel:
        """Ajusta o modelo de Poisson com os dados da fase de grupos.

        Parameters
        ----------
        grupos_df : pd.DataFrame
            Tabela agregada da fase de grupos (colunas ``Time``, ``J``, ``GP``,
            ``GC``), tipicamente retornada por ``Preprocessor.create_features``.
        aplicar_elenco : bool
            Se True, mistura índices FBref (poder de fogo, pressão, química)
            e a forma recente dos últimos jogos nas forças de ataque/defesa.

        Returns
        -------
        PoissonScoreModel
            O modelo de Poisson ajustado.
        """
        self.poisson.fit(grupos_df)
        if aplicar_elenco:
            try:
                from src.elenco_analysis import aplicar_elenco_ao_poisson, analisar_elencos, forma_recente

                elencos = analisar_elencos()
                aplicar_elenco_ao_poisson(self.poisson, elencos, forma=forma_recente())
            except FileNotFoundError:
                pass
        return self.poisson

    def predict_match_poisson(
        self, match_features: pd.DataFrame
    ) -> Dict[str, float]:
        """Prevê probabilidades 1X2 e placar usando o modelo de Poisson.

        Requer que ``fit_poisson`` tenha sido chamado previamente.
        """
        if not self.poisson.is_fitted:
            raise RuntimeError(
                "Modelo de Poisson não ajustado. "
                "Execute fit_poisson(grupos_df) primeiro."
            )

        home = match_features["Time_Mandante"].values[0]
        away = match_features["Time_Visitante"].values[0]

        probs = self.poisson.match_probabilities(home, away)

        classes = ["Visitante", "Empate", "Mandante"]
        idx = int(np.argmax([probs["p_away"], probs["p_draw"], probs["p_home"]]))

        return {
            "prob_derrota_mandante": probs["p_away"],
            "prob_empate": probs["p_draw"],
            "prob_vitoria_mandante": probs["p_home"],
            "resultado_previsto": classes[idx],
            "gols_esperados_mandante": probs["expected_goals_home"],
            "gols_esperados_visitante": probs["expected_goals_away"],
            "placar_mais_provavel": probs["most_likely_score"],
        }

    def predict_score(self, match_features: pd.DataFrame) -> Tuple[int, int]:
        """Prevê o placar a partir dos gols esperados do modelo de Poisson.

        O placar é o arredondamento dos gols esperados (``round(lambda)``) de
        cada time. Requer que ``fit_poisson`` tenha sido chamado previamente.
        """
        if not self.poisson.is_fitted:
            raise RuntimeError(
                "Modelo de Poisson não ajustado. "
                "Execute fit_poisson(grupos_df) primeiro."
            )

        home = match_features["Time_Mandante"].values[0]
        away = match_features["Time_Visitante"].values[0]

        lam_home, lam_away = self.poisson.expected_goals(home, away)

        gols_mandante = int(round(lam_home))
        gols_visitante = int(round(lam_away))

        return gols_mandante, gols_visitante
    
    def predict_quartas(self, quartas_features: List[pd.DataFrame]) -> pd.DataFrame:
        """Prevê resultados de todos os confrontos das quartas.

        Utiliza o modelo de Poisson para probabilidades 1X2 e placar.
        Requer que ``fit_poisson`` tenha sido chamado previamente.
        """
        predictions = []
        
        for i, features in enumerate(quartas_features):
            result_probs = self.predict_match_poisson(features)
            score = self.predict_score(features)
            
            predictions.append({
                'Confronto': f"QF{i+1}",
                'Mandante': features['Time_Mandante'].values[0],
                'Visitante': features['Time_Visitante'].values[0],
                'Prob_Mandante': f"{result_probs['prob_vitoria_mandante']:.1%}",
                'Prob_Empate': f"{result_probs['prob_empate']:.1%}",
                'Prob_Visitante': f"{result_probs['prob_derrota_mandante']:.1%}",
                'Placar_Previsto': f"{score[0]}x{score[1]}",
                'Favorito': result_probs['resultado_previsto']
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath: str = None):
        """Salva modelo treinado."""
        if filepath is None:
            filepath = MODEL_DIR / "classifier.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.classifier,
                'feature_names': self.feature_names
            }, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str = None):
        """Carrega modelo treinado."""
        if filepath is None:
            filepath = MODEL_DIR / "classifier.pkl"
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.classifier = data['model']
        self.scaler = data.get('scaler')
        self.feature_names = data['feature_names']
        self.is_trained = True
        
        print(f"Modelo carregado de: {filepath}")
    
    def run(self) -> pd.DataFrame:
        """Executa pipeline completo de treinamento."""
        print("=" * 50)
        print("Pipeline de Modelagem")
        print("=" * 50)
        
        # Prepara dados
        X, y = self.prepare_training_data()
        
        # Treina
        results = self.train(X, y)
        
        # Salva modelo
        self.save_model()
        
        return results


if __name__ == "__main__":
    model = LibertadoresModel()
    results = model.run()
