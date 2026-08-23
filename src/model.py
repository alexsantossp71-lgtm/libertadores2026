"""
Modelo preditivo para Libertadores 2026
Utiliza XGBoost para classificação e Regressão de Poisson para previsão de placares
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from poisson import PoissonScoreModel

# Configurações
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class LibertadoresModel:
    """Modelo preditivo para resultados da Libertadores.

    Combina duas abordagens:
      * Classificador XGBoost (1X2) treinado sobre features de confronto;
      * Modelo de Regressão de Poisson para previsão de placares e
        probabilidades 1X2 a partir de gols esperados (ver ``poisson.py``).
    """
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.poisson = PoissonScoreModel()
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara dados históricos para treinamento.
        Em produção, isso carregaria dados das edições anteriores.
        """
        # Dados simulados de treinamento (últimas edições)
        np.random.seed(42)
        
        n_samples = 200
        
        # Features simuladas
        X = pd.DataFrame({
            'Diff_Pts': np.random.uniform(-10, 10, n_samples),
            'Diff_GP': np.random.uniform(-10, 10, n_samples),
            'Diff_GC': np.random.uniform(-10, 10, n_samples),
            'Diff_SG': np.random.uniform(-15, 15, n_samples),
            'Diff_Aproveitamento': np.random.uniform(-0.5, 0.5, n_samples),
            'Diff_Media_Gols': np.random.uniform(-2, 2, n_samples),
            'Diff_Score_Forca': np.random.uniform(-50, 50, n_samples),
            'Razao_Pts': np.random.uniform(0.5, 1.5, n_samples),
            'Razao_GP': np.random.uniform(0.5, 1.5, n_samples),
            'Razao_Score': np.random.uniform(0.5, 1.5, n_samples),
            'Pais_Mandante_Cod': np.random.choice([1, 2, 3], n_samples),
            'Pais_Visitante_Cod': np.random.choice([1, 2, 3], n_samples),
            'Mesmo_Pais': np.random.choice([0, 1], n_samples),
        })
        
        # Resultado simulado (0=Derrota, 1=Empate, 2=Vitória do mandante)
        y_probs = []
        for _, row in X.iterrows():
            # Probabilidades baseadas nas features
            prob_mandante = 0.35 + row['Diff_Aproveitamento'] * 0.3 + row['Pais_Mandante_Cod'] * 0.02
            prob_empate = 0.30 - abs(row['Diff_Aproveitamento']) * 0.2
            prob_visitante = 1 - prob_mandante - prob_empate
            
            probs = [prob_visitante, prob_empate, prob_mandante]
            probs = np.clip(probs, 0, 1)
            probs = probs / sum(probs)
            
            result = np.random.choice([0, 1, 2], p=probs)
            y_probs.append(result)
        
        y = np.array(y_probs)
        
        self.feature_names = X.columns.tolist()
        
        return X.values, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Treina o modelo XGBoost."""
        print("=" * 50)
        print("Treinando modelo...")
        print("=" * 50)
        
        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normaliza features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treina XGBoost
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Avalia
        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        X_scaled = self.scaler.fit_transform(X)
        cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.is_trained = True
        
        print(f"Acurácia: {accuracy:.2%}")
        print(f"Cross-Validation: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
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
        X_scaled = self.scaler.transform(X)
        
        # Prediz probabilidades
        proba = self.classifier.predict_proba(X_scaled)[0]
        
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
                from elenco_analysis import aplicar_elenco_ao_poisson, analisar_elencos, forma_recente

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
                'scaler': self.scaler,
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
        self.scaler = data['scaler']
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
