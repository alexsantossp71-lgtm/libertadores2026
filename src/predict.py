"""
Geração de previsões das Quartas de Final com análise de elenco.

O Poisson da fase de grupos continua sendo o núcleo. Em cima dele entram:

* índices FBref (poder de fogo, pressão defensiva, disciplina);
* química de elenco (11 / nº de jogadores usados — proxy de repetição);
* forma recente dos últimos 5 jogos (openfootball).

Os CSVs de auditoria guardam o cenário-base e o cenário com elenco lado a lado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from elenco_analysis import (
    aplicar_elenco_ao_poisson,
    persistir_analise,
    prever_confronto_com_elenco,
    run as run_analise,
)
from model import LibertadoresModel
from preprocessing import Preprocessor

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# O QF3 ainda sem adversário definido vira dois cenários.
QF3_CANDIDATOS = ("Tolima", "Independiente del Valle")
QF3_PAIS = {"Tolima": "COL", "Independiente del Valle": "ECU"}


class LibertadoresPredictor:
    """Gerador de previsões para a Libertadores 2026."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = LibertadoresModel()
        self.elencos: pd.DataFrame = pd.DataFrame()

    def _resolver_quartas(self, quartas: pd.DataFrame, times_validos: set) -> pd.DataFrame:
        """Expande o QF3 indefinido nos dois cenários reais (Tolima / IDV)."""
        rows = []
        for _, row in quartas.iterrows():
            mandante = row["Mandante"]
            visitante = row["Visitante"]
            if mandante in times_validos and visitante in times_validos:
                rec = row.to_dict()
                rec["Cenario"] = "definido"
                rows.append(rec)
                continue
            if mandante in times_validos and "A DEFINIR" in str(visitante).upper():
                for cand in QF3_CANDIDATOS:
                    if cand not in times_validos:
                        continue
                    rec = row.to_dict()
                    rec["Visitante"] = cand
                    rec["Pais_Visitante"] = QF3_PAIS.get(cand, rec.get("Pais_Visitante"))
                    rec["Cenario"] = f"se {cand}"
                    rec["Confronto"] = f"{row['Confronto']} ({cand})"
                    rows.append(rec)
        return pd.DataFrame(rows)

    def generate_quartas_predictions(self) -> pd.DataFrame:
        """Gera previsões das quartas com Poisson ajustado pelo elenco."""
        print("=" * 50)
        print("Gerando Previsões - Quartas de Final (com elenco)")
        print("=" * 50)

        grupos, _, quartas = self.preprocessor.load_data()
        grupos_features = self.preprocessor.create_features(grupos, incluir_elenco=True)

        print("Analisando elencos (FBref + forma recente)...")
        analise = run_analise(persist=True)
        self.elencos = analise["elencos"]

        print("Ajustando Poisson da fase de grupos...")
        self.model.fit_poisson(grupos_features, aplicar_elenco=False)
        aplicar_elenco_ao_poisson(
            self.model.poisson, self.elencos, forma=analise["forma"]
        )

        times_validos = set(self.model.poisson.teams)
        confrontos_df = self._resolver_quartas(quartas, times_validos)

        confrontos: List[Dict] = []
        for _, row in confrontos_df.iterrows():
            mandante = row["Mandante"]
            visitante = row["Visitante"]

            match_features = self.preprocessor.create_match_features(
                grupos_features, mandante, visitante
            )
            result_probs = self.model.predict_match_poisson(match_features)
            score = self.model.predict_score(match_features)
            detalhe = prever_confronto_com_elenco(
                self.model.poisson, mandante, visitante, self.elencos
            )

            confrontos.append({
                "Confronto": row.get("Confronto"),
                "Cenario": row.get("Cenario", "definido"),
                "Mandante": mandante,
                "Visitante": visitante,
                "Pais_Mandante": row.get("Pais_Mandante"),
                "Pais_Visitante": row.get("Pais_Visitante"),
                "Data_Ida": row.get("Data_Ida"),
                "Data_Volta": row.get("Data_Volta"),
                "Prob_Mandante": result_probs["prob_vitoria_mandante"],
                "Prob_Empate": result_probs["prob_empate"],
                "Prob_Visitante": result_probs["prob_derrota_mandante"],
                "Prob_Mandante_base": detalhe.get("prob_mandante_base"),
                "Prob_Empate_base": detalhe.get("prob_empate_base"),
                "Prob_Visitante_base": detalhe.get("prob_visitante_base"),
                "Gols_Esperados_Mandante": result_probs["gols_esperados_mandante"],
                "Gols_Esperados_Visitante": result_probs["gols_esperados_visitante"],
                "Gols_Esperados_Mandante_base": detalhe.get("xg_mandante_base"),
                "Gols_Esperados_Visitante_base": detalhe.get("xg_visitante_base"),
                "Delta_xG_Mandante": detalhe.get("delta_xg_mandante"),
                "Delta_xG_Visitante": detalhe.get("delta_xg_visitante"),
                "Placar_Previsto": f"{score[0]}x{score[1]}",
                "Favorito": result_probs["resultado_previsto"],
                "Indice_Ofensivo_Mandante": detalhe.get("indice_ofensivo_mandante"),
                "Indice_Ofensivo_Visitante": detalhe.get("indice_ofensivo_visitante"),
                "Indice_Pressao_Mandante": detalhe.get("indice_pressao_mandante"),
                "Indice_Pressao_Visitante": detalhe.get("indice_pressao_visitante"),
                "Quimica_Mandante": detalhe.get("quimica_mandante"),
                "Quimica_Visitante": detalhe.get("quimica_visitante"),
                "Risco_Suspensao_Mandante": detalhe.get("risco_suspensao_mandante"),
                "Risco_Suspensao_Visitante": detalhe.get("risco_suspensao_visitante"),
                "Score_Elenco_Mandante": detalhe.get("score_elenco_mandante"),
                "Score_Elenco_Visitante": detalhe.get("score_elenco_visitante"),
                "Nota_Elenco": detalhe.get("nota_elenco"),
            })

        df_predictions = pd.DataFrame(confrontos)
        persistir_analise(
            self.elencos,
            analise.get("jogadores"),
            analise.get("forma"),
            analise.get("influencia"),
            confrontos=df_predictions,
        )
        return df_predictions

    def save_predictions(self, df: pd.DataFrame, format: str = "csv"):
        """Salva previsões em arquivo."""
        if format == "csv":
            filepath = OUTPUT_DIR / "quartas_previsao.csv"
            df.to_csv(filepath, index=False)
        elif format == "excel":
            filepath = OUTPUT_DIR / "quartas_previsao.xlsx"
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Formato não suportado: {format}")
        print(f"Previsões salvas em: {filepath}")

    def print_predictions(self, df: pd.DataFrame):
        """Imprime previsões formatadas."""
        print("\n" + "=" * 80)
        print("⚽ PREVISÕES - QUARTAS DE FINAL DA LIBERTADORES 2026")
        print("   Poisson da fase de grupos × análise de elenco (FBref)")
        print("=" * 80)

        for _, row in df.iterrows():
            cenario = f"  [{row['Cenario']}]" if row.get("Cenario") and row["Cenario"] != "definido" else ""
            print(f"\n{row['Confronto']}{cenario}: {row['Mandante']} ({row.get('Pais_Mandante','')}) x "
                  f"({row.get('Pais_Visitante','')}) {row['Visitante']}")
            print(f"   📅 Ida: {row.get('Data_Ida')} | Volta: {row.get('Data_Volta')}")
            print("   📊 Probabilidades (com elenco):")
            print(f"      {row['Mandante']}: {row['Prob_Mandante']:.1%}")
            print(f"      Empate: {row['Prob_Empate']:.1%}")
            print(f"      {row['Visitante']}: {row['Prob_Visitante']:.1%}")
            if pd.notna(row.get("Prob_Mandante_base")):
                print("   📐 Sem elenco (só grupos): "
                      f"{row['Prob_Mandante_base']:.1%} / "
                      f"{row['Prob_Empate_base']:.1%} / "
                      f"{row['Prob_Visitante_base']:.1%}")
            print("   🥅 Gols esperados: "
                  f"{row['Gols_Esperados_Mandante']:.2f} x "
                  f"{row['Gols_Esperados_Visitante']:.2f}")
            print(f"   🔮 Placar previsto: {row['Placar_Previsto']}  ·  ⭐ Favorito: {row['Favorito']}")
            if row.get("Nota_Elenco"):
                print(f"   👕 Elenco: {row['Nota_Elenco']}")

        print("\n" + "=" * 80)
        print("⚠️  Aviso: o ajuste de elenco usa estatísticas reais da FBref")
        print("   (finalizações, desarmes, interceptações, cartões, rotação)")
        print("   e a forma dos últimos 5 jogos. Lesões pontuais e XI do dia")
        print("   só entram quando a tabela de jogadores estiver raspada.")
        print("=" * 80)

    def run(self) -> pd.DataFrame:
        """Executa pipeline completo de previsões."""
        predictions = self.generate_quartas_predictions()
        self.print_predictions(predictions)
        self.save_predictions(predictions)
        return predictions


if __name__ == "__main__":
    LibertadoresPredictor().run()
