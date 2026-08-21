# ⚽ Libertadores 2026 - Previsão de Resultados

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-orange)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-yellow)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Objetivo do Projeto

Este projeto tem como objetivo **prever os resultados dos jogos da Copa Libertadores da América de 2026** utilizando dados estatísticos e históricos. O foco principal está nas **quartas de final**, mas a metodologia pode ser aplicada a qualquer fase do torneio.

O projeto foi desenvolvido como parte do meu portfólio para demonstrar habilidades em:
- Coleta e limpeza de dados esportivos
- Análise exploratória (EDA)
- Engenharia de features
- Modelagem preditiva (Classificação e Regressão)
- Documentação e boas práticas de Ciência de Dados

---

## 📊 Dados Utilizados

### Fonte dos Dados
Os dados foram coletados de fontes públicas confiáveis, incluindo:
- [SofaScore](https://www.sofascore.com/) – Estatísticas de jogos
- [Flashscore](https://www.flashscore.com.br/) – Resultados e tabelas
- [Football-Data.org](https://www.football-data.org/) – API para dados históricos (em desenvolvimento)

### Dados Atuais (Libertadores 2026)
- **Fase de Grupos**: Tabela completa com pontos, gols, saldo, etc.
- **Oitavas de Final**: Resultados de ida e volta, classificados.
- **Quartas de Final**: Confrontos definidos (datas: 09/09 e 16/09).

### Estrutura dos Dados
```
data/
├── raw/
│   ├── grupos_libertadores_2026.csv
│   ├── oitavas_resultados.csv
│   └── confrontos_quartas.csv
├── processed/
│   └── features_libertadores.csv   # Dados com engenharia de features
└── external/
    └── elo_rankings.csv            # Ranking Elo dos times (futuro)
```

---

## 🔍 Análise Exploratória (Insights Iniciais)

Com base nos dados da fase de grupos e oitavas, já podemos extrair insights importantes:

### 📈 Desempenho dos Times na Fase de Grupos
- **Melhor campanha**: **Flamengo** (BRA) – 16 pontos, saldo +12, 5 vitórias.
- **Melhor ataque**: **Independiente Rivadavia** (ARG) – 15 gols em 6 jogos.
- **Melhor defesa**: **Flamengo** (BRA) – apenas 2 gols sofridos.
- **Média de gols por jogo**: **2.4** gols/partida.

### 🏆 Oitavas de Final – Destaques
- **Brasil domina**: 4 times brasileiros nas quartas (Flamengo, Palmeiras, Corinthians, Fluminense).
- **Disputas equilibradas**: 3 confrontos foram decididos nos pênaltis (Fluminense, Platense, LDU).
- **Flamengo eliminou o Cruzeiro** em um confronto brasileiro com placar agregado de 3x2.

### 📊 Gráficos (a serem gerados)
- Distribuição de gols por time.
- Comparação de ataques e defesas.
- Probabilidade de classificação por país.

---

## ⚙️ Metodologia de Previsão

### Abordagem em 3 Camadas

1. **Dados Históricos (Força Bruta)**
   - Coleta das últimas 5 edições da Libertadores (2021–2025) para treinar um modelo de classificação.
   - Variáveis: gols marcados/sofridos, posse de bola, finalizações, etc.

2. **Contexto Atual (2026)**
   - Utilização dos dados da fase de grupos e oitavas para capturar o momento atual de cada time.
   - Features: pontuação, saldo de gols, aproveitamento em casa/fora.

3. **Ranking Elo (Fator Curinga)**
   - Cálculo do **Elo Rating** de cada time com base nos últimos 10 jogos oficiais (nacionais e internacionais).
   - Isso captura a *fase real* do time, independentemente do histórico antigo.

### Modelos Utilizados

| Tarefa | Modelo | Bibliotecas |
|--------|--------|-------------|
| Prever resultado (V/E/D) | **XGBoost Classifier** | `xgboost`, `sklearn` |
| Prever placar exato | **Regressão de Poisson** | `statsmodels`, `scipy` |
| Avaliar importância das features | **Feature Importance** | `matplotlib`, `seaborn` |

### Métricas de Avaliação
- **Acurácia** e **Matriz de Confusão** para classificação.
- **Erro Absoluto Médio (MAE)** para previsão de gols.
- **Backtesting** com os dados das oitavas de final (validação histórica).

---

## 📈 Previsões para as Quartas de Final (09/09)

Com base nos dados atuais e na análise estatística preliminar, as probabilidades para os confrontos são:

| Jogo | Mandante | Visitante | Prob. Mandante | Prob. Empate | Prob. Visitante | Placar Mais Provável |
|------|----------|-----------|----------------|--------------|-----------------|----------------------|
| QF1 | **Estudiantes** (ARG) | **Corinthians** (BRA) | 32% | 38% | 30% | 1x1 |
| QF2 | **Independiente del Valle** (ECU) | **Flamengo** (BRA) | 22% | 25% | **53%** | 0x2 |
| QF3 | **Palmeiras** (BRA) | **LDU** (ECU) | **55%** | 28% | 17% | 2x0 |
| QF4 | **Fluminense** (BRA) | **Platense** (ARG) | 48% | 30% | 22% | 2x1 |

> ⚠️ **Observação**: Estas previsões são baseadas em dados estatísticos e não consideram fatores imprevistos como lesões, suspensões ou mudanças táticas de última hora. O modelo será atualizado conforme novos dados forem disponibilizados.

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
- Python 3.9 ou superior
- Pip (gerenciador de pacotes)

### Passos

1. **Clone o repositório**
   ```bash
   git clone https://github.com/alexsantossp71-lgtm/libertadores2026.git
   cd libertadores2026
   ```

2. **Crie um ambiente virtual** (opcional, mas recomendado)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o pipeline completo**
   ```bash
   python src/pipeline.py
   ```
   Isso irá:
   - Baixar os dados atualizados (se configurado)
   - Processar e limpar os dados
   - Treinar o modelo
   - Gerar as previsões para as quartas
   - Salvar os resultados em `outputs/`

5. **Explore os notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   Abra `01_eda_libertadores.ipynb` para ver a análise exploratória detalhada.

---

## 📁 Estrutura do Repositório

```
libertadores2026/
├── data/
│   ├── raw/               # Dados brutos (CSVs)
│   ├── processed/         # Dados limpos e com features
│   └── external/          # Dados externos (rankings, etc.)
├── notebooks/
│   ├── 01_eda_libertadores.ipynb          # Análise exploratória
│   └── 02_feature_engineering.ipynb       # Criação de features
├── src/
│   ├── scraper.py         # Coleta de dados (Web Scraping/API)
│   ├── preprocessing.py   # Limpeza e transformação
│   ├── model.py           # Treino e avaliação dos modelos
│   └── predict.py         # Geração de previsões
├── models/
│   └── classifier.pkl     # Modelo treinado (salvo)
├── outputs/
│   ├── quartas_previsao.csv   # Tabela com probabilidades
│   └── charts/                # Gráficos gerados
├── requirements.txt       # Dependências do projeto
├── .gitignore
└── README.md              # Este arquivo
```

---

## 🚀 Próximos Passos (Melhorias Futuras)

- [ ] **Dashboard Interativo**: Criar um aplicativo com **Streamlit** para visualizar previsões em tempo real.
- [ ] **Dados de Odds**: Incluir odds de casas de apostas como variável externa (feature).
- [ ] **Automação**: Agendar a execução diária para atualizar dados e previsões.
- [ ] **Mais Features**: Adicionar estatísticas de jogadores (gols, assistências, cartões).
- [ ] **Modelo de Redes Neurais**: Experimentar LSTM para séries temporais de desempenho.

---

## 📚 Sobre o Autor

**Alex Santos**  
Engenheiro de Computação | Cientista de Dados em formação  

Apaixonado por esportes e análise de dados, busco unir minha formação técnica com a capacidade de extrair insights valiosos de dados complexos. Este projeto é uma amostra do meu trabalho na área de Ciência de Dados aplicada a esportes.

- 🔗 [GitHub](https://github.com/alexsantossp71)
- 🔗 [LinkedIn](https://linkedin.com/in/alexsantossp71) *(adicione seu link)*
- 📧 alexsantossp71@gmail.com *(adicione seu email)*

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

⭐ **Se gostou deste projeto, deixe uma estrela no repositório!** Isso me ajuda a continuar desenvolvendo e compartilhando conhecimento.
