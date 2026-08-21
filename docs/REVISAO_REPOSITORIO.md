# 🔍 Revisão Profunda do Repositório — Libertadores 2026

> Análise técnica realizada em 21/08/2026 sobre o commit `06fc61c` (branch `main`).
> Cobertura: código (`app.py`, `pages/`, `src/`, `tests/`), dados, README, configuração e metodologia.

---

## 📋 Sumário executivo

| Aspecto | Avaliação |
|---|---|
| **Qualidade de código (src/)** | 🟢 Boa — docstrings, type hints, tratamento de erro |
| **Clientes de API** | 🟢 Muito boa — rate limit, cache, retries, fallback |
| **Testes** | 🟡 Razoável — 51 testes passando, mas cobertura seletiva |
| **Integridade dos dados** | 🔴 Crítica — dados de exemplo contraditórios entre si |
| **Honestidade do README** | 🔴 Crítica — promete o que o repo não entrega |
| **Metodologia (modelagem)** | 🟡 Misto — Poisson correto como baseline; XGBoost circular |
| **Arquitetura/empacotamento** | 🟡 `sys.path` hacks, duplicação entre app.py e dashboard_utils |
| **CI/CD** | 🔴 Inexistente (badge aponta para workflow que não existe) |

**Em uma frase:** o projeto tem acabamento visual e de documentação acima da média, mas **a fundação (dados e metodologia) tem problemas sérios que um entrevistador técnico detectaria em minutos** — e que são todos corrigíveis.

---

## ✅ O que está muito bom (mantenha)

1. **`src/api_futebol_client.py` e `src/odds_client.py`** — engenharia madura: rate limit com `time.sleep(6)`, cache em disco, retries com backoff, exceções específicas (`ApiFutebolAuthError`, `BzzoiroRateLimitError`), fallback documentado. É o código mais "de produção" do repo.
2. **`src/poisson.py`** — classe limpa, docstrings com referências, validação de entrada, normalização pelo truncamento da cauda, `__repr__`. Didático e correto como *baseline*.
3. **`src/generate_example_data.py`** — determinístico (seed fixa), esquema de colunas canônico documentado, viéses de mercado (favorito-longshot, informação extra) modelados explicitamente.
4. **Dashboard (`app.py`)** — UI caprichada (CSS custom, métricas, tabs, download CSV), cache bem usado (`cache_data`/`cache_resource`), Monte Carlo do mata-mata com bracket correto.
5. **Análises estatísticas** — ANOVA, Pearson, Brier multiclasse, probabilidades implícitas normalizadas (remoção de overround). As ferramentas certas.
6. **51 testes passando** (`pytest tests/ -q` → `51 passed`), incluindo mocks dos clientes de API.
7. **Licença MIT, `.env.example`, dados de exemplo versionados** — boas práticas de repo.

---

## 🔴 Problemas críticos

### P1. Os dados de exemplo são internamente impossíveis (e alimentam o modelo)

A tabela da fase de grupos (hardcoded em `src/scraper.py`) **viola a contabilidade básica de gols**:

```
Somatório de GP (gols marcados) = 111
Somatório de GC (gols sofridos) = 64
```

Em qualquer tabela real esses dois números são **idênticos** — cada gol marcado por um time é sofrido pelo adversário. Além disso:

- A tabela diz `J = 6` para todos os 12 times, mas `data/examples/partidas_libertadores_2026.csv` dá **8 jogos por time** (36 de grupos + 12 de oitavas) e **GP/GC por time completamente diferentes** da tabela (ex.: Independiente Rivadavia tem GP=15 na tabela, mas marcou apenas 5 nas partidas; Palmeiras: 12 na tabela vs. 16 nas partidas).

**Efeito prático mensurado** no `PoissonScoreModel`:

```
league_avg (ataque médio) = 1.542
defesa média              = 0.889   → fator de deflação ≈ 0.58
```

Como `λ = ataque × defesa × vantagem / league_avg`, **todos os lambdas ficam ~42% abaixo do que seriam com uma tabela consistente**. Sintoma visível: o placar mais provável de Estudiantes × Corinthians no app é **0×0 com 40% de empate** (empates reais na Libertadores giram em torno de 25%). O modelo não está "errado por Poisson" — está errado porque a tabela de entrada é fictícia de forma inconsistente.

**Correção sugerida (raiz):** gere **uma única fonte de verdade** — o dataset de partidas — e *derive* a tabela de grupos dele por agregação (`groupby` + cálculo de Pts/V/E/D/GP/GC/SG). Assim `sum(GP) == sum(GC)` é garantido por construção, e tabela × partidas × oitavas nunca divergem. Ex.:

```python
def standings_from_matches(partidas: pd.DataFrame) -> pd.DataFrame:
    """Agrega a tabela de grupos a partir das partidas (fonte única de verdade)."""
    g = partidas[partidas["fase"] == "Fase de Grupos"]
    stats = []
    for time in set(g["mandante"]) | set(g["visitante"]):
        casa = g[g["mandante"] == time]
        fora = g[g["visitante"] == time]
        gp = casa["gols_mandante"].sum() + fora["gols_visitante"].sum()
        gc = casa["gols_visitante"].sum() + fora["gols_mandante"].sum()
        v = (casa["gols_mandante"] > casa["gols_visitante"]).sum() + \
            (fora["gols_visitante"] > fora["gols_mandante"]).sum()
        # ... E, D, Pts
        stats.append({...})
    return pd.DataFrame(stats)
```

Adicione também um **teste de invariantes** (impede regressão):

```python
def test_tabela_consistente():
    df = standings_from_matches(generate_partidas())
    assert df["GP"].sum() == df["GC"].sum()
    assert df["J"].sum() % 2 == 0
    assert ((df["V"] + df["E"] + df["D"]) == df["J"]).all()
```

### P2. O README promete coisas que o repositório não entrega

Para um projeto de **portfólio**, essa é a issues mais arriscada — recrutador confia no README e clica:

| Promessa no README | Realidade |
|---|---|
| Badge "CI Atualização diária" → `.github/workflows/update_data.yml` | **O diretório `.github/` não existe** (badge quebrada) |
| Checklist "[x] Automação: workflow do GitHub Actions executando o pipeline diariamente" | Não existe workflow |
| Estrutura lista `models/classifier.pkl # Modelo treinado (salvo)` | `models/*.pkl` está no `.gitignore`; não existe no clone |
| Estrutura lista `outputs/quartas_previsao.csv` | `outputs/` só tem `.gitkeep` |
| "Fontes: SofaScore, Flashscore…" | O scraper **não coleta nada** — retorna DataFrames hardcoded (`# TODO: Implementar scraping real`) |
| "Fase de Grupos: tabela completa" | 12 times fictícios; a Libertadores real tem 32 times em 8 grupos |
| LinkedIn "*(adicione seu link)*" | Placeholder esquecido |

O ponto da "fonte dos dados" é o mais delicado: o README dá a entender que há coleta de SofaScore/Flashscore quando não há. **Sugestão:** seção "Status honesta" no topo — `dados de exemplo sintéticos` em destaque; remova a badge de CI até criar o workflow (ou crie o workflow, que é rápido — ver roadmap).

### P3. O XGBoost de `model.py` é circular e tem vazamento de dados

`LibertadoresModel.prepare_training_data()` gera features com `np.random.uniform` e **o rótulo y é uma fórmula determinística dessas mesmas features**. O classificador "aprende" a regra que você mesmo escreveu — a acurácia reportada não mede nada sobre futebol. Pior:

```python
X_scaled = self.scaler.fit_transform(X)          # ← ajusta o scaler em TODOS os dados
cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)  # ← vazamento no CV
```

- **Vazamento**: o scaler deve ser ajustado só no fold de treino (use `Pipeline([('scaler', ...), ('clf', ...)])` no `cross_val_score`).
- **Scaling é inútil em XGBoost** (árvores são invariantes à escala monotônica).
- `use_label_encoder=False` está deprecado nas versões atuais do xgboost.
- O app **nem usa** esse classificador (usa só o Poisson) — mas o README o lista como metodologia.

**Sugestão:** ou remova o XGBoost, ou treine-o com algo real: o dataset `partidas_libertadores_2026.csv` tem 48 jogos com estatísticas por partida — dá para treinar 1X2 com validação temporal (rodadas 1–4 treino, 5–6 teste) e comparar com Poisson e odds de verdade.

### P4. Os "insights" do README são conclusões embutidas nos dados sintéticos

O README afirma: *"As odds melhoram o modelo. O mercado (58,3%) supera o Poisson (47,9%). A combinação atinge 60,4%…"*. Esses números vêm do gerador de exemplo, em que `generate_odds()` **injeta deliberadamente "informação extra" na direção do resultado real em 30% dos jogos**. Ou seja: o mercado vence porque você programou o mercado para vencer. Há um aviso ("base de exemplo… efeitos amplificados") — bom! — mas os títulos em negrito leem-se como descoberta.

Da mesma forma, o `generate_example_data` doc diz que as partidas são "coerentes com o modelo", mas o ajuste do Poisson usa a **tabela** (inconsistente — ver P1), não as partidas; a comparação modelo × mercado parte de bases divergentes.

**Sugestão:** reframe honesto e mais valioso: *"a base sintética é um testbed para a metodologia; com dados reais de edições anteriores (2018–2025) a pergunta fica respondida de verdade"*. Os próprios "Próximos Passos" já listam validação out-of-sample — promover isso de "futuro" para "essencial".

### P5. Duplicação `app.py` × `src/dashboard_utils.py`

`dashboard_utils.py` foi criado para centralizar, mas `app.py` mantém **cópias próprias** de `FLAGS`, `flag()`, `REQUIRED_FILES`, `_ensure_raw_data()`, `load_data()`/`load_grupos_features()` e `fit_model()`. Duas versões da mesma lógica = drift silencioso (já acontecem pequenas diferenças de assinatura do `fit_model`). Além disso, **4 arquivos fazem `sys.path.insert`** (`app.py`, `dashboard_utils.py`, `pages/*`, `pipeline.py`) em vez de importar um pacote.

**Sugestão:** torne o projeto instalável e importe de verdade:

```toml
# pyproject.toml
[project]
name = "libertadores2026"
...
[tool.setuptools.packages.find]
include = ["src*"]
```

```python
# app.py
from src.poisson import PoissonScoreModel
from src.dashboard_utils import fit_model, load_grupos_features, flag
```

E delete as cópias no `app.py`. Nos notebooks, `pip install -e .` resolve o import.

---

## 🟠 Problemas médios

### M1. Duas definições divergentes de "placar previsto"
- `app.py` usa `most_likely_score` (argmax da matriz) → Estudiantes × Corinthians = **0×0**
- `predict.py`/`model.py` usam `round(λ)` → mesmo jogo = **1×1**
- O README imprime **1×1** (o caminho do `predict.py`)

Padronize no argmax da matriz (estatisticamente correto) e exponha só uma função.

### M2. `except Exception: pass` no `app.py` (bloco de métricas de arbitragem/odds)
Erros reais (coluna renomeada, CSV corrompido) desaparecem silenciosamente e a página fica com métricas a menos sem ninguém saber. Capture o erro e mostre `st.info("análises opcionais indisponíveis: {e}")` em um `expander`.

### M3. Heurísticas arbitrárias sem justificativa (`preprocessing.py`)
- `pais_map = {'BRA': 3, 'ARG': 2, ...}` — codificação **ordinal** de país implica ordenação subjetiva (e Uruguai/Paraguai/Peru/Bolívia/Venezuela viram `NaN`→0). Países não têm ordem: use one-hot ou frequência alvo; ou elimine.
- `Score_Forca = 0.4·Pts + 0.3·SG + 0.2·GP + 0.1·V` — pesos mágicos misturando escalas; se precisar de um score, derive de componente principal ou Padronize antes.
- `Vitorias_Seq = V` — nome promete "sequência de vitórias", entrega total de vitórias (e o comentário admite "placeholder"). Ou implemente com dados de partidas (`groupby(rodada)`), ou renomeie.
- `combined_probabilities` com limiar de 8 p.p. **ajustável no dashboard sobre o mesmo conjunto avaliado** — overfitting interativo. Separe limiar (definido em treino) de avaliação (teste), ou use mistura contínua tipo `p_final = w·p_modelo + (1−w)·p_mercado` com `w` aprendido por verossimilhança.

### M4. Modelo Poisson: overclaim das referências
O cabeçalho cita Maher (1982) e Dixon & Coles (1997), mas o que está implementado é a **razão de médias brutas** (sem estimação por máxima verossimilhança, sem a correção τ de Dixon-Coles para placares baixos/empates, sem pesos temporais). Como *baseline* didático está ótimo — mas: (a) diga explicitamente "forma multiplicativa simplificada"; (b) se citar Dixon-Coles, implemente a τ ou o ajuste por MLE com `scipy.optimize` (é ~30 linhas e elevaria muito o nível do projeto); (c) o slider de vantagem de campo (1.00–1.50, com fator visitante `2−h` ficando 0.50) permite configurações extremas sem aviso — melhor **estimar** a vantagem nos dados (gols de mandante vs. visitante nas partidas) e mostrar o intervalo.

### M5. Configuração do Streamlit para produção
`enableXsrfProtection = false` e `enableCORS = false` servem para o preview local, mas **não deveriam ir para deploy público** ( CSRF desabilitado). Sugestão: `config.toml` seguro por padrão + override local documentado, ou variáveis de ambiente.

### M6. `predict.py` — caminho Excel quebrado e variável não definida
`save_predictions(format='excel')` chama `df.to_excel` → **`openpyxl` não está no `requirements.txt`** (ImportError em runtime). E se `format` for inválido, `filepath` é referenciado sem atribuição (`UnboundLocalError`). Adicione `openpyxl` ou remova o formato; use `else: raise ValueError`.

### M7. Cobertura de testes desigual
Testados: poisson, preprocessing, model (integração), clientes, gerador. **Sem testes:** `scraper.py` (poderia validar invariantes da tabela — pegaria o P1!), `pipeline.py`, `predict.py`, `dashboard_utils.py`, e as páginas (Smoke test com `streamlit AppTest` é barato: `st.testing.v1.AppTest.from_file("app.py").run()`). Sem `pytest --cov`, sem limiar de cobertura. `conftest.py` vazio.

### M8. Detalhes de UX no dashboard
- Páginas nomeadas `5_Arbitragem` e `6_Odds` sem 1–4: como existe `pages/`, o Streamlit gera navegação automática **e** a sidebar manual com `st.page_link` — navegação duplicada. Ou migre para `st.navigation`/`st.Page` (moderno, um só lugar), ou renomeie.
- Monte Carlo refaz o `fit` dentro da função cacheada em vez de reaproveitar o `fit_model` — se os parâmetros divergirem no futuro, bug sutil.
- No simulador, o padrão `index=0` do visitante escolhe o primeiro time — prefira o 2º colocado ou o adversário real das quartas.

### M9. Requisitos e notebooks
- `requirements.txt` mistura runtime e dev (`jupyter`, `pytest`) — separe `requirements.txt` / `requirements-dev.txt` (deploy no Streamlit Community Cloud fica mais leve e rápido).
- `notebooks/01` e `02` estão **sem outputs salvos** (parecem não executados); numeração pula 03/04. Execute "Restart & Run All", salve, e renumere.

---

## 🟢 Sugestões de melhoria (roadmap priorizado)

### Quick wins (1–2 dias)
1. **P1:** derivar a tabela de grupos das partidas + teste de invariantes (`sum(GP)==sum(GC)`). *Alta alavancagem: conserte o gerador e todos os números do README/dashboard mudam de fictícios-inconsistentes para fictícios-coerentes.*
2. **P2:** remover/corrigir badge de CI, checklist, estrutura de arquivos e placeholders do README; deixar explícito o que é sintético.
3. **M6:** `openpyxl` no requirements ou remover formato excel.
4. **M1:** unificar "placar previsto" no argmax da matriz.
5. **M2:** substituir `except: pass` por aviso visível.
6. Executar e salvar os notebooks 01/02.

### Médio prazo (1–2 semanas)
7. **CI real:** workflow GitHub Actions com `pytest` + `ruff` em PR, e o tal "update_data.yml" diário (existe API real de odds/estatísticas já integrada — é só chamar o pipeline e commitar `data/processed/`). Badge volta a ser verdade.
8. **P5:** `pyproject.toml` + imports `from src...`; matar os `sys.path` hacks e a duplicação app×utils.
9. **P3:** XGBoost com dados reais/históricos + validação temporal, ou remoção honesta. Corrigir vazamento do scaler com `sklearn.pipeline.Pipeline`.
10. **M4:** Poisson com MLE + τ de Dixon-Coles (diferencial real no portfólio) e vantagem de campo estimada dos dados.
11. **M3:** one-hot para país, remover `Score_Forca`/`Vitorias_Seq` ou implementar de verdade; validação do limiar da combinação em split temporal.
12. **Dados reais:** a API Futebol já tem cliente pronto — puxar a Libertadores 2025 real (32 times) para validar a metodologia e repor os "insights" por números verdadeiros.

### Longo prazo
13. Backtesting em edições anteriores (2018–2025): log-loss/Brier do Poisson vs. odds de fechamento — a pergunta "o mercado bate o modelo?" respondida com dados reais vale ouro no portfólio.
14. Modelo bayesiano hierárquico (forças de ataque/defesa com priors por país) via `pymc` — evolução natural do Maher.
15. Calibração de probabilidades (reliability curves, isotonic regression) — barato e impressiona.

---

## 🎯 Checklist "antes de mostrar a um recrutador"

- [ ] Tabela de grupos derivada das partidas (consistência garantida)
- [ ] README sem promessas quebradas (badge CI, arquivos listados, fontes)
- [ ] Disclaimer de dados sintéticos visível na primeira dobra do README **e** no topo do dashboard
- [ ] XGBoost removido ou treinado em dados reais com validação sem vazamento
- [ ] Um só "placar previsto" no código inteiro
- [ ] CI verde (pytest) + lint configurado
- [ ] Notebooks executados com outputs
- [ ] LinkedIn/email preenchidos

---

*Documento gerado como parte da revisão do repositório; nenhum arquivo de código foi alterado nesta análise.*
