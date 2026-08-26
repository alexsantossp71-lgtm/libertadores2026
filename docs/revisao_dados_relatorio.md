# Relatório de Revisão de Dados — Libertadores 2026

**Data da auditoria:** 2026-08-26
**Escopo:** `data/historical/partidas_libertadores.csv` (2012–2026) + gerador `src/real_data.py` + artefatos 2026 derivados.
**Fora de escopo:** `data/processed/*`, código de modelo (`model.py`, `poisson.py`, `predict.py`, `elenco_analysis.py`).

## 1. Objetivo

Auditar a qualidade do dataset histórico da Copa Libertadores usado pelo projeto,
cobrindo 5 dimensões: **integridade**, **completude**, **validade**, **consistência**
e **concordância entre fontes**. O resultado é um relatório estruturado reproduzível
via `python src/real_data.py review`, com orçamento de avisos testável
(`tests/test_real_data.py`).

## 2. Metodologia

A auditoria é implementada em `src/real_data.py`:

- `collect_warnings(output=...)` — captura **todos** os avisos de parsing dos arquivos
  openfootball (sem truncar nos 10 primeiros, como faz `build_dataset`).
- `review(partidas)` — orquestra 5 categorias, cada uma com `erros` / `avisos` / `info`:
  - `_cat_integridade` — reproduz `validate()` (não modificada).
  - `_cat_completude` — jogos sem placar: distingue futuro (info), em andamento (info)
    e disputado no passado (erro).
  - `_cat_validade` — domínios de gols (0–30), fase canônica, data `AAAA-MM-DD`,
    código de país de 3 letras, perna ∈ {1,2}.
  - `_cat_consistencia` — duplicatas amplas, cobertura de normalização (`is_unmapped`),
    deriva dos artefatos 2026 (`build_app_tables` vs `data/raw/*`).
  - `_cat_entre_fontes` — openfootball 2026 vs suplementos (placar por jogo).

## 3. Resultado Resumo

| Categoria | Erros | Avisos | Info |
|---|---|---|---|
| integridade (`validate`) | 0 | 24 | 0 |
| completude | 0 | 0 | 1 |
| validade | 0 | 0 | 0 |
| consistência | 0 | 0 | 0 |
| entre_fontes | 0 | 0 | 0 |
| **Total de erros** | **0** | | |

- **Partidas no dataset:** 2.227 (2012–2026, 15 temporadas)
- **Com placar:** 2.226 — **Sem placar:** 1 (Tolima×IDV, volta em 25/08/2026, em andamento)
- **Avisos de parsing openfootball:** 0

## 4. Achados por Categoria

### 4.1 Integridade (`validate`) — 0 erros, 24 avisos
Os 24 avisos são **esperados e conhecidos**: confrontos de mata-mata onde o agregado
empatou e a disputa de pênaltis **não está registrada na fonte openfootball** (vencedor
indefinido). Exemplos:

- `2013 Finals/Quarterfinals: Club Tijuana x Atlético Mineiro`
- `2013 Playoffs/Round of 16: Newell's Old Boys x Vélez Sarsfield`
- `2013 Playoffs/Round of 16: Grêmio Porto Alegre x Santa Fe`

Não há erros de integridade (gols, tabelas e jogos consistentes — validação OK).

### 4.2 Completude — 0 erros, 0 avisos, 1 info
O único item é informativo: `2026 Playoffs CAR Independiente del Valle x CD Tolima
(2026-08-25): confronto em andamento (ida/volta pendente)`. Placar ausente esperado
para jogo futuro/em andamento — correto.

### 4.3 Validade — 0 erros, 0 avisos
Todos os domínios respeitados: gols em [0,30], fases canônicas, datas `AAAA-MM-DD`,
códigos de país de 3 letras, perna ∈ {1,2}.

### 4.4 Consistência — 0 erros, 0 avisos
- Sem duplicatas na checagem ampla.
- Cobertura de normalização: 100% dos times presentes no dataset têm entrada em
  `_CANONICAL` (adicionadas 104 entradas durante esta auditoria, com proveniência
  marcada no dicionário).
- Artefatos 2026 (`grupos`, `oitavas`, `quartas`) reproduzem exatamente o rebuild
  (`build_app_tables`) — sem deriva.

### 4.5 Entre Fontes — 0 erros, 0 avisos
Openfootball 2026 concorda com os suplementos ESPN/FBref para todos os jogos com placar
(sem divergências de placar).

## 5. Correções Aplicadas (Tarefa 7)

1. **Bug de falso positivo em `_cat_consistencia`** — a checagem de cobertura de
   normalização comparava `nome_curto == mandante`, capturando times que mapeiam para
   si mesmos (ex.: `Palmeiras`→`Palmeiras`). Substituída por `is_unmapped()`, que
   verifica corretamente se `normalize_name(nome)` não está em `_CANONICAL`. Reduziu
   de 310 avisos falsos para 0.
2. **Cobertura de normalização** — adicionadas 104 entradas a `_CANONICAL` para times
   históricos não mapeados (ex.: `Grêmio Porto Alegre`→`Grêmio`, `São Paulo FC`→`São
   Paulo`, `Newell's Old Boys`→`Newell's Old Boys`). Inclui correção do apóstrofo
   (`Newell's` normaliza para `newell s`, não `newells`).
3. **Regeneração de artefatos** — `build` + `tabelas` regeneraram dataset e CSVs
   derivados com os nomes normalizados, mantendo consistência (sem deriva).

## 6. Testes de Regressão (Tarefa 8)

Adicionados 7 testes em `tests/test_real_data.py` (todos passando, 15 no total):

- `test_avisos_nao_truncados` — `collect_warnings()` devolve orçamento real (0).
- `test_cobertura_normalizacao` — nenhum time sem `_CANONICAL`.
- `test_completude_sem_inesperados` — completude só com avisos/info esperados.
- `test_validade_dominio` — domínios de gols/fase/data/país/perna respeitados.
- `test_sem_duplicatas_amplas` — sem duplicatas na checagem ampla.
- `test_concordancia_entre_fontes` — openfootball 2026 concorda com suplementos.
- `test_review_sem_erros_novos` — nenhuma categoria com erros.

## 7. Conclusão

O dataset está **íntegro e consistente** para uso no modelo. Os únicos avisos são
conhecidos (24 de integridade por penais não registrados na fonte) e 1 informativo
(jogo 2026 em andamento). Não há erros em nenhuma dimensão. A auditoria é reproduzível
e protegida por testes de regressão.
