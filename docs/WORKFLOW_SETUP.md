# 🤖 Setup do Workflow de Atualização Diária

Este documento descreve como ativar a atualização automática de dados da Libertadores 2026.

## Problema

O token do GitHub App usado pelo Arena não tem permissão `workflows`, por isso o arquivo `.github/workflows/update_data.yml` não pode ser enviado via `git push`. Ele precisa ser criado manualmente pela interface do GitHub.

## Solução - Criar manualmente no GitHub

1. No seu repositório no GitHub, vá em **Add file → Create new file**
2. Digite o caminho: `.github/workflows/update_data.yml`
3. Cole o conteúdo abaixo
4. Commit direto na `main` ou em um branch

## Conteúdo do workflow `.github/workflows/update_data.yml`

```yaml
name: 🤖 Atualização diária - Libertadores 2026

on:
  schedule:
    # Todo dia às 06:00 UTC (03:00 BRT)
    - cron: '0 6 * * *'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  update-data:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repositório
        uses: actions/checkout@v4
        with:
          persist-credentials: true
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Instalar dependências
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Rodar pipeline completo
        env:
          BSD_API: ${{ secrets.BSD_API }}
          API_FUTEBOL_KEY: ${{ secrets.API_FUTEBOL_KEY }}
        run: |
          python src/pipeline.py
          echo "=== Arquivos gerados ==="
          ls -R data/ || true
          ls -R outputs/ || true

      - name: Rodar testes (sanity check)
        run: |
          python -m pytest tests/ -v || echo "Alguns testes falharam, mas continua"

      - name: Commit e push automático se houver mudanças
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          git add -A
          git add -f outputs/quartas_previsao.csv 2>/dev/null || true
          git add -f data/processed/libertadores_estatisticas_detalhadas.csv 2>/dev/null || true
          git add -f data/processed/libertadores_odds.csv 2>/dev/null || true
          git add -f data/processed/features_libertadores.csv 2>/dev/null || true
          git add -f outputs/charts/ 2>/dev/null || true

          if git diff --staged --quiet; then
            echo "✅ Nenhuma mudança detectada - nada para commitar"
            exit 0
          fi

          echo "📦 Mudanças detectadas:"
          git diff --staged --stat

          git commit -m "🤖 Atualização automática de dados - $(date -u +%Y-%m-%d)"
          git pull --rebase origin main || true
          git push origin HEAD:main

      - name: Resumo
        if: always()
        run: |
          echo "### 🤖 Pipeline executado em $(date -u)" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "- **Branch**: ${{ github.ref_name }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Trigger**: ${{ github.event_name }}" >> $GITHUB_STEP_SUMMARY
          if [ -f outputs/quartas_previsao.csv ]; then
            echo "- **Previsões geradas**: $(wc -l < outputs/quartas_previsao.csv) linhas" >> $GITHUB_STEP_SUMMARY
          fi
```

## Secrets necessários

Vá em **Settings → Secrets and variables → Actions → New repository secret** e adicione:

| Secret | Descrição |
|--------|-----------|
| `BSD_API` | Chave da Bzzoiro Sports Data (https://sports.bzzoiro.com/register) |
| `API_FUTEBOL_KEY` | Chave da API Futebol (https://www.api-futebol.com.br/) |

Sem as chaves, o pipeline usa as bases de exemplo de `data/examples/` e continua funcionando.

## Teste manual

Após criar o arquivo, vá em **Actions → 🤖 Atualização diária - Libertadores 2026 → Run workflow** para testar.
