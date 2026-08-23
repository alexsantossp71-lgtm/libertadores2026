# Snapshot FBref — Copa Libertadores 2026

Fonte: [FBref comps/14](https://fbref.com/en/comps/14/stats/Copa-Libertadores-Stats)
Coletado em **2026-08-22** (estatísticas **não incluem** as classificatórias).

## Páginas disponíveis nesta competição

A FBref **não publica** passing, defense avançada, possession nem GCA/xG
para a Libertadores. O que existe (e o scraper pede):

| Página | URL | O que entra no snapshot |
|--------|-----|-------------------------|
| Standard | `/comps/14/stats/` | jogos, posse, gols, assistências, cartões |
| Shooting | `/comps/14/shooting/` | finalizações, no gol, G/Sh |
| Miscellaneous | `/comps/14/misc/` | faltas, impedimentos, cruzamentos, interceptações, desarmes ganhos |
| Playing Time / Keepers / Schedule | (raspagem ao vivo) | não versionados neste CSV de elencos |

## Arquivos

* `elencos_2026.csv` — 32 times da fase de grupos + mata-mata até as
  oitavas (22/08/2026), bloco *Squad Stats* + gols sofridos do bloco
  *Opponent Stats*.
* `jogadores_2026.csv` — **opcional**. Só aparece depois de
  `python src/fbref_scraper.py scrape` (a tabela de jogadores tem
  centenas de linhas; o HTML cacheado fica em `data/raw/fbref/`).

## Como atualizar

```bash
python src/fbref_scraper.py scrape --season 2026
# se a rede da FBref estiver ok, copie o processado para cá:
# cp data/processed/fbref_elencos.csv data/historical/fbref/elencos_2026.csv
```

Sports-Reference pede intervalo ≥ 3 s entre requisições. O cliente
espera 3,5 s e identifica o User-Agent do repositório.
