# Upgrade Spec — Fechar lacunas do Estudo Guiado

## Objetivo
Elevar o relatório e o pipeline para cobrir os Tópicos Sugeridos (ex-"Conteúdos Mínimos")
do Estudo Guiado que hoje estão ausentes, sem quebrar o que já funciona. Foco na nota de
**Qualidade da Solução**.

Prazo: entrega 20/07/2026. Escopo escolhido: **completo**. Segunda abordagem: **LightGBM**.

## Lacunas a fechar (mapeadas ao guia)
1. **Baseline ausente (CM 4.2)** — não há modelo de referência.
2. **Só uma abordagem (CM 4.3)** — o guia pede ao menos duas. Adicionar LightGBM.
3. **Impacto de negócio não quantificado (CM 1.2 / CM 6.1)** — traduzir recall em % de
   alertas antecipados / horas de parada evitadas.
4. **Sem matriz de confusão / análise de erros (CM 5.2)**.
5. **Relatório sem figuras (CM 2.x, 5.x, 6.x)** — nenhuma imagem embutida.

## Restrições invariáveis (NÃO violar)
- **Split temporal**: treino = timestamp <= 2025-04-30; teste = timestamp >= 2025-05-01.
  Nunca split aleatório. Mesma máscara para todos os modelos (comparação justa).
- **Sem data leakage**: features já são calculadas sem vazamento; não recomputar target.
- **Features**: usar exatamente `src/model.py::FEATURE_COLS` (8 features) para todos os
  modelos, garantindo comparabilidade.
- **Dados fora do git**: nada de `data/*.parquet|xlsx|csv` commitado (Edital 7.11).
- **Relatório sem travessões (—)**: nenhum "—" no texto final. Usar vírgula/parênteses/dois-pontos.
- **Fonte única do relatório**: `_fill_report.py` gera o `.docx`. Toda mudança de texto vai
  no `_fill_report.py`; o `.docx` é regenerado a partir dele. O texto-base já reenquadrado
  (foco previsão/prevenção, sem travessões) está no `.docx` atual — portar para o script.
- **Testes verdes**: `pytest tests/ -q` deve continuar passando; novas funções entram com teste (TDD).
- **Reprodutibilidade**: seed 42 (`src/model.py::SEED`), sem aleatoriedade não semeada.

## Insumos já disponíveis
- `data/features.parquet` (37.164.054 × 14): TAG, timestamp, alarm_count_{15m,30m,1h,2h,4h},
  critical_alarm_count_1h, time_since_last_critical, tipo_caminhao, current_state,
  label_1h, label_2h, label_4h.
- `model.pkl` — XGBoost já treinado (label_4h, threshold 0.5).
- `metrics.json` — métricas XGBoost por horizonte + thresholds.
- Figuras EDA já geradas em `data/` (não rastreadas): eda_criticidade_tipo.png,
  eda_dontgo_timeline.png, eda_top_alarmes.png, feature_correlation.png,
  feature_distributions.png. SHAP em `shap_plots/`.

## Convenção de saída
- Novas figuras rastreáveis em `figures/` (criar). Copiar para lá também as figuras EDA/SHAP
  que forem embutidas, para o repo ter os artefatos versionados.
- Resultados numéricos em `analysis/` (criar): JSONs consumidos pelo relatório.

---

## Workstream A — Modelagem comparativa (baseline + LightGBM)
Arquivos próprios (não colidir com B): `src/baselines.py`, `src/lgbm_model.py`,
`tests/test_baselines.py`, `tests/test_lgbm_model.py`, `analysis/model_comparison.json`,
`figures/roc_pr_curves.png`, `figures/model_comparison.png`.

1. **Baselines** (`src/baselines.py`):
   - `DummyClassifier(strategy="most_frequent")` e `strategy="stratified"` (seed 42).
   - Baseline heurístico de domínio: prever positivo quando `critical_alarm_count_1h > 0`
     (ou `alarm_count_4h` acima de um limiar calibrado no treino). Documentar a regra.
   - Avaliar no split temporal para **label_4h** (mínimo) — F1, precision, recall, AUC-ROC, AUC-PR.
2. **LightGBM** (`src/lgbm_model.py`): mesmas FEATURE_COLS, mesmo split, tratar
   desbalanceamento (`is_unbalance=True` ou `scale_pos_weight`). Treinar para os 3 horizontes;
   avaliar F1/precision/recall/AUC-ROC/AUC-PR + thresholds [0.3,0.5,0.7] no melhor horizonte.
3. **Tabela comparativa** (CM 5.1): baseline(s) × XGBoost × LightGBM, colunas
   Precision/Recall/F1/AUC-ROC/AUC-PR, para label_4h. Salvar em `analysis/model_comparison.json`.
4. **Figuras**:
   - `figures/roc_pr_curves.png` — curvas ROC e Precision-Recall dos modelos (Figura 9 do guia).
   - `figures/model_comparison.png` — barras F1/recall baseline vs modelos (Figura 13 do guia).
5. **Regra de utilidade** (CM 4.2): declarar explicitamente que o modelo escolhido supera o baseline.
6. Testes: shapes/типos de retorno, determinismo com seed, e que métricas ficam em [0,1].

## Workstream B — Matriz de confusão e análise de erros
Arquivos próprios: `src/error_analysis.py`, `tests/test_error_analysis.py`,
`analysis/error_analysis.json`, `figures/confusion_matrix_4h.png`.
Usa o `model.pkl` existente + `data/features.parquet` (NÃO retreina).

1. **Matriz de confusão** (CM 5.2) do XGBoost em label_4h no teste, no threshold 0.7
   (o preferido no relatório). Salvar `figures/confusion_matrix_4h.png` com anotação de
   impacto operacional (FN = parada não planejada; FP = inspeção desnecessária).
2. **Análise de falsos negativos**: distribuição dos FN por `tipo_caminhao` (caminhão vs
   escavadeira) e, se viável, por faixa de `alarm_count_4h` / `current_state`. Identificar
   onde o modelo mais falha. Salvar números em `analysis/error_analysis.json`.
3. **Degradação temporal** (CM 5.2, opcional se sobrar tempo): recall do teste dividido por
   mês (maio vs junho) para checar drift. Incluir no JSON se calculado.
4. Testes: função retorna matriz 2×2 coerente (TP+FN = positivos reais) e contagens somam N_test.

## Workstream C — Relatório (depende de A e B)
Arquivos: `_fill_report.py` (reescrever como fonte única), regenerar
`Relatorio_Final_Vale_Desenvolver2026.docx`. Também mover/copar figuras para `figures/`.

1. **Portar o texto atual reenquadrado e SEM travessões** do `.docx` para o `_fill_report.py`
   (o script hoje tem a versão antiga — substituir todo o texto pelo atual + as adições).
2. **Adicionar conteúdo novo**:
   - *Entendimento do Negócio*: métrica de sucesso de negócio + 1 cenário de aplicação (dispatcher
     com N h de antecedência) — CM 1.2.
   - *Metodologia/Resultados*: parágrafo de **baseline + LightGBM** com a **tabela comparativa**
     (de `analysis/model_comparison.json`) — CM 4.2/4.3/5.1. Afirmar que o modelo supera o baseline.
   - *Resultados*: **matriz de confusão + análise de erros** (de `analysis/error_analysis.json`) — CM 5.2.
   - *Resultados/Conclusão*: **impacto de negócio quantificado** — a partir do recall do modelo
     escolhido: "com recall de X%, o modelo teria antecipado X% dos Don't Go do período de teste"
     e uma estimativa de horas de parada evitadas (declarar premissas) — CM 6.1.
   - Atualizar *Trabalhos Futuros*: remover "comparar com LightGBM" (agora feito); manter os demais.
   - Atualizar Palavras-Chave se fizer sentido (incluir LightGBM/Baseline).
3. **Embutir figuras** no `.docx` (via `docx.add_picture`, largura ~15cm, com legenda "Figura N: ..."):
   - EDA: eda_dontgo_timeline (série temporal de alertas), eda_criticidade_tipo, feature_correlation.
   - Modelagem: figures/roc_pr_curves.png, figures/model_comparison.png.
   - Avaliação: figures/confusion_matrix_4h.png, shap_plots/shap_summary_label_4h.png.
   Copiar as figuras EDA/SHAP embutidas para `figures/` para versionar no repo.
4. **Verificar**: 0 travessões no `.docx` final; todas as figuras renderizam; texto flui.

## Definição de pronto
- `pytest tests/ -q` verde (incluindo novos testes).
- `analysis/model_comparison.json` e `analysis/error_analysis.json` gerados.
- Figuras em `figures/` versionadas.
- `_fill_report.py` regenera o `.docx` sem travessões, com tabela comparativa, matriz de
  confusão, impacto de negócio quantificado e figuras embutidas.
- Nada de `data/` commitado. Suite de testes e seeds preservam reprodutibilidade.
