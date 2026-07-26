# Decisões Técnicas: Vale Projeto Desenvolver 2026

Registro de cada decisão relevante: o que foi escolhido, por que, e o que foi descartado.

---

## D001. Modelo principal: XGBoost (não PyTorch/TensorFlow)

**Decisão:** XGBoost como modelo principal.

**Racional:**
- Dados tabulares estruturados → gradient boosting supera redes neurais na maioria dos benchmarks
- Classe extremamente rara (0.05%) → `scale_pos_weight` nativo resolve sem arquitetura especial
- Critério de avaliação exige interpretabilidade → SHAP + XGBoost = explicação visual trivial
- Tempo limitado (7 semanas, iniciante em ML aplicado) → XGBoost domínio em 2 dias

**Descartado:** PyTorch MLP, pode ser adicionado como experimento comparativo em "Trabalhos Futuros"

---

## D002. Split temporal (não random)

**Decisão:** Train Jan-Abr / Test Mai-Jun, sempre por timestamp.

**Racional:**
- Dados de série temporal têm dependência temporal. Random split vaza informação do futuro
- Random split inflaria artificialmente as métricas e invalidaria o modelo em produção
- TimeSeriesSplit para cross-validation dentro do treino

**Risco:** Conceito drift entre meses. Monitorar distribuição de features entre períodos

---

## D003. Métrica principal: F1 (não accuracy)

**Decisão:** F1 como métrica de referência, com precision e recall reportados separadamente.

**Racional:**
- Classe positiva = 0.05% → um modelo que prediz sempre 0 teria 99.95% de accuracy
- Precisamos de recall alto (não perder Don't Go reais) e precision razoável (não gerar alarmes falsos)
- AUC-ROC como métrica secundária para comparação de modelos

---

## D004. Horizonte de predição: 1h, 2h e 4h

**Decisão:** Treinar e avaliar para os 3 horizontes, reportar o melhor.

**Racional:**
- Não sabemos a priori qual horizonte tem melhor sinal
- 1h: maior precisão, menos tempo de reação
- 4h: mais tempo de reação, sinal mais fraco
- Reportar os 3 demonstra rigor metodológico

---

## D005. Desbalanceamento: scale_pos_weight (não SMOTE por padrão)

**Decisão:** `scale_pos_weight = n_negativo / n_positivo` como primeira abordagem.

**Racional:**
- Nativo no XGBoost, sem overhead computacional
- SMOTE pode criar exemplos sintéticos que não refletem padrões reais de telemetria
- SMOTE testado como ablação se `scale_pos_weight` não for suficiente

---

## D006. Notebooks numerados por step

**Decisão:** 01_clean, 02_eda, 03_features, 04_model, 05_report

**Racional:**
- Critério de Reprodutibilidade exige que o juiz consiga replicar o pipeline
- Ordem explícita elimina ambiguidade
- Cada notebook independente mas com inputs/outputs documentados

---

## D007. Dataset fora do git

**Decisão:** `.gitignore` inclui `data/` completamente.

**Racional:**
- Edital item 7.11 proíbe uso dos dados para vantagem comercial
- README documenta como obter os dados pelo programa
- Código e notebooks publicados sem os dados brutos

---

## D008. Achados ao rodar com os dados reais (37M registros)

Os testes unitários usam fixtures pequenas (5-15 linhas) e não capturam problemas
que só aparecem na escala/diversidade real. Três problemas surgiram só na execução
real e foram corrigidos com TDD antes de re-rodar os notebooks:

**1. Memória: `object` dtype não escala.** `load_telemetry` carregava todas as
colunas de string como `object`; em 37M linhas isso precisa de ~30GB (vs. 9.7GB
livres na máquina). Fix: cast para `category` por mês. Descoberta adicional: o
`pd.concat` reverte `category` para `object` quando as categorias diferem entre
meses (ex: TAGs/operadores que só aparecem em alguns meses). Corrigido unificando
o `CategoricalDtype` antes do concat. O mesmo padrão se repetiu em
`build_features` (coluna `current_state`, só 5 valores possíveis, mas calculada
como `object` em 37M linhas) e na seleção de colunas (a função só usa 5 das 19
colunas de `df_tele`, mas copiava/ordenava todas).

**2. Quarto erro de qualidade: `'NULL'` em `Valor`.** Além dos 3 erros documentados
(encoding, NULL em Classe, vírgula decimal), 237.443 registros tinham a string
`'NULL'` também em `Valor`, quebrando o cast para float. Não estava nos dados de
teste sintéticos. Corrigido em `cleaner.py` (novo campo `valor_null_fixed` no
`QualityReport`) com o mesmo padrão usado para `Classe`.

**3. Verificação pós-limpeza com falso positivo.** O regex usado para corrigir o
encoding de `Criticidade` (`N.{1,2}o Cr.{1,2}tico`) usa `.` como wildcard, o que
também casa com o valor correto "Não Crítico" já limpo. A asserção de verificação
no notebook 01 reusava esse mesmo regex para checar "nenhuma corrupção restante",
o que nunca poderia passar. Corrigido checando o marcador de corrupção (`?`)
diretamente, em vez de reusar o regex de correção.

**4. `shap` 0.49.1 incompatível com `xgboost` 3.2.0.** XGBoost ≥3.1 serializa
`base_score` como string com colchetes (`"[5E-1]"`); o parser UBJSON do shap não
lida com esse formato ([shap#4202](https://github.com/shap/shap/issues/4202),
ainda sem fix lançado no PyPI). Sem alterar `requirements.txt`, o notebook 04
aplica um monkeypatch local em `decode_ubjson_buffer` que remove os colchetes
antes do `float()`. Mesmo valor numérico, só corrige o formato da string.
