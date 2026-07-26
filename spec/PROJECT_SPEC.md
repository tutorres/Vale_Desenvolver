# Vale Programa Desenvolver 2026: Project Specification (SDD)

## 1. Objetivo

Desenvolver uma solução analítica capaz de **antecipar alertas críticos (Don't Go)**
em equipamentos de mineração (caminhões e escavadeiras), a partir de dados de
telemetria real de uma mina de minério de ferro.

**Pergunta central:**
> Dado o histórico de alarmes de um equipamento até o instante T,
> ele vai gerar um Don't Go nas próximas N horas?

---

## 2. Entradas e Saídas

### Entradas
| Arquivo | Registros | Descrição |
|---|---|---|
| `telemetry_jan..jun.parquet` | 37.164.054 | Alarmes de telemetria por equipamento |
| `desenvolver_apontamentos.parquet` | 377.907 | Estado operacional por equipamento |
| `Alarmes_Regra_de_Negocio.xlsx` | - | Regras de janela deslizante por alarme |
| `desenvolver_dontgo.xlsx` | - | Sequência de eventos que gera Don't Go |

### Saídas esperadas
| Artefato | Descrição |
|---|---|
| `cleaned_telemetry.parquet` | Dados limpos e validados |
| `features.parquet` | Dataset de features para ML |
| `model.pkl` | Modelo XGBoost treinado |
| `metrics.json` | F1, precision, recall, AUC por threshold |
| `shap_plots/` | Gráficos de explicabilidade |
| `report.pdf` | Relatório final no template Vale |
| `notebooks/` | Pipeline reproduzível passo a passo |

---

## 3. Módulos do Sistema

### 3.1 `src/loader.py`: Carregamento
- Carregar e concatenar os 6 arquivos parquet
- Validar schema esperado (colunas, tipos)
- Retornar DataFrame limpo de estrutura

### 3.2 `src/cleaner.py`: Limpeza
- Corrigir encoding de `Criticidade`
- Substituir string `"NULL"` por `np.nan` em `Classe`
- Corrigir separador decimal em `Valor`
- Detectar e reportar `Inicio > Fim` em apontamentos
- Detectar TAGs inconsistentes entre datasets
- Retornar DataFrame limpo + `QualityReport`

### 3.3 `src/validator.py`: Validação de qualidade
- Gerar relatório estruturado de cada erro encontrado
- Contar registros afetados por cada problema
- Retornar `QualityReport` (dataclass)

### 3.4 `src/business_rules.py`: Regras de negócio
- Ler regras por tipo de alarme (qty, time_window_minutes)
- Aplicar janela deslizante por TAG por Alarme
- Comparar sinal derivado vs `Is_Dont_Go` registrado
- Retornar falsos positivos e falsos negativos

### 3.5 `src/features.py`: Feature engineering
- Rolling alarm counts (15min, 30min, 1h, 2h, 4h)
- Distribuição por tipo de alarme por turno
- Tempo desde último alarme crítico por TAG
- Merge com estado operacional (apontamentos)
- One-hot de `Tipo` (Caminhao / Escavadeira)
- Label: `Is_Dont_Go` nos próximos [1h, 2h, 4h]

### 3.6 `src/model.py`: Modelo preditivo
- Split temporal (train Jan-Abr / test Mai-Jun)
- Treinamento XGBoost com `scale_pos_weight`
- Avaliação: F1, precision, recall, AUC
- SHAP values e plots
- Serialização do modelo

### 3.7 `src/utils.py`: Utilitários
- Logging padronizado
- Seeds de reprodutibilidade
- Funções de plot reutilizáveis

---

## 4. Regras de Negócio Críticas

### 4.1 Don't Go
Um Don't Go é gerado quando um equipamento acumula **Y alarmes de um tipo**
dentro de uma **janela de Z minutos**, conforme definido em `Alarmes_Regra_de_Negocio.xlsx`.

### 4.2 Criticidade
- `Id_Criticidade = 1` → Crítico (foco do modelo)
- `Id_Criticidade = 2` → Não Crítico
- `Id_Criticidade = 3` → Informacional
- `Id_Criticidade = 4` → Outros

### 4.3 Classes operacionais
- `Operando` → equipamento em operação normal
- `Parado` → parado mas disponível
- `Hibernando` → inativo por período longo
- `Manutenção` → fora de operação para reparo

---

## 5. Restrições Técnicas

| Restrição | Detalhe |
|---|---|
| Sem data leakage | Split sempre por tempo, nunca random |
| Métrica principal | F1 + precision + recall. Accuracy é enganosa (0.05% positivo) |
| Dados externos | Proibidos (Edital item 4.1) |
| Dataset no git | Proibido (Edital item 7.11) |
| Reprodutibilidade | Seeds fixos, requirements.txt pinado |

---

## 6. Critérios de Avaliação (mapeamento técnico)

| Critério | Pontuação máxima | Como garantir |
|---|---|---|
| Diagnóstico | 5 | Documentar cada erro: coluna, contagem, fix |
| Qualidade da Solução | 5 | Modelo ML com métricas + SHAP |
| Reprodutibilidade | 5 | README claro, notebooks numerados, seeds fixos |

---

## 7. Stack Técnico

```
pandas==2.2.x
pyarrow==16.x
duckdb==1.x
numpy==1.26.x
scikit-learn==1.5.x
xgboost==2.x
lightgbm==4.x
imbalanced-learn==0.12.x
shap==0.45.x
matplotlib==3.9.x
seaborn==0.13.x
plotly==5.x
jupyter==1.x
```

---

## 8. Estrutura de Pastas

```
vale_project/
├── spec/
│   ├── PROJECT_SPEC.md      # este arquivo
│   ├── contracts/           # contratos de entrada/saída por módulo
│   └── DECISIONS.md         # decisões técnicas e racional
├── src/
│   ├── loader.py
│   ├── cleaner.py
│   ├── validator.py
│   ├── business_rules.py
│   ├── features.py
│   ├── model.py
│   └── utils.py
├── tests/
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_validator.py
│   ├── test_business_rules.py
│   ├── test_features.py
│   └── test_model.py
├── notebooks/
│   ├── 01_clean.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_features.ipynb
│   ├── 04_model.ipynb
│   └── 05_report.ipynb
├── data/                    # NÃO commitar, apenas local
├── requirements.txt
├── README.md
└── .gitignore
```
