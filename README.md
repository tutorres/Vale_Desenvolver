# Vale Programa Desenvolver 2026 — Análise Avançada de Dados

**Participante:** Arthur Torres — CEFET-MG, Engenharia de Computação  
**Desafio:** Antecipar ocorrência de alertas críticos (Don't Go) em equipamentos de mineração  
**Entrega:** 20 de julho de 2026

---

## Objetivo

Dado o histórico de alarmes de telemetria de um equipamento até o instante T,
prever se um alerta Don't Go ocorrerá nas próximas 1h, 2h ou 4h.

---

## Dataset

Os dados foram fornecidos pela Vale S.A. como parte do Programa Desenvolver 2026
e **não estão incluídos neste repositório** (Edital item 7.11).

O dataset simula operação real de uma mina de minério de ferro com problemas
intencionais de qualidade de dados inseridos como parte do desafio.

Para reproduzir os resultados, o dataset deve ser obtido pelo programa e
colocado na pasta `data/` conforme estrutura abaixo.

---

## Estrutura esperada da pasta `data/`

```
data/
├── telemetry_jan.parquet
├── telemetry_feb.parquet
├── telemetry_mar.parquet
├── telemetry_abr.parquet
├── telemetry_may.parquet
├── telemetry_jun.parquet
├── desenvolver_apontamentos.parquet
├── Alarmes_-_Regra_de_Negocio.xlsx
├── Dicionario_de_Dados.xlsx
├── desenvolver_dontgo.xlsx
└── README_dataset.pdf
```

---

## Como reproduzir

```bash
# 1. Clonar o repositório
git clone git@github.com:tutorres/Vale_Desenvolver.git
cd Vale_Desenvolver

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Colocar o dataset em data/ (ver estrutura acima)

# 5. Rodar os testes
pytest tests/ -v

# 6. Executar os notebooks em ordem
jupyter notebook notebooks/01_clean.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_features.ipynb
jupyter notebook notebooks/04_model.ipynb
jupyter notebook notebooks/05_report.ipynb
```

---

## Pipeline

| Step | Notebook | Descrição |
|---|---|---|
| 1 | `01_clean.ipynb` | Limpeza e documentação dos erros de qualidade |
| 2 | `02_eda.ipynb` | EDA e validação das regras de negócio |
| 3 | `03_features.ipynb` | Feature engineering e janela deslizante |
| 4 | `04_model.ipynb` | Treinamento XGBoost + SHAP |
| 5 | `05_report.ipynb` | Geração do relatório final |

---

## Resultados (dataset real — 37.164.054 registros)

### Qualidade de dados

4 problemas de qualidade identificados e corrigidos (3 eram conhecidos previamente;
o 4º — NULL string em `Valor` — só apareceu ao rodar no dataset real completo):

| # | Coluna | Problema | Registros corrigidos |
|---|---|---|---|
| 1 | `Criticidade` | Corrupção de encoding UTF-8 | 11 |
| 2 | `Classe` | String literal `"NULL"` | 36.104.611 |
| 3 | `Valor` | String literal `"NULL"` | 237.443 |
| 4 | `Valor` | Vírgula como separador decimal | 821.849 |

### Modelo (XGBoost, split temporal Jan–Abr / Mai–Jun)

| Horizonte | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| 1h | 0.114 | 0.062 | 0.774 | 0.883 |
| 2h | 0.143 | 0.080 | 0.662 | 0.809 |
| **4h** (melhor) | **0.186** | 0.111 | 0.577 | 0.767 |

Melhor horizonte: **4h**. Avaliação por threshold (label_4h):

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.3 | 0.152 | 0.087 | 0.592 |
| 0.5 | 0.186 | 0.111 | 0.577 |
| 0.7 | **0.207** | 0.126 | 0.574 |

SHAP plots em `shap_plots/`. Métricas completas em `metrics.json`.

---

## Stack

- Python 3.11+, Pandas, PyArrow, DuckDB
- XGBoost, Scikit-learn, imbalanced-learn
- SHAP, Matplotlib, Seaborn, Plotly
- Jupyter, Git
