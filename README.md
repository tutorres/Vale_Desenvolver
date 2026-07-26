# Vale Programa Desenvolver 2026: Análise Avançada de Dados

**Participante:** Arthur Torres, CEFET-MG, Engenharia de Computação  
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

## Resultados (dataset real, 37.164.054 registros)

### Qualidade de dados

4 problemas de qualidade identificados e corrigidos (3 eram conhecidos previamente;
o 4º, NULL string em `Valor`, só apareceu ao rodar no dataset real completo):

| # | Coluna | Problema | Registros corrigidos |
|---|---|---|---|
| 1 | `Criticidade` | Corrupção de encoding UTF-8 | 11 |
| 2 | `Classe` | String literal `"NULL"` | 36.104.611 |
| 3 | `Valor` | String literal `"NULL"` | 237.443 |
| 4 | `Valor` | Vírgula como separador decimal | 821.849 |

### Como ler os números: este é um problema de evento raro

Antes de qualquer métrica absoluta, o contexto que define se ela é boa ou ruim.

Split temporal: treino em Jan a Abr (23.273.520 linhas), teste em Mai a Jun
(13.890.534 linhas). No conjunto de teste há 227.623 positivos, ou seja,
**taxa base de 1,64%**. Em um problema tão desbalanceado, F1 absoluto baixo
é o resultado esperado até para um modelo útil, e acurácia não diz nada
(prever sempre "não" acerta 98,4% das vezes e não previne uma única parada).
O que importa é o ganho sobre a alternativa que a operação teria de fato.

### Comparação com baselines (label_4h, teste Mai a Jun)

Fonte: `analysis/model_comparison.json`.

| Modelo | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Dummy (most_frequent) | 0.000 | 0.000 | 0.000 | 0.500 | 0.016 |
| Dummy (stratified) | 0.017 | 0.032 | 0.022 | 0.500 | 0.016 |
| Heuristic (`critical_1h > 0`) | 0.021 | 0.210 | 0.038 | 0.529 | 0.072 |
| **XGBoost** | **0.111** | **0.577** | **0.186** | **0.767** | **0.205** |
| LightGBM | 0.110 | 0.579 | 0.185 | 0.776 | 0.197 |

Os dois números que sustentam o resultado:

- **F1 4.9x acima da melhor alternativa não-ML.** XGBoost 0.186 contra 0.038 da
  heurística `critical_1h > 0`, que é a regra que um time de operação escreveria
  sem modelo nenhum. Os dois dummies ficam em 0.000 e 0.022.
- **AUC-PR com 12.5x de lift sobre a taxa base.** 0.205 contra os 0.0164 que um
  classificador aleatório entregaria. AUC-PR é a métrica correta aqui porque,
  ao contrário da AUC-ROC, ela não é inflada pela massa de negativos.

Traduzindo para operação: no horizonte de 4h o modelo captura **57,7% dos eventos
Don't Go**, contra 21,0% da heurística, e faz isso com precisão 5.3x maior
(0.111 contra 0.021), ou seja, com muito menos alarme falso por acerto.

Sem inflar o resultado: precision de 0.111 significa que aproximadamente 9 em cada
10 alertas emitidos não viram Don't Go nas 4h seguintes. O modelo só se paga se o
custo de uma inspeção antecipada for bem menor que o de uma parada não planejada,
o que é o caso típico em mineração, mas é uma premissa que precisa ser validada
com o custo real da operação antes de qualquer implantação.

### Métricas por horizonte (XGBoost)

| Horizonte | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| 1h | 0.114 | 0.062 | 0.774 | 0.883 |
| 2h | 0.143 | 0.080 | 0.662 | 0.809 |
| **4h** (melhor F1) | **0.186** | 0.111 | 0.577 | 0.767 |

O trade-off é claro: horizontes curtos têm recall e AUC-ROC maiores, mas
precisão muito menor. 4h maximiza F1 e dá mais tempo de reação à operação.

Avaliação por threshold (label_4h):

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.3 | 0.152 | 0.087 | 0.592 |
| 0.5 | 0.186 | 0.111 | 0.577 |
| 0.7 | **0.206** | 0.126 | 0.574 |

Subir o threshold para 0.7 melhora precisão (0.111 para 0.126) custando quase
nada de recall (0.577 para 0.574). É o ponto de operação usado na análise de
erros abaixo.

SHAP plots em `shap_plots/`. Métricas completas em `metrics.json` e
`analysis/model_comparison.json`.

---

## Análise de erros

Fonte: `analysis/error_analysis.json` (XGBoost, label_4h, threshold 0.7).

### Matriz de confusão

Sobre os 13.890.534 registros de teste:

| | Previsto: sem Don't Go | Previsto: Don't Go |
|---|---|---|
| **Real: sem Don't Go** | TN 12.755.063 | FP 907.848 |
| **Real: Don't Go** | FN 96.918 | TP 130.705 |

Recall 0.574, precision 0.126.

Significado operacional dos dois erros, e eles não custam a mesma coisa:

- **Falso negativo:** Don't Go não antecipado, ou seja, parada não planejada.
- **Falso positivo:** alerta sem Don't Go, ou seja, inspeção desnecessária.

### Limitação 1: drift temporal

O desempenho não é estável ao longo do período de teste.

| Mês | Positivos | TP | FN | Recall |
|---|---|---|---|---|
| 2025-05 | 158.766 | 67.210 | 91.556 | **0.423** |
| 2025-06 | 68.857 | 63.495 | 5.362 | **0.922** |

Recall de 0.423 em maio contra 0.922 em junho. O recall global de 0.574 é uma
média que esconde essa dispersão, e maio, o mês imediatamente após o fim do
treino, é justamente o pior. Isso indica que o modelo não generaliza de forma
uniforme e que o número honesto para planejar uma implantação é o pior mês,
não a média. Mitigação necessária antes de produção: retreino periódico e
monitoramento de drift, com validação em janelas móveis em vez de um único
split fixo.

### Limitação 2: viés por tipo de equipamento

Distribuição dos 96.918 falsos negativos por tipo:

| Tipo de equipamento | Falsos negativos | % do total |
|---|---|---|
| Escavadeira | 94.932 | 98,0% |
| Caminhão | 1.986 | 2,0% |

Praticamente todo o erro do modelo está concentrado em escavadeiras. Isso é
consistente com a leitura do SHAP, em que o modelo se apoia fortemente em um
risco-base por tipo de equipamento. Na prática, o modelo funciona bem para
caminhões e mal para escavadeiras, o que sugere que escavadeira precisa de
features próprias ou de um modelo separado.

Por estado do equipamento no momento do falso negativo:

| Estado | Falsos negativos |
|---|---|
| Operando | 92.773 |
| Parado | 3.916 |
| Manutenção | 229 |

### Limitação 3: os falsos negativos estão todos no regime de alto volume de alarmes

Distribuição dos falsos negativos por faixa de contagem de alarmes em 4h:

| Alarmes em 4h | Falsos negativos |
|---|---|
| 0-5 | 10 |
| 6-10 | 17 |
| 11-20 | 22 |
| **21+** | **96.869** |

96.869 dos 96.918 falsos negativos (99,9%) estão na faixa de 21+ alarmes em 4h.
O modelo erra quase exclusivamente quando o equipamento já está em regime de
alarme intenso, que é exatamente o cenário em que a previsão teria mais valor.
A hipótese é que nesse regime as features de contagem saturam e param de
discriminar. Testar features de taxa, aceleração e composição dos alarmes,
em vez de contagem bruta, é o próximo passo mais promissor.

---

## Stack

- Python 3.11+, Pandas, PyArrow, DuckDB
- XGBoost, LightGBM, Scikit-learn, imbalanced-learn
- SHAP, Matplotlib, Seaborn, Plotly
- Jupyter, Git
