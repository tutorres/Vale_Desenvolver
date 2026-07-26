# Contracts: Entrada e Saída por Módulo

Cada módulo tem um contrato explícito: o que recebe, o que garante, o que retorna.
A implementação e os testes devem respeitar esses contratos.

---

## loader.py

### Função: `load_telemetry(data_dir: str) -> pd.DataFrame`

**Input:**
- `data_dir`: caminho para a pasta com os 6 arquivos parquet de telemetria

**Garantias (preconditions):**
- Os 6 arquivos existem: `telemetry_jan.parquet` ... `telemetry_jun.parquet`
- Cada arquivo tem as colunas: `TAG, Alarme, Id_Criticidade, Criticidade, Valor, Classe, Is_Dont_Go, Tipo`

**Output:**
- DataFrame concatenado com todos os registros
- Coluna `_source_month` adicionada (jan, feb, mar, abr, may, jun)
- Schema validado, levanta `SchemaError` se coluna ausente

**Postconditions:**
- `len(df) == 37_164_054` (± tolerância por versão do dataset)
- Sem duplicatas de index

---

## cleaner.py

### Função: `clean_telemetry(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]`

**Input:**
- DataFrame bruto retornado por `load_telemetry`

**Output:**
- `df_clean`: DataFrame com os 3 erros intencionais corrigidos
- `report`: `QualityReport` dataclass com contagens por erro

**Postconditions:**
- `df_clean['Criticidade'].str.contains(r'N.{1,2}o').sum() == 0`
- `(df_clean['Classe'] == 'NULL').sum() == 0`
- `df_clean['Valor'].dtype == float64`
- `report.criticidade_fixed >= 0`
- `report.null_string_fixed >= 0`
- `report.decimal_fixed >= 0`

---

## validator.py

### Dataclass: `QualityReport`

```python
@dataclass
class QualityReport:
    criticidade_fixed: int       # registros com encoding corrigido
    null_string_fixed: int       # registros com NULL string substituído
    decimal_fixed: int           # registros com decimal corrigido
    inicio_after_fim: int        # apontamentos com Inicio > Fim
    tag_mismatches: list[str]    # TAGs presentes em telemetria mas não em apontamentos
    duplicate_rows: int          # linhas duplicadas encontradas
    timestamp_gaps: list[str]    # meses com gaps de timestamp detectados
    summary: str                 # texto gerado automaticamente para o relatório
```

### Função: `validate_apontamentos(df: pd.DataFrame) -> QualityReport`

**Postconditions:**
- `report.inicio_after_fim >= 0`
- `report.summary` não vazio

---

## business_rules.py

### Função: `load_rules(rules_path: str) -> dict`

**Output:**
```python
{
  "ALARME_XYZ": {"qty": 3, "window_minutes": 30},
  ...
}
```

### Função: `apply_sliding_window(df: pd.DataFrame, rules: dict) -> pd.DataFrame`

**Input:**
- `df`: telemetria limpa
- `rules`: dict retornado por `load_rules`

**Output:**
- DataFrame com coluna `Is_Dont_Go_derived` (0/1) por TAG por timestamp

**Postconditions:**
- `Is_Dont_Go_derived` contém apenas 0 e 1
- Sem leakage: janela olha apenas para trás (t-window até t)

### Função: `compare_signals(df: pd.DataFrame) -> dict`

**Input:** DataFrame com `Is_Dont_Go` e `Is_Dont_Go_derived`

**Output:**
```python
{
  "true_positives": int,
  "false_positives": int,
  "false_negatives": int,
  "true_negatives": int,
  "precision": float,
  "recall": float
}
```

---

## features.py

### Função: `build_features(df_tele: pd.DataFrame, df_apon: pd.DataFrame) -> pd.DataFrame`

**Input:**
- `df_tele`: telemetria limpa
- `df_apon`: apontamentos limpos

**Output:** DataFrame com colunas:

| Coluna | Tipo | Descrição |
|---|---|---|
| `TAG` | str | identificador do equipamento |
| `timestamp` | datetime | instante da janela |
| `alarm_count_15m` | int | alarmes nos últimos 15 min |
| `alarm_count_30m` | int | alarmes nos últimos 30 min |
| `alarm_count_1h` | int | alarmes na última 1h |
| `alarm_count_2h` | int | alarmes nas últimas 2h |
| `alarm_count_4h` | int | alarmes nas últimas 4h |
| `critical_alarm_count_1h` | int | alarmes críticos (Id_Criticidade=1) na última 1h |
| `time_since_last_critical` | float | minutos desde último alarme crítico |
| `tipo_caminhao` | int | 1 se Caminhao, 0 se Escavadeira |
| `current_state` | str | Operando / Parado / Hibernando / Manutenção |
| `label_1h` | int | Don't Go ocorre na próxima 1h |
| `label_2h` | int | Don't Go ocorre nas próximas 2h |
| `label_4h` | int | Don't Go ocorre nas próximas 4h |

**Postconditions:**
- Sem NaN nas colunas de feature numéricas
- `label_*` contém apenas 0 e 1
- Sem data leakage: features usam apenas dados anteriores ao timestamp

---

## model.py

### Função: `train_model(features: pd.DataFrame, label_col: str) -> tuple[XGBClassifier, dict]`

**Input:**
- `features`: DataFrame retornado por `build_features`
- `label_col`: coluna alvo (`label_1h`, `label_2h` ou `label_4h`)

**Split:**
- Train: Jan-Abr (baseado em `timestamp`)
- Test: Mai-Jun

**Output:**
- `model`: XGBClassifier treinado
- `metrics`: dict com `precision`, `recall`, `f1`, `auc`, `threshold`

**Postconditions:**
- `metrics['f1'] > 0` (modelo melhor que random)
- Modelo serializado em `model.pkl`
- SHAP plots salvos em `shap_plots/`

### Função: `evaluate_model(model, X_test, y_test) -> dict`

**Postconditions:**
- Retorna métricas para pelo menos 3 thresholds (0.3, 0.5, 0.7)
- Nunca usa accuracy como métrica principal
