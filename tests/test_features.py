"""
TDD: test_features.py
Testes para src/features.py
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features import build_features


FEATURE_COLUMNS = [
    "TAG", "timestamp",
    "alarm_count_15m", "alarm_count_30m", "alarm_count_1h",
    "alarm_count_2h", "alarm_count_4h",
    "critical_alarm_count_1h", "time_since_last_critical",
    "tipo_caminhao", "current_state",
    "label_1h", "label_2h", "label_4h",
]


@pytest.fixture
def df_tele():
    """Telemetria mínima limpa para testes de feature engineering."""
    base = datetime(2025, 1, 1, 8, 0, 0)
    return pd.DataFrame({
        "TAG": ["CAM001"] * 10 + ["ESC001"] * 5,
        "Alarme": ["AL01"] * 15,
        "Id_Criticidade": [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2],
        "Criticidade": ["Crítico", "Não Crítico"] * 7 + ["Crítico"],
        "Valor": [float(i) for i in range(15)],
        "Classe": [np.nan] * 15,
        "Is_Dont_Go": [0] * 13 + [1] + [0],
        "Tipo": ["Caminhao"] * 10 + ["Escavadeira"] * 5,
        "timestamp": [base + timedelta(minutes=i * 5) for i in range(15)],
    })


@pytest.fixture
def df_apon():
    """Apontamentos mínimos para testes."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return pd.DataFrame({
        "Tag": ["CAM001", "ESC001"],
        "Inicio": [base, base],
        "Fim": [base + timedelta(hours=12), base + timedelta(hours=12)],
        "Classe": ["Operando", "Parado"],
    })


class TestBuildFeatures:

    def test_retorna_dataframe(self, df_tele, df_apon):
        result = build_features(df_tele, df_apon)
        assert isinstance(result, pd.DataFrame)

    def test_colunas_esperadas_presentes(self, df_tele, df_apon):
        result = build_features(df_tele, df_apon)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Coluna ausente: {col}"

    def test_sem_nan_em_features_numericas(self, df_tele, df_apon):
        """Features numéricas não podem ter NaN. O modelo não aceita."""
        result = build_features(df_tele, df_apon)
        numeric_cols = [c for c in result.columns if result[c].dtype in [np.float64, np.int64]]
        assert result[numeric_cols].isna().sum().sum() == 0

    def test_labels_binarios(self, df_tele, df_apon):
        """Labels devem conter apenas 0 e 1."""
        result = build_features(df_tele, df_apon)
        for col in ["label_1h", "label_2h", "label_4h"]:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} tem valores inválidos"

    def test_tipo_caminhao_binario(self, df_tele, df_apon):
        """tipo_caminhao deve ser 1 para Caminhao, 0 para Escavadeira."""
        result = build_features(df_tele, df_apon)
        assert set(result["tipo_caminhao"].unique()).issubset({0, 1})

    def test_sem_data_leakage(self, df_tele, df_apon):
        """alarm_count_* não pode usar dados do futuro em relação ao timestamp."""
        result = build_features(df_tele, df_apon)
        # Primeiro registro de cada TAG deve ter alarm_count_15m == 0 ou 1 (só ele mesmo)
        first_cam = result[result["TAG"] == "CAM001"].iloc[0]
        assert first_cam["alarm_count_15m"] <= 1

    def test_alarm_count_aumenta_com_tempo(self, df_tele, df_apon):
        """Contagem de alarmes deve aumentar conforme mais alarmes ocorrem."""
        result = build_features(df_tele, df_apon)
        cam = result[result["TAG"] == "CAM001"].reset_index(drop=True)
        assert cam["alarm_count_1h"].iloc[-1] >= cam["alarm_count_1h"].iloc[0]

    def test_current_state_e_category(self, df_tele, df_apon):
        """current_state deve ser category. Com 37M registros reais, um dtype
        object (string Python por linha) explode a memória (ver spec/DECISIONS.md).
        Tags com estados diferentes não podem fazer o concat reverter para object."""
        result = build_features(df_tele, df_apon)
        assert result["current_state"].dtype.name == "category"
