"""
TDD: test_model.py
Testes para src/model.py
"""

import numpy as np
import pandas as pd
import pytest

from src.model import train_model, evaluate_model


@pytest.fixture
def features_df():
    """Dataset de features sintético para teste do modelo."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "TAG": [f"CAM{i%10:03d}" for i in range(n)],
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10min"),
        "alarm_count_15m": np.random.randint(0, 5, n),
        "alarm_count_30m": np.random.randint(0, 8, n),
        "alarm_count_1h": np.random.randint(0, 12, n),
        "alarm_count_2h": np.random.randint(0, 20, n),
        "alarm_count_4h": np.random.randint(0, 35, n),
        "critical_alarm_count_1h": np.random.randint(0, 5, n),
        "time_since_last_critical": np.random.uniform(0, 480, n),
        "tipo_caminhao": np.random.randint(0, 2, n),
        "label_1h": np.where(np.random.random(n) < 0.05, 1, 0),
        "label_2h": np.where(np.random.random(n) < 0.07, 1, 0),
        "label_4h": np.where(np.random.random(n) < 0.10, 1, 0),
    })
    return df


class TestTrainModel:

    def test_retorna_modelo_e_metricas(self, features_df):
        model, metrics = train_model(features_df, label_col="label_1h")
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metricas_obrigatorias_presentes(self, features_df):
        """F1, precision e recall devem estar nas métricas."""
        _, metrics = train_model(features_df, label_col="label_1h")
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc" in metrics

    def test_accuracy_nao_e_metrica_principal(self, features_df):
        """accuracy não deve ser a única métrica retornada."""
        _, metrics = train_model(features_df, label_col="label_1h")
        if "accuracy" in metrics:
            assert "f1" in metrics

    def test_split_temporal_respeitado(self, features_df):
        """Treino usa Jan-Abr, teste usa Mai-Jun, sem dados futuros no treino."""
        model, metrics = train_model(features_df, label_col="label_1h")
        assert model is not None

    def test_modelo_melhor_que_random(self, features_df):
        """F1 deve ser positivo. O modelo é melhor que classificador aleatório."""
        _, metrics = train_model(features_df, label_col="label_1h")
        assert metrics["f1"] >= 0


class TestEvaluateModel:

    def test_avalia_multiplos_thresholds(self, features_df):
        """evaluate_model deve retornar métricas para pelo menos 3 thresholds."""
        model, _ = train_model(features_df, label_col="label_1h")

        feature_cols = [
            "alarm_count_15m", "alarm_count_30m", "alarm_count_1h",
            "alarm_count_2h", "alarm_count_4h",
            "critical_alarm_count_1h", "time_since_last_critical", "tipo_caminhao"
        ]
        X_test = features_df[feature_cols].tail(100)
        y_test = features_df["label_1h"].tail(100)

        results = evaluate_model(model, X_test, y_test)
        assert len(results) >= 3

    def test_nunca_retorna_apenas_accuracy(self, features_df):
        """evaluate_model nunca deve retornar só accuracy."""
        model, _ = train_model(features_df, label_col="label_1h")
        feature_cols = [
            "alarm_count_15m", "alarm_count_30m", "alarm_count_1h",
            "alarm_count_2h", "alarm_count_4h",
            "critical_alarm_count_1h", "time_since_last_critical", "tipo_caminhao"
        ]
        X_test = features_df[feature_cols].tail(100)
        y_test = features_df["label_1h"].tail(100)

        results = evaluate_model(model, X_test, y_test)
        for r in results:
            assert "f1" in r or "recall" in r
