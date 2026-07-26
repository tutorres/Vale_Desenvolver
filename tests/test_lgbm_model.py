"""
TDD: test_lgbm_model.py
Testes para src/lgbm_model.py (segunda abordagem, LightGBM, CM 4.3).
"""

import numpy as np
import pandas as pd
import pytest

from src.lgbm_model import evaluate_lgbm, train_lgbm
from src.model import FEATURE_COLS

METRIC_KEYS = {"f1", "precision", "recall", "auc_roc", "auc_pr"}


@pytest.fixture
def features_df():
    """Dataset sintético com sinal aprendível pelo LightGBM."""
    rng = np.random.RandomState(42)
    n = 1500
    critical = rng.randint(0, 4, n)
    alarm_4h = rng.randint(0, 35, n)
    score = 0.15 * critical + 0.02 * alarm_4h + rng.normal(0, 0.3, n)
    label_4h = (score > np.quantile(score, 0.88)).astype(int)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10min"),
        "alarm_count_15m": rng.randint(0, 5, n),
        "alarm_count_30m": rng.randint(0, 8, n),
        "alarm_count_1h": rng.randint(0, 12, n),
        "alarm_count_2h": rng.randint(0, 20, n),
        "alarm_count_4h": alarm_4h,
        "critical_alarm_count_1h": critical,
        "time_since_last_critical": rng.uniform(0, 480, n),
        "tipo_caminhao": rng.randint(0, 2, n),
        "label_4h": label_4h,
    })
    return df


class TestTrainLgbm:

    def test_retorna_modelo_e_metricas(self, features_df):
        model, metrics = train_lgbm(features_df, "label_4h")
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metricas_obrigatorias_presentes(self, features_df):
        _, metrics = train_lgbm(features_df, "label_4h")
        assert METRIC_KEYS.issubset(metrics.keys())

    def test_metricas_no_intervalo_unitario(self, features_df):
        _, metrics = train_lgbm(features_df, "label_4h")
        for k in METRIC_KEYS:
            assert 0.0 <= metrics[k] <= 1.0, f"{k}={metrics[k]} fora de [0,1]"

    def test_determinismo_com_seed(self, features_df):
        _, m1 = train_lgbm(features_df, "label_4h", seed=42)
        _, m2 = train_lgbm(features_df, "label_4h", seed=42)
        assert m1 == m2

    def test_split_shapes(self, features_df):
        _, metrics = train_lgbm(features_df, "label_4h")
        assert metrics["train_rows"] + metrics["test_rows"] == len(features_df)


class TestEvaluateLgbm:

    def test_tres_thresholds(self, features_df):
        model, _ = train_lgbm(features_df, "label_4h")
        X_test = features_df[FEATURE_COLS].tail(200)
        y_test = features_df["label_4h"].tail(200)
        results = evaluate_lgbm(model, X_test, y_test)
        assert len(results) == 3

    def test_metricas_por_threshold_validas(self, features_df):
        model, _ = train_lgbm(features_df, "label_4h")
        X_test = features_df[FEATURE_COLS].tail(200)
        y_test = features_df["label_4h"].tail(200)
        results = evaluate_lgbm(model, X_test, y_test, thresholds=[0.3, 0.5, 0.7])
        for r in results:
            assert r["threshold"] in (0.3, 0.5, 0.7)
            for k in ("f1", "precision", "recall"):
                assert 0.0 <= r[k] <= 1.0
