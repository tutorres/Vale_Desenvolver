"""
TDD — test_baselines.py
Testes para src/baselines.py (baselines de referência CM 4.2).
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    evaluate_dummy,
    evaluate_heuristic,
    run_baselines,
    temporal_split,
)

METRIC_KEYS = {"f1", "precision", "recall", "auc_roc", "auc_pr"}


@pytest.fixture
def features_df():
    """Dataset sintético de features com sinal na feature crítica."""
    rng = np.random.RandomState(42)
    n = 1200
    critical = rng.randint(0, 4, n)
    # label correlacionado com a feature crítica, com ruído
    label_4h = ((critical > 0) & (rng.random(n) < 0.6)).astype(int)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10min"),
        "alarm_count_15m": rng.randint(0, 5, n),
        "alarm_count_30m": rng.randint(0, 8, n),
        "alarm_count_1h": rng.randint(0, 12, n),
        "alarm_count_2h": rng.randint(0, 20, n),
        "alarm_count_4h": rng.randint(0, 35, n),
        "critical_alarm_count_1h": critical,
        "time_since_last_critical": rng.uniform(0, 480, n),
        "tipo_caminhao": rng.randint(0, 2, n),
        "label_4h": label_4h,
    })
    return df


def _assert_valid_metrics(m):
    assert METRIC_KEYS.issubset(m.keys())
    for k in METRIC_KEYS:
        assert 0.0 <= m[k] <= 1.0, f"{k}={m[k]} fora de [0,1]"
        assert isinstance(m[k], float)


class TestTemporalSplit:

    def test_shapes_consistentes(self, features_df):
        X_tr, y_tr, X_te, y_te = temporal_split(features_df, "label_4h")
        assert len(X_tr) == len(y_tr)
        assert len(X_te) == len(y_te)
        assert len(X_tr) + len(X_te) == len(features_df)
        assert X_tr.shape[1] == 8  # FEATURE_COLS

    def test_teste_nao_vazio(self, features_df):
        _, _, X_te, _ = temporal_split(features_df, "label_4h")
        assert len(X_te) > 0


class TestDummyBaselines:

    def test_most_frequent_metricas_validas(self, features_df):
        X_tr, y_tr, X_te, y_te = temporal_split(features_df, "label_4h")
        m = evaluate_dummy(X_tr, y_tr, X_te, y_te, "most_frequent")
        _assert_valid_metrics(m)

    def test_stratified_metricas_validas(self, features_df):
        X_tr, y_tr, X_te, y_te = temporal_split(features_df, "label_4h")
        m = evaluate_dummy(X_tr, y_tr, X_te, y_te, "stratified")
        _assert_valid_metrics(m)

    def test_stratified_determinismo_com_seed(self, features_df):
        X_tr, y_tr, X_te, y_te = temporal_split(features_df, "label_4h")
        m1 = evaluate_dummy(X_tr, y_tr, X_te, y_te, "stratified", seed=42)
        m2 = evaluate_dummy(X_tr, y_tr, X_te, y_te, "stratified", seed=42)
        assert m1 == m2


class TestHeuristicBaseline:

    def test_metricas_validas(self, features_df):
        _, _, X_te, y_te = temporal_split(features_df, "label_4h")
        m = evaluate_heuristic(X_te, y_te)
        _assert_valid_metrics(m)

    def test_regra_critical_gt_zero(self, features_df):
        """A regra prevê positivo exatamente quando critical_alarm_count_1h > 0."""
        _, _, X_te, y_te = temporal_split(features_df, "label_4h")
        m = evaluate_heuristic(X_te, y_te)
        # com sinal embutido, o recall da heurística deve ser alto
        assert m["recall"] > 0.5


class TestRunBaselines:

    def test_retorna_tres_baselines(self, features_df):
        results = run_baselines(features_df, "label_4h")
        assert set(results.keys()) == {
            "dummy_most_frequent", "dummy_stratified", "heuristic_critical_1h"
        }

    def test_todas_metricas_validas(self, features_df):
        results = run_baselines(features_df, "label_4h")
        for m in results.values():
            _assert_valid_metrics(m)

    def test_determinismo(self, features_df):
        r1 = run_baselines(features_df, "label_4h")
        r2 = run_baselines(features_df, "label_4h")
        assert r1 == r2
