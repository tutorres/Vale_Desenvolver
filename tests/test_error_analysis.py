"""
TDD — test_error_analysis.py
Testes para src/error_analysis.py (matriz de confusao + analise de erros).

Usam dados sinteticos: nao dependem de model.pkl nem de data/features.parquet.
"""

import numpy as np
import pandas as pd
import pytest

from src.error_analysis import (
    confusion_matrix_2x2,
    false_negative_distribution,
    recall_by_month,
)


@pytest.fixture
def synthetic_test():
    """Conjunto de teste sintetico: rotulos, probabilidades e metadados."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = (rng.random(n) < 0.20).astype(int)
    # Probabilidades correlacionadas com o rotulo, mas com ruido (gera FP/FN).
    y_prob = np.clip(0.3 * y_true + rng.random(n) * 0.7, 0, 1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-05-01", periods=n, freq="90min"),
        "tipo_caminhao": rng.randint(0, 2, n),
        "current_state": rng.choice(["Operando", "Parado", "Manutencao"], n),
        "alarm_count_4h": rng.randint(0, 40, n),
        "label_4h": y_true,
    })
    return df, y_true, y_prob


class TestConfusionMatrix:

    def test_retorna_quatro_celulas(self, synthetic_test):
        _, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        assert set(cm.keys()) >= {"tp", "fp", "fn", "tn"}

    def test_matriz_2x2(self, synthetic_test):
        """A matriz deve ter exatamente 4 celulas (2x2)."""
        _, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        assert len({"tp", "fp", "fn", "tn"} & set(cm.keys())) == 4

    def test_tp_mais_fn_igual_positivos_reais(self, synthetic_test):
        """TP + FN deve ser igual ao numero de positivos reais."""
        _, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        assert cm["tp"] + cm["fn"] == int((y_true == 1).sum())

    def test_tn_mais_fp_igual_negativos_reais(self, synthetic_test):
        _, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        assert cm["tn"] + cm["fp"] == int((y_true == 0).sum())

    def test_soma_das_celulas_igual_n_test(self, synthetic_test):
        """TP + FP + FN + TN deve somar N_test."""
        _, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(y_true)

    def test_celulas_sao_inteiros_nao_negativos(self, synthetic_test):
        _, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        for k in ("tp", "fp", "fn", "tn"):
            assert isinstance(cm[k], int)
            assert cm[k] >= 0


class TestFalseNegativeDistribution:

    def test_total_fn_bate_com_matriz(self, synthetic_test):
        df, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        cm = confusion_matrix_2x2(y_true, y_pred)
        dist = false_negative_distribution(df, y_true, y_pred)
        assert dist["total_fn"] == cm["fn"]

    def test_distribuicao_por_tipo_soma_total(self, synthetic_test):
        df, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        dist = false_negative_distribution(df, y_true, y_pred)
        by_tipo = dist["by_tipo_caminhao"]
        assert by_tipo["caminhao"] + by_tipo["escavadeira"] == dist["total_fn"]

    def test_distribuicao_por_estado_soma_total(self, synthetic_test):
        df, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        dist = false_negative_distribution(df, y_true, y_pred)
        assert sum(dist["by_current_state"].values()) == dist["total_fn"]

    def test_distribuicao_por_faixa_alarmes_soma_total(self, synthetic_test):
        df, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        dist = false_negative_distribution(df, y_true, y_pred)
        assert sum(dist["by_alarm_count_4h_bin"].values()) == dist["total_fn"]


class TestRecallByMonth:

    def test_recall_por_mes_no_intervalo(self, synthetic_test):
        df, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        rec = recall_by_month(df, y_true, y_pred)
        for info in rec.values():
            assert 0.0 <= info["recall"] <= 1.0

    def test_recall_por_mes_tp_mais_fn(self, synthetic_test):
        """Em cada mes, positivos = TP + FN."""
        df, y_true, y_prob = synthetic_test
        y_pred = (y_prob >= 0.7).astype(int)
        rec = recall_by_month(df, y_true, y_pred)
        for info in rec.values():
            assert info["positives"] == info["tp"] + info["fn"]
