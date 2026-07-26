"""
TDD: test_business_rules.py
Testes para src/business_rules.py
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.business_rules import apply_sliding_window, compare_signals


@pytest.fixture
def rules():
    return {"AL01": {"qty": 3, "window_minutes": 30}}


@pytest.fixture
def df_below_threshold():
    """2 alarmes em 30min, abaixo do threshold de 3."""
    base = datetime(2025, 1, 1, 8, 0)
    return pd.DataFrame({
        "TAG": ["CAM001", "CAM001"],
        "Alarme": ["AL01", "AL01"],
        "timestamp": [base, base + timedelta(minutes=10)],
        "Is_Dont_Go": [0, 0],
    })


@pytest.fixture
def df_above_threshold():
    """3 alarmes em 30min, atinge o threshold exato."""
    base = datetime(2025, 1, 1, 8, 0)
    return pd.DataFrame({
        "TAG": ["CAM001", "CAM001", "CAM001"],
        "Alarme": ["AL01", "AL01", "AL01"],
        "timestamp": [base, base + timedelta(minutes=10), base + timedelta(minutes=20)],
        "Is_Dont_Go": [0, 0, 1],
    })


@pytest.fixture
def df_outside_window():
    """3 alarmes mas espaçados além da janela de 30min."""
    base = datetime(2025, 1, 1, 8, 0)
    return pd.DataFrame({
        "TAG": ["CAM001", "CAM001", "CAM001"],
        "Alarme": ["AL01", "AL01", "AL01"],
        "timestamp": [base, base + timedelta(minutes=40), base + timedelta(minutes=80)],
        "Is_Dont_Go": [0, 0, 0],
    })


class TestApplySlidingWindow:

    def test_retorna_dataframe(self, df_below_threshold, rules):
        result = apply_sliding_window(df_below_threshold, rules)
        assert isinstance(result, pd.DataFrame)

    def test_coluna_derivada_presente(self, df_below_threshold, rules):
        result = apply_sliding_window(df_below_threshold, rules)
        assert "Is_Dont_Go_derived" in result.columns

    def test_derivada_contem_apenas_0_e_1(self, df_below_threshold, rules):
        result = apply_sliding_window(df_below_threshold, rules)
        assert set(result["Is_Dont_Go_derived"].unique()).issubset({0, 1})

    def test_abaixo_threshold_nao_gera_dont_go(self, df_below_threshold, rules):
        """2 alarmes em 30min não deve gerar Don't Go (threshold = 3)."""
        result = apply_sliding_window(df_below_threshold, rules)
        assert result["Is_Dont_Go_derived"].sum() == 0

    def test_no_threshold_gera_dont_go(self, df_above_threshold, rules):
        """3 alarmes em 30min deve gerar pelo menos 1 Don't Go."""
        result = apply_sliding_window(df_above_threshold, rules)
        assert result["Is_Dont_Go_derived"].sum() >= 1

    def test_fora_da_janela_nao_gera_dont_go(self, df_outside_window, rules):
        """Alarmes fora da janela de 30min não devem gerar Don't Go."""
        result = apply_sliding_window(df_outside_window, rules)
        assert result["Is_Dont_Go_derived"].sum() == 0

    def test_sem_leakage_janela_olha_so_para_tras(self, df_above_threshold, rules):
        """O primeiro alarme não pode usar eventos futuros."""
        result = apply_sliding_window(df_above_threshold, rules)
        # Primeiro registro tem apenas 1 alarme na janela (só ele mesmo)
        assert result["Is_Dont_Go_derived"].iloc[0] == 0


class TestCompareSignals:

    def test_retorna_dict_com_metricas(self):
        df = pd.DataFrame({
            "Is_Dont_Go": [1, 0, 1, 0, 0],
            "Is_Dont_Go_derived": [1, 0, 0, 0, 1],
        })
        result = compare_signals(df)
        assert isinstance(result, dict)
        assert "true_positives" in result
        assert "false_positives" in result
        assert "false_negatives" in result
        assert "precision" in result
        assert "recall" in result

    def test_perfeito_match(self):
        df = pd.DataFrame({
            "Is_Dont_Go": [1, 0, 1],
            "Is_Dont_Go_derived": [1, 0, 1],
        })
        result = compare_signals(df)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_sem_positivos_derivados(self):
        df = pd.DataFrame({
            "Is_Dont_Go": [1, 1, 0],
            "Is_Dont_Go_derived": [0, 0, 0],
        })
        result = compare_signals(df)
        assert result["false_negatives"] == 2
        assert result["true_positives"] == 0
