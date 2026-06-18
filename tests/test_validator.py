"""
TDD — test_validator.py
Testes para src/validator.py
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.validator import QualityReport, validate_apontamentos


@pytest.fixture
def df_apon_clean():
    """Apontamentos sem problemas."""
    base = datetime(2025, 1, 1, 8, 0)
    return pd.DataFrame({
        "Tag": ["CAM001", "CAM002"],
        "Inicio": [base, base + timedelta(hours=1)],
        "Fim": [base + timedelta(hours=8), base + timedelta(hours=9)],
        "Classe": ["Operando", "Parado"],
    })


@pytest.fixture
def df_apon_with_errors():
    """Apontamentos com Inicio > Fim em 2 linhas."""
    base = datetime(2025, 1, 1, 8, 0)
    return pd.DataFrame({
        "Tag": ["CAM001", "CAM002", "ESC001"],
        "Inicio": [
            base + timedelta(hours=9),  # Inicio > Fim (erro)
            base,
            base + timedelta(hours=5),  # Inicio > Fim (erro)
        ],
        "Fim": [
            base,
            base + timedelta(hours=8),
            base,
        ],
        "Classe": ["Operando", "Parado", "Operando"],
    })


class TestQualityReport:

    def test_e_dataclass(self):
        """QualityReport deve ser instanciável com campos esperados."""
        rpt = QualityReport(
            criticidade_fixed=0,
            null_string_fixed=0,
            decimal_fixed=0,
            valor_null_fixed=0,
            inicio_after_fim=0,
            tag_mismatches=[],
            duplicate_rows=0,
            timestamp_gaps=[],
            summary="ok",
        )
        assert rpt.criticidade_fixed == 0

    def test_summary_nao_vazio(self):
        """summary deve ser string não vazia."""
        rpt = QualityReport(
            criticidade_fixed=5,
            null_string_fixed=3,
            decimal_fixed=10,
            valor_null_fixed=1,
            inicio_after_fim=2,
            tag_mismatches=["CAM999"],
            duplicate_rows=0,
            timestamp_gaps=[],
            summary="relatório",
        )
        assert isinstance(rpt.summary, str)
        assert len(rpt.summary) > 0


class TestValidateApontamentos:

    def test_retorna_quality_report(self, df_apon_clean):
        result = validate_apontamentos(df_apon_clean)
        assert isinstance(result, QualityReport)

    def test_detecta_inicio_after_fim(self, df_apon_with_errors):
        result = validate_apontamentos(df_apon_with_errors)
        assert result.inicio_after_fim == 2

    def test_zero_erros_em_dados_limpos(self, df_apon_clean):
        result = validate_apontamentos(df_apon_clean)
        assert result.inicio_after_fim == 0

    def test_summary_gerado(self, df_apon_with_errors):
        result = validate_apontamentos(df_apon_with_errors)
        assert len(result.summary) > 0
