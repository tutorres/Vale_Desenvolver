"""
TDD: test_cleaner.py
Testes para src/cleaner.py

Convenção:
  - Cada teste tem um nome descritivo do comportamento esperado
  - Arrange / Act / Assert claramente separados
  - Fixtures compartilhadas no topo do arquivo
"""

import numpy as np
import pandas as pd
import pytest

from src.cleaner import clean_telemetry
from src.validator import QualityReport


# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def df_raw():
    """DataFrame mínimo com os 3 erros intencionais inseridos."""
    return pd.DataFrame({
        "TAG": ["CAM001", "CAM001", "ESC001", "CAM002", "CAM003", "CAM004"],
        "Alarme": ["AL01", "AL02", "AL01", "AL03", "AL01", "AL02"],
        "Id_Criticidade": [1, 2, 1, 3, 1, 2],
        "Criticidade": [
            "N??o Crítico",       # encoding bug variante 1
            "Não Cr??tico",       # encoding bug variante 2
            "Crítico",            # correto, não deve ser alterado
            "Não Crítico",        # correto, não deve ser alterado
            "N??o Crítico",       # encoding bug variante 1
            "Crítico",            # correto, não deve ser alterado
        ],
        "Valor": ["43,7999", "100,0", "0,5", "12,3456", "99,99", "NULL"],
        "Classe": ["Activate", "NULL", "NULL", "Inactive", "NULL", "Activate"],
        "Is_Dont_Go": [0, 0, 1, 0, 0, 0],
        "Tipo": ["Caminhao", "Caminhao", "Escavadeira", "Caminhao", "Caminhao", "Caminhao"],
    })


@pytest.fixture
def df_clean(df_raw):
    """DataFrame limpo, retornado pela função clean_telemetry."""
    df, _ = clean_telemetry(df_raw)
    return df


@pytest.fixture
def report(df_raw):
    """QualityReport retornado pela função clean_telemetry."""
    _, rpt = clean_telemetry(df_raw)
    return rpt


# ── CRITICIDADE ENCODING ──────────────────────────────────────────────────────

class TestCriticidadeEncoding:

    def test_variante_1_corrigida(self, df_clean):
        """N??o Crítico deve ser normalizado para Não Crítico."""
        assert "N??o Crítico" not in df_clean["Criticidade"].values

    def test_variante_2_corrigida(self, df_clean):
        """Não Cr??tico deve ser normalizado para Não Crítico."""
        assert "Não Cr??tico" not in df_clean["Criticidade"].values

    def test_valor_correto_preservado(self, df_clean):
        """Registros que já tinham Não Crítico correto não devem ser alterados."""
        assert "Não Crítico" in df_clean["Criticidade"].values

    def test_critico_preservado(self, df_clean):
        """Valor Crítico não deve ser tocado pela limpeza."""
        assert "Crítico" in df_clean["Criticidade"].values

    def test_report_conta_corretamente(self, report):
        """QualityReport deve contar exatamente os registros corrigidos."""
        assert report.criticidade_fixed == 3  # 2x variante1 + 1x variante2


# ── CLASSE NULL STRING ────────────────────────────────────────────────────────

class TestClasseNullString:

    def test_string_null_removida(self, df_clean):
        """String literal 'NULL' não deve existir após limpeza."""
        assert (df_clean["Classe"] == "NULL").sum() == 0

    def test_string_null_vira_nan(self, df_clean):
        """Onde havia 'NULL' deve haver np.nan."""
        assert df_clean["Classe"].isna().sum() == 3  # 3 registros com NULL

    def test_valores_validos_preservados(self, df_clean):
        """Valores Activate e Inactive não devem ser alterados."""
        assert "Activate" in df_clean["Classe"].values
        assert "Inactive" in df_clean["Classe"].values

    def test_report_conta_corretamente(self, report):
        """QualityReport deve contar os registros com NULL string corrigidos."""
        assert report.null_string_fixed == 3


# ── VALOR DECIMAL ─────────────────────────────────────────────────────────────

class TestValorDecimal:

    def test_tipo_float(self, df_clean):
        """Coluna Valor deve ser float64 após limpeza."""
        assert df_clean["Valor"].dtype == np.float64

    def test_sem_virgula(self, df_clean):
        """Nenhum valor deve conter vírgula."""
        assert df_clean["Valor"].astype(str).str.contains(",").sum() == 0

    def test_valor_convertido_corretamente(self, df_clean):
        """43,7999 deve ser convertido para 43.7999."""
        assert pytest.approx(df_clean["Valor"].iloc[0], rel=1e-4) == 43.7999

    def test_report_conta_corretamente(self, report):
        """QualityReport deve contar os registros com decimal corrigido."""
        assert report.decimal_fixed == 5  # todos os 5 registros com vírgula (exclui o NULL)


# ── VALOR STRING 'NULL' ─────────────────────────────────────────────────────

class TestValorNullString:

    def test_null_vira_nan(self, df_clean):
        """String literal 'NULL' em Valor deve virar np.nan, não quebrar o cast para float."""
        assert df_clean["Valor"].isna().sum() == 1

    def test_tipo_float_mesmo_com_null(self, df_clean):
        """Coluna Valor deve permanecer float64 mesmo com 'NULL' presente."""
        assert df_clean["Valor"].dtype == np.float64

    def test_report_conta_valor_null(self, report):
        """QualityReport deve contar os registros com 'NULL' string em Valor."""
        assert report.valor_null_fixed == 1


# ── INTEGRIDADE GERAL ─────────────────────────────────────────────────────────

class TestIntegridade:

    def test_numero_de_linhas_preservado(self, df_raw, df_clean):
        """Limpeza não deve dropar linhas."""
        assert len(df_clean) == len(df_raw)

    def test_colunas_preservadas(self, df_raw, df_clean):
        """Todas as colunas originais devem estar presentes."""
        assert set(df_raw.columns).issubset(set(df_clean.columns))

    def test_is_dont_go_intacto(self, df_clean):
        """Coluna Is_Dont_Go não deve ser alterada pela limpeza."""
        assert set(df_clean["Is_Dont_Go"].unique()).issubset({0, 1})

    def test_retorno_e_tupla(self, df_raw):
        """clean_telemetry deve retornar tupla (DataFrame, QualityReport)."""
        result = clean_telemetry(df_raw)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], QualityReport)
