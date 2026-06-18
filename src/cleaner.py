from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import log
from src.validator import QualityReport

logger = log(__name__)

# Regex matches "N??o Crítico", "Não Cr??tico", and other encoding variants
_CRITICIDADE_PATTERN = r"N.{1,2}o Cr.{1,2}tico"
_CRITICIDADE_CORRECT = "Não Crítico"


def clean_telemetry(df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
    """Fix the three intentional data quality errors in the telemetry DataFrame.

    Errors fixed:
        1. UTF-8 encoding corruption in 'Criticidade' column.
        2. Literal string "NULL" in 'Classe' column (replaced with np.nan).
        3. Comma decimal separator in 'Valor' column (cast to float64).

    Args:
        df: Raw telemetry DataFrame from load_telemetry.

    Returns:
        Tuple of (cleaned DataFrame, QualityReport with fix counts).
    """
    df = df.copy()

    # 1 — Criticidade encoding
    original_crit = df["Criticidade"].copy()
    df["Criticidade"] = df["Criticidade"].str.replace(
        _CRITICIDADE_PATTERN, _CRITICIDADE_CORRECT, regex=True
    )
    criticidade_fixed = int((df["Criticidade"] != original_crit).sum())
    logger.info("Criticidade encoding fixed: %d records", criticidade_fixed)

    # 2 — NULL string → np.nan in Classe
    mask_null = df["Classe"] == "NULL"
    null_string_fixed = int(mask_null.sum())
    df.loc[mask_null, "Classe"] = np.nan
    logger.info("NULL string replaced: %d records", null_string_fixed)

    # 3a — NULL string literal in Valor → np.nan (missing sensor reading)
    mask_valor_null = df["Valor"].astype(str) == "NULL"
    valor_null_fixed = int(mask_valor_null.sum())
    df.loc[mask_valor_null, "Valor"] = np.nan
    logger.info("Valor NULL string replaced: %d records", valor_null_fixed)

    # 3b — Comma decimal separator in Valor → float64
    decimal_fixed = int(df["Valor"].astype(str).str.contains(",", na=False).sum())
    df["Valor"] = df["Valor"].astype(str).str.replace(",", ".", regex=False).astype(float)
    logger.info("Decimal separator fixed: %d records", decimal_fixed)

    report = QualityReport(
        criticidade_fixed=criticidade_fixed,
        null_string_fixed=null_string_fixed,
        decimal_fixed=decimal_fixed,
        valor_null_fixed=valor_null_fixed,
        inicio_after_fim=0,
        tag_mismatches=[],
        duplicate_rows=0,
        timestamp_gaps=[],
        summary=(
            f"Telemetria limpa: {len(df)} registros\n"
            f"  Criticidade encoding corrigido: {criticidade_fixed}\n"
            f"  NULL string → NaN (Classe): {null_string_fixed}\n"
            f"  NULL string → NaN (Valor): {valor_null_fixed}\n"
            f"  Decimal separator corrigido: {decimal_fixed}"
        ),
    )

    return df, report
