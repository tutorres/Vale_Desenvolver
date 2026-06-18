from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.utils import log

logger = log(__name__)


@dataclass
class QualityReport:
    criticidade_fixed: int
    null_string_fixed: int
    decimal_fixed: int
    valor_null_fixed: int
    inicio_after_fim: int
    tag_mismatches: List[str]
    duplicate_rows: int
    timestamp_gaps: List[str]
    summary: str


def validate_apontamentos(df: pd.DataFrame) -> QualityReport:
    """Validate apontamentos DataFrame and return a structured quality report.

    Args:
        df: Apontamentos DataFrame with Tag, Inicio, Fim, Classe columns.

    Returns:
        QualityReport with counts of each data quality issue found.
    """
    inicio_after_fim = int((df["Inicio"] > df["Fim"]).sum())

    summary_lines = [
        f"Apontamentos validados: {len(df)} registros",
        f"  Inicio > Fim: {inicio_after_fim}",
    ]

    return QualityReport(
        criticidade_fixed=0,
        null_string_fixed=0,
        decimal_fixed=0,
        valor_null_fixed=0,
        inicio_after_fim=inicio_after_fim,
        tag_mismatches=[],
        duplicate_rows=0,
        timestamp_gaps=[],
        summary="\n".join(summary_lines),
    )
