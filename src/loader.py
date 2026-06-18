from pathlib import Path

import pandas as pd

from src.utils import log

logger = log(__name__)

MONTHS = ["jan", "feb", "mar", "abr", "may", "jun"]

REQUIRED_COLUMNS = {
    "TAG", "Alarme", "Id_Criticidade", "Criticidade",
    "Valor", "Classe", "Is_Dont_Go", "Tipo", "Data_Evento",
}

# Low-cardinality string columns — cast to category to keep the 37M-row
# real dataset within memory (object dtype alone needs ~30GB; see
# spec/DECISIONS.md).
CATEGORICAL_COLUMNS = [
    "TAG", "Alarme", "Criticidade", "Classe", "Tipo",
    "Localidade", "Tag_Frota", "Inicio_Turno", "Fim_Turno",
    "Nome_Operador_Anon", "Matricula_Operador_Hash",
]


class SchemaError(Exception):
    pass


def load_telemetry(data_dir: str) -> pd.DataFrame:
    """Load and concatenate the 6 monthly telemetry parquet files.

    Args:
        data_dir: Path to directory containing telemetry_<month>.parquet files.

    Returns:
        Concatenated DataFrame with a '_source_month' column added.

    Raises:
        FileNotFoundError: If any of the 6 monthly files is missing.
        SchemaError: If any file is missing a required column.
    """
    base = Path(data_dir)
    frames = []

    for month in MONTHS:
        path = base / f"telemetry_{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_parquet(path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SchemaError(f"telemetry_{month}.parquet missing columns: {missing}")

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype("category")

        df["_source_month"] = month
        frames.append(df)
        logger.info("Loaded %s — %d rows", path.name, len(df))

    # pd.concat reverts a categorical column to object if the per-frame
    # category sets differ (they do across months for real data — e.g.
    # different alarms or operators appear in different months). Unify
    # each column onto one shared CategoricalDtype first so concat keeps
    # it categorical instead of materializing a giant object column.
    present_cat_cols = [c for c in CATEGORICAL_COLUMNS if c in frames[0].columns]
    for col in present_cat_cols:
        union_categories = sorted(
            set().union(*(frame[col].cat.categories for frame in frames))
        )
        shared_dtype = pd.CategoricalDtype(categories=union_categories)
        for frame in frames:
            frame[col] = frame[col].astype(shared_dtype)

    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"Data_Evento": "timestamp"})
    logger.info("Total rows loaded: %d", len(result))
    return result
