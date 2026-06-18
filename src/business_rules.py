from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils import log

logger = log(__name__)


def load_rules(rules_path: str) -> Dict[str, dict]:
    """Load sliding window rules from the business rules Excel file.

    For alarms with multiple severity rows, keeps the most restrictive rule
    (lowest qty; ties broken by lowest window_minutes).

    Args:
        rules_path: Path to Alarmes - Regra de Negocio.xlsx.

    Returns:
        Dict mapping alarm name → {"qty": int, "window_minutes": int}.
    """
    df = pd.read_excel(rules_path)
    # Sort so most restrictive (lowest qty, then lowest window) comes first
    df_sorted = df.sort_values(["QTD", "TEMPO"], ascending=True)
    rules: Dict[str, dict] = {}
    for _, row in df_sorted.iterrows():
        alarm = str(row["EVENTO"])
        if alarm not in rules:
            rules[alarm] = {"qty": int(row["QTD"]), "window_minutes": int(row["TEMPO"])}
    logger.info("Loaded %d alarm rules", len(rules))
    return rules


def apply_sliding_window(df: pd.DataFrame, rules: Dict[str, dict]) -> pd.DataFrame:
    """Apply per-alarm sliding window rules to derive Is_Dont_Go signal.

    For each TAG and each Alarme, counts how many alarms of that type fell
    within the configured time window ending at each event. If count >= qty
    threshold the event is flagged as Is_Dont_Go_derived = 1.

    No leakage: window is closed at t (inclusive) and looks backward only.

    Args:
        df: Cleaned telemetry DataFrame with TAG, Alarme, timestamp columns.
        rules: Dict from load_rules().

    Returns:
        Input DataFrame with 'Is_Dont_Go_derived' column added (0/1 int).
    """
    df = df.copy().sort_values(["TAG", "Alarme", "timestamp"]).reset_index(drop=True)
    derived = np.zeros(len(df), dtype=int)

    for (tag, alarm), group in df.groupby(["TAG", "Alarme"], sort=False):
        if alarm not in rules:
            continue
        qty_threshold = rules[alarm]["qty"]
        window_offset = f"{rules[alarm]['window_minutes']}min"

        grp = group.sort_values("timestamp")
        ts_index = pd.DatetimeIndex(grp["timestamp"])
        ones = pd.Series(1.0, index=ts_index)
        counts = ones.rolling(window_offset, min_periods=1, closed="right").sum()

        flagged = (counts >= qty_threshold).values
        derived[grp.index[flagged]] = 1

    df["Is_Dont_Go_derived"] = derived
    logger.info(
        "Sliding window applied — Don't Go events derived: %d", int(derived.sum())
    )
    return df


def compare_signals(df: pd.DataFrame) -> dict:
    """Compare rule-derived Don't Go signal against the registered signal.

    Args:
        df: DataFrame with both 'Is_Dont_Go' and 'Is_Dont_Go_derived' columns.

    Returns:
        Dict with tp, fp, fn, tn, precision, recall.
    """
    actual = df["Is_Dont_Go"]
    derived = df["Is_Dont_Go_derived"]

    tp = int(((actual == 1) & (derived == 1)).sum())
    fp = int(((actual == 0) & (derived == 1)).sum())
    fn = int(((actual == 1) & (derived == 0)).sum())
    tn = int(((actual == 0) & (derived == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
    }
