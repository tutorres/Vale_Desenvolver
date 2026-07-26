from typing import List

import numpy as np
import pandas as pd

from src.utils import log

logger = log(__name__)

WINDOWS = {
    "alarm_count_15m": pd.Timedelta("15min"),
    "alarm_count_30m": pd.Timedelta("30min"),
    "alarm_count_1h": pd.Timedelta("1h"),
    "alarm_count_2h": pd.Timedelta("2h"),
    "alarm_count_4h": pd.Timedelta("4h"),
}

LABEL_HORIZONS = {
    "label_1h": pd.Timedelta("1h"),
    "label_2h": pd.Timedelta("2h"),
    "label_4h": pd.Timedelta("4h"),
}

_DEFAULT_TIME_SINCE = float(pd.Timedelta("4h").total_seconds() / 60)

_OUTPUT_COLS = (
    ["TAG", "timestamp"]
    + list(WINDOWS.keys())
    + ["critical_alarm_count_1h", "time_since_last_critical",
       "tipo_caminhao", "current_state"]
    + list(LABEL_HORIZONS.keys())
)


def build_features(df_tele: pd.DataFrame, df_apon: pd.DataFrame) -> pd.DataFrame:
    """Build the ML feature matrix from cleaned telemetry and apontamentos.

    Features computed per row (each alarm event is one row):
        - Rolling alarm counts over 5 windows (15m, 30m, 1h, 2h, 4h)
        - Critical alarm count in last 1h
        - Minutes since last critical alarm (Id_Criticidade == 1)
        - Equipment type flag (1 = Caminhao, 0 = Escavadeira)
        - Current operational state from apontamentos
        - Labels: Don't Go within next 1h / 2h / 4h

    No data leakage: all lookback windows reference timestamps strictly
    before or equal to each event's timestamp.

    Args:
        df_tele: Cleaned telemetry DataFrame.
        df_apon: Apontamentos DataFrame with Tag, Inicio, Fim, Classe columns.

    Returns:
        Feature DataFrame with columns defined in MODULE_CONTRACTS.md.
    """
    # Only these 5 columns are read below. Selecting them before the
    # copy/sort avoids dragging the other ~14 columns (several large
    # categoricals) through a 37M-row sort/groupby.
    needed_cols = ["TAG", "timestamp", "Tipo", "Id_Criticidade", "Is_Dont_Go"]
    df = df_tele[needed_cols].copy().sort_values(["TAG", "timestamp"]).reset_index(drop=True)
    apon_tag_col = "Tag" if "Tag" in df_apon.columns else "TAG"

    # current_state has a handful of possible values ("Operando", "Parado", ...).
    # Fixing the categories up front keeps it category dtype across all TAGs
    # (pd.concat reverts to object if per-frame categories differ, as discovered
    # with TAG/Alarme in loader.py).
    state_categories = pd.CategoricalDtype(
        categories=sorted(set(df_apon["Classe"].dropna().unique()) | {"Desconhecido"})
    )

    tag_frames: List[pd.DataFrame] = []

    for tag, grp in df.groupby("TAG", sort=False):
        grp = grp.sort_values("timestamp").copy()
        tipo = 1 if (grp["Tipo"].iloc[0] == "Caminhao") else 0

        grp_ts = grp.set_index("timestamp")
        ts_index: pd.DatetimeIndex = grp_ts.index

        # --- rolling alarm counts (vectorized) ---
        ones = pd.Series(1.0, index=ts_index)
        for col, delta in WINDOWS.items():
            grp_ts[col] = (
                ones.rolling(delta, min_periods=0, closed="right").sum().astype(int)
            )

        # --- critical alarm count in last 1h ---
        is_critical = (grp_ts["Id_Criticidade"] == 1).astype(float)
        grp_ts["critical_alarm_count_1h"] = (
            is_critical.rolling("1h", min_periods=0, closed="right").sum().astype(int)
        )

        # --- time since last critical alarm (minutes) ---
        last_crit_ts = (
            ts_index.to_series()
            .where(grp_ts["Id_Criticidade"] == 1)  # NaT for non-critical rows
            .ffill()
        )
        time_since = (ts_index.to_series() - last_crit_ts).dt.total_seconds() / 60
        grp_ts["time_since_last_critical"] = time_since.fillna(_DEFAULT_TIME_SINCE).values

        # --- equipment type ---
        grp_ts["tipo_caminhao"] = tipo

        # --- labels: forward-looking Don't Go (no leakage, vectorized) ---
        t_arr = ts_index.values
        dg_arr = t_arr[grp_ts["Is_Dont_Go"].values == 1]

        for label_col, horizon in LABEL_HORIZONS.items():
            if len(dg_arr) == 0:
                grp_ts[label_col] = 0
            else:
                h_ns = np.timedelta64(int(horizon.total_seconds()), "s")
                t_end = t_arr + h_ns
                lo = np.searchsorted(dg_arr, t_arr, side="right")
                hi = np.searchsorted(dg_arr, t_end, side="right")
                grp_ts[label_col] = (hi > lo).astype(int)

        # --- operational state (vectorized per-TAG lookup) ---
        grp_ts["current_state"] = pd.Categorical(
            _lookup_states(tag, ts_index, df_apon, apon_tag_col), dtype=state_categories
        )

        tag_frames.append(grp_ts.reset_index()[_OUTPUT_COLS])

    result = pd.concat(tag_frames, ignore_index=True)
    logger.info("Features built: %d rows, %d columns", len(result), len(result.columns))
    return result


def _lookup_states(
    tag: str,
    timestamps: pd.DatetimeIndex,
    df_apon: pd.DataFrame,
    tag_col: str,
) -> np.ndarray:
    """Vectorized operational-state lookup for all timestamps of one TAG."""
    apon_tag = df_apon[df_apon[tag_col] == tag].sort_values("Inicio")

    if len(apon_tag) == 0:
        return np.full(len(timestamps), "Desconhecido", dtype=object)

    inicio = apon_tag["Inicio"].values
    fim = apon_tag["Fim"].values
    classes = apon_tag["Classe"].values

    ts_arr = timestamps.values
    idx = np.searchsorted(inicio, ts_arr, side="right") - 1

    states = np.full(len(ts_arr), "Desconhecido", dtype=object)
    valid = idx >= 0
    if valid.any():
        valid_idx = idx[valid]
        in_range = ts_arr[valid] <= fim[valid_idx]
        states[valid] = np.where(in_range, classes[valid_idx], "Desconhecido")

    return states
