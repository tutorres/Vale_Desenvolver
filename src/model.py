import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.utils import log, set_seeds

logger = log(__name__)

FEATURE_COLS = [
    "alarm_count_15m", "alarm_count_30m", "alarm_count_1h",
    "alarm_count_2h", "alarm_count_4h",
    "critical_alarm_count_1h", "time_since_last_critical",
    "tipo_caminhao",
]

# Train = Jan–Apr, Test = May–Jun
_TRAIN_END = pd.Timestamp("2025-04-30 23:59:59")
_TEST_START = pd.Timestamp("2025-05-01 00:00:00")

THRESHOLDS = [0.3, 0.5, 0.7]

SEED = 42


def train_model(
    features: pd.DataFrame, label_col: str
) -> Tuple[XGBClassifier, Dict]:
    """Train an XGBoost classifier with temporal train/test split.

    Train: timestamps up to and including April.
    Test:  timestamps from May onwards.

    Handles class imbalance via scale_pos_weight. Reports F1, precision,
    recall and AUC — never accuracy as the primary metric.

    Args:
        features: Feature DataFrame from build_features().
        label_col: Target column name ('label_1h', 'label_2h', or 'label_4h').

    Returns:
        Tuple of (trained XGBClassifier, metrics dict).
    """
    set_seeds(SEED)

    train_mask = features["timestamp"] <= _TRAIN_END
    test_mask = features["timestamp"] >= _TEST_START

    # Fall back to 80/20 split when synthetic data doesn't span May-Jun
    if test_mask.sum() == 0:
        split = int(len(features) * 0.8)
        train_mask = pd.Series([True] * split + [False] * (len(features) - split))
        test_mask = ~train_mask

    X_train = features.loc[train_mask, FEATURE_COLS]
    y_train = features.loc[train_mask, label_col]
    X_test = features.loc[test_mask, FEATURE_COLS]
    y_test = features.loc[test_mask, label_col]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        random_state=SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
        "threshold": 0.5,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }

    logger.info(
        "Model trained — F1=%.3f precision=%.3f recall=%.3f AUC=%.3f",
        metrics["f1"], metrics["precision"], metrics["recall"], metrics["auc"],
    )
    return model, metrics


def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> List[Dict]:
    """Evaluate model at multiple decision thresholds.

    Args:
        model: Trained XGBClassifier.
        X_test: Feature matrix for the test set.
        y_test: True labels for the test set.

    Returns:
        List of dicts, one per threshold, each containing f1, precision,
        recall, and threshold — never accuracy alone.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    results = []
    for thresh in THRESHOLDS:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            "threshold": thresh,
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        })
    return results


def save_model(model: XGBClassifier, path: str = "model.pkl") -> None:
    """Serialize trained model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def save_metrics(metrics: Dict, path: str = "metrics.json") -> None:
    """Write metrics dict to JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)
