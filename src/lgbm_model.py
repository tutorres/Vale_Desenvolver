"""
Segunda abordagem de modelagem: LightGBM (CM 4.3).

Usa exatamente as mesmas FEATURE_COLS e o mesmo split temporal do XGBoost de
referência (src/model.py), para garantir comparabilidade justa na tabela
comparativa (CM 5.1). Trata o forte desbalanceamento via scale_pos_weight
(equivalente a is_unbalance, porém explícito e reprodutível).
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.baselines import temporal_split
from src.model import FEATURE_COLS, SEED, THRESHOLDS
from src.utils import log, set_seeds

logger = log(__name__)


def train_lgbm(features: pd.DataFrame, label_col: str,
               seed: int = SEED) -> Tuple[LGBMClassifier, Dict]:
    """Treina um LGBMClassifier com o split temporal padrão.

    Trata desbalanceamento com scale_pos_weight = n_neg / n_pos calculado no
    treino. Reporta F1, precision, recall, AUC-ROC e AUC-PR no threshold 0.5.

    Returns:
        (modelo treinado, dict de métricas).
    """
    set_seeds(seed)
    X_train, y_train, X_test, y_test = temporal_split(features, label_col)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        scale_pos_weight=scale_pos,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    both = len(np.unique(y_test)) > 1

    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)) if both else 0.0,
        "auc_pr": float(average_precision_score(y_test, y_prob)) if both else 0.0,
        "threshold": 0.5,
        "scale_pos_weight": scale_pos,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
    }
    logger.info(
        "LightGBM %s — F1=%.3f precision=%.3f recall=%.3f AUC-ROC=%.3f AUC-PR=%.3f",
        label_col, metrics["f1"], metrics["precision"], metrics["recall"],
        metrics["auc_roc"], metrics["auc_pr"],
    )
    return model, metrics


def evaluate_lgbm(model: LGBMClassifier, X_test: pd.DataFrame,
                  y_test: pd.Series, thresholds: List[float] = None) -> List[Dict]:
    """Avalia o modelo em múltiplos thresholds de decisão.

    Returns:
        Lista de dicts (um por threshold) com f1, precision, recall.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    y_prob = model.predict_proba(X_test)[:, 1]
    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            "threshold": thresh,
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        })
    return results
