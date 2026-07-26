"""
Baselines de referência para o modelo preditivo Don't Go.

Fornece modelos triviais e um baseline heurístico de domínio, avaliados no
mesmo split temporal e com as mesmas FEATURE_COLS do XGBoost de referência
(src/model.py), para responder às lacunas CM 4.2 (baseline) e CM 5.1 (tabela
comparativa) do Estudo Guiado.

Baselines implementados:
  * DummyClassifier(strategy="most_frequent"): prevê sempre a classe majoritária.
  * DummyClassifier(strategy="stratified"): prevê aleatoriamente respeitando
    a proporção de classes do treino (seed 42).
  * Heurístico de domínio. REGRA: prevê Don't Go (positivo) quando houve ao menos
    um alarme crítico na última hora, isto é `critical_alarm_count_1h > 0`. É a
    regra "de bolso" que um dispatcher aplicaria sem modelo. O score contínuo
    usado para AUC-ROC/AUC-PR é o próprio `critical_alarm_count_1h`.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import FEATURE_COLS, SEED, _TEST_START, _TRAIN_END
from src.utils import log, set_seeds

logger = log(__name__)

# Coluna que dispara o baseline heurístico de domínio.
HEURISTIC_FEATURE = "critical_alarm_count_1h"


def temporal_split(features: pd.DataFrame, label_col: str):
    """Aplica o split temporal idêntico ao src/model.py.

    Treino: timestamp <= 2025-04-30 23:59:59.
    Teste:  timestamp >= 2025-05-01.

    Faz fallback para 80/20 sequencial quando os dados sintéticos não cobrem
    Mai-Jun (mesma lógica de train_model), para os testes com dados de brinquedo.
    """
    train_mask = features["timestamp"] <= _TRAIN_END
    test_mask = features["timestamp"] >= _TEST_START

    if test_mask.sum() == 0:
        split = int(len(features) * 0.8)
        train_mask = pd.Series([True] * split + [False] * (len(features) - split),
                               index=features.index)
        test_mask = ~train_mask

    X_train = features.loc[train_mask, FEATURE_COLS]
    y_train = features.loc[train_mask, label_col]
    X_test = features.loc[test_mask, FEATURE_COLS]
    y_test = features.loc[test_mask, label_col]
    return X_train, y_train, X_test, y_test


def _metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    """Calcula F1, precision, recall, AUC-ROC e AUC-PR.

    AUC-ROC/AUC-PR só são definidos quando há as duas classes no teste; caso
    contrário retorna 0.0 (evita exceção com rótulo constante nos testes).
    """
    y_true = np.asarray(y_true)
    both_classes = len(np.unique(y_true)) > 1
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if both_classes else 0.0,
        "auc_pr": float(average_precision_score(y_true, y_score)) if both_classes else 0.0,
    }


def evaluate_dummy(X_train, y_train, X_test, y_test, strategy: str,
                   seed: int = SEED) -> Dict[str, float]:
    """Treina e avalia um DummyClassifier no split fornecido.

    Args:
        strategy: "most_frequent" ou "stratified".
        seed: semente para reprodutibilidade (relevante em "stratified").
    """
    set_seeds(seed)
    clf = DummyClassifier(strategy=strategy, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # predict_proba do dummy respeita a estratégia; usamos prob da classe 1.
    if 1 in clf.classes_:
        pos_idx = list(clf.classes_).index(1)
        y_score = clf.predict_proba(X_test)[:, pos_idx]
    else:
        y_score = np.zeros(len(X_test))
    return _metrics(y_test, y_pred, y_score)


def evaluate_heuristic(X_test, y_test) -> Dict[str, float]:
    """Avalia o baseline heurístico de domínio.

    REGRA: positivo (Don't Go previsto) sse `critical_alarm_count_1h > 0`.
    Score contínuo para AUC = valor de `critical_alarm_count_1h`.
    """
    scores = X_test[HEURISTIC_FEATURE].to_numpy()
    y_pred = (scores > 0).astype(int)
    return _metrics(y_test, y_pred, scores)


def run_baselines(features: pd.DataFrame, label_col: str = "label_4h",
                  seed: int = SEED) -> Dict[str, Dict[str, float]]:
    """Roda os três baselines no split temporal e devolve as métricas.

    Returns:
        dict {nome_baseline: {f1, precision, recall, auc_roc, auc_pr}}.
    """
    X_train, y_train, X_test, y_test = temporal_split(features, label_col)
    results = {
        "dummy_most_frequent": evaluate_dummy(X_train, y_train, X_test, y_test,
                                              "most_frequent", seed),
        "dummy_stratified": evaluate_dummy(X_train, y_train, X_test, y_test,
                                           "stratified", seed),
        "heuristic_critical_1h": evaluate_heuristic(X_test, y_test),
    }
    for name, m in results.items():
        logger.info("Baseline %s: F1=%.3f recall=%.3f AUC-PR=%.3f",
                    name, m["f1"], m["recall"], m["auc_pr"])
    return results
