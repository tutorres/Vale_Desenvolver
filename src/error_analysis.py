"""
error_analysis.py: Matriz de confusao e analise de erros do XGBoost (CM 5.2).

Usa o modelo ja treinado (model.pkl) e o conjunto de TESTE (timestamp >= 2025-05-01)
de data/features.parquet. NAO retreina nada.

Foco operacional:
    - Falso Negativo (FN): Don't Go que o modelo NAO antecipou -> parada nao planejada.
    - Falso Positivo (FP): alerta sem Don't Go -> inspecao desnecessaria.

Funcoes puras (confusion_matrix_2x2, false_negative_distribution, recall_by_month)
operam sobre arrays/DataFrames e sao testaveis sem o modelo real.
"""

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.model import FEATURE_COLS
from src.utils import log

logger = log(__name__)

# Split temporal identico a src/model.py.
_TEST_START = pd.Timestamp("2025-05-01 00:00:00")

# Threshold preferido no relatorio.
REPORT_THRESHOLD = 0.7

LABEL_COL = "label_4h"

# Colunas lidas do parquet: 8 features + timestamp + label + estado.
_READ_COLS = list(dict.fromkeys(FEATURE_COLS + ["timestamp", LABEL_COL, "current_state"]))

# Faixas de alarm_count_4h para agrupar os falsos negativos.
_ALARM_BINS = [-0.5, 5, 10, 20, np.inf]
_ALARM_LABELS = ["0-5", "6-10", "11-20", "21+"]


def confusion_matrix_2x2(y_true, y_pred) -> Dict[str, int]:
    """Matriz de confusao 2x2 como dict {tp, fp, fn, tn}.

    Args:
        y_true: rotulos verdadeiros (0/1).
        y_pred: predicoes binarias (0/1).

    Returns:
        Dict com contagens inteiras tp, fp, fn, tn.
    """
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[0, 1],
    ).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def false_negative_distribution(df: pd.DataFrame, y_true, y_pred) -> Dict:
    """Distribuicao dos falsos negativos por tipo, estado e faixa de alarmes.

    Um FN e uma linha com label_4h == 1 (Don't Go real) que o modelo previu 0.

    Args:
        df: DataFrame do teste, alinhado por posicao com y_true/y_pred. Precisa
            das colunas tipo_caminhao, current_state e alarm_count_4h.
        y_true: rotulos verdadeiros (0/1).
        y_pred: predicoes binarias (0/1).

    Returns:
        Dict com total_fn e distribuicoes by_tipo_caminhao, by_current_state e
        by_alarm_count_4h_bin.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn = df.loc[fn_mask]
    total_fn = int(fn_mask.sum())

    # Por tipo de equipamento (1 = caminhao, 0 = escavadeira).
    n_caminhao = int((fn["tipo_caminhao"] == 1).sum())
    n_escav = int((fn["tipo_caminhao"] == 0).sum())
    by_tipo = {"caminhao": n_caminhao, "escavadeira": n_escav}

    # Por estado operacional.
    state_counts = fn["current_state"].astype(str).value_counts()
    by_state = {str(k): int(v) for k, v in state_counts.items()}

    # Por faixa de alarm_count_4h.
    bins = pd.cut(fn["alarm_count_4h"], bins=_ALARM_BINS, labels=_ALARM_LABELS)
    bin_counts = bins.value_counts()
    by_bin = {str(lbl): int(bin_counts.get(lbl, 0)) for lbl in _ALARM_LABELS}

    return {
        "total_fn": total_fn,
        "by_tipo_caminhao": by_tipo,
        "by_current_state": by_state,
        "by_alarm_count_4h_bin": by_bin,
    }


def recall_by_month(df: pd.DataFrame, y_true, y_pred) -> Dict[str, Dict]:
    """Recall por mes do teste (maio vs junho) para checar drift temporal.

    Args:
        df: DataFrame do teste com coluna timestamp, alinhado com y_true/y_pred.
        y_true: rotulos verdadeiros (0/1).
        y_pred: predicoes binarias (0/1).

    Returns:
        Dict {"2025-05": {...}, ...} com recall, tp, fn e positives por mes.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    months = pd.to_datetime(df["timestamp"]).dt.to_period("M").astype(str).values

    out: Dict[str, Dict] = {}
    for m in sorted(set(months)):
        m_mask = months == m
        yt = y_true[m_mask]
        yp = y_pred[m_mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        positives = tp + fn
        recall = tp / positives if positives > 0 else 0.0
        out[m] = {"recall": recall, "tp": tp, "fn": fn, "positives": positives}
    return out


def load_test_set(parquet_path: str) -> pd.DataFrame:
    """Le apenas as colunas necessarias do conjunto de TESTE (>= 2025-05-01)."""
    df = pd.read_parquet(
        parquet_path,
        columns=_READ_COLS,
        filters=[("timestamp", ">=", _TEST_START)],
    )
    logger.info("Test set carregado: %d linhas x %d colunas", len(df), df.shape[1])
    return df


def load_model(model_path: str):
    """Carrega o XGBoost serializado via pickle."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def plot_confusion_matrix(cm: Dict[str, int], threshold: float, out_path: str) -> None:
    """Salva a matriz de confusao anotada com impacto operacional."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    impact = np.array([
        ["Sem Don't Go\n(correto)", "Inspecao\ndesnecessaria (FP)"],
        ["Parada nao\nplanejada (FN)", "Don't Go\nantecipado (TP)"],
    ])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(grid, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Previsto: Normal", "Previsto: Don't Go"])
    ax.set_yticklabels(["Real: Normal", "Real: Don't Go"])
    ax.set_xlabel("Predicao do modelo")
    ax.set_ylabel("Situacao real")
    ax.set_title(
        f"Matriz de Confusao: XGBoost, label_4h, threshold {threshold:g}\n"
        f"(conjunto de teste, maio-junho 2025)"
    )

    vmax = grid.max()
    for i in range(2):
        for j in range(2):
            color = "white" if grid[i, j] > vmax * 0.5 else "black"
            ax.text(
                j, i,
                f"{grid[i, j]:,}\n{impact[i, j]}",
                ha="center", va="center", color=color, fontsize=11,
            )

    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    ax.text(
        0.5, -0.22,
        f"FN = parada nao planejada  |  FP = inspecao desnecessaria\n"
        f"Recall = {recall:.1%}  |  Precision = {precision:.1%}",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="N. de eventos")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva em %s", out_path)


def run(
    parquet_path: str = "data/features.parquet",
    model_path: str = "model.pkl",
    threshold: float = REPORT_THRESHOLD,
    figure_path: str = "figures/confusion_matrix_4h.png",
    json_path: str = "analysis/error_analysis.json",
) -> Dict:
    """Pipeline completo: matriz de confusao + analise de erros do XGBoost.

    NAO retreina o modelo. Aplica model.pkl no conjunto de teste, calcula a
    matriz de confusao no threshold informado, a distribuicao dos falsos
    negativos e o recall por mes. Salva figura e JSON.
    """
    model = load_model(model_path)
    df = load_test_set(parquet_path)

    y_true = df[LABEL_COL].to_numpy().astype(int)
    y_prob = model.predict_proba(df[FEATURE_COLS])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix_2x2(y_true, y_pred)
    fn_dist = false_negative_distribution(df, y_true, y_pred)
    monthly = recall_by_month(df, y_true, y_pred)

    n_test = len(df)
    n_pos = int((y_true == 1).sum())
    recall = cm["tp"] / n_pos if n_pos else 0.0
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0

    result = {
        "model": "XGBoost",
        "label": LABEL_COL,
        "threshold": threshold,
        "n_test": n_test,
        "n_positives": n_pos,
        "confusion_matrix": cm,
        "recall": recall,
        "precision": precision,
        "operational_meaning": {
            "false_negative": "Don't Go nao antecipado = parada nao planejada",
            "false_positive": "alerta sem Don't Go = inspecao desnecessaria",
        },
        "false_negative_distribution": fn_dist,
        "recall_by_month": monthly,
    }

    plot_confusion_matrix(cm, threshold, figure_path)

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Analise de erros salva em %s", json_path)

    return result


if __name__ == "__main__":
    out = run()
    cm = out["confusion_matrix"]
    logger.info(
        "TP=%d FP=%d FN=%d TN=%d | recall=%.3f precision=%.3f",
        cm["tp"], cm["fp"], cm["fn"], cm["tn"], out["recall"], out["precision"],
    )
