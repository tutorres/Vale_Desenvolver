import logging
import random

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost  # noqa: F401 - seed is passed per-model, not globally
    except ImportError:
        pass


def log(name: str) -> logging.Logger:
    """Return a named logger with a standard console handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
