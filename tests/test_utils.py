"""
TDD — test_utils.py
Testes para src/utils.py
"""

import logging
import numpy as np
import pytest

from src.utils import log, set_seeds


class TestSetSeeds:

    def test_numpy_reproducible(self):
        """set_seeds deve tornar numpy reproduzível."""
        set_seeds(42)
        a = np.random.randint(0, 1000, size=10)
        set_seeds(42)
        b = np.random.randint(0, 1000, size=10)
        assert list(a) == list(b)

    def test_seed_diferente_gera_resultado_diferente(self):
        """Seeds diferentes devem gerar resultados diferentes."""
        set_seeds(42)
        a = np.random.randint(0, 1000, size=10)
        set_seeds(99)
        b = np.random.randint(0, 1000, size=10)
        assert list(a) != list(b)


class TestLog:

    def test_retorna_logger(self):
        """log() deve retornar um Logger."""
        logger = log("test_module")
        assert isinstance(logger, logging.Logger)

    def test_nome_correto(self):
        """Logger deve ter o nome passado."""
        logger = log("meu_modulo")
        assert logger.name == "meu_modulo"

    def test_mesmo_nome_retorna_mesmo_logger(self):
        """Chamar log() duas vezes com mesmo nome retorna o mesmo objeto."""
        a = log("modulo_x")
        b = log("modulo_x")
        assert a is b
