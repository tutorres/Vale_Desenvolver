"""
TDD: test_loader.py
Testes para src/loader.py
"""

import pandas as pd
import pytest

from src.loader import load_telemetry


# Columns required in the raw parquet files
_FILE_COLUMNS = {
    "TAG", "Alarme", "Id_Criticidade", "Criticidade",
    "Valor", "Classe", "Is_Dont_Go", "Tipo", "Data_Evento",
}

# Expected output columns after loading (Data_Evento renamed to timestamp)
EXPECTED_COLUMNS = (_FILE_COLUMNS - {"Data_Evento"}) | {"timestamp"}


class TestLoadTelemetry:

    def test_retorna_dataframe(self, tmp_path):
        """load_telemetry deve retornar um DataFrame."""
        for month in ["jan", "feb", "mar", "abr", "may", "jun"]:
            df = pd.DataFrame({col: [] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        result = load_telemetry(str(tmp_path))

        assert isinstance(result, pd.DataFrame)

    def test_colunas_esperadas_presentes(self, tmp_path):
        """DataFrame deve conter todas as colunas do schema, com timestamp renomeado."""
        for month in ["jan", "feb", "mar", "abr", "may", "jun"]:
            df = pd.DataFrame({col: [] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        result = load_telemetry(str(tmp_path))
        assert EXPECTED_COLUMNS.issubset(set(result.columns))
        assert "timestamp" in result.columns
        assert "Data_Evento" not in result.columns

    def test_coluna_source_month_adicionada(self, tmp_path):
        """DataFrame deve ter coluna _source_month para rastreabilidade."""
        for month in ["jan", "feb", "mar", "abr", "may", "jun"]:
            df = pd.DataFrame({col: ["x"] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        result = load_telemetry(str(tmp_path))
        assert "_source_month" in result.columns

    def test_todos_os_meses_carregados(self, tmp_path):
        """Todos os 6 meses devem estar representados."""
        for month in ["jan", "feb", "mar", "abr", "may", "jun"]:
            df = pd.DataFrame({col: ["x"] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        result = load_telemetry(str(tmp_path))
        assert set(result["_source_month"].unique()) == {"jan", "feb", "mar", "abr", "may", "jun"}

    def test_levanta_erro_se_arquivo_faltando(self, tmp_path):
        """Deve levantar FileNotFoundError se algum mês estiver faltando."""
        for month in ["jan", "feb", "mar", "abr", "may"]:
            df = pd.DataFrame({col: [] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        with pytest.raises(FileNotFoundError):
            load_telemetry(str(tmp_path))

    def test_colunas_string_convertidas_para_category(self, tmp_path):
        """Colunas de string de baixa cardinalidade devem virar dtype category,
        para controlar o uso de memória nos 37M registros reais."""
        for month in ["jan", "feb", "mar", "abr", "may", "jun"]:
            df = pd.DataFrame({col: ["x"] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        result = load_telemetry(str(tmp_path))

        for col in ["TAG", "Alarme", "Criticidade", "Classe", "Tipo"]:
            assert result[col].dtype.name == "category", f"{col} deveria ser category"

    def test_categorias_diferentes_entre_meses_mantem_category(self, tmp_path):
        """Quando os meses têm valores diferentes (ex: TAGs distintos por mês,
        como no dataset real), o dtype category não deve reverter para object
        após o concat, senão a memória explode nos 37M registros reais."""
        for i, month in enumerate(["jan", "feb", "mar", "abr", "may", "jun"]):
            df = pd.DataFrame({col: [f"{col}_{month}_{i}"] for col in _FILE_COLUMNS})
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        result = load_telemetry(str(tmp_path))

        for col in ["TAG", "Alarme", "Criticidade", "Classe", "Tipo"]:
            assert result[col].dtype.name == "category", f"{col} deveria permanecer category"
        assert result["TAG"].nunique() == 6

    def test_levanta_erro_se_coluna_faltando(self, tmp_path):
        """Deve levantar SchemaError se coluna obrigatória estiver ausente."""
        from src.loader import SchemaError

        for month in ["jan", "feb", "mar", "abr", "may", "jun"]:
            df = pd.DataFrame({"TAG": [], "Alarme": []})  # schema incompleto
            df.to_parquet(tmp_path / f"telemetry_{month}.parquet")

        with pytest.raises(SchemaError):
            load_telemetry(str(tmp_path))
