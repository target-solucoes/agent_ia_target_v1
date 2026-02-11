"""
Phase 4.4 — End-to-End integration tests for non_graph_executor.

Tests the full dynamic execution flow:
  IntentAnalyzer (mocked LLM) → DynamicQueryBuilder → QueryExecutor (real DuckDB)

Each test simulates the real agent pipeline:
  1. IntentAnalyzer.analyze() returns a QueryIntent (via mocked LLM)
  2. DynamicQueryBuilder.build_query() generates SQL
  3. QueryExecutor.execute_dynamic_query() runs against real dataset
  4. OutputFormatter validates the output schema (NonGraphOutput)

All tests run OFFLINE (no API keys) thanks to mocked LLM responses.
Tests validate real query execution against the parquet dataset.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.non_graph_executor.models.intent_schema import (
    AggregationSpec,
    ColumnSpec,
    OrderSpec,
    QueryIntent,
)
from src.non_graph_executor.models.schemas import NonGraphOutput
from src.non_graph_executor.tools.dynamic_query_builder import DynamicQueryBuilder
from src.non_graph_executor.tools.intent_analyzer import IntentAnalyzer
from src.non_graph_executor.tools.metadata_cache import MetadataCache
from src.non_graph_executor.tools.query_executor import QueryExecutor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_SOURCE = "data/datasets/DadosComercial_resumido_v02.parquet"


# ---------------------------------------------------------------------------
# Mock AliasMapper
# ---------------------------------------------------------------------------


class MockAliasMapper:
    """Full-featured mock of AliasMapper for E2E tests."""

    VIRTUAL_COLUMN_MAP = {
        "Ano": 'YEAR("Data")',
        "Mes": 'MONTH("Data")',
        "Nome_Mes": 'MONTHNAME("Data")',
    }

    column_types = {
        "numeric": ["Valor_Vendido", "Peso_Vendido", "Qtd_Vendida"],
        "categorical": [
            "UF_Cliente",
            "Empresa",
            "Cod_Familia_Produto",
            "Des_Linha_Produto",
            "Municipio_Cliente",
        ],
        "temporal": ["Data"],
    }

    aliases = {
        "columns": {
            "Valor_Vendido": ["vendas", "faturamento", "receita", "valor vendido"],
            "UF_Cliente": ["estado", "UF", "estados"],
            "Cod_Cliente": ["cliente", "clientes"],
            "Empresa": ["empresa", "empresas"],
            "Data": ["data", "período"],
        }
    }

    def is_virtual_column(self, col_name: str) -> bool:
        return col_name in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, col_name: str):
        return self.VIRTUAL_COLUMN_MAP.get(col_name)

    def is_categorical_column(self, name):
        return name in self.column_types.get("categorical", [])

    def resolve(self, term: str):
        mapping = {
            "vendas": "Valor_Vendido",
            "faturamento": "Valor_Vendido",
            "ano": "Ano",
            "mes": "Mes",
            "estado": "UF_Cliente",
            "cliente": "Cod_Cliente",
            "empresa": "Empresa",
        }
        return mapping.get(term.lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(response_dict: Dict[str, Any]) -> MagicMock:
    """Create a mocked LLM that returns the given JSON dict."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_dict)
    mock_response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alias_mapper():
    return MockAliasMapper()


@pytest.fixture(scope="module")
def metadata_cache():
    return MetadataCache(DATA_SOURCE)


@pytest.fixture(scope="module")
def query_executor(metadata_cache, alias_mapper):
    return QueryExecutor(DATA_SOURCE, metadata_cache, alias_mapper)


@pytest.fixture(scope="module")
def query_builder(alias_mapper):
    return DynamicQueryBuilder(alias_mapper, DATA_SOURCE)


# ===========================================================================
# E2E: Full Flow Tests (IntentAnalyzer → DynamicQueryBuilder → QueryExecutor)
# ===========================================================================


class TestE2ETemporalAnalysis:
    """E2E tests for temporal analysis queries."""

    def test_q1_ultimo_ano_full_flow(self, alias_mapper, query_builder, query_executor):
        """E2E: 'qual o ultimo ano com vendas?' → IntentAnalyzer → SQL → DuckDB result."""
        llm_response = {
            "intent_type": "temporal_analysis",
            "aggregations": [
                {
                    "function": "max",
                    "column": {
                        "name": "Ano",
                        "is_virtual": True,
                        "expression": 'YEAR("Data")',
                    },
                    "alias": "ultimo_ano",
                }
            ],
            "confidence": 0.95,
            "reasoning": "Último ano → MAX(YEAR(Data))",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        # Step 1: Analyze intent
        intent = analyzer.analyze("qual o ultimo ano com vendas?")
        assert intent.intent_type == "temporal_analysis"

        # Step 2: Build SQL
        sql = query_builder.build_query(intent)
        assert "MAX" in sql
        assert 'YEAR("Data")' in sql

        # Step 3: Execute
        results = query_executor.execute_dynamic_query(sql)
        assert len(results) == 1
        ultimo_ano = list(results[0].values())[0]
        assert isinstance(ultimo_ano, (int, float))
        assert ultimo_ano >= 2000  # Sanity check

    def test_q2_ano_ultima_venda_full_flow(
        self, alias_mapper, query_builder, query_executor
    ):
        """E2E: 'qual o ano em que ocorreu a última venda?'"""
        llm_response = {
            "intent_type": "temporal_analysis",
            "aggregations": [
                {
                    "function": "max",
                    "column": {
                        "name": "Ano",
                        "is_virtual": True,
                        "expression": 'YEAR("Data")',
                    },
                    "alias": "ano_ultima_venda",
                }
            ],
            "confidence": 0.95,
            "reasoning": "MAX(YEAR(Data))",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("qual o ano em que ocorreu a última venda?")
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 1
        ano = list(results[0].values())[0]
        assert isinstance(ano, (int, float))


class TestE2EGroupedAggregation:
    """E2E tests for grouped aggregation queries."""

    def test_q3_mes_maior_venda_2016_full_flow(
        self, alias_mapper, query_builder, query_executor
    ):
        """E2E: 'qual foi o mes com maior valor de venda em 2016?'"""
        llm_response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "group_by": [
                {
                    "name": "Mes",
                    "is_virtual": True,
                    "expression": 'MONTH("Data")',
                    "alias": "Mes",
                }
            ],
            "order_by": {"column": "total_vendas", "direction": "DESC"},
            "limit": 1,
            "confidence": 0.95,
            "reasoning": "Mês com maior total de vendas em 2016",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("qual foi o mes com maior valor de venda em 2016?")
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = query_builder.build_query(intent, filters=filters)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 1
        mes = results[0].get("Mes")
        total_vendas = results[0].get("total_vendas")
        assert mes is not None
        assert isinstance(mes, (int, float))
        assert 1 <= mes <= 12
        assert total_vendas is not None
        assert total_vendas > 0

    def test_q4_mes_maior_valor_vendas_full_flow(
        self, alias_mapper, query_builder, query_executor
    ):
        """E2E: 'que mes foi o maior valor de vendas?' (no year filter)"""
        llm_response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "group_by": [
                {
                    "name": "Mes",
                    "is_virtual": True,
                    "expression": 'MONTH("Data")',
                    "alias": "Mes",
                }
            ],
            "order_by": {"column": "total_vendas", "direction": "DESC"},
            "limit": 1,
            "confidence": 0.95,
            "reasoning": "Mês com maior total de vendas global",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("que mes foi o maior valor de vendas?")
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 1
        mes = results[0].get("Mes")
        assert 1 <= mes <= 12

    def test_vendas_por_estado_full_flow(
        self, alias_mapper, query_builder, query_executor
    ):
        """E2E: 'vendas por estado' — grouped aggregation without ranking."""
        llm_response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "group_by": [{"name": "UF_Cliente", "alias": "Estado"}],
            "confidence": 0.92,
            "reasoning": "Vendas agrupadas por estado",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("vendas por estado")
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) > 1  # Multiple states
        for row in results:
            assert "Estado" in row
            assert "total_vendas" in row
            assert row["total_vendas"] > 0

    def test_vendas_por_mes_em_2016(self, alias_mapper, query_builder, query_executor):
        """E2E: 'vendas por mês em 2016' — all months without ranking."""
        llm_response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "group_by": [
                {
                    "name": "Mes",
                    "is_virtual": True,
                    "expression": 'MONTH("Data")',
                    "alias": "Mes",
                }
            ],
            "confidence": 0.90,
            "reasoning": "Vendas agrupadas por mês em 2016",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("vendas por mês em 2016")
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = query_builder.build_query(intent, filters=filters)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) >= 1
        months = [row["Mes"] for row in results]
        assert all(1 <= m <= 12 for m in months)
        assert len(months) == len(set(months))  # No duplicates


class TestE2ERanking:
    """E2E tests for ranking queries."""

    def test_top_5_estados_por_faturamento(
        self, alias_mapper, query_builder, query_executor
    ):
        """E2E: 'top 5 estados por faturamento'"""
        llm_response = {
            "intent_type": "ranking",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_faturamento",
                }
            ],
            "group_by": [{"name": "UF_Cliente", "alias": "Estado"}],
            "order_by": {"column": "total_faturamento", "direction": "DESC"},
            "limit": 5,
            "confidence": 0.95,
            "reasoning": "Top 5 estados por faturamento",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("top 5 estados por faturamento")
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 5
        # Check descending order
        values = [row["total_faturamento"] for row in results]
        assert values == sorted(values, reverse=True)


class TestE2ESimpleAggregation:
    """E2E tests for simple aggregation queries."""

    def test_total_de_vendas(self, alias_mapper, query_builder, query_executor):
        """E2E: 'qual o total de vendas?'"""
        llm_response = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "confidence": 0.95,
            "reasoning": "SUM(Valor_Vendido)",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("qual o total de vendas?")
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 1
        total = results[0]["total_vendas"]
        assert isinstance(total, (int, float))
        assert total > 0

    def test_count_distinct_clientes(self, alias_mapper, query_builder, query_executor):
        """E2E: 'quantos clientes temos?'"""
        llm_response = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "count",
                    "column": {"name": "Cod_Cliente"},
                    "distinct": True,
                    "alias": "total_clientes",
                }
            ],
            "confidence": 0.95,
            "reasoning": "COUNT(DISTINCT Cod_Cliente)",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("quantos clientes temos?")
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 1
        total = results[0]["total_clientes"]
        assert isinstance(total, (int, float))
        assert total > 0


class TestE2EWithFilters:
    """E2E tests with session filters applied."""

    def test_total_vendas_ano_2016(self, alias_mapper, query_builder, query_executor):
        """E2E: total de vendas com filtro de ano 2016."""
        llm_response = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "confidence": 0.95,
            "reasoning": "SUM filtrado por 2016",
        }
        mock_llm = _make_mock_llm(llm_response)
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=alias_mapper)

        intent = analyzer.analyze("total de vendas")
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = query_builder.build_query(intent, filters=filters)
        results = query_executor.execute_dynamic_query(sql)

        assert len(results) == 1
        total = results[0]["total_vendas"]
        assert total > 0

        # Compare with unfiltered total — filtered should be smaller
        sql_all = query_builder.build_query(intent)
        results_all = query_executor.execute_dynamic_query(sql_all)
        total_all = results_all[0]["total_vendas"]
        assert total < total_all  # Filtered must be less than total


# ===========================================================================
# E2E: Output Schema Compliance
# ===========================================================================


class TestOutputSchemaCompliance:
    """Tests that results can be packaged into NonGraphOutput."""

    def test_aggregation_result_fits_non_graph_output(
        self, alias_mapper, query_builder, query_executor
    ):
        """Results from dynamic execution can be wrapped in NonGraphOutput."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            confidence=0.95,
        )
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        # Package into NonGraphOutput
        output = NonGraphOutput(
            status="success",
            query_type="aggregation",
            data=results,
            summary="O total de vendas é...",
            metadata={
                "row_count": len(results),
                "engine": "DuckDB",
                "execution_path": "dynamic",
                "intent_type": "simple_aggregation",
                "sql_query": sql,
            },
            performance_metrics={"total_time": 0.1},
        )
        serialized = output.model_dump()

        assert serialized["status"] == "success"
        assert serialized["query_type"] == "aggregation"
        assert serialized["data"] is not None
        assert len(serialized["data"]) == 1
        assert serialized["metadata"]["execution_path"] == "dynamic"

    def test_grouped_result_fits_non_graph_output(
        self, alias_mapper, query_builder, query_executor
    ):
        """Multi-row grouped results can be wrapped in NonGraphOutput."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
            confidence=0.92,
        )
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)

        output = NonGraphOutput(
            status="success",
            query_type="aggregation",
            data=results,
            metadata={
                "row_count": len(results),
                "engine": "DuckDB",
                "execution_path": "dynamic",
                "intent_type": "grouped_aggregation",
                "group_by": ["UF_Cliente"],
            },
            performance_metrics={"total_time": 0.2},
        )
        serialized = output.model_dump()

        assert serialized["status"] == "success"
        assert len(serialized["data"]) > 1
        assert serialized["metadata"]["group_by"] == ["UF_Cliente"]


# ===========================================================================
# E2E: Cross-Query Consistency
# ===========================================================================


class TestE2ECrossQueryConsistency:
    """Tests that related queries produce consistent results."""

    def test_q1_q2_same_answer(self, alias_mapper, query_builder, query_executor):
        """Q1 and Q2 should return the same year."""
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano", is_virtual=True, expression='YEAR("Data")'
                    ),
                    alias="ultimo_ano",
                )
            ],
            confidence=0.95,
        )
        sql = query_builder.build_query(intent)
        results = query_executor.execute_dynamic_query(sql)
        q1_year = results[0]["ultimo_ano"]

        intent2 = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano", is_virtual=True, expression='YEAR("Data")'
                    ),
                    alias="ano_ultima_venda",
                )
            ],
            confidence=0.95,
        )
        sql2 = query_builder.build_query(intent2)
        results2 = query_executor.execute_dynamic_query(sql2)
        q2_year = results2[0]["ano_ultima_venda"]

        assert q1_year == q2_year

    def test_sum_by_estado_equals_total(
        self, alias_mapper, query_builder, query_executor
    ):
        """Sum of per-state totals should equal the overall total."""
        # Overall total
        intent_total = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
        )
        sql_total = query_builder.build_query(intent_total)
        results_total = query_executor.execute_dynamic_query(sql_total)
        overall_total = results_total[0]["total"]

        # Per-state totals
        intent_grouped = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
        )
        sql_grouped = query_builder.build_query(intent_grouped)
        results_grouped = query_executor.execute_dynamic_query(sql_grouped)
        sum_of_groups = sum(row["total"] for row in results_grouped)

        assert abs(overall_total - sum_of_groups) < 0.01


# ===========================================================================
# E2E: Error Cases
# ===========================================================================


class TestE2EErrorCases:
    """Tests for error handling in the E2E flow."""

    def test_invalid_sql_raises_exception(self, query_executor):
        """Executing invalid SQL should raise an exception."""
        with pytest.raises(Exception):
            query_executor.execute_dynamic_query("SELECT FROM INVALID_TABLE")

    def test_nonexistent_column_in_query_raises(self, query_builder, query_executor):
        """Referencing a non-existent physical column should fail at DuckDB level."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Coluna_Que_Nao_Existe"),
                    alias="result",
                )
            ],
        )
        sql = query_builder.build_query(intent)
        with pytest.raises(Exception):
            query_executor.execute_dynamic_query(sql)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
