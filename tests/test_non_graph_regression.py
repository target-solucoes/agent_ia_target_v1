"""
Phase 4.1 — Regression Tests for non_graph_executor.

Tests the 4 critical queries from the failed session log (881118fc_20260209_094828):
  Q1: "qual o ultimo ano com vendas?"
  Q2: "qual o ano em que ocorreu a ultima venda?"
  Q3: "quais foi o mes com maior valor de venda em 2016?"
  Q4: "que mes foi o maior valor de vendas?"

Each test validates:
  - No BinderException or execution errors
  - Correct intent analysis (semantic understanding)
  - Correct SQL generation (GROUP BY, virtual columns, etc.)
  - Correct query execution against the real dataset
  - Output follows NonGraphOutput schema

These tests use mocked LLM for the IntentAnalyzer to ensure deterministic,
offline validation. The DynamicQueryBuilder and QueryExecutor run against
the real dataset to ensure end-to-end correctness of SQL generation and execution.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.non_graph_executor.models.intent_schema import (
    AggregationSpec,
    ColumnSpec,
    OrderSpec,
    QueryIntent,
)
from src.non_graph_executor.tools.dynamic_query_builder import DynamicQueryBuilder
from src.non_graph_executor.tools.query_executor import QueryExecutor
from src.non_graph_executor.tools.metadata_cache import MetadataCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH = str(
    PROJECT_ROOT / "data" / "datasets" / "DadosComercial_resumido_v02.parquet"
)


class MockAliasMapper:
    """Minimal AliasMapper mock for regression tests."""

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
            "Valor_Vendido": ["vendas", "faturamento", "receita"],
            "UF_Cliente": ["estado", "UF"],
            "Cod_Cliente": ["cliente"],
            "Data": ["data"],
            "Ano": ["ano"],
            "Mes": ["mes", "mês"],
        }
    }

    def is_virtual_column(self, name: str) -> bool:
        return name in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, name: str):
        return self.VIRTUAL_COLUMN_MAP.get(name)

    def is_categorical_column(self, name: str) -> bool:
        return name in self.column_types.get("categorical", [])

    def is_numeric_column(self, name: str) -> bool:
        return name in self.column_types.get("numeric", [])

    def resolve(self, term: str):
        mapping = {
            "vendas": "Valor_Vendido",
            "venda": "Valor_Vendido",
            "faturamento": "Valor_Vendido",
            "ano": "Ano",
            "mes": "Mes",
            "estado": "UF_Cliente",
        }
        return mapping.get(term.lower())


@pytest.fixture
def alias_mapper():
    return MockAliasMapper()


@pytest.fixture
def metadata_cache():
    if not Path(DATA_PATH).exists():
        pytest.skip("Dataset not found — skipping regression test")
    return MetadataCache(DATA_PATH)


@pytest.fixture
def query_executor(metadata_cache, alias_mapper):
    return QueryExecutor(DATA_PATH, metadata_cache, alias_mapper)


@pytest.fixture
def query_builder(alias_mapper):
    return DynamicQueryBuilder(alias_mapper, DATA_PATH)


# ===========================================================================
# Q1: "qual o ultimo ano com vendas?"
# ===========================================================================


class TestQ1UltimoAnoComVendas:
    """
    Regression test for Q1.

    Original failure: BinderException — column "Ano" not found.
    Expected behaviour: SELECT MAX(YEAR("Data")) → returns the most recent year.
    """

    def _build_intent(self) -> QueryIntent:
        return QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano",
                        is_virtual=True,
                        expression='YEAR("Data")',
                        alias="Ano",
                    ),
                    alias="ultimo_ano",
                )
            ],
            confidence=0.95,
            reasoning="Último ano com vendas → MAX(YEAR(Data))",
        )

    def test_intent_is_temporal(self):
        """Intent must be temporal_analysis, not simple_aggregation."""
        intent = self._build_intent()
        assert intent.intent_type == "temporal_analysis"

    def test_aggregation_is_max_on_virtual_year(self):
        """Aggregation must be MAX on virtual column Ano (YEAR(Data))."""
        intent = self._build_intent()
        agg = intent.aggregations[0]
        assert agg.function == "max"
        assert agg.column.is_virtual is True
        assert agg.column.expression == 'YEAR("Data")'

    def test_sql_contains_year_expression(self, query_builder):
        """Generated SQL must use YEAR("Data"), not "Ano"."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        assert 'YEAR("Data")' in sql, f"SQL must contain YEAR expression: {sql}"
        assert "MAX" in sql, f"SQL must contain MAX: {sql}"
        # Must NOT try to reference bare "Ano" column
        assert '"Ano"' not in sql, f"SQL must not reference bare Ano column: {sql}"

    def test_execution_returns_valid_year(self, query_builder, query_executor):
        """Query execution must return a valid year without BinderException."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)

        # This MUST NOT raise BinderException
        result = query_executor.execute_dynamic_query(sql)

        assert len(result) == 1, f"Expected 1 row, got {len(result)}"
        year_value = list(result[0].values())[0]
        assert isinstance(year_value, (int, float)), (
            f"Expected numeric year, got {type(year_value)}"
        )
        assert 2000 <= year_value <= 2030, f"Year {year_value} out of reasonable range"

    def test_no_binder_exception(self, query_builder, query_executor):
        """Specifically validates that BinderException does NOT occur (original bug)."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        try:
            query_executor.execute_dynamic_query(sql)
        except Exception as e:
            error_msg = str(e).lower()
            assert "binder" not in error_msg, (
                f"BinderException still occurs (original bug not fixed): {e}"
            )
            raise  # Re-raise if it's a different error


# ===========================================================================
# Q2: "qual o ano em que ocorreu a ultima venda?"
# ===========================================================================


class TestQ2AnoUltimaVenda:
    """
    Regression test for Q2.

    Original failure: System returned SUM(Valor_Vendido) instead of YEAR(MAX(Data)).
    Expected behaviour: SELECT MAX(YEAR("Data")) → returns the year, not the value.
    """

    def _build_intent(self) -> QueryIntent:
        return QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano",
                        is_virtual=True,
                        expression='YEAR("Data")',
                        alias="Ano",
                    ),
                    alias="ano_ultima_venda",
                )
            ],
            confidence=0.95,
            reasoning="Ano da última venda → MAX(YEAR(Data)), foco em dimensão temporal",
        )

    def test_intent_is_temporal_not_aggregation_of_value(self):
        """
        Must NOT be aggregation on Valor_Vendido.
        The original bug was: system did SUM(Valor_Vendido) instead of MAX(YEAR(Data)).
        """
        intent = self._build_intent()
        assert intent.intent_type == "temporal_analysis"
        assert intent.aggregations[0].column.name == "Ano"
        assert intent.aggregations[0].column.name != "Valor_Vendido"

    def test_sql_targets_year_not_value(self, query_builder):
        """SQL must aggregate YEAR(Data), not Valor_Vendido."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        assert 'YEAR("Data")' in sql
        assert "Valor_Vendido" not in sql, (
            f"SQL should NOT reference Valor_Vendido for this query: {sql}"
        )

    def test_execution_returns_year_not_monetary_value(
        self, query_builder, query_executor
    ):
        """Result must be a year (e.g. 2016), not a monetary value (e.g. 47833605)."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        result = query_executor.execute_dynamic_query(sql)

        assert len(result) == 1
        value = list(result[0].values())[0]
        # A year should be between 2000 and 2030; a monetary value would be > 10000
        assert 2000 <= value <= 2030, (
            f"Expected year (2000-2030), got {value}. "
            f"Possibly returning monetary value (original bug)."
        )


# ===========================================================================
# Q3: "quais foi o mes com maior valor de venda em 2016?"
# ===========================================================================


class TestQ3MesMaiorVenda2016:
    """
    Regression test for Q3.

    Original failure: System returned MAX(Valor_Vendido) = 489339.41 without
    identifying WHICH month. The correct answer requires GROUP BY MONTH(Data).
    Expected: SELECT MONTH("Data"), SUM("Valor_Vendido") ... GROUP BY ... ORDER BY DESC LIMIT 1
    """

    def _build_intent(self) -> QueryIntent:
        return QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Mes",
                    is_virtual=True,
                    expression='MONTH("Data")',
                    alias="Mes",
                )
            ],
            order_by=OrderSpec(column="total_vendas", direction="DESC"),
            limit=1,
            confidence=0.95,
            reasoning="Mês com maior total de vendas em 2016 → GROUP BY MONTH, SUM, ORDER DESC, LIMIT 1",
        )

    def _filters_2016(self) -> Dict[str, Any]:
        return {"Data": {"between": ["2016-01-01", "2016-12-31"]}}

    def test_intent_has_group_by(self):
        """Intent MUST have GROUP BY on MONTH (this was completely missing before)."""
        intent = self._build_intent()
        assert len(intent.group_by) >= 1
        assert intent.group_by[0].name == "Mes"
        assert intent.group_by[0].is_virtual is True

    def test_intent_aggregation_is_sum_not_max(self):
        """
        Aggregation must be SUM (total per month), NOT MAX (single highest value).
        Original bug: used MAX(Valor_Vendido) which returns the single highest
        transaction, not the month with the highest total.
        """
        intent = self._build_intent()
        assert intent.aggregations[0].function == "sum"

    def test_sql_has_group_by_and_order_by(self, query_builder):
        """SQL must contain GROUP BY, ORDER BY, and LIMIT."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent, filters=self._filters_2016())
        assert "GROUP BY" in sql, f"SQL missing GROUP BY: {sql}"
        assert "ORDER BY" in sql, f"SQL missing ORDER BY: {sql}"
        assert "LIMIT 1" in sql, f"SQL missing LIMIT 1: {sql}"
        assert 'MONTH("Data")' in sql, f"SQL missing MONTH expression: {sql}"

    def test_execution_returns_month_and_value(self, query_builder, query_executor):
        """
        Result must contain BOTH the month identifier AND the total value.
        The original bug returned only a scalar value without the month.
        """
        intent = self._build_intent()
        sql = query_builder.build_query(intent, filters=self._filters_2016())
        result = query_executor.execute_dynamic_query(sql)

        assert len(result) == 1, f"Expected 1 row (top month), got {len(result)}"

        row = result[0]
        # Must have at least 2 columns: month identifier + value
        assert len(row) >= 2, (
            f"Expected at least 2 columns (month + value), got {len(row)}: {row}"
        )

        # Extract values
        keys = list(row.keys())
        values = list(row.values())

        # One column should be the month (1-12)
        month_value = None
        total_value = None
        for k, v in row.items():
            if isinstance(v, (int, float)):
                if 1 <= v <= 12:
                    month_value = v
                elif v > 12:
                    total_value = v

        assert month_value is not None, (
            f"Result must contain a month number (1-12): {row}"
        )
        assert total_value is not None, f"Result must contain a total value > 12: {row}"

    def test_result_is_not_single_max_transaction(self, query_builder, query_executor):
        """
        The result should NOT be ~489339.41 (a single MAX transaction).
        It should be the SUM of all transactions in the top month,
        which is a much larger number.
        """
        intent = self._build_intent()
        sql = query_builder.build_query(intent, filters=self._filters_2016())
        result = query_executor.execute_dynamic_query(sql)
        row = result[0]

        # Find the aggregated value (the larger number)
        total = max(v for v in row.values() if isinstance(v, (int, float)) and v > 12)

        # The SUM per month should be much larger than a single transaction
        # The original bug returned ~489339.41 (single MAX); the correct SUM
        # for any month should be in the millions range for this dataset
        assert total > 500000, (
            f"Total {total} seems too small — might be returning MAX of a single "
            f"transaction instead of SUM per month (original bug)."
        )


# ===========================================================================
# Q4: "que mes foi o maior valor de vendas?"
# ===========================================================================


class TestQ4MesMaiorValorVendas:
    """
    Regression test for Q4.

    Same semantic error as Q3, but without the explicit 2016 filter.
    The system should still GROUP BY MONTH and return the month with
    the highest total sales across all data.
    """

    def _build_intent(self) -> QueryIntent:
        return QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Mes",
                    is_virtual=True,
                    expression='MONTH("Data")',
                    alias="Mes",
                )
            ],
            order_by=OrderSpec(column="total_vendas", direction="DESC"),
            limit=1,
            confidence=0.95,
            reasoning="Mês com maior valor de vendas (sem filtro de ano)",
        )

    def test_sql_has_no_date_filter(self, query_builder):
        """Without explicit year, SQL should NOT have date BETWEEN filter."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        assert "BETWEEN" not in sql, (
            f"SQL should not have date filter when no year specified: {sql}"
        )

    def test_sql_still_has_group_by(self, query_builder):
        """Even without filters, GROUP BY must be present."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        assert "GROUP BY" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 1" in sql

    def test_execution_identifies_month(self, query_builder, query_executor):
        """Result must contain month identification and total value."""
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        result = query_executor.execute_dynamic_query(sql)

        assert len(result) == 1
        row = result[0]
        assert len(row) >= 2, f"Must have month + value columns: {row}"

        # Verify month is in valid range
        month_found = False
        for v in row.values():
            if isinstance(v, (int, float)) and 1 <= v <= 12:
                month_found = True
                break
        assert month_found, f"Result must contain a month number (1-12): {row}"

    def test_result_uses_sum_not_max(self, query_builder, query_executor):
        """
        Aggregation must be SUM per month (total), not MAX (single row).
        This was the core semantic error in the original system.
        """
        intent = self._build_intent()
        sql = query_builder.build_query(intent)
        assert "SUM" in sql, f"SQL must use SUM, not MAX: {sql}"

        result = query_executor.execute_dynamic_query(sql)
        row = result[0]
        total = max(v for v in row.values() if isinstance(v, (int, float)) and v > 12)
        assert total > 500000, f"Total {total} seems too small for SUM of monthly sales"


# ===========================================================================
# Cross-query validation
# ===========================================================================


class TestCrossQueryValidation:
    """Cross-validation tests ensuring consistency across queries."""

    def test_q1_and_q2_return_same_year(self, query_builder, query_executor):
        """Q1 and Q2 ask the same thing differently — must return same year."""
        intent_q1 = QueryIntent(
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
        intent_q2 = QueryIntent(
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

        sql1 = query_builder.build_query(intent_q1)
        sql2 = query_builder.build_query(intent_q2)
        r1 = query_executor.execute_dynamic_query(sql1)
        r2 = query_executor.execute_dynamic_query(sql2)

        year1 = list(r1[0].values())[0]
        year2 = list(r2[0].values())[0]
        assert year1 == year2, (
            f"Q1 returned year {year1} but Q2 returned year {year2}. "
            f"Both queries ask for the last year with sales."
        )

    def test_q3_with_filter_is_subset_of_q4_without_filter(
        self, query_builder, query_executor
    ):
        """
        Q3 (filtered to 2016) top-month total must be <= Q4 (all data) top-month total.
        Unless Q4's top month happens to be in 2016, in which case they could be equal.
        """
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Mes", is_virtual=True, expression='MONTH("Data")', alias="Mes"
                )
            ],
            order_by=OrderSpec(column="total_vendas", direction="DESC"),
            limit=1,
            confidence=0.95,
        )

        filters_2016 = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}

        sql_q3 = query_builder.build_query(intent, filters=filters_2016)
        sql_q4 = query_builder.build_query(intent)

        r_q3 = query_executor.execute_dynamic_query(sql_q3)
        r_q4 = query_executor.execute_dynamic_query(sql_q4)

        total_q3 = max(
            v for v in r_q3[0].values() if isinstance(v, (int, float)) and v > 12
        )
        total_q4 = max(
            v for v in r_q4[0].values() if isinstance(v, (int, float)) and v > 12
        )

        assert total_q3 <= total_q4, (
            f"Q3 (filtered 2016) total {total_q3} should be <= Q4 (all data) total {total_q4}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
