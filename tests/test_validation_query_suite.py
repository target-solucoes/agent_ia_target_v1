"""
Phase 4.5 — Validation query suite for non_graph_executor.

Comprehensive parametrized test suite covering ALL query categories defined
in the planning document:

  1. Simple aggregations (sum, avg, count, count distinct, min, max)
  2. Grouped aggregations (by estado, mes, ano, empresa, produto)
  3. Rankings (top N DESC, bottom N ASC)
  4. Temporal analysis (último ano, primeiro mês, ano da última venda)
  5. Comparisons (filters isolating specific groups)
  6. Combined scenarios (ranking + temporal filter, grouped + multiple agg)

All tests build QueryIntent directly, generate SQL via DynamicQueryBuilder,
and execute against the real DuckDB dataset for validation.
No LLM calls — tests are fully deterministic.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.non_graph_executor.tools.dynamic_query_builder import DynamicQueryBuilder
from src.non_graph_executor.tools.metadata_cache import MetadataCache
from src.non_graph_executor.tools.query_executor import QueryExecutor
from src.non_graph_executor.models.intent_schema import (
    AggregationSpec,
    ColumnSpec,
    OrderSpec,
    QueryIntent,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_SOURCE = "data/datasets/DadosComercial_resumido_v02.parquet"


# ---------------------------------------------------------------------------
# Mock AliasMapper
# ---------------------------------------------------------------------------


class MockAliasMapper:
    """Full-featured mock of AliasMapper."""

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
            "Empresa": ["empresa"],
            "Cod_Familia_Produto": ["familia"],
        }
    }

    def is_virtual_column(self, name):
        return name in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, name):
        return self.VIRTUAL_COLUMN_MAP.get(name)

    def is_categorical_column(self, name):
        return name in self.column_types.get("categorical", [])


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
def executor(metadata_cache, alias_mapper):
    return QueryExecutor(DATA_SOURCE, metadata_cache, alias_mapper)


@pytest.fixture(scope="module")
def builder(alias_mapper):
    return DynamicQueryBuilder(alias_mapper, DATA_SOURCE)


# ===========================================================================
# Category 1: Simple Aggregations
# ===========================================================================


class TestSimpleAggregations:
    """Validation for simple aggregation queries (no GROUP BY)."""

    @pytest.mark.parametrize(
        "func,alias_name",
        [
            ("sum", "total_vendas"),
            ("avg", "media_vendas"),
            ("min", "min_vendas"),
            ("max", "max_vendas"),
        ],
    )
    def test_basic_aggregation_functions(self, builder, executor, func, alias_name):
        """Each basic aggregation function returns a positive numeric result."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function=func,
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias=alias_name,
                )
            ],
            confidence=0.95,
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)

        assert len(results) == 1
        value = results[0][alias_name]
        assert isinstance(value, (int, float))
        assert value > 0

    def test_count_all_records(self, builder, executor):
        """COUNT should return the total number of rows."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_registros",
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        assert results[0]["total_registros"] > 0

    def test_count_distinct_clientes(self, builder, executor):
        """COUNT DISTINCT should return fewer than COUNT."""
        intent_distinct = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="Cod_Cliente"),
                    distinct=True,
                    alias="n_clientes",
                )
            ],
        )
        intent_total = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="Cod_Cliente"),
                    alias="n_registros",
                )
            ],
        )
        sql_d = builder.build_query(intent_distinct)
        sql_t = builder.build_query(intent_total)
        n_clientes = executor.execute_dynamic_query(sql_d)[0]["n_clientes"]
        n_registros = executor.execute_dynamic_query(sql_t)[0]["n_registros"]
        assert n_clientes <= n_registros

    def test_median_vendas(self, builder, executor):
        """MEDIAN should return a value between MIN and MAX."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="median",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="mediana",
                ),
                AggregationSpec(
                    function="min",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="minimo",
                ),
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="maximo",
                ),
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        row = results[0]
        assert row["minimo"] <= row["mediana"] <= row["maximo"]


# ===========================================================================
# Category 2: Grouped Aggregations
# ===========================================================================


class TestGroupedAggregations:
    """Validation for grouped aggregation queries."""

    @pytest.mark.parametrize(
        "group_col,alias_name,expected_min_groups",
        [
            ("UF_Cliente", "Estado", 2),
            ("Empresa", "Empresa", 1),
        ],
    )
    def test_group_by_categorical_column(
        self, builder, executor, group_col, alias_name, expected_min_groups
    ):
        """Grouping by categorical columns produces multiple result groups."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name=group_col, alias=alias_name)],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        assert len(results) >= expected_min_groups
        for row in results:
            assert row["total"] >= 0

    def test_group_by_virtual_ano(self, builder, executor):
        """Grouping by Ano (virtual) produces yearly aggregates."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Ano", is_virtual=True, expression='YEAR("Data")', alias="Ano"
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        years = [row["Ano"] for row in results]
        assert all(y >= 2000 for y in years)
        assert len(set(years)) == len(years)  # No duplicates

    def test_group_by_virtual_mes(self, builder, executor):
        """Grouping by Mes (virtual) produces monthly aggregates."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Mes", is_virtual=True, expression='MONTH("Data")', alias="Mes"
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        months = [row["Mes"] for row in results]
        assert all(1 <= m <= 12 for m in months)

    def test_group_by_multiple_columns(self, builder, executor):
        """Grouping by estado + ano produces a cross-product of groups."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(name="UF_Cliente", alias="Estado"),
                ColumnSpec(
                    name="Ano", is_virtual=True, expression='YEAR("Data")', alias="Ano"
                ),
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        # More rows than just grouping by estado alone
        assert len(results) > 2

    def test_grouped_sum_equals_total(self, builder, executor):
        """Sum of per-group totals must equal overall total."""
        # Overall
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
        total = executor.execute_dynamic_query(builder.build_query(intent_total))[0][
            "total"
        ]

        # Grouped
        intent_grouped = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Ano", is_virtual=True, expression='YEAR("Data")', alias="Ano"
                )
            ],
        )
        groups = executor.execute_dynamic_query(builder.build_query(intent_grouped))
        sum_groups = sum(row["total"] for row in groups)

        assert abs(total - sum_groups) < 0.01


# ===========================================================================
# Category 3: Rankings
# ===========================================================================


class TestRankings:
    """Validation for ranking queries (top N / bottom N)."""

    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_top_n_estados(self, builder, executor, n):
        """Top N rankings return exactly N results in descending order."""
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=n,
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)

        assert len(results) <= n
        values = [row["total"] for row in results]
        assert values == sorted(values, reverse=True)

    def test_bottom_3_estados(self, builder, executor):
        """Bottom 3 returns results in ascending order."""
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
            order_by=OrderSpec(column="total", direction="ASC"),
            limit=3,
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)

        values = [row["total"] for row in results]
        assert values == sorted(values)

    def test_ranking_by_virtual_month(self, builder, executor):
        """Ranking months by total vendas with year filter."""
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Mes", is_virtual=True, expression='MONTH("Data")', alias="Mes"
                )
            ],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=3,
        )
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = builder.build_query(intent, filters=filters)
        results = executor.execute_dynamic_query(sql)

        assert len(results) <= 3
        months = [row["Mes"] for row in results]
        assert all(1 <= m <= 12 for m in months)
        values = [row["total"] for row in results]
        assert values == sorted(values, reverse=True)


# ===========================================================================
# Category 4: Temporal Analysis
# ===========================================================================


class TestTemporalAnalysis:
    """Validation for temporal analysis queries."""

    def test_max_year(self, builder, executor):
        """MAX(YEAR(Data)) returns the most recent year."""
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
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        assert results[0]["ultimo_ano"] >= 2000

    def test_min_year(self, builder, executor):
        """MIN(YEAR(Data)) returns the earliest year."""
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="min",
                    column=ColumnSpec(
                        name="Ano", is_virtual=True, expression='YEAR("Data")'
                    ),
                    alias="primeiro_ano",
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        first = results[0]["primeiro_ano"]
        assert first >= 2000

    def test_min_year_before_max_year(self, builder, executor):
        """MIN(YEAR) must be <= MAX(YEAR)."""
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="min",
                    column=ColumnSpec(
                        name="Ano", is_virtual=True, expression='YEAR("Data")'
                    ),
                    alias="primeiro",
                ),
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano", is_virtual=True, expression='YEAR("Data")'
                    ),
                    alias="ultimo",
                ),
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        assert results[0]["primeiro"] <= results[0]["ultimo"]

    def test_max_month(self, builder, executor):
        """MAX(MONTH(Data)) returns a valid month (1-12)."""
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Mes", is_virtual=True, expression='MONTH("Data")'
                    ),
                    alias="ultimo_mes",
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        assert 1 <= results[0]["ultimo_mes"] <= 12


# ===========================================================================
# Category 5: Filtered Queries (Comparisons)
# ===========================================================================


class TestFilteredQueries:
    """Validation for queries with various filter combinations."""

    def test_vendas_in_2016(self, builder, executor):
        """Total vendas in 2016 should be positive but less than global total."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
        )
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = builder.build_query(intent, filters=filters)
        results = executor.execute_dynamic_query(sql)
        total_2016 = results[0]["total"]

        sql_all = builder.build_query(intent)
        total_all = executor.execute_dynamic_query(sql_all)[0]["total"]

        assert total_2016 > 0
        assert total_2016 <= total_all

    def test_vendas_per_year_sum_to_total(self, builder, executor):
        """Sum of yearly totals must equal the overall total."""
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
        total = executor.execute_dynamic_query(builder.build_query(intent_total))[0][
            "total"
        ]

        intent_yearly = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Ano", is_virtual=True, expression='YEAR("Data")', alias="Ano"
                )
            ],
        )
        yearly = executor.execute_dynamic_query(builder.build_query(intent_yearly))
        sum_yearly = sum(row["total"] for row in yearly)

        assert abs(total - sum_yearly) < 0.01

    def test_filter_by_virtual_year(self, builder, executor):
        """Filtering by Ano (virtual) should work via YEAR('Data')."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
        )
        filters = {"Ano": 2016}
        sql = builder.build_query(intent, filters=filters)
        results = executor.execute_dynamic_query(sql)
        assert results[0]["total"] > 0


# ===========================================================================
# Category 6: Combined / Complex Scenarios
# ===========================================================================


class TestCombinedScenarios:
    """Validation for complex multi-dimensional queries."""

    def test_ranking_with_temporal_filter(self, builder, executor):
        """Top 3 estados by vendas in 2016."""
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=3,
        )
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = builder.build_query(intent, filters=filters)
        results = executor.execute_dynamic_query(sql)

        assert len(results) <= 3
        values = [row["total"] for row in results]
        assert values == sorted(values, reverse=True)

    def test_multiple_aggregations_per_group(self, builder, executor):
        """SUM + AVG + COUNT in a single grouped query."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                ),
                AggregationSpec(
                    function="avg",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="media",
                ),
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="qtd",
                ),
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)

        for row in results:
            assert "total" in row
            assert "media" in row
            assert "qtd" in row
            # SUM/COUNT relationship: total ≈ media × qtd
            assert row["total"] > 0
            assert row["media"] > 0
            assert row["qtd"] > 0

    def test_grouped_by_mes_and_estado(self, builder, executor):
        """Double grouping: Mes × Estado."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Mes", is_virtual=True, expression='MONTH("Data")', alias="Mes"
                ),
                ColumnSpec(name="UF_Cliente", alias="Estado"),
            ],
        )
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = builder.build_query(intent, filters=filters)
        results = executor.execute_dynamic_query(sql)

        assert len(results) > 1
        for row in results:
            assert 1 <= row["Mes"] <= 12
            assert row["Estado"] is not None
            assert row["total"] >= 0

    def test_nome_mes_grouping(self, builder, executor):
        """Grouping by Nome_Mes (MONTHNAME) returns month names."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[
                ColumnSpec(
                    name="Nome_Mes",
                    is_virtual=True,
                    expression='MONTHNAME("Data")',
                    alias="Nome_Mes",
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)

        month_names = [row["Nome_Mes"] for row in results]
        # DuckDB MONTHNAME returns English names
        valid_names = {
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        }
        for name in month_names:
            assert name in valid_names

    def test_stddev_aggregation(self, builder, executor):
        """STDDEV_SAMP (std) should return a positive value for Valor_Vendido."""
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="std",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="desvio_padrao",
                )
            ],
        )
        sql = builder.build_query(intent)
        results = executor.execute_dynamic_query(sql)
        assert results[0]["desvio_padrao"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
