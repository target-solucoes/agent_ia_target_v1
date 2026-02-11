"""
Phase 4.3 — Comprehensive unit tests for DynamicQueryBuilder.

Tests the SQL generation engine across:
  - SELECT clause: aggregations, GROUP BY dimensions, select columns
  - GROUP BY clause: real columns, virtual columns, multiple groupings
  - ORDER BY clause: alias references, column references, ASC/DESC
  - LIMIT clause: explicit limits, max-limit cap
  - WHERE clause: equality, IN, BETWEEN, operators, virtual column filters,
    case-insensitive categorical filters
  - Column resolution: real columns, virtual columns (Ano/Mes/Nome_Mes)
  - Aggregation functions: sum, avg, count, min, max, median, std
  - COUNT DISTINCT
  - Filter merging: session filters + additional_filters from intent
  - validate_intent(): valid/invalid intents
  - _safe_alias(): sanitization
  - Edge cases: empty intents, unknown columns, negative limits
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.non_graph_executor.tools.dynamic_query_builder import (
    AGG_FUNCTION_MAP,
    DEFAULT_MAX_LIMIT,
    DynamicQueryBuilder,
)
from src.non_graph_executor.models.intent_schema import (
    AggregationSpec,
    ColumnSpec,
    OrderSpec,
    QueryIntent,
)


# ---------------------------------------------------------------------------
# Mock AliasMapper
# ---------------------------------------------------------------------------


class MockAliasMapper:
    """Full-featured mock of AliasMapper for DynamicQueryBuilder tests."""

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


DATA_SOURCE = "data/datasets/DadosComercial_resumido_v02.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder():
    return DynamicQueryBuilder(MockAliasMapper(), DATA_SOURCE)


# ===========================================================================
# Q1-Q4: Critical Query Scenarios (from diagnosis)
# ===========================================================================


class TestCriticalQueryScenarios:
    """Reproduce and validate the 4 critical query scenarios."""

    def test_q1_ultimo_ano_com_vendas(self, builder):
        """Q1: MAX(YEAR('Data')) for last year with sales."""
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
        sql = builder.build_query(intent)
        assert 'YEAR("Data")' in sql
        assert "MAX" in sql
        assert "ultimo_ano" in sql
        assert "GROUP BY" not in sql

    def test_q2_ano_ultima_venda(self, builder):
        """Q2: Same as Q1 — MAX(YEAR('Data'))."""
        intent = QueryIntent(
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
        sql = builder.build_query(intent)
        assert 'MAX(YEAR("Data"))' in sql

    def test_q3_mes_maior_venda_2016(self, builder):
        """Q3: GROUP BY MONTH + SUM + ORDER DESC + LIMIT 1 + year filter."""
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
        filters = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
        sql = builder.build_query(intent, filters=filters)

        assert 'MONTH("Data")' in sql
        assert "GROUP BY" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 1" in sql
        assert "Valor_Vendido" in sql
        assert "BETWEEN" in sql

    def test_q4_mes_maior_valor_vendas_sem_filtro(self, builder):
        """Q4: Same as Q3 but without date filter."""
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
        sql = builder.build_query(intent)
        assert "GROUP BY" in sql
        assert "BETWEEN" not in sql
        assert "WHERE" not in sql


# ===========================================================================
# SELECT Clause Tests
# ===========================================================================


class TestSelectClause:
    """Tests for SELECT clause generation."""

    def test_single_aggregation(self, builder):
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
        sql = builder.build_query(intent)
        assert 'SUM("Valor_Vendido") as total' in sql

    def test_multiple_aggregations(self, builder):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                ),
                AggregationSpec(
                    function="avg",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="media_vendas",
                ),
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="Cod_Cliente"),
                    distinct=True,
                    alias="n_clientes",
                ),
            ],
        )
        sql = builder.build_query(intent)
        assert "SUM" in sql
        assert "AVG" in sql
        assert "COUNT(DISTINCT" in sql

    def test_group_by_dimension_in_select(self, builder):
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
        )
        sql = builder.build_query(intent)
        assert '"UF_Cliente" as Estado' in sql
        assert "total" in sql

    def test_virtual_column_in_select(self, builder):
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
        assert 'MONTH("Data") as Mes' in sql

    def test_empty_intent_produces_select_star(self, builder):
        intent = QueryIntent(intent_type="metadata")
        sql = builder.build_query(intent)
        assert "SELECT *" in sql


# ===========================================================================
# Aggregation Function Tests
# ===========================================================================


class TestAggregationFunctions:
    """Tests for all supported aggregation functions."""

    @pytest.mark.parametrize(
        "func,sql_func",
        [
            ("sum", "SUM"),
            ("avg", "AVG"),
            ("count", "COUNT"),
            ("min", "MIN"),
            ("max", "MAX"),
            ("median", "MEDIAN"),
            ("std", "STDDEV_SAMP"),
        ],
    )
    def test_all_aggregation_functions(self, builder, func, sql_func):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function=func,
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="result",
                )
            ],
        )
        sql = builder.build_query(intent)
        assert sql_func in sql

    def test_count_distinct(self, builder):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="Cod_Cliente"),
                    distinct=True,
                    alias="total_clientes",
                )
            ],
        )
        sql = builder.build_query(intent)
        assert "COUNT(DISTINCT" in sql
        assert "Cod_Cliente" in sql

    def test_invalid_aggregation_function_raises(self, builder):
        """Pydantic Literal validation rejects invalid function at schema level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AggregationSpec(function="invalid", column=ColumnSpec(name="Valor_Vendido"))

    def test_virtual_column_in_aggregation(self, builder):
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
        assert 'MAX(YEAR("Data"))' in sql


# ===========================================================================
# GROUP BY Clause Tests
# ===========================================================================


class TestGroupByClause:
    """Tests for GROUP BY clause generation."""

    def test_single_real_column(self, builder):
        intent = QueryIntent(
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
        sql = builder.build_query(intent)
        assert 'GROUP BY "UF_Cliente"' in sql

    def test_single_virtual_column(self, builder):
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
                ColumnSpec(name="Mes", is_virtual=True, expression='MONTH("Data")')
            ],
        )
        sql = builder.build_query(intent)
        assert 'GROUP BY MONTH("Data")' in sql

    def test_multiple_group_by_columns(self, builder):
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
                ColumnSpec(name="UF_Cliente"),
                ColumnSpec(name="Ano", is_virtual=True, expression='YEAR("Data")'),
            ],
        )
        sql = builder.build_query(intent)
        assert "GROUP BY" in sql
        assert '"UF_Cliente"' in sql
        assert 'YEAR("Data")' in sql

    def test_no_group_by_when_simple_aggregation(self, builder):
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
        sql = builder.build_query(intent)
        assert "GROUP BY" not in sql

    def test_nome_mes_virtual_column(self, builder):
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
                    name="Nome_Mes", is_virtual=True, expression='MONTHNAME("Data")'
                )
            ],
        )
        sql = builder.build_query(intent)
        assert 'MONTHNAME("Data")' in sql
        assert "GROUP BY" in sql


# ===========================================================================
# ORDER BY Clause Tests
# ===========================================================================


class TestOrderByClause:
    """Tests for ORDER BY clause generation."""

    def test_order_by_aggregation_alias_desc(self, builder):
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=5,
        )
        sql = builder.build_query(intent)
        assert "ORDER BY total DESC" in sql

    def test_order_by_aggregation_alias_asc(self, builder):
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total", direction="ASC"),
        )
        sql = builder.build_query(intent)
        assert "ORDER BY total ASC" in sql

    def test_no_order_by_when_not_specified(self, builder):
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
                ColumnSpec(name="Mes", is_virtual=True, expression='MONTH("Data")')
            ],
        )
        sql = builder.build_query(intent)
        assert "ORDER BY" not in sql

    def test_invalid_direction_rejected_by_schema(self, builder):
        """Pydantic Literal validation rejects invalid direction at schema level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OrderSpec(column="total", direction="INVALID")


# ===========================================================================
# LIMIT Clause Tests
# ===========================================================================


class TestLimitClause:
    """Tests for LIMIT clause generation."""

    def test_explicit_limit(self, builder):
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=5,
        )
        sql = builder.build_query(intent)
        assert "LIMIT 5" in sql

    def test_limit_1(self, builder):
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
                ColumnSpec(name="Mes", is_virtual=True, expression='MONTH("Data")')
            ],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=1,
        )
        sql = builder.build_query(intent)
        assert "LIMIT 1" in sql

    def test_no_limit_when_not_specified(self, builder):
        intent = QueryIntent(
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
        sql = builder.build_query(intent)
        assert "LIMIT" not in sql

    def test_large_limit_capped_at_max(self, builder):
        intent = QueryIntent(
            intent_type="tabular",
            limit=99999,
        )
        sql = builder.build_query(intent)
        assert f"LIMIT {DEFAULT_MAX_LIMIT}" in sql


# ===========================================================================
# WHERE Clause Tests
# ===========================================================================


class TestWhereClause:
    """Tests for WHERE clause generation."""

    def test_equality_filter_numeric(self, builder):
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
        assert "WHERE" in sql
        assert "2016" in sql
        # "Ano" is a virtual column — should resolve to YEAR("Data")
        assert 'YEAR("Data")' in sql

    def test_between_filter(self, builder):
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
        assert "BETWEEN" in sql
        assert "2016-01-01" in sql
        assert "2016-12-31" in sql

    def test_in_filter_with_list(self, builder):
        intent = QueryIntent(
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
        filters = {"UF_Cliente": ["SP", "RJ", "MG"]}
        sql = builder.build_query(intent, filters=filters)
        assert "IN" in sql
        # UF_Cliente is categorical → case-insensitive with UPPER
        assert "UPPER" in sql

    def test_operator_filter(self, builder):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="count", column=ColumnSpec(name="Cod_Cliente"), alias="n"
                )
            ],
        )
        filters = {"Valor_Vendido": {"operator": ">=", "value": 1000}}
        sql = builder.build_query(intent, filters=filters)
        assert ">=" in sql
        assert "1000" in sql

    def test_no_where_when_no_filters(self, builder):
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
        sql = builder.build_query(intent)
        assert "WHERE" not in sql

    def test_virtual_column_filter_mes(self, builder):
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
        filters = {"Mes": 3}
        sql = builder.build_query(intent, filters=filters)
        assert 'MONTH("Data")' in sql
        assert "3" in sql

    def test_categorical_equality_filter_case_insensitive(self, builder):
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
        filters = {"UF_Cliente": "SP"}
        sql = builder.build_query(intent, filters=filters)
        assert "UPPER" in sql


# ===========================================================================
# Filter Merging Tests
# ===========================================================================


class TestFilterMerging:
    """Tests for merging session filters with additional_filters."""

    def test_session_filters_only(self, builder):
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
        assert "2016" in sql

    def test_additional_filters_only(self, builder):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            additional_filters={"Ano": 2016},
        )
        sql = builder.build_query(intent, filters=None)
        assert "2016" in sql

    def test_merged_filters(self, builder):
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            additional_filters={"Ano": 2016},
        )
        session_filters = {"UF_Cliente": "SP"}
        sql = builder.build_query(intent, filters=session_filters)
        assert "2016" in sql
        assert "SP" in sql


# ===========================================================================
# Column Resolution Tests
# ===========================================================================


class TestColumnResolution:
    """Tests for column expression resolution."""

    def test_real_column_gets_double_quotes(self, builder):
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
        sql = builder.build_query(intent)
        assert '"Valor_Vendido"' in sql

    def test_virtual_column_with_expression(self, builder):
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
                ColumnSpec(name="Ano", is_virtual=True, expression='YEAR("Data")')
            ],
        )
        sql = builder.build_query(intent)
        assert 'YEAR("Data")' in sql
        # Should NOT have "Ano" as a quoted column
        assert '"Ano"' not in sql

    def test_virtual_column_without_expression_resolved_from_registry(self, builder):
        """If ColumnSpec has is_virtual=True but no expression, builder resolves it."""
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
                ColumnSpec(name="Ano", is_virtual=True)  # No expression provided
            ],
        )
        sql = builder.build_query(intent)
        assert 'YEAR("Data")' in sql

    def test_untagged_virtual_column_resolved_from_registry(self, builder):
        """If ColumnSpec is NOT tagged as virtual but name matches, builder resolves it."""
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
                ColumnSpec(name="Ano")  # Not tagged as virtual
            ],
        )
        sql = builder.build_query(intent)
        # DynamicQueryBuilder checks _is_virtual_column internally
        assert 'YEAR("Data")' in sql


# ===========================================================================
# Ranking Tests
# ===========================================================================


class TestRankings:
    """Tests for ranking query generation."""

    def test_top_5_estados_por_faturamento(self, builder):
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_faturamento",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
            order_by=OrderSpec(column="total_faturamento", direction="DESC"),
            limit=5,
        )
        sql = builder.build_query(intent)
        assert '"UF_Cliente"' in sql
        assert "GROUP BY" in sql
        assert "ORDER BY" in sql
        assert "DESC" in sql
        assert "LIMIT 5" in sql

    def test_top_1_is_equivalent_to_max_group(self, builder):
        """Top 1 by DESC = finding the maximum group."""
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="Empresa")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=1,
        )
        sql = builder.build_query(intent)
        assert "LIMIT 1" in sql
        assert "GROUP BY" in sql

    def test_bottom_5_ascending(self, builder):
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total", direction="ASC"),
            limit=5,
        )
        sql = builder.build_query(intent)
        assert "ASC" in sql
        assert "LIMIT 5" in sql


# ===========================================================================
# Validation Tests
# ===========================================================================


class TestValidateIntent:
    """Tests for validate_intent()."""

    def test_valid_intent_no_warnings(self, builder):
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=5,
        )
        warnings = builder.validate_intent(intent)
        assert len(warnings) == 0

    def test_invalid_aggregation_function_rejected_by_schema(self, builder):
        """Pydantic Literal validation rejects invalid function names at schema level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AggregationSpec(
                function="invalid_func", column=ColumnSpec(name="Valor_Vendido")
            )

    def test_unknown_column_warning(self, builder):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum", column=ColumnSpec(name="Coluna_Inexistente")
                )
            ],
        )
        warnings = builder.validate_intent(intent)
        assert any(
            "não reconhecida" in w or "Coluna_Inexistente" in w for w in warnings
        )

    def test_valid_virtual_column_no_warning(self, builder):
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max", column=ColumnSpec(name="Ano", is_virtual=True)
                )
            ],
        )
        warnings = builder.validate_intent(intent)
        assert len(warnings) == 0

    def test_negative_limit_warning(self, builder):
        intent = QueryIntent(intent_type="tabular", limit=-5)
        warnings = builder.validate_intent(intent)
        assert any("inválido" in w.lower() or "limit" in w.lower() for w in warnings)


# ===========================================================================
# _safe_alias Tests
# ===========================================================================


class TestSafeAlias:
    """Tests for _safe_alias() sanitization."""

    def test_normal_alias(self, builder):
        assert builder._safe_alias("total_vendas") == "total_vendas"

    def test_alias_with_spaces(self, builder):
        result = builder._safe_alias("total vendas")
        assert " " not in result

    def test_alias_with_special_chars(self, builder):
        result = builder._safe_alias("total@vendas!")
        assert "@" not in result
        assert "!" not in result

    def test_alias_starting_with_number(self, builder):
        result = builder._safe_alias("123abc")
        assert not result[0].isdigit()

    def test_empty_alias(self, builder):
        assert builder._safe_alias("") == "resultado"

    def test_none_alias(self, builder):
        assert builder._safe_alias(None) == "resultado"


# ===========================================================================
# FROM Clause Test
# ===========================================================================


class TestFromClause:
    """Tests for FROM clause."""

    def test_from_clause_contains_data_source(self, builder):
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
        sql = builder.build_query(intent)
        assert DATA_SOURCE in sql
        assert "FROM" in sql


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_metadata_intent_produces_select_star(self, builder):
        intent = QueryIntent(intent_type="metadata")
        sql = builder.build_query(intent)
        assert "SELECT *" in sql

    def test_agg_function_map_has_all_expected_functions(self):
        expected = {"sum", "avg", "count", "min", "max", "median", "std"}
        assert set(AGG_FUNCTION_MAP.keys()) == expected

    def test_default_max_limit_is_1000(self):
        assert DEFAULT_MAX_LIMIT == 1000

    def test_sql_structure_order(self, builder):
        """SQL parts appear in correct order: SELECT → FROM → WHERE → GROUP BY → ORDER BY → LIMIT."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total", direction="DESC"),
            limit=10,
        )
        filters = {"Ano": 2016}
        sql = builder.build_query(intent, filters=filters)

        select_pos = sql.index("SELECT")
        from_pos = sql.index("FROM")
        where_pos = sql.index("WHERE")
        group_pos = sql.index("GROUP BY")
        order_pos = sql.index("ORDER BY")
        limit_pos = sql.index("LIMIT")

        assert select_pos < from_pos < where_pos < group_pos < order_pos < limit_pos


# ===========================================================================
# Temporal Date Range Detection Tests
# ===========================================================================


class TestTemporalDateRangeDetection:
    """Tests for _is_temporal_date_range() and BETWEEN generation from date lists."""

    def test_date_list_generates_between_not_in(self, builder):
        """Critical fix: Date list filter should produce BETWEEN, not IN."""
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
        # This is the format produced by filter_classifier for date ranges
        filters = {"Data": ["2016-01-01", "2016-12-31"]}
        sql = builder.build_query(intent, filters=filters)
        assert "BETWEEN" in sql
        assert "IN" not in sql
        assert "'2016-01-01'" in sql
        assert "'2016-12-31'" in sql

    def test_date_list_with_dict_between_also_works(self, builder):
        """Dict BETWEEN format should still work as before."""
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
        assert "BETWEEN" in sql

    def test_non_date_list_still_uses_in(self, builder):
        """Non-temporal list filters should still use IN clause."""
        intent = QueryIntent(
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
        filters = {"UF_Cliente": ["SP", "RJ"]}
        sql = builder.build_query(intent, filters=filters)
        assert "IN" in sql
        assert "BETWEEN" not in sql

    def test_three_element_date_list_uses_in(self, builder):
        """A list with 3+ dates should use IN even on temporal columns."""
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
        filters = {"Data": ["2016-01-01", "2016-06-15", "2016-12-31"]}
        sql = builder.build_query(intent, filters=filters)
        assert "IN" in sql
        assert "BETWEEN" not in sql

    def test_is_temporal_date_range_positive(self, builder):
        """Directly test _is_temporal_date_range helper."""
        assert builder._is_temporal_date_range("Data", ["2016-01-01", "2016-12-31"])
        assert builder._is_temporal_date_range("data", ["2015-05-01", "2015-12-31"])

    def test_is_temporal_date_range_negative(self, builder):
        """Non-temporal columns or non-date values should return False."""
        # Non-temporal column
        assert not builder._is_temporal_date_range("UF_Cliente", ["SP", "RJ"])
        # 3 values
        assert not builder._is_temporal_date_range(
            "Data", ["2016-01-01", "2016-06-15", "2016-12-31"]
        )
        # 1 value
        assert not builder._is_temporal_date_range("Data", ["2016-01-01"])
        # Non-date strings
        assert not builder._is_temporal_date_range("Data", ["abc", "def"])
        # Numeric values
        assert not builder._is_temporal_date_range("Data", [2016, 2017])

    def test_date_range_filter_with_dynamic_query_execution(self, builder):
        """Full E2E: date list filter → BETWEEN SQL → valid DuckDB query."""
        import duckdb

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
        filters = {"Data": ["2016-01-01", "2016-12-31"]}
        sql = builder.build_query(intent, filters=filters)
        assert "BETWEEN" in sql
        # Execute against real dataset
        with duckdb.connect() as conn:
            result = conn.execute(sql).fetchall()
        assert len(result) == 1
        assert result[0][0] == 2016


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
