"""
Quick validation test for DynamicQueryBuilder (Phase 3.1).

This script tests SQL generation for the 4 critical query scenarios
described in the non_graph_executor diagnosis document.
"""

from src.non_graph_executor.tools.dynamic_query_builder import DynamicQueryBuilder
from src.non_graph_executor.models.intent_schema import (
    QueryIntent,
    ColumnSpec,
    AggregationSpec,
    OrderSpec,
)


class MockAliasMapper:
    """Minimal mock for AliasMapper to test DynamicQueryBuilder."""

    VIRTUAL_COLUMN_MAP = {
        "Ano": 'YEAR("Data")',
        "Mes": 'MONTH("Data")',
        "Nome_Mes": 'MONTHNAME("Data")',
    }
    column_types = {
        "numeric": ["Valor_Vendido", "Peso_Vendido", "Qtd_Vendida"],
        "categorical": ["UF_Cliente", "Empresa", "Cod_Familia_Produto"],
        "temporal": ["Data"],
    }
    aliases = {
        "columns": {
            "Valor_Vendido": ["vendas", "faturamento"],
            "UF_Cliente": ["estado"],
            "Cod_Cliente": ["cliente"],
        }
    }

    def is_virtual_column(self, name):
        return name in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, name):
        return self.VIRTUAL_COLUMN_MAP.get(name)

    def is_categorical_column(self, name):
        return name in self.column_types.get("categorical", [])


def test_all():
    builder = DynamicQueryBuilder(
        MockAliasMapper(),
        "data/datasets/DadosComercial_resumido_v02.parquet",
    )

    # Test 1: Q1 - "qual o ultimo ano com vendas?" → MAX(YEAR("Data"))
    print("=" * 60)
    print("Test 1: 'qual o ultimo ano com vendas?'")
    print('Expected: SELECT MAX(YEAR("Data")) as ultimo_ano FROM ...')
    intent1 = QueryIntent(
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
        reasoning="MAX(YEAR(Data))",
    )
    sql1 = builder.build_query(intent1)
    print(f"Generated:\n{sql1}")
    assert 'YEAR("Data")' in sql1, "Should contain YEAR expression"
    assert "MAX" in sql1, "Should contain MAX"
    print("✓ PASSED\n")

    # Test 2: Q2 - "qual o ano em que ocorreu a ultima venda?"
    print("=" * 60)
    print("Test 2: 'qual o ano em que ocorreu a ultima venda?'")
    print('Expected: SELECT MAX(YEAR("Data")) as ano_ultima_venda FROM ...')
    intent2 = QueryIntent(
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
        reasoning="MAX(YEAR(Data)) - year of last sale",
    )
    sql2 = builder.build_query(intent2)
    print(f"Generated:\n{sql2}")
    assert 'YEAR("Data")' in sql2
    assert "MAX" in sql2
    print("✓ PASSED\n")

    # Test 3: Q3 - "qual foi o mes com maior valor de venda em 2016?"
    # Requires GROUP BY + ORDER BY + LIMIT
    print("=" * 60)
    print("Test 3: 'qual foi o mes com maior valor de venda em 2016?'")
    print('Expected: SELECT MONTH("Data") as Mes, SUM("Valor_Vendido") as total_vendas')
    print('          FROM ... WHERE ... GROUP BY MONTH("Data") ORDER BY ... LIMIT 1')
    intent3 = QueryIntent(
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
        reasoning="Month with highest total sales",
    )
    filters_2016 = {"Data": {"between": ["2016-01-01", "2016-12-31"]}}
    sql3 = builder.build_query(intent3, filters=filters_2016)
    print(f"Generated:\n{sql3}")
    assert 'MONTH("Data")' in sql3, "Should contain MONTH expression"
    assert "GROUP BY" in sql3, "Should contain GROUP BY"
    assert "ORDER BY" in sql3, "Should contain ORDER BY"
    assert "LIMIT 1" in sql3, "Should contain LIMIT 1"
    assert "Valor_Vendido" in sql3, "Should aggregate Valor_Vendido"
    assert "BETWEEN" in sql3, "Should have date filter"
    print("✓ PASSED\n")

    # Test 4: Q4 - "que mes foi o maior valor de vendas?" (same without filter)
    print("=" * 60)
    print("Test 4: 'que mes foi o maior valor de vendas?'")
    sql4 = builder.build_query(intent3)  # Same intent, no filters
    print(f"Generated:\n{sql4}")
    assert 'MONTH("Data")' in sql4
    assert "GROUP BY" in sql4
    assert "BETWEEN" not in sql4, "Should NOT have date filter"
    print("✓ PASSED\n")

    # Test 5: Ranking - "top 5 estados por faturamento"
    print("=" * 60)
    print("Test 5: 'top 5 estados por faturamento'")
    intent5 = QueryIntent(
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
        confidence=0.95,
        reasoning="Top 5 states by revenue",
    )
    sql5 = builder.build_query(intent5)
    print(f"Generated:\n{sql5}")
    assert "UF_Cliente" in sql5, "Should contain UF_Cliente"
    assert "GROUP BY" in sql5
    assert "ORDER BY" in sql5
    assert "LIMIT 5" in sql5
    print("✓ PASSED\n")

    # Test 6: Count distinct - "quantos clientes temos?"
    print("=" * 60)
    print("Test 6: 'quantos clientes temos?'")
    intent6 = QueryIntent(
        intent_type="simple_aggregation",
        aggregations=[
            AggregationSpec(
                function="count",
                column=ColumnSpec(name="Cod_Cliente"),
                distinct=True,
                alias="total_clientes",
            )
        ],
        confidence=0.95,
        reasoning="COUNT(DISTINCT Cod_Cliente)",
    )
    sql6 = builder.build_query(intent6)
    print(f"Generated:\n{sql6}")
    assert "COUNT(DISTINCT" in sql6, "Should use COUNT(DISTINCT)"
    assert "Cod_Cliente" in sql6
    print("✓ PASSED\n")

    # Test 7: Vendas por mes (no ranking, just grouped)
    print("=" * 60)
    print("Test 7: 'vendas por mes em 2016'")
    intent7 = QueryIntent(
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
        confidence=0.90,
        reasoning="Sales grouped by month",
    )
    sql7 = builder.build_query(intent7, filters=filters_2016)
    print(f"Generated:\n{sql7}")
    assert "GROUP BY" in sql7
    assert "ORDER BY" not in sql7, "Should NOT have ORDER BY"
    assert "LIMIT" not in sql7, "Should NOT have LIMIT"
    print("✓ PASSED\n")

    # Test 8: Validation
    print("=" * 60)
    print("Test 8: Validation")
    warnings = builder.validate_intent(intent3)
    print(f"Warnings for valid intent: {warnings}")
    assert len(warnings) == 0, "Should have no warnings for valid intent"
    print("✓ PASSED\n")

    print("=" * 60)
    print("ALL 8 TESTS PASSED ✓")


if __name__ == "__main__":
    test_all()
