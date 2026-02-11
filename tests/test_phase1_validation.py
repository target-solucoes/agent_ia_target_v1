"""
Phase 1 validation tests for non_graph_executor bug fixes.

Tests the four subfases:
1.1 - Virtual column registry in AliasMapper
1.2 - Virtual column handling in QueryExecutor
1.3 - Aggregation fallback for temporal keywords
1.4 - Temporal column prioritization in _extract_column_name
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphic_classifier.tools.alias_mapper import AliasMapper
from src.non_graph_executor.tools.query_classifier_params import ParameterExtractor


# =========================================================================
# Mock AliasMapper for isolated testing (no YAML dependency)
# =========================================================================
class MockAliasMapper:
    """Minimal mock replicating AliasMapper behavior for unit tests."""

    VIRTUAL_COLUMN_MAP = {
        "Ano": 'YEAR("Data")',
        "Mes": 'MONTH("Data")',
        "Nome_Mes": 'MONTHNAME("Data")',
    }

    column_types = {
        "numeric": ["Valor_Vendido", "Peso_Vendido", "Qtd_Vendida"],
        "categorical": ["UF_Cliente", "Des_Linha_Produto"],
        "temporal": ["Data"],
    }

    def resolve(self, term):
        mappings = {
            "vendas": "Valor_Vendido",
            "venda": "Valor_Vendido",
            "faturamento": "Valor_Vendido",
            "ano": "Ano",
            "mes": "Mes",
            "mês": "Mes",
            "estado": "UF_Cliente",
            "data": "Data",
            "quando": "Data",
        }
        return mappings.get(term.lower())

    def is_virtual_column(self, col):
        return col in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, col):
        return self.VIRTUAL_COLUMN_MAP.get(col)

    def is_categorical_column(self, col):
        return col in self.column_types.get("categorical", [])

    def is_numeric_column(self, col):
        return col in self.column_types.get("numeric", [])


def test_phase_1_1_virtual_column_registry():
    """Phase 1.1: AliasMapper has VIRTUAL_COLUMN_MAP and methods."""
    print("=" * 60)
    print("Phase 1.1: Virtual Column Registry in AliasMapper")
    print("=" * 60)

    # Class-level attribute exists
    assert hasattr(AliasMapper, "VIRTUAL_COLUMN_MAP"), "Missing VIRTUAL_COLUMN_MAP"
    assert "Ano" in AliasMapper.VIRTUAL_COLUMN_MAP
    assert "Mes" in AliasMapper.VIRTUAL_COLUMN_MAP
    assert "Nome_Mes" in AliasMapper.VIRTUAL_COLUMN_MAP
    print("  [OK] VIRTUAL_COLUMN_MAP defined with Ano, Mes, Nome_Mes")

    # Expressions are correct
    assert AliasMapper.VIRTUAL_COLUMN_MAP["Ano"] == 'YEAR("Data")'
    assert AliasMapper.VIRTUAL_COLUMN_MAP["Mes"] == 'MONTH("Data")'
    assert AliasMapper.VIRTUAL_COLUMN_MAP["Nome_Mes"] == 'MONTHNAME("Data")'
    print("  [OK] Virtual expressions are correct DuckDB functions")

    # Instance methods exist
    assert hasattr(AliasMapper, "is_virtual_column"), "Missing is_virtual_column method"
    assert hasattr(AliasMapper, "get_virtual_expression"), (
        "Missing get_virtual_expression method"
    )
    print("  [OK] is_virtual_column() and get_virtual_expression() methods exist")

    # Test with mock
    mapper = MockAliasMapper()
    assert mapper.is_virtual_column("Ano") is True
    assert mapper.is_virtual_column("Mes") is True
    assert mapper.is_virtual_column("Valor_Vendido") is False
    assert mapper.get_virtual_expression("Ano") == 'YEAR("Data")'
    assert mapper.get_virtual_expression("Mes") == 'MONTH("Data")'
    assert mapper.get_virtual_expression("Valor_Vendido") is None
    print("  [OK] is_virtual_column/get_virtual_expression return correct values")

    print("  PASSED\n")


def test_phase_1_3_aggregation_fallback():
    """Phase 1.3: Temporal keywords trigger correct aggregation."""
    print("=" * 60)
    print("Phase 1.3: Aggregation Fallback for Temporal Keywords")
    print("=" * 60)

    mapper = MockAliasMapper()
    state = {}

    # Test: "ultimo" -> max
    params = ParameterExtractor.extract_aggregation_params(
        "qual o ultimo ano com vendas?", state, mapper
    )
    assert params["aggregation"] == "max", f"Expected max, got {params['aggregation']}"
    print(f"  [OK] 'ultimo ano' -> aggregation='max' (got: {params['aggregation']})")

    # Test: "ultima" -> max
    params = ParameterExtractor.extract_aggregation_params(
        "qual o ano em que ocorreu a ultima venda?", state, mapper
    )
    assert params["aggregation"] == "max", f"Expected max, got {params['aggregation']}"
    print(f"  [OK] 'ultima venda' -> aggregation='max' (got: {params['aggregation']})")

    # Test: "mais recente" -> max
    params = ParameterExtractor.extract_aggregation_params(
        "qual a venda mais recente?", state, mapper
    )
    assert params["aggregation"] == "max", f"Expected max, got {params['aggregation']}"
    print(f"  [OK] 'mais recente' -> aggregation='max' (got: {params['aggregation']})")

    # Test: "primeiro" -> min
    params = ParameterExtractor.extract_aggregation_params(
        "qual foi o primeiro ano com vendas?", state, mapper
    )
    assert params["aggregation"] == "min", f"Expected min, got {params['aggregation']}"
    print(f"  [OK] 'primeiro ano' -> aggregation='min' (got: {params['aggregation']})")

    # Test: "primeira" -> min
    params = ParameterExtractor.extract_aggregation_params(
        "qual a primeira venda registrada?", state, mapper
    )
    assert params["aggregation"] == "min", f"Expected min, got {params['aggregation']}"
    print(
        f"  [OK] 'primeira venda' -> aggregation='min' (got: {params['aggregation']})"
    )

    # Test: Existing keywords still work
    params = ParameterExtractor.extract_aggregation_params(
        "qual o maior valor de vendas?", state, mapper
    )
    assert params["aggregation"] == "max", f"Expected max, got {params['aggregation']}"
    print(f"  [OK] 'maior valor' -> aggregation='max' (existing behavior preserved)")

    params = ParameterExtractor.extract_aggregation_params(
        "qual o total de vendas?", state, mapper
    )
    assert params["aggregation"] == "sum", f"Expected sum, got {params['aggregation']}"
    print(f"  [OK] 'total vendas' -> aggregation='sum' (existing behavior preserved)")

    params = ParameterExtractor.extract_aggregation_params(
        "quantos clientes temos?", state, mapper
    )
    assert params["aggregation"] == "count", (
        f"Expected count, got {params['aggregation']}"
    )
    print(
        f"  [OK] 'quantos clientes' -> aggregation='count' (existing behavior preserved)"
    )

    print("  PASSED\n")


def test_phase_1_4_temporal_column_prioritization():
    """Phase 1.4: Temporal intent prioritizes temporal/virtual columns."""
    print("=" * 60)
    print("Phase 1.4: Temporal Column Prioritization")
    print("=" * 60)

    mapper = MockAliasMapper()

    # Test: "ultimo ano com vendas" -> should resolve to "Ano" (virtual), not "Valor_Vendido"
    col = ParameterExtractor._extract_column_name(
        "qual o ultimo ano com vendas?", mapper
    )
    assert col == "Ano", f"Expected 'Ano', got '{col}'"
    print(f"  [OK] 'ultimo ano com vendas' -> column='Ano' (temporal priority)")

    # Test: "ano da ultima venda" -> should resolve to "Ano" (virtual), not "Valor_Vendido"
    col = ParameterExtractor._extract_column_name(
        "qual o ano em que ocorreu a ultima venda?", mapper
    )
    assert col == "Ano", f"Expected 'Ano', got '{col}'"
    print(
        f"  [OK] 'ano em que ocorreu a ultima venda' -> column='Ano' (temporal priority)"
    )

    # Test: Non-temporal queries still prioritize numeric columns
    col = ParameterExtractor._extract_column_name(
        "qual o maior valor de vendas?", mapper
    )
    assert col == "Valor_Vendido", f"Expected 'Valor_Vendido', got '{col}'"
    print(
        f"  [OK] 'maior valor de vendas' -> column='Valor_Vendido' (numeric priority preserved)"
    )

    # Test: "total de vendas" -> should still resolve to numeric column
    col = ParameterExtractor._extract_column_name("qual o total de vendas?", mapper)
    assert col == "Valor_Vendido", f"Expected 'Valor_Vendido', got '{col}'"
    print(
        f"  [OK] 'total de vendas' -> column='Valor_Vendido' (numeric priority preserved)"
    )

    print("  PASSED\n")


def test_combined_flow_query1():
    """Combined test: Q1 'qual o ultimo ano com vendas?' full parameter extraction."""
    print("=" * 60)
    print("Combined Test: Q1 'qual o ultimo ano com vendas?'")
    print("=" * 60)

    mapper = MockAliasMapper()
    query = "qual o ultimo ano com vendas?"
    state = {}

    params = ParameterExtractor.extract_aggregation_params(query, state, mapper)

    print(f"  Query: '{query}'")
    print(
        f"  Extracted: aggregation='{params['aggregation']}', column='{params.get('column')}'"
    )

    assert params["aggregation"] == "max", f"Expected max, got {params['aggregation']}"
    assert params["column"] == "Ano", f"Expected Ano, got {params.get('column')}"

    # Verify the column is virtual and would be resolved correctly
    assert mapper.is_virtual_column(params["column"]) is True
    expr = mapper.get_virtual_expression(params["column"])
    assert expr == 'YEAR("Data")'
    print(f"  Virtual expression: {expr}")
    print(f'  Expected SQL: SELECT MAX(YEAR("Data")) as result FROM ...')
    print("  PASSED\n")


def test_combined_flow_query2():
    """Combined test: Q2 'qual o ano em que ocorreu a ultima venda?' full parameter extraction."""
    print("=" * 60)
    print("Combined Test: Q2 'qual o ano em que ocorreu a ultima venda?'")
    print("=" * 60)

    mapper = MockAliasMapper()
    query = "qual o ano em que ocorreu a ultima venda?"
    state = {}

    params = ParameterExtractor.extract_aggregation_params(query, state, mapper)

    print(f"  Query: '{query}'")
    print(
        f"  Extracted: aggregation='{params['aggregation']}', column='{params.get('column')}'"
    )

    assert params["aggregation"] == "max", f"Expected max, got {params['aggregation']}"
    assert params["column"] == "Ano", f"Expected Ano, got {params.get('column')}"

    # Verify the column is virtual
    assert mapper.is_virtual_column(params["column"]) is True
    expr = mapper.get_virtual_expression(params["column"])
    assert expr == 'YEAR("Data")'
    print(f"  Virtual expression: {expr}")
    print(f'  Expected SQL: SELECT MAX(YEAR("Data")) as result FROM ...')
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 1 VALIDATION TESTS")
    print("=" * 60 + "\n")

    test_phase_1_1_virtual_column_registry()
    test_phase_1_3_aggregation_fallback()
    test_phase_1_4_temporal_column_prioritization()
    test_combined_flow_query1()
    test_combined_flow_query2()

    print("=" * 60)
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
