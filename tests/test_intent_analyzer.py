"""
Phase 4.2 — Unit tests for IntentAnalyzer.

Tests the LLM-based intent analysis engine across multiple query categories:
  - Simple aggregations (sum, avg, count)
  - Aggregations with grouping (by month, state, product)
  - Rankings (top N)
  - Temporal queries (último ano, primeiro mês)
  - Metadata (quantas linhas, quais colunas)
  - Conversational (saudações)
  - Lookups (dados do cliente X)
  - Tabular (mostre tabela)

All tests use mocked LLM to run offline without API keys.
Tests validate both parsing/construction logic and enrichment of virtual columns.
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
from src.non_graph_executor.tools.intent_analyzer import (
    IntentAnalyzer,
    build_schema_context,
)


# ---------------------------------------------------------------------------
# Mock AliasMapper
# ---------------------------------------------------------------------------


class MockAliasMapper:
    """Full-featured mock of AliasMapper for IntentAnalyzer tests."""

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
            "Ano": ["ano", "anual"],
            "Mes": ["mes", "mês", "mensal"],
        }
    }

    def is_virtual_column(self, col_name: str) -> bool:
        return col_name in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, col_name: str):
        return self.VIRTUAL_COLUMN_MAP.get(col_name)

    def resolve(self, term: str):
        mapping = {
            "vendas": "Valor_Vendido",
            "venda": "Valor_Vendido",
            "faturamento": "Valor_Vendido",
            "ano": "Ano",
            "mes": "Mes",
            "estado": "UF_Cliente",
            "cliente": "Cod_Cliente",
            "empresa": "Empresa",
        }
        return mapping.get(term.lower())


# ---------------------------------------------------------------------------
# Helper to create analyzer with mocked LLM returning a specific JSON
# ---------------------------------------------------------------------------


def _make_analyzer_with_response(response_dict: Dict[str, Any]) -> IntentAnalyzer:
    """Create IntentAnalyzer with LLM mocked to return specific JSON response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_dict)
    mock_response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    mock_llm.invoke.return_value = mock_response

    return IntentAnalyzer(llm=mock_llm, alias_mapper=MockAliasMapper())


def _make_analyzer_no_response() -> IntentAnalyzer:
    """Create IntentAnalyzer with LLM mocked (no invoke)."""
    return IntentAnalyzer(llm=MagicMock(), alias_mapper=MockAliasMapper())


# ===========================================================================
# Schema Context Builder
# ===========================================================================


class TestSchemaContextBuilder:
    """Tests for build_schema_context()."""

    def test_includes_column_types(self):
        ctx = build_schema_context(MockAliasMapper())
        assert "numeric" in ctx
        assert "Valor_Vendido" in ctx

    def test_includes_virtual_columns(self):
        ctx = build_schema_context(MockAliasMapper())
        assert "YEAR" in ctx
        assert "MONTH" in ctx
        assert "Ano" in ctx

    def test_includes_alias_mappings(self):
        ctx = build_schema_context(MockAliasMapper())
        assert "vendas" in ctx or "Valor_Vendido" in ctx


# ===========================================================================
# _build_intent_from_dict — Parsing Tests
# ===========================================================================


class TestBuildIntentFromDict:
    """Tests for IntentAnalyzer._build_intent_from_dict (internal parsing)."""

    def test_simple_aggregation(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "confidence": 0.95,
            "reasoning": "Soma de vendas",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "simple_aggregation"
        assert len(intent.aggregations) == 1
        assert intent.aggregations[0].function == "sum"
        assert intent.aggregations[0].column.name == "Valor_Vendido"

    def test_grouped_aggregation_with_order_and_limit(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "group_by": [{"name": "UF_Cliente", "alias": "Estado"}],
            "order_by": {"column": "total", "direction": "DESC"},
            "limit": 5,
            "confidence": 0.92,
            "reasoning": "Top 5 estados",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) == 1
        assert intent.group_by[0].name == "UF_Cliente"
        assert intent.order_by is not None
        assert intent.order_by.direction == "DESC"
        assert intent.limit == 5

    def test_virtual_column_in_group_by(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "grouped_aggregation",
            "aggregations": [{"function": "avg", "column": {"name": "Valor_Vendido"}}],
            "group_by": [
                {
                    "name": "Mes",
                    "is_virtual": True,
                    "expression": 'MONTH("Data")',
                    "alias": "Mes",
                }
            ],
            "confidence": 0.90,
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.group_by[0].is_virtual is True
        assert intent.group_by[0].expression == 'MONTH("Data")'

    def test_count_distinct(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "count",
                    "column": {"name": "Cod_Cliente"},
                    "distinct": True,
                    "alias": "n_clientes",
                }
            ],
            "confidence": 0.95,
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.aggregations[0].function == "count"
        assert intent.aggregations[0].distinct is True

    def test_metadata_intent(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "metadata",
            "confidence": 0.95,
            "reasoning": "Estrutura do dataset",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "metadata"
        assert intent.aggregations == []

    def test_conversational_intent(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "conversational",
            "confidence": 0.99,
            "reasoning": "Saudação",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "conversational"

    def test_lookup_intent(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "lookup",
            "confidence": 0.85,
            "reasoning": "Busca registro",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "lookup"

    def test_tabular_intent(self):
        analyzer = _make_analyzer_no_response()
        data = {"intent_type": "tabular", "limit": 50, "confidence": 0.90}
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "tabular"
        assert intent.limit == 50

    def test_invalid_intent_type_defaults_to_simple_aggregation(self):
        analyzer = _make_analyzer_no_response()
        data = {"intent_type": "nonexistent_type", "confidence": 0.50}
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "simple_aggregation"

    def test_invalid_aggregation_function_defaults_to_sum(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {"function": "invalid_fn", "column": {"name": "Valor_Vendido"}}
            ],
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.aggregations[0].function == "sum"

    def test_negative_limit_becomes_none(self):
        analyzer = _make_analyzer_no_response()
        data = {"intent_type": "ranking", "limit": -5}
        intent = analyzer._build_intent_from_dict(data)
        assert intent.limit is None

    def test_invalid_order_direction_defaults_to_desc(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "ranking",
            "order_by": {"column": "total", "direction": "INVALID"},
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.order_by.direction == "DESC"

    def test_confidence_clamped_to_valid_range(self):
        analyzer = _make_analyzer_no_response()
        data = {"intent_type": "metadata", "confidence": 1.5}
        intent = analyzer._build_intent_from_dict(data)
        assert intent.confidence == 1.0

        data2 = {"intent_type": "metadata", "confidence": -0.5}
        intent2 = analyzer._build_intent_from_dict(data2)
        assert intent2.confidence == 0.0

    def test_missing_optional_fields_use_defaults(self):
        analyzer = _make_analyzer_no_response()
        data = {"intent_type": "simple_aggregation"}
        intent = analyzer._build_intent_from_dict(data)
        assert intent.select_columns == []
        assert intent.aggregations == []
        assert intent.group_by == []
        assert intent.order_by is None
        assert intent.limit is None
        assert intent.additional_filters == {}
        assert intent.confidence == 0.8
        assert intent.reasoning == ""

    def test_additional_filters_preserved(self):
        analyzer = _make_analyzer_no_response()
        data = {
            "intent_type": "simple_aggregation",
            "additional_filters": {"Ano": 2016},
            "confidence": 0.90,
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.additional_filters == {"Ano": 2016}


# ===========================================================================
# _enrich_virtual_columns — Enrichment Tests
# ===========================================================================


class TestEnrichVirtualColumns:
    """Tests for virtual column enrichment logic."""

    def test_tagged_virtual_gets_expression(self):
        """A column already tagged as virtual gets its SQL expression filled."""
        analyzer = _make_analyzer_no_response()
        intent = QueryIntent(
            intent_type="temporal_analysis",
            group_by=[ColumnSpec(name="Ano", is_virtual=True)],
            confidence=0.90,
        )
        enriched = analyzer._enrich_virtual_columns(intent)
        assert enriched.group_by[0].expression == 'YEAR("Data")'

    def test_untagged_virtual_detected_and_enriched(self):
        """A column not tagged as virtual but known to mapper gets enriched."""
        analyzer = _make_analyzer_no_response()
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            group_by=[ColumnSpec(name="Mes")],  # NOT tagged as virtual
            aggregations=[
                AggregationSpec(function="sum", column=ColumnSpec(name="Valor_Vendido"))
            ],
            confidence=0.85,
        )
        enriched = analyzer._enrich_virtual_columns(intent)
        assert enriched.group_by[0].is_virtual is True
        assert enriched.group_by[0].expression == 'MONTH("Data")'

    def test_real_column_not_modified(self):
        """Non-virtual columns must NOT be modified by enrichment."""
        analyzer = _make_analyzer_no_response()
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            group_by=[ColumnSpec(name="UF_Cliente")],
            aggregations=[
                AggregationSpec(function="sum", column=ColumnSpec(name="Valor_Vendido"))
            ],
            confidence=0.90,
        )
        enriched = analyzer._enrich_virtual_columns(intent)
        assert enriched.group_by[0].is_virtual is False
        assert enriched.group_by[0].expression is None

    def test_aggregation_column_enrichment(self):
        """Virtual columns in aggregations also get enriched."""
        analyzer = _make_analyzer_no_response()
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(name="Ano"),  # Not tagged as virtual
                )
            ],
            confidence=0.90,
        )
        enriched = analyzer._enrich_virtual_columns(intent)
        assert enriched.aggregations[0].column.is_virtual is True
        assert enriched.aggregations[0].column.expression == 'YEAR("Data")'

    def test_nome_mes_enrichment(self):
        """Nome_Mes virtual column gets correct MONTHNAME expression."""
        analyzer = _make_analyzer_no_response()
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            group_by=[ColumnSpec(name="Nome_Mes")],
            aggregations=[
                AggregationSpec(function="sum", column=ColumnSpec(name="Valor_Vendido"))
            ],
            confidence=0.85,
        )
        enriched = analyzer._enrich_virtual_columns(intent)
        assert enriched.group_by[0].is_virtual is True
        assert enriched.group_by[0].expression == 'MONTHNAME("Data")'


# ===========================================================================
# _parse_response — JSON Parsing Tests
# ===========================================================================


class TestParseResponse:
    """Tests for JSON parsing from LLM responses."""

    def test_clean_json(self):
        analyzer = _make_analyzer_no_response()
        raw = json.dumps(
            {"intent_type": "metadata", "confidence": 0.95, "reasoning": "test"}
        )
        intent = analyzer._parse_response(raw, "quantas linhas?")
        assert intent.intent_type == "metadata"

    def test_json_in_markdown_code_block(self):
        analyzer = _make_analyzer_no_response()
        raw = '```json\n{"intent_type": "conversational", "confidence": 0.98, "reasoning": "oi"}\n```'
        intent = analyzer._parse_response(raw, "olá")
        assert intent.intent_type == "conversational"

    def test_json_with_surrounding_text(self):
        analyzer = _make_analyzer_no_response()
        raw = 'Here is the analysis: {"intent_type": "tabular", "confidence": 0.80} end of analysis'
        intent = analyzer._parse_response(raw, "mostre tabela")
        assert intent.intent_type == "tabular"

    def test_completely_invalid_text_raises_error(self):
        analyzer = _make_analyzer_no_response()
        with pytest.raises(ValueError, match="Could not parse"):
            analyzer._parse_response("this is not json at all", "qualquer coisa")

    def test_partial_json_raises_error(self):
        analyzer = _make_analyzer_no_response()
        with pytest.raises(ValueError):
            analyzer._parse_response('{"intent_type": "meta', "incompleto")


# ===========================================================================
# _create_fallback_intent — Fallback Tests
# ===========================================================================


class TestCreateFallbackIntent:
    """Tests for fallback intent on error."""

    def test_fallback_has_low_confidence(self):
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent("qualquer query", "some error")
        assert intent.confidence == 0.3

    def test_fallback_for_metadata_query(self):
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent(
            "quantas linhas tem o dataset?", "error"
        )
        assert intent.intent_type == "metadata"

    def test_fallback_for_tabular_query(self):
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent("mostre tabela completa", "error")
        assert intent.intent_type == "tabular"

    def test_fallback_default_is_simple_aggregation(self):
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent("alguma coisa genérica", "error")
        assert intent.intent_type == "simple_aggregation"

    def test_fallback_includes_error_in_reasoning(self):
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent("query", "Connection timeout")
        assert "Connection timeout" in intent.reasoning


# ===========================================================================
# analyze() — Full Analysis Tests (Mocked LLM)
# ===========================================================================


class TestAnalyzeSimpleAggregations:
    """Tests for simple aggregation queries via analyze()."""

    def test_total_de_vendas(self):
        response = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "confidence": 0.95,
            "reasoning": "Soma de vendas",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("qual o total de vendas?")
        assert intent.intent_type == "simple_aggregation"
        assert intent.aggregations[0].function == "sum"
        assert intent.aggregations[0].column.name == "Valor_Vendido"

    def test_media_de_valor_vendido(self):
        response = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "avg",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "media",
                }
            ],
            "confidence": 0.93,
            "reasoning": "Média de vendas",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("qual a média de valor vendido?")
        assert intent.aggregations[0].function == "avg"

    def test_quantos_clientes(self):
        response = {
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
            "reasoning": "Contagem distinta de clientes",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("quantos clientes temos?")
        assert intent.aggregations[0].function == "count"
        assert intent.aggregations[0].distinct is True


class TestAnalyzeGroupedAggregations:
    """Tests for grouped aggregation queries via analyze()."""

    def test_vendas_por_estado(self):
        response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "group_by": [{"name": "UF_Cliente", "alias": "Estado"}],
            "confidence": 0.92,
            "reasoning": "Vendas agrupadas por estado",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("vendas por estado")
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) == 1
        assert intent.group_by[0].name == "UF_Cliente"

    def test_vendas_por_mes(self):
        response = {
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
            "reasoning": "Vendas agrupadas por mês",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("vendas por mês em 2016")
        assert intent.intent_type == "grouped_aggregation"
        assert intent.group_by[0].is_virtual is True
        assert intent.group_by[0].expression == 'MONTH("Data")'

    def test_total_por_familia_produto(self):
        response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "group_by": [{"name": "Cod_Familia_Produto", "alias": "Familia"}],
            "confidence": 0.88,
            "reasoning": "Vendas por família de produto",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("total de vendas por família de produto")
        assert len(intent.group_by) == 1


class TestAnalyzeRankings:
    """Tests for ranking queries via analyze()."""

    def test_top_5_estados(self):
        response = {
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
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("top 5 estados por faturamento")
        assert intent.intent_type == "ranking"
        assert intent.limit == 5
        assert intent.order_by.direction == "DESC"

    def test_produto_que_vendeu_mais(self):
        response = {
            "intent_type": "ranking",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "group_by": [{"name": "Des_Linha_Produto", "alias": "Produto"}],
            "order_by": {"column": "total", "direction": "DESC"},
            "limit": 1,
            "confidence": 0.93,
            "reasoning": "Produto com maior total de vendas",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("qual produto vendeu mais?")
        assert intent.limit == 1
        assert intent.order_by.direction == "DESC"


class TestAnalyzeTemporalQueries:
    """Tests for temporal analysis queries via analyze()."""

    def test_ultimo_ano_com_vendas(self):
        response = {
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
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("qual o último ano com vendas?")
        assert intent.intent_type == "temporal_analysis"
        assert intent.aggregations[0].function == "max"
        # Virtual column must be enriched
        assert intent.aggregations[0].column.is_virtual is True
        assert intent.aggregations[0].column.expression == 'YEAR("Data")'

    def test_primeiro_mes_com_vendas(self):
        response = {
            "intent_type": "temporal_analysis",
            "aggregations": [
                {
                    "function": "min",
                    "column": {
                        "name": "Mes",
                        "is_virtual": True,
                        "expression": 'MONTH("Data")',
                    },
                    "alias": "primeiro_mes",
                }
            ],
            "confidence": 0.90,
            "reasoning": "Primeiro mês → MIN(MONTH(Data))",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("qual foi o primeiro mês com vendas?")
        assert intent.aggregations[0].function == "min"
        assert intent.aggregations[0].column.name == "Mes"

    def test_em_que_mes_maior_venda(self):
        """This is the critical Q3 scenario — must produce grouped_aggregation."""
        response = {
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
            "reasoning": "Mês com maior total → GROUP BY MONTH, SUM, ORDER DESC, LIMIT 1",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("em que mês ocorreu a maior venda?")
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) >= 1
        assert intent.order_by is not None
        assert intent.limit == 1


class TestAnalyzeMetadata:
    """Tests for metadata queries."""

    def test_quantas_linhas(self):
        response = {
            "intent_type": "metadata",
            "confidence": 0.98,
            "reasoning": "Número de linhas do dataset",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("quantas linhas tem o dataset?")
        assert intent.intent_type == "metadata"

    def test_quais_colunas(self):
        response = {
            "intent_type": "metadata",
            "confidence": 0.95,
            "reasoning": "Lista de colunas",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("quais são as colunas disponíveis?")
        assert intent.intent_type == "metadata"


class TestAnalyzeConversational:
    """Tests for conversational queries."""

    def test_saudacao(self):
        response = {
            "intent_type": "conversational",
            "confidence": 0.99,
            "reasoning": "Saudação genérica",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("olá, tudo bem?")
        assert intent.intent_type == "conversational"


class TestAnalyzeLookup:
    """Tests for lookup queries."""

    def test_dados_do_cliente(self):
        response = {
            "intent_type": "lookup",
            "confidence": 0.88,
            "reasoning": "Busca de registro específico",
        }
        analyzer = _make_analyzer_with_response(response)
        intent = analyzer.analyze("dados do cliente 12345")
        assert intent.intent_type == "lookup"


# ===========================================================================
# Error Handling
# ===========================================================================


class TestAnalyzeErrorHandling:
    """Tests for graceful error handling."""

    def test_llm_exception_returns_fallback(self):
        """If LLM throws, analyze() returns a safe fallback intent."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API quota exceeded")
        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=MockAliasMapper())

        intent = analyzer.analyze("total de vendas")
        assert intent.confidence == 0.3  # Fallback has low confidence
        assert "API quota exceeded" in intent.reasoning

    def test_llm_returns_nonsense_produces_fallback(self):
        """If LLM returns unparseable text, analyze() returns fallback."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I don't understand what you mean."
        mock_response.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
        }
        mock_llm.invoke.return_value = mock_response

        analyzer = IntentAnalyzer(llm=mock_llm, alias_mapper=MockAliasMapper())
        intent = analyzer.analyze("qualquer query")
        # Should be a fallback, not crash
        assert intent.confidence <= 0.5

    def test_token_accumulator_is_called(self):
        """Token accumulator should receive token counts when provided."""
        response = {
            "intent_type": "metadata",
            "confidence": 0.95,
            "reasoning": "test",
        }
        analyzer = _make_analyzer_with_response(response)

        mock_accumulator = MagicMock()
        mock_accumulator.add = MagicMock()

        analyzer.analyze("quantas linhas?", token_accumulator=mock_accumulator)
        mock_accumulator.add.assert_called_once()


# ===========================================================================
# Prompt Construction
# ===========================================================================


class TestPromptConstruction:
    """Tests for prompt building."""

    def test_prompt_includes_query(self):
        analyzer = _make_analyzer_no_response()
        prompt = analyzer._build_prompt("total de vendas", filters={})
        assert "total de vendas" in prompt

    def test_prompt_includes_schema_context(self):
        analyzer = _make_analyzer_no_response()
        prompt = analyzer._build_prompt("qualquer query", filters={})
        assert "Tipos de Colunas" in prompt or "numeric" in prompt

    def test_prompt_includes_filters_when_present(self):
        analyzer = _make_analyzer_no_response()
        prompt = analyzer._build_prompt("vendas", filters={"Ano": 2016})
        assert "2016" in prompt

    def test_prompt_omits_filters_when_empty(self):
        analyzer = _make_analyzer_no_response()
        prompt = analyzer._build_prompt("vendas", filters={})
        assert "Filtros já aplicados" not in prompt


# ===========================================================================
# JSON Fix Issues
# ===========================================================================


class TestFixJsonIssues:
    """Tests for _fix_json_issues() which handles common LLM JSON problems."""

    def test_fix_unescaped_year_expression(self):
        """LLM may produce unescaped YEAR("Data") inside JSON strings."""
        analyzer = _make_analyzer_no_response()
        broken = '{"expression": "YEAR("Data")", "name": "Ano"}'
        fixed = analyzer._fix_json_issues(broken)
        assert '"expression": null' in fixed

    def test_fix_unescaped_month_expression(self):
        """LLM may produce unescaped MONTH("Data") inside JSON strings."""
        analyzer = _make_analyzer_no_response()
        broken = '{"expression": "MONTH("Data")", "name": "Mes"}'
        fixed = analyzer._fix_json_issues(broken)
        assert '"expression": null' in fixed

    def test_fix_unescaped_monthname_expression(self):
        analyzer = _make_analyzer_no_response()
        broken = '{"expression": "MONTHNAME("Data")", "name": "Nome_Mes"}'
        fixed = analyzer._fix_json_issues(broken)
        assert '"expression": null' in fixed

    def test_fix_trailing_comma(self):
        analyzer = _make_analyzer_no_response()
        broken = '{"a": 1, "b": 2,}'
        fixed = analyzer._fix_json_issues(broken)
        assert fixed == '{"a": 1, "b": 2}'

    def test_valid_json_unchanged(self):
        """Valid JSON should pass through unchanged (except trailing commas)."""
        analyzer = _make_analyzer_no_response()
        valid = '{"intent_type": "simple_aggregation", "confidence": 0.95}'
        fixed = analyzer._fix_json_issues(valid)
        assert fixed == valid

    def test_null_expression_unchanged(self):
        """expression: null should not be modified."""
        analyzer = _make_analyzer_no_response()
        valid = '{"expression": null, "name": "Ano"}'
        fixed = analyzer._fix_json_issues(valid)
        assert '"expression": null' in fixed

    def test_fix_enables_parsing_of_broken_json(self):
        """After fixing, the JSON should be parseable."""
        analyzer = _make_analyzer_no_response()
        # Simulate real LLM response with unescaped expressions
        broken = (
            '{"intent_type": "grouped_aggregation", "aggregations": '
            '[{"function": "sum", "column": {"name": "Valor_Vendido", '
            '"is_virtual": false, "expression": null}, "alias": "total"}], '
            '"group_by": [{"name": "Mes", "is_virtual": true, '
            '"expression": "MONTH("Data")", "alias": "Mes"}], '
            '"confidence": 0.95, "reasoning": "test"}'
        )
        fixed = analyzer._fix_json_issues(broken)
        import json

        parsed = json.loads(fixed)
        assert parsed["intent_type"] == "grouped_aggregation"
        assert len(parsed["group_by"]) == 1


# ===========================================================================
# Improved Fallback Intent
# ===========================================================================


class TestImprovedFallbackIntent:
    """Tests for _create_fallback_intent with grouped query detection."""

    def test_fallback_detects_grouped_query_by_mes(self):
        """'qual mes com maior venda' should produce grouped fallback."""
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent(
            "qual mes com maior valor de venda em 2016",
            "parse error",
        )
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) > 0
        assert intent.group_by[0].name == "Mes"
        assert intent.order_by is not None
        assert intent.limit == 1

    def test_fallback_detects_grouped_query_by_estado(self):
        """'vendas por estado' should produce grouped fallback."""
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent(
            "vendas por estado",
            "parse error",
        )
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) > 0
        assert intent.group_by[0].name == "UF_Cliente"

    def test_fallback_detects_ranking_with_menor(self):
        """'menor' keyword should produce ASC ordering."""
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent(
            "qual estado com menor venda",
            "parse error",
        )
        assert intent.order_by is not None
        assert intent.order_by.direction == "ASC"

    def test_fallback_simple_aggregation_max(self):
        """'último' keyword should produce max aggregation."""
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent(
            "qual o total de vendas",
            "parse error",
        )
        assert intent.intent_type == "simple_aggregation"
        assert len(intent.aggregations) > 0

    def test_fallback_metadata(self):
        """Metadata queries should be detected."""
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent(
            "quantas linhas tem o dataset",
            "parse error",
        )
        assert intent.intent_type == "metadata"

    def test_fallback_has_low_confidence(self):
        """Fallback intent should always have low confidence."""
        analyzer = _make_analyzer_no_response()
        intent = analyzer._create_fallback_intent("qualquer query", "error")
        assert intent.confidence == 0.3

    def test_detect_grouped_intent_patterns(self):
        """Test _detect_grouped_intent with various patterns."""
        analyzer = _make_analyzer_no_response()
        assert analyzer._detect_grouped_intent("vendas por mes")
        assert analyzer._detect_grouped_intent("vendas por estado")
        assert analyzer._detect_grouped_intent("qual mes teve maior venda")
        assert analyzer._detect_grouped_intent("qual estado com maior faturamento")
        assert not analyzer._detect_grouped_intent("total de vendas")
        assert not analyzer._detect_grouped_intent("quantos clientes temos")

    def test_detect_group_dimension(self):
        """Test _detect_group_dimension maps terms to correct columns."""
        analyzer = _make_analyzer_no_response()
        assert analyzer._detect_group_dimension("vendas por mes") == "Mes"
        assert analyzer._detect_group_dimension("vendas por estado") == "UF_Cliente"
        assert analyzer._detect_group_dimension("por ano") == "Ano"
        assert analyzer._detect_group_dimension("total geral") is None


# ===========================================================================
# Parse Response with Strategy 4 (JSON fix)
# ===========================================================================


class TestParseResponseWithJsonFix:
    """Tests for _parse_response Strategy 4: fix broken JSON before parsing."""

    def test_parse_response_with_unescaped_expression(self):
        """Should successfully parse JSON even with unescaped SQL expressions."""
        analyzer = _make_analyzer_no_response()
        broken_content = (
            '{"intent_type": "temporal_analysis", "select_columns": [], '
            '"aggregations": [{"function": "max", "column": {"name": "Ano", '
            '"is_virtual": true, "expression": "YEAR("Data")", "alias": "Ano"}, '
            '"distinct": false, "alias": "ultimo_ano"}], "group_by": [], '
            '"order_by": null, "limit": null, "additional_filters": {}, '
            '"confidence": 0.95, "reasoning": "test"}'
        )
        intent = analyzer._parse_response(broken_content, "qual o ultimo ano?")
        assert intent.intent_type == "temporal_analysis"
        assert len(intent.aggregations) == 1
        assert intent.aggregations[0].function == "max"

    def test_parse_response_with_markdown_wrapping(self):
        """Should handle ```json ... ``` wrapping."""
        analyzer = _make_analyzer_no_response()
        wrapped = (
            "```json\n"
            '{"intent_type": "simple_aggregation", "select_columns": [], '
            '"aggregations": [{"function": "sum", "column": {"name": "Valor_Vendido", '
            '"is_virtual": false, "expression": null}, "distinct": false, '
            '"alias": "total"}], "group_by": [], "order_by": null, '
            '"limit": null, "additional_filters": {}, '
            '"confidence": 0.9, "reasoning": "total de vendas"}\n'
            "```"
        )
        intent = analyzer._parse_response(wrapped, "total de vendas")
        assert intent.intent_type == "simple_aggregation"

    def test_parse_response_with_extra_text(self):
        """Should extract JSON even with extra text around it."""
        analyzer = _make_analyzer_no_response()
        content = (
            "Aqui está a análise:\n\n"
            '{"intent_type": "metadata", "select_columns": [], '
            '"aggregations": [], "group_by": [], "order_by": null, '
            '"limit": null, "additional_filters": {}, '
            '"confidence": 0.9, "reasoning": "metadata"}\n\n'
            "Espero que ajude!"
        )
        intent = analyzer._parse_response(content, "quantas linhas?")
        assert intent.intent_type == "metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
