"""
Phase 2 validation tests for non_graph_executor — Intent Comprehension Engine.

Tests the three subfases:
2.1 - IntentAnalyzer (LLM-based semantic intent analysis)
2.2 - QueryIntent schema (Pydantic models for structured intent)
2.3 - QueryClassifier hybrid refactoring (pre-filters + IntentAnalyzer + legacy)

These tests use mocks for the LLM to run offline without API keys.
"""

import sys
import os
import json
import logging
from typing import Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# =========================================================================
# Phase 2.2 — QueryIntent Schema Tests
# =========================================================================

from src.non_graph_executor.models.intent_schema import (
    ColumnSpec,
    AggregationSpec,
    OrderSpec,
    QueryIntent,
)
from src.non_graph_executor.models.schemas import QueryTypeClassification


class TestColumnSpec:
    """Tests for ColumnSpec model."""

    def test_basic_column(self):
        col = ColumnSpec(name="Valor_Vendido")
        assert col.name == "Valor_Vendido"
        assert col.is_virtual is False
        assert col.expression is None
        assert col.alias is None

    def test_virtual_column(self):
        col = ColumnSpec(
            name="Ano",
            is_virtual=True,
            expression='YEAR("Data")',
            alias="Ano",
        )
        assert col.name == "Ano"
        assert col.is_virtual is True
        assert col.expression == 'YEAR("Data")'
        assert col.alias == "Ano"

    def test_column_serialization(self):
        col = ColumnSpec(name="UF_Cliente", alias="Estado")
        data = col.model_dump()
        assert data["name"] == "UF_Cliente"
        assert data["alias"] == "Estado"
        assert data["is_virtual"] is False


class TestAggregationSpec:
    """Tests for AggregationSpec model."""

    def test_sum_aggregation(self):
        agg = AggregationSpec(
            function="sum",
            column=ColumnSpec(name="Valor_Vendido"),
            alias="total_vendas",
        )
        assert agg.function == "sum"
        assert agg.column.name == "Valor_Vendido"
        assert agg.distinct is False
        assert agg.alias == "total_vendas"

    def test_count_distinct(self):
        agg = AggregationSpec(
            function="count",
            column=ColumnSpec(name="UF_Cliente"),
            distinct=True,
            alias="total_estados",
        )
        assert agg.function == "count"
        assert agg.distinct is True

    def test_all_aggregation_functions(self):
        """Validate all supported aggregation functions."""
        for func in ["sum", "avg", "count", "min", "max", "median", "std"]:
            agg = AggregationSpec(
                function=func,
                column=ColumnSpec(name="Valor_Vendido"),
            )
            assert agg.function == func

    def test_invalid_aggregation_function(self):
        """Invalid function should raise ValidationError."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AggregationSpec(
                function="invalid_function",
                column=ColumnSpec(name="Valor_Vendido"),
            )


class TestOrderSpec:
    """Tests for OrderSpec model."""

    def test_default_desc(self):
        order = OrderSpec(column="total_vendas")
        assert order.column == "total_vendas"
        assert order.direction == "DESC"

    def test_asc_direction(self):
        order = OrderSpec(column="Ano", direction="ASC")
        assert order.direction == "ASC"


class TestQueryIntent:
    """Tests for QueryIntent model."""

    def test_simple_aggregation(self):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                )
            ],
            confidence=0.95,
            reasoning="Soma simples de vendas",
        )
        assert intent.intent_type == "simple_aggregation"
        assert len(intent.aggregations) == 1
        assert intent.aggregations[0].function == "sum"
        assert intent.group_by == []
        assert intent.order_by is None
        assert intent.limit is None

    def test_grouped_aggregation(self):
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
                    name="Mes",
                    is_virtual=True,
                    expression='MONTH("Data")',
                    alias="Mes",
                )
            ],
            confidence=0.90,
        )
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) == 1
        assert intent.group_by[0].is_virtual is True
        assert intent.group_by[0].expression == 'MONTH("Data")'

    def test_ranking(self):
        intent = QueryIntent(
            intent_type="ranking",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente")],
            order_by=OrderSpec(column="total_vendas", direction="DESC"),
            limit=5,
            confidence=0.92,
        )
        assert intent.intent_type == "ranking"
        assert intent.limit == 5
        assert intent.order_by.direction == "DESC"

    def test_temporal_analysis(self):
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano",
                        is_virtual=True,
                        expression='YEAR("Data")',
                    ),
                )
            ],
            confidence=0.95,
            reasoning="Último ano com vendas",
        )
        assert intent.intent_type == "temporal_analysis"
        assert intent.aggregations[0].column.is_virtual is True

    def test_all_intent_types(self):
        """Validate all supported intent types."""
        valid_types = [
            "simple_aggregation",
            "grouped_aggregation",
            "ranking",
            "temporal_analysis",
            "comparison",
            "lookup",
            "metadata",
            "tabular",
            "conversational",
        ]
        for intent_type in valid_types:
            intent = QueryIntent(intent_type=intent_type, confidence=0.80)
            assert intent.intent_type == intent_type

    def test_invalid_intent_type(self):
        with pytest.raises(Exception):
            QueryIntent(intent_type="invalid_type", confidence=0.80)

    def test_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            QueryIntent(intent_type="metadata", confidence=1.5)
        with pytest.raises(Exception):
            QueryIntent(intent_type="metadata", confidence=-0.1)

    def test_default_values(self):
        intent = QueryIntent(intent_type="metadata")
        assert intent.select_columns == []
        assert intent.aggregations == []
        assert intent.group_by == []
        assert intent.order_by is None
        assert intent.limit is None
        assert intent.additional_filters == {}
        assert intent.confidence == 0.8  # default
        assert intent.reasoning == ""

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="avg",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="media_vendas",
                )
            ],
            group_by=[ColumnSpec(name="UF_Cliente", alias="Estado")],
            order_by=OrderSpec(column="media_vendas", direction="DESC"),
            limit=10,
            confidence=0.88,
            reasoning="Média de vendas por estado, top 10",
        )
        json_str = intent.model_dump_json()
        restored = QueryIntent.model_validate_json(json_str)
        assert restored.intent_type == intent.intent_type
        assert len(restored.aggregations) == 1
        assert restored.group_by[0].alias == "Estado"
        assert restored.limit == 10


# =========================================================================
# Phase 2.2 — QueryTypeClassification with intent field
# =========================================================================


class TestQueryTypeClassificationIntent:
    """Tests for the new 'intent' field on QueryTypeClassification."""

    def test_intent_default_none(self):
        cls = QueryTypeClassification(
            query_type="aggregation",
            confidence=0.85,
            requires_llm=True,
            parameters={},
        )
        assert cls.intent is None

    def test_intent_attached(self):
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                )
            ],
            confidence=0.95,
        )
        cls = QueryTypeClassification(
            query_type="aggregation",
            confidence=0.95,
            requires_llm=True,
            parameters={"column": "Valor_Vendido", "aggregation": "sum"},
            intent=intent,
        )
        assert cls.intent is not None
        assert cls.intent.intent_type == "simple_aggregation"

    def test_intent_excluded_from_serialization(self):
        """The intent field must be excluded from model_dump / JSON."""
        intent = QueryIntent(
            intent_type="ranking",
            confidence=0.90,
        )
        cls = QueryTypeClassification(
            query_type="aggregation",
            confidence=0.90,
            requires_llm=True,
            parameters={},
            intent=intent,
        )
        data = cls.model_dump()
        assert "intent" not in data

        json_str = cls.model_dump_json()
        parsed = json.loads(json_str)
        assert "intent" not in parsed


# =========================================================================
# Phase 2.1 — IntentAnalyzer Tests (with mocked LLM)
# =========================================================================

from src.non_graph_executor.tools.intent_analyzer import IntentAnalyzer


class MockAliasMapper:
    """Minimal mock replicating AliasMapper behavior for unit tests."""

    VIRTUAL_COLUMN_MAP = {
        "Ano": 'YEAR("Data")',
        "Mes": 'MONTH("Data")',
        "Nome_Mes": 'MONTHNAME("Data")',
    }

    column_types = {
        "numeric": ["Valor_Vendido", "Peso_Vendido", "Qtd_Vendida"],
        "categorical": ["UF_Cliente", "Des_Linha_Produto", "Empresa"],
        "temporal": ["Data"],
    }

    alias_map = {
        "vendas": "Valor_Vendido",
        "venda": "Valor_Vendido",
        "faturamento": "Valor_Vendido",
        "ano": "Ano",
        "mes": "Mes",
        "mês": "Mes",
        "estado": "UF_Cliente",
    }

    def resolve(self, term):
        return self.alias_map.get(term.lower())

    @classmethod
    def is_virtual_column(cls, col_name):
        return col_name in cls.VIRTUAL_COLUMN_MAP

    @classmethod
    def get_virtual_expression(cls, col_name):
        return cls.VIRTUAL_COLUMN_MAP.get(col_name)


class TestIntentAnalyzerParsing:
    """Tests for IntentAnalyzer internal parsing (no LLM calls)."""

    def _make_analyzer(self):
        """Create an IntentAnalyzer with mocked LLM."""
        mock_llm = MagicMock()
        analyzer = IntentAnalyzer(
            llm=mock_llm,
            alias_mapper=MockAliasMapper(),
        )
        return analyzer

    def test_build_intent_from_dict_simple_aggregation(self):
        """Test building QueryIntent from a parsed dict."""
        analyzer = self._make_analyzer()
        data = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "confidence": 0.95,
            "reasoning": "Soma total de vendas",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "simple_aggregation"
        assert len(intent.aggregations) == 1
        assert intent.aggregations[0].function == "sum"
        assert intent.aggregations[0].column.name == "Valor_Vendido"
        assert intent.confidence == 0.95

    def test_build_intent_from_dict_grouped_aggregation(self):
        analyzer = self._make_analyzer()
        data = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total_vendas",
                }
            ],
            "group_by": [{"name": "UF_Cliente", "alias": "Estado"}],
            "order_by": {"column": "total_vendas", "direction": "DESC"},
            "limit": 5,
            "confidence": 0.92,
            "reasoning": "Top 5 estados por vendas",
        }
        intent = analyzer._build_intent_from_dict(data)
        assert intent.intent_type == "grouped_aggregation"
        assert len(intent.group_by) == 1
        assert intent.group_by[0].name == "UF_Cliente"
        assert intent.order_by is not None
        assert intent.order_by.direction == "DESC"
        assert intent.limit == 5

    def test_build_intent_from_dict_with_virtual_column(self):
        """Virtual columns should be enriched with expressions."""
        analyzer = self._make_analyzer()
        data = {
            "intent_type": "temporal_analysis",
            "aggregations": [
                {
                    "function": "max",
                    "column": {"name": "Ano", "is_virtual": True},
                }
            ],
            "confidence": 0.90,
        }
        intent = analyzer._build_intent_from_dict(data)
        # _enrich_virtual_columns should fill in the expression
        assert intent.aggregations[0].column.name == "Ano"
        assert intent.aggregations[0].column.is_virtual is True

    def test_enrich_virtual_columns(self):
        """Test virtual column enrichment post-processing."""
        analyzer = self._make_analyzer()
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            group_by=[
                ColumnSpec(name="Ano", is_virtual=True),
                ColumnSpec(name="UF_Cliente"),
            ],
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                )
            ],
            confidence=0.90,
        )
        enriched = analyzer._enrich_virtual_columns(intent)

        # Ano should get expression filled in
        ano_col = enriched.group_by[0]
        assert ano_col.name == "Ano"
        assert ano_col.is_virtual is True
        assert ano_col.expression == 'YEAR("Data")'

        # UF_Cliente should remain unchanged
        uf_col = enriched.group_by[1]
        assert uf_col.is_virtual is False

    def test_enrich_virtual_columns_detects_untagged_virtuals(self):
        """Columns that are virtual but not tagged should be detected."""
        analyzer = self._make_analyzer()
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            group_by=[
                ColumnSpec(name="Mes"),  # Not tagged as virtual
            ],
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                )
            ],
            confidence=0.85,
        )
        enriched = analyzer._enrich_virtual_columns(intent)

        mes_col = enriched.group_by[0]
        assert mes_col.name == "Mes"
        assert mes_col.is_virtual is True
        assert mes_col.expression == 'MONTH("Data")'

    def test_create_fallback_intent(self):
        """Fallback intent should be safe defaults."""
        analyzer = self._make_analyzer()
        intent = analyzer._create_fallback_intent("alguma query aqui", "test error")
        assert intent.intent_type == "simple_aggregation"
        assert intent.confidence == 0.3
        assert intent.reasoning != ""

    def test_parse_response_valid_json(self):
        """Test parsing a clean JSON response."""
        analyzer = self._make_analyzer()
        valid_json = json.dumps(
            {
                "intent_type": "metadata",
                "confidence": 0.95,
                "reasoning": "Pergunta sobre estrutura",
            }
        )
        intent = analyzer._parse_response(valid_json, "quantas linhas?")
        assert intent.intent_type == "metadata"
        assert intent.confidence == 0.95

    def test_parse_response_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        analyzer = self._make_analyzer()
        response = '```json\n{"intent_type": "conversational", "confidence": 0.98, "reasoning": "Saudação"}\n```'
        intent = analyzer._parse_response(response, "olá")
        assert intent.intent_type == "conversational"

    def test_parse_response_invalid_json_raises_error(self):
        """Invalid JSON should raise ValueError (caller handles fallback)."""
        analyzer = self._make_analyzer()
        with pytest.raises(ValueError, match="Could not parse LLM response"):
            analyzer._parse_response("this is not json at all", "alguma query")


class TestIntentAnalyzerAnalyze:
    """Tests for IntentAnalyzer.analyze() with mocked LLM."""

    def _make_analyzer_with_response(self, response_json: dict):
        """Create analyzer with LLM mocked to return specific JSON."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(response_json)

        # For token tracking
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_llm.invoke.return_value = mock_response

        analyzer = IntentAnalyzer(
            llm=mock_llm,
            alias_mapper=MockAliasMapper(),
        )
        return analyzer

    def test_analyze_simple_aggregation(self):
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
        analyzer = self._make_analyzer_with_response(response)
        intent = analyzer.analyze(query="qual o total de vendas?", filters={})
        assert intent.intent_type == "simple_aggregation"
        assert intent.aggregations[0].function == "sum"
        assert intent.aggregations[0].column.name == "Valor_Vendido"

    def test_analyze_grouped_with_virtual_column(self):
        """Virtual columns should be enriched after LLM response."""
        response = {
            "intent_type": "grouped_aggregation",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "group_by": [{"name": "Ano", "is_virtual": True}],
            "confidence": 0.90,
            "reasoning": "Vendas por ano",
        }
        analyzer = self._make_analyzer_with_response(response)
        intent = analyzer.analyze(query="vendas por ano", filters={})
        assert intent.intent_type == "grouped_aggregation"
        # Virtual column should have expression filled
        assert intent.group_by[0].expression == 'YEAR("Data")'

    def test_analyze_llm_exception_returns_fallback(self):
        """If LLM call throws, should return fallback intent."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API quota exceeded")
        analyzer = IntentAnalyzer(
            llm=mock_llm,
            alias_mapper=MockAliasMapper(),
        )
        intent = analyzer.analyze(query="total de vendas", filters={})
        assert intent.intent_type == "simple_aggregation"
        assert intent.confidence == 0.3


# =========================================================================
# Phase 2.3 — QueryClassifier Hybrid Flow Tests
# =========================================================================

from src.non_graph_executor.tools.query_classifier import QueryClassifier


class TestQueryClassifierPreFilters:
    """Tests for pre-filter classification (no LLM/IntentAnalyzer needed)."""

    def _make_classifier(self, use_intent_analyzer=False):
        """Create classifier with mocked dependencies and NO IntentAnalyzer."""
        mock_llm = MagicMock()
        return QueryClassifier(
            alias_mapper=MockAliasMapper(),
            llm=mock_llm,
            intent_analyzer=None,
            use_intent_analyzer=use_intent_analyzer,
        )

    def test_conversational_greeting(self):
        classifier = self._make_classifier()
        state = {}
        result = classifier.classify("olá", state)
        assert result.query_type == "conversational"
        assert result.confidence >= 0.90

    def test_conversational_bom_dia(self):
        classifier = self._make_classifier()
        state = {}
        result = classifier.classify("bom dia", state)
        assert result.query_type == "conversational"

    def test_tabular_mostrar_tabela(self):
        classifier = self._make_classifier()
        state = {}
        result = classifier.classify("mostrar tabela", state)
        assert result.query_type == "tabular"
        assert result.confidence >= 0.90

    def test_tabular_with_limit(self):
        classifier = self._make_classifier()
        state = {}
        result = classifier.classify("mostre 50 registros", state)
        # Could be tabular or metadata sample_rows
        assert result.query_type in ("tabular", "metadata")

    def test_sample_rows(self):
        classifier = self._make_classifier()
        state = {}
        result = classifier.classify("mostre 5 linhas", state)
        # Pre-filter should catch "mostre N linhas" as metadata/sample_rows
        assert result.query_type in ("metadata", "tabular")


class TestQueryClassifierHybridFlow:
    """Tests for the hybrid IntentAnalyzer → legacy flow."""

    def _make_classifier_with_intent_response(self, response_json: dict):
        """Create classifier with IntentAnalyzer mocked to return specific response."""
        mock_llm = MagicMock()

        # Mock IntentAnalyzer's LLM
        mock_ia_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(response_json)
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        mock_ia_llm.invoke.return_value = mock_response

        intent_analyzer = IntentAnalyzer(
            llm=mock_ia_llm,
            alias_mapper=MockAliasMapper(),
        )

        classifier = QueryClassifier(
            alias_mapper=MockAliasMapper(),
            llm=mock_llm,
            intent_analyzer=intent_analyzer,
            use_intent_analyzer=True,
        )
        return classifier

    def test_intent_simple_aggregation_maps_to_aggregation(self):
        """IntentAnalyzer simple_aggregation → query_type=aggregation."""
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
        classifier = self._make_classifier_with_intent_response(response)
        result = classifier.classify("qual o total de vendas?", {})
        assert result.query_type == "aggregation"
        assert result.confidence == 0.95
        assert result.intent is not None
        assert result.intent.intent_type == "simple_aggregation"
        assert result.parameters.get("column") == "Valor_Vendido"
        assert result.parameters.get("aggregation") == "sum"

    def test_intent_grouped_aggregation_maps_to_aggregation(self):
        """IntentAnalyzer grouped_aggregation → query_type=aggregation."""
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
            "order_by": {"column": "total", "direction": "DESC"},
            "limit": 5,
            "confidence": 0.92,
            "reasoning": "Top 5 estados por vendas",
        }
        classifier = self._make_classifier_with_intent_response(response)
        result = classifier.classify("top 5 estados por vendas", {})
        assert result.query_type == "aggregation"
        assert result.parameters.get("group_by") is not None
        assert len(result.parameters["group_by"]) == 1
        assert result.parameters["group_by"][0]["name"] == "UF_Cliente"
        assert result.parameters.get("order_by") is not None
        assert result.parameters["order_by"]["direction"] == "DESC"
        assert result.parameters.get("limit") == 5

    def test_intent_ranking_maps_to_aggregation(self):
        """IntentAnalyzer ranking → query_type=aggregation."""
        response = {
            "intent_type": "ranking",
            "aggregations": [
                {
                    "function": "sum",
                    "column": {"name": "Valor_Vendido"},
                    "alias": "total",
                }
            ],
            "group_by": [{"name": "Empresa"}],
            "order_by": {"column": "total", "direction": "DESC"},
            "limit": 3,
            "confidence": 0.93,
            "reasoning": "Top 3 empresas",
        }
        classifier = self._make_classifier_with_intent_response(response)
        result = classifier.classify("top 3 empresas por vendas", {})
        assert result.query_type == "aggregation"
        assert result.intent.intent_type == "ranking"

    def test_intent_temporal_maps_to_aggregation(self):
        """IntentAnalyzer temporal_analysis → query_type=aggregation."""
        response = {
            "intent_type": "temporal_analysis",
            "aggregations": [
                {
                    "function": "max",
                    "column": {"name": "Ano", "is_virtual": True},
                }
            ],
            "confidence": 0.95,
            "reasoning": "Último ano",
        }
        classifier = self._make_classifier_with_intent_response(response)
        result = classifier.classify("qual o último ano com vendas", {})
        assert result.query_type == "aggregation"
        assert result.intent.intent_type == "temporal_analysis"

    def test_intent_metadata_maps_to_metadata(self):
        """IntentAnalyzer metadata → query_type=metadata."""
        response = {
            "intent_type": "metadata",
            "confidence": 0.95,
            "reasoning": "Pergunta sobre estrutura do dataset",
        }
        classifier = self._make_classifier_with_intent_response(response)
        result = classifier.classify("quantas linhas tem no dataset?", {})
        # Pre-filter should catch this before IntentAnalyzer
        # But if IntentAnalyzer catches it, should map correctly
        assert result.query_type == "metadata"

    def test_intent_lookup_maps_to_lookup(self):
        """IntentAnalyzer lookup → query_type=lookup."""
        response = {
            "intent_type": "lookup",
            "confidence": 0.88,
            "reasoning": "Busca de registro específico",
        }
        classifier = self._make_classifier_with_intent_response(response)
        # Use a query that won't be caught by pre-filters
        result = classifier.classify("detalhes do pedido 12345", {})
        assert result.query_type == "lookup"
        assert result.intent is not None

    def test_intent_tabular_maps_to_tabular(self):
        """IntentAnalyzer tabular → query_type=tabular."""
        response = {
            "intent_type": "tabular",
            "limit": 50,
            "confidence": 0.90,
            "reasoning": "Dados brutos",
        }
        classifier = self._make_classifier_with_intent_response(response)
        # Note: "mostrar tabela" would be caught by pre-filter.
        # This test validates the mapping path even though pre-filter usually catches it.
        # We use a query that wouldn't be caught by pre-filters.
        result = classifier.classify("exiba dados sem agregação", {})
        assert result.query_type == "tabular"
        assert result.parameters.get("limit") == 50

    def test_intent_conversational_maps_correctly(self):
        """IntentAnalyzer conversational → query_type=conversational."""
        response = {
            "intent_type": "conversational",
            "confidence": 0.99,
            "reasoning": "Saudação",
        }
        classifier = self._make_classifier_with_intent_response(response)
        # Note: simple greetings are caught by pre-filter.
        # This tests the mapping for ambiguous cases.
        result = classifier.classify("como você pode me ajudar?", {})
        assert result.query_type == "conversational"

    def test_intent_excluded_from_json(self):
        """When serializing, intent should not appear."""
        response = {
            "intent_type": "simple_aggregation",
            "aggregations": [
                {
                    "function": "avg",
                    "column": {"name": "Valor_Vendido"},
                }
            ],
            "confidence": 0.90,
            "reasoning": "Média de vendas",
        }
        classifier = self._make_classifier_with_intent_response(response)
        result = classifier.classify("média de vendas", {})
        # Intent exists in memory
        assert result.intent is not None
        # But not in serialization
        data = result.model_dump()
        assert "intent" not in data


class TestQueryClassifierLegacyFallback:
    """Tests for legacy fallback when IntentAnalyzer is disabled."""

    def _make_legacy_classifier(self):
        """Create classifier with IntentAnalyzer disabled."""
        mock_llm = MagicMock()
        # Mock LLM response for _llm_classify fallback
        mock_response = MagicMock()
        mock_response.content = "aggregation"
        mock_response.usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
        }
        mock_llm.invoke.return_value = mock_response

        return QueryClassifier(
            alias_mapper=MockAliasMapper(),
            llm=mock_llm,
            intent_analyzer=None,
            use_intent_analyzer=False,
        )

    def test_legacy_metadata_detection(self):
        classifier = self._make_legacy_classifier()
        result = classifier.classify("quantas linhas tem?", {})
        assert result.query_type == "metadata"
        assert result.intent is None  # No intent in legacy mode

    def test_legacy_aggregation_detection(self):
        classifier = self._make_legacy_classifier()
        result = classifier.classify("qual a soma de vendas?", {})
        assert result.query_type == "aggregation"
        assert result.intent is None

    def test_legacy_conversational_preserved(self):
        """Pre-filter conversational works regardless of mode."""
        classifier = self._make_legacy_classifier()
        result = classifier.classify("olá, tudo bem?", {})
        assert result.query_type == "conversational"

    def test_legacy_tabular_preserved(self):
        """Pre-filter tabular works regardless of mode."""
        classifier = self._make_legacy_classifier()
        result = classifier.classify("mostrar tabela", {})
        assert result.query_type == "tabular"


class TestQueryClassifierIntentAnalyzerFailure:
    """Tests for graceful fallback when IntentAnalyzer fails."""

    def test_fallback_on_llm_error(self):
        """If IntentAnalyzer throws, should fall back to legacy."""
        mock_llm = MagicMock()
        # Legacy LLM response
        mock_response = MagicMock()
        mock_response.content = "aggregation"
        mock_response.usage_metadata = {
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
        }
        mock_llm.invoke.return_value = mock_response

        # IntentAnalyzer that always fails
        mock_ia = MagicMock(spec=IntentAnalyzer)
        mock_ia.analyze.side_effect = Exception("LLM connection error")

        classifier = QueryClassifier(
            alias_mapper=MockAliasMapper(),
            llm=mock_llm,
            intent_analyzer=mock_ia,
            use_intent_analyzer=True,
        )

        result = classifier.classify("soma de vendas", {})
        # Should fall back to legacy and still classify correctly
        assert result.query_type == "aggregation"
        assert result.intent is None  # Legacy doesn't produce intent


# =========================================================================
# Phase 2.3 — Parameter Extraction from Intent Tests
# =========================================================================


class TestExtractParamsFromIntent:
    """Tests for _extract_params_from_intent method."""

    def _make_classifier(self):
        mock_llm = MagicMock()
        return QueryClassifier(
            alias_mapper=MockAliasMapper(),
            llm=mock_llm,
            use_intent_analyzer=False,
        )

    def test_extract_aggregation_params(self):
        classifier = self._make_classifier()
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                    alias="total_vendas",
                )
            ],
            confidence=0.95,
        )
        params = classifier._extract_params_from_intent(intent, "total vendas", {})
        assert params["column"] == "Valor_Vendido"
        assert params["aggregation"] == "sum"
        assert params["distinct"] is False

    def test_extract_virtual_column_params(self):
        classifier = self._make_classifier()
        intent = QueryIntent(
            intent_type="temporal_analysis",
            aggregations=[
                AggregationSpec(
                    function="max",
                    column=ColumnSpec(
                        name="Ano",
                        is_virtual=True,
                        expression='YEAR("Data")',
                    ),
                )
            ],
            confidence=0.95,
        )
        params = classifier._extract_params_from_intent(intent, "último ano", {})
        assert params["column"] == "Ano"
        assert params["column_is_virtual"] is True
        assert params["column_expression"] == 'YEAR("Data")'

    def test_extract_group_by_params(self):
        classifier = self._make_classifier()
        intent = QueryIntent(
            intent_type="grouped_aggregation",
            aggregations=[
                AggregationSpec(
                    function="avg",
                    column=ColumnSpec(name="Valor_Vendido"),
                )
            ],
            group_by=[
                ColumnSpec(name="UF_Cliente", alias="Estado"),
                ColumnSpec(name="Ano", is_virtual=True, expression='YEAR("Data")'),
            ],
            confidence=0.90,
        )
        params = classifier._extract_params_from_intent(
            intent, "média por estado e ano", {}
        )
        assert "group_by" in params
        assert len(params["group_by"]) == 2
        assert params["group_by"][0]["name"] == "UF_Cliente"
        assert params["group_by"][1]["is_virtual"] is True

    def test_extract_order_and_limit_params(self):
        classifier = self._make_classifier()
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
            limit=10,
            confidence=0.92,
        )
        params = classifier._extract_params_from_intent(intent, "top 10 empresas", {})
        assert params["order_by"]["direction"] == "DESC"
        assert params["limit"] == 10

    def test_extract_filters_from_state(self):
        classifier = self._make_classifier()
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="sum",
                    column=ColumnSpec(name="Valor_Vendido"),
                )
            ],
            confidence=0.95,
        )
        state = {"filter_final": {"Ano": 2024, "UF_Cliente": "SP"}}
        params = classifier._extract_params_from_intent(intent, "vendas", state)
        assert params["filters"] == {"Ano": 2024, "UF_Cliente": "SP"}

    def test_extract_distinct_param(self):
        classifier = self._make_classifier()
        intent = QueryIntent(
            intent_type="simple_aggregation",
            aggregations=[
                AggregationSpec(
                    function="count",
                    column=ColumnSpec(name="UF_Cliente"),
                    distinct=True,
                )
            ],
            confidence=0.90,
        )
        params = classifier._extract_params_from_intent(intent, "quantos estados", {})
        assert params["distinct"] is True
        assert params["aggregation"] == "count"


# =========================================================================
# Integration — Models Export Validation
# =========================================================================


class TestModelsExport:
    """Test that models/__init__.py exports new schemas correctly."""

    def test_import_from_models_package(self):
        from src.non_graph_executor.models import (
            QueryIntent,
            ColumnSpec,
            AggregationSpec,
            OrderSpec,
        )

        assert QueryIntent is not None
        assert ColumnSpec is not None
        assert AggregationSpec is not None
        assert OrderSpec is not None

    def test_import_from_tools_package(self):
        from src.non_graph_executor.tools import IntentAnalyzer

        assert IntentAnalyzer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
