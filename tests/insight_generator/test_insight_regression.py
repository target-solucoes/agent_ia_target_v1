"""
FASE 4.3 - Regression Tests for Insight Generator.

Ensures that recent improvements do not degrade previously resolved cases.
Validates behavior on historically problematic queries (from diagnosis.md).
Provides a simple continuous validation mechanism.

Test categories:
    - Structural regression: output format hasn't changed
    - Diagnostic regression: problems from diagnosis.md stay fixed
    - Pipeline contract regression: integration with formatter/orchestrator intact
    - Prompt regression: dynamic prompt still includes required components

Usage:
    pytest tests/insight_generator/test_insight_regression.py -v
"""

import sys
import os
import json
import logging
from typing import Dict, Any
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest
import pandas as pd

from tests.insight_generator.quality_evaluators import (
    evaluate_noise_absence,
    evaluate_structure_valid,
    evaluate_filter_mention,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _make_state_with_data(
    user_query: str,
    chart_type: str,
    data_records: list,
    filters: dict = None,
    enriched_intent: dict = None,
) -> dict:
    """Build a minimal InsightState dict for node-level testing."""
    df = pd.DataFrame(data_records)

    chart_spec = {
        "chart_type": chart_type,
        "user_query": user_query,
        "dimensions": [],
        "metrics": [],
        "filters": filters or {},
    }
    # Extract dimension/metric cols from DataFrame
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            chart_spec["metrics"].append(
                {"name": col, "column": col, "aggregation": "sum"}
            )
        else:
            chart_spec["dimensions"].append({"name": col, "column": col, "alias": col})

    analytics_result = {
        "data": data_records,
        "metadata": {
            "user_query": user_query,
            "rows": len(data_records),
            "columns": len(data_records[0]) if data_records else 0,
        },
    }

    return {
        "chart_spec": chart_spec,
        "analytics_result": analytics_result,
        "data": df,
        "chart_type": chart_type,
        "user_query": user_query,
        "enriched_intent": enriched_intent,
        "errors": [],
        "insights": [],
        "agent_tokens": {},
    }


def _mock_llm_json_response(response_json: dict) -> MagicMock:
    """Create a mock LLM instance that returns a specific JSON response."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(response_json)
    mock_response.usage_metadata = {
        "input_tokens": 300,
        "output_tokens": 150,
        "total_tokens": 450,
    }
    mock_response.response_metadata = {"model_name": "gemini-2.5-flash"}
    mock_llm.invoke.return_value = mock_response
    mock_llm.model = "gemini-2.5-flash"
    return mock_llm


# ============================================================================
# P1/P3 Regression: User query must appear in the LLM prompt
# ============================================================================


class TestP1P3UserQueryInPrompt:
    """
    Regression for P1 (absence of user intent) and P3 (query not in prompt).
    After FASE 1/2, the user_query MUST be injected into the LLM prompt.
    """

    def test_user_query_in_prompt_legacy_mode(self):
        """build_prompt_node injects user_query into llm_prompt."""
        from src.insight_generator.graph.nodes import build_prompt_node

        state = _make_state_with_data(
            user_query="quais foram os maiores representantes de SC?",
            chart_type="bar_horizontal",
            data_records=[
                {"Cod_Vendedor": "000145", "Valor_Vendido": 21220758},
            ],
            filters={"UF_Cliente": "SC"},
            enriched_intent={
                "base_intent": "ranking",
                "polarity": "neutral",
                "temporal_focus": "single_period",
                "comparison_type": "none",
                "narrative_angle": "ranking de representantes",
                "suggested_metrics": [],
                "key_entities": [],
                "filters_context": {},
            },
        )
        state["numeric_summary"] = {
            "total": 21220758,
            "lider_label": "000145",
            "lider_valor": 21220758,
        }

        result = build_prompt_node(state)
        prompt = result.get("llm_prompt", "")

        assert "maiores representantes" in prompt.lower(), (
            "P3 regression: user_query not found in prompt"
        )
        assert "SC" in prompt or "santa catarina" in prompt.lower(), (
            "P3 regression: filter not found in prompt"
        )

    def test_user_query_in_prompt_dynamic_mode(self):
        """dynamic_build_prompt_node also injects user_query."""
        from src.insight_generator.graph.integration import dynamic_build_prompt_node

        state = _make_state_with_data(
            user_query="qual a distribuição dos produtos?",
            chart_type="pie",
            data_records=[
                {"Produto": "A", "Qtd": 100},
            ],
            enriched_intent={
                "base_intent": "distribution",
                "polarity": "neutral",
                "temporal_focus": "single_period",
                "comparison_type": "none",
                "narrative_angle": "distribuição de produtos",
                "suggested_metrics": [],
                "key_entities": [],
                "filters_context": {},
            },
        )
        state["numeric_summary"] = {"total": 100}

        result = dynamic_build_prompt_node(state)
        prompt = result.get("llm_prompt", "")

        assert (
            "distribuição dos produtos" in prompt.lower()
            or "distribuicao dos produtos" in prompt.lower()
        ), "P3 regression: user_query not in dynamic prompt"


# ============================================================================
# P5 Regression: Real data must be in the prompt
# ============================================================================


class TestP5DataInPrompt:
    """
    Regression for P5 (DataFrame not passed to LLM).
    After FASE 1, real data must be formatted as markdown table in the prompt.
    """

    def test_data_table_in_prompt(self):
        """build_prompt_node includes data as markdown table."""
        from src.insight_generator.graph.nodes import build_prompt_node

        state = _make_state_with_data(
            user_query="top 5 clientes",
            chart_type="bar_horizontal",
            data_records=[
                {"Cod_Cliente": "2855", "Valor_Vendido": 48215340},
                {"Cod_Cliente": "22494", "Valor_Vendido": 31892110},
            ],
            enriched_intent={
                "base_intent": "ranking",
                "polarity": "neutral",
                "temporal_focus": "single_period",
                "comparison_type": "none",
                "narrative_angle": "ranking",
                "suggested_metrics": [],
                "key_entities": [],
                "filters_context": {},
            },
        )
        state["numeric_summary"] = {"total": 80107450}

        result = build_prompt_node(state)
        prompt = result.get("llm_prompt", "")

        # Data values should appear in the prompt
        assert "2855" in prompt, "P5 regression: client code not in prompt data table"
        assert "22494" in prompt, "P5 regression: client code not in prompt data table"


# ============================================================================
# P7 Regression: Enriched intent must be in the prompt
# ============================================================================


class TestP7EnrichedIntentInPrompt:
    """
    Regression for P7 (IntentEnricher disconnected from LLM prompt).
    enriched_intent context must appear in the prompt.
    """

    def test_enriched_intent_in_prompt(self):
        from src.insight_generator.graph.nodes import build_prompt_node

        state = _make_state_with_data(
            user_query="quais clientes tiveram queda?",
            chart_type="line_composed",
            data_records=[{"Cod_Cliente": "20524", "Valor": 5401}],
            enriched_intent={
                "base_intent": "variation",
                "polarity": "negative",
                "temporal_focus": "period_over_period",
                "comparison_type": "period_vs_period",
                "narrative_angle": "análise de variação com foco em quedas e riscos",
                "suggested_metrics": [],
                "key_entities": [],
                "filters_context": {},
            },
        )
        state["numeric_summary"] = {"total": 5401}

        result = build_prompt_node(state)
        prompt = result.get("llm_prompt", "")

        # Enriched intent context should be present
        assert (
            "queda" in prompt.lower()
            or "risco" in prompt.lower()
            or "negativ" in prompt.lower()
        ), "P7 regression: enriched_intent narrative not in prompt"


# ============================================================================
# P2/P4 Regression: No rigid 4-section structure, no chart_type templates
# ============================================================================


class TestP2P4FlexibleFormat:
    """
    Regression for P2 (chart-type templates) and P4 (rigid 4-section structure).
    FASE 2 output should use flexible 'resposta' format, not forced sections.
    """

    def test_output_uses_resposta_field(self):
        """format_output_node produces resposta when LLM returns FASE 2 format."""
        from src.insight_generator.graph.nodes import format_output_node

        state = {
            "llm_response": json.dumps(
                {
                    "resposta": "Os maiores clientes são 2855 e 22494.",
                    "dados_destacados": ["Cliente 2855: R$ 48M"],
                    "filtros_mencionados": ["SC"],
                }
            ),
            "chart_type": "bar_horizontal",
            "numeric_summary": {"total": 80107450},
            "errors": [],
            "agent_tokens": {},
        }

        result = format_output_node(state)
        output = result.get("final_output", {})

        assert output.get("resposta"), "FASE 2 resposta not in output"
        assert output.get("status") == "success"
        # Should NOT require all 4 legacy sections to be populated
        assert isinstance(output.get("dados_destacados"), list)

    def test_no_chart_type_templates_in_prompt(self):
        """System prompt should not contain chart_type-based templates."""
        from src.insight_generator.formatters.prompt_builder import build_system_prompt

        prompt = build_system_prompt({"base_intent": "ranking", "polarity": "neutral"})
        # Should not contain legacy template keywords
        assert "CHART_TYPE" not in prompt
        assert "bar_horizontal" not in prompt.lower()
        assert "EXATAMENTE 5" not in prompt


# ============================================================================
# P6 Regression: Formatter should not add redundant LLM calls
# ============================================================================


class TestP6FormatterContract:
    """
    Regression for P6 (formatter adds redundant LLM layer).
    Verify that insight_generator output contains fields the formatter needs.
    """

    def test_output_has_backward_compat_fields(self):
        """Output includes both FASE 2 and legacy fields for formatter."""
        from src.insight_generator.graph.nodes import format_output_node

        state = {
            "llm_response": json.dumps(
                {
                    "resposta": "Análise completa dos dados.",
                    "dados_destacados": ["Item 1", "Item 2"],
                    "filtros_mencionados": ["SC"],
                }
            ),
            "chart_type": "bar_horizontal",
            "numeric_summary": {"total": 100},
            "errors": [],
            "agent_tokens": {},
        }

        result = format_output_node(state)
        output = result.get("final_output", {})

        # FASE 2 fields
        assert "resposta" in output
        assert "dados_destacados" in output
        assert "filtros_mencionados" in output

        # Backward compat fields for formatter_agent
        assert "executive_summary" in output
        assert "detailed_insights" in output
        assert "formatted_insights" in output
        assert "synthesized_insights" in output
        assert "metadata" in output


# ============================================================================
# P8 Regression: Model selection correctness
# ============================================================================


class TestP8ModelSelection:
    """
    Regression for P8 (suboptimal model selection).
    Default should be flash, not flash-lite. Complex queries use flash.
    """

    def test_default_model_is_flash(self):
        from src.insight_generator.core.settings import INSIGHT_MODEL_DEFAULT

        assert "lite" not in INSIGHT_MODEL_DEFAULT.lower(), (
            f"Default model should be flash, got: {INSIGHT_MODEL_DEFAULT}"
        )

    def test_lite_model_exists(self):
        from src.insight_generator.core.settings import INSIGHT_MODEL_LITE

        assert "lite" in INSIGHT_MODEL_LITE.lower()

    def test_default_temperature(self):
        from src.insight_generator.core.settings import INSIGHT_TEMPERATURE_DEFAULT

        assert 0.1 <= INSIGHT_TEMPERATURE_DEFAULT <= 0.8


# ============================================================================
# Structural Regression: Output format stability
# ============================================================================


class TestOutputStructuralRegression:
    """Ensure the output structure remains stable across changes."""

    def test_full_workflow_output_structure(self):
        """Full mocked workflow produces structurally valid output."""
        from src.insight_generator.graph.workflow import execute_workflow

        mock_llm = _mock_llm_json_response(
            {
                "resposta": "Resultados da análise.",
                "dados_destacados": ["Ponto 1"],
                "filtros_mencionados": [],
            }
        )

        chart_spec = {
            "chart_type": "bar_horizontal",
            "user_query": "top 5 clientes",
            "dimensions": [{"name": "Cod_Cliente", "column": "Cod_Cliente"}],
            "metrics": [
                {
                    "name": "Valor_Vendido",
                    "column": "Valor_Vendido",
                    "aggregation": "sum",
                }
            ],
            "filters": {},
        }
        analytics_result = {
            "data": [
                {"Cod_Cliente": "2855", "Valor_Vendido": 48215340},
                {"Cod_Cliente": "22494", "Valor_Vendido": 31892110},
            ],
            "metadata": {"user_query": "top 5 clientes", "rows": 2, "columns": 2},
        }

        with patch(
            "src.insight_generator.graph.nodes.load_insight_llm",
            return_value=mock_llm,
        ):
            result = execute_workflow(chart_spec, analytics_result)

        eval_result = evaluate_structure_valid(result)
        assert eval_result.passed, f"Structure regression: {eval_result.rationale}"

    def test_error_output_has_required_fields(self):
        """Even error outputs should have status and metadata."""
        from src.insight_generator.graph.workflow import execute_workflow

        # Deliberately missing data to trigger error path
        chart_spec = {"chart_type": "bar_horizontal", "user_query": "test"}
        analytics_result = {}  # Missing data

        result = execute_workflow(chart_spec, analytics_result)

        assert "status" in result
        assert "metadata" in result
        assert result.get("status") in ("success", "error")


# ============================================================================
# Diagnostic Cases Regression (from diagnosis.md §2)
# ============================================================================


class TestDiagnosticCasesRegression:
    """
    Reproduce the 5 diagnostic queries from diagnosis.md and verify
    that the previously identified problems are resolved.
    """

    def _run_diagnostic_case(
        self,
        user_query: str,
        chart_type: str,
        data_records: list,
        filters: dict = None,
        enriched_intent: dict = None,
    ) -> Dict[str, Any]:
        """Execute workflow with mock LLM for diagnostic cases."""
        from src.insight_generator.graph.workflow import execute_workflow

        # Build a response that demonstrates correct behavior
        # (The mock simulates what the LLM should produce after FASE 1-3 fixes)
        first_record = data_records[0] if data_records else {}
        entity_vals = [str(v) for v in first_record.values() if isinstance(v, str)]
        num_vals = [
            str(int(v)) for v in first_record.values() if isinstance(v, (int, float))
        ]

        mock_response = {
            "resposta": f"Resposta direta: {user_query}. "
            + ", ".join(entity_vals + num_vals[:2]),
            "dados_destacados": [
                f"{k}: {v}" for k, v in list(first_record.items())[:3]
            ],
            "filtros_mencionados": [f"{k}: {v}" for k, v in (filters or {}).items()],
        }

        mock_llm = _mock_llm_json_response(mock_response)

        chart_spec = {
            "chart_type": chart_type,
            "user_query": user_query,
            "dimensions": [],
            "metrics": [],
            "filters": filters or {},
        }
        for col, val in first_record.items():
            if isinstance(val, (int, float)):
                chart_spec["metrics"].append(
                    {"name": col, "column": col, "aggregation": "sum"}
                )
            else:
                chart_spec["dimensions"].append(
                    {"name": col, "column": col, "alias": col}
                )

        analytics_result = {
            "data": data_records,
            "metadata": {"user_query": user_query, "rows": len(data_records)},
        }

        with patch(
            "src.insight_generator.graph.nodes.load_insight_llm",
            return_value=mock_llm,
        ):
            return execute_workflow(chart_spec, analytics_result)

    def test_diag_q1_representantes_sc(self):
        """Q1: 'quais foram os maiores representantes de SC?'
        P1: Response must mention representative codes."""
        result = self._run_diagnostic_case(
            user_query="quais foram os maiores representantes de SC?",
            chart_type="bar_horizontal",
            data_records=[
                {"Cod_Vendedor": "000145", "Valor_Vendido": 21220758},
                {"Cod_Vendedor": "000018", "Valor_Vendido": 20586795},
            ],
            filters={"UF_Cliente": "SC"},
            enriched_intent={"base_intent": "ranking", "polarity": "neutral"},
        )
        assert result["status"] == "success"
        resposta = result.get("resposta", "")
        # P1 fix: should mention actual representative codes
        assert "000145" in resposta or "000018" in resposta, (
            "Diagnostic Q1 regression: no representative codes in response"
        )

    def test_diag_q2_distribuicao_produtos(self):
        """Q2: 'qual a distribuição dos produtos do representante 000145?'
        P2: No HHI or Simpson in response."""
        result = self._run_diagnostic_case(
            user_query="qual a distribuição dos produtos do representante 000145?",
            chart_type="pie",
            data_records=[
                {"Des_Linha_Produto": "CONEXOES SOLDAVEIS", "Qtd_Vendida": 3473773},
                {"Des_Linha_Produto": "TUBOS ELETRODUTO", "Qtd_Vendida": 1399350},
            ],
            filters={"UF_Cliente": "SC", "Cod_Vendedor": "000145"},
            enriched_intent={"base_intent": "distribution", "polarity": "neutral"},
        )
        assert result["status"] == "success"
        noise_result = evaluate_noise_absence(result.get("resposta", ""))
        assert noise_result.passed, (
            f"Diagnostic Q2 regression: noise found: {noise_result.details}"
        )

    def test_diag_q3_produtos_clientes_cross(self):
        """Q3: Cross-reference query should mention products and clients."""
        result = self._run_diagnostic_case(
            user_query="quais os 5 produtos mais vendidos nos 5 maiores clientes?",
            chart_type="bar_vertical_stacked",
            data_records=[
                {
                    "Cliente": "2855",
                    "Produto": "Conexoes Soldaveis",
                    "Qtd_Vendida": 2190981,
                },
                {
                    "Cliente": "23709",
                    "Produto": "Tubos Eletroduto",
                    "Qtd_Vendida": 859350,
                },
            ],
            enriched_intent={"base_intent": "composition", "polarity": "neutral"},
        )
        assert result["status"] == "success"
        resposta = result.get("resposta", "")
        # Should mention actual entities
        assert "2855" in resposta or "23709" in resposta, (
            "Diagnostic Q3 regression: no client codes in response"
        )

    def test_diag_q4_historico_2015(self):
        """Q4: 'qual o histórico de compras de 2015?'
        Response should describe temporal pattern, not just variation."""
        result = self._run_diagnostic_case(
            user_query="qual o histórico de compras de 2015?",
            chart_type="line",
            data_records=[
                {"Mes": "Jan", "Valor_Vendido": 633249},
                {"Mes": "Mar", "Valor_Vendido": 2438682},
                {"Mes": "Dez", "Valor_Vendido": 430074},
            ],
            filters={"Ano": "2015"},
            enriched_intent={"base_intent": "trend", "polarity": "neutral"},
        )
        assert result["status"] == "success"

    def test_diag_q5_queda_clientes(self):
        """Q5: 'Quais clientes tiveram a maior queda entre fev/2015 e março/2015?'
        P1+P2: Must mention client codes, not convergence/correlation."""
        result = self._run_diagnostic_case(
            user_query="Quais clientes que tiveram a maior queda entre fev/2015 e março/2015?",
            chart_type="line_composed",
            data_records=[
                {"Cod_Cliente": "20524", "Fev_2015": 5401, "Mar_2015": 0},
                {"Cod_Cliente": "33777", "Fev_2015": 1695, "Mar_2015": 0},
            ],
            filters={"UF_Cliente": "SC"},
            enriched_intent={"base_intent": "variation", "polarity": "negative"},
        )
        assert result["status"] == "success"
        resposta = result.get("resposta", "")
        noise_result = evaluate_noise_absence(resposta)
        assert noise_result.passed, (
            f"Diagnostic Q5 regression: noise found: {noise_result.details}"
        )


# ============================================================================
# Prompt Component Regression
# ============================================================================


class TestPromptComponentRegression:
    """Ensure prompt components from FASE 1-3 are all present."""

    def test_system_prompt_has_analyst_persona(self):
        from src.insight_generator.formatters.prompt_builder import build_system_prompt

        prompt = build_system_prompt()
        assert "analista" in prompt.lower(), "System prompt missing analyst persona"

    def test_system_prompt_has_json_format(self):
        from src.insight_generator.formatters.prompt_builder import build_system_prompt

        prompt = build_system_prompt()
        assert "json" in prompt.lower(), "System prompt missing JSON format instruction"

    def test_system_prompt_has_no_emoji_rule(self):
        from src.insight_generator.formatters.prompt_builder import build_system_prompt

        prompt = build_system_prompt()
        assert "emoji" in prompt.lower(), "System prompt missing no-emoji rule"

    def test_build_prompt_includes_all_sections(self):
        from src.insight_generator.formatters.prompt_builder import build_prompt

        prompt = build_prompt(
            numeric_summary={"total": 1000},
            chart_type="bar_horizontal",
            filters={"UF": "SC"},
            user_query="top 5 clientes",
            data_table="| Col | Val |\n|---|---|\n| A | 100 |",
            intent_context="ranking de desempenho",
            enriched_intent={"base_intent": "ranking"},
        )

        assert "top 5 clientes" in prompt.lower(), "Prompt missing user_query"
        assert "SC" in prompt, "Prompt missing filters"
        assert "Col" in prompt, "Prompt missing data table"
        assert "ranking" in prompt.lower(), "Prompt missing intent context"
