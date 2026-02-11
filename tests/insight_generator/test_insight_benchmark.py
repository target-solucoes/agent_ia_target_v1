"""
FASE 4.1 - Benchmark Test Suite for Insight Generator.

Structured benchmark covering multiple intents, chart types, complexities,
and filter conditions. Each test exercises the full insight_generator workflow
(parse_input → build_prompt → invoke_llm → format_output) with mocked or
real LLM calls, then evaluates quality via the FASE 4.4 evaluators.

Test categories:
    - Unit: deterministic evaluator logic (always run, no API key)
    - Integration: full workflow with mocked LLM (no API key)
    - E2E: full workflow with real Gemini API (requires GEMINI_API_KEY,
            skipped when unavailable, tagged with @pytest.mark.e2e)

Usage:
    # Unit + integration (offline):
    pytest tests/insight_generator/test_insight_benchmark.py -v

    # E2E (requires GEMINI_API_KEY):
    pytest tests/insight_generator/test_insight_benchmark.py -v -m e2e
"""

import sys
import os
import json
import logging
import re
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pytest

from tests.insight_generator.benchmark_fixtures import (
    BenchmarkCase,
    get_benchmark_cases,
    get_simple_cases,
    get_complex_cases,
)
from tests.insight_generator.quality_evaluators import (
    evaluate_insight_quality,
    evaluate_intent_adherence,
    evaluate_data_coherence,
    evaluate_filter_mention,
    evaluate_format_adequacy,
    evaluate_noise_absence,
    evaluate_structure_valid,
    evaluate_conciseness,
    QualityReport,
    EvalResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _has_gemini_key() -> bool:
    """Check whether GEMINI_API_KEY is configured."""
    return bool(os.getenv("GEMINI_API_KEY"))


def _build_mock_llm_response(case: BenchmarkCase) -> str:
    """
    Build a realistic mock LLM JSON response for a benchmark case.

    This simulates what a well-functioning insight_generator should produce,
    allowing the evaluators to be tested independently of the real LLM.
    """
    # Build a realistic response text from data
    data = case.data_records
    query = case.user_query

    lines = [f"Análise para: {query}"]

    if case.filters:
        filter_parts = [f"{k}: {v}" for k, v in case.filters.items()]
        lines.append(f"Filtros: {', '.join(filter_parts)}")

    # Add data mentions
    dados_destacados = []
    for i, record in enumerate(data[:5]):
        parts = []
        for col, val in record.items():
            if isinstance(val, (int, float)):
                parts.append(f"{col} = {val:,.0f}")
            else:
                parts.append(f"{col}: {val}")
        line = " | ".join(parts)
        lines.append(f"- {line}")
        if i < 4:
            dados_destacados.append(line)

    resposta_text = "\n".join(lines)
    filtros_mencionados = [f"{k}: {v}" for k, v in case.filters.items()]

    return json.dumps(
        {
            "resposta": resposta_text,
            "dados_destacados": dados_destacados,
            "filtros_mencionados": filtros_mencionados,
        }
    )


def _build_well_formed_output(case: BenchmarkCase) -> Dict[str, Any]:
    """Build a well-formed insight_generator output dict for evaluator testing."""
    mock_response = json.loads(_build_mock_llm_response(case))
    return {
        "status": "success",
        "chart_type": case.chart_type,
        "resposta": mock_response["resposta"],
        "dados_destacados": mock_response["dados_destacados"],
        "filtros_mencionados": mock_response["filtros_mencionados"],
        "metadata": {
            "llm_model": "gemini-2.5-flash",
            "timestamp": "2026-02-11T00:00:00",
            "transparency_validated": True,
            "pipeline_version": "fase_3",
            "calculation_time": 0.0,
            "metrics_count": 5,
        },
        "executive_summary": {},
        "detailed_insights": [],
        "formatted_insights": mock_response["resposta"],
        "synthesized_insights": {"narrative": "", "key_findings": []},
        "next_steps": [],
    }


def _run_workflow_with_mock_llm(case: BenchmarkCase) -> Dict[str, Any]:
    """
    Execute the full insight_generator workflow with a mocked LLM.

    Patches load_insight_llm to return a mock that produces realistic JSON.
    This tests the pipeline integration without requiring an API key.
    """
    from src.insight_generator.graph.workflow import execute_workflow

    mock_response_json = _build_mock_llm_response(case)

    # Create mock LLM that returns the expected JSON
    mock_llm = MagicMock()
    mock_response_obj = MagicMock()
    mock_response_obj.content = mock_response_json
    mock_response_obj.usage_metadata = {
        "input_tokens": 500,
        "output_tokens": 200,
        "total_tokens": 700,
    }
    mock_response_obj.response_metadata = {
        "model_name": "gemini-2.5-flash",
    }
    mock_llm.invoke.return_value = mock_response_obj
    # Set model attribute for token tracking
    mock_llm.model = "gemini-2.5-flash"

    chart_spec = case.build_chart_spec()
    analytics_result = case.build_analytics_result()

    with patch(
        "src.insight_generator.graph.nodes.load_insight_llm",
        return_value=mock_llm,
    ):
        result = execute_workflow(chart_spec, analytics_result)

    return result


# ============================================================================
# Unit Tests: Evaluator Correctness
# ============================================================================


class TestEvaluatorUnit:
    """Unit tests for individual quality evaluators (no API key needed)."""

    def test_intent_adherence_good_response(self):
        result = evaluate_intent_adherence(
            resposta="Os 5 maiores clientes por faturamento são: Cliente 2855 com R$ 48,2M",
            user_query="Top 5 clientes por faturamento",
            enriched_intent={"base_intent": "ranking"},
            expected_entities=["2855"],
        )
        assert result.score >= 0.7
        assert result.criterion == "intent_adherence"

    def test_intent_adherence_bad_response(self):
        result = evaluate_intent_adherence(
            resposta="O spread das series reduziu 19.22%, indicando convergencia.",
            user_query="Quais clientes tiveram a maior queda?",
            enriched_intent={"base_intent": "variation"},
            expected_entities=["20524", "33777"],
        )
        assert result.score < 0.7

    def test_intent_adherence_empty(self):
        result = evaluate_intent_adherence("", "qualquer pergunta")
        assert result.score == 0.0
        assert not result.passed

    def test_data_coherence_values_present(self):
        result = evaluate_data_coherence(
            resposta="Cliente 2855 lidera com R$ 48.215.340 em vendas.",
            data_records=[{"Cod_Cliente": "2855", "Valor_Vendido": 48215340}],
            key_columns=["Cod_Cliente", "Valor_Vendido"],
        )
        assert result.score >= 0.5

    def test_data_coherence_values_absent(self):
        result = evaluate_data_coherence(
            resposta="A análise mostra tendência de convergencia.",
            data_records=[{"Cod_Cliente": "2855", "Valor_Vendido": 48215340}],
            key_columns=["Cod_Cliente"],
        )
        assert result.score < 0.5

    def test_filter_mention_correct(self):
        result = evaluate_filter_mention(
            resposta="Os maiores representantes em Santa Catarina são...",
            active_filters={"UF_Cliente": "SC"},
        )
        assert result.passed

    def test_filter_mention_missing(self):
        result = evaluate_filter_mention(
            resposta="O total de vendas é de R$ 100M.",
            active_filters={"UF_Cliente": "SC"},
        )
        assert not result.passed

    def test_filter_mention_no_filters(self):
        result = evaluate_filter_mention("Qualquer resposta", {})
        assert result.passed
        assert result.score == 1.0

    def test_noise_absence_clean(self):
        result = evaluate_noise_absence(
            "Os 5 maiores clientes são 2855 (R$ 48M), 22494 (R$ 32M)."
        )
        assert result.passed
        assert result.score == 1.0

    def test_noise_absence_with_hhi(self):
        result = evaluate_noise_absence(
            "Índice Herfindahl-Hirschman: 1.602 (moderada concentração)."
        )
        assert not result.passed

    def test_noise_absence_with_simpson(self):
        result = evaluate_noise_absence(
            "Diversidade = 85.89; Equilíbrio = 0.00 (Simpson)."
        )
        assert not result.passed

    def test_structure_valid_good(self):
        output = {
            "status": "success",
            "chart_type": "bar_horizontal",
            "resposta": "Dados analisados.",
            "dados_destacados": ["Item 1"],
            "filtros_mencionados": [],
            "metadata": {
                "llm_model": "gemini-2.5-flash",
                "timestamp": "2026-02-11T00:00:00",
            },
        }
        result = evaluate_structure_valid(output)
        assert result.passed

    def test_structure_valid_missing_fields(self):
        output = {"status": "error"}
        result = evaluate_structure_valid(output)
        assert not result.passed

    def test_conciseness_good(self):
        result = evaluate_conciseness("Uma resposta curta e direta com valores.")
        assert result.passed

    def test_conciseness_too_long(self):
        long_text = " ".join(["palavra"] * 800)
        result = evaluate_conciseness(long_text)
        assert result.score < 0.8

    def test_conciseness_filler_sections(self):
        result = evaluate_conciseness(
            "Resposta normal.\n\nRecomendações:\n- Fazer X\n- Fazer Y\n\n"
            "Próximos Passos:\n- Passo 1"
        )
        assert result.score < 1.0

    def test_format_adequacy_ranking(self):
        result = evaluate_format_adequacy(
            "1. **Cliente A** - R$ 48M\n2. **Cliente B** - R$ 32M",
            enriched_intent={"base_intent": "ranking"},
        )
        assert result.score >= 0.8

    def test_format_adequacy_distribution(self):
        result = evaluate_format_adequacy(
            "| Produto | Participação |\n|---|---|\n| A | 31,5% |",
            enriched_intent={"base_intent": "distribution"},
        )
        assert result.score >= 0.8


class TestQualityReportComposite:
    """Test the composite QualityReport aggregation."""

    def test_all_criteria_attached(self):
        cases = get_benchmark_cases()
        case = cases[0]  # S01 ranking simple
        output = _build_well_formed_output(case)
        report = evaluate_insight_quality(
            output=output,
            user_query=case.user_query,
            data_records=case.data_records,
            active_filters=case.filters,
            enriched_intent=case.enriched_intent,
            expected_entities=case.expected_entities,
            key_columns=case.key_columns,
        )
        assert len(report.results) == 7
        criteria_names = {r.criterion for r in report.results}
        assert "intent_adherence" in criteria_names
        assert "data_coherence" in criteria_names
        assert "noise_absence" in criteria_names
        assert "structure_valid" in criteria_names
        assert 0.0 <= report.overall_score <= 1.0

    def test_summary_has_all_criteria(self):
        case = get_benchmark_cases()[0]
        output = _build_well_formed_output(case)
        report = evaluate_insight_quality(
            output=output,
            user_query=case.user_query,
            data_records=case.data_records,
        )
        summary = report.summary()
        assert "overall_score" in summary
        assert "criteria" in summary
        assert len(summary["criteria"]) == 7


# ============================================================================
# Integration Tests: Full Workflow with Mocked LLM
# ============================================================================


class TestBenchmarkIntegration:
    """Integration tests using the full workflow with mocked LLM."""

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases(),
        ids=[c.id for c in get_benchmark_cases()],
    )
    def test_workflow_produces_valid_output(self, case: BenchmarkCase):
        """Each benchmark case should produce a structurally valid output."""
        result = _run_workflow_with_mock_llm(case)

        assert result is not None, f"Workflow returned None for {case.id}"
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result.get("status") == "success", (
            f"[{case.id}] status={result.get('status')}, error={result.get('error')}"
        )

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases(),
        ids=[c.id for c in get_benchmark_cases()],
    )
    def test_workflow_output_structure(self, case: BenchmarkCase):
        """Output must have all required FASE 2 fields."""
        result = _run_workflow_with_mock_llm(case)
        eval_result = evaluate_structure_valid(result)
        assert eval_result.passed, (
            f"[{case.id}] Structure invalid: {eval_result.rationale}"
        )

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases(),
        ids=[c.id for c in get_benchmark_cases()],
    )
    def test_workflow_has_resposta(self, case: BenchmarkCase):
        """FASE 2: Output must contain a non-empty 'resposta' field."""
        result = _run_workflow_with_mock_llm(case)
        resposta = result.get("resposta", "")
        assert resposta, f"[{case.id}] Empty resposta"
        assert len(resposta) > 10, f"[{case.id}] resposta too short: {len(resposta)}"

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases(),
        ids=[c.id for c in get_benchmark_cases()],
    )
    def test_workflow_noise_free(self, case: BenchmarkCase):
        """Response should not contain academic/irrelevant noise metrics."""
        result = _run_workflow_with_mock_llm(case)
        resposta = result.get("resposta", "")
        eval_result = evaluate_noise_absence(resposta)
        assert eval_result.passed, (
            f"[{case.id}] Noise found: {eval_result.details.get('noise_keywords_found')}"
        )


# ============================================================================
# E2E Tests: Real Gemini API (skipped without API key)
# ============================================================================


@pytest.mark.e2e
@pytest.mark.skipif(not _has_gemini_key(), reason="GEMINI_API_KEY not set")
class TestBenchmarkE2E:
    """
    End-to-end benchmark tests with real Gemini API calls.

    These tests run the full insight_generator pipeline against the real
    LLM and evaluate quality using the FASE 4.4 evaluators.
    Requires GEMINI_API_KEY to be set in the environment.

    Run with: pytest -m e2e -v
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up environment for E2E tests."""
        # Ensure dynamic prompt mode
        os.environ.setdefault("INSIGHT_PROMPT_MODE", "legacy")

    def _run_e2e(self, case: BenchmarkCase) -> Dict[str, Any]:
        """Execute full workflow with real LLM for a benchmark case."""
        from src.insight_generator.graph.workflow import execute_workflow

        chart_spec = case.build_chart_spec()
        analytics_result = case.build_analytics_result()
        return execute_workflow(chart_spec, analytics_result)

    @pytest.mark.parametrize(
        "case",
        get_simple_cases(),
        ids=[c.id for c in get_simple_cases()],
    )
    def test_e2e_simple_cases(self, case: BenchmarkCase):
        """Simple cases should pass all quality criteria with real LLM."""
        result = self._run_e2e(case)
        assert result.get("status") == "success", f"Error: {result.get('error')}"

        report = evaluate_insight_quality(
            output=result,
            user_query=case.user_query,
            data_records=case.data_records,
            active_filters=case.filters,
            enriched_intent=case.enriched_intent,
            expected_entities=case.expected_entities,
            key_columns=case.key_columns,
        )
        summary = report.summary()
        logger.info(f"[{case.id}] E2E quality: {json.dumps(summary, indent=2)}")

        # Minimum overall score for simple cases
        assert report.overall_score >= 0.6, (
            f"[{case.id}] Quality too low: {report.overall_score:.2f}\n"
            f"Details: {json.dumps(summary, indent=2)}"
        )

    @pytest.mark.parametrize(
        "case",
        get_complex_cases(),
        ids=[c.id for c in get_complex_cases()],
    )
    def test_e2e_complex_cases(self, case: BenchmarkCase):
        """Complex cases evaluated with real LLM (quality report logged)."""
        result = self._run_e2e(case)
        assert result.get("status") == "success", f"Error: {result.get('error')}"

        report = evaluate_insight_quality(
            output=result,
            user_query=case.user_query,
            data_records=case.data_records,
            active_filters=case.filters,
            enriched_intent=case.enriched_intent,
            expected_entities=case.expected_entities,
            key_columns=case.key_columns,
        )
        summary = report.summary()
        logger.info(f"[{case.id}] E2E quality: {json.dumps(summary, indent=2)}")

        # Minimum overall score for complex cases (relaxed threshold)
        assert report.overall_score >= 0.5, (
            f"[{case.id}] Quality too low: {report.overall_score:.2f}\n"
            f"Details: {json.dumps(summary, indent=2)}"
        )


# ============================================================================
# Parametrized Evaluator Tests Across All Benchmark Cases
# ============================================================================


class TestEvaluatorsAgainstBenchmark:
    """Test evaluators produce reasonable scores across all benchmark cases."""

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases(),
        ids=[c.id for c in get_benchmark_cases()],
    )
    def test_well_formed_output_passes_structure(self, case: BenchmarkCase):
        """A well-formed output should always pass structure validation."""
        output = _build_well_formed_output(case)
        result = evaluate_structure_valid(output)
        assert result.passed, f"[{case.id}] {result.rationale}"

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases(),
        ids=[c.id for c in get_benchmark_cases()],
    )
    def test_well_formed_output_quality_report(self, case: BenchmarkCase):
        """Well-formed outputs should have overall score > 0.5."""
        output = _build_well_formed_output(case)
        report = evaluate_insight_quality(
            output=output,
            user_query=case.user_query,
            data_records=case.data_records,
            active_filters=case.filters,
            enriched_intent=case.enriched_intent,
            expected_entities=case.expected_entities,
            key_columns=case.key_columns,
        )
        assert report.overall_score >= 0.5, (
            f"[{case.id}] Score too low: {report.overall_score:.2f}\n"
            f"{json.dumps(report.summary(), indent=2)}"
        )
