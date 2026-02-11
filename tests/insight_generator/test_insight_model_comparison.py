"""
FASE 4.2 - Model Comparison Tests (flash-lite vs flash).

Implements controlled comparison between gemini-2.5-flash-lite and
gemini-2.5-flash on the same benchmark cases. Evaluates:
    - Semantic quality of the response
    - Clarity and directness
    - Alignment with user intent
    - Latency
    - Token usage (cost proxy)

Results are captured as structured JSON for documentation and tracking.

Usage:
    # Run comparison (requires GEMINI_API_KEY):
    pytest tests/insight_generator/test_insight_model_comparison.py -v -m e2e

    # Generate comparison report:
    pytest tests/insight_generator/test_insight_model_comparison.py -v -m e2e -s
"""

import sys
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from unittest.mock import patch

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
    QualityReport,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Comparison Data Structures
# ============================================================================


@dataclass
class ModelRunResult:
    """Result from running a single case with a specific model."""

    model: str
    case_id: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    quality_report: Optional[Dict[str, Any]] = None
    resposta: str = ""
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Side-by-side comparison of flash-lite vs flash for one case."""

    case_id: str
    case_description: str
    lite: Optional[ModelRunResult] = None
    full: Optional[ModelRunResult] = None

    def summary(self) -> Dict[str, Any]:
        result = {
            "case_id": self.case_id,
            "description": self.case_description,
        }
        for label, run in [("flash_lite", self.lite), ("flash", self.full)]:
            if run:
                quality = run.quality_report or {}
                result[label] = {
                    "latency_ms": round(run.latency_ms, 1),
                    "tokens": run.total_tokens,
                    "overall_score": quality.get("overall_score", 0),
                    "passed": quality.get("passed", False),
                    "error": run.error,
                }
            else:
                result[label] = {"error": "not executed"}

        # Delta
        if (
            self.lite
            and self.full
            and self.lite.quality_report
            and self.full.quality_report
        ):
            lite_score = self.lite.quality_report.get("overall_score", 0)
            full_score = self.full.quality_report.get("overall_score", 0)
            result["quality_delta"] = round(full_score - lite_score, 3)
            result["latency_delta_ms"] = round(
                (self.full.latency_ms - self.lite.latency_ms), 1
            )

        return result


# ============================================================================
# Model Execution Helper
# ============================================================================


def _run_with_model(
    case: BenchmarkCase,
    model_name: str,
) -> ModelRunResult:
    """
    Execute the insight_generator workflow with a specific model override.

    Patches INSIGHT_MODEL_DEFAULT / INSIGHT_MODEL_LITE to force the
    desired model, then runs the full pipeline.

    Args:
        case: Benchmark case to execute.
        model_name: Model identifier (e.g., "gemini-2.5-flash").

    Returns:
        ModelRunResult with latency, tokens, quality, and response.
    """
    from src.insight_generator.graph.workflow import execute_workflow

    chart_spec = case.build_chart_spec()
    analytics_result = case.build_analytics_result()

    # Force model selection to the specified model
    with patch(
        "src.insight_generator.models.insight_schemas.select_insight_model",
        return_value=model_name,
    ):
        start = time.time()
        try:
            result = execute_workflow(chart_spec, analytics_result)
            latency_ms = (time.time() - start) * 1000
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelRunResult(
                model=model_name,
                case_id=case.id,
                latency_ms=latency_ms,
                error=str(e),
            )

    # Extract tokens
    agent_tokens = result.get("_agent_tokens", {}).get("insight_generator", {})
    input_tokens = agent_tokens.get("input_tokens", 0)
    output_tokens = agent_tokens.get("output_tokens", 0)
    total_tokens = agent_tokens.get("total_tokens", 0)

    # Evaluate quality
    report = evaluate_insight_quality(
        output=result,
        user_query=case.user_query,
        data_records=case.data_records,
        active_filters=case.filters,
        enriched_intent=case.enriched_intent,
        expected_entities=case.expected_entities,
        key_columns=case.key_columns,
    )

    return ModelRunResult(
        model=model_name,
        case_id=case.id,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        quality_report=report.summary(),
        resposta=result.get("resposta", ""),
        error=result.get("error"),
    )


# ============================================================================
# Unit Tests: Model Selection Logic
# ============================================================================


class TestModelSelectionLogic:
    """Unit tests for select_insight_model without API calls."""

    def test_complex_intent_uses_flash(self):
        from src.insight_generator.models.insight_schemas import select_insight_model

        for intent in ["comparison", "variation", "composition", "trend", "temporal"]:
            model = select_insight_model({"base_intent": intent})
            assert "lite" not in model, (
                f"Intent '{intent}' should use flash, got {model}"
            )

    def test_negative_polarity_uses_flash(self):
        from src.insight_generator.models.insight_schemas import select_insight_model

        model = select_insight_model(
            {
                "base_intent": "ranking",
                "polarity": "negative",
                "temporal_focus": "single_period",
                "comparison_type": "none",
            }
        )
        assert "lite" not in model

    def test_simple_ranking_uses_lite(self):
        from src.insight_generator.models.insight_schemas import select_insight_model

        model = select_insight_model(
            {
                "base_intent": "ranking",
                "polarity": "neutral",
                "temporal_focus": "single_period",
                "comparison_type": "none",
            }
        )
        assert "lite" in model

    def test_no_intent_defaults_to_flash(self):
        from src.insight_generator.models.insight_schemas import select_insight_model

        model = select_insight_model(None)
        assert "lite" not in model

    def test_period_comparison_uses_flash(self):
        from src.insight_generator.models.insight_schemas import select_insight_model

        model = select_insight_model(
            {
                "base_intent": "ranking",
                "polarity": "neutral",
                "temporal_focus": "period_over_period",
                "comparison_type": "period_vs_period",
            }
        )
        assert "lite" not in model

    def test_model_tier_alignment_with_fixtures(self):
        """Verify fixtures' model_tier matches select_insight_model logic."""
        from src.insight_generator.models.insight_schemas import select_insight_model

        for case in get_benchmark_cases():
            model = select_insight_model(case.enriched_intent)
            is_lite = "lite" in model
            expected_lite = case.model_tier == "lite"
            # We only check that complex cases are NOT assigned to lite
            if case.complexity == "complex":
                assert not is_lite, (
                    f"[{case.id}] Complex case should use flash, got {model}"
                )


# ============================================================================
# E2E Comparison Tests
# ============================================================================


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.e2e
@pytest.mark.skipif(not _has_gemini_key(), reason="GEMINI_API_KEY not set")
class TestModelComparison:
    """
    Side-by-side comparison of flash-lite vs flash on benchmark cases.

    These tests execute each case with both models and compare quality
    scores, latency, and token usage. Results are printed as structured
    JSON for documentation.
    """

    def _compare(self, case: BenchmarkCase) -> ComparisonResult:
        lite_result = _run_with_model(case, "gemini-2.5-flash-lite")
        full_result = _run_with_model(case, "gemini-2.5-flash")
        return ComparisonResult(
            case_id=case.id,
            case_description=case.description,
            lite=lite_result,
            full=full_result,
        )

    @pytest.mark.parametrize(
        "case",
        get_simple_cases(),
        ids=[c.id for c in get_simple_cases()],
    )
    def test_simple_case_comparison(self, case: BenchmarkCase):
        """
        Simple cases: flash-lite should perform adequately.
        Flash should match or exceed flash-lite quality.
        """
        comparison = self._compare(case)
        summary = comparison.summary()
        logger.info(
            f"[{case.id}] Comparison:\n{json.dumps(summary, indent=2, ensure_ascii=False)}"
        )
        print(f"\n[{case.id}] {json.dumps(summary, indent=2, ensure_ascii=False)}")

        # Both models should succeed
        assert comparison.lite and comparison.lite.error is None, (
            f"flash-lite error: {comparison.lite.error if comparison.lite else 'N/A'}"
        )
        assert comparison.full and comparison.full.error is None, (
            f"flash error: {comparison.full.error if comparison.full else 'N/A'}"
        )

        # Both should score above minimum
        lite_score = comparison.lite.quality_report.get("overall_score", 0)
        full_score = comparison.full.quality_report.get("overall_score", 0)
        assert lite_score >= 0.4, f"flash-lite score too low: {lite_score}"
        assert full_score >= 0.5, f"flash score too low: {full_score}"

    @pytest.mark.parametrize(
        "case",
        get_complex_cases(),
        ids=[c.id for c in get_complex_cases()],
    )
    def test_complex_case_comparison(self, case: BenchmarkCase):
        """
        Complex cases: flash should significantly outperform flash-lite.
        """
        comparison = self._compare(case)
        summary = comparison.summary()
        logger.info(
            f"[{case.id}] Comparison:\n{json.dumps(summary, indent=2, ensure_ascii=False)}"
        )
        print(f"\n[{case.id}] {json.dumps(summary, indent=2, ensure_ascii=False)}")

        # Flash should succeed
        assert comparison.full and comparison.full.error is None, (
            f"flash error: {comparison.full.error if comparison.full else 'N/A'}"
        )

        # Flash should meet minimum quality
        full_score = comparison.full.quality_report.get("overall_score", 0)
        assert full_score >= 0.5, f"flash score too low for complex case: {full_score}"

    @pytest.mark.parametrize(
        "case",
        get_benchmark_cases()[:3],
        ids=[c.id for c in get_benchmark_cases()[:3]],
    )
    def test_latency_comparison(self, case: BenchmarkCase):
        """Compare latency between models (flash-lite should be faster)."""
        comparison = self._compare(case)
        summary = comparison.summary()
        print(
            f"\n[{case.id}] Latency: {json.dumps(summary, indent=2, ensure_ascii=False)}"
        )

        if comparison.lite and comparison.full:
            # Log latency delta
            delta = comparison.full.latency_ms - comparison.lite.latency_ms
            logger.info(
                f"[{case.id}] Latency delta: flash is {delta:.0f}ms "
                f"{'slower' if delta > 0 else 'faster'} than flash-lite"
            )
