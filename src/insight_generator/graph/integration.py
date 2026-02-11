"""
Integration module for DynamicPromptBuilder into insight_generator pipeline.

FASE 3 - Integration Guide

This module provides integration helpers and examples for using the
DynamicPromptBuilder in the existing LangGraph workflow.

Key Integration Points:
    1. parse_input_node: Add enriched_intent to state
    2. calculate_metrics_node: Use MetricComposer output
    3. build_prompt_node: Replace with dynamic_build_prompt_node
    4. invoke_llm_node: Use dynamic prompt

Usage:
    # In graph/nodes.py, replace build_prompt_node with:
    from .integration import dynamic_build_prompt_node

    # In graph/graph.py, update workflow:
    workflow.add_node("build_prompt", dynamic_build_prompt_node)
"""

from typing import Dict, Any
import logging

from ..core.intent_enricher import (
    IntentEnricher,
    EnrichedIntent,
    Polarity,
    TemporalFocus,
    ComparisonType,
)
from ..formatters.dynamic_prompt_builder import build_dynamic_prompt
from .state import InsightState

logger = logging.getLogger(__name__)


# ============================================================================
# Integration Node - Replacement for build_prompt_node
# ============================================================================


def dynamic_build_prompt_node(state: InsightState) -> InsightState:
    """
    Node 3 (Alternative): Build LLM prompt dynamically using intent + metrics.

    This node REPLACES the current build_prompt_node, using DynamicPromptBuilder
    instead of chart-type-based templates.

    Required State Fields:
        - enriched_intent: EnrichedIntent (from FASE 1)
        - numeric_summary: Dict (metrics from FASE 2 - MetricComposer)
        - chart_spec: Dict (optional, from graphic_classifier)
        - analytics_metadata: Dict (optional, from analytics_executor)

    Updated State Fields:
        - llm_prompt: str (dynamic prompt)

    Args:
        state: Current workflow state

    Returns:
        Updated state with dynamic llm_prompt

    Raises:
        Adds errors to state if prompt building fails
    """
    logger.info("[dynamic_build_prompt_node] Starting dynamic prompt building")

    try:
        # Validate required fields
        if "enriched_intent" not in state:
            raise ValueError("Missing required field: enriched_intent (FASE 1 output)")
        if "numeric_summary" not in state:
            raise ValueError("Missing required field: numeric_summary (FASE 2 output)")

        enriched_intent = state["enriched_intent"]
        composed_metrics = state["numeric_summary"]

        # Compatibility: current pipeline stores enriched_intent as dict
        if isinstance(enriched_intent, dict):
            enriched_intent = EnrichedIntent(
                base_intent=enriched_intent.get("base_intent", "ranking"),
                polarity=Polarity(enriched_intent.get("polarity", "neutral")),
                temporal_focus=TemporalFocus(
                    enriched_intent.get("temporal_focus", "single_period")
                ),
                comparison_type=ComparisonType(
                    enriched_intent.get("comparison_type", "none")
                ),
                suggested_metrics=enriched_intent.get("suggested_metrics", []),
                key_entities=enriched_intent.get("key_entities", []),
                filters_context=enriched_intent.get("filters_context", {}),
                narrative_angle=enriched_intent.get("narrative_angle", ""),
            )

        # Optional fields
        chart_spec = state.get("chart_spec")
        analytics_metadata = state.get("analytics_metadata")
        if analytics_metadata is None:
            analytics_metadata = state.get("analytics_result", {}).get("metadata")

        # FASE 1: Extract user_query for dynamic prompt
        user_query = ""
        if chart_spec:
            user_query = chart_spec.get("user_query", "")
        if not user_query:
            user_query = (
                state.get("analytics_result", {})
                .get("metadata", {})
                .get("user_query", "")
            )

        # Build dynamic prompt
        llm_prompt = build_dynamic_prompt(
            enriched_intent=enriched_intent,
            composed_metrics=composed_metrics,
            chart_spec=chart_spec,
            analytics_metadata=analytics_metadata,
        )

        # FASE 1: Prepend user_query and data table to dynamic prompt
        prefix_sections = []

        if user_query:
            prefix_sections.append(
                f'PERGUNTA DO USUARIO:\n"{user_query}"\n\n'
                f"Sua resposta DEVE responder diretamente a esta pergunta."
            )

        # Include real data if available
        import pandas as pd

        df = state.get("data")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            from .nodes import _format_dataframe_as_markdown

            data_table = _format_dataframe_as_markdown(df, max_rows=20)
            if data_table:
                prefix_sections.append(
                    f"DADOS DISPONÍVEIS:\n{data_table}\n\n"
                    f"Use estes dados para fundamentar sua resposta com valores específicos."
                )

        if prefix_sections:
            llm_prompt = "\n\n".join(prefix_sections) + "\n\n" + llm_prompt

        state["llm_prompt"] = llm_prompt

        logger.info(
            "[dynamic_build_prompt_node] Built dynamic prompt with %d characters",
            len(llm_prompt),
        )
        logger.debug(
            "[dynamic_build_prompt_node] Intent: %s, Polarity: %s",
            enriched_intent.base_intent,
            enriched_intent.polarity.value,
        )

        return state

    except Exception as e:
        error_msg = f"Failed to build dynamic prompt: {str(e)}"
        logger.error(f"[dynamic_build_prompt_node] {error_msg}")

        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)

        # Fallback: use empty prompt (will be caught downstream)
        state["llm_prompt"] = ""

        return state


# ============================================================================
# Helper: Enrich Intent in parse_input_node
# ============================================================================


def enrich_intent_in_state(state: InsightState) -> InsightState:
    """
    Helper function to add intent enrichment to parse_input_node.

    This function should be called within parse_input_node after
    the basic parsing is complete.

    Required State Fields:
        - intent: str (from graphic_classifier)
        - original_query: str (user query)
        - chart_spec: Dict (optional)
        - analytics_metadata: Dict (optional)

    Updated State Fields:
        - enriched_intent: EnrichedIntent

    Args:
        state: Current workflow state

    Returns:
        Updated state with enriched_intent
    """
    logger.info("[enrich_intent_in_state] Starting intent enrichment")

    try:
        # Validate required fields
        if "intent" not in state:
            raise ValueError("Missing required field: intent")

        intent = state["intent"]
        original_query = state.get("original_query", "")
        chart_spec = state.get("chart_spec", {})
        analytics_metadata = state.get("analytics_metadata", {})

        # Create enricher
        enricher = IntentEnricher()

        # Enrich intent
        enriched_intent = enricher.enrich(
            base_intent=intent,
            query=original_query,
            chart_spec=chart_spec,
            analytics_metadata=analytics_metadata,
        )

        state["enriched_intent"] = enriched_intent

        logger.info(
            "[enrich_intent_in_state] Intent enriched: %s -> polarity=%s, temporal=%s",
            intent,
            enriched_intent.polarity.value,
            enriched_intent.temporal_focus.value,
        )

        return state

    except Exception as e:
        error_msg = f"Failed to enrich intent: {str(e)}"
        logger.error(f"[enrich_intent_in_state] {error_msg}")

        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_msg)

        return state


# ============================================================================
# Integration Example - Full Node Replacement
# ============================================================================


def integrated_parse_input_node(state: InsightState) -> InsightState:
    """
    Example: parse_input_node with integrated intent enrichment.

    This is a COMPLETE REPLACEMENT example showing how to integrate
    FASE 1 (IntentEnricher) into the existing parse_input_node.

    Args:
        state: Current workflow state

    Returns:
        Updated state with enriched_intent
    """
    logger.info("[integrated_parse_input_node] Starting with intent enrichment")

    # ... (existing parse_input_node logic) ...
    # This would parse analytics_result, chart_type, etc.

    # NEW: Add intent enrichment
    state = enrich_intent_in_state(state)

    return state


# ============================================================================
# Backward Compatibility - Feature Flag
# ============================================================================


def should_use_dynamic_prompt(state: InsightState) -> bool:
    """
    Feature flag to enable/disable dynamic prompt builder.

    This allows gradual rollout and A/B testing.

    Args:
        state: Current workflow state

    Returns:
        True if dynamic prompt should be used, False for legacy
    """
    # Check for feature flag in state
    use_dynamic = state.get("use_dynamic_prompt", False)

    # Check for environment variable (if needed)
    # import os
    # use_dynamic = os.getenv("USE_DYNAMIC_PROMPT", "false").lower() == "true"

    logger.debug("[should_use_dynamic_prompt] Dynamic prompt enabled: %s", use_dynamic)

    return use_dynamic


def adaptive_build_prompt_node(state: InsightState) -> InsightState:
    """
    Adaptive node that chooses between dynamic and legacy prompt builder.

    This node provides backward compatibility during rollout.

    Args:
        state: Current workflow state

    Returns:
        Updated state with llm_prompt
    """
    if should_use_dynamic_prompt(state):
        logger.info("[adaptive_build_prompt_node] Using DYNAMIC prompt builder")
        return dynamic_build_prompt_node(state)
    else:
        logger.info("[adaptive_build_prompt_node] Using LEGACY prompt builder")
        # Call existing build_prompt_node
        from .nodes import build_prompt_node

        return build_prompt_node(state)


# ============================================================================
# Validation Helper
# ============================================================================


def validate_dynamic_prompt_requirements(state: InsightState) -> Dict[str, bool]:
    """
    Validates that all requirements for dynamic prompt are met.

    Use this helper to check if the state has all necessary fields
    for dynamic prompt building.

    Args:
        state: Current workflow state

    Returns:
        Dict with validation results:
        {
            "has_enriched_intent": bool,
            "has_composed_metrics": bool,
            "has_chart_spec": bool,
            "has_analytics_metadata": bool,
            "ready_for_dynamic_prompt": bool,
        }
    """
    validation = {
        "has_enriched_intent": "enriched_intent" in state
        and state["enriched_intent"] is not None,
        "has_composed_metrics": "numeric_summary" in state
        and state["numeric_summary"] is not None,
        "has_chart_spec": "chart_spec" in state and state["chart_spec"] is not None,
        "has_analytics_metadata": "analytics_metadata" in state
        and state["analytics_metadata"] is not None,
    }

    # Ready if at least enriched_intent and composed_metrics are present
    validation["ready_for_dynamic_prompt"] = (
        validation["has_enriched_intent"] and validation["has_composed_metrics"]
    )

    logger.debug("[validate_dynamic_prompt_requirements] Validation: %s", validation)

    return validation


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    "dynamic_build_prompt_node",
    "enrich_intent_in_state",
    "integrated_parse_input_node",
    "adaptive_build_prompt_node",
    "should_use_dynamic_prompt",
    "validate_dynamic_prompt_requirements",
]
