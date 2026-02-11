"""
Pydantic schemas and LLM configuration for insight generator.

This module defines the output schemas for insights, the LLM loader function,
and the dynamic model selection logic (FASE 3).

GEMINI MIGRATION:
- Uses ChatGoogleGenerativeAI (Google Gemini)
- Model: gemini-2.5-flash (default, FASE 3 upgrade)
- Optimizations: timeout=30s, max_retries=2
- temperature=0.4 for balanced output
- JSON mode automatic (response_mime_type="application/json")

References:
- Authentication: https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb
"""

import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from src.shared_lib.core.config import get_insight_config
from ..core.settings import (
    INSIGHT_MODEL_DEFAULT,
    INSIGHT_MODEL_LITE,
    INSIGHT_TEMPERATURE_DEFAULT,
)

logger = logging.getLogger(__name__)


class InsightItem(BaseModel):
    """Um insight individual"""

    title: str = Field(..., description="Título do insight (para negrito)")
    content: str = Field(..., description="Conteúdo completo do insight")
    metrics: Dict[str, Any] = Field(..., description="Métricas usadas no cálculo")
    confidence: float = Field(default=0.8, ge=0, le=1)
    chart_context: str = Field(..., description="Tipo de gráfico relacionado")


class InsightMetadata(BaseModel):
    """Metadados da geração"""

    calculation_time: float
    metrics_count: int
    llm_model: str
    timestamp: str
    transparency_validated: bool


class InsightOutput(BaseModel):
    """Output completo do insight_generator"""

    status: str = Field(..., pattern="^(success|error)$")
    chart_type: str
    insights: List[InsightItem] = Field(..., max_length=5)
    metadata: InsightMetadata
    error: Optional[str] = None


def select_insight_model(enriched_intent: Optional[Dict[str, Any]] = None) -> str:
    """
    Select the appropriate LLM model based on query complexity.

    FASE 3: Dynamic model selection based on enriched_intent metadata.
    Complex queries (comparisons, variations, negative polarity, temporal)
    use the full flash model. Simple queries (rankings, distributions with
    single period) can use flash-lite for lower latency and cost.

    Decision criteria:
        flash (default):
            - base_intent in {comparison, variation, composition, trend}
            - polarity == NEGATIVE
            - comparison_type != NONE
            - temporal_focus in {PERIOD_OVER_PERIOD, TIME_SERIES}
            - No enriched_intent available (safe default)

        flash-lite:
            - base_intent in {ranking, distribution}
            - temporal_focus == SINGLE_PERIOD
            - polarity != NEGATIVE
            - comparison_type == NONE

    Args:
        enriched_intent: Optional dict with base_intent, polarity,
                        temporal_focus, comparison_type, etc.

    Returns:
        Model name string (e.g., "gemini-2.5-flash")
    """
    if enriched_intent is None:
        logger.debug("[select_insight_model] No enriched_intent, using default model")
        return INSIGHT_MODEL_DEFAULT

    base_intent = enriched_intent.get("base_intent", "")
    polarity = enriched_intent.get("polarity", "neutral")
    comparison_type = enriched_intent.get("comparison_type", "none")
    temporal_focus = enriched_intent.get("temporal_focus", "single_period")

    # Complex intents always use the full model
    complex_intents = {"comparison", "variation", "composition", "trend", "temporal"}
    if base_intent in complex_intents:
        logger.info(
            f"[select_insight_model] Complex intent '{base_intent}' -> {INSIGHT_MODEL_DEFAULT}"
        )
        return INSIGHT_MODEL_DEFAULT

    # Negative polarity requires deeper reasoning
    if polarity == "negative":
        logger.info(
            f"[select_insight_model] Negative polarity -> {INSIGHT_MODEL_DEFAULT}"
        )
        return INSIGHT_MODEL_DEFAULT

    # Explicit comparisons require the full model
    if comparison_type and comparison_type != "none":
        logger.info(
            f"[select_insight_model] Comparison type '{comparison_type}' -> {INSIGHT_MODEL_DEFAULT}"
        )
        return INSIGHT_MODEL_DEFAULT

    # Multi-period temporal analysis requires the full model
    complex_temporal = {"period_over_period", "time_series", "seasonality"}
    if temporal_focus in complex_temporal:
        logger.info(
            f"[select_insight_model] Temporal focus '{temporal_focus}' -> {INSIGHT_MODEL_DEFAULT}"
        )
        return INSIGHT_MODEL_DEFAULT

    # Simple queries: ranking or distribution with single period
    simple_intents = {"ranking", "distribution"}
    if base_intent in simple_intents and temporal_focus == "single_period":
        logger.info(
            f"[select_insight_model] Simple intent '{base_intent}' + single_period -> {INSIGHT_MODEL_LITE}"
        )
        return INSIGHT_MODEL_LITE

    # Default to the full model for safety
    logger.debug(f"[select_insight_model] Default fallback -> {INSIGHT_MODEL_DEFAULT}")
    return INSIGHT_MODEL_DEFAULT


def load_insight_llm(
    model_override: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """
    Load Google Gemini LLM instance with FASE 3 configuration.

    FASE 3 Updates:
    - Default model upgraded to gemini-2.5-flash
    - Temperature reduced to 0.4 for balanced output
    - Supports model_override for dynamic model selection
    - JSON mode automatic (response_mime_type="application/json")

    Args:
        model_override: Optional model name to override the default.
                       Used by select_insight_model() for dynamic selection.

    Returns:
        ChatGoogleGenerativeAI: Configured instance with timeout, retry and JSON mode
    """
    overrides = {}
    if model_override:
        overrides["model"] = model_override
        logger.info(f"[load_insight_llm] Using model override: {model_override}")

    config = get_insight_config(**overrides)
    return ChatGoogleGenerativeAI(**config.to_gemini_kwargs())
