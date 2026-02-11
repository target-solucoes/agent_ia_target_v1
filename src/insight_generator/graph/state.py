"""
State definition for the Insight Generator LangGraph workflow.

This module defines InsightState, a TypedDict that manages the flow of data
through all nodes in the insight generation workflow.
"""

from typing import TypedDict, Dict, Any, List, Optional
import pandas as pd


class InsightState(TypedDict, total=False):
    """
    State schema for the Insight Generator LangGraph workflow.

    This TypedDict tracks data flow through the workflow nodes:
    1. parse_input_node: Populates chart_spec, analytics_result, data
    2. calculate_metrics_node: Populates numeric_summary, chart_type, cache_key
    3. build_prompt_node: Populates llm_prompt
    4. invoke_llm_node: Populates llm_response
    5. validate_insights_node: Validates and structures insights
    6. format_output_node: Populates final_output

    All fields are optional (total=False) to allow incremental state building.
    """

    # ========== INPUT (from upstream agents) ==========
    chart_spec: Dict[str, Any]
    """
    Chart specification from graphic_classifier.
    Contains: chart_type, dimensions, metrics, filters, etc.
    Example:
        {
            "chart_type": "bar_horizontal",
            "dimensions": [{"column": "cliente", "alias": "Cliente"}],
            "metrics": [{"column": "valor", "aggregation": "sum"}],
            ...
        }
    """

    analytics_result: Dict[str, Any]
    """
    Analytics output from analytics_executor.
    Contains: data, metadata, plotly_config, etc.
    Example:
        {
            "data": [...],
            "metadata": {"rows": 10, "columns": 2},
            "plotly_config": {...}
        }
    """

    data: Optional[pd.DataFrame]
    """
    Processed DataFrame extracted from analytics_result.
    Used for metric calculations.
    """

    # ========== USER CONTEXT (FASE 1) ==========
    user_query: Optional[str]
    """
    Original user query extracted from chart_spec or analytics_result.
    Used by build_prompt_node (FASE 1) to inject the user's question into the
    LLM prompt so the response directly addresses their intent.
    """

    # ========== INTERMEDIATE (processing) ==========
    chart_type: str
    """
    Chart type extracted from chart_spec.
    Used for routing to appropriate calculator.
    Values: bar_horizontal, bar_vertical, line, pie, etc.
    """

    enriched_intent: Optional[Dict[str, Any]]
    """
    Enriched intent with semantic metadata.
    Populated by IntentEnricher in parse_input_node.
    Contains: base_intent, polarity, temporal_focus, comparison_type,
              suggested_metrics, key_entities, filters_context, narrative_angle
    Example:
        {
            "base_intent": "variation",
            "polarity": "negative",
            "temporal_focus": "period_over_period",
            "comparison_type": "period_vs_period",
            "suggested_metrics": ["delta", "growth_rate", "loss_magnitude"],
            "key_entities": ["Produto", "Maio", "Junho", "2016"],
            "filters_context": {"has_filters": true, "temporal_filter": true},
            "narrative_angle": "análise de variação e mudança, com foco em quedas e riscos"
        }
    """

    numeric_summary: Dict[str, Any]
    """
    Calculated metrics specific to chart_type.
    Populated by calculator classes.
    Example for bar_horizontal:
        {
            "total": 12000,
            "top_n": 5,
            "sum_top_n": 8000,
            "concentracao_top_n_pct": 66.7,
            "lider_valor": 3000,
            "gap_percentual": 45.0,
            ...
        }
    """

    cache_key: str
    """
    Hash key for caching metric calculations.
    Prevents redundant calculations for similar queries.
    """

    # ========== LLM (language model processing) ==========
    llm_prompt: str
    """
    Formatted prompt sent to LLM.
    Built from numeric_summary and chart_type specific template.
    """

    system_prompt: Optional[str]
    """
    Dynamic system prompt built by build_system_prompt().
    FASE 2: Replaces the rigid SYSTEM_PROMPT constant with an
    intent-driven system message tailored to the user's question.
    """

    llm_response: str
    """
    Raw text response from LLM.
    Contains unstructured insights that need parsing.
    """

    # ========== FASE 2: INTENTION-DRIVEN OUTPUT ==========
    resposta: Optional[str]
    """
    FASE 2: Primary LLM response text addressing the user's question directly.
    This is the main content the user sees. May contain markdown formatting
    (bold, lists, tables). Replaces the rigid 4-section structure.
    """

    dados_destacados: Optional[List[str]]
    """
    FASE 2: Key data points highlighted in the analysis (3-5 items).
    Each item contains a concrete finding with values.
    Example: ["Loja A lidera com R$ 1.2M (32% do total)", "Gap de 45% entre 1º e 2º"]
    """

    filtros_mencionados: Optional[List[str]]
    """
    FASE 2: Filters mentioned/contextualized in the response.
    Tracks which active filters were referenced in the narrative.
    Example: ["Santa Catarina", "Janeiro a Março 2024"]
    """

    # ========== OUTPUT (final results - FASE 4 UNIFIED) ==========
    executive_summary: Dict[str, str]
    """
    Executive summary with title and introduction.
    Populated by validate_insights_node from unified LLM response (FASE 4).
    Structure:
        {
            "title": "Professional analysis title (max 80 chars)",
            "introduction": "Contextual introduction (50-300 chars)"
        }
    """

    insights: List[Dict[str, Any]]
    """
    Structured list of detailed insights (detailed_insights from LLM).
    Each insight contains:
        - title: str (for bold formatting)
        - content: str (insight text with values)
        - formula: str (calculation with operators) [JSON mode]
        - interpretation: str (strategic implication) [JSON mode]
        - metrics: Dict (base values used)
        - confidence: float (0-1)
    Example:
        [
            {
                "title": "Concentracao Critica no Top 3",
                "content": "Top 3 = 8.66M / Top 5 = 12.68M → 68.3%...",
                "formula": "Top 3 = 8.66M / Total 12.68M → 68.3%",
                "interpretation": "Dependência crítica indica risco...",
                "metrics": {"top3_sum": 8.66, "top5_sum": 12.68},
                "confidence": 0.9
            }
        ]
    """

    synthesized_narrative: str
    """
    Cohesive executive narrative connecting key insights (400-800 chars).
    Populated by validate_insights_node from unified LLM response (FASE 4).
    Uses natural language, not telegraphic.
    """

    key_findings: List[str]
    """
    List of 3-5 concise key findings (bullet points, max 140 chars each).
    Populated by validate_insights_node from unified LLM response (FASE 4).
    Actionable and with concrete values.
    """

    next_steps: List[str]
    """
    List of exactly 3 strategic recommendations (max 200 chars each).
    Populated by validate_insights_node from unified LLM response (FASE 4).
    Direct, actionable, and contextualized to the insights.
    """

    formatted_insights: str
    """
    Executive markdown-formatted insights.
    Transformed from insights list using ExecutiveMarkdownFormatter.
    Contains H3 headers, bullet points, bold formatting, and separators.
    Example:
        ### **Concentração de Poder**

        * Top 3 = **R$ 8,66M** / Total **R$ 12,68M** → **68,3%**
        * Indica **alta concentração** e **dependência crítica**.

        ---

        ### **Gap Competitivo**
        ...
    """

    final_output: Dict[str, Any]
    """
    Complete output of the insight generator.
    Structure:
        {
            "status": "success" | "error",
            "chart_type": str,
            "insights": List[InsightItem],
            "metadata": {
                "calculation_time": float,
                "metrics_count": int,
                "llm_model": str,
                "timestamp": str,
                "transparency_validated": bool
            },
            "error": Optional[str]
        }
    """

    # ========== ERROR HANDLING ==========
    errors: List[str]
    """
    List of error messages encountered during workflow execution.
    Allows tracking multiple errors without stopping the workflow.
    """

    # ========== METADATA (optional tracking) ==========
    calculation_time: Optional[float]
    """Time taken for metric calculations in seconds."""

    transparency_validated: Optional[bool]
    """Whether insights passed transparency validation (showing base values)."""

    metrics_count: Optional[int]
    """Number of metrics calculated."""

    # ========== FASE 5: ALIGNMENT VALIDATION METADATA ==========
    alignment_score: Optional[float]
    """
    Alignment score between narrative and detailed_insights (0.0 to 1.0).
    Populated by validate_insights_node (FASE 5).
    Score >= 0.95 indicates good alignment.
    """

    alignment_validated: Optional[bool]
    """
    Whether alignment validation passed (score >= 0.95).
    Populated by validate_insights_node (FASE 5).
    """

    corrections_applied: Optional[List[str]]
    """
    List of automatic corrections applied by AlignmentCorrector.
    Populated by validate_insights_node (FASE 5).
    Examples:
        - "Added placeholder for missing metric: concentração"
        - "Added key_finding from narrative sentence"
        - "Generated executive_summary title"
    """

    alignment_warnings: Optional[List[str]]
    """
    List of alignment warnings detected during validation.
    Populated by validate_insights_node (FASE 5).
    Examples:
        - "Found 2 metric(s) in narrative without detailed explanation"
        - "Found 1 numeric value mismatch(es)"
    """

    _alignment_retry_count: Optional[int]
    """
    Internal counter for alignment retry attempts (max 2).
    Used by validate_insights_node for retry logic.
    """

    _retry_reason: Optional[str]
    """
    Internal field storing the reason for retry.
    Used by validate_insights_node for debugging.
    """

    # ========== TOKEN TRACKING ==========
    agent_tokens: Optional[Dict[str, Dict[str, int]]]
    """
    Token usage tracking per agent.
    Structure:
        {
            "insight_generator": {
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int
            }
        }
    Populated by invoke_llm_node when LLM is called.
    """
