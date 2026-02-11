"""
Prompt builder for generating intention-driven LLM prompts.

FASE 2 Implementation: Replaces rigid chart-type templates with dynamic,
intent-driven prompt construction. The LLM receives the user's question,
real data, and contextual guidance based on enriched intent rather than
chart type, enabling adaptive responses that directly address user needs.

Key changes from legacy:
    - Removed rigid SYSTEM_PROMPT (177-line fixed structure)
    - Removed CHART_TYPE_TEMPLATES (8 chart-type templates)
    - Response format is flexible JSON with 'resposta' as primary field
    - Intent-based guidelines replace chart-type templates
    - Simplified metric formatting (no academic metrics)
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Intent → Guideline mapping (replaces CHART_TYPE_TEMPLATES)
# ============================================================================

INTENT_GUIDELINES: Dict[str, str] = {
    "ranking": (
        "Liste os itens em ordem, com valores. "
        "Identifique líderes e gaps entre posições consecutivas."
    ),
    "trend": (
        "Descreva a evolução cronológica. Destaque picos, vales e tendências gerais."
    ),
    "temporal": (
        "Descreva a evolução cronológica. Destaque picos, vales e tendências gerais."
    ),
    "comparison": (
        "Compare explicitamente os elementos. Use diferenças absolutas e percentuais."
    ),
    "distribution": (
        "Identifique os dominantes e suas participações. "
        "Destaque concentração e proporções."
    ),
    "composition": (
        "Descreva a composição por componente. Identifique padrões e desequilíbrios."
    ),
    "variation": (
        "Foque nas mudanças. Ranking de maiores variações (positivas ou negativas)."
    ),
}

POLARITY_GUIDANCE: Dict[str, str] = {
    "positive": "Foque em oportunidades, crescimento e destaques positivos.",
    "negative": "Foque em riscos, quedas e pontos de atenção.",
    "neutral": "Apresente panorama geral de forma equilibrada.",
}


# ============================================================================
# Metrics that should be excluded from simplified prompt (academic/noise)
# ============================================================================

_EXCLUDED_METRIC_KEYS = {
    "hhi",
    "diversidade_pct",
    "equilibrio",
    "balanceamento",
    "coeficiente_variacao",
    "cv",
    "shannon",
    "simpson",
    "gini",
    "spread_inicial",
    "spread_final",
    "convergencia_pct",
    "correlacao_media",
    "score_balanceamento",
    "diversity_score",
}


# ============================================================================
# System prompt builder (replaces rigid SYSTEM_PROMPT constant)
# ============================================================================


def build_system_prompt(enriched_intent: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a dynamic system prompt based on intent context.

    Instead of a rigid 177-line fixed structure, this generates a concise,
    intention-driven system message that positions the LLM as a commercial
    data analyst and defines a flexible response format.

    Args:
        enriched_intent: Optional enriched intent dict with base_intent,
                        polarity, narrative_angle, etc.

    Returns:
        System prompt string
    """
    parts = []

    # Core persona and behavioral rules
    parts.append(
        "Você é um analista de dados comerciais. "
        "Responda a pergunta do usuário de forma direta, clara e útil "
        "para tomada de decisão."
    )

    parts.append(
        "\nREGRAS FUNDAMENTAIS:\n"
        "- Responda a pergunta diretamente na primeira frase.\n"
        "- Use valores específicos dos dados (nomes, números, percentuais).\n"
        "- Seja conciso. Não gere seções desnecessárias.\n"
        "- Não use emojis.\n"
        "- Linguagem profissional e acessível (evite jargão estatístico).\n"
        "- Mencione filtros ativos naturalmente no texto quando relevante."
    )

    # Intent-specific guideline
    if enriched_intent:
        base_intent = enriched_intent.get("base_intent", "")
        guideline = INTENT_GUIDELINES.get(base_intent, "")
        if guideline:
            parts.append(f"\nDIRETRIZ DE ANÁLISE:\n{guideline}")

        polarity = enriched_intent.get("polarity", "neutral")
        polarity_guide = POLARITY_GUIDANCE.get(polarity, "")
        if polarity_guide:
            parts.append(f"Tom: {polarity_guide}")

    # Flexible response format (JSON)
    parts.append(
        "\nFORMATO DE RESPOSTA:\n"
        "Retorne APENAS um JSON válido com a seguinte estrutura:\n"
        "{\n"
        '  "resposta": "Texto da resposta direta ao usuário. '
        "Pode incluir texto corrido, listas com marcadores (- item), "
        'ou tabelas markdown quando fizer sentido para os dados.",\n'
        '  "dados_destacados": ["dado chave 1 com valor", "dado chave 2 com valor"],\n'
        '  "filtros_mencionados": ["filtro 1", "filtro 2"]\n'
        "}\n"
        "\nDIRETRIZES DE FORMATO:\n"
        "- Escolha o formato mais adequado à pergunta:\n"
        "  * Para rankings: liste os itens com valores.\n"
        "  * Para tendências: descreva a evolução com marcos importantes.\n"
        "  * Para comparações: destaque diferenças absolutas e percentuais.\n"
        "  * Para distribuições: identifique dominantes e oportunidades.\n"
        '- O campo "resposta" é o texto principal que o usuário verá.\n'
        "- Use negrito (**texto**) para destacar nomes e valores importantes.\n"
        '- "dados_destacados" deve conter 3-5 descobertas-chave com valores concretos.\n'
        "- Não gere seções artificiais, recomendações genéricas ou métricas acadêmicas."
    )

    return "\n".join(parts)


# ============================================================================
# Simplified metric formatting (replaces chart-type-specific formatters)
# ============================================================================


def _format_number(value: float, is_percentage: bool = False) -> str:
    """Format number with thousand separators for readability."""
    if is_percentage:
        return f"{value:.2f}%"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"


def _format_simplified_metrics(numeric_summary: Dict[str, Any]) -> str:
    """
    Format metrics for the prompt, keeping only essential context.

    FASE 2: Simplifies metric formatting by excluding academic/statistical
    metrics (HHI, diversity indices, balance scores, etc.) and keeping
    only actionable values: totals, top N, min/max, variations.

    Args:
        numeric_summary: Dict with calculated metrics

    Returns:
        Formatted string with simplified metrics
    """
    if not numeric_summary:
        return ""

    lines = []

    # Total geral
    total = numeric_summary.get("total")
    if total is not None and isinstance(total, (int, float)):
        lines.append(f"Total geral: {_format_number(total)}")

    # Top N with values
    top_n = numeric_summary.get("top_n")
    sum_top_n = numeric_summary.get("sum_top_n")
    concentracao = numeric_summary.get("concentracao_top_n_pct")
    if top_n and sum_top_n:
        line = f"Top {top_n}: {_format_number(sum_top_n)}"
        if concentracao:
            line += f" ({concentracao:.1f}% do total)"
        lines.append(line)

    # Leader
    lider_label = numeric_summary.get("lider_label")
    lider_valor = numeric_summary.get("lider_valor")
    if lider_label and lider_valor:
        lines.append(f"Líder: {lider_label} = {_format_number(lider_valor)}")

    # Gap between 1st and 2nd
    gap_absoluto = numeric_summary.get("gap_absoluto")
    gap_percentual = numeric_summary.get("gap_percentual")
    if gap_absoluto and gap_percentual:
        lines.append(
            f"Gap líder-segundo: {_format_number(gap_absoluto)} ({gap_percentual:.1f}%)"
        )

    # Min/Max with labels
    max_valor = numeric_summary.get("max_valor")
    max_label = numeric_summary.get("max_label")
    min_valor = numeric_summary.get("min_valor")
    min_label = numeric_summary.get("min_label")
    if max_valor is not None and min_valor is not None:
        max_str = f"Max: {_format_number(max_valor)}"
        if max_label:
            max_str = f"Max ({max_label}): {_format_number(max_valor)}"
        min_str = f"Min: {_format_number(min_valor)}"
        if min_label:
            min_str = f"Min ({min_label}): {_format_number(min_valor)}"
        lines.append(f"{max_str} | {min_str}")

    # Total variation (when temporal)
    variacao_percentual = numeric_summary.get("variacao_percentual")
    valor_inicial = numeric_summary.get("valor_inicial")
    valor_final = numeric_summary.get("valor_final")
    if variacao_percentual is not None and valor_inicial and valor_final:
        lines.append(
            f"Variação: {_format_number(valor_inicial)} → "
            f"{_format_number(valor_final)} ({variacao_percentual:+.1f}%)"
        )

    # Trend if detected
    tendencia = numeric_summary.get("tendencia")
    if tendencia:
        lines.append(f"Tendência: {tendencia}")

    # Include remaining numeric metrics that are NOT excluded
    seen_keys = {
        "total",
        "top_n",
        "sum_top_n",
        "concentracao_top_n_pct",
        "lider_label",
        "lider_valor",
        "peso_lider_total_pct",
        "segundo_valor",
        "segundo_label",
        "gap_absoluto",
        "gap_percentual",
        "max_valor",
        "max_label",
        "min_valor",
        "min_label",
        "variacao_percentual",
        "valor_inicial",
        "valor_final",
        "variacao_absoluta",
        "tendencia",
        "metadata",
        "modules_used",
        "top3_sum",
        "concentracao_top3_pct",
        "total_items",
        "tail_sum",
        "tail_pct",
    }

    for key, value in numeric_summary.items():
        if key in seen_keys:
            continue
        if key.startswith("_") or key.endswith("_col") or key.endswith("_label"):
            continue
        if isinstance(value, (dict, list)):
            continue
        # Exclude academic/noise metrics
        key_lower = key.lower()
        if any(excl in key_lower for excl in _EXCLUDED_METRIC_KEYS):
            continue
        if isinstance(value, (int, float)):
            if key.endswith("_pct") or "percentual" in key_lower:
                lines.append(f"{key}: {value:.1f}%")
            else:
                lines.append(f"{key}: {_format_number(value)}")

    if not lines:
        return ""

    return "\n".join(lines)


# ============================================================================
# Main prompt builder (FASE 2: intention-driven)
# ============================================================================


def build_prompt(
    numeric_summary: Dict[str, Any],
    chart_type: str,
    filters: Dict[str, Any] = None,
    user_query: str = "",
    data_table: str = "",
    intent_context: str = "",
    enriched_intent: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build intention-driven prompt for the LLM.

    FASE 2: Constructs a dynamic prompt centered on the user's question,
    with real data, contextual filters, and simplified metrics as auxiliary
    context. No chart-type templates or rigid structure imposed.

    Args:
        numeric_summary: Calculated metrics (auxiliary context only)
        chart_type: Chart type (used only for logging, not for template selection)
        filters: Active filters (optional)
        user_query: Original user question (primary directive)
        data_table: Real data as markdown table
        intent_context: Semantic context from enriched_intent
        enriched_intent: Full enriched intent dict (for guideline selection)

    Returns:
        Formatted prompt string for the LLM
    """
    sections = []

    # Primary: User question
    if user_query:
        sections.append(
            f'PERGUNTA DO USUÁRIO:\n"{user_query}"\n\n'
            f"Sua resposta DEVE responder diretamente a esta pergunta."
        )

    # Filters with semantic context
    if filters:
        filters_text = _format_filters_for_prompt(filters)
        if filters_text:
            sections.append(
                f"FILTROS ATIVOS:\n{filters_text}\n\n"
                f"Mencione os filtros naturalmente na resposta quando relevante, "
                f"usando nomes legíveis (ex: 'Santa Catarina' em vez de 'SC')."
            )

    # Real data as markdown table
    if data_table:
        sections.append(
            f"DADOS DISPONÍVEIS:\n{data_table}\n\n"
            f"Use estes dados para fundamentar sua resposta com valores específicos "
            f"(nomes, números, percentuais). Cite os dados reais."
        )

    # Enriched intent context
    if intent_context:
        sections.append(f"CONTEXTO DA ANÁLISE:\n{intent_context}")

    # Simplified metrics as auxiliary context (not as a template to fill)
    metrics_text = _format_simplified_metrics(numeric_summary)
    if metrics_text:
        sections.append(f"MÉTRICAS AUXILIARES (contexto adicional):\n{metrics_text}")

    return "\n\n".join(sections)


# ============================================================================
# Filter formatting (preserved from FASE 1)
# ============================================================================


def _format_filters_for_prompt(filters: Dict[str, Any]) -> str:
    """
    Format filters for prompt inclusion with semantic context.

    Produces human-readable filter descriptions that allow the LLM
    to contextualize them naturally in the response narrative.

    Args:
        filters: Dict of applied filters

    Returns:
        Formatted string with readable filters
    """
    if not filters:
        return ""

    descriptions = []
    for key, value in filters.items():
        readable_key = key.replace("_", " ")

        if isinstance(value, list):
            if len(value) == 1:
                descriptions.append(f"{readable_key}: {value[0]}")
            else:
                descriptions.append(f"{readable_key}: {', '.join(map(str, value))}")
        elif isinstance(value, dict):
            if "between" in value:
                descriptions.append(
                    f"{readable_key}: entre {value['between'][0]} e {value['between'][1]}"
                )
            else:
                descriptions.append(f"{readable_key}: {value}")
        else:
            descriptions.append(f"{readable_key}: {value}")

    return " | ".join(descriptions)
