"""
FASE 4.4 - Insight Quality Evaluators (Success Criteria).

Defines measurable, deterministic quality evaluation functions that
score insight_generator responses against objective criteria.

Each evaluator returns a float score in [0.0, 1.0] and a short rationale,
enabling automated benchmark tracking and regression detection.

Criteria implemented:
    1. intent_adherence  - Does the response address the user's question?
    2. data_coherence    - Does the response cite real data values?
    3. filter_mention    - Are active filters contextualized in the response?
    4. format_adequacy   - Is the response format appropriate for the intent?
    5. noise_absence     - Is the response free of academic/irrelevant metrics?
    6. structure_valid   - Is the output JSON structurally correct?
    7. conciseness       - Is the response concise (no inflated sections)?
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Result
# ============================================================================


@dataclass
class EvalResult:
    """Result of a single quality evaluation."""

    criterion: str
    score: float  # 0.0 - 1.0
    passed: bool
    rationale: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Aggregated quality report for a single insight response."""

    results: List[EvalResult]
    overall_score: float = 0.0
    passed: bool = False

    def __post_init__(self):
        if self.results:
            self.overall_score = sum(r.score for r in self.results) / len(self.results)
            self.passed = all(r.passed for r in self.results)

    def summary(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "passed": self.passed,
            "criteria": {
                r.criterion: {
                    "score": round(r.score, 3),
                    "passed": r.passed,
                    "rationale": r.rationale,
                }
                for r in self.results
            },
        }


# ============================================================================
# Quality Thresholds (from planning.md §4.4)
# ============================================================================

DEFAULT_THRESHOLDS = {
    "intent_adherence": 0.90,
    "data_coherence": 0.85,
    "filter_mention": 0.90,
    "format_adequacy": 0.80,
    "noise_absence": 0.95,
    "structure_valid": 1.00,
    "conciseness": 0.80,
}


# ============================================================================
# Academic / noise metric keywords to detect
# ============================================================================

NOISE_KEYWORDS = [
    "hhi",
    "herfindahl",
    "hirschman",
    "simpson",
    "shannon",
    "gini",
    "coeficiente de variação",
    "coeficiente de variacao",
    "diversidade =",
    "equilibrio =",
    "equilíbrio =",
    "score de balanceamento",
    "spread das series",
    "spread das séries",
    "correlação média",
    "correlacao media",
    "convergência",
    "convergencia",
    "divergência",
    "divergencia",
]


# ============================================================================
# 1. Intent Adherence
# ============================================================================


def evaluate_intent_adherence(
    resposta: str,
    user_query: str,
    enriched_intent: Optional[Dict[str, Any]] = None,
    expected_entities: Optional[List[str]] = None,
) -> EvalResult:
    """
    Evaluate whether the response directly addresses the user's question.

    Checks:
    - Key entities from the query appear in the response
    - The response's opening sentence is related to the query topic
    - Intent-specific keywords are present

    Args:
        resposta: The generated response text.
        user_query: Original user question.
        enriched_intent: Enriched intent dict (optional).
        expected_entities: Specific entity names expected in the response.

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not resposta or not user_query:
        return EvalResult(
            criterion="intent_adherence",
            score=0.0,
            passed=False,
            rationale="Empty response or query",
        )

    resposta_lower = resposta.lower()
    query_lower = user_query.lower()
    score_components = []

    # Check 1: Key query nouns appear in response
    query_tokens = set(re.findall(r"[a-záàâãéèêíïóôõúüç]{4,}", query_lower))
    # Filter out very common words
    stopwords = {
        "quais",
        "qual",
        "como",
        "foram",
        "para",
        "entre",
        "esse",
        "desse",
        "mesmo",
        "mais",
        "maior",
        "menor",
        "todos",
        "todas",
        "vendas",
        "compras",  # too generic
    }
    query_tokens -= stopwords

    if query_tokens:
        found = sum(1 for t in query_tokens if t in resposta_lower)
        token_ratio = found / len(query_tokens)
        score_components.append(token_ratio)

    # Check 2: Expected entities are mentioned
    if expected_entities:
        found_entities = sum(
            1 for e in expected_entities if e.lower() in resposta_lower
        )
        entity_ratio = found_entities / len(expected_entities)
        score_components.append(entity_ratio)

    # Check 3: Intent-keyword presence
    if enriched_intent:
        base_intent = enriched_intent.get("base_intent", "")
        intent_signals = {
            "ranking": ["maior", "menor", "top", "primeiro", "lider", "líder"],
            "variation": ["queda", "cresci", "variação", "variacao", "aument"],
            "trend": [
                "evolução",
                "evolucao",
                "tendência",
                "tendencia",
                "historico",
                "histórico",
            ],
            "distribution": [
                "distribuição",
                "distribuicao",
                "participação",
                "participacao",
                "%",
            ],
            "comparison": ["compara", "diferença", "diferenca", "versus", "vs"],
            "composition": ["composição", "composicao", "componente", "agrup"],
            "temporal": ["período", "periodo", "mes", "mês", "ano", "trimestre"],
        }
        keywords = intent_signals.get(base_intent, [])
        if keywords:
            found = any(k in resposta_lower for k in keywords)
            score_components.append(1.0 if found else 0.3)

    # Check 4: First sentence is not generic (should mention specifics)
    first_sentence = resposta.split(".")[0] if resposta else ""
    has_specific = bool(re.search(r"R?\$?\s?\d[\d.,]*|\d{3,}|%", first_sentence))
    if len(resposta) > 50:
        score_components.append(0.8 if has_specific else 0.5)

    score = sum(score_components) / len(score_components) if score_components else 0.0
    threshold = DEFAULT_THRESHOLDS["intent_adherence"]

    return EvalResult(
        criterion="intent_adherence",
        score=round(min(score, 1.0), 3),
        passed=score >= threshold,
        rationale=f"Query tokens matched: {score_components}",
        details={
            "query_tokens": list(query_tokens) if query_tokens else [],
            "expected_entities_found": bool(expected_entities),
        },
    )


# ============================================================================
# 2. Data Coherence
# ============================================================================


def evaluate_data_coherence(
    resposta: str,
    data_records: List[Dict[str, Any]],
    key_columns: Optional[List[str]] = None,
) -> EvalResult:
    """
    Evaluate whether the response cites concrete data values from the dataset.

    Checks:
    - Numeric values from the data appear in the response
    - Entity names (column values) from the data appear in the response

    Args:
        resposta: The generated response text.
        data_records: List of dicts representing the source data rows.
        key_columns: Column names whose values should appear in the response.

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not resposta or not data_records:
        return EvalResult(
            criterion="data_coherence",
            score=0.0,
            passed=False,
            rationale="Empty response or data",
        )

    resposta_clean = resposta.replace(".", "").replace(",", "").lower()
    resposta_lower = resposta.lower()
    found_values = 0
    total_checked = 0

    for record in data_records[:20]:  # Limit check to 20 rows
        for col, value in record.items():
            if key_columns and col not in key_columns:
                continue

            if isinstance(value, (int, float)):
                total_checked += 1
                # Check if the numeric value appears (approximately)
                str_val = str(int(value)) if value == int(value) else str(value)
                # Check raw number
                if str_val in resposta_clean:
                    found_values += 1
                # Check abbreviated (e.g., 2.44M, 633K)
                elif _check_abbreviated_number(value, resposta):
                    found_values += 1

            elif isinstance(value, str) and len(value) > 2:
                total_checked += 1
                if value.lower() in resposta_lower:
                    found_values += 1

    score = found_values / total_checked if total_checked > 0 else 0.0
    threshold = DEFAULT_THRESHOLDS["data_coherence"]

    return EvalResult(
        criterion="data_coherence",
        score=round(min(score, 1.0), 3),
        passed=score >= threshold,
        rationale=f"Found {found_values}/{total_checked} data values in response",
        details={"found": found_values, "total_checked": total_checked},
    )


def _check_abbreviated_number(value: float, text: str) -> bool:
    """Check if a number appears in abbreviated form (K, M, B) in text."""
    text_lower = text.lower()
    abs_val = abs(value)

    if abs_val >= 1_000_000:
        abbrev = f"{abs_val / 1_000_000:.1f}".rstrip("0").rstrip(".")
        if f"{abbrev}m" in text_lower or f"{abbrev} m" in text_lower:
            return True
    if abs_val >= 1_000:
        abbrev = f"{abs_val / 1_000:.0f}"
        if f"{abbrev}k" in text_lower or f"{abbrev} k" in text_lower:
            return True
        abbrev_comma = f"{abs_val:,.0f}".replace(",", ".")
        if abbrev_comma in text:
            return True

    return False


# ============================================================================
# 3. Filter Mention
# ============================================================================


def evaluate_filter_mention(
    resposta: str,
    active_filters: Dict[str, Any],
) -> EvalResult:
    """
    Evaluate whether active filters are contextualized in the response.

    Checks that filter values are mentioned naturally in the response text.
    Maps common abbreviations to full names (e.g., SC -> Santa Catarina).

    Args:
        resposta: The generated response text.
        active_filters: Dict of active filter key-value pairs.

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not active_filters:
        return EvalResult(
            criterion="filter_mention",
            score=1.0,
            passed=True,
            rationale="No filters to check",
        )

    if not resposta:
        return EvalResult(
            criterion="filter_mention",
            score=0.0,
            passed=False,
            rationale="Empty response with active filters",
        )

    # Common UF mappings
    UF_MAP = {
        "SC": "santa catarina",
        "SP": "são paulo",
        "PR": "paraná",
        "RS": "rio grande do sul",
        "MG": "minas gerais",
        "RJ": "rio de janeiro",
        "BA": "bahia",
        "GO": "goiás",
        "PE": "pernambuco",
        "CE": "ceará",
    }

    resposta_lower = resposta.lower()
    found = 0
    total = 0

    for key, value in active_filters.items():
        values = value if isinstance(value, list) else [value]
        for v in values:
            total += 1
            v_str = str(v)
            # Direct match
            if v_str.lower() in resposta_lower:
                found += 1
            # UF expansion match
            elif v_str.upper() in UF_MAP:
                if UF_MAP[v_str.upper()] in resposta_lower:
                    found += 1
            # Partial match for long values
            elif len(v_str) > 5 and v_str[:5].lower() in resposta_lower:
                found += 1

    score = found / total if total > 0 else 1.0
    threshold = DEFAULT_THRESHOLDS["filter_mention"]

    return EvalResult(
        criterion="filter_mention",
        score=round(score, 3),
        passed=score >= threshold,
        rationale=f"Filters mentioned: {found}/{total}",
        details={"found": found, "total": total, "filters": active_filters},
    )


# ============================================================================
# 4. Format Adequacy
# ============================================================================


def evaluate_format_adequacy(
    resposta: str,
    enriched_intent: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """
    Evaluate whether the response format is appropriate for the query type.

    Format expectations by intent:
    - ranking: ordered list with values
    - distribution: table or percentages
    - trend/temporal: chronological description with milestones
    - comparison: explicit value differences
    - single_metric: concise (< 3 sentences)

    Args:
        resposta: The generated response text.
        enriched_intent: Enriched intent dict (optional).

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not resposta:
        return EvalResult(
            criterion="format_adequacy",
            score=0.0,
            passed=False,
            rationale="Empty response",
        )

    base_intent = (enriched_intent or {}).get("base_intent", "")
    score = 0.5  # baseline
    rationale_parts = []

    # Generic quality: has specific values
    has_numbers = bool(re.search(r"\d", resposta))
    has_bold = "**" in resposta
    if has_numbers:
        score += 0.2
        rationale_parts.append("has numbers")
    if has_bold:
        score += 0.1
        rationale_parts.append("has bold formatting")

    # Intent-specific checks
    if base_intent == "ranking":
        # Expect ordered list (1., 2., ... or -, *)
        has_list = bool(re.search(r"(^|\n)\s*[\d]+[\.\)]|\n\s*[-\*]", resposta))
        if has_list:
            score += 0.2
            rationale_parts.append("has ordered/unordered list")

    elif base_intent in ("distribution", "composition"):
        # Expect percentages or table
        has_pct = "%" in resposta
        has_table = "|" in resposta
        if has_pct:
            score += 0.1
            rationale_parts.append("has percentages")
        if has_table:
            score += 0.1
            rationale_parts.append("has table")

    elif base_intent in ("trend", "temporal"):
        # Expect chronological/temporal references
        temporal_refs = re.findall(
            r"(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez|"
            r"janeiro|fevereiro|março|marco|abril|maio|junho|"
            r"julho|agosto|setembro|outubro|novembro|dezembro|"
            r"2014|2015|2016|trimestre|semestre|mês|mes)",
            resposta.lower(),
        )
        if len(temporal_refs) >= 2:
            score += 0.2
            rationale_parts.append(f"has {len(temporal_refs)} temporal refs")

    elif base_intent in ("variation", "comparison"):
        # Expect comparison words and deltas
        has_comparison = bool(
            re.search(
                r"queda|cresci|variação|variacao|diferença|diferenca|aument",
                resposta.lower(),
            )
        )
        has_delta = bool(re.search(r"[+-]?\d+[.,]?\d*\s*%", resposta))
        if has_comparison:
            score += 0.1
            rationale_parts.append("has comparison language")
        if has_delta:
            score += 0.1
            rationale_parts.append("has delta values")

    score = min(score, 1.0)
    threshold = DEFAULT_THRESHOLDS["format_adequacy"]

    return EvalResult(
        criterion="format_adequacy",
        score=round(score, 3),
        passed=score >= threshold,
        rationale="; ".join(rationale_parts) or "baseline only",
    )


# ============================================================================
# 5. Noise Absence
# ============================================================================


def evaluate_noise_absence(resposta: str) -> EvalResult:
    """
    Evaluate whether the response is free of academic/irrelevant metrics.

    Checks for presence of noise keywords like HHI, Simpson diversity,
    Gini coefficient, spread/convergence metrics, etc.

    Args:
        resposta: The generated response text.

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not resposta:
        return EvalResult(
            criterion="noise_absence",
            score=1.0,
            passed=True,
            rationale="Empty response – no noise",
        )

    resposta_lower = resposta.lower()
    found_noise = []

    for keyword in NOISE_KEYWORDS:
        if keyword.lower() in resposta_lower:
            found_noise.append(keyword)

    if found_noise:
        # Each noise keyword reduces score
        penalty = min(len(found_noise) * 0.15, 0.6)
        score = max(1.0 - penalty, 0.0)
    else:
        score = 1.0

    threshold = DEFAULT_THRESHOLDS["noise_absence"]

    return EvalResult(
        criterion="noise_absence",
        score=round(score, 3),
        passed=score >= threshold,
        rationale=f"Noise keywords found: {found_noise}" if found_noise else "Clean",
        details={"noise_keywords_found": found_noise},
    )


# ============================================================================
# 6. Structure Valid
# ============================================================================


def evaluate_structure_valid(output: Dict[str, Any]) -> EvalResult:
    """
    Evaluate whether the insight_generator output has correct structure.

    Required fields for FASE 2 output:
    - status: "success"
    - resposta: non-empty string
    - dados_destacados: list
    - filtros_mencionados: list
    - metadata: dict with required keys

    Args:
        output: The full insight_generator output dict.

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not isinstance(output, dict):
        return EvalResult(
            criterion="structure_valid",
            score=0.0,
            passed=False,
            rationale="Output is not a dict",
        )

    checks = []

    # Status check
    status = output.get("status")
    checks.append(("status == success", status == "success"))

    # Resposta exists and is non-empty
    resposta = output.get("resposta", "")
    checks.append(("resposta non-empty", bool(resposta)))

    # dados_destacados is a list
    dd = output.get("dados_destacados")
    checks.append(("dados_destacados is list", isinstance(dd, list)))

    # filtros_mencionados is a list
    fm = output.get("filtros_mencionados")
    checks.append(("filtros_mencionados is list", isinstance(fm, list)))

    # metadata exists and has key fields
    meta = output.get("metadata", {})
    checks.append(("metadata exists", isinstance(meta, dict)))
    if isinstance(meta, dict):
        checks.append(("has llm_model", "llm_model" in meta))
        checks.append(("has timestamp", "timestamp" in meta))

    # Backward compat fields
    checks.append(("has chart_type", "chart_type" in output))

    passed_count = sum(1 for _, ok in checks if ok)
    total = len(checks)
    score = passed_count / total if total > 0 else 0.0
    threshold = DEFAULT_THRESHOLDS["structure_valid"]

    failed = [name for name, ok in checks if not ok]

    return EvalResult(
        criterion="structure_valid",
        score=round(score, 3),
        passed=score >= threshold,
        rationale=f"Passed {passed_count}/{total}"
        + (f" (failed: {failed})" if failed else ""),
        details={"checks": {name: ok for name, ok in checks}},
    )


# ============================================================================
# 7. Conciseness
# ============================================================================


def evaluate_conciseness(
    resposta: str,
    max_tokens: int = 500,
) -> EvalResult:
    """
    Evaluate response conciseness (no inflated sections, reasonable length).

    Checks:
    - Token count <= max_tokens target
    - No artificially generated sections ("Recomendações:", "Próximos Passos:")
    - No filler language

    Args:
        resposta: The generated response text.
        max_tokens: Target maximum word count (~tokens).

    Returns:
        EvalResult with score in [0.0, 1.0].
    """
    if not resposta:
        return EvalResult(
            criterion="conciseness",
            score=1.0,
            passed=True,
            rationale="Empty response",
        )

    words = resposta.split()
    word_count = len(words)

    score = 1.0
    rationale_parts = []

    # Length penalty
    if word_count > max_tokens * 1.5:
        score -= 0.3
        rationale_parts.append(f"too long ({word_count} words)")
    elif word_count > max_tokens:
        excess_ratio = (word_count - max_tokens) / max_tokens
        score -= min(excess_ratio * 0.5, 0.2)
        rationale_parts.append(f"slightly long ({word_count} words)")
    else:
        rationale_parts.append(f"good length ({word_count} words)")

    # Check for artificial/filler sections
    filler_patterns = [
        r"recomendações?:",
        r"recomendacoes?:",
        r"próximos?\s+passos?:",
        r"proximos?\s+passos?:",
        r"conclusão:",
        r"conclusao:",
        r"considerações?\s+finais:",
        r"consideracoes?\s+finais:",
    ]
    filler_found = []
    resposta_lower = resposta.lower()
    for pat in filler_patterns:
        if re.search(pat, resposta_lower):
            filler_found.append(pat)

    if filler_found:
        score -= 0.2
        rationale_parts.append(f"filler sections: {len(filler_found)}")

    score = max(score, 0.0)
    threshold = DEFAULT_THRESHOLDS["conciseness"]

    return EvalResult(
        criterion="conciseness",
        score=round(score, 3),
        passed=score >= threshold,
        rationale="; ".join(rationale_parts),
        details={"word_count": word_count, "filler_found": filler_found},
    )


# ============================================================================
# Composite Evaluator
# ============================================================================


def evaluate_insight_quality(
    output: Dict[str, Any],
    user_query: str,
    data_records: List[Dict[str, Any]],
    active_filters: Optional[Dict[str, Any]] = None,
    enriched_intent: Optional[Dict[str, Any]] = None,
    expected_entities: Optional[List[str]] = None,
    key_columns: Optional[List[str]] = None,
) -> QualityReport:
    """
    Run all quality evaluators on an insight_generator output.

    This is the primary entry point for quality assessment. It combines
    all individual evaluators into a single QualityReport with per-criterion
    scores and an overall pass/fail judgment.

    Args:
        output: Full insight_generator output dict.
        user_query: Original user question.
        data_records: Source data as list of dicts.
        active_filters: Active filters dict (optional).
        enriched_intent: Enriched intent dict (optional).
        expected_entities: Entity names expected in the response.
        key_columns: Columns whose values should appear in the response.

    Returns:
        QualityReport with all criterion results and overall score.
    """
    resposta = output.get("resposta", "")

    results = [
        evaluate_intent_adherence(
            resposta, user_query, enriched_intent, expected_entities
        ),
        evaluate_data_coherence(resposta, data_records, key_columns),
        evaluate_filter_mention(resposta, active_filters or {}),
        evaluate_format_adequacy(resposta, enriched_intent),
        evaluate_noise_absence(resposta),
        evaluate_structure_valid(output),
        evaluate_conciseness(resposta),
    ]

    return QualityReport(results=results)
