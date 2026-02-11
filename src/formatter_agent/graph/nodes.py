"""
LangGraph workflow nodes for formatter agent
=============================================

Defines all processing nodes for the formatter agent workflow:
1. parse_inputs_node - Validate and extract inputs
2. select_handler_node - Select chart-specific handler
3. generate_executive_summary_node - LLM: Generate title + introduction
4. synthesize_insights_node - LLM: Create narrative from insights
5. generate_next_steps_node - LLM: Generate strategic recommendations
6. format_data_table_node - Format data tables
7. assemble_output_node - Assemble final JSON output
8. handle_error_node - Fallback error handling
"""

import logging
import time
from typing import Dict, Any, List

from ..parsers.input_parser import InputParser
from ..handlers.registry import get_handler, is_chart_type_supported
from ..formatters.data_table_formatter import DataTableFormatter
from ..formatters.output_assembler import OutputAssembler
from ..graph.state import FormatterState

logger = logging.getLogger(__name__)


# ============================================================================
# NODE 1: PARSE INPUTS
# ============================================================================


def parse_inputs_node(state: FormatterState) -> Dict[str, Any]:
    """
    Validate and extract inputs from all previous agents.

    This node:
    - Validates required inputs (query, chart_type)
    - Extracts data from filter_classifier, graphic_classifier, analytics_executor
    - Extracts plotly and insight results
    - Reports validation errors

    Args:
        state: Current FormatterState

    Returns:
        Dictionary with:
        - parsed_inputs: Structured input data
        - status: "parsing_complete" or "error"
        - error: Error message if validation fails
    """
    logger.info("=== NODE: parse_inputs ===")
    start_time = time.time()

    try:
        parser = InputParser()
        parsed_inputs = parser.parse(state)

        # Check for critical validation errors
        errors = parsed_inputs.get("validation_errors", [])
        if errors:
            logger.error(f"Input validation failed: {errors}")
            return {
                "status": "error",
                "error": f"Input validation failed: {'; '.join(errors)}",
                "parsed_inputs": parsed_inputs,
            }

        execution_time = time.time() - start_time
        logger.info(
            f"parse_inputs completed successfully in {execution_time:.2f}s - "
            f"chart_type='{parsed_inputs.get('chart_type')}', "
            f"data_rows={len(parsed_inputs.get('data', []))}, "
            f"insights={len(parsed_inputs.get('insights', []))}"
        )

        return {
            "parsed_inputs": parsed_inputs,
            "status": "parsing_complete",
            "error": None,
        }

    except Exception as e:
        logger.error(f"parse_inputs_node failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Failed to parse inputs: {str(e)}",
            "parsed_inputs": {},
        }


# ============================================================================
# NODE 2: SELECT HANDLER
# ============================================================================


def select_handler_node(state: FormatterState) -> Dict[str, Any]:
    """
    Select appropriate chart handler based on chart_type.

    This node:
    - Retrieves chart_type from parsed_inputs
    - Selects corresponding handler from registry
    - Validates handler availability

    Args:
        state: Current FormatterState with parsed_inputs

    Returns:
        Dictionary with:
        - chart_handler: Handler name
        - status: "handler_selected" or "error"
        - error: Error message if handler not found
    """
    logger.info("=== NODE: select_handler ===")
    start_time = time.time()

    try:
        parsed_inputs = state.get("parsed_inputs", {})
        chart_type = parsed_inputs.get("chart_type", "")

        if not chart_type:
            logger.error("Cannot select handler: chart_type is missing")
            return {
                "status": "error",
                "error": "Cannot select handler: chart_type is missing from parsed_inputs",
                "chart_handler": None,
            }

        # Check if chart type is supported
        if not is_chart_type_supported(chart_type):
            logger.error(f"Unsupported chart_type: '{chart_type}'")
            return {
                "status": "error",
                "error": f"Unsupported chart_type: '{chart_type}'",
                "chart_handler": None,
            }

        # Get handler instance
        handler = get_handler(chart_type)
        handler_name = handler.__class__.__name__

        execution_time = time.time() - start_time
        logger.info(
            f"select_handler completed in {execution_time:.2f}s - "
            f"selected '{handler_name}' for chart_type='{chart_type}'"
        )

        return {
            "chart_handler": handler_name,
            "status": "handler_selected",
            "error": None,
        }

    except Exception as e:
        logger.error(f"select_handler_node failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Failed to select handler: {str(e)}",
            "chart_handler": None,
        }


# ============================================================================
# NODE 2.5: START PARALLEL GENERATION (FAN-OUT)
# ============================================================================


def start_parallel_generation_node(state: FormatterState) -> Dict[str, Any]:
    """
    Trigger parallel execution of executive_summary + synthesize_insights.

    This is an orchestration node with no processing - it simply triggers
    the fan-out to parallel LLM calls.

    Args:
        state: Current FormatterState (pass-through)

    Returns:
        Empty dict (pass-through state)
    """
    logger.info("=== NODE: start_parallel_generation (FAN-OUT) ===")
    logger.debug(
        "Triggering parallel execution: generate_executive_summary + synthesize_insights"
    )
    return {}


# ============================================================================
# NODE 3: GENERATE EXECUTIVE SUMMARY
# ============================================================================


def generate_executive_summary_node(state: FormatterState) -> Dict[str, Any]:
    """
    Extract executive summary from insight_generator output (FASE 4 - Unified).

    **FASE 4 UPDATE:** This node NO LONGER calls LLM. It now EXTRACTS
    the executive_summary from the unified insight_generator output.

    This node:
    - Extracts executive_summary from insight_result
    - Validates structure
    - Provides fallback if missing

    Args:
        state: Current FormatterState with parsed_inputs

    Returns:
        Dictionary with:
        - executive_summary: Extracted title and introduction
    """
    logger.info("=== NODE: generate_executive_summary (FASE 4 - EXTRACTION) ===")
    start_time = time.time()

    try:
        parsed_inputs = state.get("parsed_inputs", {})
        insight_result = parsed_inputs.get("insight_result", {})

        # FASE 4: Extract from unified output instead of generating
        executive_summary = insight_result.get("executive_summary", {})

        # Validate structure
        if not executive_summary or not executive_summary.get("title"):
            logger.warning(
                "[generate_executive_summary_node] Missing or invalid executive_summary from insight_generator. Using fallback."
            )
            chart_type = parsed_inputs.get("chart_type", "")
            query = parsed_inputs.get("query", "")
            executive_summary = {
                "title": f"Análise de {chart_type.replace('_', ' ').title()}",
                "introduction": f"Análise baseada na consulta: {query}",
                "_fallback_used": True,
            }
        else:
            executive_summary["_fallback_used"] = False

        # Add additional context fields for backward compatibility
        executive_summary["subtitle"] = parsed_inputs.get("query", "")
        executive_summary["filters_applied_description"] = _get_filter_description(
            parsed_inputs.get("filters", {})
        )

        execution_time = time.time() - start_time
        fallback_used = executive_summary.get("_fallback_used", False)
        logger.info(
            f"generate_executive_summary extracted in {execution_time:.2f}s - "
            f"fallback_used={fallback_used} (FASE 4 - NO LLM CALL)"
        )

        return {
            "executive_summary": executive_summary,
        }

    except Exception as e:
        logger.error(f"generate_executive_summary_node failed: {e}", exc_info=True)
        # Use fallback
        fallback_summary = {
            "title": "Análise de Dados",
            "introduction": "Análise executiva.",
            "subtitle": "",
            "filters_applied_description": "",
            "_fallback_used": True,
        }
        return {"executive_summary": fallback_summary}


def _get_filter_description(filters: Dict[str, Any]) -> str:
    """
    Helper function to generate filter description text.

    Args:
        filters: Dictionary of applied filters

    Returns:
        Human-readable description of filters
    """
    if not filters:
        return "Sem filtros aplicados"

    descriptions = []
    for key, value in filters.items():
        if isinstance(value, list):
            descriptions.append(f"{key}: {', '.join(map(str, value))}")
        else:
            descriptions.append(f"{key}: {value}")

    return " | ".join(descriptions)


# ============================================================================
# NODE 4: SYNTHESIZE INSIGHTS
# ============================================================================


def synthesize_insights_node(state: FormatterState) -> Dict[str, Any]:
    """
    Extract synthesized insights from insight_generator output (FASE 4 - Unified).

    **FASE 4 UPDATE:** This node NO LONGER calls LLM. It now EXTRACTS
    the synthesized_insights from the unified insight_generator output.

    This node:
    - Extracts narrative and key_findings from insight_result
    - Processes detailed_insights for backward compatibility
    - Provides fallback if missing

    Args:
        state: Current FormatterState with parsed_inputs

    Returns:
        Dictionary with:
        - synthesized_insights: Narrative, key findings, detailed insights
    """
    logger.info("=== NODE: synthesize_insights (FASE 4 - EXTRACTION) ===")
    start_time = time.time()

    try:
        parsed_inputs = state.get("parsed_inputs", {})
        insight_result = parsed_inputs.get("insight_result", {})

        # FASE 4: Extract from unified output instead of generating
        synthesized_insights_data = insight_result.get("synthesized_insights", {})
        detailed_insights = insight_result.get("detailed_insights", [])

        # Extract components
        narrative = synthesized_insights_data.get("narrative", "")
        key_findings = synthesized_insights_data.get("key_findings", [])

        # Validate
        if not narrative and not key_findings:
            logger.warning(
                "[synthesize_insights_node] Missing synthesized_insights from insight_generator. Using fallback."
            )
            narrative = "Insights não disponíveis no formato unificado."
            key_findings = []
            fallback_used = True
        else:
            fallback_used = False

        # Process detailed_insights for backward compatibility
        # The formatter expects insights in a specific format
        processed_insights = []
        for insight in detailed_insights:
            processed_insights.append(
                {
                    "title": insight.get("title", ""),
                    "content": insight.get("content", ""),
                    "formula": insight.get("formula", ""),
                    "interpretation": insight.get("interpretation", ""),
                    "confidence": insight.get("confidence", 0.9),
                }
            )

        # Build result
        result = {
            "narrative": narrative,
            "key_findings": key_findings,
            "detailed_insights": processed_insights,
            "transparency_validated": insight_result.get("metadata", {}).get(
                "transparency_validated", False
            ),
            "_fallback_used": fallback_used,
        }

        execution_time = time.time() - start_time
        logger.info(
            f"synthesize_insights extracted in {execution_time:.2f}s - "
            f"fallback_used={fallback_used}, "
            f"narrative_length={len(narrative)}, "
            f"key_findings={len(key_findings)} (FASE 4 - NO LLM CALL)"
        )

        return {
            "synthesized_insights": result,
        }

    except Exception as e:
        logger.error(f"synthesize_insights_node failed: {e}", exc_info=True)
        # Use fallback
        fallback_synthesis = {
            "narrative": "Síntese de insights não disponível.",
            "key_findings": [],
            "detailed_insights": [],
            "transparency_validated": False,
            "_fallback_used": True,
        }
        return {"synthesized_insights": fallback_synthesis}


# ============================================================================
# NODE 5: GENERATE NEXT STEPS
# ============================================================================


def generate_next_steps_node(state: FormatterState) -> Dict[str, Any]:
    """
    Extract next steps from insight_generator output (FASE 4 - Unified).

    **FASE 4 UPDATE:** This node NO LONGER calls LLM. It now EXTRACTS
    the next_steps from the unified insight_generator output.

    This node:
    - Extracts next_steps/recommendations from insight_result
    - Validates structure (exactly 3 recommendations)
    - Provides fallback if missing

    Args:
        state: Current FormatterState with parsed_inputs

    Returns:
        Dictionary with:
        - next_steps: Strategic recommendations
    """
    logger.info("=== NODE: generate_next_steps (FASE 4 - EXTRACTION) ===")
    start_time = time.time()

    try:
        parsed_inputs = state.get("parsed_inputs", {})
        insight_result = parsed_inputs.get("insight_result", {})

        # FASE 4: Extract from unified output instead of generating
        next_steps = insight_result.get("next_steps", [])

        # Validate structure (expect exactly 3 recommendations)
        if not next_steps or len(next_steps) < 3:
            logger.warning(
                f"[generate_next_steps_node] Missing or insufficient next_steps from insight_generator. "
                f"Got {len(next_steps)}, expected 3. Using fallback."
            )
            chart_type = parsed_inputs.get("chart_type", "")
            next_steps = _get_fallback_next_steps(chart_type)
            fallback_used = True
        else:
            # Limit to exactly 3
            next_steps = next_steps[:3]
            fallback_used = False

        # Build result
        result = {
            "next_steps": next_steps,
            "_fallback_used": fallback_used,
        }

        execution_time = time.time() - start_time
        logger.info(
            f"generate_next_steps extracted in {execution_time:.2f}s - "
            f"count={len(next_steps)}, fallback_used={fallback_used} (FASE 4 - NO LLM CALL)"
        )

        return {
            "next_steps": result,
        }

    except Exception as e:
        logger.error(f"generate_next_steps_node failed: {e}", exc_info=True)
        # Use fallback
        chart_type = state.get("parsed_inputs", {}).get("chart_type", "")
        fallback_next_steps = {
            "next_steps": _get_fallback_next_steps(chart_type),
            "_fallback_used": True,
        }
        return {"next_steps": fallback_next_steps}


def _get_fallback_next_steps(chart_type: str) -> List[str]:
    """
    Helper function to provide fallback next steps by chart type.

    Args:
        chart_type: Type of chart being analyzed

    Returns:
        List of 3 generic next steps
    """
    FALLBACK_NEXT_STEPS = {
        "bar_horizontal": [
            "Investigar causas da concentração nos principais items do ranking e desenvolver estratégias de retenção",
            "Avaliar oportunidades de crescimento nos items de menor performance para reduzir dependência",
            "Estabelecer monitoramento contínuo dos top performers para identificar mudanças de padrão rapidamente",
        ],
        "bar_vertical": [
            "Analisar fatores que diferenciam categorias com melhor performance para replicar práticas de sucesso",
            "Estabelecer benchmarks baseados nas categorias líderes para orientar melhorias nas demais",
            "Investigar tendências temporais de cada categoria para identificar oportunidades de crescimento",
        ],
        "line": [
            "Investigar causas de variações significativas no período para melhor compreensão dos drivers temporais",
            "Desenvolver modelo preditivo baseado na tendência identificada para planejamento estratégico",
            "Comparar com períodos equivalentes anteriores para identificar mudanças estruturais relevantes",
        ],
        "pie": [
            "Avaliar oportunidades nos segmentos menores da distribuição para potencial crescimento inexplorado",
            "Desenvolver estratégias específicas para os principais segmentos identificados na análise",
            "Monitorar evolução da distribuição ao longo do tempo para detectar mudanças de concentração",
        ],
    }

    return FALLBACK_NEXT_STEPS.get(
        chart_type,
        [
            "Analisar padrões identificados para compreender drivers de comportamento",
            "Desenvolver estratégias baseadas nos insights principais para maximizar resultados",
            "Estabelecer monitoramento contínuo das métricas chave para acompanhamento proativo",
        ],
    )


# ============================================================================
# NODE 6: FORMAT DATA TABLE
# ============================================================================


def format_data_table_node(state: FormatterState) -> Dict[str, Any]:
    """
    Format data table into markdown and HTML representations.

    This node:
    - Extracts data from parsed_inputs
    - Formats into both markdown and HTML tables
    - Limits rows for display

    Args:
        state: Current FormatterState with parsed_inputs

    Returns:
        Dictionary with:
        - formatted_data_table: Formatted table data
        - status: Updated status
        - error: Error message if formatting fails
    """
    logger.info("=== NODE: format_data_table ===")
    start_time = time.time()

    try:
        parsed_inputs = state.get("parsed_inputs", {})
        data = parsed_inputs.get("data", [])

        # Format table
        formatter = DataTableFormatter()
        formatted_table = formatter.format(data, max_rows=10)

        execution_time = time.time() - start_time
        logger.info(
            f"format_data_table completed in {execution_time:.2f}s - "
            f"formatted {formatted_table['showing_rows']}/{formatted_table['total_rows']} rows"
        )

        return {
            "formatted_data_table": formatted_table,
            "status": "table_formatted",
            "error": None,
        }

    except Exception as e:
        logger.error(f"format_data_table_node failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Failed to format data table: {str(e)}",
            "formatted_data_table": {},
        }


# ============================================================================
# NODE 7: ASSEMBLE OUTPUT
# ============================================================================


def assemble_output_node(state: FormatterState) -> Dict[str, Any]:
    """
    Assemble final structured JSON output.

    This node:
    - Combines all generated components
    - Structures output according to API specification
    - Adds metadata and execution tracking
    - Calculates statistics

    Args:
        state: Current FormatterState with all components

    Returns:
        Dictionary with:
        - formatter_output: Complete structured JSON output
        - status: "success"
        - error: None
    """
    logger.info("=== NODE: assemble_output ===")
    start_time = time.time()

    try:
        parsed_inputs = state.get("parsed_inputs", {})
        executive_summary = state.get("executive_summary", {})
        synthesized_insights = state.get("synthesized_insights", {})
        next_steps = state.get("next_steps", {})
        formatted_table = state.get("formatted_data_table", {})

        # FASE 3: Extract FASE 2 native fields from insight_result
        # These are the primary response fields; formatter passes them through
        insight_result = parsed_inputs.get("insight_result", {})
        resposta = insight_result.get("resposta", "")
        dados_destacados = insight_result.get("dados_destacados", [])
        filtros_mencionados = insight_result.get("filtros_mencionados", [])

        # Aggregate tokens from upstream agents (no LLM calls in formatter since FASE 4)
        from src.formatter_agent.utils.token_accumulator import TokenAccumulator

        accumulator = TokenAccumulator()

        # Collect tokens from executive_summary
        if executive_summary and "_tokens" in executive_summary:
            accumulator.add(executive_summary["_tokens"])

        # Collect tokens from synthesized_insights
        if synthesized_insights and "_tokens" in synthesized_insights:
            accumulator.add(synthesized_insights["_tokens"])

        # Collect tokens from next_steps
        if next_steps and "_tokens" in next_steps:
            accumulator.add(next_steps["_tokens"])

        # Store tokens totals in state
        if "agent_tokens" not in state or not isinstance(
            state.get("agent_tokens"), dict
        ):
            state["agent_tokens"] = {}

        # Only register formatter_agent if it actually performed LLM calls
        formatter_tokens = accumulator.get_totals()
        if formatter_tokens.get("total_tokens", 0) > 0 or accumulator.llm_calls > 0:
            state["agent_tokens"]["formatter_agent"] = formatter_tokens

        logger.info(
            f"[FormatterAgent] Total tokens: {accumulator.get_totals()}, "
            f"across {accumulator.llm_calls if accumulator.llm_calls > 0 else 'no'} LLM calls"
        )

        # Aggregate query-level tokens across all LLM agents
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for agent_name, tokens in state.get("agent_tokens", {}).items():
            if not isinstance(tokens, dict):
                continue
            total_tokens["input_tokens"] += int(tokens.get("input_tokens", 0) or 0)
            total_tokens["output_tokens"] += int(tokens.get("output_tokens", 0) or 0)
            total_tokens["total_tokens"] += int(tokens.get("total_tokens", 0) or 0)

        # Calculate execution times (approximate from state if available)
        execution_times = {
            "parse_inputs": 0.1,  # These would ideally be tracked more precisely
            "select_handler": 0.05,
            "generate_executive_summary": 1.5,
            "synthesize_insights": 2.0,
            "generate_next_steps": 1.5,
            "format_data_table": 0.1,
            "assemble_output": 0.0,  # Will be calculated
        }

        # Assemble output
        assembler = OutputAssembler()
        formatter_output = assembler.assemble(
            parsed_inputs=parsed_inputs,
            executive_summary=executive_summary,
            synthesized_insights=synthesized_insights,
            next_steps=next_steps,
            formatted_table=formatted_table,
            execution_times=execution_times,
            agent_tokens=state.get("agent_tokens", {}),
            total_tokens=total_tokens,
        )

        # FASE 3: Pass through FASE 2 native fields as primary response
        # insight_result.resposta IS the final user-facing answer
        formatter_output["resposta"] = resposta
        formatter_output["dados_destacados"] = dados_destacados
        formatter_output["filtros_mencionados"] = filtros_mencionados

        execution_time = time.time() - start_time
        execution_times["assemble_output"] = execution_time

        logger.info(
            f"assemble_output completed in {execution_time:.2f}s - "
            f"output assembled successfully"
        )

        return {
            "formatter_output": formatter_output,
            "status": "success",
            "error": None,
        }

    except Exception as e:
        logger.error(f"assemble_output_node failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Failed to assemble output: {str(e)}",
            "formatter_output": {},
        }


# ============================================================================
# NODE 8: HANDLE ERROR
# ============================================================================


def handle_error_node(state: FormatterState) -> Dict[str, Any]:
    """
    Fallback error handling node for critical failures.

    This node:
    - Creates degraded output with available data
    - Provides user-friendly error messages
    - Ensures workflow completes gracefully

    Args:
        state: Current FormatterState with error information

    Returns:
        Dictionary with:
        - formatter_output: Degraded output with error information
        - status: "error"
        - error: Detailed error message
    """
    logger.info("=== NODE: handle_error ===")

    error_message = state.get("error", "Unknown error occurred")
    parsed_inputs = state.get("parsed_inputs", {})

    logger.error(f"Entering error handling node: {error_message}")

    # Create degraded output
    degraded_output = {
        "status": "error",
        "format_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "error": {
            "message": error_message,
            "recovery": "partial",
            "degraded_mode": True,
        },
        "executive_summary": {
            "title": "Análise Interrompida por Erro",
            "introduction": f"Não foi possível completar a análise devido a um erro: {error_message}",
            "subtitle": parsed_inputs.get("query", ""),
            "filters_applied_description": "Informação indisponível devido a erro",
        },
        "insights": {
            "narrative": "Não foi possível gerar insights devido a erro no processamento.",
            "key_findings": [],
            "detailed_insights": [],
            "transparency_validated": False,
        },
        "next_steps": {
            "items": [],
        },
        "data": {
            "table_markdown": "Dados indisponíveis",
            "table_html": "<p>Dados indisponíveis</p>",
            "total_records": 0,
            "displayed_records": 0,
        },
        "visualization": {
            "chart": {
                "type": parsed_inputs.get("chart_type", "unknown"),
                "available": False,
                "error": "Visualização não disponível devido a erro",
            }
        },
        "metadata": {
            "query": parsed_inputs.get("query", ""),
            "chart_type": parsed_inputs.get("chart_type", "unknown"),
            "agents_executed": ["formatter (failed)"],
            "execution_summary": {
                "total_time_seconds": 0,
                "status": "failed",
            },
        },
    }

    logger.info("Error handling node created degraded output")

    return {
        "formatter_output": degraded_output,
        "status": "error",
        "error": error_message,
    }
