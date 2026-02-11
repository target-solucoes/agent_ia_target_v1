"""
LangGraph workflow definition for the Insight Generator.

This module defines the complete workflow graph with all nodes and edges
that orchestrate the insight generation pipeline.

FASE 3 Simplified Workflow (4 nodes):
    START → parse_input → build_prompt → invoke_llm → format_output → END

    parse_input now includes metric calculation (formerly calculate_metrics).
    format_output now includes validation and markdown transform
    (formerly validate_insights + transform_to_markdown + format_output).
"""

import logging
from typing import Optional
from langgraph.graph import StateGraph, END

from .state import InsightState
from .nodes import (
    parse_input_node,
    build_prompt_node,
    invoke_llm_node,
    format_output_node,
)
from .router import should_continue, should_invoke_llm
from ..core.settings import INSIGHT_PROMPT_MODE
from .integration import dynamic_build_prompt_node

logger = logging.getLogger(__name__)


def create_insight_generator_workflow(
    verbose: bool = False,
    use_conditional_edges: bool = False,
    prompt_mode: Optional[str] = None,
) -> StateGraph:
    """
    Create and compile the Insight Generator LangGraph workflow.

    FASE 3 Simplified Pipeline (4 nodes):
        1. parse_input: Extract/validate input + calculate metrics (inline)
        2. build_prompt: Format metrics into LLM prompt
        3. invoke_llm: Generate insights using Gemini (dynamic model selection)
        4. format_output: Validate response, transform, assemble final output

    Args:
        verbose: If True, enable verbose logging for debugging
        use_conditional_edges: If True, use conditional routing for error handling
        prompt_mode: Override for prompt mode ("legacy" or "dynamic")

    Returns:
        Compiled StateGraph ready for execution via invoke()
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    prompt_mode_effective = (prompt_mode or INSIGHT_PROMPT_MODE or "legacy").lower()
    logger.info(
        "Creating Insight Generator workflow (FASE 3, prompt_mode=%s)",
        prompt_mode_effective,
    )

    # Initialize StateGraph with InsightState schema
    workflow = StateGraph(InsightState)

    # ========== Add Nodes (4 nodes - FASE 3) ==========
    logger.debug("Adding workflow nodes (FASE 3 simplified)")

    workflow.add_node("parse_input", parse_input_node)

    if prompt_mode_effective == "dynamic":
        workflow.add_node("build_prompt", dynamic_build_prompt_node)
    else:
        workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("invoke_llm", invoke_llm_node)
    workflow.add_node("format_output", format_output_node)

    # ========== Define Entry Point ==========
    workflow.set_entry_point("parse_input")

    # ========== Add Edges ==========
    logger.debug("Adding workflow edges")

    if use_conditional_edges:
        # Advanced workflow with conditional routing
        logger.info("Using conditional edge routing (FASE 3)")

        # parse_input → check errors → build_prompt or end
        workflow.add_conditional_edges(
            "parse_input",
            should_continue,
            {"continue": "build_prompt", "end": "format_output"},
        )

        # build_prompt → check if should invoke LLM → invoke_llm or skip
        workflow.add_conditional_edges(
            "build_prompt",
            should_invoke_llm,
            {"invoke": "invoke_llm", "skip": "format_output"},
        )

        # invoke_llm → format_output
        workflow.add_edge("invoke_llm", "format_output")

        # format_output → END
        workflow.add_edge("format_output", END)

    else:
        # Simple linear workflow (default)
        logger.info("Using linear edge routing (FASE 3)")

        workflow.add_edge("parse_input", "build_prompt")
        workflow.add_edge("build_prompt", "invoke_llm")
        workflow.add_edge("invoke_llm", "format_output")
        workflow.add_edge("format_output", END)

    logger.info("Workflow created successfully (4 nodes)")

    # Compile and return
    return workflow.compile()


def visualize_workflow(output_path: Optional[str] = None) -> str:
    """
    Generate Mermaid diagram of the workflow.

    Args:
        output_path: Optional file path to save the diagram.
                    If None, returns the Mermaid string.

    Returns:
        Mermaid diagram string

    Example:
        >>> mermaid = visualize_workflow("workflow_diagram.mmd")
        >>> # Diagram saved to workflow_diagram.mmd
    """
    workflow = create_insight_generator_workflow()

    try:
        # Get Mermaid representation
        mermaid_graph = workflow.get_graph().draw_mermaid()

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(mermaid_graph)
            logger.info(f"Workflow diagram saved to: {output_path}")

        return mermaid_graph

    except Exception as e:
        logger.error(f"Failed to generate workflow diagram: {e}")
        raise


def execute_workflow(
    chart_spec: dict,
    analytics_result: dict,
    verbose: bool = False,
    use_conditional_edges: bool = False,
) -> dict:
    """
    Convenience function to execute the workflow with inputs.

    Args:
        chart_spec: Chart specification from graphic_classifier
        analytics_result: Analytics output from analytics_executor
        verbose: Enable verbose logging
        use_conditional_edges: Use conditional routing for error handling

    Returns:
        Final output dictionary with insights and metadata

    Example:
        >>> result = execute_workflow(chart_spec, analytics_result)
        >>> if result["status"] == "success":
        ...     for insight in result["insights"]:
        ...         print(f"{insight['title']}: {insight['content']}")
    """
    from .nodes import initialize_state

    # Create workflow
    workflow = create_insight_generator_workflow(
        verbose=verbose, use_conditional_edges=use_conditional_edges
    )

    # Initialize state
    initial_state = initialize_state(chart_spec, analytics_result)

    try:
        # Execute workflow
        logger.info("Executing Insight Generator workflow")
        final_state = workflow.invoke(initial_state)

        # DEBUG: Log agent_tokens presence
        if "agent_tokens" in final_state:
            logger.info(
                f"[execute_workflow] agent_tokens found in final_state: {list(final_state['agent_tokens'].keys())}"
            )
        else:
            logger.warning("[execute_workflow] agent_tokens NOT found in final_state")

        # Extract final output
        output = final_state.get("final_output", {})

        if not output:
            logger.warning("Workflow completed but no final_output generated")
            return {
                "status": "error",
                "error": "No output generated",
                "insights": [],
                "metadata": {
                    "calculation_time": 0.0,
                    "metrics_count": 0,
                    "llm_model": "gemini-2.5-flash",
                    "timestamp": "",
                    "transparency_validated": False,
                    "pipeline_version": "fase_3",
                },
            }

        # CRITICAL: Include agent_tokens from final_state for token tracking
        if "agent_tokens" in final_state:
            output["_agent_tokens"] = final_state["agent_tokens"]

        logger.info(f"Workflow completed with status: {output.get('status')}")
        return output

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "insights": [],
            "metadata": {
                "calculation_time": 0.0,
                "metrics_count": 0,
                "llm_model": "gemini-2.5-flash",
                "timestamp": "",
                "transparency_validated": False,
                "pipeline_version": "fase_3",
            },
        }


# Convenience alias for backward compatibility
create_workflow = create_insight_generator_workflow
