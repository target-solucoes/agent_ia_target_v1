"""Graph module for insight_generator LangGraph workflow.

FASE 3: Simplified to 4-node pipeline:
    parse_input → build_prompt → invoke_llm → format_output
"""

from .state import InsightState
from .workflow import (
    create_insight_generator_workflow,
    create_workflow,
    visualize_workflow,
    execute_workflow,
)
from .nodes import (
    parse_input_node,
    calculate_metrics_node,  # backward compat (now inlined in parse_input)
    build_prompt_node,
    invoke_llm_node,
    validate_insights_node,  # backward compat (now inlined in format_output)
    format_output_node,
    initialize_state,
)
from .router import (
    route_by_chart_type,
    should_continue,
)

__all__ = [
    # State
    "InsightState",
    # Workflow
    "create_insight_generator_workflow",
    "create_workflow",
    "visualize_workflow",
    "execute_workflow",
    # Nodes (active in FASE 3 pipeline)
    "parse_input_node",
    "build_prompt_node",
    "invoke_llm_node",
    "format_output_node",
    "initialize_state",
    # Nodes (backward compat - no longer in workflow graph)
    "calculate_metrics_node",
    "validate_insights_node",
    # Router
    "route_by_chart_type",
    "should_continue",
]
