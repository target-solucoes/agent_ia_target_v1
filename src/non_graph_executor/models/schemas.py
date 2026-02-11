"""
Pydantic schemas for non_graph_executor.

This module defines the data models for non-graph query processing,
including output schemas and query classification results.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


def _default_token_usage() -> Dict[str, int]:
    """Return default token usage payload."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class NonGraphOutput(BaseModel):
    """
    Schema principal de output do non_graph_executor.

    Este schema define a estrutura de resposta para queries não-gráficas,
    incluindo dados, metadados, métricas de performance e informações de erro.

    Attributes:
        status: Status da execução (success, error, partial)
        query_type: Tipo de query processada
        data: Dados retornados pela query (opcional)
        summary: Resumo textual do resultado (opcional)
        metadata: Metadados da execução (row_count, columns, dtypes, etc)
        performance_metrics: Métricas de performance (tempos, cache hit, etc)
        conversational_response: Resposta para queries conversacionais (opcional)
        error: Informações de erro se status != success (opcional)

    Example:
        >>> output = NonGraphOutput(
        ...     status="success",
        ...     query_type="aggregation",
        ...     data=[{"avg_vendas": 15420.50}],
        ...     summary="A média de vendas é R$ 15.420,50",
        ...     metadata={"row_count": 1, "execution_time": 0.245},
        ...     performance_metrics={"total_time": 0.287}
        ... )
    """

    status: Literal["success", "error", "partial"]
    query_type: Literal[
        "metadata",
        "aggregation",
        "lookup",
        "textual",
        "statistical",
        "conversational",
        "tabular",
    ]

    # Dados principais
    data: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None

    # Metadados da execução
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Contém: row_count, columns, dtypes, execution_time, engine, filters_applied

    # Métricas de performance
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    # Contém: total_time, classification_time, execution_time, llm_time, cache_hit

    # Resposta conversacional
    conversational_response: Optional[str] = None

    # Informações de erro
    error: Optional[Dict[str, str]] = None

    # Rastreamento de tokens
    total_tokens: Dict[str, int] = Field(default_factory=_default_token_usage)
    agent_tokens: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "query_type": "aggregation",
                "data": [{"avg_vendas": 15420.50}],
                "summary": "A média de vendas é R$ 15.420,50",
                "metadata": {
                    "row_count": 1,
                    "execution_time": 0.245,
                    "engine": "DuckDB",
                    "filters_applied": {},
                },
                "performance_metrics": {
                    "total_time": 0.287,
                    "classification_time": 0.012,
                    "execution_time": 0.245,
                    "llm_time": 0.030,
                },
            }
        }
    )


class QueryTypeClassification(BaseModel):
    """
    Resultado da classificação de query.

    Este schema define o resultado da classificação de uma query não-gráfica,
    incluindo o tipo identificado, nível de confiança e parâmetros extraídos.

    Attributes:
        query_type: Tipo de query classificada
        confidence: Nível de confiança da classificação (0.0 a 1.0)
        requires_llm: Se a query precisa usar LLM para processamento
        parameters: Parâmetros específicos extraídos da query

    Parameter Fields por Tipo de Query:
        - metadata: metadata_type (row_count, column_list, dtypes, sample_rows),
                   n (para sample), column (para unique_values)
        - aggregation: column, aggregation (sum/avg/etc), filters
        - lookup: lookup_column, lookup_value, return_columns
        - textual: column, search_term, case_sensitive
        - statistical: column, filters
        - tabular: limit (default 100)
        - conversational: {} (vazio)

    Example:
        >>> classification = QueryTypeClassification(
        ...     query_type="aggregation",
        ...     confidence=0.85,
        ...     requires_llm=True,
        ...     parameters={
        ...         "column": "Valor_Vendido",
        ...         "aggregation": "avg",
        ...         "filters": {"Ano": 2015}
        ...     }
        ... )
    """

    query_type: Literal[
        "metadata",
        "aggregation",
        "lookup",
        "textual",
        "statistical",
        "conversational",
        "tabular",
    ]
    subtype: Optional[str] = Field(
        default=None,
        description="Subtipo específico (ex: sample_rows, row_count para metadata)",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    requires_llm: bool
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # QueryIntent from IntentAnalyzer (Phase 2) — carries full semantic analysis
    # for downstream components (DynamicQueryBuilder in Phase 3).
    # Typed as Any to avoid circular imports; at runtime this will be a QueryIntent instance.
    intent: Optional[Any] = Field(
        default=None,
        description="Full semantic intent from IntentAnalyzer (when available)",
        exclude=True,  # Excluded from JSON serialization
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_type": "aggregation",
                "confidence": 0.85,
                "requires_llm": True,
                "parameters": {
                    "column": "Valor_Vendido",
                    "aggregation": "avg",
                    "filters": {"Ano": 2015},
                },
            }
        }
    )
