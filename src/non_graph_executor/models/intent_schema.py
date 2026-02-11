"""
Pydantic schemas for LLM-based intent analysis.

This module defines the structured output models used by the IntentAnalyzer
to represent the user's query intent, including column specifications,
aggregation details, grouping dimensions, and ordering.

These schemas replace the keyword-based classification approach with a
semantic, LLM-driven understanding of what the user actually wants.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class ColumnSpec(BaseModel):
    """
    Especificação de uma coluna referenciada na query.

    Representa tanto colunas físicas do dataset quanto colunas virtuais
    (derivadas de transformações, como Ano → YEAR("Data")).

    Attributes:
        name: Nome real da coluna ou nome virtual (ex: "Valor_Vendido", "Ano")
        is_virtual: Se a coluna requer transformação SQL (ex: Ano → YEAR("Data"))
        expression: Expressão SQL correspondente, se virtual
        alias: Nome de exibição para o resultado (ex: "total_vendas", "Ano")

    Example:
        >>> col = ColumnSpec(name="Ano", is_virtual=True,
        ...     expression='YEAR("Data")', alias="Ano")
    """

    name: str = Field(description="Nome real da coluna ou nome virtual")
    is_virtual: bool = Field(
        default=False,
        description="Se a coluna requer transformação SQL (ex: Ano → YEAR(Data))",
    )
    expression: Optional[str] = Field(
        default=None,
        description="Expressão SQL equivalente se coluna virtual",
    )
    alias: Optional[str] = Field(
        default=None,
        description="Alias para exibição no resultado",
    )


class AggregationSpec(BaseModel):
    """
    Especificação de uma operação de agregação.

    Define a função de agregação a ser aplicada sobre uma coluna,
    incluindo suporte a DISTINCT e alias personalizado.

    Attributes:
        function: Função de agregação (sum, avg, count, min, max, median, std)
        column: Coluna sobre a qual aplicar a agregação
        distinct: Se deve usar DISTINCT (ex: COUNT(DISTINCT col))
        alias: Nome de exibição para o resultado agregado

    Example:
        >>> agg = AggregationSpec(
        ...     function="sum",
        ...     column=ColumnSpec(name="Valor_Vendido"),
        ...     alias="total_vendas"
        ... )
    """

    function: Literal["sum", "avg", "count", "min", "max", "median", "std"] = Field(
        description="Função de agregação a aplicar"
    )
    column: ColumnSpec = Field(description="Coluna alvo da agregação")
    distinct: bool = Field(
        default=False,
        description="Se deve usar DISTINCT na agregação",
    )
    alias: Optional[str] = Field(
        default=None,
        description="Alias para o resultado da agregação",
    )


class OrderSpec(BaseModel):
    """
    Especificação de ordenação de resultados.

    Attributes:
        column: Nome da coluna ou alias de agregação para ordenação
        direction: Direção da ordenação (ASC ou DESC)

    Example:
        >>> order = OrderSpec(column="total_vendas", direction="DESC")
    """

    column: str = Field(description="Coluna ou alias para ordenação")
    direction: Literal["ASC", "DESC"] = Field(
        default="DESC",
        description="Direção da ordenação",
    )


class QueryIntent(BaseModel):
    """
    Especificação completa da intenção do usuário.

    Este modelo captura a compreensão semântica total de uma query,
    incluindo tipo de intenção, colunas de seleção, agregações,
    dimensões de agrupamento, ordenação e filtros adicionais.

    Tipos de intenção:
        - simple_aggregation: Agregação sem agrupamento (ex: "total de vendas")
        - grouped_aggregation: Agregação com GROUP BY (ex: "vendas por mês")
        - ranking: Top N com ordenação (ex: "top 5 estados por vendas")
        - temporal_analysis: Consulta sobre dimensão temporal (ex: "último ano")
        - comparison: Comparação entre grupos (ex: "SP vs RJ")
        - lookup: Busca de registro específico (ex: "dados do cliente X")
        - metadata: Informação sobre o dataset (ex: "quantas linhas")
        - tabular: Dados brutos tabulares (ex: "mostre a tabela")
        - conversational: Saudações ou ajuda (ex: "olá")

    Attributes:
        intent_type: Tipo de intenção identificada
        select_columns: Colunas que o usuário quer visualizar
        aggregations: Operações de agregação a aplicar
        group_by: Dimensões de agrupamento (GROUP BY)
        order_by: Especificação de ordenação (ORDER BY)
        limit: Limite de resultados (LIMIT)
        additional_filters: Filtros detectados na própria query
        confidence: Nível de confiança na interpretação (0.0 a 1.0)
        reasoning: Explicação do raciocínio do LLM (para debug/logging)

    Example:
        >>> intent = QueryIntent(
        ...     intent_type="grouped_aggregation",
        ...     select_columns=[],
        ...     aggregations=[AggregationSpec(
        ...         function="sum",
        ...         column=ColumnSpec(name="Valor_Vendido"),
        ...         alias="total_vendas"
        ...     )],
        ...     group_by=[ColumnSpec(name="Mes", is_virtual=True,
        ...         expression='MONTH("Data")', alias="Mes")],
        ...     order_by=OrderSpec(column="total_vendas", direction="DESC"),
        ...     limit=1,
        ...     confidence=0.95,
        ...     reasoning="Usuário quer o mês com maior total de vendas"
        ... )
    """

    intent_type: Literal[
        "simple_aggregation",
        "grouped_aggregation",
        "ranking",
        "temporal_analysis",
        "comparison",
        "lookup",
        "metadata",
        "tabular",
        "conversational",
    ] = Field(description="Tipo de intenção identificada")

    # Colunas de seleção (o que o usuário quer ver além de agregações)
    select_columns: List[ColumnSpec] = Field(
        default_factory=list,
        description="Colunas de seleção direta (sem agregação)",
    )

    # Agregações a aplicar
    aggregations: List[AggregationSpec] = Field(
        default_factory=list,
        description="Operações de agregação a aplicar",
    )

    # Dimensões de agrupamento
    group_by: List[ColumnSpec] = Field(
        default_factory=list,
        description="Dimensões de agrupamento (GROUP BY)",
    )

    # Ordenação
    order_by: Optional[OrderSpec] = Field(
        default=None,
        description="Especificação de ordenação (ORDER BY)",
    )

    # Limite de resultados
    limit: Optional[int] = Field(
        default=None,
        description="Limite de resultados (LIMIT), usado em rankings",
    )

    # Filtros adicionais detectados na query (além dos do filter_classifier)
    additional_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filtros adicionais detectados na query do usuário",
    )

    # Confiança na interpretação
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Nível de confiança na interpretação (0.0 a 1.0)",
    )

    # Explicação do raciocínio (para debug)
    reasoning: str = Field(
        default="",
        description="Explicação do raciocínio do LLM sobre a intenção identificada",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "intent_type": "simple_aggregation",
                    "aggregations": [
                        {
                            "function": "max",
                            "column": {
                                "name": "Ano",
                                "is_virtual": True,
                                "expression": 'YEAR("Data")',
                                "alias": "Ano",
                            },
                        }
                    ],
                    "confidence": 0.95,
                    "reasoning": "Usuário quer saber o último ano com vendas → MAX(YEAR(Data))",
                },
                {
                    "intent_type": "grouped_aggregation",
                    "aggregations": [
                        {
                            "function": "sum",
                            "column": {"name": "Valor_Vendido"},
                            "alias": "total_vendas",
                        }
                    ],
                    "group_by": [
                        {
                            "name": "Mes",
                            "is_virtual": True,
                            "expression": 'MONTH("Data")',
                            "alias": "Mes",
                        }
                    ],
                    "order_by": {"column": "total_vendas", "direction": "DESC"},
                    "limit": 1,
                    "confidence": 0.95,
                    "reasoning": "Usuário quer o mês com maior valor de vendas",
                },
            ]
        }
    )
