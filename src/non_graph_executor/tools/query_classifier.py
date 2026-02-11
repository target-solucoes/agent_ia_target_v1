"""
Query classifier for non_graph_executor.

This module implements query classification using a hybrid approach:
1. Fast pre-filter for conversational queries (no LLM needed)
2. LLM-based IntentAnalyzer for semantic understanding of all other queries
3. Legacy keyword-based fallback (configurable)

Phase 2 refactoring: The IntentAnalyzer replaces keyword-based classification
for non-conversational queries, enabling proper handling of GROUP BY,
ORDER BY, temporal functions, and rankings.
"""

import logging
import json
import re
from typing import Dict, Optional, Any

from src.non_graph_executor.models.schemas import QueryTypeClassification
from src.non_graph_executor.models.intent_schema import (
    QueryIntent,
    ColumnSpec,
    AggregationSpec,
)
from src.non_graph_executor.tools.intent_analyzer import IntentAnalyzer
from src.non_graph_executor.tools.query_classifier_params import ParameterExtractor

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classificador de queries não-gráficas.

    Estratégia de classificação:
    1. Detecção rápida via keywords (prioridade alta)
    2. Uso de LLM apenas para casos ambíguos
    3. Extração de parâmetros específicos por tipo

    Ordem de prioridade:
    tabular > conversational > metadata > aggregation >
    statistical > lookup > textual > LLM fallback
    """

    # ========================================================================
    # KEYWORDS POR CATEGORIA (Estratégia escalável e contextual)
    # ========================================================================

    # TABULAR: Solicitação explícita de visualização de dados em formato tabela
    TABULAR_KEYWORDS = [
        "mostrar tabela",
        "mostre a tabela",
        "mostre tabela",
        "exibir tabela",
        "exiba tabela",
        "ver tabela",
        "veja tabela",
        "dados brutos",
        "tabela completa",
        "tabela de dados",
        "todos os dados",
        "ver todos os dados",
        "mostrar registros",
        "mostre registros",
        "exibir registros",
        "ver registros",
        "listar registros",
        "mostre os dados",
    ]

    # METADATA: Perguntas sobre estrutura e meta-informações do dataset
    # (Excluindo queries com termos de negócio que indicam agregação)
    METADATA_KEYWORDS = [
        "quantas linhas",
        "quantas colunas",
        "quantos registros",
        "número de registros",
        "total de linhas",
        "total de registros",
        "total de colunas",
        "quais colunas",
        "quais são as colunas",
        "quais os tipos",
        "quais campos",
        "liste as colunas",
        "lista de colunas",
        "tipos de dados",
        "tipo das colunas",
        "mostre os tipos",
        "valores únicos de",
        "valores únicos tem",
        "valores unicos de",
        "valores unicos tem",
        "quantos valores únicos",
        "quantos valores unicos",
        "valores nulos",
        "primeiras linhas",
        "últimas linhas",
        "mostre linhas",
        "mostre 5 linhas",
        "mostre 10 linhas",
        "mostre algumas linhas",
        "mostrar linhas",
        "mostrar 5 linhas",
        "mostrar 10 linhas",
        "mostre os campos",
        "amostra",
        "sample",
        "shape",
        "estrutura",
        "schema",
        "número de linhas",
        "número de colunas",
        "valores distintos de",
        "distinct count de",
        "distinct count",
        "linhas de exemplo",
        "exemplos de dados",
        "exemplos",
        "preview dos dados",
        "preview",
        "quantos registros tem",
    ]

    # AGGREGATION: Operações de agregação (soma, média, contagem, min, max, etc)
    # Palavras específicas que indicam cálculos agregados
    AGGREGATION_PATTERNS = [
        # AVG patterns
        ("média", None),
        ("media", None),
        ("average", None),
        ("valor médio", None),
        ("valor medio", None),
        # SUM patterns
        ("soma", None),
        ("somatório", None),
        ("somatorio", None),
        (
            "total de",
            ["vendas", "valor", "quantidade", "pedidos", "clientes"],
        ),  # "total de [business]" is aggregation
        ("qual o total", None),  # "qual o total de vendas"
        ("qual a total", None),
        # COUNT patterns
        (
            "quantos",
            ["clientes", "produtos", "pedidos", "vendas"],
        ),  # Must have business context
        ("quantas", ["vendas", "compras", "transações", "transacoes"]),
        ("número de", ["clientes", "produtos", "pedidos", "vendas"]),
        ("numero de", ["clientes", "produtos", "pedidos", "vendas"]),
        ("count de", None),
        ("count", None),
        ("contagem de", None),
        # MIN/MAX patterns
        ("menor", None),
        ("mínimo", None),
        ("minimo", None),
        ("qual o menor", None),
        ("qual a menor", None),
        ("maior", None),
        ("máximo", None),
        ("maximo", None),
        ("qual o maior", None),
        ("qual a maior", None),
        # MEDIAN patterns
        ("mediana", None),
        ("median", None),
        ("valor mediano", None),
    ]

    # STATISTICAL: Análises estatísticas avançadas (desvio, variância, quartis, etc)
    # Diferente de aggregation simples
    STATISTICAL_KEYWORDS = [
        "estatísticas",
        "estatisticas",
        "resumo estatístico",
        "resumo estatistico",
        "análise estatística",
        "analise estatistica",
        "quartis",
        "quartil",
        "variância",
        "variancia",
        "desvio padrão",
        "desvio padrao",
        "desvio-padrão",
        "iqr",
        "percentil",
        "percentis",
        "distribuição",
        "distribuicao",
        "q1",
        "q3",
        "std de",
        "variance",
    ]

    # TEXTUAL: Listagens e buscas textuais
    # Diferente de TABULAR (que mostra tudo) e LOOKUP (que busca registro específico)
    TEXTUAL_PATTERNS = [
        ("liste todos", None),  # "liste todos os X"
        ("listar todos", None),
        ("liste os", None),  # "liste os X"
        ("listar os", None),
        ("mostre todos os", None),  # "mostre todos os X" (quando não é tabela)
        ("mostrar todos os", None),
        ("contém", None),
        ("contem", None),
        ("que contém", None),
        ("que contem", None),
        ("buscar texto", None),
        ("procurar por", ["texto", "palavra", "nome"]),  # Busca textual específica
    ]

    # LOOKUP: Busca de registro específico por ID/código
    # Diferente de AGGREGATION (min/max)
    LOOKUP_PATTERNS = [
        ("cliente", ["123", "abc", "xyz"]),  # Indica ID específico
        ("pedido", ["123", "abc", "xyz"]),
        ("produto", ["123", "abc", "xyz"]),
        ("detalhes do", None),
        ("dados do", ["cliente", "pedido", "produto"]),
        ("informações do", None),
        ("informacoes do", None),
        ("registro", ["123", "abc", "xyz"]),
    ]

    # Business keywords para excluir conversationais
    # Qualquer termo de negócio/domínio indica que não é conversacional
    BUSINESS_KEYWORDS = [
        "vendas",
        "clientes",
        "produtos",
        "produto",
        "cliente",
        "valor",
        "valores",
        "quantidade",
        "quantidades",
        "tabela",
        "dados",
        "pedidos",
        "pedido",
        "faturamento",
        "receita",
        "preço",
        "preco",
        "preços",
        "precos",
        "peso",
        "qtd",
        "empresa",
        "filial",
        "matriz",
        "estado",
        "uf",
        "cidade",
        "data",
        "ano",
        "mes",
        "mês",
    ]

    # Saudações simples
    GREETINGS = ["oi", "olá", "ola", "hello", "hi", "bom dia", "boa tarde", "boa noite"]

    def __init__(
        self,
        alias_mapper,
        llm,
        intent_analyzer: Optional[IntentAnalyzer] = None,
        use_intent_analyzer: bool = True,
    ):
        """
        Initialize query classifier.

        Args:
            alias_mapper: AliasMapper instance for column resolution
            llm: LLM instance for ambiguous cases (legacy fallback)
            intent_analyzer: IntentAnalyzer instance for semantic comprehension
                (Phase 2). If None and use_intent_analyzer=True, one will be
                created using the provided llm and alias_mapper.
            use_intent_analyzer: Whether to use the IntentAnalyzer for
                semantic classification. Set to False to use legacy
                keyword-based classification only.
        """
        self.alias_mapper = alias_mapper
        self.llm = llm
        self.use_intent_analyzer = use_intent_analyzer
        self.intent_analyzer = intent_analyzer

        if self.use_intent_analyzer and self.intent_analyzer is None:
            logger.info(
                "QueryClassifier: No IntentAnalyzer provided, will be set externally"
            )

        mode = (
            "IntentAnalyzer (semantic)"
            if use_intent_analyzer
            else "keyword-based (legacy)"
        )
        logger.info(f"QueryClassifier initialized with {mode} classification")

    def classify(self, query: str, state: Dict) -> QueryTypeClassification:
        """
        Classifica query usando estratégia híbrida.

        Fluxo de classificação (Phase 2):
        1. PRE-FILTRO RÁPIDO: Detecta queries conversacionais sem LLM
        2. INTENT ANALYZER (se habilitado): Compreensão semântica via LLM
        3. LEGACY FALLBACK: Classificação keyword-based (se IntentAnalyzer falhar
           ou estiver desabilitado)

        O IntentAnalyzer produz um QueryIntent com informação dimensional
        completa (group_by, order_by, limit, etc.), que é mapeado para o
        QueryTypeClassification existente para manter compatibilidade.

        Args:
            query: Query do usuário
            state: State do pipeline

        Returns:
            QueryTypeClassification com tipo, confidence, parâmetros e intent
        """
        query_lower = query.lower()

        # ====================================================================
        # PRE-FILTRO 1: Conversational (sem LLM, muito barato)
        # ====================================================================
        if self._is_conversational(query, query_lower):
            logger.debug(f"[QueryClassifier] Pre-filter: CONVERSATIONAL: {query}")
            return QueryTypeClassification(
                query_type="conversational",
                confidence=0.98,
                requires_llm=True,
                parameters={},
            )

        # ====================================================================
        # PRE-FILTRO 2: Tabular explícito (keywords diretos, sem LLM)
        # ====================================================================
        if any(kw in query_lower for kw in self.TABULAR_KEYWORDS):
            logger.debug(f"[QueryClassifier] Pre-filter: TABULAR: {query}")
            limit_match = re.search(r"(\d+)\s*(linhas|registros|rows)", query_lower)
            limit = int(limit_match.group(1)) if limit_match else 100
            return QueryTypeClassification(
                query_type="tabular",
                confidence=0.95,
                requires_llm=False,
                parameters={"limit": limit},
            )

        # ====================================================================
        # PRE-FILTRO 3: Sample rows explícito (metadata com número)
        # ====================================================================
        sample_match = re.search(
            r"(mostre|mostrar|exibir|ver|primeiras?|últimas?)\s+(\d+)\s*(linhas|registros|rows)",
            query_lower,
        )
        if sample_match:
            n = int(sample_match.group(2))
            if n <= 100:
                logger.debug(
                    f"[QueryClassifier] Pre-filter: METADATA (sample_rows): {query}"
                )
                return QueryTypeClassification(
                    query_type="metadata",
                    subtype="sample_rows",
                    confidence=0.95,
                    requires_llm=False,
                    parameters={"metadata_type": "sample_rows", "n": n},
                )

        # ====================================================================
        # INTENT ANALYZER: Compreensão semântica via LLM (Phase 2)
        # ====================================================================
        if self.use_intent_analyzer and self.intent_analyzer is not None:
            try:
                logger.info(
                    f"[QueryClassifier] Delegating to IntentAnalyzer: '{query}'"
                )
                token_accumulator = state.get("_token_accumulator")
                filters = state.get("filter_final", {})

                intent = self.intent_analyzer.analyze(
                    query=query,
                    filters=filters,
                    token_accumulator=token_accumulator,
                )

                # Map QueryIntent → QueryTypeClassification
                classification = self._map_intent_to_classification(
                    intent=intent,
                    query=query,
                    state=state,
                )

                logger.info(
                    f"[QueryClassifier] IntentAnalyzer result: "
                    f"intent_type={intent.intent_type} → "
                    f"query_type={classification.query_type} "
                    f"(confidence={classification.confidence:.2f})"
                )

                return classification

            except Exception as e:
                logger.warning(
                    f"[QueryClassifier] IntentAnalyzer failed, "
                    f"falling back to legacy: {e}"
                )
                # Fall through to legacy classification

        # ====================================================================
        # LEGACY FALLBACK: Classificação keyword-based
        # ====================================================================
        logger.info(f"[QueryClassifier] Using legacy keyword classification: '{query}'")
        return self._legacy_classify(query, query_lower, state)

    def _map_intent_to_classification(
        self,
        intent: QueryIntent,
        query: str,
        state: Dict,
    ) -> QueryTypeClassification:
        """
        Mapeia QueryIntent do IntentAnalyzer para QueryTypeClassification.

        Converte o intent_type semântico para o query_type do schema existente
        e extrai parâmetros compatíveis com o fluxo atual de _execute_by_type.

        Mapeamento:
        - simple_aggregation → aggregation
        - grouped_aggregation → aggregation (com intent para Phase 3)
        - ranking → aggregation (com intent para Phase 3)
        - temporal_analysis → aggregation (com intent para Phase 3)
        - comparison → aggregation (com intent para Phase 3)
        - lookup → lookup
        - metadata → metadata
        - tabular → tabular
        - conversational → conversational

        Args:
            intent: QueryIntent do IntentAnalyzer
            query: Query original
            state: State do pipeline

        Returns:
            QueryTypeClassification com tipo mapeado e intent anexado
        """
        intent_type = intent.intent_type

        # === CONVERSATIONAL ===
        if intent_type == "conversational":
            return QueryTypeClassification(
                query_type="conversational",
                confidence=intent.confidence,
                requires_llm=True,
                parameters={},
                intent=intent,
            )

        # === METADATA ===
        if intent_type == "metadata":
            params = self._extract_metadata_params(query, query.lower())
            return QueryTypeClassification(
                query_type="metadata",
                confidence=intent.confidence,
                requires_llm=False,
                parameters=params,
                intent=intent,
            )

        # === TABULAR ===
        if intent_type == "tabular":
            limit = intent.limit or 100
            return QueryTypeClassification(
                query_type="tabular",
                confidence=intent.confidence,
                requires_llm=False,
                parameters={"limit": limit},
                intent=intent,
            )

        # === LOOKUP ===
        if intent_type == "lookup":
            params = self._extract_lookup_params(query, state)
            return QueryTypeClassification(
                query_type="lookup",
                confidence=intent.confidence,
                requires_llm=True,
                parameters=params,
                intent=intent,
            )

        # === AGGREGATION TYPES (simple, grouped, ranking, temporal, comparison) ===
        # All map to "aggregation" query_type for backward compatibility.
        # The full QueryIntent is carried via the `intent` field for Phase 3.
        params = self._extract_params_from_intent(intent, query, state)

        return QueryTypeClassification(
            query_type="aggregation",
            confidence=intent.confidence,
            requires_llm=True,
            parameters=params,
            intent=intent,
        )

    def _extract_params_from_intent(
        self,
        intent: QueryIntent,
        query: str,
        state: Dict,
    ) -> Dict[str, Any]:
        """
        Extrai parâmetros de agregação a partir do QueryIntent.

        Converte a representação semântica do IntentAnalyzer para os
        parâmetros esperados pelo _execute_by_type existente:
        - column: coluna alvo da agregação
        - aggregation: tipo de função (sum, avg, count, etc.)
        - distinct: se deve usar DISTINCT
        - filters: filtros do state

        Args:
            intent: QueryIntent com análise semântica
            query: Query original
            state: State do pipeline

        Returns:
            Dict com parâmetros compatíveis com o fluxo atual
        """
        params: Dict[str, Any] = {}

        if intent.aggregations:
            agg = intent.aggregations[0]  # Primary aggregation
            params["aggregation"] = agg.function
            params["column"] = agg.column.name
            params["distinct"] = agg.distinct

            # If the column is virtual, note it for QueryExecutor
            if agg.column.is_virtual and agg.column.expression:
                params["column_expression"] = agg.column.expression
                params["column_is_virtual"] = True
        else:
            # No aggregations in intent, try to extract from query (legacy)
            legacy_params = ParameterExtractor.extract_aggregation_params(
                query, state, self.alias_mapper
            )
            params.update(legacy_params)

        # Carry group_by info for future Phase 3
        if intent.group_by:
            params["group_by"] = [
                {
                    "name": col.name,
                    "is_virtual": col.is_virtual,
                    "expression": col.expression,
                    "alias": col.alias,
                }
                for col in intent.group_by
            ]

        # Carry order_by info for future Phase 3
        if intent.order_by:
            params["order_by"] = {
                "column": intent.order_by.column,
                "direction": intent.order_by.direction,
            }

        # Carry limit for future Phase 3
        if intent.limit is not None:
            params["limit"] = intent.limit

        # Get filters from state
        params["filters"] = state.get("filter_final", {})

        return params

    def _legacy_classify(
        self, query: str, query_lower: str, state: Dict
    ) -> QueryTypeClassification:
        """
        Classificação legacy baseada em keywords (fallback).

        Mantida para backward compatibility e como fallback quando
        o IntentAnalyzer não está disponível ou falha.

        Ordem de prioridade:
        1. METADATA - Estrutura do dataset
        2. STATISTICAL - Análises avançadas
        3. AGGREGATION - Cálculos simples
        4. TEXTUAL - Listagens
        5. LOOKUP - Busca específica
        6. LLM fallback

        Args:
            query: Query original
            query_lower: Query em lowercase
            state: State do pipeline

        Returns:
            QueryTypeClassification com tipo, confidence e parâmetros
        """
        # METADATA
        if any(kw in query_lower for kw in self.METADATA_KEYWORDS):
            has_business_context = any(
                kw in query_lower for kw in self.BUSINESS_KEYWORDS
            )
            metadata_terms = [
                "linhas",
                "registros",
                "colunas",
                "campos",
                "rows",
                "columns",
                "valores únicos",
                "valores unicos",
                "distinct count",
                "tipos de dados",
                "tipo das",
            ]
            has_metadata_terms = any(term in query_lower for term in metadata_terms)

            if has_metadata_terms or not has_business_context:
                logger.debug(f"[Legacy] Query classified as METADATA: {query}")
                params = self._extract_metadata_params(query, query_lower)
                return QueryTypeClassification(
                    query_type="metadata",
                    confidence=0.90,
                    requires_llm=False,
                    parameters=params,
                )

        # STATISTICAL
        if any(kw in query_lower for kw in self.STATISTICAL_KEYWORDS):
            logger.debug(f"[Legacy] Query classified as STATISTICAL: {query}")
            params = self._extract_statistical_params(query, state)
            return QueryTypeClassification(
                query_type="statistical",
                confidence=0.85,
                requires_llm=True,
                parameters=params,
            )

        # AGGREGATION
        if self._is_aggregation(query_lower):
            logger.debug(f"[Legacy] Query classified as AGGREGATION: {query}")
            params = self._extract_aggregation_params(query, state)
            return QueryTypeClassification(
                query_type="aggregation",
                confidence=0.85,
                requires_llm=True,
                parameters=params,
            )

        # TEXTUAL
        if self._is_textual(query_lower):
            logger.debug(f"[Legacy] Query classified as TEXTUAL: {query}")
            params = self._extract_textual_params(query, state)
            return QueryTypeClassification(
                query_type="textual",
                confidence=0.80,
                requires_llm=True,
                parameters=params,
            )

        # LOOKUP
        if self._is_lookup(query_lower):
            logger.debug(f"[Legacy] Query classified as LOOKUP: {query}")
            params = self._extract_lookup_params(query, state)
            return QueryTypeClassification(
                query_type="lookup",
                confidence=0.80,
                requires_llm=True,
                parameters=params,
            )

        # LLM Fallback
        logger.debug(f"[Legacy] No keyword match, using LLM fallback for: {query}")
        return self._llm_classify(query, state)

    def _is_aggregation(self, query_lower: str) -> bool:
        """
        Verifica se query é uma agregação usando padrões contextuais.

        Evita falsos positivos verificando contexto ao redor das keywords.

        Args:
            query_lower: Query em lowercase

        Returns:
            True se agregação, False caso contrário
        """
        for pattern, context_words in self.AGGREGATION_PATTERNS:
            if pattern in query_lower:
                # If no context required, it's aggregation
                if context_words is None:
                    return True
                # If context required, check if any context word is present
                if any(ctx in query_lower for ctx in context_words):
                    return True
        return False

    def _is_textual(self, query_lower: str) -> bool:
        """
        Verifica se query é textual usando padrões contextuais.

        Args:
            query_lower: Query em lowercase

        Returns:
            True se textual, False caso contrário
        """
        for pattern, context_words in self.TEXTUAL_PATTERNS:
            if pattern in query_lower:
                # If no context required, it's textual
                if context_words is None:
                    return True
                # If context required, check if any context word is present
                if any(ctx in query_lower for ctx in context_words):
                    return True
        return False

    def _is_lookup(self, query_lower: str) -> bool:
        """
        Verifica se query é lookup usando padrões contextuais.

        Lookup requer indicação de registro específico (ID, código, etc).

        Args:
            query_lower: Query em lowercase

        Returns:
            True se lookup, False caso contrário
        """
        for pattern, context_words in self.LOOKUP_PATTERNS:
            if pattern in query_lower:
                # If no context required, it's lookup
                if context_words is None:
                    return True
                # If context required, check if any context word is present
                if any(ctx in query_lower for ctx in context_words):
                    return True
        return False

    def _is_conversational(self, query: str, query_lower: str) -> bool:
        """
        Detecta se query é conversacional.

        Critérios:
        - Query é saudação simples (GREETINGS)
        - Query tem <= 3 palavras E não contém keywords de negócio
        - NÃO deve classificar queries que contenham keywords de metadata
        - Queries de ajuda ("como funciona?", "o que você faz?", etc)

        Args:
            query: Query original
            query_lower: Query em lowercase

        Returns:
            True se conversacional, False caso contrário
        """
        # Check if it's a greeting
        if query_lower.strip() in self.GREETINGS:
            return True

        # Check for help/question patterns
        help_patterns = [
            "como funciona",
            "o que você faz",
            "o que voce faz",
            "pode me ajudar",
            "ajuda",
            "help",
            "como usar",
            "o que é isso",
            "o que e isso",
        ]
        if any(pattern in query_lower for pattern in help_patterns):
            return True

        # Check if query has metadata keywords (not conversational)
        if any(kw in query_lower for kw in self.METADATA_KEYWORDS):
            return False

        # Check if short and generic (no business/data keywords)
        # Must be TRULY conversational (greeting-like)
        words = query.split()
        if len(words) <= 3:
            has_business = any(kw in query_lower for kw in self.BUSINESS_KEYWORDS)
            # Also check for data-related terms
            data_terms = ["linhas", "registros", "rows", "dados", "tabela", "colunas"]
            has_data_terms = any(term in query_lower for term in data_terms)

            # Only conversational if no business keywords AND no data terms
            if not has_business and not has_data_terms:
                return True

        return False

    def _extract_metadata_params(self, query: str, query_lower: str) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de metadata.

        Args:
            query: Query original
            query_lower: Query em lowercase

        Returns:
            Dict com metadata_type e parâmetros adicionais
        """
        return ParameterExtractor.extract_metadata_params(
            query, query_lower, self.alias_mapper
        )

    def _extract_aggregation_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de agregação.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com aggregation, column e filters
        """
        return ParameterExtractor.extract_aggregation_params(
            query, state, self.alias_mapper
        )

    def _extract_lookup_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de lookup usando LLM.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com lookup_column e lookup_value
        """
        token_accumulator = state.get("_token_accumulator")
        return ParameterExtractor.extract_lookup_params(
            query,
            state,
            self.llm,
            self.alias_mapper,
            token_accumulator=token_accumulator,
        )

    def _extract_textual_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries textuais.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com column e search_term
        """
        return ParameterExtractor.extract_textual_params(
            query, state, self.alias_mapper
        )

    def _extract_statistical_params(self, query: str, state: Dict) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries estatísticas.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            Dict com column e filters
        """
        return ParameterExtractor.extract_statistical_params(
            query, state, self.alias_mapper
        )

    def _llm_classify(self, query: str, state: Dict) -> QueryTypeClassification:
        """
        Fallback para classificação via LLM quando keywords não funcionam.

        Args:
            query: Query original
            state: State do pipeline

        Returns:
            QueryTypeClassification com resultado da classificação
        """
        try:
            prompt = f"""Classifique a seguinte query em UMA das categorias abaixo:

Categorias:
- metadata: Perguntas sobre estrutura dos dados (linhas, colunas, tipos)
- aggregation: Agregações simples (média, soma, total, min, max)
- lookup: Busca de registros específicos
- textual: Buscas textuais ou listagens
- statistical: Estatísticas descritivas completas
- tabular: Solicitação de dados em formato tabela

Query: "{query}"

Retorne APENAS um JSON válido no formato:
{{"query_type": "tipo", "confidence": 0.0-1.0}}

Use confidence alto (0.8-0.9) se tiver certeza, médio (0.5-0.7) se ambíguo."""

            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Capture and accumulate tokens
            from src.shared_lib.utils.token_tracker import extract_token_usage

            tokens = extract_token_usage(response, self.llm)
            if "_token_accumulator" in state:
                state["_token_accumulator"].add(tokens)
                logger.debug(f"[QueryClassifier] Tokens accumulated: {tokens}")

            # Try to parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    # Try to find any JSON object
                    match = re.search(r"\{.*?\}", content, re.DOTALL)
                    if match:
                        result = json.loads(match.group(0))
                    else:
                        raise ValueError("No JSON found in response")

            query_type = result.get("query_type", "tabular")
            confidence = float(result.get("confidence", 0.5))

            logger.info(
                f"LLM classified query as: {query_type} (confidence: {confidence})"
            )

            # Extract parameters based on type
            parameters = {}
            if query_type == "aggregation":
                parameters = self._extract_aggregation_params(query, state)
            elif query_type == "lookup":
                parameters = self._extract_lookup_params(query, state)
            elif query_type == "textual":
                parameters = self._extract_textual_params(query, state)
            elif query_type == "statistical":
                parameters = self._extract_statistical_params(query, state)
            elif query_type == "metadata":
                parameters = self._extract_metadata_params(query, query.lower())
            elif query_type == "tabular":
                parameters = {"limit": 100}

            return QueryTypeClassification(
                query_type=query_type,
                confidence=confidence,
                requires_llm=True,
                parameters=parameters,
            )

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Fallback to tabular with low confidence
            return QueryTypeClassification(
                query_type="tabular",
                confidence=0.5,
                requires_llm=False,
                parameters={"limit": 100},
            )
