"""
Output formatter for non_graph_executor.

This module implements JSON output formatting and summary generation
for non-graph query results using Google Gemini LLM when appropriate.
"""

import logging
from typing import Any, Dict, List, Optional, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.non_graph_executor.models.schemas import NonGraphOutput

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Formatador de output final do non_graph_executor.

    Gera JSON estruturado seguindo schema NonGraphOutput.
    Adiciona summaries via Google Gemini LLM quando apropriado (aggregation, statistical, lookup).

    Esta classe é responsável por:
    1. Formatar output final em JSON estruturado
    2. Gerar summaries textuais para queries que requerem interpretação
    3. Garantir compatibilidade com o schema NonGraphOutput
    4. Formatar respostas conversacionais

    Attributes:
        llm: Google Gemini LLM instance configurado para geração de summaries
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize output formatter.

        Args:
            llm: ChatGoogleGenerativeAI instance for generating summaries
                (configured with temperature=0.3 and max_output_tokens=800)
        """
        self.llm = llm
        logger.info("OutputFormatter initialized successfully")

    def format(
        self,
        query_type: Literal[
            "metadata",
            "aggregation",
            "lookup",
            "textual",
            "statistical",
            "conversational",
            "tabular",
        ],
        data: Optional[List[Dict[str, Any]]],
        metadata: Dict[str, Any],
        performance: Dict[str, float],
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, str]] = None,
        token_accumulator=None,
    ) -> Dict[str, Any]:
        """
        Formata output final completo do non_graph_executor.

        Este método orquestra a formatação do resultado final:
        1. Se há erro, retorna output de erro
        2. Gera summary via LLM (para query types apropriados)
        3. Cria NonGraphOutput estruturado
        4. Retorna dict serializado

        Args:
            query_type: Tipo de query processada
            data: Dados retornados pela query (None se erro)
            metadata: Metadados da execução (row_count, columns, etc)
            performance: Métricas de performance (total_time, llm_time, etc)
            query: Query original do usuário
            filters: Filtros aplicados (opcional)
            error: Informações de erro (opcional)

        Returns:
            Dict com output formatado seguindo schema NonGraphOutput

        Example:
            >>> formatter = OutputFormatter(llm)
            >>> output = formatter.format(
            ...     query_type="aggregation",
            ...     data=[{"avg_vendas": 15420.50}],
            ...     metadata={"row_count": 1},
            ...     performance={"total_time": 0.287},
            ...     query="qual a média de vendas?",
            ...     filters={"Ano": 2015}
            ... )
            >>> print(output['status'])
            'success'
        """
        logger.debug(f"Formatting output for query_type={query_type}")

        # Copy metadata to avoid mutating caller state and sync execution timing info
        metadata = dict(metadata or {})
        perf_execution_time = performance.get("execution_time")
        perf_total_time = performance.get("total_time")
        if perf_execution_time is not None:
            metadata["execution_time"] = perf_execution_time
        if perf_total_time is not None:
            metadata["total_execution_time"] = perf_total_time

        # Handle error case
        if error is not None:
            logger.warning(f"Formatting error output: {error.get('type', 'Unknown')}")
            return NonGraphOutput(
                status="error",
                query_type=query_type,
                error=error,
                metadata=metadata,
                performance_metrics=performance,
            ).model_dump()

        # Generate summary for appropriate query types
        summary = None
        if query_type in ["aggregation", "statistical", "lookup", "metadata"]:
            try:
                logger.debug(f"Generating summary for {query_type} query")
                summary = self._generate_summary(
                    query_type=query_type,
                    query=query,
                    data=data,
                    metadata=metadata,
                    filters=filters or {},
                    token_accumulator=token_accumulator,
                )
            except Exception as e:
                logger.warning(f"Failed to generate summary: {str(e)}")
                # Continue without summary - not critical

        # Create structured output
        output = NonGraphOutput(
            status="success",
            query_type=query_type,
            data=data,
            summary=summary,
            metadata=metadata,
            performance_metrics=performance,
        )

        logger.debug(
            f"Output formatted successfully with summary={summary is not None}"
        )
        return output.model_dump()

    def _generate_summary(
        self,
        query_type: str,
        query: str,
        data: Optional[List[Dict[str, Any]]],
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
        token_accumulator=None,
    ) -> Optional[str]:
        """
        Gera summary textual via LLM para queries específicas.

        Este método usa LLM com reasoning_effort='minimal' para gerar
        summaries concisos e objetivos. Apenas query types que se
        beneficiam de interpretação recebem summaries.

        Args:
            query_type: Tipo de query (aggregation, statistical, lookup)
            query: Query original do usuário
            data: Dados retornados
            metadata: Metadados da execução
            filters: Filtros aplicados

        Returns:
            Summary textual (1-3 frases) ou None se não aplicável

        Raises:
            Exception: Erros são propagados para o caller tratar

        Example:
            >>> summary = formatter._generate_summary(
            ...     query_type="aggregation",
            ...     query="qual a média de vendas?",
            ...     data=[{"avg_vendas": 15420.50}],
            ...     metadata={"row_count": 1},
            ...     filters={"Ano": 2015}
            ... )
            >>> print(summary)
            'A média de vendas em 2015 é R$ 15.420,50.'
        """
        if not data:
            logger.debug("No data to summarize")
            return None

        # Generate prompt based on query type
        if query_type == "aggregation":
            prompt = self._create_aggregation_prompt(query, data, filters)
        elif query_type == "statistical":
            prompt = self._create_statistical_prompt(query, data, filters)
        elif query_type == "lookup":
            prompt = self._create_lookup_prompt(query, data)
        elif query_type == "metadata":
            prompt = self._create_metadata_prompt(query, data, filters)
        else:
            logger.debug(f"No summary generation for query_type={query_type}")
            return None

        # Generate summary using LLM
        logger.debug(f"Calling LLM for summary generation (query_type={query_type})")
        try:
            messages = [
                SystemMessage(
                    content=(
                        "Você é um assistente de análise de dados especializado em "
                        "resumir resultados de queries de forma clara e concisa. "
                        "Responda sempre em português brasileiro, de forma objetiva "
                        "e profissional."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            summary = response.content.strip()

            # Capture and accumulate tokens
            from src.shared_lib.utils.token_tracker import extract_token_usage

            tokens = extract_token_usage(response, self.llm)
            if token_accumulator is not None:
                token_accumulator.add(tokens)
                logger.debug(f"[OutputFormatter] Tokens accumulated: {tokens}")

            logger.debug(f"Summary generated successfully (length={len(summary)})")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def _create_aggregation_prompt(
        self, query: str, data: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> str:
        """
        Cria prompt para summarizar resultado de agregação.

        Suporta tanto resultados escalares simples (1 valor) quanto
        resultados multi-dimensionais (com GROUP BY, rankings, etc.).

        Para dados multi-dimensionais, inclui TODOS os dados retornados
        no prompt para que o LLM possa gerar uma resposta completa
        identificando dimensões (mês, estado, etc.) e valores.

        Args:
            query: Query original
            data: Resultado da agregação (pode ser multi-dimensional)
            filters: Filtros aplicados

        Returns:
            Prompt formatado para o LLM
        """
        filters_text = (
            f"\nFiltros aplicados: {filters}" if filters else "\nSem filtros aplicados"
        )

        # Verificar se resultado é multi-dimensional (mais de 1 coluna ou mais de 1 linha)
        is_multi_dimensional = False
        if data:
            first_row = data[0]
            # Multi-dimensional: mais de 1 chave no dict ou mais de 1 linha
            if len(first_row) > 1 or len(data) > 1:
                is_multi_dimensional = True

        if is_multi_dimensional:
            # Formatar dados completos de forma legível
            # Limitar a 20 linhas para evitar prompts muito longos
            display_data = data[:20]
            data_text = ""
            for i, row in enumerate(display_data, 1):
                row_items = [f"{k}: {v}" for k, v in row.items()]
                data_text += f"  {i}. {', '.join(row_items)}\n"

            if len(data) > 20:
                data_text += f"  ... e mais {len(data) - 20} resultados\n"

            return f"""Resuma o resultado da análise de forma concisa e completa:

Query: "{query}"
Dados retornados ({len(data)} resultado(s)):
{data_text}
{filters_text}

INSTRUÇÕES IMPORTANTES:
- Responda DIRETAMENTE à pergunta feita pelo usuário.
- Se os dados contêm dimensões (mês, estado, etc.), SEMPRE identifique-as pelo nome na resposta.
- Para valores monetários, use formato R$ com separadores de milhar.
- Para meses numéricos, converta para nomes (1=janeiro, 2=fevereiro, 3=março, 4=abril, 5=maio, 6=junho, 7=julho, 8=agosto, 9=setembro, 10=outubro, 11=novembro, 12=dezembro).
- Se houver ranking, mencione a posição e o valor.
- Responda em 1-3 frases, de forma clara e objetiva.
- NÃO adicione sugestões ou contexto extra além do solicitado."""

        else:
            # Resultado escalar simples (comportamento original)
            return f"""Resuma o resultado da agregação de forma concisa:

Query: "{query}"
Resultado: {data[0] if data else "Nenhum resultado"}
{filters_text}

Responda em 1-2 frases, de forma clara e objetiva. 
Use formatação de moeda (R$) quando apropriado.
Não adicione contexto extra ou sugestões, apenas resuma o resultado."""

    def _create_statistical_prompt(
        self, query: str, data: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> str:
        """
        Cria prompt para summarizar estatísticas descritivas.

        Args:
            query: Query original
            data: Estatísticas calculadas
            filters: Filtros aplicados

        Returns:
            Prompt formatado para o LLM
        """
        filters_text = (
            f"\nFiltros aplicados: {filters}" if filters else "\nSem filtros aplicados"
        )

        return f"""Resuma as estatísticas descritivas de forma executiva:

Query: "{query}"
Estatísticas: {data[0] if data else "Nenhum resultado"}
{filters_text}

Destaque os insights principais em 2-3 frases.
Use formatação numérica apropriada (moeda, percentual, etc).
Foque nos valores mais relevantes (média, mediana, outliers)."""

    def _create_lookup_prompt(self, query: str, data: List[Dict[str, Any]]) -> str:
        """
        Cria prompt para summarizar resultado de lookup.

        Args:
            query: Query original
            data: Registro(s) encontrado(s)

        Returns:
            Prompt formatado para o LLM
        """
        return f"""Resuma o resultado da busca:

Query: "{query}"
Registro encontrado: {data[0] if data else "Nenhum registro encontrado"}

Responda de forma clara em 1-2 frases.
Se não encontrou resultado, indique isso claramente.
Não adicione contexto extra, apenas resuma o que foi encontrado."""

    def _create_metadata_prompt(
        self, query: str, data: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> str:
        """
        Cria prompt para summarizar informações de metadata.

        Args:
            query: Query original
            data: Metadados retornados
            filters: Filtros aplicados

        Returns:
            Prompt formatado para o LLM
        """
        filters_text = (
            f"\nFiltros aplicados: {filters}" if filters else "\nSem filtros aplicados"
        )

        data_dict = data[0] if data else {}

        # Identificar tipo de informação
        if "row_count" in data_dict and len(data_dict) <= 3:
            # Query sobre quantidade de linhas
            return f"""Responda à pergunta de forma natural e direta:

Query: "{query}"
Resposta: O dataset possui {data_dict["row_count"]:,} linhas.
{filters_text}

Reformule a resposta acima em português brasileiro, de forma clara e natural.
Apenas responda diretamente, sem adicionar contexto extra."""
        else:
            # Outras informações de metadata
            return f"""Resuma as informações do dataset de forma concisa:

Query: "{query}"
Dados: {data_dict}
{filters_text}

Responda em 1-2 frases, de forma clara e objetiva.
Não adicione contexto extra ou sugestões."""

    def format_conversational(self, response: str) -> Dict[str, Any]:
        """
        Formata resposta conversacional específica.

        Para queries conversacionais (saudações, ajuda), formata
        a resposta de forma apropriada sem dados ou metadados.

        Args:
            response: Resposta conversacional gerada

        Returns:
            Dict com output formatado para query conversacional

        Example:
            >>> formatter = OutputFormatter(llm)
            >>> output = formatter.format_conversational(
            ...     "Olá! Estou aqui para ajudar com análise de dados."
            ... )
            >>> print(output['query_type'])
            'conversational'
        """
        logger.debug("Formatting conversational response")

        output = NonGraphOutput(
            status="success",
            query_type="conversational",
            conversational_response=response,
            metadata={},
            performance_metrics={},
        )

        return output.model_dump()
