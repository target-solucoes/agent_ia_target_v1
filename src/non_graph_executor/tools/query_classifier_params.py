"""
Parameter extraction methods for QueryClassifier.

This module contains the parameter extraction logic for different query types.
Separated for better organization and maintainability.
"""

import logging
import json
import re
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ParameterExtractor:
    """Helper class for parameter extraction."""

    @staticmethod
    def extract_metadata_params(
        query: str, query_lower: str, alias_mapper
    ) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de metadata.

        Detecta tipo específico:
        - row_count: "quantas linhas", "número de linhas"
        - column_count: "quantas colunas", "número de colunas"
        - column_list: "quais colunas", "lista de colunas"
        - dtypes: "tipos de dados", "dtypes"
        - sample_rows: "primeiras linhas", "sample", "amostra" (extrai número via regex)
        - unique_values: "valores únicos", "unique" (identifica coluna)

        Args:
            query: Query original
            query_lower: Query em lowercase
            alias_mapper: AliasMapper para resolução de colunas

        Returns:
            Dict com metadata_type e parâmetros adicionais
        """
        params = {}

        # Row count
        if (
            "quantas linhas" in query_lower
            or "número de linhas" in query_lower
            or "número de registros" in query_lower
            or "quantos registros" in query_lower
            or "total de linhas" in query_lower
            or "total de registros" in query_lower
        ):
            params["metadata_type"] = "row_count"

        # Column count
        elif "quantas colunas" in query_lower or "número de colunas" in query_lower:
            params["metadata_type"] = "column_count"

        # Column list
        elif (
            "quais colunas" in query_lower
            or "quais são as colunas" in query_lower
            or "quais campos" in query_lower
            or "lista de colunas" in query_lower
            or "liste as colunas" in query_lower
            or "mostre os campos" in query_lower
        ):
            params["metadata_type"] = "column_list"

        # Data types
        elif (
            "tipos de dados" in query_lower
            or "tipo das colunas" in query_lower
            or "mostre os tipos" in query_lower
            or "dtypes" in query_lower
        ):
            params["metadata_type"] = "dtypes"

        # Sample rows
        elif (
            any(
                kw in query_lower
                for kw in [
                    "primeiras linhas",
                    "últimas linhas",
                    "mostre linhas",
                    "primeiras",
                    "últimas",
                    "sample",
                    "amostra",
                    "exemplos de dados",
                    "preview dos dados",
                ]
            )
            or ("mostre" in query_lower and "linhas" in query_lower)
            or ("primeiras" in query_lower and "linhas" in query_lower)
            or ("últimas" in query_lower and "linhas" in query_lower)
        ):
            params["metadata_type"] = "sample_rows"
            # Extract number via regex
            match = re.search(r"(\d+)", query)
            params["n"] = int(match.group(1)) if match else 5

        # Unique count (quantos valores únicos OR valores únicos de [column])
        elif (
            "quantos valores únicos" in query_lower
            or "quantos valores unicos" in query_lower
            or "distinct count" in query_lower
            or ("valores únicos de" in query_lower)
            or ("valores unicos de" in query_lower)
        ):
            params["metadata_type"] = "unique_count"
            # Try to extract column name
            try:
                column = ParameterExtractor._extract_column_name(query, alias_mapper)
                if column:
                    params["column"] = column
            except AttributeError:
                # alias_mapper might be a Mock in tests
                pass

        # Unique values (lista de valores únicos - when NOT followed by "de [column]")
        elif (
            "valores únicos" in query_lower
            or "valores unicos" in query_lower
            or "valores distintos" in query_lower
        ):
            params["metadata_type"] = "unique_values"
            # Try to extract column name
            try:
                column = ParameterExtractor._extract_column_name(query, alias_mapper)
                if column:
                    params["column"] = column
            except AttributeError:
                # alias_mapper might be a Mock in tests
                pass

        else:
            # Default to sample_rows
            params["metadata_type"] = "sample_rows"
            params["n"] = 5

        return params

    @staticmethod
    def extract_aggregation_params(
        query: str, state: Dict, alias_mapper
    ) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de agregação.

        Detecta tipo de agregação e coluna alvo.

        Args:
            query: Query original
            state: State do pipeline
            alias_mapper: AliasMapper para resolução de colunas

        Returns:
            Dict com aggregation, column e filters
        """
        query_lower = query.lower()
        params = {}

        # Detect aggregation type (ordem de prioridade para evitar conflitos)
        # 1. Mediana (antes de média para não confundir)
        if "mediana" in query_lower or "median" in query_lower:
            params["aggregation"] = "median"
        # 2. Média/Average
        elif (
            "média" in query_lower
            or "media" in query_lower
            or "average" in query_lower
            or "valor médio" in query_lower
            or "valor medio" in query_lower
        ):
            params["aggregation"] = "avg"
        # 3. Menor/Mínimo (antes de min para capturar "menor")
        elif (
            "menor" in query_lower
            or "mínimo" in query_lower
            or "minimo" in query_lower
            or "minima" in query_lower  # feminino
            or "mínima" in query_lower  # feminino com acento
            or "qual o menor" in query_lower
            or "qual a menor" in query_lower
            or "qual o minimo" in query_lower
            or "qual a minima" in query_lower
        ):
            params["aggregation"] = "min"
        # 4. Maior/Máximo (antes de max para capturar "maior")
        elif (
            "maior" in query_lower
            or "máximo" in query_lower
            or "maximo" in query_lower
            or "maxima" in query_lower  # feminino
            or "máxima" in query_lower  # feminino com acento
            or "qual o maior" in query_lower
            or "qual a maior" in query_lower
            or "qual o maximo" in query_lower
            or "qual a maxima" in query_lower
        ):
            params["aggregation"] = "max"
        # 5. Count (quantos, quantas, número de)
        elif (
            "count" in query_lower
            or "quantos" in query_lower
            or "quantas" in query_lower
            or "número de" in query_lower
            or "numero de" in query_lower
            or "contagem" in query_lower
        ):
            params["aggregation"] = "count"
            # Detectar se precisa de COUNT DISTINCT (valores únicos de entidades)
            # Para entidades como clientes, produtos, vendedores, etc.
            params["distinct"] = any(
                [
                    "clientes" in query_lower,
                    "produtos" in query_lower,
                    "vendedores" in query_lower,
                    "pedidos" in query_lower,
                    "categorias" in query_lower,
                    "famílias" in query_lower,
                    "familias" in query_lower,
                    "estados" in query_lower,
                    "cidades" in query_lower,
                    "uf" in query_lower,
                    "distinct" in query_lower,
                    "únicos" in query_lower,
                    "unicos" in query_lower,
                ]
            )
        # 6. Soma
        elif (
            "soma" in query_lower
            or "sum" in query_lower
            or "somatório" in query_lower
            or "somatorio" in query_lower
        ):
            params["aggregation"] = "sum"
        # 7. Total (verificar contexto)
        elif "total" in query_lower:
            # Check if it's count or sum
            if (
                "quantidade" in query_lower
                or "número" in query_lower
                or "quantos" in query_lower
            ):
                params["aggregation"] = "count"
            else:
                params["aggregation"] = "sum"
        # 8. Desvio padrão (deveria ser statistical, mas por segurança)
        elif "desvio" in query_lower or "std" in query_lower:
            params["aggregation"] = "std"
        # 9. Quantidade
        elif "quantidade" in query_lower:
            params["aggregation"] = "count"
        # 10. Último/Mais recente → MAX (ex: "último ano", "última venda", "mais recente")
        elif any(
            kw in query_lower
            for kw in [
                "ultimo",
                "última",
                "ultimo",
                "ultima",
                "mais recente",
                "mais novo",
                "mais nova",
            ]
        ):
            params["aggregation"] = "max"
        # 11. Primeiro/Mais antigo → MIN (ex: "primeiro ano", "primeira venda", "mais antigo")
        elif any(
            kw in query_lower
            for kw in [
                "primeiro",
                "primeira",
                "mais antigo",
                "mais antiga",
            ]
        ):
            params["aggregation"] = "min"
        else:
            # Default to sum
            params["aggregation"] = "sum"

        # Extract column name
        column = ParameterExtractor._extract_column_name(query, alias_mapper)
        if column:
            params["column"] = column
        else:
            # Default to first numeric column if available
            params["column"] = None

        # Get filters from state
        params["filters"] = state.get("filter_final", {})

        return params

    @staticmethod
    def extract_lookup_params(
        query: str,
        state: Dict,
        llm,
        alias_mapper,
        token_accumulator=None,
    ) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries de lookup usando LLM.

        Args:
            query: Query original
            state: State do pipeline
            llm: LLM para extração
            alias_mapper: AliasMapper para resolução de colunas
            token_accumulator: TokenAccumulator compartilhado para tracking

        Returns:
            Dict com lookup_column e lookup_value
        """
        try:
            prompt = f"""Extraia as informações de busca da seguinte query:
Query: "{query}"

Identifique:
1. lookup_column: coluna sendo usada para buscar (ex: "Cod_Cliente", "id", "pedido")
2. lookup_value: valor sendo buscado (ex: "123", "C001", "ABC")

Retorne APENAS um JSON válido no formato:
{{"lookup_column": "nome_da_coluna", "lookup_value": "valor"}}"""

            response = llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Track token usage for lookup parameter extraction
            from src.shared_lib.utils.token_tracker import extract_token_usage

            tokens = extract_token_usage(response, llm)
            if token_accumulator is not None:
                token_accumulator.add(tokens)
                logger.debug(
                    f"[ParameterExtractor] Lookup tokens accumulated: {tokens}"
                )

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

            # Resolve column name via alias_mapper
            lookup_column = result.get("lookup_column")
            if lookup_column:
                resolved = alias_mapper.resolve(lookup_column)
                if resolved:
                    result["lookup_column"] = resolved

            return result

        except Exception as e:
            logger.error(f"Error extracting lookup params: {e}")
            return {"lookup_column": None, "lookup_value": None}

    @staticmethod
    def extract_textual_params(query: str, state: Dict, alias_mapper) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries textuais.

        Args:
            query: Query original
            state: State do pipeline
            alias_mapper: AliasMapper para resolução de colunas

        Returns:
            Dict com column e search_term
        """
        params = {}

        # Extract column name
        column = ParameterExtractor._extract_column_name(query, alias_mapper)
        params["column"] = column

        # Extract search term
        # Try to find quoted text
        match = re.search(r'"([^"]*)"', query) or re.search(r"'([^']*)'", query)
        if match:
            params["search_term"] = match.group(1)
        else:
            # Try to find text after "contém"
            match = re.search(r"contém\s+(\w+)", query.lower())
            if match:
                params["search_term"] = match.group(1)
            else:
                # Default to empty
                params["search_term"] = ""

        params["case_sensitive"] = False  # Default

        return params

    @staticmethod
    def extract_statistical_params(
        query: str, state: Dict, alias_mapper
    ) -> Dict[str, Any]:
        """
        Extrai parâmetros para queries estatísticas.

        Args:
            query: Query original
            state: State do pipeline
            alias_mapper: AliasMapper para resolução de colunas

        Returns:
            Dict com column e filters
        """
        params = {}

        # Extract column name
        column = ParameterExtractor._extract_column_name(query, alias_mapper)
        params["column"] = column

        # Get filters from state
        params["filters"] = state.get("filter_final", {})

        return params

    @staticmethod
    def _extract_column_name(query: str, alias_mapper) -> Optional[str]:
        """
        Extrai nome de coluna da query usando AliasMapper.

        Itera sobre possíveis termos na query e tenta resolver via alias_mapper.

        IMPORTANTE: Para queries de agregação (soma, média, maior, menor, etc.),
        prioriza colunas NUMÉRICAS sobre colunas categóricas, pois a agregação
        deve ser aplicada a métricas numéricas, não a dimensões categóricas.

        Para queries com intenção temporal (último ano, em que mês, quando, etc.),
        prioriza colunas TEMPORAIS ou VIRTUAIS TEMPORAIS (Ano, Mes, Data) sobre
        colunas numéricas, pois o usuário quer saber QUANDO, não QUANTO.

        Args:
            query: Query original
            alias_mapper: AliasMapper para resolução

        Returns:
            Nome real da coluna ou None
        """
        query_lower = query.lower()

        # Detectar se a query tem intenção temporal
        # Estas keywords indicam que o usuário quer saber QUANDO algo ocorreu,
        # não QUANTO foi o valor — portanto a coluna alvo deve ser temporal.
        temporal_intent_keywords = [
            "ultimo ano",
            "última ano",
            "ultimo ano",
            "ultima ano",
            "primeiro ano",
            "primeira ano",
            "em que ano",
            "qual ano",
            "qual o ano",
            "ultimo mes",
            "última mes",
            "ultimo mês",
            "ultima mês",
            "primeiro mes",
            "primeira mes",
            "primeiro mês",
            "em que mes",
            "em que mês",
            "qual mes",
            "qual mês",
            "qual o mes",
            "qual o mês",
            "quando foi",
            "quando ocorreu",
            "em que data",
            "em que periodo",
            "em que período",
        ]
        is_temporal_intent = any(kw in query_lower for kw in temporal_intent_keywords)

        # Detectar se é uma query de agregação numérica
        is_numeric_aggregation = any(
            kw in query_lower
            for kw in [
                "soma",
                "sum",
                "media",
                "média",
                "average",
                "maior",
                "menor",
                "máximo",
                "maximo",
                "maxima",
                "máxima",
                "mínimo",
                "minimo",
                "minima",
                "mínima",
                "total",
                "totais",
                "somatório",
                "somatorio",
                "mediana",
                "median",
                "aumento",
                "queda",
                "variação",
                "variacao",
                "crescimento",
                "decrescimento",
                "delta",
            ]
        )

        # Split query into words and try to resolve each
        words = query.split()

        # Coletar TODAS as colunas resolvidas (tanto numéricas quanto categóricas)
        resolved_columns = []

        # Try 2-word combinations first (more specific)
        for i in range(len(words) - 1):
            term = f"{words[i]} {words[i + 1]}"
            resolved = alias_mapper.resolve(term)
            if resolved:
                resolved_columns.append((term, resolved))

        # Try single words
        for word in words:
            resolved = alias_mapper.resolve(word)
            if resolved:
                # Evitar duplicatas
                if not any(r[1] == resolved for r in resolved_columns):
                    resolved_columns.append((word, resolved))

        # Se encontrou colunas, aplicar lógica de priorização
        if resolved_columns:
            # Para queries com intenção temporal, PRIORIZAR colunas temporais/virtuais
            # Isso garante que "qual o último ano com vendas?" resolva para "Ano"
            # (coluna virtual → YEAR("Data")) em vez de "Valor_Vendido"
            if is_temporal_intent:
                try:
                    temporal_cols = alias_mapper.column_types.get("temporal", [])
                    # Verificar colunas virtuais temporais (Ano, Mes, Nome_Mes)
                    has_virtual = hasattr(alias_mapper, "is_virtual_column")

                    for term, col in resolved_columns:
                        is_temporal = col in temporal_cols
                        is_virtual_temporal = (
                            has_virtual and alias_mapper.is_virtual_column(col)
                        )
                        if is_temporal or is_virtual_temporal:
                            logger.debug(
                                f"Resolved column (temporal priority): '{term}' -> '{col}' "
                                f"(temporal={is_temporal}, virtual={is_virtual_temporal})"
                            )
                            return col

                    logger.debug(
                        f"Temporal intent detected but no temporal column found in resolved: "
                        f"{[c[1] for c in resolved_columns]}"
                    )
                except (AttributeError, KeyError):
                    pass

            # Para agregações numéricas, PRIORIZAR colunas numéricas
            if is_numeric_aggregation:
                try:
                    numeric_cols = alias_mapper.column_types.get("numeric", [])

                    # Procurar primeira coluna numérica nas resolvidas
                    for term, col in resolved_columns:
                        if col in numeric_cols:
                            logger.debug(
                                f"Resolved column (numeric priority): '{term}' -> '{col}'"
                            )
                            return col

                    # Se não encontrou numérica nas resolvidas, tentar fallback
                    # mas apenas se não encontrou NENHUMA coluna categórica também
                    logger.warning(
                        f"Aggregation query but no numeric column found in query. "
                        f"Resolved columns: {[c[1] for c in resolved_columns]}"
                    )

                except (AttributeError, KeyError):
                    # alias_mapper might not have column_types
                    pass

            # Retornar primeira coluna resolvida (comportamento original para COUNT ou quando não é agregação numérica)
            term, col = resolved_columns[0]
            logger.debug(f"Resolved column: '{term}' -> '{col}'")
            return col

        # NOVO: Fallback para coluna numérica default em agregações
        query_lower = query.lower()
        if any(
            kw in query_lower
            for kw in [
                "total",
                "totais",
                "soma",
                "media",
                "média",
                "maior",
                "menor",
                "máximo",
                "maximo",
                "maxima",  # feminino
                "máxima",  # feminino com acento
                "mínimo",
                "minimo",
                "minima",  # feminino
                "mínima",  # feminino com acento
            ]
        ):
            try:
                numeric_cols = alias_mapper.column_types.get("numeric", [])
                if numeric_cols:
                    default = numeric_cols[0]
                    logger.debug(
                        f"No column found, using default numeric column: {default}"
                    )
                    return default
            except (AttributeError, KeyError):
                # alias_mapper might not have column_types
                pass

        logger.debug(f"No column found in query: {query}")
        return None
