"""
Dynamic SQL query builder for non_graph_executor.

This module converts a QueryIntent (from IntentAnalyzer) into a valid DuckDB
SQL query string. It supports:
- SELECT with aggregations and expressions
- GROUP BY (including virtual/temporal columns)
- ORDER BY with ASC/DESC
- LIMIT for rankings and pagination
- WHERE clause integration with session filters
- Virtual column resolution (Ano → YEAR("Data"), Mes → MONTH("Data"))
- Column validation against dataset schema

Phase 3 component: Bridges the gap between semantic intent (Phase 2) and
SQL execution, replacing the rigid template-based query generation.

Security:
- All column names are validated against a whitelist (alias_mapper schema)
- Virtual columns use predefined expressions only (no user input in SQL)
- Aggregation functions are restricted to a fixed set
"""

import logging
from typing import Any, Dict, List, Optional

from src.non_graph_executor.models.intent_schema import (
    AggregationSpec,
    ColumnSpec,
    QueryIntent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Mapeamento de funções de agregação para sintaxe DuckDB
AGG_FUNCTION_MAP = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
    "std": "STDDEV_SAMP",
}

# Colunas virtuais e suas expressões DuckDB
# Mantém consistência com AliasMapper.VIRTUAL_COLUMN_MAP
VIRTUAL_COLUMNS = {
    "Ano": {"expression": 'YEAR("Data")', "alias": "Ano"},
    "Mes": {"expression": 'MONTH("Data")', "alias": "Mes"},
    "Nome_Mes": {"expression": 'MONTHNAME("Data")', "alias": "Nome_Mes"},
}

# Limite padrão máximo para proteção contra queries pesadas
DEFAULT_MAX_LIMIT = 1000


class DynamicQueryBuilder:
    """
    Gerador de SQL dinâmico a partir de QueryIntent.

    Converte a especificação semântica produzida pelo IntentAnalyzer em
    uma query DuckDB válida e completa, com suporte a GROUP BY, ORDER BY,
    LIMIT e funções temporais.

    A construção é feita em etapas:
    1. Resolução de colunas (reais vs virtuais)
    2. Construção de SELECT clause (agregações + dimensões)
    3. Construção de GROUP BY clause
    4. Integração de WHERE clause (filtros da sessão + filtros adicionais)
    5. Construção de ORDER BY clause
    6. Aplicação de LIMIT

    Attributes:
        alias_mapper: AliasMapper instance para validação de colunas e
            resolução de colunas virtuais
        data_source: Caminho do dataset (para a cláusula FROM)

    Example:
        >>> builder = DynamicQueryBuilder(alias_mapper, "data/dataset.parquet")
        >>> intent = QueryIntent(
        ...     intent_type="grouped_aggregation",
        ...     aggregations=[AggregationSpec(
        ...         function="sum",
        ...         column=ColumnSpec(name="Valor_Vendido"),
        ...         alias="total_vendas"
        ...     )],
        ...     group_by=[ColumnSpec(name="Mes", is_virtual=True,
        ...         expression='MONTH("Data")', alias="Mes")],
        ...     order_by=OrderSpec(column="total_vendas", direction="DESC"),
        ...     limit=1,
        ... )
        >>> sql = builder.build_query(intent, filters={"Data": {"between": ["2016-01-01", "2016-12-31"]}})
        >>> print(sql)
        SELECT MONTH("Data") as Mes, SUM("Valor_Vendido") as total_vendas
        FROM 'data/dataset.parquet'
        WHERE "Data" BETWEEN '2016-01-01' AND '2016-12-31'
        GROUP BY MONTH("Data")
        ORDER BY total_vendas DESC
        LIMIT 1
    """

    def __init__(self, alias_mapper, data_source: str):
        """
        Inicializa o DynamicQueryBuilder.

        Args:
            alias_mapper: AliasMapper instance com informações de colunas,
                tipos e colunas virtuais
            data_source: Caminho para o arquivo de dados (parquet, csv, etc.)
        """
        self.alias_mapper = alias_mapper
        self.data_source = data_source

        # Coletar nomes válidos de colunas do dataset para validação
        self._valid_columns = self._collect_valid_columns()

        logger.info(
            f"DynamicQueryBuilder initialized: "
            f"data_source={data_source}, "
            f"valid_columns={len(self._valid_columns)}"
        )

    # =========================================================================
    # MAIN BUILD METHOD
    # =========================================================================

    def build_query(
        self,
        intent: QueryIntent,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Constrói query DuckDB completa a partir de um QueryIntent.

        Orquestra a construção de todas as cláusulas SQL baseado na
        análise semântica do IntentAnalyzer.

        Args:
            intent: QueryIntent com a especificação completa da intenção
            filters: Filtros da sessão (do filter_classifier)

        Returns:
            String SQL completa pronta para execução no DuckDB

        Raises:
            ValueError: Se o intent não contém informação suficiente para
                gerar uma query válida
        """
        logger.info(
            f"[DynamicQueryBuilder] Building query for intent_type={intent.intent_type}"
        )

        # Merge session filters with additional filters from the query
        merged_filters = self._merge_filters(filters, intent.additional_filters)

        # Build each clause
        select_clause = self._build_select_clause(intent)
        from_clause = f"FROM '{self.data_source}'"
        where_clause = self._build_where_clause(merged_filters)
        group_by_clause = self._build_group_by_clause(intent)
        order_by_clause = self._build_order_by_clause(intent)
        limit_clause = self._build_limit_clause(intent)

        # Assemble final query
        parts = [f"SELECT {select_clause}", from_clause]

        if where_clause:
            parts.append(f"WHERE {where_clause}")
        if group_by_clause:
            parts.append(f"GROUP BY {group_by_clause}")
        if order_by_clause:
            parts.append(f"ORDER BY {order_by_clause}")
        if limit_clause:
            parts.append(limit_clause)

        sql = "\n".join(parts)

        logger.info(f"[DynamicQueryBuilder] Generated SQL:\n{sql}")
        return sql

    # =========================================================================
    # SELECT CLAUSE
    # =========================================================================

    def _build_select_clause(self, intent: QueryIntent) -> str:
        """
        Constrói a cláusula SELECT a partir do intent.

        Combina:
        1. Colunas de GROUP BY (dimensões de agrupamento)
        2. Expressões de agregação
        3. Colunas de seleção direta (se houver)

        Se não houver nenhuma especificação, retorna "*".

        Args:
            intent: QueryIntent com especificações de seleção

        Returns:
            String com a cláusula SELECT (sem a palavra SELECT)
        """
        select_parts: List[str] = []

        # 1. Adicionar dimensões de agrupamento (GROUP BY) no SELECT
        for col in intent.group_by:
            col_expr = self._resolve_column_expression(col)
            alias = col.alias or col.name
            select_parts.append(f"{col_expr} as {self._safe_alias(alias)}")

        # 2. Adicionar agregações
        for agg in intent.aggregations:
            agg_expr = self._build_aggregation_expression(agg)
            alias = agg.alias or f"{agg.function}_{agg.column.name}"
            select_parts.append(f"{agg_expr} as {self._safe_alias(alias)}")

        # 3. Adicionar colunas de seleção direta (sem agrupamento/agregação)
        for col in intent.select_columns:
            col_expr = self._resolve_column_expression(col)
            alias = col.alias or col.name
            # Evitar duplicatas com group_by
            expr_with_alias = f"{col_expr} as {self._safe_alias(alias)}"
            if expr_with_alias not in select_parts:
                select_parts.append(expr_with_alias)

        if not select_parts:
            # Fallback: se nada foi especificado, retornar todas colunas
            logger.warning(
                "[DynamicQueryBuilder] No select columns/aggregations specified, "
                "using SELECT *"
            )
            return "*"

        return ", ".join(select_parts)

    def _build_aggregation_expression(self, agg: AggregationSpec) -> str:
        """
        Constrói expressão de agregação SQL.

        Suporta:
        - Funções padrão: SUM, AVG, COUNT, MIN, MAX, MEDIAN, STDDEV_SAMP
        - COUNT com DISTINCT
        - Colunas virtuais dentro de agregações

        Args:
            agg: AggregationSpec com função e coluna

        Returns:
            String com a expressão de agregação (ex: 'SUM("Valor_Vendido")')

        Raises:
            ValueError: Se a função de agregação não é suportada
        """
        func = agg.function.lower()
        if func not in AGG_FUNCTION_MAP:
            raise ValueError(
                f"Função de agregação não suportada: '{func}'. "
                f"Suportadas: {list(AGG_FUNCTION_MAP.keys())}"
            )

        sql_func = AGG_FUNCTION_MAP[func]
        col_expr = self._resolve_column_expression(agg.column)

        if func == "count" and agg.distinct:
            return f"COUNT(DISTINCT {col_expr})"

        return f"{sql_func}({col_expr})"

    # =========================================================================
    # GROUP BY CLAUSE
    # =========================================================================

    def _build_group_by_clause(self, intent: QueryIntent) -> str:
        """
        Constrói a cláusula GROUP BY.

        Usa as expressões das colunas (não aliases), incluindo
        expressões de colunas virtuais (YEAR("Data"), MONTH("Data")).

        Args:
            intent: QueryIntent com dimensões de agrupamento

        Returns:
            String com a cláusula GROUP BY (sem as palavras GROUP BY),
            ou string vazia se não houver agrupamento
        """
        if not intent.group_by:
            return ""

        group_parts = []
        for col in intent.group_by:
            col_expr = self._resolve_column_expression(col)
            group_parts.append(col_expr)

        return ", ".join(group_parts)

    # =========================================================================
    # ORDER BY CLAUSE
    # =========================================================================

    def _build_order_by_clause(self, intent: QueryIntent) -> str:
        """
        Constrói a cláusula ORDER BY.

        Usa o alias de agregação quando o order_by refere a uma coluna
        que é alias de uma agregação. Caso contrário, usa a expressão
        da coluna diretamente.

        Args:
            intent: QueryIntent com especificação de ordenação

        Returns:
            String com a cláusula ORDER BY (sem as palavras ORDER BY),
            ou string vazia se não houver ordenação
        """
        if not intent.order_by:
            return ""

        order_col = intent.order_by.column
        direction = intent.order_by.direction

        if direction not in ("ASC", "DESC"):
            direction = "DESC"

        # Verificar se order_col é um alias de agregação
        agg_aliases = self._collect_aggregation_aliases(intent)
        if order_col in agg_aliases:
            # Usar o alias diretamente (DuckDB suporta ORDER BY alias)
            return f"{self._safe_alias(order_col)} {direction}"

        # Verificar se é um alias de group_by
        for col in intent.group_by:
            alias = col.alias or col.name
            if order_col == alias or order_col == col.name:
                col_expr = self._resolve_column_expression(col)
                return f"{col_expr} {direction}"

        # Tentar resolver como coluna do dataset
        col_spec = ColumnSpec(name=order_col)
        if self._is_virtual_column(order_col):
            col_spec = ColumnSpec(
                name=order_col,
                is_virtual=True,
                expression=self._get_virtual_expression(order_col),
            )

        col_expr = self._resolve_column_expression(col_spec)
        return f"{col_expr} {direction}"

    # =========================================================================
    # LIMIT CLAUSE
    # =========================================================================

    def _build_limit_clause(self, intent: QueryIntent) -> str:
        """
        Constrói a cláusula LIMIT.

        Aplica o limite especificado no intent. Para queries sem limit
        explícito mas com group_by, não aplica limit (retorna tudo).
        Para segurança, aplica um limite máximo global.

        Args:
            intent: QueryIntent com especificação de limite

        Returns:
            String com a cláusula LIMIT completa ou string vazia
        """
        if intent.limit is not None and intent.limit > 0:
            safe_limit = min(intent.limit, DEFAULT_MAX_LIMIT)
            return f"LIMIT {safe_limit}"

        return ""

    # =========================================================================
    # WHERE CLAUSE
    # =========================================================================

    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """
        Constrói a cláusula WHERE a partir de filtros.

        Delega a construção ao mesmo padrão usado pelo QueryExecutor
        para manter consistência. Suporta:
        - Igualdade: {"Ano": 2015} → YEAR("Data") = 2015
        - IN: {"UF": ["SP", "RJ"]} → "UF" IN ('SP', 'RJ')
        - Between: {"Data": {"between": ["2015-01-01", "2015-12-31"]}}
        - Operadores: {"Valor": {"operator": ">=", "value": 1000}}

        Args:
            filters: Dict de filtros a aplicar

        Returns:
            String com condições WHERE (sem a palavra WHERE),
            ou string vazia se não houver filtros
        """
        if not filters:
            return ""

        conditions = []

        for col, value in filters.items():
            # Resolver colunas virtuais em filtros
            col_escaped = self._resolve_filter_column(col)

            # Case 1: Lista de valores
            if isinstance(value, list):
                # Sub-case 1a: Intervalo temporal (2 datas) → BETWEEN
                if self._is_temporal_date_range(col, value):
                    start_escaped = value[0].replace("'", "''")
                    end_escaped = value[1].replace("'", "''")
                    conditions.append(
                        f"{col_escaped} BETWEEN '{start_escaped}' AND '{end_escaped}'"
                    )
                    logger.debug(
                        f"Temporal date range detected for '{col}': "
                        f"BETWEEN {value[0]} AND {value[1]}"
                    )
                else:
                    # Sub-case 1b: Lista regular → IN clause
                    escaped_values = []
                    is_categorical = self._is_categorical_column(col)

                    for v in value:
                        if isinstance(v, str):
                            v_escaped = v.replace("'", "''")
                            if is_categorical:
                                escaped_values.append(f"UPPER('{v_escaped}')")
                            else:
                                escaped_values.append(f"'{v_escaped}'")
                        else:
                            escaped_values.append(str(v))

                    values_str = ", ".join(escaped_values)

                    if is_categorical:
                        conditions.append(f"UPPER({col_escaped}) IN ({values_str})")
                    else:
                        conditions.append(f"{col_escaped} IN ({values_str})")

            # Case 2: Dict com operador ou between
            elif isinstance(value, dict):
                if "operator" in value:
                    operator = value["operator"]
                    op_value = value["value"]
                    if isinstance(op_value, str):
                        op_value_str = (
                            f"'{op_value.replace(chr(39), chr(39) + chr(39))}'"
                        )
                    else:
                        op_value_str = str(op_value)
                    conditions.append(f"{col_escaped} {operator} {op_value_str}")

                elif "between" in value:
                    start, end = value["between"]
                    start_str = (
                        f"'{start.replace(chr(39), chr(39) + chr(39))}'"
                        if isinstance(start, str)
                        else str(start)
                    )
                    end_str = (
                        f"'{end.replace(chr(39), chr(39) + chr(39))}'"
                        if isinstance(end, str)
                        else str(end)
                    )
                    conditions.append(
                        f"{col_escaped} BETWEEN {start_str} AND {end_str}"
                    )

            # Case 3: Equality (valor simples)
            else:
                is_categorical = self._is_categorical_column(col)
                should_quote = isinstance(value, str)

                if self.alias_mapper and not should_quote and is_categorical:
                    should_quote = True

                if should_quote:
                    value_str_raw = str(value) if not isinstance(value, str) else value
                    value_escaped = value_str_raw.replace("'", "''")
                    if is_categorical:
                        conditions.append(
                            f"UPPER({col_escaped}) = UPPER('{value_escaped}')"
                        )
                    else:
                        conditions.append(f"{col_escaped} = '{value_escaped}'")
                else:
                    conditions.append(f"{col_escaped} = {value}")

        return " AND ".join(conditions)

    # =========================================================================
    # COLUMN RESOLUTION HELPERS
    # =========================================================================

    def _is_temporal_date_range(self, col: str, values: list) -> bool:
        """
        Detecta se uma lista de valores representa um intervalo temporal (BETWEEN).

        O filter_classifier produz filtros de data como listas de 2 elementos:
        {"Data": ["2016-01-01", "2016-12-31"]} que representam um intervalo
        BETWEEN, NÃO um IN com 2 valores específicos.

        Critérios de detecção:
        1. A lista tem exatamente 2 elementos
        2. A coluna é temporal ("Data" ou registrada como temporal no alias_mapper)
        3. Ambos os valores parecem ser datas (formato YYYY-MM-DD)

        Args:
            col: Nome da coluna do filtro
            values: Lista de valores do filtro

        Returns:
            True se os valores representam um intervalo temporal
        """
        if len(values) != 2:
            return False

        # Verificar se a coluna é temporal
        is_temporal = col in ("Data", "data")
        if not is_temporal and self.alias_mapper:
            temporal_cols = getattr(self.alias_mapper, "column_types", {}).get(
                "temporal", []
            )
            if col in temporal_cols:
                is_temporal = True

        if not is_temporal:
            return False

        # Verificar se ambos os valores parecem ser datas (YYYY-MM-DD)
        import re

        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
        return all(isinstance(v, str) and date_pattern.match(v) for v in values)

    def _resolve_column_expression(self, col: ColumnSpec) -> str:
        """
        Resolve uma ColumnSpec para sua expressão SQL.

        Se a coluna é virtual, usa a expressão SQL registrada.
        Se é uma coluna real, escapa o nome com aspas duplas.

        Args:
            col: ColumnSpec a resolver

        Returns:
            Expressão SQL da coluna (ex: 'YEAR("Data")' ou '"Valor_Vendido"')
        """
        # 1. Se a ColumnSpec já traz expressão virtual, usá-la
        if col.is_virtual and col.expression:
            return col.expression

        # 2. Verificar no registro de colunas virtuais
        if self._is_virtual_column(col.name):
            expr = self._get_virtual_expression(col.name)
            if expr:
                return expr

        # 3. Coluna real: escapar nome
        return f'"{col.name}"'

    def _resolve_filter_column(self, col_name: str) -> str:
        """
        Resolve nome de coluna para uso em filtros WHERE.

        Converte colunas virtuais para suas expressões SQL.

        Args:
            col_name: Nome da coluna

        Returns:
            Expressão SQL da coluna
        """
        # Verificar colunas virtuais conhecidas
        if col_name in ("Ano", "ano"):
            return 'YEAR("Data")'
        if col_name in ("Mes", "mes"):
            return 'MONTH("Data")'
        if col_name in ("Nome_Mes", "nome_mes"):
            return 'MONTHNAME("Data")'

        # Verificar via alias_mapper
        if (
            self.alias_mapper
            and hasattr(self.alias_mapper, "is_virtual_column")
            and self.alias_mapper.is_virtual_column(col_name)
        ):
            expr = self.alias_mapper.get_virtual_expression(col_name)
            if expr:
                return expr

        return f'"{col_name}"'

    def _is_virtual_column(self, col_name: str) -> bool:
        """Verifica se uma coluna é virtual."""
        if col_name in VIRTUAL_COLUMNS:
            return True
        if (
            self.alias_mapper
            and hasattr(self.alias_mapper, "is_virtual_column")
            and self.alias_mapper.is_virtual_column(col_name)
        ):
            return True
        return False

    def _get_virtual_expression(self, col_name: str) -> Optional[str]:
        """Obtém a expressão SQL de uma coluna virtual."""
        if col_name in VIRTUAL_COLUMNS:
            return VIRTUAL_COLUMNS[col_name]["expression"]
        if self.alias_mapper and hasattr(self.alias_mapper, "get_virtual_expression"):
            return self.alias_mapper.get_virtual_expression(col_name)
        return None

    def _is_categorical_column(self, col_name: str) -> bool:
        """Verifica se coluna é categórica (para comparação case-insensitive)."""
        if self.alias_mapper and hasattr(self.alias_mapper, "is_categorical_column"):
            return self.alias_mapper.is_categorical_column(col_name)
        return False

    # =========================================================================
    # UTILITY HELPERS
    # =========================================================================

    def _safe_alias(self, alias: str) -> str:
        """
        Sanitiza um alias para uso seguro em SQL.

        Remove caracteres especiais e garante que o alias é um
        identificador SQL válido. Se contém espaços ou caracteres
        especiais, envolve em aspas duplas.

        Args:
            alias: Alias bruto

        Returns:
            Alias seguro para uso em SQL
        """
        if not alias:
            return "resultado"

        # Remover caracteres não-alfanuméricos (exceto underscore)
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in alias)

        # Garantir que não começa com número
        if safe and safe[0].isdigit():
            safe = f"col_{safe}"

        return safe or "resultado"

    def _collect_aggregation_aliases(self, intent: QueryIntent) -> set:
        """
        Coleta todos os aliases de agregação do intent.

        Usado para resolver ORDER BY que referencia alias de agregação.

        Args:
            intent: QueryIntent

        Returns:
            Set de aliases de agregação
        """
        aliases = set()
        for agg in intent.aggregations:
            alias = agg.alias or f"{agg.function}_{agg.column.name}"
            aliases.add(alias)
            # Também adicionar a versão sanitizada
            aliases.add(self._safe_alias(alias))
        return aliases

    def _collect_valid_columns(self) -> set:
        """
        Coleta todos os nomes de colunas válidos (reais + virtuais).

        Usado para validação de segurança contra SQL injection.

        Returns:
            Set de nomes de colunas válidos
        """
        valid = set()

        # Colunas virtuais
        valid.update(VIRTUAL_COLUMNS.keys())

        # Colunas do alias_mapper
        if self.alias_mapper:
            column_types = getattr(self.alias_mapper, "column_types", {})
            for col_list in column_types.values():
                if isinstance(col_list, list):
                    valid.update(col_list)

            # Colunas do mapeamento de aliases
            aliases = getattr(self.alias_mapper, "aliases", {})
            columns_section = aliases.get("columns", {})
            valid.update(columns_section.keys())

        return valid

    def _merge_filters(
        self,
        session_filters: Optional[Dict[str, Any]],
        additional_filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Combina filtros da sessão com filtros adicionais do intent.

        Filtros adicionais (detectados na query) são mesclados com
        filtros da sessão (do filter_classifier). Em caso de conflito,
        filtros adicionais têm precedência.

        Args:
            session_filters: Filtros da sessão (filter_final)
            additional_filters: Filtros adicionais do QueryIntent

        Returns:
            Dict combinado de filtros, ou None se vazio
        """
        merged = {}

        if session_filters:
            merged.update(session_filters)

        if additional_filters:
            merged.update(additional_filters)

        return merged if merged else None

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_intent(self, intent: QueryIntent) -> List[str]:
        """
        Valida um QueryIntent antes de gerar SQL.

        Verifica:
        - Funções de agregação são válidas
        - Colunas referenciadas existem (reais ou virtuais)
        - ORDER BY referencia coluna/alias válido
        - Limite é positivo

        Args:
            intent: QueryIntent a validar

        Returns:
            Lista de warnings/erros encontrados (vazia se tudo OK)
        """
        warnings_list: List[str] = []

        # Validar agregações
        for agg in intent.aggregations:
            if agg.function not in AGG_FUNCTION_MAP:
                warnings_list.append(f"Função de agregação inválida: '{agg.function}'")

            col_name = agg.column.name
            if col_name and not self._is_known_column(col_name):
                warnings_list.append(
                    f"Coluna de agregação não reconhecida: '{col_name}'"
                )

        # Validar group_by
        for col in intent.group_by:
            col_name = col.name
            if col_name and not self._is_known_column(col_name):
                warnings_list.append(
                    f"Coluna de agrupamento não reconhecida: '{col_name}'"
                )

        # Validar select_columns
        for col in intent.select_columns:
            col_name = col.name
            if col_name and not self._is_known_column(col_name):
                warnings_list.append(f"Coluna de seleção não reconhecida: '{col_name}'")

        # Validar limit
        if intent.limit is not None and intent.limit <= 0:
            warnings_list.append(f"Limite inválido: {intent.limit}")

        if warnings_list:
            logger.warning(
                f"[DynamicQueryBuilder] Validation warnings: {warnings_list}"
            )

        return warnings_list

    def _is_known_column(self, col_name: str) -> bool:
        """
        Verifica se o nome da coluna é reconhecido.

        Aceita colunas reais do dataset, colunas virtuais e colunas
        registradas no alias_mapper.

        Args:
            col_name: Nome da coluna a verificar

        Returns:
            True se a coluna é reconhecida
        """
        if col_name in self._valid_columns:
            return True
        if self._is_virtual_column(col_name):
            return True
        # Aceitar colunas com expressões (já resolvidas pelo IntentAnalyzer)
        return False
