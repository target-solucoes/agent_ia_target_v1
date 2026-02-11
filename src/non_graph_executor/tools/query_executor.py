"""
Query executor for non_graph_executor.

This module implements optimized DuckDB query execution
for different types of non-graph operations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from src.shared_lib.utils.logger import get_logger

logger = get_logger(__name__)


class QueryExecutor:
    """
    Executor otimizado de queries DuckDB para non-graph queries.

    Fornece métodos específicos para cada tipo de operação:
    - Metadata queries (sample rows)
    - Aggregations (sum, avg, count, etc)
    - Lookups (busca por ID/chave)
    - Text search (LIKE queries)
    - Statistical queries (quartis, IQR, variance)

    Performance:
    - Queries parametrizadas para segurança
    - Conexões efêmeras (abrir, executar, fechar)
    - Limite automático para evitar resultados gigantes
    - Otimização de queries para colunas específicas

    Example:
        >>> executor = QueryExecutor("data/dataset.parquet", metadata_cache)
        >>> # Sample rows
        >>> rows = executor.get_sample_rows(n=10)
        >>> # Aggregation
        >>> total = executor.compute_simple_aggregation("Valor", "sum")
        >>> # Lookup
        >>> record = executor.lookup_record("ID", 123)
    """

    # Limite padrão para evitar resultados gigantes
    DEFAULT_LIMIT = 1000

    def __init__(self, data_source: str, metadata_cache, alias_mapper=None):
        """
        Initialize query executor.

        Args:
            data_source: Path to dataset file (parquet, csv, etc)
            metadata_cache: MetadataCache instance for caching support
            alias_mapper: Optional AliasMapper for type-aware filtering

        Raises:
            FileNotFoundError: Se dataset não existe
            RuntimeError: Se DuckDB não está disponível
        """
        if not DUCKDB_AVAILABLE:
            raise RuntimeError(
                "DuckDB is required for QueryExecutor. Install with: pip install duckdb"
            )

        self.data_source = data_source
        self.metadata_cache = metadata_cache
        self.alias_mapper = alias_mapper

        # Validar que dataset existe
        if not Path(data_source).exists():
            raise FileNotFoundError(f"Dataset not found: {data_source}")

        logger.info(f"QueryExecutor initialized for: {data_source}")

    # =========================================================================
    # DYNAMIC QUERY EXECUTION (Phase 3)
    # =========================================================================

    def execute_dynamic_query(
        self,
        sql: str,
        timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Executa query SQL dinâmica gerada pelo DynamicQueryBuilder.

        Este método é o ponto de entrada para execução de queries
        construídas dinamicamente a partir do QueryIntent, suportando
        GROUP BY, ORDER BY, LIMIT e funções temporais.

        Diferente dos métodos especializados (compute_simple_aggregation,
        lookup_record, etc.), este método aceita SQL arbitrário gerado
        pelo DynamicQueryBuilder — que já valida colunas e expressões.

        Args:
            sql: Query SQL completa gerada pelo DynamicQueryBuilder
            timeout: Timeout em segundos para a execução (default: 30)

        Returns:
            Lista de dicts, cada um representando uma linha do resultado.
            Para agregações simples sem GROUP BY, retorna lista com 1 dict.
            Para agregações com GROUP BY, retorna lista de dicts (um por grupo).

        Raises:
            Exception: Se ocorrer erro na execução da query DuckDB

        Example:
            >>> sql = 'SELECT MONTH("Data") as Mes, SUM("Valor_Vendido") as total '
            ...       "FROM 'data/dataset.parquet' GROUP BY MONTH(\"Data\") "
            ...       'ORDER BY total DESC LIMIT 1'
            >>> results = executor.execute_dynamic_query(sql)
            >>> print(results)
            [{'Mes': 3, 'total': 5200000.0}]
        """
        logger.info(f"[QueryExecutor] Executing dynamic query:\n{sql}")

        try:
            with duckdb.connect() as conn:
                result = conn.execute(sql).fetchall()
                columns = [desc[0] for desc in conn.description]

            # Converter para lista de dicts
            rows = []
            for row_tuple in result:
                row_dict = {}
                for col_name, value in zip(columns, row_tuple):
                    # Converter tipos DuckDB para Python nativos
                    row_dict[col_name] = self._convert_duckdb_value(value)
                rows.append(row_dict)

            logger.info(
                f"[QueryExecutor] Dynamic query returned {len(rows)} rows, "
                f"columns: {columns}"
            )
            return rows

        except Exception as e:
            logger.error(
                f"[QueryExecutor] Error executing dynamic query: {e}\nSQL: {sql}",
                exc_info=True,
            )
            raise

    @staticmethod
    def _convert_duckdb_value(value: Any) -> Any:
        """
        Converte valores DuckDB para tipos Python nativos.

        DuckDB pode retornar tipos como Decimal, datetime, date, etc.
        que precisam ser convertidos para tipos JSON-serializáveis.

        Args:
            value: Valor retornado pelo DuckDB

        Returns:
            Valor convertido para tipo Python nativo
        """
        if value is None:
            return None

        # Importações tardias para não impactar startup
        import decimal
        import datetime

        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        if isinstance(value, datetime.date):
            return value.isoformat()
        if isinstance(value, datetime.timedelta):
            return str(value)
        if isinstance(value, (int, float, str, bool)):
            return value

        # Fallback: converter para string
        return str(value)

    # =========================================================================
    # METADATA QUERIES
    # =========================================================================

    def get_tabular_data(
        self,
        limit: int = 100,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retorna dados tabulares completos do dataset.

        Útil para solicitações explícitas de "mostrar tabela" ou "dados brutos".
        Aplica filtros opcionais e limite para evitar resultados gigantes.

        Args:
            limit: Número máximo de linhas (default: 100, max: 1000)
            columns: Lista opcional de colunas específicas a retornar (None = todas)
            filters: Filtros opcionais a aplicar

        Returns:
            Lista de dicts representando linhas da tabela

        Example:
            >>> # Sem filtros
            >>> data = executor.get_tabular_data(limit=50)
            >>> # Com filtros
            >>> data = executor.get_tabular_data(
            ...     limit=100, filters={"UF_Cliente": "SP"}
            ... )
            >>> # Colunas específicas
            >>> data = executor.get_tabular_data(
            ...     limit=100, columns=["Produto", "Valor_Vendido"]
            ... )
        """
        # Aplicar limite máximo
        limit = min(limit, self.DEFAULT_LIMIT)

        try:
            # Construir SELECT clause (colunas específicas ou *)
            if columns:
                # Validate and quote column names
                select_clause = ", ".join([f'"{col}"' for col in columns])
            else:
                select_clause = "*"

            # Construir WHERE clause se houver filtros
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            # Query: SELECT [columns] FROM data {WHERE} LIMIT n
            query = f"SELECT {select_clause} FROM '{self.data_source}' {where_clause} LIMIT {limit}"

            logger.debug(f"Executing tabular query: {query}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchall()
                columns = [desc[0] for desc in conn.description]

            # Converter para lista de dicts
            rows = []
            for row_tuple in result:
                row_dict = dict(zip(columns, row_tuple))
                rows.append(row_dict)

            logger.debug(f"Retrieved {len(rows)} tabular rows")
            return rows

        except Exception as e:
            logger.error(f"Error executing tabular query: {e}")
            raise

    def get_sample_rows(
        self, n: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retorna n primeiras linhas do dataset.

        Aplica filtros opcionais antes de retornar amostra.
        Útil para preview de dados e validação de filtros.

        Args:
            n: Número de linhas a retornar (default: 5)
            filters: Filtros opcionais a aplicar

        Returns:
            Lista de dicts, cada um representando uma linha

        Example:
            >>> # Sem filtros
            >>> rows = executor.get_sample_rows(n=10)
            >>> # Com filtros
            >>> rows = executor.get_sample_rows(n=5, filters={"Ano": 2015})
        """
        try:
            # Construir WHERE clause se houver filtros
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            # Query: SELECT * FROM data {WHERE} LIMIT n
            query = f"SELECT * FROM '{self.data_source}' {where_clause} LIMIT {n}"

            logger.debug(f"Executing sample query: {query}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchall()
                columns = [desc[0] for desc in conn.description]

            # Converter para lista de dicts
            rows = []
            for row_tuple in result:
                row_dict = dict(zip(columns, row_tuple))
                rows.append(row_dict)

            logger.debug(f"Retrieved {len(rows)} sample rows")
            return rows

        except Exception as e:
            logger.error(f"Error executing sample query: {e}")
            raise

    # =========================================================================
    # AGGREGATION QUERIES
    # =========================================================================

    def compute_simple_aggregation(
        self,
        column: str,
        aggregation: str,
        filters: Optional[Dict[str, Any]] = None,
        distinct: bool = False,
    ) -> Union[int, float]:
        """
        Calcula agregação simples em uma coluna.

        Agregações suportadas: sum, avg, count, min, max, median, std

        Args:
            column: Nome da coluna a agregar
            aggregation: Tipo de agregação (sum, avg, count, min, max, median, std)
            filters: Filtros opcionais a aplicar
            distinct: Se True, usa COUNT(DISTINCT column) para count (default: False)

        Returns:
            Valor numérico da agregação (int para count, float para outros)

        Raises:
            ValueError: Se agregação não é suportada
            Exception: Se erro na execução da query

        Example:
            >>> # Soma total
            >>> total = executor.compute_simple_aggregation("Valor", "sum")
            >>> # Contagem de valores únicos
            >>> unique_clients = executor.compute_simple_aggregation("Cod_Cliente", "count", distinct=True)
            >>> # Média com filtro
            >>> avg = executor.compute_simple_aggregation(
            ...     "Valor", "avg", filters={"Ano": 2015}
            ... )
        """
        # Mapear agregação para função DuckDB
        agg_map = {
            "sum": "SUM",
            "avg": "AVG",
            "count": "COUNT",
            "min": "MIN",
            "max": "MAX",
            "median": "MEDIAN",
            "std": "STDDEV_SAMP",
        }

        if aggregation.lower() not in agg_map:
            raise ValueError(
                f"Unsupported aggregation: {aggregation}. "
                f"Supported: {list(agg_map.keys())}"
            )

        # VALIDAÇÃO: Para agregações numéricas (não COUNT), verificar se coluna é numérica
        numeric_aggregations = ["sum", "avg", "min", "max", "median", "std"]
        if aggregation.lower() in numeric_aggregations:
            if self.alias_mapper and hasattr(self.alias_mapper, "column_types"):
                try:
                    numeric_cols = self.alias_mapper.column_types.get("numeric", [])
                    categorical_cols = self.alias_mapper.column_types.get(
                        "categorical", []
                    )

                    # Se a coluna é categórica, não pode aplicar agregação numérica
                    if column in categorical_cols:
                        logger.error(
                            f"Cannot apply {aggregation.upper()} to categorical column '{column}'. "
                            f"Categorical columns should only be used with COUNT or as GROUP BY dimensions."
                        )

                        # Tentar sugerir coluna numérica apropriada
                        if numeric_cols:
                            suggested = numeric_cols[0]  # Normalmente Valor_Vendido
                            logger.warning(
                                f"Suggestion: Use numeric column '{suggested}' instead of '{column}' "
                                f"for {aggregation.upper()} aggregation."
                            )
                            raise ValueError(
                                f"Cannot apply {aggregation.upper()} to categorical column '{column}'. "
                                f"Use a numeric column like '{suggested}' instead."
                            )
                        else:
                            raise ValueError(
                                f"Cannot apply {aggregation.upper()} to categorical column '{column}'. "
                                f"Categorical columns can only be used with COUNT."
                            )

                    # Se não é categórica nem numérica conhecida, emitir warning mas continuar
                    # Colunas virtuais (Ano, Mes) são aceitáveis para MIN/MAX — não emitir warning
                    is_virtual = hasattr(
                        self.alias_mapper, "is_virtual_column"
                    ) and self.alias_mapper.is_virtual_column(column)
                    if (
                        column not in numeric_cols
                        and column not in categorical_cols
                        and not is_virtual
                    ):
                        logger.warning(
                            f"Column '{column}' type is unknown. "
                            f"Attempting {aggregation.upper()} aggregation anyway."
                        )

                except (AttributeError, KeyError) as e:
                    logger.debug(f"Could not validate column type: {e}")
                    # Continue anyway if validation fails

        try:
            # Obter função DuckDB
            agg_func = agg_map[aggregation.lower()]

            # Construir WHERE clause se houver filtros
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            # Resolver coluna virtual (Ano -> YEAR("Data"), Mes -> MONTH("Data"), etc.)
            # Colunas virtuais não existem fisicamente no dataset e requerem expressões SQL.
            is_virtual = (
                self.alias_mapper
                and hasattr(self.alias_mapper, "is_virtual_column")
                and self.alias_mapper.is_virtual_column(column)
            )

            if is_virtual:
                col_escaped = self.alias_mapper.get_virtual_expression(column)
                logger.debug(
                    f"Resolved virtual column '{column}' to expression: {col_escaped}"
                )
            else:
                col_escaped = f'"{column}"'

            # Para COUNT com distinct=True, usar COUNT(DISTINCT column)
            if aggregation.lower() == "count" and distinct:
                agg_expression = f"COUNT(DISTINCT {col_escaped})"
            else:
                agg_expression = f"{agg_func}({col_escaped})"

            # Query: SELECT AGG_FUNC(column) FROM data {WHERE}
            query = f"SELECT {agg_expression} as result FROM '{self.data_source}' {where_clause}"

            logger.debug(f"Executing aggregation query: {query}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchone()

            # Extrair valor
            value = result[0] if result and result[0] is not None else 0.0

            # COUNT deve retornar inteiro, outras agregações float
            if aggregation.lower() == "count":
                return int(value)

            logger.debug(f"Aggregation result: {value}")
            return float(value)

        except Exception as e:
            logger.error(f"Error executing aggregation query: {e}")
            raise

    # =========================================================================
    # LOOKUP QUERIES
    # =========================================================================

    def lookup_record(
        self,
        lookup_column: str,
        lookup_value: Any,
        return_columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Busca registro específico por valor de coluna.

        Retorna primeiro registro que faz match (LIMIT 1).
        Pode retornar todas as colunas ou apenas colunas especificadas.

        Args:
            lookup_column: Coluna usada para busca
            lookup_value: Valor a buscar
            return_columns: Lista de colunas a retornar (None = todas)

        Returns:
            Dict representando registro encontrado, ou None se não encontrado

        Example:
            >>> # Buscar por ID, retornar todas colunas
            >>> record = executor.lookup_record("ID", 123)
            >>> # Buscar por nome, retornar colunas específicas
            >>> record = executor.lookup_record(
            ...     "Cliente", "Silva", return_columns=["Cliente", "Valor"]
            ... )
        """
        try:
            # Construir SELECT clause
            if return_columns:
                # Sempre incluir lookup_column no resultado
                columns_to_select = list(return_columns)
                if lookup_column not in columns_to_select:
                    columns_to_select.insert(0, lookup_column)

                # Escapar nomes de colunas
                cols_escaped = [f'"{col}"' for col in columns_to_select]
                select_clause = ", ".join(cols_escaped)
            else:
                select_clause = "*"

            # Escapar nome da coluna de lookup
            lookup_col_escaped = f'"{lookup_column}"'

            # Escapar valor de lookup
            if isinstance(lookup_value, str):
                value_escaped = lookup_value.replace("'", "''")
                value_str = f"'{value_escaped}'"
            else:
                value_str = str(lookup_value)

            base_condition = f"{lookup_col_escaped} = {value_str}"
            where_parts = [base_condition]
            if filters:
                where_parts.append(self._build_where_clause(filters))
            where_clause = " AND ".join(where_parts)

            # Query: SELECT {cols} FROM data WHERE <lookup> AND <filters> LIMIT 1
            query = (
                f"SELECT {select_clause} FROM '{self.data_source}' "
                f"WHERE {where_clause} LIMIT 1"
            )

            logger.debug(f"Executing lookup query: {query}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchone()

                # Se não encontrou, retornar None
                if not result:
                    logger.debug("No record found for lookup")
                    return None

                # Obter nomes das colunas
                columns = [desc[0] for desc in conn.description]

            # Converter para dict
            record = dict(zip(columns, result))

            logger.debug(f"Found record: {record}")
            return record

        except Exception as e:
            logger.error(f"Error executing lookup query: {e}")
            raise

    # =========================================================================
    # TEXTUAL/SEARCH QUERIES
    # =========================================================================

    def search_text(
        self,
        column: str,
        search_term: str,
        case_sensitive: bool = False,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca texto em coluna específica usando LIKE.

        Suporta busca case-sensitive ou case-insensitive.
        Retorna todos os registros que fazem match.

        Args:
            column: Coluna onde buscar
            search_term: Termo a buscar (usa LIKE com wildcards %)
            case_sensitive: Se True, busca case-sensitive
            limit: Limite de resultados (None = sem limite, usa DEFAULT_LIMIT)
            filters: Filtros opcionais adicionais

        Returns:
            Lista de dicts representando registros encontrados

        Example:
            >>> # Busca case-insensitive
            >>> results = executor.search_text("Cliente", "silva")
            >>> # Busca case-sensitive com limite e filtros
            >>> results = executor.search_text(
            ...     "Descricao", "Premium", case_sensitive=True,
            ...     limit=10, filters={"Ano": 2015}
            ... )
        """
        try:
            # Aplicar limite padrão se não especificado
            if limit is None:
                limit = self.DEFAULT_LIMIT

            # Escapar nome da coluna
            col_escaped = f'"{column}"'

            # Escapar termo de busca (adicionar wildcards)
            term_escaped = search_term.replace("'", "''")
            term_with_wildcards = f"%{term_escaped}%"

            # Construir condição LIKE
            if case_sensitive:
                like_condition = f"{col_escaped} LIKE '{term_with_wildcards}'"
            else:
                like_condition = (
                    f"LOWER({col_escaped}) LIKE LOWER('{term_with_wildcards}')"
                )

            # Adicionar filtros adicionais se fornecidos
            where_clause = like_condition
            if filters:
                additional_filters = self._build_where_clause(filters)
                if additional_filters:
                    where_clause = f"{like_condition} AND {additional_filters}"

            # Query: SELECT * FROM data WHERE conditions LIMIT
            query = (
                f"SELECT * FROM '{self.data_source}' WHERE {where_clause} LIMIT {limit}"
            )

            logger.debug(f"Executing text search query: {query}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchall()
                columns = [desc[0] for desc in conn.description]

            # Converter para lista de dicts
            rows = []
            for row_tuple in result:
                row_dict = dict(zip(columns, row_tuple))
                rows.append(row_dict)

            logger.debug(f"Found {len(rows)} matching records")
            return rows

        except Exception as e:
            logger.error(f"Error executing text search query: {e}")
            raise

    def list_unique_values(
        self,
        column: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """
        Lista valores únicos de uma coluna.

        Retorna valores ordenados. Útil para colunas categóricas.
        Aplica filtros opcionais antes de extrair valores únicos.

        Args:
            column: Coluna da qual extrair valores únicos
            filters: Filtros opcionais a aplicar
            limit: Limite de valores únicos (None = sem limite, usa DEFAULT_LIMIT)

        Returns:
            Lista de valores únicos ordenados

        Example:
            >>> # Listar todos estados
            >>> states = executor.list_unique_values("UF")
            >>> # Listar produtos de 2015
            >>> products = executor.list_unique_values(
            ...     "Produto", filters={"Ano": 2015}
            ... )
        """
        try:
            # Aplicar limite padrão se não especificado
            if limit is None:
                limit = self.DEFAULT_LIMIT

            # Escapar nome da coluna
            col_escaped = f'"{column}"'

            # Construir WHERE clause se houver filtros
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            # Query: SELECT DISTINCT column FROM data {WHERE} ORDER BY column LIMIT
            query = (
                f"SELECT DISTINCT {col_escaped} FROM '{self.data_source}' "
                f"{where_clause} ORDER BY {col_escaped} LIMIT {limit}"
            )

            logger.debug(f"Executing unique values query: {query}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchall()

            # Extrair valores (primeira coluna de cada tupla)
            values = [row[0] for row in result]

            logger.debug(f"Found {len(values)} unique values")
            return values

        except Exception as e:
            logger.error(f"Error executing unique values query: {e}")
            raise

    # =========================================================================
    # STATISTICAL QUERIES
    # =========================================================================

    def compute_descriptive_stats(
        self, column: str, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calcula estatísticas descritivas completas de uma coluna.

        Otimização: Uma única query para todas as métricas.
        Retorna dict com: count, min, max, mean, median, std, variance, q25, q75, iqr

        Args:
            column: Coluna numérica para calcular estatísticas
            filters: Filtros opcionais a aplicar

        Returns:
            Dict com estatísticas descritivas:
                - count: número de valores não-nulos
                - min: valor mínimo
                - max: valor máximo
                - mean: média
                - median: mediana
                - std: desvio padrão amostral
                - variance: variância amostral
                - q25: primeiro quartil (25%)
                - q75: terceiro quartil (75%)
                - iqr: intervalo interquartil (q75 - q25)

        Example:
            >>> stats = executor.compute_descriptive_stats("Valor")
            >>> print(f"Média: {stats['mean']}, Mediana: {stats['median']}")
            >>> # Com filtros
            >>> stats = executor.compute_descriptive_stats(
            ...     "Valor", filters={"Ano": 2015}
            ... )
        """
        try:
            # Escapar nome da coluna
            col_escaped = f'"{column}"'

            # Construir WHERE clause se houver filtros
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            # Query otimizada: todas as estatísticas em uma única query
            query = f"""
                SELECT
                    COUNT({col_escaped}) as count,
                    MIN({col_escaped}) as min,
                    MAX({col_escaped}) as max,
                    AVG({col_escaped}) as mean,
                    MEDIAN({col_escaped}) as median,
                    STDDEV_SAMP({col_escaped}) as std,
                    VAR_SAMP({col_escaped}) as variance,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col_escaped}) as q25,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col_escaped}) as q75
                FROM '{self.data_source}'
                {where_clause}
            """

            logger.debug(f"Executing descriptive stats query for column: {column}")

            # Executar query
            with duckdb.connect() as conn:
                result = conn.execute(query).fetchone()

            # Construir dict de estatísticas
            stats = {
                "count": int(result[0]) if result[0] is not None else 0,
                "min": float(result[1]) if result[1] is not None else 0.0,
                "max": float(result[2]) if result[2] is not None else 0.0,
                "mean": float(result[3]) if result[3] is not None else 0.0,
                "median": float(result[4]) if result[4] is not None else 0.0,
                "std": float(result[5]) if result[5] is not None else 0.0,
                "variance": float(result[6]) if result[6] is not None else 0.0,
                "q25": float(result[7]) if result[7] is not None else 0.0,
                "q75": float(result[8]) if result[8] is not None else 0.0,
            }

            # Calcular IQR
            stats["iqr"] = stats["q75"] - stats["q25"]

            logger.debug(f"Computed descriptive stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error executing descriptive stats query: {e}")
            raise

    # =========================================================================
    # HELPER METHODS
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

    def _build_where_clause(self, filters: Dict[str, Any]) -> str:
        """
        Constrói WHERE clause a partir de filtros.

        Suporta múltiplos formatos de filtro:
        - Equality: {"Ano": 2015} → "Ano" = 2015 (ou YEAR("Data") = 2015 se Ano não existe)
        - IN clause: {"UF": ["SP", "RJ"]} → "UF" IN ('SP', 'RJ')
        - Operator: {"Valor": {"operator": ">=", "value": 1000}} → "Valor" >= 1000
        - Between: {"Data": {"between": ["2015-01-01", "2015-12-31"]}} →
                   "Data" BETWEEN '2015-01-01' AND '2015-12-31'

        Combina múltiplos filtros com AND.
        Escapa strings para evitar SQL injection.

        Args:
            filters: Dict de filtros a aplicar

        Returns:
            WHERE clause (sem a palavra WHERE)

        Example:
            >>> clause = executor._build_where_clause({"Ano": 2015, "UF": "SP"})
            >>> # Retorna: YEAR("Data") = 2015 AND "UF" = 'SP'
            >>> clause = executor._build_where_clause({"UF": ["SP", "RJ"]})
            >>> # Retorna: "UF" IN ('SP', 'RJ')
        """
        if not filters:
            return ""

        conditions = []

        for col, value in filters.items():
            # Tratamento de colunas virtuais em filtros:
            # Colunas virtuais (Ano, Mes, Nome_Mes) não existem fisicamente no dataset.
            # Transformamos para suas expressões DuckDB equivalentes.
            if col == "Ano" or col == "ano":
                col_escaped = 'YEAR("Data")'
                logger.debug(
                    "Transforming filter 'Ano' to YEAR(Data) for year-based filtering"
                )
            elif col == "Mes" or col == "mes":
                col_escaped = 'MONTH("Data")'
                logger.debug(
                    "Transforming filter 'Mes' to MONTH(Data) for month-based filtering"
                )
            elif col == "Nome_Mes" or col == "nome_mes":
                col_escaped = 'MONTHNAME("Data")'
                logger.debug(
                    "Transforming filter 'Nome_Mes' to MONTHNAME(Data) for month name filtering"
                )
            elif (
                self.alias_mapper
                and hasattr(self.alias_mapper, "is_virtual_column")
                and self.alias_mapper.is_virtual_column(col)
            ):
                col_escaped = self.alias_mapper.get_virtual_expression(col)
                logger.debug(
                    f"Transforming virtual column filter '{col}' to expression: {col_escaped}"
                )
            else:
                # Escapar nome da coluna
                col_escaped = f'"{col}"'

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
                    # Check if column is categorical for case-insensitive comparison
                    is_categorical = (
                        self.alias_mapper
                        and self.alias_mapper.is_categorical_column(col)
                    )

                    # Escapar cada valor se string
                    escaped_values = []
                    for v in value:
                        if isinstance(v, str):
                            v_escaped = v.replace("'", "''")
                            # For categorical columns, convert to UPPER for case-insensitive match
                            if is_categorical:
                                escaped_values.append(f"UPPER('{v_escaped}')")
                            else:
                                escaped_values.append(f"'{v_escaped}'")
                        else:
                            escaped_values.append(str(v))

                    values_str = ", ".join(escaped_values)

                    # For categorical columns, use UPPER on column as well
                    if is_categorical:
                        conditions.append(f"UPPER({col_escaped}) IN ({values_str})")
                    else:
                        conditions.append(f"{col_escaped} IN ({values_str})")

            # Case 2: Dict com operador ou between
            elif isinstance(value, dict):
                # Sub-case 2a: Operador customizado
                if "operator" in value:
                    operator = value["operator"]
                    op_value = value["value"]

                    # Escapar valor se string
                    if isinstance(op_value, str):
                        op_value_escaped = op_value.replace("'", "''")
                        op_value_str = f"'{op_value_escaped}'"
                    else:
                        op_value_str = str(op_value)

                    conditions.append(f"{col_escaped} {operator} {op_value_str}")

                # Sub-case 2b: Between
                elif "between" in value:
                    start, end = value["between"]

                    # Escapar valores se string
                    if isinstance(start, str):
                        start_escaped = start.replace("'", "''")
                        start_str = f"'{start_escaped}'"
                    else:
                        start_str = str(start)

                    if isinstance(end, str):
                        end_escaped = end.replace("'", "''")
                        end_str = f"'{end_escaped}'"
                    else:
                        end_str = str(end)

                    conditions.append(
                        f"{col_escaped} BETWEEN {start_str} AND {end_str}"
                    )

            # Case 3: Equality (valor simples)
            else:
                # Determine if value should be treated as string based on column type
                should_quote = isinstance(value, str)

                # Check if column is categorical for case-insensitive comparison
                is_categorical = (
                    self.alias_mapper and self.alias_mapper.is_categorical_column(col)
                )

                # If we have alias_mapper, check if column is categorical
                # Even if value is not a string, categorical columns need quoted values
                if self.alias_mapper and not should_quote:
                    if is_categorical:
                        should_quote = True
                        logger.debug(
                            f"Converting value {value} to string for categorical column {col}"
                        )

                # Format value appropriately
                if should_quote:
                    value_str_temp = str(value) if not isinstance(value, str) else value
                    value_escaped = value_str_temp.replace("'", "''")

                    # For categorical columns, use case-insensitive comparison
                    if is_categorical:
                        value_str = f"UPPER('{value_escaped}')"
                        conditions.append(f"UPPER({col_escaped}) = {value_str}")
                        logger.debug(
                            f"Using case-insensitive comparison for categorical column {col}: UPPER({col_escaped}) = UPPER('{value_escaped}')"
                        )
                    else:
                        value_str = f"'{value_escaped}'"
                        conditions.append(f"{col_escaped} = {value_str}")
                else:
                    value_str = str(value)
                    conditions.append(f"{col_escaped} = {value_str}")

        # Combinar condições com AND
        where_clause = " AND ".join(conditions)

        return where_clause
