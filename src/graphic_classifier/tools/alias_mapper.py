"""
Dynamic alias mapper for column resolution.

This module provides the AliasMapper class that resolves user query terms
to actual column names using the alias.yaml configuration.

Key features:
- Exact matching
- Fuzzy matching with configurable threshold
- Reverse index for performance
- Caching for repeated queries
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process

from src.graphic_classifier.utils.text_cleaner import (
    normalize_text,
    fuzzy_normalize,
    similarity_key,
)
from src.graphic_classifier.core.settings import ALIAS_PATH, FUZZY_MATCH_THRESHOLD


class AliasMapper:
    """
    Handles dynamic column alias resolution.

    This class loads the alias configuration from YAML and provides
    methods to resolve user query terms to actual column names using
    various matching strategies.
    """

    # =========================================================================
    # VIRTUAL COLUMN REGISTRY
    # =========================================================================
    # Colunas que não existem fisicamente no dataset, mas são derivadas de
    # colunas reais via expressões DuckDB. O AliasMapper resolve aliases como
    # "ano" -> "Ano", porém "Ano" não é uma coluna física — requer YEAR("Data").
    # Este registro permite que componentes downstream (QueryExecutor, etc.)
    # detectem colunas virtuais e obtenham a expressão SQL correspondente.
    # =========================================================================
    VIRTUAL_COLUMN_MAP: Dict[str, str] = {
        "Ano": 'YEAR("Data")',
        "Mes": 'MONTH("Data")',
        "Nome_Mes": 'MONTHNAME("Data")',
    }

    def __init__(
        self,
        alias_path: str = ALIAS_PATH,
        fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
        dataset_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the AliasMapper.

        Args:
            alias_path: Path to the alias.yaml configuration file
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0.0 to 1.0)
            dataset_columns: Optional list of exact column names from dataset
                            (enables case-insensitive matching)
        """
        self.logger = logging.getLogger(__name__)
        self.alias_path = Path(alias_path)
        self.fuzzy_threshold = fuzzy_threshold

        # Cache for successful resolutions
        self._cache: Dict[str, Optional[str]] = {}

        # Load aliases from YAML
        self.aliases = self._load_aliases()

        # NOVO: Build dataset column map for case-insensitive matching
        self.dataset_column_map = self._build_dataset_column_map(dataset_columns)

        # Build reverse index for fast lookup
        self.reverse_index = self._build_reverse_index()

        # Build fuzzy search index
        self.fuzzy_index = self._build_fuzzy_index()

        # Load column types for type-aware operations
        self.column_types = self._load_column_types()

        case_mode = "case-insensitive" if dataset_columns else "case-preserving"
        self.logger.info(
            f"AliasMapper initialized with {len(self.aliases.get('columns', {}))} columns, "
            f"{len(self.aliases.get('metrics', {}))} metrics, "
            f"{len(self.column_types.get('categorical', []))} categorical, "
            f"{len(self.column_types.get('numeric', []))} numeric, "
            f"fuzzy_threshold={fuzzy_threshold}, mode={case_mode}"
        )

    def _load_aliases(self) -> Dict:
        """
        Load alias configuration from YAML file.

        Returns:
            Dictionary containing alias configuration

        Raises:
            FileNotFoundError: If alias file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not self.alias_path.exists():
            raise FileNotFoundError(f"Alias file not found: {self.alias_path}")

        try:
            with open(self.alias_path, "r", encoding="utf-8") as f:
                aliases = yaml.safe_load(f)

            if not isinstance(aliases, dict):
                raise ValueError("Alias file must contain a dictionary")

            return aliases

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing alias YAML: {e}")
            raise

    def _build_dataset_column_map(
        self, dataset_columns: Optional[List[str]]
    ) -> Dict[str, str]:
        """
        Constroi mapeamento de nomes normalizados para nomes exatos do dataset.

        Permite case-insensitive matching retornando sempre o nome EXATO
        do dataset para uso em SQL queries.

        Args:
            dataset_columns: Lista de nomes EXATOS das colunas do dataset

        Returns:
            Dict mapeando normalized_name -> exact_name

        Example:
            >>> dataset_columns = ["VALOR_VENDIDO", "MUNICIPIO_CLIENTE"]
            >>> mapper._build_dataset_column_map(dataset_columns)
            {
                "valorvendido": "VALOR_VENDIDO",
                "municipiocliente": "MUNICIPIO_CLIENTE"
            }
        """
        if not dataset_columns:
            return {}

        column_map = {}
        for exact_name in dataset_columns:
            normalized = normalize_text(exact_name)
            if normalized:
                # Se houver duplicadas apos normalizacao, primeira ocorrencia prevalece
                if normalized in column_map:
                    self.logger.warning(
                        f"Duplicate normalized column name '{normalized}': "
                        f"'{column_map[normalized]}' vs '{exact_name}'. "
                        f"Using first occurrence."
                    )
                else:
                    column_map[normalized] = exact_name

        self.logger.debug(
            f"Built dataset column map with {len(column_map)} exact names"
        )
        return column_map

    def _build_reverse_index(self) -> Dict[str, str]:
        """
        Build reverse lookup index for exact matching.

        Creates a mapping from normalized alias -> column name.
        MODIFICADO: Agora retorna nomes EXATOS do dataset quando disponivel.

        Returns:
            Dictionary mapping normalized aliases to column names
        """
        reverse_index = {}

        # Index columns
        for column, aliases in self.aliases.get("columns", {}).items():
            # NOVO: Resolver nome exato do dataset
            normalized_col = normalize_text(column)
            exact_name = self.dataset_column_map.get(normalized_col, column)

            # Add the column name itself
            reverse_index[normalized_col] = exact_name

            # Add all aliases
            for alias in aliases:
                normalized = normalize_text(alias)
                if normalized and normalized not in reverse_index:
                    reverse_index[normalized] = exact_name  # Nome EXATO!

        # Index metrics
        for metric, aliases in self.aliases.get("metrics", {}).items():
            # NOVO: Resolver nome exato do dataset
            normalized_metric = normalize_text(metric)
            exact_name = self.dataset_column_map.get(normalized_metric, metric)

            reverse_index[normalized_metric] = exact_name

            for alias in aliases:
                normalized = normalize_text(alias)
                if normalized and normalized not in reverse_index:
                    reverse_index[normalized] = exact_name  # Nome EXATO!

        self.logger.debug(f"Built reverse index with {len(reverse_index)} entries")
        return reverse_index

    def _load_column_types(self) -> Dict[str, List[str]]:
        """
        Load column type classifications from alias configuration.

        Returns:
            Dictionary with 'numeric', 'categorical', and 'temporal' lists
        """
        column_types = self.aliases.get("column_types", {})

        return {
            "numeric": column_types.get("numeric", []),
            "categorical": column_types.get("categorical", []),
            "temporal": column_types.get("temporal", []),
        }

    def is_categorical_column(self, column_name: str) -> bool:
        """
        Check if a column is categorical/textual.

        Args:
            column_name: Name of the column to check

        Returns:
            True if column is categorical, False otherwise
        """
        return column_name in self.column_types.get("categorical", [])

    def is_numeric_column(self, column_name: str) -> bool:
        """
        Check if a column is numeric.

        Args:
            column_name: Name of the column to check

        Returns:
            True if column is numeric, False otherwise
        """
        return column_name in self.column_types.get("numeric", [])

    def get_column_type(self, column_name: str) -> Optional[str]:
        """
        Get the type classification of a column.

        Args:
            column_name: Name of the column

        Returns:
            'numeric', 'categorical', 'temporal', or None if not classified
        """
        if column_name in self.column_types.get("numeric", []):
            return "numeric"
        elif column_name in self.column_types.get("categorical", []):
            return "categorical"
        elif column_name in self.column_types.get("temporal", []):
            return "temporal"
        return None

    def _build_fuzzy_index(self) -> Dict[str, str]:
        """
        Build index for fuzzy matching.

        Creates a mapping from similarity keys to column names.
        MODIFICADO: Usa nomes exatos do dataset quando disponivel.

        Returns:
            Dictionary mapping similarity keys to column names
        """
        fuzzy_index = {}

        # Index columns
        for column, aliases in self.aliases.get("columns", {}).items():
            # NOVO: Resolver nome exato do dataset
            normalized_col = normalize_text(column)
            exact_name = self.dataset_column_map.get(normalized_col, column)

            # Add column name
            key = similarity_key(column)
            fuzzy_index[key] = exact_name

            # Add all aliases
            for alias in aliases:
                key = similarity_key(alias)
                if key and key not in fuzzy_index:
                    fuzzy_index[key] = exact_name  # Nome EXATO!

        # Index metrics
        for metric, aliases in self.aliases.get("metrics", {}).items():
            # NOVO: Resolver nome exato do dataset
            normalized_metric = normalize_text(metric)
            exact_name = self.dataset_column_map.get(normalized_metric, metric)

            key = similarity_key(metric)
            fuzzy_index[key] = exact_name

            for alias in aliases:
                key = similarity_key(alias)
                if key and key not in fuzzy_index:
                    fuzzy_index[key] = exact_name  # Nome EXATO!

        return fuzzy_index

    def resolve(self, query_term: str) -> Optional[str]:
        """
        Resolve a query term to an actual column name.

        Tries multiple strategies in order:
        1. Check cache
        2. Exact match (after normalization)
        3. Fuzzy match (if above threshold)

        Args:
            query_term: Term extracted from user query

        Returns:
            Actual column name if found, None otherwise

        Examples:
            >>> mapper = AliasMapper()
            >>> mapper.resolve("faturamento")
            'Valor_Vendido'
            >>> mapper.resolve("UF")
            'UF_Cliente'
        """
        if not query_term:
            return None

        # Check cache first
        if query_term in self._cache:
            return self._cache[query_term]

        # Try exact match
        result = self._exact_match(query_term)
        if result:
            self._cache[query_term] = result
            return result

        # Try fuzzy match
        result = self._fuzzy_match(query_term)
        if result:
            self._cache[query_term] = result
            return result

        # No match found
        self.logger.debug(f"No mapping found for term: '{query_term}'")
        self._cache[query_term] = None
        return None

    def _exact_match(self, query_term: str) -> Optional[str]:
        """
        Attempt exact match using reverse index.

        Args:
            query_term: Query term to match

        Returns:
            Column name if exact match found, None otherwise
        """
        normalized = normalize_text(query_term)
        result = self.reverse_index.get(normalized)

        if result:
            self.logger.debug(f"Exact match: '{query_term}' -> '{result}'")

        return result

    def _fuzzy_match(self, query_term: str) -> Optional[str]:
        """
        Attempt fuzzy match using string similarity.

        Uses RapidFuzz for efficient fuzzy string matching.

        Args:
            query_term: Query term to match

        Returns:
            Column name if fuzzy match above threshold, None otherwise
        """
        normalized = normalize_text(query_term)

        # Search in reverse index keys
        search_keys = list(self.reverse_index.keys())

        if not search_keys:
            return None

        # Use RapidFuzz to find best match
        # Using token_sort_ratio for better handling of word order differences
        result = process.extractOne(
            normalized,
            search_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.fuzzy_threshold * 100,  # RapidFuzz uses 0-100 scale
        )

        if result:
            matched_key, score, _ = result
            column = self.reverse_index[matched_key]

            self.logger.debug(
                f"Fuzzy match: '{query_term}' -> '{column}' "
                f"(score: {score / 100:.2f}, key: '{matched_key}')"
            )

            return column

        return None

    def resolve_batch(self, terms: List[str]) -> Dict[str, Optional[str]]:
        """
        Resolve multiple terms at once.

        Args:
            terms: List of query terms to resolve

        Returns:
            Dictionary mapping query terms to column names (or None if not found)

        Examples:
            >>> mapper = AliasMapper()
            >>> mapper.resolve_batch(["vendas", "estado", "produto"])
            {'vendas': 'Valor_Vendido', 'estado': 'UF_Cliente', 'produto': 'Des_Linha_Produto'}
        """
        return {term: self.resolve(term) for term in terms}

    def get_all_columns(self) -> List[str]:
        """
        Get list of all available column names.

        Returns:
            List of column names from alias configuration
        """
        return list(self.aliases.get("columns", {}).keys())

    def get_all_metrics(self) -> List[str]:
        """
        Get list of all available metric names.

        Returns:
            List of metric names from alias configuration
        """
        return list(self.aliases.get("metrics", {}).keys())

    def get_column_aliases(self, column: str) -> List[str]:
        """
        Get all aliases for a specific column.

        Args:
            column: Column name

        Returns:
            List of aliases for the column
        """
        return self.aliases.get("columns", {}).get(column, [])

    def get_column_category(self, column: str) -> Optional[str]:
        """
        Get the category of a column (temporal, produto, vendas, etc.).

        Args:
            column: Column name

        Returns:
            Category name if found, None otherwise
        """
        categories = self.aliases.get("categories", {})

        for category, columns in categories.items():
            if column in columns:
                return category

        return None

    def is_temporal_column(self, column: str) -> bool:
        """
        Check if a column is temporal (date/time related).

        Args:
            column: Column name

        Returns:
            True if column is temporal, False otherwise
        """
        temporal_columns = self.aliases.get("categories", {}).get("temporal", [])
        return column in temporal_columns

    def is_metric_column(self, column: str) -> bool:
        """
        Check if a column is a metric (numeric measure).

        Args:
            column: Column name

        Returns:
            True if column is a metric, False otherwise
        """
        # Check if in vendas category or metrics section
        vendas_columns = self.aliases.get("categories", {}).get("vendas", [])
        metrics = self.aliases.get("metrics", {})

        return column in vendas_columns or column in metrics

    # =========================================================================
    # VIRTUAL COLUMN METHODS
    # =========================================================================

    def is_virtual_column(self, column_name: str) -> bool:
        """
        Verifica se uma coluna é virtual (derivada de outra coluna via expressão SQL).

        Colunas virtuais como "Ano" e "Mes" são mapeadas pelo alias.yaml mas não
        existem fisicamente no dataset. Elas requerem expressões DuckDB como
        YEAR("Data") ou MONTH("Data") para serem utilizadas em queries.

        Args:
            column_name: Nome da coluna a verificar

        Returns:
            True se a coluna é virtual, False caso contrário

        Example:
            >>> mapper.is_virtual_column("Ano")
            True
            >>> mapper.is_virtual_column("Valor_Vendido")
            False
        """
        return column_name in self.VIRTUAL_COLUMN_MAP

    def get_virtual_expression(self, column_name: str) -> Optional[str]:
        """
        Obtém a expressão DuckDB equivalente para uma coluna virtual.

        Args:
            column_name: Nome da coluna virtual

        Returns:
            Expressão SQL DuckDB correspondente, ou None se não for virtual

        Example:
            >>> mapper.get_virtual_expression("Ano")
            'YEAR("Data")'
            >>> mapper.get_virtual_expression("Mes")
            'MONTH("Data")'
            >>> mapper.get_virtual_expression("Valor_Vendido")
            None
        """
        return self.VIRTUAL_COLUMN_MAP.get(column_name)

    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()
        self.logger.debug("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and hit statistics
        """
        return {
            "cache_size": len(self._cache),
            "successful_resolutions": sum(
                1 for v in self._cache.values() if v is not None
            ),
            "failed_resolutions": sum(1 for v in self._cache.values() if v is None),
        }
