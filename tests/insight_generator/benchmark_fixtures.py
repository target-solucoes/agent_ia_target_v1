"""
FASE 4.1 - Benchmark Fixtures for Insight Generator Testing.

Provides structured test scenarios (BenchmarkCase) covering multiple
intents, chart types, complexities, and filter conditions.

Each case includes:
    - user_query, chart_type, enriched_intent, filters
    - chart_spec / analytics_result stubs matching production contracts
    - data_records representing realistic aggregated data
    - expected_entities: names/values expected in a good response
    - model_tier: "lite" if achievable by flash-lite, else "full"

These fixtures are consumed by:
    - test_insight_benchmark.py  (4.1)
    - test_insight_model_comparison.py  (4.2)
    - test_insight_regression.py  (4.3)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class BenchmarkCase:
    """A single benchmark test case for insight quality evaluation."""

    id: str
    description: str
    user_query: str
    chart_type: str
    data_records: List[Dict[str, Any]]
    enriched_intent: Dict[str, Any]
    filters: Dict[str, Any] = field(default_factory=dict)
    expected_entities: List[str] = field(default_factory=list)
    key_columns: Optional[List[str]] = None
    model_tier: str = "full"  # "lite" or "full"
    complexity: str = "medium"  # "simple", "medium", "complex"

    # Pre-built stubs for the pipeline
    def build_chart_spec(self) -> Dict[str, Any]:
        """Build a chart_spec dict matching production contract."""
        dims = []
        metrics = []

        # Infer dimensions and metrics from data
        if self.data_records:
            first_row = self.data_records[0]
            for col, val in first_row.items():
                if isinstance(val, (int, float)):
                    metrics.append({"name": col, "column": col, "aggregation": "sum"})
                else:
                    dims.append({"name": col, "column": col, "alias": col})

        return {
            "chart_type": self.chart_type,
            "user_query": self.user_query,
            "dimensions": dims,
            "metrics": metrics,
            "filters": self.filters,
            "intent": self.enriched_intent.get("base_intent", "ranking"),
        }

    def build_analytics_result(self) -> Dict[str, Any]:
        """Build an analytics_result dict matching production contract."""
        return {
            "data": self.data_records,
            "metadata": {
                "user_query": self.user_query,
                "rows": len(self.data_records),
                "columns": len(self.data_records[0]) if self.data_records else 0,
            },
        }


# ============================================================================
# Benchmark Cases (derived from insight_generator_samples.md)
# ============================================================================


def get_benchmark_cases() -> List[BenchmarkCase]:
    """Return all benchmark cases for the test suite."""
    return [
        _case_01_ranking_simple(),
        _case_02_ranking_with_filters(),
        _case_03_distribution(),
        _case_04_temporal_trend(),
        _case_05_comparison_negative(),
        _case_06_composition_crosstab(),
        _case_07_category_comparison(),
        _case_08_variation_negative(),
        _case_09_single_metric(),
        _case_10_temporal_seasonality(),
        _case_11_contextual_filter(),
        _case_12_temporal_filtered_range(),
        _case_13_complex_multi_series(),
    ]


def get_simple_cases() -> List[BenchmarkCase]:
    """Return only simple-complexity cases (suitable for flash-lite)."""
    return [c for c in get_benchmark_cases() if c.complexity == "simple"]


def get_complex_cases() -> List[BenchmarkCase]:
    """Return only complex cases (require flash)."""
    return [c for c in get_benchmark_cases() if c.complexity == "complex"]


# ============================================================================
# Individual Case Constructors
# ============================================================================


def _case_01_ranking_simple() -> BenchmarkCase:
    return BenchmarkCase(
        id="S01",
        description="Ranking simples - top 5 clientes",
        user_query="Top 5 clientes por faturamento",
        chart_type="bar_horizontal",
        data_records=[
            {"Cod_Cliente": "2855", "Valor_Vendido": 48215340},
            {"Cod_Cliente": "22494", "Valor_Vendido": 31892110},
            {"Cod_Cliente": "23709", "Valor_Vendido": 27456890},
            {"Cod_Cliente": "40461", "Valor_Vendido": 22118750},
            {"Cod_Cliente": "41410", "Valor_Vendido": 19887320},
        ],
        enriched_intent={
            "base_intent": "ranking",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "none",
            "narrative_angle": "ranking de desempenho por volume",
            "suggested_metrics": ["total", "top_n", "gap"],
            "key_entities": ["cliente", "faturamento"],
            "filters_context": {"has_filters": False},
        },
        expected_entities=["2855", "22494", "23709", "40461", "41410"],
        key_columns=["Cod_Cliente", "Valor_Vendido"],
        model_tier="lite",
        complexity="simple",
    )


def _case_02_ranking_with_filters() -> BenchmarkCase:
    return BenchmarkCase(
        id="S02",
        description="Ranking com filtro UF",
        user_query="quais foram os maiores representantes de SC?",
        chart_type="bar_horizontal",
        data_records=[
            {"Cod_Vendedor": "000145", "Valor_Vendido": 21220758},
            {"Cod_Vendedor": "000018", "Valor_Vendido": 20586795},
            {"Cod_Vendedor": "000146", "Valor_Vendido": 15967630},
            {"Cod_Vendedor": "000114", "Valor_Vendido": 10586136},
            {"Cod_Vendedor": "000176", "Valor_Vendido": 9832415},
        ],
        enriched_intent={
            "base_intent": "ranking",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "none",
            "narrative_angle": "ranking de representantes por vendas",
            "suggested_metrics": ["total", "top_n", "gap"],
            "key_entities": ["representante", "SC"],
            "filters_context": {"has_filters": True, "temporal_filter": False},
        },
        filters={"UF_Cliente": "SC"},
        expected_entities=["000145", "000018", "000146"],
        key_columns=["Cod_Vendedor", "Valor_Vendido"],
        model_tier="lite",
        complexity="simple",
    )


def _case_03_distribution() -> BenchmarkCase:
    return BenchmarkCase(
        id="S03",
        description="Distribuição de produtos (pie)",
        user_query="qual a distribuição dos produtos do representante 000145?",
        chart_type="pie",
        data_records=[
            {
                "Des_Linha_Produto": "CONEXOES SOLDAVEIS",
                "Qtd_Vendida": 3473773,
                "Participacao_pct": 31.5,
            },
            {
                "Des_Linha_Produto": "TUBOS ELETRODUTO CORRUGADO",
                "Qtd_Vendida": 1399350,
                "Participacao_pct": 12.7,
            },
            {
                "Des_Linha_Produto": "CONEXOES ESGOTO PRIMARIO",
                "Qtd_Vendida": 1310456,
                "Participacao_pct": 11.9,
            },
            {
                "Des_Linha_Produto": "TUBOS PVC SOLDAVEL",
                "Qtd_Vendida": 1089230,
                "Participacao_pct": 9.9,
            },
            {
                "Des_Linha_Produto": "CONEXOES ROSCAVEIS",
                "Qtd_Vendida": 987654,
                "Participacao_pct": 8.9,
            },
        ],
        enriched_intent={
            "base_intent": "distribution",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "none",
            "narrative_angle": "análise de distribuição e participação de produtos",
            "suggested_metrics": ["total", "participacao", "concentracao"],
            "key_entities": ["produto", "representante", "000145"],
            "filters_context": {"has_filters": True},
        },
        filters={"UF_Cliente": "SC", "Cod_Vendedor": "000145"},
        expected_entities=["CONEXOES SOLDAVEIS", "TUBOS ELETRODUTO", "000145"],
        key_columns=["Des_Linha_Produto", "Qtd_Vendida"],
        model_tier="lite",
        complexity="simple",
    )


def _case_04_temporal_trend() -> BenchmarkCase:
    return BenchmarkCase(
        id="S04",
        description="Tendência temporal série única",
        user_query="qual o histórico de compras de 2015?",
        chart_type="line",
        data_records=[
            {"Mes": "Jan", "Valor_Vendido": 633249},
            {"Mes": "Fev", "Valor_Vendido": 1053361},
            {"Mes": "Mar", "Valor_Vendido": 2438682},
            {"Mes": "Abr", "Valor_Vendido": 1238411},
            {"Mes": "Mai", "Valor_Vendido": 974582},
            {"Mes": "Jun", "Valor_Vendido": 606921},
            {"Mes": "Jul", "Valor_Vendido": 845591},
            {"Mes": "Ago", "Valor_Vendido": 1076049},
            {"Mes": "Set", "Valor_Vendido": 1328773},
            {"Mes": "Out", "Valor_Vendido": 2388519},
            {"Mes": "Nov", "Valor_Vendido": 639065},
            {"Mes": "Dez", "Valor_Vendido": 430074},
        ],
        enriched_intent={
            "base_intent": "trend",
            "polarity": "neutral",
            "temporal_focus": "time_series",
            "comparison_type": "none",
            "narrative_angle": "análise de evolução temporal e padrões sazonais",
            "suggested_metrics": ["variacao", "pico", "vale", "tendencia"],
            "key_entities": ["2015", "compras"],
            "filters_context": {"has_filters": True, "temporal_filter": True},
        },
        filters={"UF_Cliente": "SC", "Cod_Vendedor": "000145", "Ano": "2015"},
        expected_entities=["Mar", "Out", "Dez"],
        key_columns=["Mes", "Valor_Vendido"],
        model_tier="full",
        complexity="medium",
    )


def _case_05_comparison_negative() -> BenchmarkCase:
    return BenchmarkCase(
        id="S05",
        description="Comparação temporal - queda entre meses",
        user_query="Quais clientes que tiveram a maior queda entre fev/2015 e março/2015?",
        chart_type="line_composed",
        data_records=[
            {
                "Cod_Cliente": "20524",
                "Fev_2015": 5401,
                "Mar_2015": 0,
                "Variacao_pct": -100.0,
            },
            {
                "Cod_Cliente": "33777",
                "Fev_2015": 1695,
                "Mar_2015": 0,
                "Variacao_pct": -100.0,
            },
            {
                "Cod_Cliente": "17761",
                "Fev_2015": 4129,
                "Mar_2015": 430,
                "Variacao_pct": -89.6,
            },
            {
                "Cod_Cliente": "14061",
                "Fev_2015": 7115,
                "Mar_2015": 3566,
                "Variacao_pct": -49.9,
            },
            {
                "Cod_Cliente": "2330",
                "Fev_2015": 8085,
                "Mar_2015": 4098,
                "Variacao_pct": -49.3,
            },
        ],
        enriched_intent={
            "base_intent": "variation",
            "polarity": "negative",
            "temporal_focus": "period_over_period",
            "comparison_type": "period_vs_period",
            "narrative_angle": "análise de variação e mudança, com foco em quedas e riscos, entre períodos específicos",
            "suggested_metrics": ["delta", "growth_rate", "loss_magnitude"],
            "key_entities": ["cliente", "queda", "fevereiro", "março", "2015"],
            "filters_context": {"has_filters": True, "temporal_filter": True},
        },
        filters={"UF_Cliente": "SC", "Cod_Vendedor": "000145"},
        expected_entities=["20524", "33777", "17761", "100%"],
        key_columns=["Cod_Cliente", "Variacao_pct"],
        model_tier="full",
        complexity="complex",
    )


def _case_06_composition_crosstab() -> BenchmarkCase:
    return BenchmarkCase(
        id="S06",
        description="Composição cross-tab (stacked)",
        user_query="quais os 5 produtos mais vendidos nos 5 maiores clientes desse mesmo representante?",
        chart_type="bar_vertical_stacked",
        data_records=[
            {
                "Cliente": "2855",
                "Produto": "Conexoes Soldaveis",
                "Qtd_Vendida": 2190981,
            },
            {"Cliente": "2855", "Produto": "Tubos Eletroduto", "Qtd_Vendida": 273725},
            {
                "Cliente": "2855",
                "Produto": "Conex. Esgoto Prim.",
                "Qtd_Vendida": 808452,
            },
            {"Cliente": "23709", "Produto": "Tubos Eletroduto", "Qtd_Vendida": 859350},
            {"Cliente": "23709", "Produto": "Conexoes Soldaveis", "Qtd_Vendida": 64223},
            {
                "Cliente": "22494",
                "Produto": "Conexoes Soldaveis",
                "Qtd_Vendida": 375339,
            },
            {
                "Cliente": "40461",
                "Produto": "Conexoes Soldaveis",
                "Qtd_Vendida": 164984,
            },
            {"Cliente": "41410", "Produto": "Conexoes Soldaveis", "Qtd_Vendida": 79423},
        ],
        enriched_intent={
            "base_intent": "composition",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "category_vs_category",
            "narrative_angle": "análise de composição e mix de produtos por cliente",
            "suggested_metrics": ["total", "participacao", "dominancia"],
            "key_entities": ["produto", "cliente", "representante"],
            "filters_context": {"has_filters": True},
        },
        filters={"UF_Cliente": "SC", "Cod_Vendedor": "000145"},
        expected_entities=["2855", "23709", "Conexoes Soldaveis", "Tubos Eletroduto"],
        key_columns=["Cliente", "Produto", "Qtd_Vendida"],
        model_tier="full",
        complexity="complex",
    )


def _case_07_category_comparison() -> BenchmarkCase:
    return BenchmarkCase(
        id="S07",
        description="Comparação entre categorias (vendas por estado)",
        user_query="vendas por estado",
        chart_type="bar_vertical",
        data_records=[
            {"UF_Cliente": "SP", "Valor_Vendido": 89234560},
            {"UF_Cliente": "SC", "Valor_Vendido": 67891230},
            {"UF_Cliente": "PR", "Valor_Vendido": 54321780},
            {"UF_Cliente": "MG", "Valor_Vendido": 43210890},
            {"UF_Cliente": "RS", "Valor_Vendido": 38765430},
        ],
        enriched_intent={
            "base_intent": "ranking",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "category_vs_category",
            "narrative_angle": "comparação de desempenho por estado",
            "suggested_metrics": ["total", "participacao"],
            "key_entities": ["estado", "vendas"],
            "filters_context": {"has_filters": False},
        },
        expected_entities=["SP", "SC", "PR"],
        key_columns=["UF_Cliente", "Valor_Vendido"],
        model_tier="lite",
        complexity="simple",
    )


def _case_08_variation_negative() -> BenchmarkCase:
    return BenchmarkCase(
        id="S08",
        description="Variação negativa período a período",
        user_query="quais produtos tiveram queda em 2016 comparado a 2015?",
        chart_type="bar_horizontal",
        data_records=[
            {
                "Des_Linha_Produto": "Tubos PVC Soldavel",
                "Ano_2015": 12300000,
                "Ano_2016": 8100000,
                "Variacao_pct": -34.1,
            },
            {
                "Des_Linha_Produto": "Conexoes Roscaveis",
                "Ano_2015": 9800000,
                "Ano_2016": 7200000,
                "Variacao_pct": -26.5,
            },
            {
                "Des_Linha_Produto": "Caixas de Luz",
                "Ano_2015": 5400000,
                "Ano_2016": 4500000,
                "Variacao_pct": -16.7,
            },
        ],
        enriched_intent={
            "base_intent": "variation",
            "polarity": "negative",
            "temporal_focus": "period_over_period",
            "comparison_type": "period_vs_period",
            "narrative_angle": "análise de queda e variação negativa entre períodos anuais",
            "suggested_metrics": ["delta", "variacao_pct", "loss_magnitude"],
            "key_entities": ["produto", "queda", "2015", "2016"],
            "filters_context": {"has_filters": False, "temporal_filter": True},
        },
        expected_entities=["Tubos PVC", "Conexoes Roscaveis", "34,1%", "-34"],
        key_columns=["Des_Linha_Produto", "Variacao_pct"],
        model_tier="full",
        complexity="medium",
    )


def _case_09_single_metric() -> BenchmarkCase:
    return BenchmarkCase(
        id="S09",
        description="Métrica única - total de vendas",
        user_query="qual o total de vendas?",
        chart_type="bar_horizontal",
        data_records=[
            {"Metrica": "Total Valor_Vendido", "Valor": 109711696},
        ],
        enriched_intent={
            "base_intent": "ranking",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "none",
            "narrative_angle": "valor agregado",
            "suggested_metrics": ["total"],
            "key_entities": ["vendas", "total"],
            "filters_context": {"has_filters": True},
        },
        filters={"UF_Cliente": "SC"},
        expected_entities=["109", "SC"],
        model_tier="lite",
        complexity="simple",
    )


def _case_10_temporal_seasonality() -> BenchmarkCase:
    return BenchmarkCase(
        id="S10",
        description="Tendência multi-ano com sazonalidade",
        user_query="como foram as vendas mês a mês nos últimos 3 anos?",
        chart_type="line_composed",
        data_records=[
            {
                "Ano": "2014",
                "Melhor_Mes": "Mar",
                "Valor_Melhor": 28500000,
                "Pior_Mes": "Dez",
                "Valor_Pior": 11200000,
                "Media_Mensal": 19800000,
                "Total_Ano": 237600000,
            },
            {
                "Ano": "2015",
                "Melhor_Mes": "Mar",
                "Valor_Melhor": 26100000,
                "Pior_Mes": "Dez",
                "Valor_Pior": 9800000,
                "Media_Mensal": 18200000,
                "Total_Ano": 218400000,
            },
            {
                "Ano": "2016",
                "Melhor_Mes": "Out",
                "Valor_Melhor": 24300000,
                "Pior_Mes": "Jun",
                "Valor_Pior": 10100000,
                "Media_Mensal": 17500000,
                "Total_Ano": 210000000,
            },
        ],
        enriched_intent={
            "base_intent": "temporal",
            "polarity": "neutral",
            "temporal_focus": "time_series",
            "comparison_type": "period_vs_period",
            "narrative_angle": "análise de tendência anual com padrões sazonais multi-ano",
            "suggested_metrics": ["variacao", "tendencia", "sazonalidade"],
            "key_entities": ["2014", "2015", "2016", "mês"],
            "filters_context": {"has_filters": False},
        },
        expected_entities=["2014", "2015", "2016"],
        model_tier="full",
        complexity="complex",
    )


def _case_11_contextual_filter() -> BenchmarkCase:
    return BenchmarkCase(
        id="S11",
        description="Filtro persistente com referencia contextual",
        user_query="e para o representante 000018?",
        chart_type="bar_horizontal",
        data_records=[
            {"Des_Linha_Produto": "CONEXOES SOLDAVEIS", "Qtd_Vendida": 2891456},
            {"Des_Linha_Produto": "TUBOS PVC SOLDAVEL", "Qtd_Vendida": 1234567},
            {"Des_Linha_Produto": "CONEXOES ESGOTO PRIMARIO", "Qtd_Vendida": 987654},
        ],
        enriched_intent={
            "base_intent": "ranking",
            "polarity": "neutral",
            "temporal_focus": "single_period",
            "comparison_type": "none",
            "narrative_angle": "ranking de produtos por volume",
            "suggested_metrics": ["total", "top_n"],
            "key_entities": ["produto", "representante", "000018"],
            "filters_context": {"has_filters": True},
        },
        filters={"UF_Cliente": "SC"},
        expected_entities=["000018", "CONEXOES SOLDAVEIS"],
        key_columns=["Des_Linha_Produto", "Qtd_Vendida"],
        model_tier="lite",
        complexity="simple",
    )


def _case_12_temporal_filtered_range() -> BenchmarkCase:
    return BenchmarkCase(
        id="S12",
        description="Tendência com filtro temporal explícito (jan-jun)",
        user_query="como foram as vendas do representante 000145 entre janeiro e junho de 2015?",
        chart_type="line",
        data_records=[
            {"Mes": "Jan", "Valor_Vendido": 633249},
            {"Mes": "Fev", "Valor_Vendido": 1053361},
            {"Mes": "Mar", "Valor_Vendido": 2438682},
            {"Mes": "Abr", "Valor_Vendido": 1238411},
            {"Mes": "Mai", "Valor_Vendido": 974582},
            {"Mes": "Jun", "Valor_Vendido": 606921},
        ],
        enriched_intent={
            "base_intent": "trend",
            "polarity": "neutral",
            "temporal_focus": "time_series",
            "comparison_type": "none",
            "narrative_angle": "análise de evolução temporal em período delimitado",
            "suggested_metrics": ["variacao", "pico", "vale"],
            "key_entities": ["representante", "000145", "janeiro", "junho", "2015"],
            "filters_context": {"has_filters": True, "temporal_filter": True},
        },
        filters={"UF_Cliente": "SC", "Cod_Vendedor": "000145"},
        expected_entities=["000145", "Mar"],
        key_columns=["Mes", "Valor_Vendido"],
        model_tier="full",
        complexity="medium",
    )


def _case_13_complex_multi_series() -> BenchmarkCase:
    return BenchmarkCase(
        id="S13",
        description="Evolução multi-produto multi-ano (line_composed)",
        user_query="como os 3 maiores produtos evoluíram de 2014 para 2016?",
        chart_type="line_composed",
        data_records=[
            {
                "Produto": "Conexoes Soldaveis",
                "Ano_2014": 45200000,
                "Ano_2015": 42800000,
                "Ano_2016": 40100000,
                "Variacao_pct": -11.3,
            },
            {
                "Produto": "Tubos Eletroduto",
                "Ano_2014": 28700000,
                "Ano_2015": 30100000,
                "Ano_2016": 31500000,
                "Variacao_pct": 9.8,
            },
            {
                "Produto": "Conex. Esgoto Prim.",
                "Ano_2014": 22100000,
                "Ano_2015": 21500000,
                "Ano_2016": 19800000,
                "Variacao_pct": -10.4,
            },
        ],
        enriched_intent={
            "base_intent": "temporal",
            "polarity": "neutral",
            "temporal_focus": "time_series",
            "comparison_type": "period_vs_period",
            "narrative_angle": "análise de evolução temporal multi-série com divergência de tendência",
            "suggested_metrics": ["variacao", "tendencia", "divergencia"],
            "key_entities": ["produto", "2014", "2015", "2016"],
            "filters_context": {"has_filters": False},
        },
        expected_entities=["Conexoes Soldaveis", "Tubos Eletroduto", "2014", "2016"],
        key_columns=["Produto", "Variacao_pct"],
        model_tier="full",
        complexity="complex",
    )
