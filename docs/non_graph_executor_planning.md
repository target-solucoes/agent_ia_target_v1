# Plano de Refatoracao: Agente non_graph_executor

## 1. Visao Geral

### Objetivo
Refatorar o agente `non_graph_executor` para substituir a arquitetura atual (baseada em classificacao por keywords e SQL deterministico) por um mecanismo orientado por raciocinio de LLM, capaz de compreender a intencao real do usuario e gerar queries DuckDB dinamicas que respondam corretamente a qualquer pergunta analitica sobre os dados comerciais.

### Principios de Design
- **LLM como motor de raciocinio**: O LLM deve interpretar a intencao e definir a estrategia de consulta
- **Tools como interface de execucao**: O LLM aciona ferramentas especializadas via function calling
- **alias.yaml como base semantica**: Toda resolucao de colunas e tipos usa o mapeamento existente
- **DuckDB como unico motor SQL**: Nenhuma query fora do DuckDB
- **Compatibilidade de output**: Manter schema `NonGraphOutput` existente
- **Modularidade preservada**: Novos componentes se integram sem quebrar o pipeline

### Restricoes Tecnicas
- Modelos LLM: apenas `gemini-2.5-flash` e `gemini-2.5-flash-lite`
- Motor de dados: DuckDB 1.3.2 exclusivamente
- Framework: LangChain + LangGraph
- Output: compativel com `NonGraphOutput` (Pydantic model em `schemas.py`)
- Pipeline: compativel com `GraphState` TypedDict e fluxo do `pipeline_orchestrator.py`

---

## 2. Plano de Fases

### FASE 1: Correcoes Criticas de Bugs

**Objetivo**: Eliminar erros de execucao (BinderException) e corrigir tratamento de colunas virtuais sem alterar a arquitetura fundamental.

#### 1.1 Registro de Colunas Virtuais no AliasMapper

**Arquivo**: `src/graphic_classifier/tools/alias_mapper.py`

**Mudanca**: Adicionar um mapeamento interno que identifique colunas virtuais e suas expressoes DuckDB equivalentes:
```
VIRTUAL_COLUMN_MAP = {
    "Ano": 'YEAR("Data")',
    "Mes": 'MONTH("Data")',
    "Nome_Mes": 'MONTHNAME("Data")',
}
```

Adicionar metodo `is_virtual_column(col_name) -> bool` e `get_virtual_expression(col_name) -> str` para que componentes downstream possam verificar se uma coluna resolvida e virtual e obter a expressao SQL correspondente.

**Resultado esperado**: Componentes podem consultar se "Ano" e virtual e obter `YEAR("Data")` em vez de `"Ano"`.

#### 1.2 Tratamento de Colunas Virtuais no QueryExecutor

**Arquivo**: `src/non_graph_executor/tools/query_executor.py`

**Mudanca**: Em `compute_simple_aggregation()` (linha 211), antes de construir a query, verificar se a coluna e virtual usando `alias_mapper.is_virtual_column()`. Se sim, substituir `"Ano"` por `YEAR("Data")` no SQL.

Estender o tratamento existente em `_build_where_clause()` (linhas 722-728) que ja trata "Ano" em filtros para tambem tratar "Mes":
```python
if col == "Mes" or col == "mes":
    col_escaped = 'MONTH("Data")'
```

**Resultado esperado**: Query 1 ("qual o ultimo ano com vendas?") executaria `SELECT MAX(YEAR("Data")) FROM data` em vez de `SELECT SUM("Ano")`.

#### 1.3 Correcao do Fallback de Agregacao

**Arquivo**: `src/non_graph_executor/tools/query_classifier_params.py`

**Mudanca**: Na funcao `extract_aggregation_params()` (linha 258-260), o default para agregacao e `"sum"`. Quando a query contem palavras como "ultimo/a", "primeiro/a", "mais recente", o default deveria ser `"max"` ou `"min"` respectivamente. Adicionar deteccao:
```python
if any(kw in query_lower for kw in ["ultimo", "ultima", "mais recente", "mais novo"]):
    params["aggregation"] = "max"
elif any(kw in query_lower for kw in ["primeiro", "primeira", "mais antigo", "mais antiga"]):
    params["aggregation"] = "min"
```

**Resultado esperado**: Query 2 ("qual o ano em que ocorreu a ultima venda?") usaria `max` em vez de `sum`.

#### 1.4 Priorizacao de Coluna Temporal em Contexto Temporal

**Arquivo**: `src/non_graph_executor/tools/query_classifier_params.py`

**Mudanca**: Na funcao `_extract_column_name()` (linha 414), quando a query contem palavras temporais ("ano", "mes", "data", "quando", "periodo") E palavras como "ultimo/primeiro/mais recente", priorizar a resolucao para coluna temporal ("Data" ou suas expressoes virtuais) em vez de colunas numericas.

Adicionar logica:
```python
temporal_intent_keywords = ["ultimo", "ultima", "primeiro", "primeira", "quando", "em que ano", "em que mes"]
if any(kw in query_lower for kw in temporal_intent_keywords):
    # Priorizar coluna temporal
    temporal_cols = alias_mapper.column_types.get("temporal", [])
    ...
```

**Resultado esperado**: Query 2 resolveria para coluna "Data" (com YEAR) em vez de "Valor_Vendido".

#### Risco da Fase 1
- **Baixo**: Mudancas pontuais e retrocompativeis
- As correcoes melhoram o comportamento para queries simples temporais mas NAO resolvem o problema fundamental de GROUP BY (CR-1)

---

### FASE 2: Motor de Compreensao de Intent via LLM

**Objetivo**: Criar um componente que use o LLM para entender a intencao completa do usuario e produzir uma especificacao estruturada da consulta necessaria, substituindo a classificacao keyword-based.

#### 2.1 Novo Componente: IntentAnalyzer

**Arquivo novo**: `src/non_graph_executor/tools/intent_analyzer.py`

**Responsabilidade**: Receber a query do usuario e o contexto semantico (alias.yaml, metadata do dataset) e produzir uma especificacao estruturada do que o usuario quer saber.

**Input**:
- `query`: Query em linguagem natural
- `schema_context`: Informacoes do dataset (colunas, tipos, alias)
- `filters`: Filtros ativos da sessao

**Output estruturado** (Pydantic model `QueryIntent`):
```python
class QueryIntent(BaseModel):
    """Especificacao da intencao do usuario."""
    intent_type: Literal["simple_aggregation", "grouped_aggregation", "ranking",
                          "temporal_analysis", "comparison", "lookup", "metadata",
                          "tabular", "conversational"]
    # Colunas de selecao (o que o usuario quer ver)
    select_columns: List[ColumnSpec]
    # Agregacoes a aplicar
    aggregations: List[AggregationSpec]
    # Dimensoes de agrupamento
    group_by: List[ColumnSpec]
    # Ordenacao
    order_by: Optional[OrderSpec]
    # Limite de resultados
    limit: Optional[int]
    # Filtros adicionais detectados na query
    additional_filters: Dict[str, Any]
    # Confianca na interpretacao
    confidence: float
    # Explicacao do raciocinio (para debug)
    reasoning: str

class ColumnSpec(BaseModel):
    """Especificacao de uma coluna."""
    name: str            # Nome real da coluna ou expressao
    is_virtual: bool     # Se requer transformacao (Ano -> YEAR(Data))
    expression: Optional[str]  # Expressao SQL se virtual
    alias: Optional[str]  # Alias para exibicao

class AggregationSpec(BaseModel):
    """Especificacao de uma agregacao."""
    function: Literal["sum", "avg", "count", "min", "max", "median", "std"]
    column: ColumnSpec
    distinct: bool = False
    alias: Optional[str]

class OrderSpec(BaseModel):
    """Especificacao de ordenacao."""
    column: str
    direction: Literal["ASC", "DESC"]
```

**Estrategia de prompt**: O prompt do IntentAnalyzer deve incluir:
1. O schema do dataset (colunas, tipos, a partir de alias.yaml `column_types`)
2. Exemplos do mapeamento de alias (para o LLM saber que "vendas" = Valor_Vendido)
3. A lista de colunas virtuais e suas expressoes (Ano = YEAR(Data), Mes = MONTH(Data))
4. Instrucoes claras sobre como preencher o output estruturado
5. Exemplos de intent para cada tipo (few-shot)

**Modelo recomendado**: `gemini-2.5-flash` (nao lite) para o IntentAnalyzer, pois requer raciocinio mais sofisticado. Usar temperature=0.1 para determinismo.

**Exemplos de transformacao:**

| Query do Usuario | QueryIntent Esperado |
|-----------------|---------------------|
| "qual o ultimo ano com vendas?" | intent_type=simple_aggregation, aggregations=[max(YEAR(Data))], group_by=[], limit=None |
| "qual mes teve maior valor de vendas em 2016?" | intent_type=grouped_aggregation, aggregations=[sum(Valor_Vendido)], group_by=[MONTH(Data)], order_by=DESC, limit=1 |
| "top 5 estados por faturamento" | intent_type=ranking, aggregations=[sum(Valor_Vendido)], group_by=[UF_Cliente], order_by=DESC, limit=5 |
| "quantos clientes temos?" | intent_type=simple_aggregation, aggregations=[count_distinct(Cod_Cliente)], group_by=[] |
| "vendas por mes em 2016" | intent_type=grouped_aggregation, aggregations=[sum(Valor_Vendido)], group_by=[MONTH(Data)], order_by=None |

#### 2.2 Novo Schema: QueryIntent

**Arquivo novo**: `src/non_graph_executor/models/intent_schema.py`

Definir os Pydantic models `QueryIntent`, `ColumnSpec`, `AggregationSpec`, `OrderSpec` conforme descrito acima.

#### 2.3 Refatoracao do QueryClassifier

**Arquivo**: `src/non_graph_executor/tools/query_classifier.py`

**Mudanca**: O QueryClassifier atual (745 linhas) sera simplificado. Em vez de ter 8 niveis de prioridade baseados em keywords, passara a:

1. **Pre-filtro rapido** (mantido): Detectar queries puramente conversacionais (saudacoes, "oi", "ajuda") sem chamar LLM. Este filtro e barato e evita chamadas LLM desnecessarias.

2. **Delegacao ao IntentAnalyzer**: Para todas as outras queries, delegar ao `IntentAnalyzer` para compreensao semantica completa.

3. **Mapeamento de intent_type para query_type**: Converter o `intent_type` do IntentAnalyzer para o `query_type` do schema existente (para compatibilidade):
   - `simple_aggregation`, `grouped_aggregation`, `ranking`, `temporal_analysis` -> `"aggregation"`
   - `comparison` -> `"aggregation"` (com dados multi-dimensionais)
   - `lookup` -> `"lookup"`
   - `metadata` -> `"metadata"`
   - `tabular` -> `"tabular"`
   - `conversational` -> `"conversational"`

**Resultado esperado**: A classificacao passa a ser semantica e contextual, eliminando falsos positivos de keywords e capturando informacao dimensional.

#### Risco da Fase 2
- **Medio**: Mudanca significativa na logica de classificacao
- Mitigacao: Manter o classificador antigo como fallback configuravel durante transicao
- Custo: Aumento de chamadas LLM (1 chamada extra por query para analise de intent)
- Latencia: ~0.3-0.8s adicionais por query (aceitavel para UX do chatbot)

---

### FASE 3: Gerador de SQL Dinamico

**Objetivo**: Criar um componente que converta o `QueryIntent` em uma query DuckDB valida e completa, com suporte a GROUP BY, ORDER BY, LIMIT, e funcoes temporais.

#### 3.1 Novo Componente: DynamicQueryBuilder

**Arquivo novo**: `src/non_graph_executor/tools/dynamic_query_builder.py`

**Responsabilidade**: Receber um `QueryIntent` e gerar a query DuckDB correspondente.

**Metodo principal**:
```python
def build_query(self, intent: QueryIntent, filters: Dict, data_source: str) -> str:
    """Constroi query DuckDB a partir da intencao analisada."""
```

**Capacidades**:
1. **SELECT com expressoes**: Suporte a colunas reais, virtuais (YEAR/MONTH), e alias
2. **Funcoes de agregacao**: SUM, AVG, COUNT, MIN, MAX, MEDIAN, STDDEV_SAMP
3. **GROUP BY**: Agrupamento por uma ou mais dimensoes (incluindo expressoes temporais)
4. **ORDER BY**: Ordenacao por qualquer coluna ou agregacao, ASC ou DESC
5. **LIMIT**: Limitacao de resultados para rankings
6. **WHERE com filtros**: Reutilizar logica existente de `_build_where_clause` do QueryExecutor
7. **HAVING** (futuro): Filtros pos-agrupamento
8. **Validacao de colunas**: Verificar que todas as colunas referenciadas existem (reais ou virtuais)

**Resolucao de colunas virtuais**:
```python
VIRTUAL_COLUMNS = {
    "Ano": {"expression": 'YEAR("Data")', "alias": "Ano"},
    "Mes": {"expression": 'MONTH("Data")', "alias": "Mes"},
    "Nome_Mes": {"expression": 'MONTHNAME("Data")', "alias": "Nome_Mes"},
}
```

**Exemplos de SQL gerado**:

| QueryIntent | SQL DuckDB |
|------------|------------|
| max(YEAR(Data)) | `SELECT MAX(YEAR("Data")) as max_ano FROM 'data/...'` |
| sum(Valor_Vendido) GROUP BY MONTH(Data) ORDER BY DESC LIMIT 1 | `SELECT MONTH("Data") as Mes, SUM("Valor_Vendido") as total_vendas FROM 'data/...' WHERE ... GROUP BY MONTH("Data") ORDER BY total_vendas DESC LIMIT 1` |
| sum(Valor_Vendido) GROUP BY UF_Cliente ORDER BY DESC LIMIT 5 | `SELECT "UF_Cliente", SUM("Valor_Vendido") as total_vendas FROM 'data/...' GROUP BY "UF_Cliente" ORDER BY total_vendas DESC LIMIT 5` |
| count_distinct(Cod_Cliente) | `SELECT COUNT(DISTINCT "Cod_Cliente") as total_clientes FROM 'data/...'` |

#### 3.2 Refatoracao do QueryExecutor

**Arquivo**: `src/non_graph_executor/tools/query_executor.py`

**Mudanca**: Adicionar novo metodo `execute_dynamic_query()` que:
1. Recebe uma query SQL do DynamicQueryBuilder
2. Executa via DuckDB
3. Retorna lista de dicts (mesmo formato atual)
4. Trata erros com mensagens claras

Os metodos existentes (`compute_simple_aggregation`, `lookup_record`, etc.) sao mantidos para backward compatibility e para uso direto quando a query e simples o suficiente.

```python
def execute_dynamic_query(self, sql: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Executa query SQL dinamica gerada pelo DynamicQueryBuilder."""
```

#### 3.3 Refatoracao do Fluxo no Agent

**Arquivo**: `src/non_graph_executor/agent.py`

**Mudanca no metodo `execute()`** (linhas 170-374):

O fluxo atual:
```
1. Conversational check
2. QueryClassifier.classify() -> QueryTypeClassification
3. _execute_by_type(classification) -> data, metadata
4. OutputFormatter.format() -> NonGraphOutput
```

Novo fluxo:
```
1. Conversational check (mantido, sem LLM)
2. IntentAnalyzer.analyze(query, schema_context) -> QueryIntent
3. DynamicQueryBuilder.build_query(intent) -> SQL string
4. QueryExecutor.execute_dynamic_query(sql) -> data
5. OutputFormatter.format(query, data, intent) -> NonGraphOutput
```

**Mudanca no `_execute_by_type()`**: Este metodo com seus 7 branches seria gradualmente substituido. Para queries que o IntentAnalyzer classifica com alta confianca, usar o fluxo dinamico. Para queries de metadata simples (row_count, column_list), manter o fluxo existente via MetadataCache.

#### 3.4 Melhoria do OutputFormatter

**Arquivo**: `src/non_graph_executor/utils/output_formatter.py`

**Mudanca**: O prompt de summary deve receber TODOS os dados retornados pela query dinamica, nao apenas um valor escalar. Por exemplo, se a query retorna `[{Mes: 3, total_vendas: 5200000}]`, o LLM pode gerar "O mes com maior valor de vendas em 2016 foi marco, com R$ 5.200.000,00."

Atualizar `_create_aggregation_prompt()` (linha 244) para lidar com dados multi-dimensionais:
```python
# Se dados tem mais de uma coluna, incluir todas no prompt
if data and len(data[0]) > 1:
    prompt += f"\nDados completos: {data}"
```

#### Risco da Fase 3
- **Medio-Alto**: Mudanca significativa no fluxo de execucao
- Mitigacao: Feature flag para alternar entre fluxo antigo e novo
- Risco de SQL injection: Validar todas as colunas e expressoes contra whitelist do schema
- Risco de queries pesadas: Manter limite de LIMIT padrao e timeout do DuckDB

---

### FASE 4: Testes e Validacao

**Objetivo**: Garantir que a refatoracao corrige os problemas observados e nao introduz regressoes.

#### 4.1 Testes de Regressao (Queries do Log)

**Arquivo novo**: `tests/test_non_graph_regression.py`

Criar testes automatizados para as 4 queries que falharam:

```
Test 1: "qual o ultimo ano com vendas?"
  - Esperado: Resposta contendo o ano (ex: "2016") sem BinderException
  - Verificar: status="success", summary contem um ano valido

Test 2: "qual o ano em que ocorreu a ultima venda?"
  - Esperado: Resposta identificando o ano correto via MAX(YEAR(Data))
  - Verificar: data contem campo com valor de ano, nao soma de vendas

Test 3: "qual foi o mes com maior valor de venda em 2016?"
  - Esperado: Resposta identificando o MES e o VALOR
  - Verificar: data contem campo de mes E campo de valor, summary menciona o nome do mes

Test 4: "que mes foi o maior valor de vendas?"
  - Esperado: Mesmo que Test 3 (sem filtro de ano explicito)
  - Verificar: data contem campo de mes E campo de valor
```

#### 4.2 Testes Unitarios do IntentAnalyzer

**Arquivo novo**: `tests/test_intent_analyzer.py`

Testar que o IntentAnalyzer produz o `QueryIntent` correto para diversas categorias:
- Agregacoes simples (soma, media, contagem)
- Agregacoes com agrupamento (por mes, por estado, por produto)
- Rankings (top N)
- Consultas temporais (ultimo ano, primeiro mes)
- Metadata (quantas linhas, quais colunas)
- Conversacional (saudacoes)
- Lookups (dados do cliente X)

#### 4.3 Testes Unitarios do DynamicQueryBuilder

**Arquivo novo**: `tests/test_dynamic_query_builder.py`

Testar que o SQL gerado e valido e correto:
- Queries com GROUP BY
- Queries com ORDER BY + LIMIT
- Queries com colunas virtuais (YEAR, MONTH)
- Queries com filtros combinados
- Validacao contra SQL injection
- Tratamento de colunas inexistentes

#### 4.4 Testes de Integracao End-to-End

**Arquivo novo**: `tests/test_non_graph_e2e.py`

Testar o fluxo completo: query -> IntentAnalyzer -> DynamicQueryBuilder -> QueryExecutor -> OutputFormatter:
- Verificar que o output final segue schema NonGraphOutput
- Verificar que summaries respondem a pergunta real
- Verificar compatibilidade com pipeline_orchestrator (GraphState)
- Testar com dataset real (DadosComercial_resumido_v02.parquet)

#### 4.5 Suite de Queries de Validacao

Criar um conjunto amplo de queries de teste cobrindo:

**Agregacoes simples:**
- "qual o total de vendas?"
- "quantos clientes temos?"
- "qual a media de valor vendido?"

**Agregacoes com agrupamento:**
- "vendas por estado"
- "vendas por mes em 2016"
- "total de vendas por familia de produto"

**Rankings:**
- "top 5 estados por faturamento"
- "qual produto vendeu mais?"
- "quais os 3 vendedores com maior receita?"

**Temporal:**
- "qual o ultimo ano com vendas?"
- "em que mes ocorreu a maior venda?"
- "vendas no primeiro trimestre de 2016"

**Comparacoes:**
- "vendas em SP vs RJ"
- "comparar faturamento 2015 vs 2016"

#### Risco da Fase 4
- **Baixo**: Testes nao alteram codigo de producao
- Necessario acesso ao dataset real para testes de integracao
- Chamadas LLM nos testes podem ser lentas; considerar mocks para testes unitarios

---

## 3. Arquivos Afetados por Fase

### Fase 1 (Correcoes Criticas)
| Arquivo | Tipo de Mudanca |
|---------|----------------|
| `src/graphic_classifier/tools/alias_mapper.py` | Adicionar virtual column registry |
| `src/non_graph_executor/tools/query_executor.py` | Tratar colunas virtuais em aggregation |
| `src/non_graph_executor/tools/query_classifier_params.py` | Corrigir fallback de agregacao e coluna temporal |

### Fase 2 (IntentAnalyzer)
| Arquivo | Tipo de Mudanca |
|---------|----------------|
| `src/non_graph_executor/tools/intent_analyzer.py` | **NOVO** - Motor de compreensao de intent |
| `src/non_graph_executor/models/intent_schema.py` | **NOVO** - Schemas Pydantic para QueryIntent |
| `src/non_graph_executor/tools/query_classifier.py` | Simplificar, delegar ao IntentAnalyzer |
| `src/non_graph_executor/models/llm_loader.py` | Possivel ajuste de modelo (flash vs flash-lite) |

### Fase 3 (DynamicQueryBuilder)
| Arquivo | Tipo de Mudanca |
|---------|----------------|
| `src/non_graph_executor/tools/dynamic_query_builder.py` | **NOVO** - Gerador de SQL dinamico |
| `src/non_graph_executor/tools/query_executor.py` | Adicionar execute_dynamic_query() |
| `src/non_graph_executor/agent.py` | Refatorar fluxo execute() |
| `src/non_graph_executor/utils/output_formatter.py` | Melhorar prompts para dados multi-dimensionais |

### Fase 4 (Testes)
| Arquivo | Tipo de Mudanca |
|---------|----------------|
| `tests/test_non_graph_regression.py` | **NOVO** - Testes de regressao |
| `tests/test_intent_analyzer.py` | **NOVO** - Testes do IntentAnalyzer |
| `tests/test_dynamic_query_builder.py` | **NOVO** - Testes do DynamicQueryBuilder |
| `tests/test_non_graph_e2e.py` | **NOVO** - Testes end-to-end |

---

## 4. Criterios de Sucesso

### Criterios Minimos (Fase 1)
- [ ] Query "qual o ultimo ano com vendas?" nao gera BinderException
- [ ] Colunas virtuais Ano/Mes sao resolvidas para YEAR(Data)/MONTH(Data)
- [ ] Fallback de agregacao detecta "ultimo/primeiro" corretamente

### Criterios Intermediarios (Fases 2-3)
- [ ] Query "qual mes teve maior valor de vendas em 2016?" retorna o nome/numero do mes E o valor
- [ ] Query "top 5 estados por faturamento" retorna 5 estados com seus valores
- [ ] Queries com GROUP BY executam corretamente via DuckDB
- [ ] IntentAnalyzer produz QueryIntent correto para >90% dos casos de teste
- [ ] SQL gerado pelo DynamicQueryBuilder e valido e seguro (sem SQL injection)

### Criterios Finais (Fase 4)
- [ ] Todas as 4 queries do log da sessao de teste produzem respostas corretas
- [ ] Suite de testes de regressao passa 100%
- [ ] Testes unitarios cobrem IntentAnalyzer e DynamicQueryBuilder
- [ ] Output final compativel com schema NonGraphOutput
- [ ] Pipeline completo funciona sem erros (integracao com pipeline_orchestrator)
- [ ] Nenhum impacto em agentes que nao foram modificados (analytics_executor, insight_generator, etc.)

---

## 5. Consideracoes de Performance

### Latencia
- **Fase 1**: Zero impacto (apenas logica condicional extra)
- **Fases 2-3**: +0.3-0.8s por query (1 chamada LLM adicional para IntentAnalyzer)
- Mitigacao: Usar `gemini-2.5-flash-lite` para queries simples, `gemini-2.5-flash` para complexas

### Custo de Tokens
- **Atual**: ~200 tokens por query (classificacao + summary)
- **Projetado**: ~500-800 tokens por query (intent analysis + summary)
- Mitigacao: Schema context compacto no prompt, few-shot examples minimos

### Cache
- IntentAnalyzer: Nao cachear (queries sao unicas)
- MetadataCache: Manter cache existente para schema/metadata
- DynamicQueryBuilder: Nao cachear (queries dependem de filtros ativos)

---

## 6. Dependencias e Ordem de Execucao

```
Fase 1 (Correcoes Criticas)
    |
    v
Fase 2 (IntentAnalyzer) -----> Fase 3 (DynamicQueryBuilder)
                                    |
                                    v
                               Fase 4 (Testes)
```

- Fase 1 pode ser executada independentemente e ja traz melhorias imediatas
- Fase 2 e pre-requisito para Fase 3 (DynamicQueryBuilder precisa de QueryIntent)
- Fase 4 deve ser desenvolvida em paralelo com Fases 2 e 3 (TDD)
- Feature flags permitem rollback por fase

---

## 7. Riscos e Mitigacoes

| Risco | Probabilidade | Impacto | Mitigacao |
|-------|---------------|---------|-----------|
| LLM gera QueryIntent incorreto | Media | Alto | Few-shot examples, temperatura baixa (0.1), validacao Pydantic |
| SQL gerado e invalido | Baixa | Alto | Validacao de colunas contra schema, try/catch com fallback |
| Aumento de latencia inaceitavel | Baixa | Medio | Modelo lite para queries simples, cache de schema context |
| Regressao em queries que funcionavam | Media | Alto | Feature flag, suite de regressao, rollback por fase |
| SQL injection via prompt | Baixa | Critico | Whitelist de colunas/expressoes, nunca usar input do usuario direto no SQL |
| Incompatibilidade com pipeline existente | Baixa | Alto | Manter NonGraphOutput schema, testes de integracao |
| Custo de tokens excede orcamento | Baixa | Medio | Monitorar via TokenAccumulator existente, ajustar prompts |
