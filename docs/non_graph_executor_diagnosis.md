# Diagnostico Completo: Agente non_graph_executor

## 1. Resumo Executivo

O agente `non_graph_executor` apresenta falhas sistematicas ao responder perguntas analiticas sobre dados comerciais. Em uma sessao de teste com 4 queries (sessao `881118fc_20260209_094828`), **100% das respostas falharam em atender a intencao real do usuario**: 1 erro de execucao (BinderException) e 3 respostas semanticamente incorretas (marcadas como "success" pelo sistema, mas que nao respondem a pergunta feita).

A causa fundamental e uma **arquitetura rigida baseada em regras hardcoded e SQL deterministico**, que nao consegue:
- Compreender a intencao semantica real do usuario
- Gerar queries com GROUP BY, ORDER BY ou funcoes temporais
- Diferenciar colunas reais de colunas virtuais no mapeamento de alias

O resultado e um agente que executa operacoes tecnicamente corretas (SELECT MAX, SELECT SUM) que **nao correspondem a pergunta feita**, produzindo respostas incompletas ou enganosas.

---

## 2. Analise Detalhada por Query

### 2.1 Query 1: "qual o ultimo ano com vendas?"

| Campo | Detalhe |
|-------|---------|
| **Intencao do usuario** | Descobrir em qual ano ocorreu a venda mais recente no dataset |
| **Resposta esperada** | "O ultimo ano com vendas no dataset e 2016." |
| **Resposta real** | ERRO: BinderException - coluna "Ano" nao encontrada |
| **Status reportado** | `error` |

**Rastreamento no codigo:**

1. **Classificacao** (`query_classifier.py:268-448`): A query nao faz match com nenhum keyword pattern. A palavra "ano" aparece em `BUSINESS_KEYWORDS` (linha 248), impedindo classificacao como conversational. Nenhum pattern de aggregation faz match direto. Cai no **LLM fallback** (linha 448 -> `_llm_classify`).

2. **LLM Fallback** (`query_classifier.py:649-744`): O LLM classifica como `metadata`. O classificador entao chama `_extract_metadata_params` (linha 725).

3. **Extracao de parametros** (`query_classifier_params.py:20-143`): Nenhum keyword de metadata faz match ("ultimo ano" nao esta listado). Cai no `else` default (linha 138-141), retornando `metadata_type: "sample_rows", n: 5`.

4. **Execucao no agente** (`agent.py:442-517`): Porem, o log mostra que a query real executada foi `SELECT SUM("Ano")`, indicando que houve uma reclassificacao ou um fluxo diferente onde o LLM retornou `aggregation` com `column=Ano`. A resolucao de alias mapeia "ano" -> "Ano" (`alias.yaml:67-72`).

5. **Erro no QueryExecutor** (`query_executor.py:308-347`): O SQL `SELECT SUM("Ano") as result FROM 'data/datasets/...'` falha porque a coluna "Ano" **nao existe fisicamente** no dataset. O dataset possui apenas a coluna "Data" (tipo temporal/DATE).

**Causas-raiz identificadas:**
- **CR-2**: alias.yaml mapeia "ano" -> "Ano", mas "Ano" nao e uma coluna real do dataset
- **CR-8**: AliasMapper resolve "ano" para "Ano" sem validar se a coluna existe fisicamente
- **CR-7**: O sistema tenta SUM("Ano") em vez de MAX(YEAR("Data")), pois nao sabe gerar funcoes temporais

---

### 2.2 Query 2: "qual o ano em que ocorreu a ultima venda?"

| Campo | Detalhe |
|-------|---------|
| **Intencao do usuario** | Descobrir o ano da venda mais recente |
| **Resposta esperada** | "A ultima venda ocorreu no ano de 2016." (via `SELECT YEAR(MAX("Data")) FROM data`) |
| **Resposta real** | "A ultima venda ocorreu em 2016, totalizando R$ 47.833.605,23." |
| **Status reportado** | `success` (semanticamente incorreto) |

**Rastreamento no codigo:**

1. **Filtros aplicados**: O filter_classifier detectou "2016" na query e aplicou filtro `Data BETWEEN '2016-01-01' AND '2016-12-31'`. Isso ja limitou os dados a 2016 antes da execucao.

2. **Classificacao** (`query_classifier.py:408-416`): O keyword "ultima" (feminino) nao consta nos patterns de aggregation diretamente. Porem, a palavra "venda" (presente em `AGGREGATION_PATTERNS` via `("total de", ["vendas", ...])`) ou "ano" em `BUSINESS_KEYWORDS` pode ter influenciado. O LLM fallback foi usado (0.63s de classification_time).

3. **Extracao de parametros** (`query_classifier_params.py:145-273`): Nenhum keyword de agregacao especifico faz match ("ultimo/ultima" nao esta nos patterns). Cai no default (linha 259-260): `aggregation: "sum"`.

4. **Resolucao de coluna** (`query_classifier_params.py:414-556`): A funcao `_extract_column_name` detecta "venda" na query, que via AliasMapper resolve para `Valor_Vendido` (alias.yaml:163-173). Como e uma agregacao numerica e `Valor_Vendido` e coluna numerica, e selecionada com prioridade (linhas 493-503).

5. **Execucao** (`query_executor.py:211-347`): Executa `SELECT SUM("Valor_Vendido") as result FROM data WHERE "Data" BETWEEN '2016-01-01' AND '2016-12-31'`. Resultado: 47,833,605.23.

6. **Summary via LLM** (`output_formatter.py:244-270`): O prompt de aggregation envia ao LLM: `Query: "qual o ano em que ocorreu a ultima venda?", Resultado: {sum_Valor_Vendido: 47833605.23}, Filtros: {Data: [2016-01-01, 2016-12-31]}`. O LLM **infere/hallucina** "2016" a partir dos filtros, nao do dado real.

**Causas-raiz identificadas:**
- **CR-4**: A classificacao nao entende que "ultimo ano" requer MAX sobre dimensao temporal, nao SUM sobre valor
- **CR-5**: ParameterExtractor resolve "venda" -> Valor_Vendido, ignorando que o usuario quer o ANO, nao o VALOR
- **CR-7**: Deveria gerar `SELECT YEAR(MAX("Data")) FROM data`, mas o sistema so sabe fazer `SELECT AGG(col) FROM data`
- **CR-6**: O OutputFormatter recebe `{sum_Valor_Vendido: 47833605.23}` e nao tem como determinar o ano correto

---

### 2.3 Query 3: "quais foi o mes com maior valor de venda em 2016?"

| Campo | Detalhe |
|-------|---------|
| **Intencao do usuario** | Identificar QUAL mes de 2016 teve o maior valor total de vendas |
| **Resposta esperada** | "O mes com maior valor de venda em 2016 foi [nome do mes], com R$ [valor]." (via `SELECT MONTHNAME("Data"), SUM("Valor_Vendido") FROM data WHERE YEAR("Data")=2016 GROUP BY MONTHNAME("Data") ORDER BY 2 DESC LIMIT 1`) |
| **Resposta real** | "Em 2016, o mes com o maior valor de venda registrou R$ 489.339,41." |
| **Status reportado** | `success` (semanticamente incorreto - nao informa o mes) |

**Rastreamento no codigo:**

1. **Classificacao** (`query_classifier.py:405-416`): O keyword "maior" faz match com `AGGREGATION_PATTERNS` em `("maior", None)` (linha 149). Classificado como `aggregation` com confianca 0.85.

2. **Extracao de parametros** (`query_classifier_params.py:191-202`): "maior" detectado -> `aggregation: "max"`. A funcao `_extract_column_name` (linha 263) resolve "venda" -> `Valor_Vendido`.

3. **Parametros finais**: `{aggregation: "max", column: "Valor_Vendido"}`. **O parametro "mes" (dimensao de agrupamento) e completamente ignorado.** Nao existe nenhum campo para capturar a dimensao de GROUP BY no schema `QueryTypeClassification`.

4. **Execucao** (`query_executor.py:211-347`): Gera `SELECT MAX("Valor_Vendido") as result FROM data WHERE "Data" BETWEEN '2016-01-01' AND '2016-12-31'`. Resultado: 489,339.41. Isto retorna o valor maximo de UMA UNICA LINHA de venda, nao a soma mensal maxima.

5. **Summary via LLM** (`output_formatter.py:244-270`): Recebe `{max_Valor_Vendido: 489339.41}` com filtro de 2016. O LLM nao tem como saber qual mes corresponde a esse valor. Gera summary generico que **omite a informacao principal pedida** (o nome do mes).

**Causas-raiz identificadas:**
- **CR-1**: O QueryExecutor nao suporta GROUP BY. A query correta seria `SELECT MONTH("Data") as mes, SUM("Valor_Vendido") as total FROM data WHERE ... GROUP BY mes ORDER BY total DESC LIMIT 1`
- **CR-3**: O tipo "aggregation" na taxonomia atual so suporta `SELECT AGG(col) FROM data WHERE...`, sem dimensoes
- **CR-5**: ParameterExtractor captura `aggregation=max` e `column=Valor_Vendido`, mas ignora completamente o "mes" como dimensao de agrupamento
- **CR-6**: O OutputFormatter recebe um unico numero e nao consegue identificar o mes correspondente
- **CR-2**: "mes" resolveria para coluna virtual "Mes" que nao existe fisicamente

**Este e o exemplo critico citado na descricao do problema.** A resposta "registrou R$ 489.339,41" sem identificar o mes e uma falha direta na capacidade do agente.

---

### 2.4 Query 4: "que mes foi o maior valor de vendas?"

| Campo | Detalhe |
|-------|---------|
| **Intencao do usuario** | Mesmo que Query 3, sem especificar o ano |
| **Resposta esperada** | "O mes com maior valor de vendas [no periodo filtrado] foi [nome do mes de 2016], com R$ [valor]." |
| **Resposta real** | "O maior valor de vendas registrado em 2016 foi de R$ 489.339,41." |
| **Status reportado** | `success` (semanticamente incorreto - nao informa o mes) |

**Rastreamento no codigo:**

Identico a Query 3. Os mesmos problemas se aplicam:
1. "maior" -> aggregation=max
2. "vendas" -> column=Valor_Vendido
3. "mes" -> ignorado (nao existe campo para dimensao de agrupamento)
4. Filtro de 2016 aplicado pelo filter_classifier (persistido da sessao anterior)
5. `SELECT MAX("Valor_Vendido") FROM data WHERE...` -> 489,339.41
6. Summary omite o mes

**Agravante adicional:** O filtro de 2016 foi aplicado automaticamente por **persistencia de sessao** do filter_classifier, nao porque o usuario especificou 2016 nesta query. O usuario perguntou genericamente "que mes foi o maior valor de vendas?" sem restricao temporal, mas recebeu resposta limitada a 2016.

---

## 3. Causas-Raiz Arquiteturais

### CR-1: Ausencia de GROUP BY no QueryExecutor

**Severidade: CRITICA**

O `QueryExecutor` (`query_executor.py`) so implementa queries com a estrutura `SELECT AGG(col) FROM data WHERE...`. Nao existe nenhum metodo que suporte:
- `GROUP BY` (agrupamento por dimensao)
- `ORDER BY` (ordenacao de resultados)
- `LIMIT` com contexto de ranking (top N)
- Sub-selects ou CTEs

Metodos existentes e suas limitacoes:
- `compute_simple_aggregation()` (linha 211): Apenas `SELECT AGG(col) FROM data WHERE...`
- `get_tabular_data()` (linha 85): Apenas `SELECT * FROM data WHERE... LIMIT N`
- `get_sample_rows()` (linha 156): Apenas `SELECT * FROM data WHERE... LIMIT N`
- `lookup_record()` (linha 353): Apenas `SELECT * FROM data WHERE col=val LIMIT 1`

**Impacto:** Impossivel responder perguntas como "qual mes/estado/produto teve maior/menor X", "top N X por Y", "comparacao entre periodos", etc.

---

### CR-2: Colunas Virtuais no alias.yaml Sem Tratamento

**Severidade: CRITICA**

O arquivo `alias.yaml` (linhas 46-72) define mapeamentos para "Mes" e "Ano":
```yaml
Mes:
  - "mes", "mensal", "por mes", ...
Ano:
  - "ano", "anual", "por ano", ...
```

Porem, **"Mes" e "Ano" nao sao colunas fisicas do dataset**. O dataset possui apenas a coluna "Data" (tipo DATE/TIMESTAMP). Para obter mes ou ano, seria necessario:
- `YEAR("Data")` para ano
- `MONTH("Data")` ou `MONTHNAME("Data")` para mes

O `_build_where_clause` no QueryExecutor (linha 722-728) trata parcialmente o caso do "Ano" em filtros:
```python
if col == "Ano" or col == "ano":
    col_escaped = 'YEAR("Data")'
```

Porem, este tratamento:
- So funciona para filtros WHERE, nao para SELECT/GROUP BY
- Nao existe tratamento equivalente para "Mes"
- Nao esta documentado como comportamento esperado

**Impacto:** Qualquer tentativa de usar "Ano" ou "Mes" como coluna de SELECT/GROUP BY falha com BinderException.

---

### CR-3: Taxonomia Rigida de Query Types

**Severidade: ALTA**

O sistema define 7 tipos fixos de query (`schemas.py`):
- `metadata` - informacoes sobre o dataset
- `aggregation` - uma unica agregacao simples
- `lookup` - busca de registro especifico
- `textual` - busca textual ou listagem
- `statistical` - estatisticas descritivas
- `tabular` - dados tabulares brutos
- `conversational` - saudacoes/ajuda

**Tipos de perguntas NAO cobertos:**
- Agregacao com agrupamento ("vendas POR mes", "total POR estado")
- Rankings ("top 5 produtos por vendas")
- Comparacoes ("vendas em SP vs RJ")
- Perguntas temporais ("ultimo ano", "primeiro mes")
- Agregacoes compostas ("media e total de vendas")
- Perguntas de proporcao ("% de vendas por estado")

A taxonomia foi desenhada para operacoes atomicas simples, mas perguntas analiticas reais quase sempre envolvem dimensoes e agrupamentos.

**Impacto:** Perguntas analiticas comuns sao forcadas em categorias inadequadas, resultando em respostas incompletas.

---

### CR-4: Classificacao Keyword-Based Sem Compreensao Semantica

**Severidade: ALTA**

O `QueryClassifier` (`query_classifier.py:268-448`) opera em 8 niveis de prioridade baseados em keywords estaticos. Problemas:

1. **Falta de contexto semantico**: "qual foi o mes com maior valor" contem "maior" (-> aggregation/max), mas o significado real e "agrupamento por mes + ordenacao por valor + limite 1". O classificador so ve "maior" -> max.

2. **Prioridade fixa**: A ordem tabular > conversational > metadata > statistical > aggregation > textual > lookup e fixa. Nao ha ponderacao por contexto.

3. **Keywords incompletos**: Termos como "ultimo/ultima", "primeiro/primeira", "qual foi o", "em que" nao possuem patterns dedicados. Caem no LLM fallback, que tem prompt limitado (linhas 661-676) e nao conhece o schema de dados.

4. **Falsos positivos de agregacao**: A presenca de "maior" ou "total" classifica imediatamente como aggregation, sem avaliar se ha dimensao de agrupamento (por mes, por estado, etc).

**Impacto:** Queries com intencao complexa sao reduzidas a operacoes simples, perdendo informacao critica.

---

### CR-5: ParameterExtractor Nao Captura Contexto Dimensional

**Severidade: ALTA**

O `ParameterExtractor` (`query_classifier_params.py`) extrai apenas:
- `aggregation`: tipo de funcao (sum, avg, count, min, max, median, std)
- `column`: coluna alvo da agregacao
- `distinct`: se deve usar DISTINCT
- `filters`: filtros do state

**O que nao extrai:**
- `group_by`: dimensao(oes) de agrupamento
- `order_by`: criterio de ordenacao
- `limit`: numero de resultados (para rankings)
- `temporal_function`: YEAR(), MONTH(), etc.
- `having`: filtro pos-agrupamento

Na query "qual foi o mes com maior valor de venda em 2016":
- Extraido: `{aggregation: "max", column: "Valor_Vendido"}`
- Deveria extrair: `{aggregation: "sum", column: "Valor_Vendido", group_by: "MONTH(Data)", order_by: "DESC", limit: 1}`

Note que ate o tipo de agregacao esta errado: o usuario quer o mes com maior TOTAL (SUM) de vendas, nao o maior valor individual (MAX).

**Impacto:** Perda total de informacao dimensional, tornando impossivel gerar a query SQL correta.

---

### CR-6: OutputFormatter Recebe Dados Incompletos

**Severidade: MEDIA**

O `OutputFormatter` (`output_formatter.py:244-270`) gera summaries via LLM usando prompt de aggregation:
```
Query: "{query}"
Resultado: {data}
Filtros: {filters}
```

Quando os dados sao incompletos (ex: `{max_Valor_Vendido: 489339.41}` sem informacao de mes), o LLM:
- Nao pode inventar a informacao faltante (o mes)
- Gera summary generico que omite a resposta principal
- Ou hallucina informacao dos filtros (como inferir "2016" do filtro temporal)

Este problema e consequencia direta de CR-1 e CR-5: se a query SQL nao retorna os dados necessarios, o formatter nao pode compensa-los.

**Impacto:** Summaries enganosos que aparentam responder a pergunta mas omitem a informacao central.

---

### CR-7: Ausencia de Geracao SQL via LLM

**Severidade: CRITICA**

O LLM e usado em 3 pontos do non_graph_executor:
1. **Classificacao fallback** (`query_classifier.py:649-744`): Determina o tipo de query quando keywords falham
2. **Extracao de parametros de lookup** (`query_classifier_params.py:275-349`): Extrai coluna e valor de busca
3. **Geracao de summary** (`output_formatter.py:154-242`): Gera texto resumo a partir dos resultados

O LLM **NUNCA** e usado para:
- Entender a intencao completa da query (quais informacoes o usuario quer)
- Determinar a estrutura SQL necessaria (GROUP BY, ORDER BY, funcoes)
- Gerar ou validar a query SQL
- Resolver ambiguidades semanticas

A construcao SQL e 100% deterministica, baseada em templates fixos por tipo de query. Isso significa que toda a "inteligencia" do sistema esta nas regras de keyword e nos templates SQL, nao no raciocinio do LLM.

**Impacto:** O sistema nao consegue lidar com perguntas que fogem dos templates pre-definidos, que representam a maioria das perguntas analiticas reais.

---

### CR-8: AliasMapper Nao Diferencia Colunas Reais de Virtuais

**Severidade: MEDIA**

O `AliasMapper` (em `src/graphic_classifier/tools/alias_mapper.py`, reutilizado pelo non_graph_executor) resolve aliases baseado no `alias.yaml`:
- "vendas" -> `Valor_Vendido` (coluna real)
- "ano" -> `Ano` (coluna VIRTUAL)
- "mes" -> `Mes` (coluna VIRTUAL)

Nao existe distincao no AliasMapper entre:
- Colunas fisicas do dataset (Valor_Vendido, Data, UF_Cliente, etc.)
- Colunas virtuais/derivadas (Ano, Mes) que requerem transformacao

O `column_types` no alias.yaml define:
```yaml
column_types:
  numeric: [Valor_Vendido, Peso_Vendido, Qtd_Vendida]
  categorical: [Empresa, Cod_Familia_Produto, ...]
  temporal: [Data]
```

Note que "Ano" e "Mes" nao aparecem em nenhum `column_types`, mas ainda assim possuem mapeamentos em `columns`. O AliasMapper retorna "Ano" ou "Mes" como se fossem colunas validas, sem sinalizar que sao virtuais.

**Impacto:** Componentes downstream (QueryExecutor) recebem nomes de colunas que nao existem e falham com BinderException.

---

## 4. Matriz de Severidade e Impacto

| Causa-Raiz | Severidade | Queries Afetadas | Componente Principal | Tipo de Falha |
|------------|-----------|-------------------|---------------------|---------------|
| **CR-1**: Sem GROUP BY | CRITICA | Q3, Q4 | `query_executor.py` | Resposta incompleta |
| **CR-2**: Colunas virtuais | CRITICA | Q1, Q2 | `alias.yaml` + `query_executor.py` | BinderException / Resposta errada |
| **CR-3**: Taxonomia rigida | ALTA | Q2, Q3, Q4 | `schemas.py` + `query_classifier.py` | Classificacao inadequada |
| **CR-4**: Keywords sem semantica | ALTA | Q1, Q2, Q3, Q4 | `query_classifier.py` | Classificacao errada |
| **CR-5**: Sem contexto dimensional | ALTA | Q3, Q4 | `query_classifier_params.py` | Parametros incompletos |
| **CR-6**: Dados incompletos no formatter | MEDIA | Q2, Q3, Q4 | `output_formatter.py` | Summary enganoso |
| **CR-7**: Sem SQL via LLM | CRITICA | Q1, Q2, Q3, Q4 | Arquitetural | Incapacidade fundamental |
| **CR-8**: Alias sem validacao | MEDIA | Q1 | `alias_mapper.py` | Coluna inexistente |

### Distribuicao de impacto por query:

| Query | Causas-raiz envolvidas | Resultado |
|-------|----------------------|-----------|
| Q1: "ultimo ano com vendas" | CR-2, CR-7, CR-8 | BinderException (erro total) |
| Q2: "ano da ultima venda" | CR-4, CR-5, CR-6, CR-7 | SUM(Valor_Vendido) em vez de YEAR(MAX(Data)) |
| Q3: "mes com maior valor 2016" | CR-1, CR-2, CR-3, CR-5, CR-6 | MAX(Valor_Vendido) sem identificar o mes |
| Q4: "mes maior valor vendas" | CR-1, CR-2, CR-3, CR-5, CR-6 | Identico a Q3 |

---

## 5. Conclusao

O `non_graph_executor` foi projetado para responder perguntas atomicas simples (contagem de linhas, soma total, busca por ID), mas e rotineiramente direcionado a responder perguntas analiticas complexas pelo `graphic_classifier`. O gap entre o que o agente sabe fazer (operacoes simples sem dimensoes) e o que e solicitado (perguntas com agrupamento, ordenacao e contexto temporal) e a raiz de todos os problemas observados.

As 8 causas-raiz se conectam em cascata:
1. O **classificador** (CR-4) nao entende a intencao real
2. O **extrator de parametros** (CR-5) perde informacao dimensional
3. O **mapeamento de alias** (CR-2, CR-8) nao distingue colunas reais de virtuais
4. O **executor** (CR-1) nao sabe gerar SQL com GROUP BY
5. O **formatador** (CR-6) recebe dados incompletos e gera respostas enganosas
6. E em nenhum ponto (CR-7) o LLM e usado para raciocinar sobre a query SQL necessaria

A refatoracao deve abordar estas causas de forma integrada, substituindo a logica deterministica por um mecanismo orientado por raciocinio de LLM, mantendo o DuckDB como motor de execucao e preservando a modularidade do sistema.
