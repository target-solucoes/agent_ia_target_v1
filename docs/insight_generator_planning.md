# Plano de Refatoracao do Insight Generator

**Data:** 2026-02-11
**Referencia:** `docs/insight_generator_diagnosis.md`
**Objetivo:** Transformar o insight_generator de um gerador de relatorio estatistico acoplado ao chart_type em um interpretador inteligente, LLM-first e intention-driven.

---

## 1. Visao Geral da Estrategia

A refatoracao segue o principio de que **a qualidade da resposta depende primariamente da qualidade do prompt**. Os problemas diagnosticados sao em sua maioria estruturais (o que entra no prompt), nao de capacidade do modelo. A estrategia prioriza corrigir o fluxo de informacao antes de otimizar componentes individuais.

```
ANTES:
  Calculator(chart_type) -> metricas -> template(chart_type) -> LLM -> relatorio fixo

DEPOIS:
  user_query + data + filters + enriched_intent -> prompt_dinamico -> LLM -> resposta adaptativa
```

---

## 2. Fases de Implementacao

### FASE 1: Injetar intencao do usuario e dados no prompt da LLM

**Prioridade:** CRITICA
**Impacto:** Resolve P1, P3, P5, P7
**Risco:** Baixo (aditivo, nao quebra contrato existente)

#### 1.1 Passar `user_query` ao prompt

**Arquivo:** `src/insight_generator/graph/nodes.py` - `build_prompt_node()`

Extrair `user_query` do state (ja disponivel via `chart_spec.user_query` ou `analytics_result.metadata.user_query`) e passa-lo para `build_prompt()`.

**Arquivo:** `src/insight_generator/formatters/prompt_builder.py` - `build_prompt()`

Adicionar parametro `user_query` a assinatura. Incluir a pergunta do usuario como primeiro elemento do prompt:

```
PERGUNTA DO USUARIO:
"{user_query}"

Sua resposta DEVE responder diretamente a esta pergunta.
```

#### 1.2 Incluir dados reais no prompt

**Arquivo:** `src/insight_generator/graph/nodes.py` - `build_prompt_node()`

Formatar o DataFrame (`state["data"]`) como tabela markdown e incluir no prompt. Limitar a 20 linhas para controle de tokens.

**Arquivo:** `src/insight_generator/formatters/prompt_builder.py`

Adicionar parametro `data_table` a `build_prompt()`. Incluir no prompt:

```
DADOS DISPONVEIS:
{data_table}

Use estes dados para fundamentar sua resposta com valores especificos.
```

#### 1.3 Incluir enriched_intent no prompt

**Arquivo:** `src/insight_generator/graph/nodes.py` - `build_prompt_node()`

Extrair `enriched_intent` do state e formatar como contexto para a LLM.

**Arquivo:** `src/insight_generator/formatters/prompt_builder.py`

Adicionar parametro `intent_context` a `build_prompt()`:

```
CONTEXTO DA ANALISE:
- Intencao: {narrative_angle}
- Polaridade: {polarity} (focar em {positive: oportunidades / negative: riscos / neutral: panorama geral})
- Foco temporal: {temporal_focus}
- Tipo de comparacao: {comparison_type}
```

#### 1.4 Incluir filtros com contexto semantico

Manter a funcionalidade existente de `_format_filters_for_prompt()`, mas integrar de forma mais natural no prompt (nao como secao separada com instrucoes de bold obrigatorio, mas como contexto da analise).

**Criterios de sucesso da FASE 1:**
- [ ] user_query aparece no prompt enviado a LLM
- [ ] Dados tabulares (ate 20 linhas) aparecem no prompt
- [ ] enriched_intent.narrative_angle aparece no prompt
- [ ] Filtros ativos aparecem no prompt com contexto
- [ ] Contrato de saida (InsightOutput) permanece compativel

---

### FASE 2: Redesenhar o prompt para ser intention-driven

**Prioridade:** ALTA
**Impacto:** Resolve P2, P4
**Risco:** Medio (altera comportamento de saida)

#### 2.1 Substituir SYSTEM_PROMPT rigido por prompt dinamico

**Arquivo:** `src/insight_generator/formatters/prompt_builder.py`

Remover o `SYSTEM_PROMPT` atual de 177 linhas com estrutura fixa de 4 secoes.

Substituir por um prompt que:
1. Posiciona a LLM como analista de dados comerciais
2. Apresenta a pergunta do usuario como objetivo principal
3. Fornece dados e contexto
4. Orienta o formato de resposta baseado no tipo de intencao (nao no chart_type)
5. Permite formato livre (texto, tabela, comparacao)

**Novo prompt (estrutura conceitual):**

```
Voce e um analista de dados comerciais. Responda a pergunta do usuario
de forma direta, clara e util para tomada de decisao.

PERGUNTA: "{user_query}"

FILTROS ATIVOS: {filters_context}
(Mencione os filtros naturalmente na resposta quando relevante.)

DADOS:
{data_table}

METRICAS AUXILIARES:
{numeric_summary_resumido}

CONTEXTO: {narrative_angle}

DIRETRIZES:
- Responda a pergunta diretamente na primeira frase.
- Use valores especificos dos dados (nomes, numeros, percentuais).
- Escolha o formato mais adequado:
  * Para rankings: liste os itens com valores.
  * Para tendencias: descreva a evolucao com pontos-chave.
  * Para comparacoes: destaque diferencas e padroes.
  * Para distribuicoes: identifique dominantes e oportunidades.
- Seja conciso. Nao gere secoes desnecessarias.
- Nao use emojis.
- Linguagem profissional e acessivel (evite jargao estatistico).
```

#### 2.2 Remover CHART_TYPE_TEMPLATES

**Arquivo:** `src/insight_generator/formatters/prompt_builder.py`

Remover o dicionario `CHART_TYPE_TEMPLATES` que contem 8 templates fixos por chart_type. Em seu lugar, usar diretrizes baseadas na `enriched_intent`:

**Mapeamento intent -> diretriz de formato:**

| Intent | Diretriz |
|--------|----------|
| ranking | "Liste os itens em ordem, com valores. Identifique lideres e gaps." |
| trend/temporal | "Descreva a evolucao cronologica. Destaque picos, vales e tendencias." |
| comparison | "Compare explicitamente os elementos. Use diferencas absolutas e percentuais." |
| distribution | "Identifique os dominantes e suas participacoes. Destaque concentracao." |
| composition | "Descreva a composicao por componente. Identifique padroes." |
| variation | "Foque nas mudancas. Ranking de maiores variacoes (positivas ou negativas)." |

#### 2.3 Flexibilizar estrutura de saida

**Arquivo:** `src/insight_generator/formatters/prompt_builder.py`

A nova resposta da LLM sera um JSON simples:

```json
{
  "resposta": "Texto da resposta direta ao usuario",
  "dados_destacados": [
    {"item": "nome", "valor": 1234, "contexto": "maior queda"}
  ],
  "filtros_mencionados": ["UF_Cliente: SC", "Periodo: 2015"]
}
```

Ou, para respostas tabulares:

```json
{
  "resposta": "Texto introdutorio",
  "tabela": {
    "headers": ["Cliente", "Fev/2015", "Mar/2015", "Variacao"],
    "rows": [...]
  },
  "filtros_mencionados": [...]
}
```

A estrutura e flexivel -- a LLM escolhe se inclui tabela, lista, ou texto corrido baseado na pergunta.

#### 2.4 Simplificar metricas pre-calculadas

As metricas pre-calculadas mudam de papel: de "roteiro para a LLM" para "contexto auxiliar". Manter apenas:
- Total geral
- Top N com valores
- Min/Max com labels
- Variacao total (quando temporal)

Remover metricas academicas: HHI, diversidade de Simpson, score de balanceamento, coeficiente de variacao.

**Criterios de sucesso da FASE 2:**
- [ ] Prompt nao contem CHART_TYPE_TEMPLATES
- [ ] Resposta da LLM e texto livre (nao 4 secoes fixas)
- [ ] user_query e o primeiro elemento do prompt
- [ ] Dados reais sao acessiveis pela LLM
- [ ] Formato de resposta adapta-se a pergunta

---

### FASE 3: Simplificar pipeline e otimizar modelo

**Prioridade:** MEDIA
**Impacto:** Resolve P6, P8
**Risco:** Medio (altera integracao com formatter e pipeline)

#### 3.1 Upgrade do modelo padrao

**Arquivo:** `src/insight_generator/models/insight_schemas.py` - `load_insight_llm()`

Alterar modelo padrao de `gemini-2.5-flash-lite` para `gemini-2.5-flash`.

**Arquivo:** `src/insight_generator/core/settings.py`

Adicionar configuracao de selecao de modelo:

```python
INSIGHT_MODEL_DEFAULT = "gemini-2.5-flash"
INSIGHT_MODEL_LITE = "gemini-2.5-flash-lite"
INSIGHT_TEMPERATURE_DEFAULT = 0.4  # balanco entre criatividade e consistencia
```

#### 3.2 Logica de selecao de modelo

**Arquivo:** `src/insight_generator/models/insight_schemas.py`

Adicionar funcao `select_insight_model(enriched_intent)`:

```python
def select_insight_model(enriched_intent):
    """
    Seleciona modelo baseado na complexidade da query.

    flash: comparacoes, variacoes, composicoes, polaridade negativa
    flash-lite: rankings simples, metricas unicas, metadata
    """
    if enriched_intent is None:
        return INSIGHT_MODEL_DEFAULT

    # Queries complexas -> flash
    complex_intents = {"comparison", "variation", "composition", "trend"}
    if enriched_intent.base_intent in complex_intents:
        return INSIGHT_MODEL_DEFAULT

    if enriched_intent.polarity == Polarity.NEGATIVE:
        return INSIGHT_MODEL_DEFAULT

    if enriched_intent.comparison_type != ComparisonType.NONE:
        return INSIGHT_MODEL_DEFAULT

    # Queries simples -> flash-lite
    simple_intents = {"ranking", "distribution"}
    if enriched_intent.base_intent in simple_intents:
        if enriched_intent.temporal_focus == TemporalFocus.SINGLE_PERIOD:
            return INSIGHT_MODEL_LITE

    return INSIGHT_MODEL_DEFAULT
```

**Criterios objetivos para fallback:**

| Criterio | Modelo |
|----------|--------|
| `base_intent` in {comparison, variation, composition, trend} | flash |
| `polarity` == NEGATIVE | flash |
| `comparison_type` != NONE | flash |
| `temporal_focus` in {PERIOD_OVER_PERIOD, TIME_SERIES} | flash |
| `base_intent` in {ranking, distribution} + SINGLE_PERIOD | flash-lite |
| Sem enriched_intent (fallback) | flash |

#### 3.3 Simplificar validacao

**Arquivo:** `src/insight_generator/graph/nodes.py`

Simplificar `validate_insights_node()`:
- Remover FASE 4/5 alignment validation (projetada para estrutura rigida de 4 secoes)
- Manter validacao basica: JSON valido, resposta nao vazia, sem emojis
- Remover retry logic para alignment (nao mais necessario com formato livre)

**Arquivos a simplificar ou remover:**
- `src/insight_generator/utils/alignment_validator.py` - Reduzir a validacao basica
- `src/insight_generator/utils/alignment_corrector.py` - Remover (nao mais necessario)
- `src/insight_generator/utils/transparency_validator.py` - Simplificar (sem formulas obrigatorias)

#### 3.4 Simplificar formatter

**Arquivo:** `src/formatter_agent/graph/nodes.py`

Reduzir o formatter a:
1. Consolidar outputs dos agentes paralelos (insight + plotly)
2. Montar JSON de saida com metadata
3. Nao fazer chamadas LLM adicionais (pass-through do insight_generator)

O `insight_result.resposta` ja e a resposta final para o usuario. O formatter apenas estrutura a saida do pipeline.

#### 3.5 Simplificar workflow do insight_generator

**Arquivo:** `src/insight_generator/graph/workflow.py`

Simplificar de 7 nodes para 4:

```
ANTES: parse_input -> calculate_metrics -> build_prompt -> invoke_llm -> validate -> to_markdown -> format_output
DEPOIS: parse_input -> build_prompt -> invoke_llm -> format_output
```

- `calculate_metrics`: Reduzir a calculo de metricas auxiliares simples (inline em parse_input)
- `to_markdown`: Remover (LLM ja gera resposta formatada)
- `validate`: Simplificar e mover para dentro de format_output

**Criterios de sucesso da FASE 3:**
- [ ] Modelo padrao e gemini-2.5-flash
- [ ] Selecao de modelo e automatica baseada em enriched_intent
- [ ] Pipeline tem 4 nodes em vez de 7
- [ ] Formatter nao faz chamadas LLM
- [ ] Validacao e basica (JSON valido, sem emojis)

---

### FASE 4: Testes e validacao de qualidade

**Prioridade:** ALTA (pos-implementacao)
**Risco:** Baixo

#### 4.1 Suite de testes de benchmark

Criar suite de testes usando os samples de `docs/insight_generator_samples.md`.

**Estrutura do teste:**

```python
# tests/test_insight_quality.py

def test_ranking_query_responds_with_list():
    """Verifica que ranking queries listam itens com valores."""
    query = "top 5 clientes por faturamento"
    result = run_insight_pipeline(query, filters={})
    assert "R$" in result["resposta"]
    # Verifica que nomes de clientes aparecem na resposta

def test_drop_query_identifies_biggest_drop():
    """Verifica que queries sobre queda identificam o maior drop."""
    query = "quais clientes tiveram maior queda entre fev e mar/2015?"
    result = run_insight_pipeline(query, filters={"Cod_Vendedor": "000145"})
    assert "20524" in result["resposta"]  # cliente com maior queda
    assert "100%" in result["resposta"] or "zero" in result["resposta"].lower()

def test_history_query_describes_pattern():
    """Verifica que queries de historico descrevem o padrao temporal."""
    query = "qual o historico de compras de 2015?"
    result = run_insight_pipeline(query, filters={"Cod_Vendedor": "000145"})
    assert "marco" in result["resposta"].lower() or "R$ 2,4" in result["resposta"]  # pico
```

#### 4.2 Teste comparativo flash-lite vs flash

Para cada sample, executar com ambos os modelos e comparar:
- Aderencia a pergunta (resposta direta?)
- Uso de dados reais (nomes e valores especificos?)
- Qualidade narrativa (fluida vs template?)
- Ausencia de ruido (metricas irrelevantes?)

#### 4.3 Teste de regressao

Garantir que o contrato de saida do insight_generator permanece compativel com:
- `pipeline_orchestrator.py` - Estado do pipeline
- `formatter_agent/` - Formatter recebe output esperado
- `plotly_generator/` - Execucao paralela nao afetada

#### 4.4 Criterios objetivos de sucesso

| Criterio | Metrica | Alvo |
|----------|---------|------|
| Resposta direta | % de respostas que respondem a pergunta na 1a frase | >= 90% |
| Uso de dados | % de respostas com nomes/valores especificos | >= 85% |
| Ausencia de ruido | % de respostas sem metricas irrelevantes (HHI, CV, etc.) | >= 95% |
| Mencao de filtros | % de respostas que contextualizam filtros ativos | >= 90% |
| Formato adequado | % de respostas com formato apropriado a pergunta | >= 80% |
| Latencia | Tempo do insight_generator (p95) | <= 3s |
| Tokens | Media de tokens de output | <= 500 |

---

## 3. Estrategia de Modelos

### Configuracao Recomendada

| Agente | Modelo | Justificativa |
|--------|--------|---------------|
| filter_classifier | flash-lite | Classificacao estruturada, baixa complexidade |
| graphic_classifier | flash | Compreensao semantica necessaria para chart mapping |
| analytics_executor | N/A (DuckDB) | Execucao deterministica |
| **insight_generator** | **flash (padrao)** | **Resposta ao usuario, requer qualidade narrativa** |
| plotly_generator | N/A (Plotly) | Geracao deterministica de grafico |
| formatter | flash-lite (ou nenhum) | Pass-through com formatacao minima |

### Criterios para Escalar para Modelos Mais Robustos

Se, apos a refatoracao, `gemini-2.5-flash` demonstrar limitacoes em cenarios especificos, considerar escalar para `gemini-2.5-pro` (ou equivalente) nos seguintes casos:

1. **Raciocinio multi-step:** Queries que requerem encadeamento logico (ex: "por que as vendas cairam?" -> identificar periodo de queda -> correlacionar com fatores -> sugerir causa)
2. **Analise causal:** Perguntas de "por que?" que requerem inferencia alem dos dados disponiveis
3. **Comparacoes multi-dimensionais aninhadas:** "Top 5 produtos nos top 5 clientes comparado com o mesmo periodo do ano anterior" (3 dimensoes simultaneas)
4. **Predicao com confianca:** "Qual a tendencia para os proximos meses?" requer capacidade preditiva
5. **Sintese de contexto extenso:** Quando o prompt excede 4K tokens com dados + contexto

**Abordagem incremental:** Implementar com flash. Monitorar qualidade via metricas da FASE 4.4. Escalar para pro apenas com evidencia empirica de limitacao.

---

## 4. Ajustes na Orquestracao e Contratos de Entrada

### Preservar

- Contrato de entrada: `chart_spec` + `analytics_result` (+ `plotly_result` opcional)
- Execucao paralela: insight_generator || plotly_generator
- Token tracking: `agent_tokens["insight_generator"]`
- Estado do pipeline: `FilterGraphState`

### Modificar

- **Saida do insight_generator:** De 4 secoes fixas para JSON flexivel com `resposta` como campo principal
- **Formatter:** De 3 chamadas LLM para pass-through
- **Contrato com formatter:** `insight_result` contera `resposta` (string), `dados_destacados` (list), `filtros_mencionados` (list) em vez de `executive_summary`, `detailed_insights`, `synthesized_insights`, `next_steps`

### Backward Compatibility

Para garantir transicao suave:
1. Manter campos legados (`formatted_insights`, `insights`) no output por 1 ciclo de release
2. Novo campo `resposta` coexiste com campos legados
3. Formatter detecta versao do output e adapta processamento
4. Apos validacao, remover campos legados

---

## 5. Principios Arquiteturais

### Escalabilidade
- Prompt dinamico baseado em intent (nao hardcoded por chart_type)
- Selecao de modelo baseada em complexidade (nao fixa)
- Metricas auxiliares como contexto (nao como roteiro)

### Modularidade
- Separacao clara: visualizacao (Plotly) vs interpretacao (Insight)
- IntentEnricher como unica fonte de contexto semantico
- Calculator reduzido a provider de metricas auxiliares

### Ausencia de Hardcoding
- Nenhum template fixo por chart_type no prompt
- Nenhuma estrutura de saida fixa (5 insights, 3 recomendacoes)
- Formato de resposta determinado pela LLM baseado na pergunta

---

## 6. Cronograma Sugerido

| Fase | Escopo | Dependencia |
|------|--------|-------------|
| FASE 1 | Injecao de user_query, dados e intent no prompt | Nenhuma |
| FASE 2 | Redesign do prompt e formato de saida | FASE 1 |
| FASE 3 | Simplificacao do pipeline e upgrade de modelo | FASE 2 |
| FASE 4 | Testes e validacao | FASE 3 |

Cada fase pode ser implementada e validada independentemente. FASE 1 ja produz melhoria significativa de qualidade sem quebrar o pipeline existente.

---

## 7. Riscos e Mitigacoes

| Risco | Probabilidade | Impacto | Mitigacao |
|-------|--------------|---------|-----------|
| Prompt longo excede limite de tokens | Baixa | Alto | Limitar dados a 20 linhas, comprimir metricas auxiliares |
| flash produz respostas inconsistentes | Media | Medio | Temperature=0.4, validacao basica de estrutura |
| Formatter quebra com novo formato | Media | Alto | Backward compatibility por 1 ciclo de release |
| Latencia aumenta com flash | Alta | Baixo | flash-lite para queries simples, parallelismo mantido |
| Regressao em queries existentes | Media | Alto | Suite de testes de regressao com 5 queries de baseline |
