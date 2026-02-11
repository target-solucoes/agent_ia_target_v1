# Diagnostico do Insight Generator

**Data:** 2026-02-11
**Versao do Pipeline:** v07_optimization
**Agente analisado:** `src/insight_generator/`
**Base de evidencias:** `logs/sessions/2026-02-10/outputs/` (5 queries de teste)

---

## 1. Resumo Executivo

O `insights_generator` apresenta falhas estruturais que comprometem fundamentalmente a qualidade das respostas entregues ao usuario. O problema central e que o agente opera como um **gerador de relatorio estatistico acoplado ao tipo de grafico**, em vez de funcionar como um **interpretador inteligente da intencao do usuario**. As respostas sao genericas, desconectadas da pergunta original, e infladas com metricas irrelevantes que nao agregam valor analitico.

A analise de 5 queries de teste revelou que **nenhuma das respostas responde diretamente a pergunta do usuario**. Em todos os casos, o agente produz uma analise estatistica padronizada baseada no chart_type, ignorando completamente o que foi perguntado.

---

## 2. Problemas Identificados

### P1 - Ausencia total da intencao do usuario na geracao de resposta

**Gravidade:** CRITICA
**Impacto:** O usuario nunca recebe uma resposta direta a sua pergunta.

**Evidencias:**

**Query 5:** "Quais clientes que tiveram a maior queda entre fev/2015 e marco/2015?"
- **O que o usuario quer:** Saber quais clientes cairam mais. Resposta esperada: "O cliente 20524 teve a maior queda, caindo 100% (de 5.401 para 0 unidades). O cliente 17761 teve a segunda maior queda, caindo 89.6% (de 4.129 para 430)."
- **O que o agente retorna:** "O spread das series reduziu 19.22% (de 2.521 para 2.037), indicando convergencia. Correlacao media de 1.00 aponta para comportamento altamente sincronizado."
- **Problema:** A resposta nao menciona nenhum cliente, nenhuma queda, nenhum valor de vendas. Fala sobre "convergencia" e "correlacao" -- metricas estatisticas que nao respondem a pergunta.

**Query 4:** "qual o historico de compras de 2015?"
- **O que o usuario quer:** Uma visao geral do comportamento de compras ao longo de 2015.
- **O que o agente retorna:** "A serie apresentou uma variacao de -32.08%, com um valor final de 430,075. A ausencia de outras series na analise impediu a observacao de dinamicas de divergencia ou convergencia."
- **Problema:** Reduz 12 meses de dados ricos (pico em marco de R$ 2.4M, vale em dezembro de R$ 430K, recuperacao em outubro de R$ 2.3M) a uma unica metrica de variacao jan-dez. Menciona "ausencia de series comparativas" -- informacao irrelevante para quem perguntou sobre historico.

**Query 3:** "quais os 5 produtos mais vendidos nos 5 maiores clientes desse mesmo representante?"
- **O que o usuario quer:** Uma lista cruzada: quais produtos e quais clientes.
- **O que o agente retorna:** "Contribuicao dominante de R$ 2,87M (41,96%). Desvio padrao alto (R$ 972.318) indica heterogeneidade."
- **Problema:** Nunca lista os 5 produtos (Conexoes Soldaveis, Tubos Eletroduto Corrugado, etc.) nem os 5 clientes (2855, 22494, etc.). O usuario nao consegue responder "quais sao?" apos ler a resposta.

**Query 1:** "quais foram os maiores representantes de SC?"
- **O que o usuario quer:** Uma lista dos maiores representantes com seus valores.
- **O que o agente retorna:** "Top 15 geram 99.37% da receita. Top 3 respondem por 52.66%." Focus em concentracao e gap competitivo.
- **Problema:** Nao nomeia os representantes claramente no texto narrativo. O usuario precisa consultar a tabela de dados para saber que sao 000145, 000018, 000146.

**Query 2:** "qual a distribuicao dos produtos do representante 000145?"
- **O que o usuario quer:** Como os produtos se distribuem (quais sao maiores, quais sao menores).
- **O que o agente retorna:** "Indice Herfindahl-Hirschman: 1.602 (moderada concentracao). Diversidade = 85.89; Equilibrio = 0.00."
- **Problema:** Usa indice HHI e "equilibrio de Simpson" -- metricas academicas que nao comunicam nada ao usuario comercial. Deveria dizer: "Conexoes Soldaveis domina com 31.4% (3.47M unidades), seguido por Tubos Eletroduto Corrugado com 12.7%."

**Causa raiz:** A `user_query` original nao e passada para o prompt da LLM. A funcao `build_prompt()` recebe apenas `numeric_summary`, `chart_type` e `filters`. O LLM nunca sabe o que o usuario perguntou.

**Localizacao no codigo:**
- `src/insight_generator/graph/nodes.py` - `build_prompt_node()` (linha 561-618): nao injeta `user_query`
- `src/insight_generator/formatters/prompt_builder.py` - `build_prompt()` (linha 715): assinatura nao aceita `user_query`

---

### P2 - Calculadoras de metricas impoem frameworks estatisticos irrelevantes

**Gravidade:** ALTA
**Impacto:** Metricas calculadas nao tem relacao com a pergunta do usuario.

**Evidencias:**

| Query | Chart Type | Metricas Impostas | Metricas Relevantes |
|-------|-----------|-------------------|---------------------|
| Q5 (quedas entre meses) | `line_composed` | spread, correlacao, convergencia, lideranca | variacao por cliente, ranking de quedas |
| Q4 (historico 2015) | `line_composed` | variacao jan-dez, num_series=1, spread=0 | valores mensais, picos, vales, tendencia |
| Q3 (produtos x clientes) | `bar_vertical_stacked` | dominancia, balanceamento, desvio padrao | lista de produtos por cliente, volumes |
| Q2 (distribuicao) | `pie` | HHI, diversidade, equilibrio | participacao % de cada produto |
| Q1 (maiores representantes) | `bar_horizontal` | concentracao top_n, gap, cauda | ranking com valores absolutos |

**Causa raiz:** O sistema de calculadoras (`src/insight_generator/calculators/`) e indexado por `chart_type`. Cada chart_type tem um calculator fixo que produz metricas pre-definidas:
- `bar_horizontal` -> `RankingCalculator` (concentracao, gap, top_n)
- `line_composed` -> `ComposedCalculator` (spread, correlacao, convergencia)
- `bar_vertical_stacked` -> `StackedCalculator` (dominancia, balanceamento)
- `pie` -> `DistributionCalculator` (HHI, diversidade)

O `MetricComposer` (FASE 2) tenta selecionar modulos por `enriched_intent`, mas os modulos internos ainda sao acoplados ao chart_type e produzem metricas estatisticas genericas em vez de metricas orientadas a pergunta.

**Localizacao:**
- `src/insight_generator/calculators/` - Todos os calculadores
- `src/insight_generator/calculators/metric_composer.py` - MetricComposer
- `src/insight_generator/graph/nodes.py` - `calculate_metrics_node()` (linha 193)

---

### P3 - A query do usuario NAO e passada ao prompt da LLM

**Gravidade:** CRITICA
**Impacto:** A LLM nao sabe o que o usuario perguntou. E o problema mais fundamental.

**Analise tecnica:**

O fluxo atual e:
```
parse_input_node -> calculate_metrics_node -> build_prompt_node -> invoke_llm_node
```

Em `build_prompt_node()` (nodes.py:561-618):
```python
def build_prompt_node(state):
    numeric_summary = state["numeric_summary"]
    chart_type = state["chart_type"]
    filters = chart_spec.get("filters", {})
    llm_prompt = build_prompt(numeric_summary, chart_type, filters=filters)
```

A `user_query` esta disponivel no state (via `chart_spec.user_query` ou `analytics_result.metadata.user_query`), mas **nunca e extraida nem passada** para `build_prompt()`.

Em `build_prompt()` (prompt_builder.py:715-759):
```python
def build_prompt(numeric_summary, chart_type, filters=None):
    template = CHART_TYPE_TEMPLATES.get(chart_type, ...)
    dados_formatados = _format_metrics_for_prompt(numeric_summary, chart_type)
    prompt = template.format(dados=dados_formatados)
```

A assinatura da funcao nao aceita `user_query`. O prompt enviado a LLM contem apenas:
1. SYSTEM_PROMPT (regras de formato)
2. Metricas pre-calculadas formatadas
3. Template do chart_type
4. Filtros (quando existem)

**A LLM recebe zero contexto sobre o que o usuario quer saber.**

---

### P4 - Estrutura rigida de 4 secoes imposta em todas as respostas

**Gravidade:** ALTA
**Impacto:** Respostas infladas com conteudo de preenchimento. Ruido excessivo.

O `SYSTEM_PROMPT` (prompt_builder.py:11-177) exige:
1. `executive_summary` (titulo + introducao 50-300 chars)
2. `detailed_insights` (**EXATAMENTE 5** insights com formula + interpretacao)
3. `synthesized_insights` (narrativa 400-800 chars + 3-5 key_findings)
4. `next_steps` (exatamente 3 recomendacoes)

**Problemas desta rigidez:**

- **5 insights obrigatorios:** Para a Query 4 (historico de 2015 com serie unica), o agente e forcado a criar 5 insights. Produz: "Numero de Series = 1.00" e "Spread Inicial = 0.00; Spread Final = 0.00; Convergencia = 0.00%" -- insights sem valor informativo, criados apenas para preencher a cota.
- **3 recomendacoes obrigatorias:** Para queries descritivas como "qual a distribuicao?", o agente gera recomendacoes genericas como "Estabelecer monitoramento continuo dos top performers para identificar mudancas de padrao" -- texto consultoria padrao sem valor contextual.
- **Narrativa 400-800 chars:** Forca narrativas longas mesmo quando a resposta ideal seria 1-2 frases diretas.
- **Formulas obrigatorias:** Cada insight deve ter uma formula com operadores. Para metricas simples, isso gera absurdos como "Numero de Series = 1.00" apresentado como "formula".

**Causa raiz:** SYSTEM_PROMPT desenhado para relatorios analiticos completos, nao para respostas de chatbot.

---

### P5 - Dados reais (DataFrame) NAO sao passados a LLM

**Gravidade:** CRITICA
**Impacto:** A LLM nao pode responder perguntas que requerem dados granulares.

**Fluxo atual:**
```
DataFrame (10-25 linhas de dados)
    |
    v
Calculator.calculate(df, config)  -> numeric_summary (5-15 metricas agregadas)
    |
    v
_format_metrics_for_prompt()      -> string de 5-10 linhas de metricas
    |
    v
LLM prompt (so ve metricas)
```

**O que a LLM ve para a Query 5 (quedas entre fev e mar/2015):**
```
Variacao = (Final - Inicial) / Inicial = (4,098 - 8,085) / 8,085 = -3,987 (-49.31%)
num_series: 5.00
num_periodos: 2.00
spread_inicial: 2,521
spread_final: 2,037
correlacao_media: 1.00
```

**O que a LLM deveria ver:**
```
| Cliente | Fev/2015 | Mar/2015 | Variacao | Variacao % |
|---------|----------|----------|----------|------------|
| 20524   | 5.401    | 0        | -5.401   | -100.0%    |
| 17761   | 4.129    | 430      | -3.699   | -89.6%     |
| 2330    | 8.085    | 4.098    | -3.987   | -49.3%     |
| 14061   | 7.115    | 3.566    | -3.549   | -49.9%     |
| 33777   | 1.695    | 0        | -1.695   | -100.0%    |
```

Sem os dados reais, a LLM nao pode identificar que o cliente 20524 caiu 100%, nem que dois clientes zeraram completamente. Ela so sabe que houve "convergencia" nas series.

**Causa raiz:** `build_prompt_node()` passa apenas `numeric_summary` para `build_prompt()`. O `state["data"]` (DataFrame) esta disponivel mas nunca e incluido no prompt.

---

### P6 - Formatter Agent adiciona camada LLM redundante

**Gravidade:** MEDIA
**Impacto:** Duplicacao de processamento, potencial inconsistencia entre camadas.

O `formatter_agent` (src/formatter_agent/) faz 3 chamadas LLM adicionais:
1. `generate_executive_summary` - Reescreve o executive_summary ja gerado pelo insight_generator
2. `synthesize_insights` - Reescreve a narrativa ja gerada
3. `generate_next_steps` - Reescreve os next_steps ja gerados

**Evidencia:** Nos outputs de teste, o formatter registra `llm_calls_execution_time: 0.0` para todas as chamadas -- indicando que em v07_optimization essas chamadas ja foram otimizadas para pass-through. Porem, a arquitetura ainda suporta essas chamadas LLM redundantes, criando risco de regressao.

O formato final do formatter (`formatter_output`) reorganiza os componentes do insight_generator em uma estrutura propria, potencialmente alterando ou perdendo informacao no processo.

---

### P7 - IntentEnricher desconectado do prompt da LLM

**Gravidade:** ALTA
**Impacto:** Enriquecimento semantico calculado mas desperdicado.

O `IntentEnricher` (src/insight_generator/core/intent_enricher.py) calcula:
- `polarity`: POSITIVE/NEGATIVE/NEUTRAL
- `temporal_focus`: SINGLE_PERIOD/PERIOD_OVER_PERIOD/TIME_SERIES
- `comparison_type`: NONE/CATEGORY_VS_CATEGORY/PERIOD_VS_PERIOD
- `narrative_angle`: string descritiva combinando contextos
- `suggested_metrics`: lista de metricas relevantes

**Esses valores influenciam apenas a selecao de modulos do MetricComposer.** Eles NAO sao passados ao prompt da LLM.

Para a Query 5 ("maior queda entre fev e mar"):
- `polarity` = NEGATIVE (detectou "queda")
- `temporal_focus` = PERIOD_OVER_PERIOD (detectou "entre fev e marco")
- `comparison_type` = PERIOD_VS_PERIOD
- `narrative_angle` = "analise de variacao e mudanca, com foco em quedas e riscos, entre periodos especificos, comparando desempenho temporal"

**Este `narrative_angle` perfeito NUNCA chega a LLM.** Se o LLM recebesse essa informacao, poderia gerar uma resposta focada em quedas e riscos, comparando periodos -- exatamente o que o usuario pediu.

**Causa raiz:** `build_prompt_node()` nao extrai `enriched_intent` do state para injetar no prompt.

---

### P8 - Selecao de modelo subotima para qualidade de insight

**Gravidade:** MEDIA-ALTA
**Impacto:** Respostas formulaicas e template-like, sem profundidade semantica.

**Configuracao atual:**
- `insight_generator`: `gemini-2.5-flash-lite` (confirmado em todos os 5 outputs de teste)
- `graphic_classifier`: `gemini-2.5-flash` (mais capaz)
- `filter_classifier`: `gemini-2.5-flash-lite`

**Analise comparativa:**

| Aspecto | flash-lite | flash |
|---------|-----------|-------|
| Velocidade | ~0.3-0.5s | ~0.8-1.5s |
| Custo por token | ~50% menor | baseline |
| Compreensao semantica | Basica | Boa |
| Qualidade narrativa | Template-like | Contextual e fluida |
| Raciocinio multi-step | Limitado | Adequado |
| Uso ideal | Classificacao, extracao | Interpretacao, narrativa |

O `insight_generator` e o agente mais visivel para o usuario -- e a resposta final que ele le. Usar o modelo mais fraco (flash-lite) neste ponto e uma escolha subotima. O `graphic_classifier` usa flash (modelo mais capaz) para classificacao interna, enquanto a resposta que o usuario realmente le usa flash-lite.

**Recomendacao:**
- `gemini-2.5-flash` como padrao para insight_generator
- `gemini-2.5-flash-lite` apenas para queries simples (single_metric, metadata)
- Considerar Gemini 2.5 Pro no futuro para: analise causal, raciocinio de cadeia longa, predicoes com intervalo de confianca, comparacoes multi-dimensionais aninhadas (top N dentro de top N)

---

## 3. Mapa de Causas Raiz

```
PROBLEMA CENTRAL: Respostas nao respondem a pergunta do usuario
    |
    +-- [P3] user_query nao passada ao LLM (RAIZ PRIMARIA)
    |     |
    |     +-- build_prompt() nao aceita user_query
    |     +-- build_prompt_node() nao extrai user_query do state
    |
    +-- [P5] Dados reais nao passados ao LLM (RAIZ SECUNDARIA)
    |     |
    |     +-- Apenas numeric_summary chega ao prompt
    |     +-- DataFrame disponivel no state mas ignorado
    |
    +-- [P7] IntentEnricher desconectado (AMPLIFICADOR)
    |     |
    |     +-- Metadados semanticos calculados mas nao usados
    |     +-- narrative_angle perfeito desperdicado
    |
    +-- [P2] Calculadoras acopladas ao chart_type (DISTORCAO)
    |     |
    |     +-- Metricas estatisticas irrelevantes dominam o prompt
    |     +-- Framework de analise nao corresponde a pergunta
    |
    +-- [P4] Estrutura rigida de 4 secoes (INFLACAO)
    |     |
    |     +-- 5 insights obrigatorios geram conteudo de preenchimento
    |     +-- 3 recomendacoes genericas sem valor contextual
    |
    +-- [P8] Modelo flash-lite para resposta ao usuario (DEGRADACAO)
    |     |
    |     +-- Narrativas template-like sem profundidade
    |     +-- Incapaz de raciocinio complexo
    |
    +-- [P6] Formatter LLM redundante (COMPLEXIDADE)
          |
          +-- Dupla geracao de executive_summary e narrativa
          +-- Risco de inconsistencia entre camadas
```

---

## 4. Analise da Relacao Grafico-Narrativa

### Distorcoes Identificadas

**line_composed com serie unica (Query 4):**
O chart_type `line_composed` e projetado para multi-series (multiplas linhas no mesmo grafico). Quando usado com serie unica (historico mensal), o calculator ComposedCalculator gera metricas de "convergencia entre series" e "correlacao media" que sao matematicamente sem sentido para N=1. A LLM recebe `num_series: 1.00, spread_inicial: 0.00, correlacao_media: NaN` e e forcada a interpretar essas metricas vazias.

**line_composed para pergunta sobre quedas (Query 5):**
O chart_type correto seria possivelmente `bar_horizontal` (ranking de quedas) ou um `line_composed` orientado a variacao. O template `line_composed` forca analise de "dinamica competitiva entre series", enquanto o usuario quer um ranking de quedas. A classificacao grafica correta nao garante insight correto -- o insight precisa ser orientado pela **intencao**, nao pelo **grafico**.

**bar_vertical_stacked para cross-reference (Query 3):**
O stacked bar e adequado para visualizar composicao, mas os insights de "contribuicao dominante" e "score de balanceamento" nao respondem "quais produtos em quais clientes". O insight deveria ser tabular: cada cliente com seus top produtos.

**pie para distribuicao (Query 2):**
O indice HHI e relevante para economistas, nao para gestores comerciais. A resposta deveria listar os produtos por participacao percentual, nao calcular indices academicos de concentracao.

### Conclusao

A relacao entre tipo de grafico e narrativa e excessivamente acoplada. O `chart_type` deveria influenciar apenas a **visualizacao** (Plotly Generator), enquanto o **insight** deveria ser guiado pela **intencao do usuario** + **dados reais**. A separacao de responsabilidades esta incorreta: o chart_type esta controlando tanto a visualizacao quanto a narrativa.

---

## 5. Avaliacao Comparativa: flash-lite vs flash

### Teste com os 5 Outputs (todos gerados com flash-lite)

| Criterio | Performance Observada (flash-lite) | Expectativa (flash) |
|----------|-----------------------------------|---------------------|
| Aderencia a pergunta | 0/5 respostas diretas | Melhoria moderada (com P3 corrigido) |
| Profundidade semantica | Template-like, formulaico | Contextual, fluido |
| Raciocinio multi-step | Incapaz (ex: ranking de quedas) | Capaz com dados adequados |
| Qualidade narrativa | Repeticao de metricas sem interpretacao | Interpretacao com insight real |
| Adequacao ao publico | Linguagem tecnica/academica (HHI, CV) | Linguagem executiva adaptavel |

### Limitacoes Estruturais vs Limitacoes de Modelo

E fundamental distinguir:
- **Limitacoes estruturais (P1-P5, P7):** A LLM nao recebe informacao suficiente para responder bem. Nenhum modelo, por mais avancado, pode responder uma pergunta que nao recebeu.
- **Limitacoes de modelo (P8):** Dado o mesmo prompt, flash-lite produz respostas mais genericas que flash.

**A correcao dos problemas estruturais (P1-P5, P7) teria impacto MUITO MAIOR que a troca de modelo isoladamente.** Um flash-lite com prompt correto (user_query + dados reais + enriched_intent) superaria um flash com o prompt atual (apenas metricas pre-calculadas).

### Necessidade Futura de Modelos Mais Robustos

Mesmo com flash e prompt corrigido, existem cenarios que podem requerer modelos mais robustos (Gemini 2.5 Pro ou equivalente):

1. **Analise causal:** "Por que as vendas cairam?" requer raciocinio inferencial que flash pode nao sustentar.
2. **Comparacoes aninhadas:** "Top 5 produtos nos top 5 clientes vs mesmo periodo do ano anterior" requer manipulacao mental de 3 dimensoes simultaneas.
3. **Predicao com confianca:** "Qual a tendencia para os proximos meses?" requer capacidade de forecast que modelos menores nao tem.
4. **Sintese de multiplas fontes:** Combinar insights de diferentes chart_types em uma narrativa coerente.

**Recomendacao:** Implementar a refatoracao com `gemini-2.5-flash` como padrao. Monitorar qualidade e escalar para modelos superiores apenas quando evidencia empririca demonstrar limitacao.

---

## 6. Inventario de Problemas por Arquivo

| Arquivo | Problemas | Severidade |
|---------|-----------|------------|
| `src/insight_generator/formatters/prompt_builder.py` | P1, P3, P4, P5, P7 | CRITICA |
| `src/insight_generator/graph/nodes.py` | P1, P3, P5, P7 | CRITICA |
| `src/insight_generator/calculators/*` | P2 | ALTA |
| `src/insight_generator/models/insight_schemas.py` | P8 | MEDIA-ALTA |
| `src/insight_generator/core/intent_enricher.py` | P7 | ALTA |
| `src/insight_generator/core/settings.py` | P8 | MEDIA |
| `src/formatter_agent/agent.py` | P6 | MEDIA |
| `src/pipeline_orchestrator.py` | P6 | MEDIA |

---

## 7. Impacto na Experiencia do Usuario

### Estado Atual
O usuario faz uma pergunta simples e recebe um relatorio estatistico de 5 secoes que nao responde sua pergunta. Ele precisa ignorar o texto e ler a tabela de dados brutos para encontrar a resposta. O insight_generator atualmente **reduz** a utilidade da resposta em vez de **aumenta-la**.

### Estado Desejado
O usuario faz uma pergunta e recebe uma resposta direta, contextualizada, que interpreta os dados em funcao da intencao original. O texto complementa a visualizacao (Plotly) em vez de competir com ela.

### Gap
A distancia entre o estado atual e o desejado e grande, mas a maioria dos problemas e estrutural (como o prompt e construido), nao conceitual. A arquitetura do LangGraph, o IntentEnricher, e os dados disponveis no state sao suficientes -- eles apenas nao estao conectados corretamente.
