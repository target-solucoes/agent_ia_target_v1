# Samples de Qualidade - Insight Generator

**Data:** 2026-02-11
**Objetivo:** Benchmark manual de qualidade para validacao futura do insight_generator refatorado.
**Referencia:** `docs/insight_generator_diagnosis.md`, `docs/insight_generator_planning.md`

Cada sample demonstra o nivel de qualidade esperado para a resposta do insight_generator. As respostas sao escritas como se fossem geradas por uma LLM com prompt corrigido (user_query + dados + enriched_intent), representando o comportamento alvo pos-refatoracao.

---

## Convencoes

- **Pergunta:** Query original do usuario em portugues
- **Filtros ativos:** Filtros persistentes/explicitados no contexto
- **Chart type:** Tipo de grafico classificado pelo graphic_classifier
- **Dados:** Resumo dos dados retornados pelo analytics_executor (simulados com base no dataset real)
- **Resposta ideal:** O que o insight_generator DEVE retornar
- **Notas:** Observacoes sobre formato, modelo e criterios

---

## Sample 1: Ranking simples (bar_horizontal)

**Pergunta:** "Top 5 clientes por faturamento"
**Filtros ativos:** Nenhum
**Chart type:** `bar_horizontal`

**Dados:**

| Cod_Cliente | Valor_Vendido |
|-------------|---------------|
| 2855        | R$ 48.215.340 |
| 22494       | R$ 31.892.110 |
| 23709       | R$ 27.456.890 |
| 40461       | R$ 22.118.750 |
| 41410       | R$ 19.887.320 |

**Resposta ideal:**

> Os 5 maiores clientes por faturamento sao:
>
> 1. **Cliente 2855** - R$ 48,2M (lider com ampla vantagem)
> 2. **Cliente 22494** - R$ 31,9M
> 3. **Cliente 23709** - R$ 27,5M
> 4. **Cliente 40461** - R$ 22,1M
> 5. **Cliente 41410** - R$ 19,9M
>
> O cliente 2855 lidera com R$ 48,2M, representando 32% do total dos top 5 (R$ 149,6M). A diferenca entre o primeiro e o segundo colocado e de R$ 16,3M, a maior distancia entre posicoes consecutivas no ranking.

**Notas:**
- Formato lista: adequado para rankings
- Resposta direta na primeira frase
- Valores especificos com nomes
- Contexto adicional (concentracao, gap) em paragrafo complementar, nao como secao separada
- Alcancavel com flash-lite (ranking simples)

---

## Sample 2: Ranking com filtros contextuais (bar_horizontal)

**Pergunta:** "quais foram os maiores representantes de SC?"
**Filtros ativos:** UF_Cliente: SC
**Chart type:** `bar_horizontal`

**Dados:**

| Cod_Vendedor | Valor_Vendido |
|--------------|---------------|
| 000145       | R$ 21.220.758 |
| 000018       | R$ 20.586.795 |
| 000146       | R$ 15.967.630 |
| 000114       | R$ 10.586.136 |
| 000176       | R$ 9.832.415  |

**Resposta ideal:**

> Os maiores representantes em **Santa Catarina** por valor vendido sao:
>
> 1. **Vendedor 000145** - R$ 21,2M
> 2. **Vendedor 000018** - R$ 20,6M
> 3. **Vendedor 000146** - R$ 16,0M
> 4. **Vendedor 000114** - R$ 10,6M
> 5. **Vendedor 000176** - R$ 9,8M
>
> Os dois lideres (000145 e 000018) estao muito proximos, com diferenca de apenas R$ 634K (3%). Juntos, representam 38% do faturamento total dos representantes de SC. A partir do terceiro colocado (000146), ha uma queda de 22% no volume.

**Notas:**
- Filtro "SC" mencionado naturalmente como "Santa Catarina"
- Destaque para a proximidade entre 1o e 2o (insight real e util)
- Sem HHI, sem "concentracao extrema", sem "cauda longa"
- Alcancavel com flash-lite

---

## Sample 3: Distribuicao (pie)

**Pergunta:** "qual a distribuicao dos produtos do representante 000145?"
**Filtros ativos:** UF_Cliente: SC, Cod_Vendedor: 000145
**Chart type:** `pie`

**Dados:**

| Des_Linha_Produto            | Qtd_Vendida | Participacao |
|------------------------------|-------------|-------------|
| CONEXOES SOLDAVEIS           | 3.473.773   | 31,5%       |
| TUBOS ELETRODUTO CORRUGADO   | 1.399.350   | 12,7%       |
| CONEXOES ESGOTO PRIMARIO     | 1.310.456   | 11,9%       |
| TUBOS PVC SOLDAVEL           | 1.089.230   | 9,9%        |
| CONEXOES ROSCAVEIS           | 987.654     | 8,9%        |
| CONEXOES ESGOTO SECUNDARIO   | 876.543     | 7,9%        |
| CAIXAS DE LUZ                | 654.321     | 5,9%        |
| Outros (4 categorias)        | 1.252.855   | 11,3%       |

**Resposta ideal:**

> A distribuicao de produtos do representante 000145 em **SC** mostra que **Conexoes Soldaveis** domina com 31,5% do volume (3,47M unidades), quase o triplo do segundo colocado.
>
> | Produto | Qtd Vendida | Participacao |
> |---------|-------------|-------------|
> | Conexoes Soldaveis | 3,47M | 31,5% |
> | Tubos Eletroduto Corrugado | 1,40M | 12,7% |
> | Conexoes Esgoto Primario | 1,31M | 11,9% |
> | Tubos PVC Soldavel | 1,09M | 9,9% |
> | Conexoes Roscaveis | 988K | 8,9% |
>
> Os 3 principais produtos concentram 56,1% do volume total. As 4 categorias menores (agrupadas em "Outros") representam 11,3%.

**Notas:**
- Formato hibrido: texto introdutorio + tabela
- Ideal para distribuicoes (resposta visual)
- Sem HHI, sem indice de diversidade de Simpson
- Filtros mencionados naturalmente
- Alcancavel com flash-lite (dados claros, pergunta simples)

---

## Sample 4: Tendencia temporal serie unica (line)

**Pergunta:** "qual o historico de compras de 2015?"
**Filtros ativos:** UF_Cliente: SC, Cod_Vendedor: 000145, Ano: 2015
**Chart type:** `line` (ou `line_composed` com serie unica)

**Dados:**

| Mes | Valor_Vendido |
|-----|---------------|
| Jan | R$ 633.249    |
| Fev | R$ 1.053.361  |
| Mar | R$ 2.438.682  |
| Abr | R$ 1.238.411  |
| Mai | R$ 974.582    |
| Jun | R$ 606.921    |
| Jul | R$ 845.591    |
| Ago | R$ 1.076.049  |
| Set | R$ 1.328.773  |
| Out | R$ 2.388.519  |
| Nov | R$ 639.065    |
| Dez | R$ 430.074    |

**Resposta ideal:**

> O historico de compras do representante 000145 em **SC** durante **2015** mostra um padrao com dois picos e fechamento em queda:
>
> - **Marco** foi o melhor mes, com R$ 2,44M -- quase 4x o valor de janeiro (R$ 633K).
> - **Outubro** teve o segundo pico, com R$ 2,39M, mostrando recuperacao apos um vale no meio do ano.
> - **Dezembro** fechou o ano no menor nivel: R$ 430K, representando queda de 82% em relacao ao pico de marco.
>
> O total do ano foi de **R$ 13,65M**, com media mensal de R$ 1,14M. O padrao sugere sazonalidade, com picos no 1o e 4o trimestres e vale no 2o trimestre (junho: R$ 607K).

**Notas:**
- Descreve o padrao temporal, nao apenas variacao jan-dez
- Identifica picos e vales com valores e meses especificos
- Interpreta o padrao (sazonalidade)
- Fornece total e media como contexto
- Requer flash para interpretar padrao sazonal (flash-lite tenderia a focar so na variacao geral)

---

## Sample 5: Comparacao temporal multi-series - polaridade negativa (line_composed)

**Pergunta:** "Quais clientes que tiveram a maior queda entre fev/2015 e marco/2015?"
**Filtros ativos:** UF_Cliente: SC, Cod_Vendedor: 000145, Data: fev-mar/2015
**Chart type:** `line_composed`

**Dados:**

| Cod_Cliente | Fev/2015 | Mar/2015 | Variacao  | Variacao % |
|-------------|----------|----------|-----------|------------|
| 20524       | 5.401    | 0        | -5.401    | -100,0%    |
| 33777       | 1.695    | 0        | -1.695    | -100,0%    |
| 17761       | 4.129    | 430      | -3.699    | -89,6%     |
| 14061       | 7.115    | 3.566    | -3.549    | -49,9%     |
| 2330        | 8.085    | 4.098    | -3.987    | -49,3%     |

**Resposta ideal:**

> Os clientes com maior queda em quantidade vendida entre **fevereiro e marco de 2015** (vendedor 000145, SC) foram:
>
> | Cliente | Fev/2015 | Mar/2015 | Queda | Queda % |
> |---------|----------|----------|-------|---------|
> | **20524** | 5.401 | 0 | -5.401 | **-100%** |
> | **33777** | 1.695 | 0 | -1.695 | **-100%** |
> | 17761 | 4.129 | 430 | -3.699 | -89,6% |
> | 14061 | 7.115 | 3.566 | -3.549 | -49,9% |
> | 2330 | 8.085 | 4.098 | -3.987 | -49,3% |
>
> Dois clientes (20524 e 33777) zeraram completamente as compras em marco, representando perda total. O cliente 17761 tambem teve queda drastica de 89,6%. Todos os 5 clientes analisados apresentaram queda, indicando uma retracao generalizada no periodo.

**Notas:**
- Formato tabular: ideal para comparacao periodo a periodo
- Resposta direta: identifica os clientes com maior queda
- Destaca os clientes que zeraram (insight critico)
- Polaridade negativa refletida no tom (retracao, perda)
- Nao fala de "convergencia", "correlacao", ou "spread"
- Requer flash para raciocinio comparativo e ranking de quedas

---

## Sample 6: Composicao/cross-tab (bar_vertical_stacked)

**Pergunta:** "quais os 5 produtos mais vendidos nos 5 maiores clientes desse mesmo representante?"
**Filtros ativos:** UF_Cliente: SC, Cod_Vendedor: 000145
**Chart type:** `bar_vertical_stacked`

**Dados:** (cross-tab Clientes x Produtos, Qtd_Vendida)

| Cliente | Conexoes Soldaveis | Tubos Eletroduto | Conex. Esgoto Prim. | Conex. Roscaveis | Conex. Esgoto Sec. | Total |
|---------|-------------------|-----------------|--------------------|-----------------|--------------------|-------|
| 2855    | 2.190.981 | 273.725 | 808.452 | 393.496 | 345.075 | 4.011.729 |
| 23709   | 64.223 | 859.350 | 50.110 | - | - | 1.200.349 |
| 22494   | 375.339 | 63.475 | 146.454 | 83.324 | 64.080 | 732.672 |
| 40461   | 164.984 | 151.125 | 148.006 | - | 51.485 | 585.152 |
| 41410   | 79.423 | 63.250 | 83.298 | - | 22.070 | 310.149 |

**Resposta ideal:**

> Os 5 produtos mais vendidos nos 5 maiores clientes do representante 000145 em **SC**:
>
> **Cliente 2855** (maior, 4,01M un. total):
> - Conexoes Soldaveis: 2,19M un. (produto dominante)
> - Conexoes Esgoto Primario: 808K un.
> - Conexoes Roscaveis: 393K un.
>
> **Cliente 23709** (1,20M un. total):
> - Tubos Eletroduto Corrugado: 859K un. (85% do volume deste cliente - perfil diferenciado)
>
> **Cliente 22494** (733K un. total):
> - Conexoes Soldaveis: 375K un. (mix mais diversificado)
>
> **Cliente 40461** (585K un.) e **Cliente 41410** (310K un.): volumes menores, com Conexoes Soldaveis e Tubos Eletroduto como principais.
>
> **Conexoes Soldaveis** e o produto dominante em 4 dos 5 clientes, exceto no cliente 23709, onde **Tubos Eletroduto Corrugado** domina com 85% do volume. Essa diferenca de perfil de compra do cliente 23709 merece atencao.

**Notas:**
- Formato agrupado por cliente (responde "quais produtos em quais clientes")
- Identifica padrao cross-tab: produto dominante varia por cliente
- Destaca anomalia (cliente 23709 com perfil diferente)
- Nao fala de "score de balanceamento", "desvio padrao dos stacks", "contribuicao dominante %"
- Requer flash para estruturar cross-tab e identificar padroes

---

## Sample 7: Comparacao entre categorias (bar_vertical)

**Pergunta:** "vendas por estado"
**Filtros ativos:** Nenhum
**Chart type:** `bar_vertical`

**Dados:**

| UF_Cliente | Valor_Vendido |
|------------|---------------|
| SP         | R$ 89.234.560 |
| SC         | R$ 67.891.230 |
| PR         | R$ 54.321.780 |
| MG         | R$ 43.210.890 |
| RS         | R$ 38.765.430 |

**Resposta ideal:**

> As vendas por estado mostram **Sao Paulo** na lideranca com R$ 89,2M, seguido por **Santa Catarina** (R$ 67,9M) e **Parana** (R$ 54,3M).
>
> | Estado | Valor Vendido | Participacao |
> |--------|---------------|-------------|
> | SP | R$ 89,2M | 30,4% |
> | SC | R$ 67,9M | 23,2% |
> | PR | R$ 54,3M | 18,5% |
> | MG | R$ 43,2M | 14,7% |
> | RS | R$ 38,8M | 13,2% |
>
> A regiao Sul (SC + PR + RS) soma R$ 161,0M, superando SP isoladamente. SP e SC juntos representam 53,6% do total.

**Notas:**
- Formato hibrido (texto + tabela)
- Adiciona participacao % como valor derivado
- Insight contextual: Sul vs SP (comparacao regional util)
- Alcancavel com flash-lite (comparacao simples)

---

## Sample 8: Pergunta comparativa com polaridade negativa (bar_horizontal)

**Pergunta:** "quais produtos tiveram queda em 2016 comparado a 2015?"
**Filtros ativos:** Nenhum (anos na query)
**Chart type:** `bar_horizontal` (ranking de variacoes negativas)

**Dados:**

| Des_Linha_Produto | 2015 | 2016 | Variacao % |
|-------------------|------|------|------------|
| Tubos PVC Soldavel | R$ 12,3M | R$ 8,1M | -34,1% |
| Conexoes Roscaveis | R$ 9,8M | R$ 7,2M | -26,5% |
| Caixas de Luz | R$ 5,4M | R$ 4,5M | -16,7% |

**Resposta ideal:**

> Tres linhas de produto apresentaram queda entre **2015 e 2016**:
>
> | Produto | 2015 | 2016 | Queda |
> |---------|------|------|-------|
> | **Tubos PVC Soldavel** | R$ 12,3M | R$ 8,1M | **-34,1%** (maior queda) |
> | Conexoes Roscaveis | R$ 9,8M | R$ 7,2M | -26,5% |
> | Caixas de Luz | R$ 5,4M | R$ 4,5M | -16,7% |
>
> **Tubos PVC Soldavel** teve a maior queda absoluta (R$ 4,2M) e percentual (-34,1%). A perda total das tres linhas soma R$ 7,7M, equivalente a reduzir o faturamento em aproximadamente 8% em relacao ao total de 2015.

**Notas:**
- Filtro temporal na query, nao como filtro persistente
- Foco exclusivo em quedas (polaridade negativa)
- Destaca maior queda absoluta e percentual
- Contextualiza impacto total
- Requer flash para raciocinio comparativo period-over-period

---

## Sample 9: Pergunta estrategica/percentual (pie)

**Pergunta:** "qual a participacao de SP nas vendas totais?"
**Filtros ativos:** Nenhum
**Chart type:** `pie` (ou metrica unica)

**Dados:**

| UF_Cliente | Valor_Vendido | Participacao |
|------------|---------------|-------------|
| SP         | R$ 89.234.560 | 30,4%       |
| Total      | R$ 293.423.890| 100%        |

**Resposta ideal:**

> **Sao Paulo** responde por **30,4%** do faturamento total, com R$ 89,2M de um total geral de R$ 293,4M. E o estado com maior participacao individual, mas nao representa maioria absoluta -- os demais estados somam 69,6% do total.

**Notas:**
- Resposta concisa e direta (3 linhas)
- Sem tabela (pergunta de metrica unica)
- Contextualiza: maior, mas nao maioria
- Sem secoes, sem recomendacoes, sem next_steps
- Alcancavel com flash-lite

---

## Sample 10: Tendencia com sazonalidade (line)

**Pergunta:** "como foram as vendas mes a mes nos ultimos 3 anos?"
**Filtros ativos:** Nenhum (3 ultimos anos inferidos)
**Chart type:** `line` (ou `line_composed` multi-ano)

**Dados (resumidos):**

| Ano | Melhor Mes | Valor | Pior Mes | Valor | Media Mensal | Total Ano |
|-----|-----------|-------|----------|-------|-------------|-----------|
| 2014 | Mar | R$ 28,5M | Dez | R$ 11,2M | R$ 19,8M | R$ 237,6M |
| 2015 | Mar | R$ 26,1M | Dez | R$ 9,8M  | R$ 18,2M | R$ 218,4M |
| 2016 | Out | R$ 24,3M | Jun | R$ 10,1M | R$ 17,5M | R$ 210,0M |

**Resposta ideal:**

> As vendas mensais nos ultimos 3 anos (2014-2016) mostram tendencia de leve queda anual, com padrao sazonal consistente:
>
> - **2014:** Media mensal de R$ 19,8M, total de R$ 237,6M. Pico em marco (R$ 28,5M).
> - **2015:** Media mensal de R$ 18,2M (-8% vs 2014), total de R$ 218,4M. Pico em marco (R$ 26,1M).
> - **2016:** Media mensal de R$ 17,5M (-4% vs 2015), total de R$ 210,0M. Pico deslocou para outubro (R$ 24,3M).
>
> O padrao sazonal se repete: os melhores meses sao marco e outubro (1o e 4o trimestres), enquanto junho e dezembro sao consistentemente os mais fracos. Em 2016, o pico principal migrou de marco para outubro, o que pode indicar mudanca no ciclo de compras.

**Notas:**
- Formato lista por ano: organiza 36 meses de dados de forma legivel
- Identifica tendencia anual + padrao sazonal
- Destaca anomalia (deslocamento do pico em 2016)
- Requer flash para: raciocinio multi-ano, deteccao de sazonalidade, identificacao de anomalia

---

## Sample 11: Pergunta contextualizada com filtro persistente

**Pergunta:** "e para o representante 000018?"
**Filtros ativos:** UF_Cliente: SC (persistente da conversa anterior)
**Chart type:** `bar_horizontal` (ranking de produtos, inferido do contexto)

**Dados:**

| Des_Linha_Produto | Qtd_Vendida |
|-------------------|-------------|
| CONEXOES SOLDAVEIS | 2.891.456 |
| TUBOS PVC SOLDAVEL | 1.234.567 |
| CONEXOES ESGOTO PRIMARIO | 987.654 |

**Resposta ideal:**

> Os principais produtos do representante **000018** em **SC**:
>
> 1. **Conexoes Soldaveis** - 2,89M unidades
> 2. **Tubos PVC Soldavel** - 1,23M unidades
> 3. **Conexoes Esgoto Primario** - 988K unidades
>
> Comparado ao representante 000145 (consulta anterior), o 000018 tambem tem Conexoes Soldaveis como produto dominante, mas com menor concentracao (nao informacao suficiente para comparacao detalhada neste contexto).

**Notas:**
- Filtro persistente (SC) mencionado naturalmente
- Referencia a conversa anterior (continuidade)
- Nao repete toda a estrutura da pergunta anterior
- Alcancavel com flash-lite (ranking simples)
- O insight comparativo (vs 000145) e um bonus se o modelo tiver capacidade

---

## Sample 12: Pergunta de metrica unica (possivel non_graph)

**Pergunta:** "qual o total de vendas?"
**Filtros ativos:** UF_Cliente: SC
**Chart type:** Nenhum (metrica unica) ou `single_metric`

**Dados:**

| Metrica | Valor |
|---------|-------|
| Total Valor_Vendido | R$ 109.711.696 |

**Resposta ideal:**

> O total de vendas em **Santa Catarina** e de **R$ 109,7M**.

**Notas:**
- Resposta minima: 1 frase
- Sem tabela, sem lista, sem secoes
- Filtro contextualizado
- Este tipo de query pode/deveria ser roteado para non_graph_executor
- Se chegar ao insight_generator, a resposta deve ser igualmente concisa
- Alcancavel com flash-lite

---

## Sample 13: Pergunta sobre tendencia com filtro temporal explicito (line)

**Pergunta:** "como foram as vendas do representante 000145 entre janeiro e junho de 2015?"
**Filtros ativos:** UF_Cliente: SC, Cod_Vendedor: 000145, Data: jan-jun/2015
**Chart type:** `line`

**Dados:**

| Mes | Valor_Vendido |
|-----|---------------|
| Jan | R$ 633.249    |
| Fev | R$ 1.053.361  |
| Mar | R$ 2.438.682  |
| Abr | R$ 1.238.411  |
| Mai | R$ 974.582    |
| Jun | R$ 606.921    |

**Resposta ideal:**

> As vendas do representante 000145 em **SC** entre **janeiro e junho de 2015** mostram um pico acentuado em marco seguido de queda progressiva:
>
> - Jan: R$ 633K (inicio do periodo)
> - Fev: R$ 1,05M (+66%)
> - **Mar: R$ 2,44M** (pico do semestre, +131% vs fevereiro)
> - Abr: R$ 1,24M (queda de 49% apos pico)
> - Mai: R$ 975K
> - Jun: R$ 607K (menor valor, -75% vs pico)
>
> O total do semestre foi de **R$ 6,94M**, com 35% concentrado em marco. Apos o pico, cada mes subsequente teve queda consistente, sugerindo que marco pode ter concentrado demanda antecipada ou compras de estoque.

**Notas:**
- Descreve evolucao mes a mes (nao apenas inicio vs fim)
- Destaca pico com contexto de variacao
- Interpreta o padrao (demanda concentrada)
- Total e concentracao como contexto adicional
- Requer flash para interpretacao de padrao e sugestao de causa

---

## Sample 14: Pergunta analitica complexa - composicao com temporalidade

**Pergunta:** "como os 3 maiores produtos evoluiram de 2014 para 2016?"
**Filtros ativos:** Nenhum
**Chart type:** `line_composed` (3 series temporais por ano)

**Dados (resumidos):**

| Produto | 2014 | 2015 | 2016 | Variacao 2014-2016 |
|---------|------|------|------|--------------------|
| Conexoes Soldaveis | R$ 45,2M | R$ 42,8M | R$ 40,1M | -11,3% |
| Tubos Eletroduto | R$ 28,7M | R$ 30,1M | R$ 31,5M | +9,8% |
| Conex. Esgoto Prim. | R$ 22,1M | R$ 21,5M | R$ 19,8M | -10,4% |

**Resposta ideal:**

> A evolucao dos 3 maiores produtos entre **2014 e 2016** mostra trajetorias divergentes:
>
> | Produto | 2014 | 2015 | 2016 | Variacao |
> |---------|------|------|------|----------|
> | Conexoes Soldaveis | R$ 45,2M | R$ 42,8M | R$ 40,1M | **-11,3%** |
> | Tubos Eletroduto | R$ 28,7M | R$ 30,1M | R$ 31,5M | **+9,8%** |
> | Conex. Esgoto Prim. | R$ 22,1M | R$ 21,5M | R$ 19,8M | **-10,4%** |
>
> **Conexoes Soldaveis** permanece lider mas perde forca progressivamente (-11,3% no periodo). Em contraste, **Tubos Eletroduto Corrugado** e a unica linha em crescimento (+9,8%), ganhando participacao relativa. Se a tendencia continuar, pode se aproximar do segundo lugar em 2017.
>
> **Conexoes Esgoto Primario** acompanha a queda do lider (-10,4%), indicando possivel correlacao de mercado entre as duas linhas.

**Notas:**
- Formato hibrido: tabela + analise interpretativa
- Identifica divergencia entre series (insight real)
- Projeta tendencia futura (valor agregado)
- Sugere correlacao entre linhas (raciocinio analitico)
- Requer flash para: deteccao de divergencia, projecao, correlacao
- Flash-lite provavelmente descreveria numeros sem interpretar padroes

---

## Criterios de Qualidade por Modelo

### flash-lite (gemini-2.5-flash-lite) - Expectativa realista

- Capaz de: listar dados, formatar tabelas, calcular variacoes simples
- Limitacoes: narrativas template-like, dificuldade com interpretacao de padroes, recomendacoes genericas
- Samples alcancaveis com qualidade aceitavel: 1, 2, 3, 7, 9, 11, 12
- Samples com qualidade degradada: 4, 5, 6, 8, 10, 13, 14

### flash (gemini-2.5-flash) - Expectativa alvo

- Capaz de: interpretacao contextual, deteccao de padroes, raciocinio comparativo, narrativas fluidas
- Limitacoes: raciocinio causal profundo, predicoes com confianca estatistica
- Samples alcancaveis com qualidade alvo: 1-14 (todos)
- Samples onde pode haver limitacao: 14 (correlacao entre linhas pode ser superficial)

### Cenarios que exigiriam Gemini 2.5 Pro (futuro)

- "Por que as vendas cairam em 2016?" (analise causal)
- "Qual a previsao para 2017 com base na tendencia?" (predicao com confianca)
- "Quais fatores explicam a diferenca de performance entre SP e SC?" (analise multivariada)
- "Compare a sazonalidade de cada produto nos ultimos 5 anos" (raciocinio sobre padroes complexos)

---

## Resumo

Estes 14 samples cobrem:
- 7 chart types: bar_horizontal, bar_vertical, bar_vertical_stacked, line, line_composed, pie, single_metric
- 3 polaridades: positiva (top, maiores), negativa (queda, pior), neutra (historico, distribuicao)
- 4 focos temporais: periodo unico, periodo vs periodo, serie temporal, sazonalidade
- Filtros: sem filtros, filtro explicito, filtro persistente, filtro temporal na query
- Complexidade: simples (1 metrica), media (ranking com contexto), alta (cross-tab, multi-serie temporal)

Cada sample prioriza: **resposta direta -> dados especificos -> interpretacao contextual**, eliminando ruido (secoes forcadas, metricas academicas, recomendacoes genericas).
