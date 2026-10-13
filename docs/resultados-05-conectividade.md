# Resultados 05 — A informação que separa rio de lago está dentro do tile?

**Data:** 12/10/2026 · **Dataset:** RiverScope (teste: 235 imagens; validação: 123) · **Sem treino novo** — análise das predições salvas

## Pergunta

Os modelos de rio marcam outros corpos d'água (rótulo 2) como rio, e nem o 3 classes nem o DeepLabV3+ resolvem
(`resultados-04`). Uma regra simples de conectividade, aplicada às predições, consegue separar rio de lago?

- Se sim: a informação está no tile (forma/conectividade), mas a rede não a usa.
- Se não: é preciso entender **onde** estão os erros que sobram.

## Método

| Item | Valor |
|---|---|
| Predições | Rio binário: {Unet, FPN} × {resnet50, efficientnet-b2} × seeds 42/20/83 (**12**) + DeepLabV3+ effnet-b2 × 3 seeds (**3**); dice, sem DA |
| Manchas | Componentes conectados (vizinhança 8) da predição de rio, por tile, na área original (sem *padding*) |
| **Borda** | Mantém as manchas que tocam a borda do tile (um rio atravessa o tile; um lago isolado, não). Sem parâmetro |
| **Maior** | Mantém só a maior mancha. Sem parâmetro |
| **Tamanho** (controle) | Mantém manchas com ≥ N pixels; N escolhido **na validação** (por execução) e aplicado ao teste |
| **Oráculo por mancha** | Remove as manchas que são majoritariamente rótulo 2 (usa o rótulo: limite superior de qualquer regra por mancha) |
| **Teto por pixel** | Remove todo FP em rótulo 2 (= contrafactual de A2) |
| Métricas | F1, precisão, recall do rio; lago → rio; % dos FP em lagos removidos; **% da lacuna fechada** = (F1 − F1 base) / (teto − F1 base); % do rio corretamente previsto que a regra apaga; tiles em que o F1 piora > 0,01 |
| Scripts | `analysis/connectivity_postproc.py` → `analysis/out/connectivity_{runs.csv,summary.txt}`; `analysis/lake_fp_by_reach_type.py` → `analysis/out/lake_fp_by_reach_type.txt` |
| Conferência | Sem regra, os números batem com `compare_formulations.py` (ex.: Unet + resnet50, seed 42: F1 0,9151, lago → rio 44,9%, teto 0,941) |

## 1. Pelos rótulos: o que uma regra de borda poderia fazer

| | Teste | Validação |
|---|---:|---:|
| Área de outra água que **toca a borda** do tile | **56.9%** | 78.4% |
| Área de rio que **não** toca a borda (risco da regra) | 8.9% | 0.6% |
| Área de outra água encostada no rio (dilatação de 2 px) | 1.4% | — |

Os corpos d'água quase nunca encostam no rio, mas **mais da metade da sua área é cortada pela borda do tile** —
para a regra da borda, eles parecem rios atravessando o tile.

## 2. Resultado: as regras simples não funcionam

Teste, Unet/FPN, média ± desvio sobre 12 execuções:

| Regra | F1 | Precisão | Recall | Lago → rio | FP em lagos removidos | Lacuna fechada | Rio apagado | Tiles piores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nenhuma (base) | 0.915 ± 0.009 | 0.899 | 0.931 | 47.9% | — | — | — | — |
| Borda | 0.913 ± 0.010 | 0.906 | 0.921 | 43.5% | 8.1% | **−5%** | 1.0% | 16 |
| Maior | 0.919 ± 0.007 | 0.926 | 0.913 | 27.8% | 41.2% | 14% | 1.9% | **34** |
| Tamanho (N da validação) | 0.915 ± 0.008 | 0.904 | 0.928 | 45.2% | 4.5% | 3% | 0.3% | 11 |
| **Oráculo por mancha** | **0.943 ± 0.007** | 0.955 | 0.931 | 1.2% | **97.1%** | **103%** | 0.0% | 0 |
| Teto por pixel | 0.942 ± 0.008 | 0.953 | 0.931 | 0.0% | 100% | 100% | 0.0% | 0 |

N escolhido na validação variou de 0 a 10 000 px entre execuções (instável).
DeepLabV3+ (3 execuções) segue o mesmo padrão: borda +1% da lacuna, maior 15% (35 tiles piores), oráculo 105%.
Na validação, nenhuma regra melhora o F1 (borda −90% da lacuna; maior −19%).

**Leitura:**
- **Borda** não funciona: remove só 8% dos FP em lagos e piora o F1, porque apaga rio verdadeiro (canais e
  meandros cortados) e mantém os lagos que tocam a borda.
- **Maior** remove 41% dos FP em lagos, mas apaga rio verdadeiro (recall −0,018) e piora 34 tiles; ganho líquido pequeno
  (+0,004 de F1, 14% da lacuna).
- **Tamanho** praticamente não faz nada: os FP em lagos não são manchas pequenas.

## 3. Mas o erro é "por mancha inteira"

- O **oráculo por mancha fecha ~100% da lacuna** (97% dos FP em lagos removidos, nenhum rio apagado).
  Ou seja: os FP em lagos estão em **manchas separadas do rio previsto** — o modelo desenha corretamente a forma
  do corpo d'água, mas **dá a identidade errada à mancha inteira**. Não é erro de borda entre rio e lago.
- Depois da regra da borda, os FP em lagos que sobram estão **99% em manchas isoladas do rio que tocam a borda**
  e só 1% em manchas grudadas no rio (validação: 78% / 22%).

**Consequência:** decidir "esta mancha é rio ou não" resolveria quase todo o problema, mas a geometria dentro do
tile não basta para decidir — a maior parte dessas manchas é cortada pela borda, como um rio seria. A informação que
falta é **externa ao tile**: onde está o curso do rio (rede SWORD, contexto maior). Isso é consistente com o
processo de anotação: os anotadores usaram vista ampliada, linhas do SWORD e Google Maps (RiverScope, §3.2).

## 4. Onde estão esses erros: tipo do trecho no SWORD

O último dígito do `reach_id` do SWORD é o tipo do trecho (1 rio, 3 lago no curso do rio, 5 topologia não confiável).

| Tipo do trecho (teste) | Tiles | Trechos | Rio | Outra água | % de toda a outra água do teste |
|---|---:|---:|---:|---:|---:|
| 1 — rio | 211 | 39 | 17.4% | 1.6% | 61.0% |
| 3 — lago no curso do rio | 12 | 5 | 23.5% | 7.3% | 16.0% |
| 5 — topologia não confiável | 12 | 4 | 45.8% | 10.5% | 23.0% |

- **Os 24 tiles de trechos tipo 3 e 5 (10% dos tiles) têm 39% da outra água** e respondem, em média, por
  **46% dos FP em lagos** (tipo 5: 33 ± 15%; tipo 3: 14 ± 4%).
- Exemplos (Unet + effnet-b2, seed 42; `analysis/out/top_lake_fp_tiles.png`): o pior tile (45% dos FP em lagos
  dessa execução) é **inteiramente água turva rotulada como outra água**, de um trecho tipo 5; outros são canais
  largos rotulados como outra água e lagos atravessados pelo rio (tipo 3). **Nesses casos, "rio vs. lago" é
  ambíguo mesmo para um humano olhando só o tile.**

### Os achados principais resistem nos trechos de rio comuns (tipo 1, 211 tiles)

| Medida (12 execuções) | Todos os tiles | Só tipo 1 |
|---|---:|---:|
| FP do rio que são outra água | 56.6 ± 8.8% | **46.3 ± 7.3%** (mín. 32%) |
| Outra água prevista como rio | 47.9 ± 12.7% | 41.6 ± 13.4% |
| Custo em F1 (contrafactual) | +0.027 ± 0.007 | **+0.018 ± 0.005** (0.011–0.027) |

Nos trechos comuns, a outra água é ~2% dos pixels não-rio e ainda concentra ~46% dos FP (~24× o esperado pela
área). **O fenômeno é real; parte da sua magnitude vem de poucos trechos ambíguos.**

## 5. Verificação do A4 só nos trechos de rio comuns (tipo 1)

3 classes − rio binário, mesma config e seed (12 pares), 211 tiles de teste de trechos tipo 1
(`analysis/robustness_checks.py` → `analysis/out/robustness_checks.txt`):

| Métrica | Rio | 3 classes | Diferença média | Mediana | 3 classes maior | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| IoU do rio | 0.883 ± 0.020 | 0.880 ± 0.016 | −0.003 | −0.008 | 5/12 | 0.34 |
| Precisão | 0.920 | 0.924 | +0.004 | −0.004 | 5/12 | 1.00 |
| Recall | 0.957 | 0.949 | −0.008 | −0.008 | 3/12 | 0.06 |
| Lago → rio | 41.6% | 35.7% | −5.9 pp | −3.1 pp | 5/12 | 0.23 |
| FP em lagos | 46.3% | 41.7% | −4.6 pp | −4.4 pp | 4/12 | 0.11 |

**Leitura:** nos trechos comuns aparece uma **tendência pequena e não significativa** de o 3 classes confundir menos
lago com rio (−3 a −6 pp), que não aparecia com todos os tiles; ela é **compensada por uma perda de recall** do rio
(−0,008, 9/12 pares), e o IoU fica igual. A conclusão se mantém: **sem ganho líquido**. Formular como "não reduz de
forma significativa", e não como "não muda nada".

## 6. Verificação do A5: o recall perdido na validação está nos rios muito largos?

Largura de cada tile: `p_width` (largura do SWORD) mediana dos nós do trecho (`SWOT/nodes/*.geojson`). Recall dos
pixels de rio, 12 execuções por modelo:

| Validação | Tiles | Trechos | % do rio | Recall — modelo de rio | Recall — modelo de água | % do rio perdido (modelo de rio) |
|---|---:|---:|---:|---:|---:|---:|
| < 100 m | 34 | 7 | 7.3% | 0.799 ± 0.044 | 0.826 | 9.4% |
| 100–300 m | 35 | 6 | 4.3% | 0.929 ± 0.009 | 0.926 | 2.1% |
| 300–1000 m | 18 | 5 | 12.1% | 0.936 ± 0.020 | 0.942 | 5.1% |
| **≥ 1000 m** | 36 | 5 | **76.3%** | **0.801 ± 0.072** | **0.962** | **83.5%** |

| Teste | Tiles | Trechos | % do rio | Recall — rio | Recall — água |
|---|---:|---:|---:|---:|---:|
| < 100 m | 45 | 16 | 6.7% | 0.906 | 0.899 |
| 100–300 m | 149 | 19 | 45.9% | 0.922 | 0.929 |
| 300–1000 m | 27 | 8 | 27.1% | 0.923 | 0.952 |
| ≥ 1000 m | 14 | 5 | 20.3% | **0.970** | 0.996 |

Por trecho (≥ 1000 m), recall do modelo de rio / de água:

| Conjunto | Trecho (tipo) | Largura SWORD | Tiles | Tiles > 95% rio | Rio | Água |
|---|---|---:|---:|---:|---:|---:|
| Validação | 32215000241 (1) | 2150 m | 11 | 55% | **0.680** | 0.939 |
| Validação | 72510001005 (5) | 1701 m | 19 | 5% | **0.853** | 0.971 |
| Validação | 32259000021 (1) | 1218 m | 3 | 0% | 0.962 | 0.994 |
| Validação | 53155000045 (5) | 2192 m | 1 | 100% | 0.995 | 0.996 |
| Teste | 34217900381 (1) | 1044 m | 7 | 71% | 0.996 | 0.997 |
| Teste | 22733000093 (3) | 1894 m | 1 | 100% | 0.791 | 0.994 |

**Leitura:**
- **Confirmado:** na validação, **83,5% do rio perdido está em trechos com ≥ 1 km de largura** (mais largos que boa
  parte do tile de ~1,5 km), onde o modelo de rio acha 80% do rio e o de **água acha 96%** — o modelo vê a água,
  mas não a chama de rio. Os dois trechos do A5 têm 1,7 e 2,15 km.
- **Mas largura não basta como explicação:** outros trechos igualmente largos, inclusive com tiles inteiramente de
  rio, são bem segmentados (teste 34217900381: 0,996; validação 53155000045: 0,995). Com só ~5 trechos largos por
  conjunto, não dá para separar "rio largo sem margem" de "aparência específica desses trechos".
- **Rios estreitos (< 100 m) também são difíceis**, mas para os **dois** modelos (validação 0,80 vs. 0,83) — é
  limite de resolução (3 m/pixel), não confusão rio–lago. Questão diferente, para a análise por largura (C4).

## Conclusões

1. **Regras geométricas simples (borda, maior mancha, tamanho) não resolvem** a confusão rio–lago: fecham no máximo
   ~14% da lacuna e apagam rio verdadeiro.
2. **O erro é de identidade da mancha, não de contorno:** um oráculo que decide "rio ou não" por mancha recupera
   ~100% da perda. O modelo segmenta bem a água; erra *qual* água é rio.
3. **A informação para essa decisão está, em grande parte, fora do tile:** 57% da área de outros corpos d'água toca
   a borda do tile, e 99% dos FP que resistem à regra da borda estão nessas manchas. Motiva usar a rede SWORD ou
   contexto maior — o mesmo contexto que os anotadores tiveram.
4. **Parte do problema é ambiguidade dos rótulos:** trechos SWORD de lago no rio e de topologia não confiável
   (10% dos tiles) concentram ~46% dos FP em lagos. Os achados A2–A4 continuam valendo nos trechos de rio comuns,
   com custo menor (+0,018 de F1).

5. **A4 nos trechos comuns:** o 3 classes mostra uma tendência pequena e não significativa de menos confusão
   (−3 a −6 pp), compensada por perda de recall; sem ganho líquido.
6. **A5:** o recall perdido na validação está nos trechos ≥ 1 km (83,5%), que o modelo de água acha; mas outros
   trechos igualmente largos são bem segmentados — largura é um fator, não a explicação completa.

## Limitações

- Os tiles são recortados ao longo do rio — isso favorece a regra da borda; mesmo assim ela não funciona.
- O oráculo usa os rótulos: é um limite superior, não um método.
- Não testado: uma regra **com o SWORD** (manter as manchas que contêm nós do rio). Os nós estão em
  `SWOT/nodes/*.geojson` (CSV, com lat/lon); exige converter coordenadas para os pixels do tile. É o teste direto da
  conclusão 3 — candidato a trabalho futuro ou a um último experimento.
