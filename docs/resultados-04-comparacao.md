# Resultados 04 — Comparação das formulações com 3 seeds

**Data:** 12/10/2026 · **Dataset:** RiverScope (PlanetScope, R+G+B+NIR) · **Conjuntos:** teste (235 imagens) e validação (123 imagens)

Compara as três formulações — **rio** (binário), **3 classes** e **água** (binário) — nas mesmas 4 configurações,
cada uma com **3 seeds**. Junta os baselines com seed 42 (B1, B2, B3) e as seeds 20 e 83 (S1).

## Configuração

| Item | Valor |
|---|---|
| Formulações | **Rio:** rótulo 1 vs. resto · **3 classes:** terra / rio / outros corpos d'água (rótulo 2) · **Água:** rótulos 1 + 2 vs. resto |
| Configurações | {Unet, FPN} × {resnet50, efficientnet-b2}, **dice**, **sem DA** (escolhidas pelas tendências da validação, não pelo teste) |
| Seeds | 42, 20, 83 (definidos antes dos resultados; controlam pesos, embaralhamento e DA) |
| Execuções | 4 configs × 3 seeds × 3 formulações = **36** |
| Fixos | Adam, LR 1e-4, weight decay 4e-4, batch 8, `plateau`, até 400 épocas, *early stopping* pela loss de validação (paciência 21) |
| Pastas | `exp/exp_riverscope{,_3classes,_water}{,_s20,_s83}/` |
| Script | `analysis/compare_formulations.py` → `analysis/out/runs.csv` (uma linha por execução) e `analysis/out/summary.txt` |

## Métricas

- **IoU, F1, precisão, recall**: da classe de interesse, somando os pixels de todo o conjunto (`micro`).
  No 3 classes, o rio e os lagos são avaliados **cada um como classe contra o resto** (`report_smp_(…)_per_class.csv`),
  o que torna o IoU do rio diretamente comparável com o binário.
- **IoU-img**: média do IoU por imagem.
- **Lago → rio**: % dos pixels de rótulo 2 previstos como rio (das predições salvas, só na área original do tile).
- **FP em lagos**: % dos pixels falsos positivos de rio que são rótulo 2.
- **F1 sem lagos** (contrafactual): F1 do rio do mesmo modelo, ignorando os pixels de rótulo 2 na avaliação.
- Valores: **média ± desvio padrão** (amostral). "Pareado": mesma configuração e mesmo seed nas duas formulações.
  Testes de Wilcoxon pareados com n = 12 têm pouco poder; servem como indicação, não como prova.
- Conferência: o cálculo pelas predições reproduz os números já registrados (Unet + ResNet-50 + CE, seed 42:
  lago → rio 47,8%, FP em lagos 67,1%, F1 0,904 → 0,932 sem lagos).

## 1. Tabela principal (teste, média ± desvio sobre 3 seeds)

### Rio (binário)

| Modelo | Encoder | IoU | F1 | Precisão | Recall | IoU-img | Lago → rio | Melhor época |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unet | resnet50 | 0.838 ± 0.006 | 0.912 ± 0.003 | 0.894 ± 0.013 | 0.930 ± 0.011 | 0.794 ± 0.014 | 57.3 ± 10.9% | 22 ± 10 |
| Unet | efficientnet-b2 | 0.848 ± 0.015 | 0.918 ± 0.009 | 0.914 ± 0.001 | 0.921 ± 0.017 | 0.816 ± 0.006 | 43.4 ± 3.0% | 24 ± 7 |
| FPN | resnet50 | 0.822 ± 0.012 | 0.902 ± 0.007 | 0.867 ± 0.030 | 0.942 ± 0.025 | 0.782 ± 0.017 | 57.4 ± 4.5% | 11 ± 9 |
| FPN | efficientnet-b2 | **0.858 ± 0.010** | **0.923 ± 0.006** | **0.917 ± 0.012** | 0.930 ± 0.019 | 0.813 ± 0.005 | **33.4 ± 11.0%** | 30 ± 7 |
| **Todas (12)** | | **0.841 ± 0.017** | **0.914 ± 0.010** | 0.898 ± 0.026 | 0.931 ± 0.018 | 0.801 ± 0.018 | 47.9 ± 12.7% | 22 ± 10 |

### 3 classes (métricas do rio e dos outros corpos d'água)

| Modelo | Encoder | IoU rio | F1 rio | Precisão rio | Recall rio | IoU lagos | Recall lagos | mIoU | Lago → rio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unet | resnet50 | 0.826 ± 0.014 | 0.905 ± 0.008 | 0.881 ± 0.011 | 0.930 ± 0.027 | 0.257 ± 0.054 | 0.323 ± 0.087 | 0.684 ± 0.023 | 55.5 ± 7.2% |
| Unet | efficientnet-b2 | 0.844 ± 0.012 | 0.916 ± 0.007 | **0.920 ± 0.013** | 0.912 ± 0.024 | **0.367 ± 0.038** | **0.468 ± 0.033** | **0.727 ± 0.017** | **35.0 ± 10.4%** |
| FPN | resnet50 | 0.822 ± 0.021 | 0.902 ± 0.013 | 0.897 ± 0.010 | 0.908 ± 0.032 | 0.253 ± 0.056 | 0.332 ± 0.034 | 0.680 ± 0.023 | 50.3 ± 4.7% |
| FPN | efficientnet-b2 | **0.849 ± 0.008** | **0.919 ± 0.005** | 0.906 ± 0.012 | 0.932 ± 0.019 | 0.322 ± 0.049 | 0.394 ± 0.105 | 0.714 ± 0.018 | 43.0 ± 7.7% |
| **Todas (12)** | | **0.835 ± 0.018** | **0.910 ± 0.010** | 0.901 ± 0.018 | 0.920 ± 0.025 | 0.300 ± 0.065 | 0.379 ± 0.087 | 0.701 ± 0.027 | 46.0 ± 10.4% |

Classe de lagos (12 execuções): precisão 0.608 ± 0.123, recall 0.379 ± 0.087 — o modelo encontra só ~38% dos pixels de lago.

### Água (binário)

| Modelo | Encoder | IoU | F1 | Precisão | Recall | IoU-img |
|---|---|---:|---:|---:|---:|---:|
| Unet | resnet50 | **0.907 ± 0.003** | **0.951 ± 0.001** | 0.955 ± 0.005 | **0.947 ± 0.004** | **0.871 ± 0.004** |
| Unet | efficientnet-b2 | 0.896 ± 0.003 | 0.945 ± 0.002 | 0.960 ± 0.010 | 0.931 ± 0.013 | 0.855 ± 0.006 |
| FPN | resnet50 | 0.898 ± 0.007 | 0.946 ± 0.004 | 0.955 ± 0.003 | 0.938 ± 0.011 | 0.861 ± 0.007 |
| FPN | efficientnet-b2 | 0.882 ± 0.016 | 0.937 ± 0.009 | **0.962 ± 0.009** | 0.914 ± 0.024 | 0.839 ± 0.020 |
| **Todas (12)** | | **0.896 ± 0.012** | **0.945 ± 0.007** | 0.958 ± 0.007 | 0.933 ± 0.018 | 0.857 ± 0.015 |

## 2. Ruído de seed vs. diferença entre configurações

IoU da classe de interesse no teste:

| Formulação | Desvio entre seeds (média por config) | Desvio entre médias das configs | Amplitude das médias |
|---|---:|---:|---:|
| Rio | 0.011 | 0.015 | 0.036 |
| 3 classes — rio | 0.014 | 0.014 | 0.028 |
| 3 classes — lagos | 0.049 | 0.055 | 0.115 |
| Água | 0.007 | 0.010 | 0.024 |

O ruído de seed (~0.01 de IoU) é **da mesma ordem** da diferença entre configurações. Diferenças de 0.01 entre
duas execuções isoladas não significam nada; só diferenças consistentes entre seeds (abaixo) devem ser interpretadas.

## 3. 3 classes vs. rio binário (pareado, 12 pares)

| Métrica (teste) | Rio | 3 classes | Diferença média | Mediana | 3 classes melhor | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| IoU do rio | 0.841 | 0.835 | −0.006 | −0.004 | 3/12 | 0.27 |
| F1 do rio | 0.914 | 0.910 | −0.004 | −0.002 | 3/12 | 0.27 |
| Precisão do rio | 0.898 | 0.901 | +0.003 | −0.003 | 5/12 | 0.91 |
| Recall do rio | 0.931 | 0.920 | −0.011 | −0.007 | 5/12 | 0.42 |
| **Lago → rio** | 47.9% | 46.0% | −1.9 pp | −0.1 pp | 6/12 menor | 0.68 |
| **FP em lagos** | 56.6% | 56.3% | −0.2 pp | −1.1 pp | — | 0.97 |
| IoU do rio (validação) | 0.792 | 0.771 | −0.021 | −0.019 | 5/12 | 0.47 |
| Lago → rio (validação) | 15.9% | 26.5% | +10.6 pp | +3.0 pp | 3/12 menor | 0.13 |

**Leitura:** treinar com a classe "outros corpos d'água" **não reduz** a confusão rio–lago: a fração de lagos
previstos como rio é praticamente a mesma (mediana −0,1 pp), e a fração de falsos positivos em lagos também.
O IoU do rio fica ligeiramente abaixo (−0,006; 3 classes melhor em só 3 de 12 pares), mas sem significância.
A afirmação segura é **"sem ganho"**, não "piora". Vale **nas mesmas condições** (LR, épocas e loss não ajustados
para o multiclasse, sem pesos por classe) — o que motiva o teste com BCE + Dice / pesos por classe.

## 4. Custo dos outros corpos d'água para o modelo de rio (12 execuções)

| Medida (teste) | Valor | Mín.–máx. |
|---|---:|---:|
| F1 do rio | 0.915 ± 0.009 | 0.901–0.927 |
| F1 do rio ignorando os pixels de lago (contrafactual) | 0.942 ± 0.008 | 0.932–0.953 |
| **Custo atribuível aos lagos (F1)** | **+0.027 ± 0.007** | 0.014–0.037 |
| **FP do rio que são rótulo 2** | **56.6 ± 8.8%** | 40.5–65.8% |
| Pixels de rótulo 2 previstos como rio | 47.9 ± 12.7% | 24.5–65.2% |
| Tiles com FP em lagos | 54 ± 6 | 45–64 |
| Parte dos FP em lagos nos 6 piores tiles | 53 ± 10% | 39–70% |

- O rótulo 2 é só **~3% dos pixels não-rio** do teste, mas concentra **57% dos falsos positivos** — cerca de
  **19× o esperado** pela área. O efeito aparece em todas as 12 execuções (mínimo 40%).
- O custo de ~0,027 de F1 é consistente com a comparação entre modelos rio vs. água (mesma config e seed):
  F1 0.914 vs. 0.945 (**+0.031, água melhor em 12/12**), com a diferença inteira na **precisão**
  (0.898 vs. 0.958, +0.060, 12/12) e **não** no recall (0.931 vs. 0.933, 7/12). As duas medidas convergem:
  o que o rio perde em relação à água são falsos positivos em outros corpos d'água.
- **Concentrado em poucos tiles:** metade dos FP em lagos está em 6 dos ~54 tiles afetados — reportar por tile,
  não só por pixel.
- **Na validação o custo é pequeno** (+0.007 ± 0.007): há menos lagos (2,6% dos pixels) e lagos são previstos
  como rio em só 15.9% dos casos. O erro dominante na validação é o oposto (seção 6).

## 5. Efeitos de fator (pareados, 6 pares por comparação)

| Formulação | Métrica | efficientnet-b2 − resnet50 | FPN − Unet |
|---|---|---|---|
| Rio | IoU | **+0.023** (melhor em 5/6) | −0.003 (2/6) |
| 3 classes | IoU do rio | **+0.023** (6/6) | +0.000 (5/6) |
| 3 classes | IoU dos lagos | **+0.090** (5/6) | −0.025 (2/6) |
| Água | IoU | **−0.013** (0/6) | −0.011 (1/6) |
| Rio | Lago → rio | **−18.9 pp** (menor em 6/6) | — |
| Rio | Precisão | **+0.035** (6/6) | — |
| 3 classes | Lago → rio | **−13.9 pp** (5/6) | — |

- **O encoder é o único fator com efeito consistente, e o sinal muda com a formulação:** o efficientnet-b2 é
  melhor para **rio** e para **lagos** (as tarefas que exigem separar tipos de água), mas o resnet50 é melhor para
  **toda a água** (0/6). O ganho do efficientnet-b2 no rio vem da **precisão**, por confundir menos lagos com rio
  (−19 pp, 6/6). Correlação entre "lago → rio" e precisão do rio nas 12 execuções: −0,76.
- O modelo (Unet vs. FPN) não tem efeito consistente no rio nem no 3 classes; na água o Unet é levemente melhor (5/6).
- 🔴 Interpretação possível (não testada): o efficientnet-b2 capturaria melhor o contexto/forma que distingue rio
  de lago. É a mesma hipótese do experimento com DeepLabV3+ (D1); os dois resultados devem ser lidos juntos.

## 6. Validação vs. teste

| Formulação | IoU teste | IoU validação | Recall teste | Recall validação |
|---|---:|---:|---:|---:|
| Rio | 0.841 ± 0.017 | 0.792 ± 0.042 | 0.931 ± 0.018 | **0.823 ± 0.056** |
| 3 classes (rio) | 0.835 ± 0.018 | 0.771 ± 0.046 | 0.920 ± 0.025 | 0.810 |
| Água | 0.896 ± 0.012 | **0.898 ± 0.026** | 0.933 ± 0.018 | — |

- **A queda na validação é do alvo "rio", não do modelo:** o modelo de água não cai (0.896 → 0.898), e o de rio
  cai ~0,05 de IoU, com o **recall** caindo de 0,93 para 0,82. Confirma o achado A5 (rio perdido na validação)
  nas 12 execuções, não só em 1 modelo.
- **Ranking:** a correlação de Spearman entre IoU de validação e de teste por execução é fraca no rio (0,15; p = 0,65)
  e na água (0,26; p = 0,42) e moderada no 3 classes (0,62; p = 0,03). Mas, **pela média dos 3 seeds**, a validação
  escolhe a mesma melhor configuração do teste no rio (FPN + efficientnet-b2). Na água e no 3 classes, não.
- Conclusão: escolher a configuração por **uma** execução não é confiável; pela média de seeds, a escolha melhora.

## 7. Convergência

Melhor época (mediana / máx.): rio 24 / 37 · 3 classes 18 / 63 · água 18 / 27. Total de épocas treinadas:
mediana 40–47, máximo 85. Nenhuma execução chegou perto do limite de 400 — o *early stopping* sempre atua.

## Conclusões

1. **Desempenho com 3 seeds:** rio IoU **0.841 ± 0.017** (F1 0.914 ± 0.010); 3 classes, rio IoU **0.835 ± 0.018**
   e lagos **0.300 ± 0.065**; água IoU **0.896 ± 0.012** (F1 0.945 ± 0.007).
2. **Os outros corpos d'água custam ~0,027 de F1 ao modelo de rio** (contrafactual, 12/12 execuções, 0,014–0,037),
   e o modelo de água supera o de rio em 12/12 pares, com a diferença inteira na precisão.
3. **57% dos falsos positivos do rio estão em lagos** (~19× o esperado pela área), em todas as execuções,
   e concentrados em poucos tiles.
4. **O multiclasse, nas mesmas condições, não reduz a confusão:** lago → rio e FP em lagos praticamente iguais
   (pareado, 12 pares); IoU do rio sem ganho. A classe de lagos é mal aprendida (IoU 0,30, recall 0,38).
5. **O encoder importa mais que a formulação para a confusão rio–lago:** efficientnet-b2 reduz lago → rio em
   ~19 pp (6/6) e aumenta a precisão do rio; mas é pior que o resnet50 para segmentar toda a água.
6. **Na validação, o erro do rio é o inverso** (recall 0,82 vs. 0,93 no teste), enquanto o modelo de água não cai.
7. **Ruído de seed ≈ diferenças entre configurações** (~0,01 de IoU): comparações só com média de seeds e pareadas.

## Pendências

- **D1 (DeepLabV3+, rodando):** comparar com o Unet + efficientnet-b2 desta tabela (mesmos seeds) — lago → rio,
  precisão e recall na validação.
- **BCE + Dice / pesos por classe:** testar se a classe de lagos melhora e se isso reduz lago → rio no 3 classes.
- **Análise por largura de rio** e replicação dos diagnósticos por trecho na validação (A5).
- Formulação "água": o encoder resnet50 é melhor aqui — considerar ao comparar com o artigo original (A1).
