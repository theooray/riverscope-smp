# Resultados 01 — Baseline binário (rio vs. resto)

**Período:** 06–07/10/2026 · **Dataset:** RiverScope (PlanetScope, R+G+B+NIR) · **Conjunto avaliado:** teste (235 imagens)

## Configuração

| Item | Valor |
|---|---|
| Tarefa | Binária: rio (rótulo 1) vs. todo o resto (terra, lagos e outros corpos d'água) |
| Grade | 2 modelos × 2 encoders × 2 losses × 2 DAs = **16 experimentos** |
| Modelos | Unet, FPN |
| Encoders | resnet50, efficientnet-b2 (pré-treinados no ImageNet) |
| Losses | crossentropy, dice |
| Data augmentation | none, moderate |
| Fixos | Adam, LR 1e-4, weight decay 4e-4, batch 8, scheduler `plateau`, até 400 épocas, *early stopping* com paciência 21, seed 42 |
| Entrada | 512×512 (tiles de até 500×500 com *padding*), 4 canais |

## Resultados por experimento

**IoU** = IoU global do rio (todos os pixels do teste juntos) · **IoU-img** = média do IoU por imagem ·
**Val IoU** = IoU de validação na melhor época · **Épocas** = melhor época / total treinado.

| # | Modelo | Encoder | Loss | DA | Épocas | IoU | F1 | Precisão | Recall | IoU-img | Val IoU | Tempo |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Unet | resnet50 | crossentropy | none | 8 / 30 | 0.825 | 0.904 | 0.911 | 0.897 | 0.788 | 0.754 | 46 min |
| 1 | Unet | resnet50 | dice | none | 26 / 48 | 0.843 | 0.915 | 0.909 | 0.920 | 0.778 | 0.786 | 74 min |
| 2 | Unet | resnet50 | crossentropy | moderate | 13 / 35 | 0.835 | 0.910 | 0.899 | 0.922 | 0.796 | 0.742 | 57 min |
| 3 | Unet | resnet50 | dice | moderate | 50 / 72 | 0.843 | 0.915 | 0.895 | 0.935 | 0.801 | 0.746 | 117 min |
| 4 | Unet | efficientnet-b2 | crossentropy | none | 10 / 32 | 0.857 | 0.923 | 0.906 | 0.941 | 0.817 | 0.806 | 28 min |
| 5 | Unet | efficientnet-b2 | dice | none | 30 / 52 | 0.834 | 0.909 | 0.915 | 0.904 | 0.815 | **0.817** | 45 min |
| 6 | Unet | efficientnet-b2 | crossentropy | moderate | 32 / 54 | 0.853 | 0.921 | 0.912 | 0.929 | **0.818** | 0.775 | 52 min |
| 7 | Unet | efficientnet-b2 | dice | moderate | 15 / 37 | 0.841 | 0.913 | 0.878 | 0.951 | 0.815 | 0.790 | 36 min |
| 8 | FPN | resnet50 | crossentropy | none | 21 / 43 | 0.842 | 0.914 | 0.907 | 0.922 | 0.809 | 0.789 | 84 min |
| 9 | FPN | resnet50 | dice | none | 7 / 29 | 0.833 | 0.909 | 0.876 | 0.944 | 0.794 | 0.814 | 56 min |
| 10 | FPN | resnet50 | crossentropy | moderate | 14 / 36 | 0.836 | 0.911 | 0.893 | 0.929 | 0.793 | 0.776 | 70 min |
| 11 | FPN | resnet50 | dice | moderate | 5 / 27 | 0.825 | 0.904 | 0.865 | 0.946 | 0.788 | 0.777 | 52 min |
| 12 | FPN | efficientnet-b2 | crossentropy | none | 6 / 28 | 0.848 | 0.918 | 0.905 | 0.932 | 0.792 | 0.739 | 35 min |
| 13 | FPN | efficientnet-b2 | dice | none | 29 / 51 | **0.863** | **0.927** | **0.927** | 0.926 | 0.809 | 0.809 | 64 min |
| 14 | FPN | efficientnet-b2 | crossentropy | moderate | 15 / 37 | 0.835 | 0.910 | 0.878 | 0.945 | 0.804 | 0.774 | 50 min |
| 15 | FPN | efficientnet-b2 | dice | moderate | 25 / 47 | 0.861 | 0.925 | 0.908 | 0.943 | **0.818** | 0.755 | 64 min |

> No modo binário, as reduções `micro` e `macro` dão valores idênticos.

## Efeito médio de cada fator

Média dos 8 experimentos de cada lado:

| Fator | Opção | IoU | IoU-img | Precisão | Recall |
|---|---|---:|---:|---:|---:|
| **Encoder** | **efficientnet-b2** | **0.849** | **0.811** | 0.904 | 0.934 |
| | resnet50 | 0.835 | 0.793 | 0.895 | 0.927 |
| Modelo | FPN | 0.843 | 0.801 | 0.895 | 0.936 |
| | Unet | 0.841 | 0.804 | 0.903 | 0.925 |
| Loss | dice | 0.843 | 0.802 | 0.897 | 0.934 |
| | crossentropy | 0.842 | 0.802 | 0.901 | 0.927 |
| DA | none | 0.843 | 0.800 | 0.907 | 0.923 |
| | moderate | 0.841 | 0.804 | 0.891 | 0.938 |

## Principais conclusões

1. **Desempenho geral:** IoU do rio entre **0.825 e 0.863** (média 0.842 ± 0.012). Média por imagem:
   0.802 ± 0.013. As diferenças entre configurações são pequenas.
2. **O encoder é o único fator com efeito consistente:** o efficientnet-b2 supera o resnet50 em
   +0.014 de IoU e +0.018 de IoU-img, mesmo tendo menos parâmetros. Modelo, loss e DA praticamente
   não mudam a média.
3. **O modelo erra mais por excesso do que por falta:** o recall médio (0.93) é maior que a precisão
   (0.90), ou seja, há mais falsos positivos que falsos negativos. Uma análise do experimento 0 mostrou
   que **~67% dos falsos positivos estão em lagos e outros corpos d'água (rótulo 2)**: o modelo
   detecta água corretamente, mas não distingue rio de lago.
4. **O DA `moderate` troca precisão por recall** (−0.016 de precisão, +0.015 de recall), sem ganho
   líquido de IoU.
5. **A validação não está predizendo o teste:** a correlação (Spearman) entre o IoU de validação e o
   IoU de teste é praticamente nula (−0.04). A melhor configuração pela validação (#5) ficou em 0.834
   no teste, enquanto a melhor no teste (#13) foi a terceira na validação. Contribuem para isso: a
   validação é pequena (123 imagens) e ruidosa, a melhor época é escolhida por ela, e o IoU de
   validação do treino é uma média por batch, não exatamente a mesma métrica do teste.
6. **Overfitting cedo:** a melhor época ficou entre 5 e 50 (mediana ~15). Depois dela, a loss de
   treino continua caindo e a de validação sobe; o *early stopping* encerra o treino 21 épocas depois.
7. **Custo:** 15,5 h de GPU no total (TITAN Xp), 58 min por experimento em média (28 a 117 min).

## Melhores configurações

| Critério | Configuração | IoU teste | IoU-img teste |
|---|---|---:|---:|
| Maior IoU no teste | FPN + efficientnet-b2 + dice + none (#13) | 0.863 | 0.809 |
| Maior IoU-img no teste | Unet + efficientnet-b2 + crossentropy + moderate (#6) e FPN + efficientnet-b2 + dice + moderate (#15) | 0.853 / 0.861 | 0.818 |
| Maior IoU na validação | Unet + efficientnet-b2 + dice + none (#5) | 0.834 | 0.815 |

Com **uma única seed** e a validação pouco confiável, ainda **não dá para afirmar que uma configuração
é melhor que outra**. O que se sustenta é: efficientnet-b2 > resnet50, e o resto está empatado.

## Limitações

- Uma seed por configuração; sem estimativa de variância.
- Escolha da melhor época pela validação (pequena e ruidosa).
- Formulação binária penaliza a detecção de lagos como falso positivo.

## Próximos passos

1. Reportar IoU por classe e avaliar também a validação (`eval_val = True`).
2. **Multiclasse** (terra, rio, outros corpos d'água) com a mesma grade, comparando o IoU do rio.
3. Binário água vs. não água (rótulos 1 + 2) como referência.
4. Repetir as melhores configurações com **3 seeds** antes de comparar.
5. Fusão de dados (NDWI, Sentinel-2, SWOT) sobre a melhor configuração.
