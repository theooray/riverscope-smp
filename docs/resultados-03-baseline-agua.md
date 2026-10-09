# Resultados 03 — Baseline binário de toda a água (água vs. resto)

**Período:** 08–09/10/2026 · **Dataset:** RiverScope (PlanetScope, R+G+B+NIR) · **Conjuntos avaliados:** validação (123 imagens) e teste (235 imagens)

## Configuração

| Item | Valor |
|---|---|
| Tarefa | Binária: **água** (rótulos 1 e 2: rio + lagos e outros corpos d'água) vs. resto (terra). O rótulo 3 (raríssimo) conta como resto. É a mesma formulação usada no artigo original do RiverScope |
| Grade | 2 modelos × 2 encoders × 2 losses × 2 DAs = **16 experimentos** |
| Modelos | Unet, FPN |
| Encoders | resnet50, efficientnet-b2 (pré-treinados no ImageNet) |
| Losses | crossentropy (BCE, `SoftBCEWithLogitsLoss`), dice |
| Data augmentation | none, moderate |
| Fixos | Adam, LR 1e-4, weight decay 4e-4, batch 8, scheduler `plateau`, até 400 épocas, *early stopping* com paciência 21, seed 42 |
| Entrada | 512×512 (tiles de até 500×500 com *padding*), 4 canais |
| Avaliação | Teste **e validação** (`eval_val = True`), com as mesmas métricas |
| Comando | `nohup python -u run-batch.py --ds riverscope --target water` → `exp/exp_riverscope_water/` |

### Distribuição das classes

| Conjunto | Água | Resto |
|---|---:|---:|
| Treino | 23,0% | 77,0% |
| Validação | 30,0% | 70,0% |
| Teste | 21,5% | 78,5% |

As proporções de água coincidem com as reportadas no artigo do RiverScope (Figura 1: 23,04%, 29,95% e 21,49%),
o que confirma que a máscara de água (rótulos 1 + 2) corresponde à usada pelos autores.

## Métricas usadas

- **IoU, F1, precisão e recall**: da classe água, somando todos os pixels do conjunto (redução `micro`).
  No binário, `micro` e `macro` são idênticos.
- **IoU-img**: média do IoU por imagem (`micro-imagewise`).
- **Val IoU / Val F1**: as mesmas métricas no conjunto de validação, calculadas com o melhor modelo
  (`gen_report_(val)_(micro).csv`).

## Resultados por experimento

**Épocas** = melhor época / total treinado. Métricas no teste, exceto as colunas de validação.

| # | Modelo | Encoder | Loss | DA | Épocas | IoU | F1 | Precisão | Recall | IoU-img | Val IoU | Val F1 | Tempo |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Unet | resnet50 | crossentropy | none | 31 / 53 | 0.898 | 0.946 | 0.958 | 0.935 | 0.865 | 0.902 | 0.948 | 94 min |
| 1 | Unet | resnet50 | dice | none | 27 / 49 | **0.905** | **0.950** | 0.956 | 0.944 | **0.867** | 0.907 | 0.951 | 78 min |
| 2 | Unet | resnet50 | crossentropy | moderate | 17 / 39 | 0.898 | 0.946 | 0.951 | 0.941 | 0.862 | 0.846 | 0.917 | 63 min |
| 3 | Unet | resnet50 | dice | moderate | 13 / 35 | 0.903 | 0.949 | 0.941 | **0.957** | 0.866 | **0.923** | **0.960** | 57 min |
| 4 | Unet | efficientnet-b2 | crossentropy | none | 6 / 28 | 0.878 | 0.935 | **0.968** | 0.904 | 0.830 | 0.857 | 0.923 | 24 min |
| 5 | Unet | efficientnet-b2 | dice | none | 18 / 40 | 0.896 | 0.945 | 0.965 | 0.926 | 0.852 | 0.878 | 0.935 | 35 min |
| 6 | Unet | efficientnet-b2 | crossentropy | moderate | 6 / 28 | 0.885 | 0.939 | 0.961 | 0.918 | 0.783 | 0.900 | 0.947 | 27 min |
| 7 | Unet | efficientnet-b2 | dice | moderate | 19 / 41 | 0.893 | 0.944 | 0.956 | 0.931 | 0.865 | 0.846 | 0.917 | 40 min |
| 8 | FPN | resnet50 | crossentropy | none | 4 / 26 | 0.886 | 0.939 | 0.932 | 0.947 | 0.840 | 0.907 | 0.951 | 50 min |
| 9 | FPN | resnet50 | dice | none | 8 / 30 | 0.890 | 0.942 | 0.958 | 0.926 | 0.853 | 0.911 | 0.953 | 58 min |
| 10 | FPN | resnet50 | crossentropy | moderate | 2 / 24 | 0.853 | 0.921 | 0.958 | 0.886 | 0.804 | 0.886 | 0.940 | 46 min |
| 11 | FPN | resnet50 | dice | moderate | 14 / 36 | 0.888 | 0.941 | 0.960 | 0.922 | 0.851 | 0.914 | 0.955 | 70 min |
| 12 | FPN | efficientnet-b2 | crossentropy | none | 9 / 31 | 0.885 | 0.939 | 0.965 | 0.914 | 0.838 | 0.825 | 0.904 | 39 min |
| 13 | FPN | efficientnet-b2 | dice | none | 5 / 27 | 0.865 | 0.928 | 0.967 | 0.892 | 0.821 | 0.842 | 0.914 | 34 min |
| 14 | FPN | efficientnet-b2 | crossentropy | moderate | 4 / 26 | 0.879 | 0.936 | 0.937 | 0.934 | 0.825 | 0.832 | 0.908 | 36 min |
| 15 | FPN | efficientnet-b2 | dice | moderate | 10 / 32 | 0.887 | 0.940 | 0.956 | 0.924 | 0.844 | 0.912 | 0.954 | 44 min |

## Efeito médio de cada fator

Média dos 8 experimentos de cada lado (teste):

| Fator | Opção | IoU | F1 | Precisão | Recall | IoU-img |
|---|---|---:|---:|---:|---:|---:|
| **Modelo** | **Unet** | **0.894** | **0.944** | 0.957 | **0.932** | **0.849** |
| | FPN | 0.879 | 0.936 | 0.954 | 0.918 | 0.835 |
| **Loss** | **dice** | **0.891** | **0.942** | 0.957 | 0.928 | **0.852** |
| | crossentropy | 0.883 | 0.938 | 0.954 | 0.923 | 0.831 |
| Encoder | resnet50 | 0.890 | 0.942 | 0.952 | 0.932 | 0.851 |
| | efficientnet-b2 | 0.884 | 0.938 | 0.959 | 0.918 | 0.832 |
| DA | none | 0.888 | 0.941 | 0.959 | 0.923 | 0.846 |
| | moderate | 0.886 | 0.939 | 0.952 | 0.927 | 0.838 |

## Comparação com a literatura (mesmo dataset e mesma tarefa)

| Trabalho | Configuração | F1 | Precisão | Recall |
|---|---|---:|---:|---:|
| RiverScope (Daroya et al., 2025), Tabela A4 — média de 5 execuções | Unet + ResNet-50 + BCE + ImageNet | 94.05 | 93.33 | 94.80 |
| RiverScope (Daroya et al., 2025), melhor modelo | Unet + ResNet-50 + BCE + SeCo | 94.39 | 94.21 | 94.62 |
| SuperRivolution (Daroya & Maji, 2025), limite superior | Unet + Swin-B + BCE + ImageNet | 94.1 ± 0.6 | 93.0 | 95.2 |
| **Este trabalho** (1 execução) | Unet + ResNet-50 + crossentropy (BCE) + none (#0) | **94.63** | 95.83 | 93.47 |
| **Este trabalho, melhor no teste** | Unet + ResNet-50 + dice + none (#1) | **94.98** | 95.59 | 94.38 |

**O pipeline reproduz os resultados publicados.** A configuração equivalente à do artigo do RiverScope
alcança F1 de 94.63, no mesmo nível (ligeiramente acima) do valor reportado. Diferenças de protocolo:
o artigo usa adaptador linear de 4→3 canais, batch 32, 50 épocas fixas, LR ajustado por configuração,
seleção pelo F1 de validação e média de 5 execuções; este trabalho usa a primeira camada com 4 canais,
batch 8, *early stopping* pela loss de validação, LR fixo e 1 execução.

## Principais conclusões

1. **Desempenho geral:** F1 médio de **0.940 ± 0.008** (IoU 0.887 ± 0.013), com valores entre 0.921 e 0.950.
   A variação entre configurações é pequena.
2. **Precisão maior que recall** (0.956 contra 0.925, em média): o modelo erra mais por **deixar de detectar
   água** do que por marcar água onde não há. Os falsos positivos são relativamente raros.
3. **Unet supera FPN** (+0.015 de IoU, +0.014 de IoU-img), principalmente pelo recall (0.932 contra 0.918).
4. **Dice supera crossentropy** (+0.008 de IoU, +0.021 de IoU-img), com ganho maior na métrica por imagem.
5. **resnet50 é ligeiramente melhor que efficientnet-b2** (+0.006 de IoU, +0.019 de IoU-img). O efficientnet-b2
   tem precisão um pouco maior, mas recall menor.
6. **O DA `moderate` não ajuda** na média (diferenças abaixo de 0.01).
7. **A validação passou a ser útil para seleção:** com as mesmas métricas no teste e na validação, a correlação
   de Spearman entre as duas é **0.48**. A melhor configuração pela validação (#3, Unet + resnet50 + dice +
   moderate, Val F1 0.960) ficou em **segundo lugar no teste** (F1 0.949), a 0.001 da melhor.
8. **Convergência rápida:** a melhor época ficou entre 2 e 31 (mediana ~10); vários modelos atingem o melhor
   resultado em menos de 10 épocas.
9. **Custo:** 13,2 h de GPU no total (TITAN Xp), 50 min por experimento em média (24 a 94 min).

## Melhores configurações

| Critério | Configuração | Val F1 | F1 teste | IoU teste |
|---|---|---:|---:|---:|
| Maior F1 / IoU no teste | Unet + resnet50 + dice + none (#1) | 0.951 | **0.950** | **0.905** |
| **Maior F1 na validação** (seleção recomendada) | Unet + resnet50 + dice + moderate (#3) | **0.960** | 0.949 | 0.903 |
| Maior precisão no teste | Unet + efficientnet-b2 + crossentropy + none (#4) | 0.923 | 0.935 | 0.878 |

As 4 melhores configurações no teste são todas **Unet + resnet50**.

## Limitações

- Uma seed por configuração; sem estimativa de variância.
- Escolha da melhor época pela loss de validação, não pelo F1 (o artigo original usa o F1).
- LR fixo (1e-4) para todas as configurações; o artigo original ajusta o LR por configuração.

## Próximos passos

1. Repetir as melhores configurações com **3 seeds**, para comparar com a média de 5 execuções do artigo.
2. Comparação com os batches de rio e multiclasse em um documento separado.
