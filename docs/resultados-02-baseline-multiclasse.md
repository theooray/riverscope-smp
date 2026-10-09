# Resultados 02 — Baseline multiclasse (terra, rio, outros corpos d'água)

**Período:** 07–08/10/2026 · **Dataset:** RiverScope (PlanetScope, R+G+B+NIR) · **Conjunto avaliado:** teste (235 imagens)

## Configuração

| Item | Valor |
|---|---|
| Tarefa | Multiclasse com 3 classes: **terra** (rótulo 0), **rio** (rótulo 1) e **outros corpos d'água** — lagos, lagoas, reservatórios (rótulo 2). O rótulo 3 (raríssimo) é mapeado para terra |
| Grade | 2 modelos × 2 encoders × 2 losses × 2 DAs = **16 experimentos** |
| Modelos | Unet, FPN |
| Encoders | resnet50, efficientnet-b2 (pré-treinados no ImageNet) |
| Losses | crossentropy (`SoftCrossEntropyLoss`), dice (modo multiclasse) |
| Data augmentation | none, moderate |
| Fixos | Adam, LR 1e-4, weight decay 4e-4, batch 8, scheduler `plateau`, até 400 épocas, *early stopping* com paciência 21, seed 42 |
| Entrada | 512×512 (tiles de até 500×500 com *padding*), 4 canais |
| Saída | 3 canais (um por classe), predição por `argmax` |
| Comando | `nohup python -u run-batch.py --ds riverscope --n_classes 3` → `exp/exp_riverscope_3classes/` |

### Distribuição das classes

| Conjunto | Terra | Rio | Outros corpos d'água | Imagens com outros corpos d'água |
|---|---:|---:|---:|---:|
| Treino | 77,0% | 18,5% | **4,5%** | 240 / 787 |
| Validação | 70,1% | 27,3% | **2,6%** | 40 / 123 |
| Teste | 78,5% | 19,1% | **2,4%** | 110 / 235 |

A classe "outros corpos d'água" é **muito minoritária** (2–5% dos pixels) e aparece em menos da metade das imagens.

## Métricas usadas

- **mIoU / mF1**: média das 3 classes (redução `macro`). Cada classe pesa igual, então a classe rara influencia bastante.
- **IoU por classe**: cada classe avaliada como "classe vs. resto", somando todos os pixels do teste (`report_smp_(test)_per_class.csv`).
- **Val mIoU**: mIoU de validação na melhor época, calculado durante o treino (média por batch).

> **Atenção:** a redução `micro` no multiclasse (≈0,92 em todos os experimentos) é dominada pela classe terra
> e não é informativa; não é usada nesta análise. Pelo mesmo motivo, o IoU *por imagem* (`iou-iw`) da classe
> "outros corpos d'água" fica inflado (~0,5): em imagens sem essa classe e sem predição dela, o SMP conta IoU = 1.
> Para essa classe, use o IoU global.

## Resultados por experimento

**Épocas** = melhor época / total treinado. Métricas de classe = IoU global no teste.

| # | Modelo | Encoder | Loss | DA | Épocas | mIoU | mF1 | IoU terra | IoU rio | IoU outros | Val mIoU | Tempo |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Unet | resnet50 | crossentropy | none | 7 / 29 | 0.653 | 0.719 | 0.970 | 0.838 | 0.150 | 0.620 | 48 min |
| 1 | Unet | resnet50 | dice | none | 16 / 38 | 0.707 | 0.791 | 0.967 | 0.841 | 0.312 | 0.638 | 63 min |
| 2 | Unet | resnet50 | crossentropy | moderate | 7 / 29 | 0.601 | 0.631 | 0.968 | 0.835 | **0.000** | 0.664 | 49 min |
| 3 | Unet | resnet50 | dice | moderate | 15 / 37 | 0.629 | 0.718 | 0.945 | 0.745 | 0.197 | 0.653 | 62 min |
| 4 | Unet | efficientnet-b2 | crossentropy | none | 16 / 38 | 0.652 | 0.721 | 0.969 | 0.826 | 0.159 | 0.611 | 35 min |
| 5 | Unet | efficientnet-b2 | dice | none | 13 / 35 | 0.746 | **0.830** | 0.970 | 0.857 | **0.411** | 0.633 | 32 min |
| 6 | Unet | efficientnet-b2 | crossentropy | moderate | 6 / 28 | 0.599 | 0.630 | 0.968 | 0.830 | **0.000** | 0.609 | 29 min |
| 7 | Unet | efficientnet-b2 | dice | moderate | 19 / 41 | 0.729 | 0.814 | 0.970 | 0.847 | 0.371 | 0.648 | 42 min |
| 8 | FPN | resnet50 | crossentropy | none | 6 / 28 | 0.682 | 0.766 | 0.962 | 0.818 | 0.265 | 0.593 | 57 min |
| 9 | FPN | resnet50 | dice | none | 25 / 47 | 0.658 | 0.738 | 0.970 | 0.798 | 0.205 | 0.626 | 95 min |
| 10 | FPN | resnet50 | crossentropy | moderate | 3 / 25 | 0.637 | 0.704 | 0.962 | 0.817 | 0.132 | 0.597 | 51 min |
| 11 | FPN | resnet50 | dice | moderate | 34 / 56 | 0.684 | 0.766 | 0.963 | 0.833 | 0.255 | 0.658 | 114 min |
| 12 | FPN | efficientnet-b2 | crossentropy | none | 7 / 29 | 0.683 | 0.760 | 0.968 | 0.846 | 0.233 | 0.618 | 39 min |
| 13 | FPN | efficientnet-b2 | dice | none | 31 / 53 | 0.726 | 0.809 | 0.970 | 0.859 | 0.350 | 0.652 | 71 min |
| 14 | FPN | efficientnet-b2 | crossentropy | moderate | 16 / 38 | 0.695 | 0.777 | 0.970 | 0.835 | 0.280 | 0.615 | 55 min |
| 15 | FPN | efficientnet-b2 | dice | moderate | 34 / 56 | **0.747** | 0.829 | **0.972** | **0.866** | 0.403 | **0.673** | 80 min |

### Detalhe das classes rio e outros corpos d'água

| # | Configuração | Rio: Precisão | Rio: Recall | Rio: F1 | Outros: Precisão | Outros: Recall | Outros: F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Unet · resnet50 · CE · none | 0.895 | 0.930 | 0.912 | 0.673 | 0.162 | 0.261 |
| 1 | Unet · resnet50 · dice · none | 0.871 | 0.961 | 0.914 | 0.608 | 0.391 | 0.476 |
| 2 | Unet · resnet50 · CE · moderate | 0.882 | 0.940 | 0.910 | 0.041 | 0.000 | 0.000 |
| 3 | Unet · resnet50 · dice · moderate | 0.840 | 0.868 | 0.854 | 0.497 | 0.246 | 0.329 |
| 4 | Unet · effnet-b2 · CE · none | 0.877 | 0.935 | 0.905 | 0.732 | 0.169 | 0.274 |
| 5 | Unet · effnet-b2 · dice · none | **0.917** | 0.929 | 0.923 | 0.725 | 0.487 | **0.583** |
| 6 | Unet · effnet-b2 · CE · moderate | 0.882 | 0.934 | 0.907 | 0.146 | 0.000 | 0.000 |
| 7 | Unet · effnet-b2 · dice · moderate | 0.911 | 0.923 | 0.917 | 0.574 | **0.513** | 0.541 |
| 8 | FPN · resnet50 · CE · none | 0.881 | 0.919 | 0.900 | 0.746 | 0.291 | 0.419 |
| 9 | FPN · resnet50 · dice · none | 0.899 | 0.876 | 0.888 | 0.380 | 0.309 | 0.341 |
| 10 | FPN · resnet50 · CE · moderate | 0.901 | 0.898 | 0.900 | 0.807 | 0.136 | 0.233 |
| 11 | FPN · resnet50 · dice · moderate | 0.884 | 0.935 | 0.909 | 0.650 | 0.296 | 0.406 |
| 12 | FPN · effnet-b2 · CE · none | 0.895 | 0.940 | 0.917 | **0.820** | 0.246 | 0.378 |
| 13 | FPN · effnet-b2 · dice · none | 0.900 | **0.949** | 0.924 | 0.772 | 0.390 | 0.518 |
| 14 | FPN · effnet-b2 · CE · moderate | 0.904 | 0.917 | 0.910 | 0.525 | 0.374 | 0.437 |
| 15 | FPN · effnet-b2 · dice · moderate | 0.913 | 0.944 | **0.928** | 0.788 | 0.452 | 0.574 |

## Efeito médio de cada fator

Média dos 8 experimentos de cada lado:

| Fator | Opção | mIoU | IoU terra | IoU rio | IoU outros | Precisão outros | Recall outros |
|---|---|---:|---:|---:|---:|---:|---:|
| **Loss** | **dice** | **0.703** | 0.966 | 0.831 | **0.313** | 0.624 | **0.385** |
| | crossentropy | 0.650 | 0.967 | 0.831 | 0.152 | 0.561 | 0.172 |
| **Encoder** | **efficientnet-b2** | **0.697** | 0.970 | **0.846** | **0.276** | 0.635 | 0.329 |
| | resnet50 | 0.656 | 0.963 | 0.816 | 0.190 | 0.550 | 0.229 |
| DA | none | 0.688 | 0.968 | 0.835 | 0.261 | 0.682 | 0.306 |
| | moderate | 0.665 | 0.965 | 0.826 | 0.205 | 0.504 | 0.252 |
| Modelo | FPN | 0.689 | 0.967 | 0.834 | 0.265 | 0.686 | 0.312 |
| | Unet | 0.665 | 0.966 | 0.828 | 0.200 | 0.499 | 0.246 |

## Principais conclusões

1. **Desempenho por classe:** terra é praticamente resolvida (IoU médio 0.967); rio fica em
   **0.831 ± 0.029** (F1 0.907); outros corpos d'água ficam em apenas **0.233 ± 0.126** (F1 ~0.36).
   O mIoU médio é 0.677 ± 0.047, puxado para baixo pela classe minoritária.
2. **A classe "outros corpos d'água" é o gargalo.** O recall médio é de só **0.28**: o modelo deixa de
   detectar ~3 de cada 4 pixels dessa classe. A precisão (0.59) é bem maior que o recall, ou seja, quando o
   modelo prevê essa classe costuma acertar, mas prevê pouco. É o comportamento típico de uma classe
   minoritária (2–5% dos pixels).
3. **Colapso da classe minoritária com crossentropy + DA moderate:** nos dois Unet com essa combinação (#2 e #6),
   o modelo **nunca prevê** outros corpos d'água (IoU 0.000), com melhores épocas muito cedo (6–7).
4. **A dice é o fator mais importante para a classe minoritária:** dobra o IoU de outros corpos d'água
   (0.313 vs. 0.152) e o recall (0.385 vs. 0.172), elevando o mIoU em +0.053. No rio, as duas losses
   empatam (0.831). Isso é coerente com a dice ser menos sensível ao desbalanceamento de classes.
5. **efficientnet-b2 supera resnet50 em todas as classes** (+0.030 no rio, +0.086 em outros corpos d'água,
   +0.041 no mIoU), repetindo o padrão de encoder mais eficiente com menos parâmetros.
6. **O DA `moderate` não ajuda na média** (mIoU −0.023), principalmente por prejudicar a classe minoritária
   com crossentropy; com dice, os resultados com e sem DA ficam próximos.
7. **A validação prediz mal o teste:** correlação de Spearman de 0.35 entre o mIoU de validação e o de teste
   (0.41 para o IoU do rio). Ressalva: o mIoU de validação do treino é uma média por batch, não exatamente
   a mesma métrica do teste. A validação também tem proporções de classe diferentes (27% de rio contra 19% no teste).
8. **Custo:** 15,4 h de GPU no total (TITAN Xp), 58 min por experimento em média (29 a 114 min).

## Melhores configurações

| Critério | Configuração | mIoU | IoU rio | IoU outros |
|---|---|---:|---:|---:|
| Maior mIoU / IoU do rio | FPN + efficientnet-b2 + dice + moderate (#15) | **0.747** | **0.866** | 0.403 |
| Maior IoU de outros corpos d'água / mF1 | Unet + efficientnet-b2 + dice + none (#5) | 0.746 | 0.857 | **0.411** |
| Maior mIoU na validação | FPN + efficientnet-b2 + dice + moderate (#15) | 0.747 | 0.866 | 0.403 |

As 4 melhores configurações em mIoU (#15, #5, #7, #13) usam **efficientnet-b2 + dice**. Diferentemente do
baseline binário, aqui a melhor configuração no teste também foi a melhor na validação.

## Limitações

- Uma seed por configuração; sem estimativa de variância (o IoU da classe minoritária varia muito: desvio de 0.126).
- Experimentos rodados antes da ativação do `eval_val`: não há relatórios completos de validação.
- Nenhum tratamento explícito do desbalanceamento (pesos por classe, focal loss, amostragem).
- Escolha da melhor época pela loss de validação, não pela métrica.

## Próximos passos

1. Tratar o desbalanceamento: **pesos por classe** na loss ou **focal loss**, na configuração efficientnet-b2 + dice.
2. Repetir as melhores configurações com **3 seeds**.
3. Gerar os relatórios de validação a partir dos `best_model.pt` existentes.
4. Comparação com os demais batches em um documento separado.
