# RiverScope SMP — Segmentação de rios em imagens PlanetScope

Pipeline em PyTorch para **segmentação semântica de rios** em imagens de satélite
PlanetScope do dataset [RiverScope](#dataset), usando arquiteturas e encoders pré-treinados da
biblioteca [Segmentation Models PyTorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch).

O projeto treina e avalia, em lote, combinações de arquitetura, encoder, função de perda e
estratégia de data augmentation, gerando métricas, gráficos e mapas de erro para cada experimento.

---

## Sumário

- [Visão geral](#visão-geral)
- [Dataset](#dataset)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Saídas de cada experimento](#saídas-de-cada-experimento)
- [Resultados](#resultados)
- [Próximos passos](#próximos-passos)
- [Créditos](#créditos)

---

## Visão geral

```
Imagem PlanetScope (R, G, B, NIR) ──► Modelo SMP (Unet / FPN / ...) ──► Máscara de rio
```

- **Tarefa:** segmentação binária — **rio** vs. **todo o resto** (terra e outros corpos d'água).
- **Entrada:** tiles de até 500×500 px com 4 bandas (R, G, B, NIR), ajustados para 512×512 com
  *padding* (sem redimensionar, para não distorcer tiles pequenos).
- **Normalização:** estatísticas do ImageNet para RGB e média/desvio calculados no treino para o NIR.
- **Treino:** Adam, scheduler `ReduceLROnPlateau`, *early stopping* pela loss de validação e
  salvamento do melhor modelo.
- **Retomada:** um checkpoint completo é salvo a cada época; um treino interrompido continua de
  onde parou (`--resume`).

## Dataset

O **RiverScope** reúne, para cada trecho de rio, dados de várias fontes alinhadas:

| Fonte | Conteúdo | Usado aqui |
|---|---|---|
| **PlanetScope** | Imagens de 4 bandas (~3 m) e máscaras rotuladas | ✅ entrada e rótulos |
| Sentinel-2 | Imagens de 12 bandas (10–20 m), reprojetadas para o tile PlanetScope | — |
| SWOT | Nuvem de pixels de água (`pixc`) e nós do rio (`nodes`) | — |
| SWORD | Base vetorial de rios (linhas centrais e larguras) | — |

Rótulos das máscaras:

| Valor | Significado | No modo binário |
|---|---|---|
| 0 | Terra / fundo | não rio |
| 1 | **Rio** | **rio** |
| 2 | Lagos e outros corpos d'água | não rio |
| 3 | Raro (1 imagem no teste) | não rio |

São usadas as divisões oficiais do dataset (`train.csv`, `valid.csv`, `test.csv`):
**787 / 123 / 235** amostras. Arquivos TIFF truncados são detectados e ignorados automaticamente.

> O dataset **não** está incluído neste repositório. Baixe-o da fonte oficial e coloque-o em
> `Datasets/RiverScope/RiverScope_dataset/` (ou informe outro caminho com `--ds_path`).

## Estrutura do repositório

```
.
├── train-test.py        # Treina e avalia UM experimento (dados, modelo, treino, métricas, relatórios)
├── run-batch.py         # Executa vários experimentos em sequência, chamando o train-test.py
├── .gitignore
└── README.md
```

Pastas criadas localmente (fora do Git):

```
Datasets/RiverScope/RiverScope_dataset/   # dataset
exp_riverscope/                           # resultados (uma subpasta por experimento)
log_batch.txt                             # log da execução em lote
```

## Instalação

Testado com Python 3.12, CUDA 12.6 e uma GPU NVIDIA TITAN Xp (12 GB).

```bash
conda create -n env-smp python=3.12
conda activate env-smp
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install segmentation-models-pytorch==0.5.0 albumentations==2.0.8 \
            numpy pandas matplotlib pillow scikit-image scikit-learn tifffile
```

Versões usadas nos experimentos: NumPy 2.5.2, Pandas 3.0.6, Matplotlib 3.11.2, Pillow 12.3.0,
scikit-image 0.26.0, scikit-learn 1.9.1, tifffile 2026.9.20.

## Como usar

### Um único experimento

```bash
python train-test.py --dataset_name riverscope --n_classes 2 --in_channels 4 \
    --h_size 512 --w_size 512 --model Unet --backbone resnet50 --loss dice \
    --da_train none --max_epochs 400 --batch_size 8 --lr 0.0001 --scheduler plateau \
    --save_images --segmap_mode darker
```

Principais argumentos:

| Argumento | Descrição | Valores |
|---|---|---|
| `--model` | Arquitetura | `Unet`, `Unet++`, `FPN`, `DeepLabV3+`, `MAnet`, `Linknet`, `PSPNet`, `PAN`, `Segformer`… |
| `--backbone` | Encoder | `resnet50`, `efficientnet-b2`, … (qualquer encoder do SMP) |
| `--loss` | Função de perda | `crossentropy`, `dice`, `jaccard`, `tversky`, `focal`, `lavosz` |
| `--da_train` | Data augmentation no treino | `none`, `mild`, `moderate`, `strong` |
| `--scheduler` | Ajuste da taxa de aprendizado | `plateau`, `cosine`, `onecycle` |
| `--n_classes` | Nº de classes (`2` = binário rio vs. resto; `3` = terra, rio e outros corpos d'água) | `2`, `3` |
| `--in_channels` | `3` = RGB, `4` = RGB + NIR | `3`, `4` |
| `--patience` | Épocas sem melhora antes do *early stopping* | padrão `21` |
| `--resume` | Continua um treino interrompido a partir do último checkpoint | — |
| `--ds_path` | Caminho do dataset, se diferente do padrão | — |
| `--debug` | Usa uma fração dos dados para testes rápidos | — |

### Vários experimentos (lote)

Edite as listas no início do `run-batch.py` (modelos, encoders, losses, augmentations, etc.) e rode:

```bash
nohup python -u run-batch.py --ds riverscope >> log_batch.txt 2>&1 &
```

Acompanhe com:

```bash
tail -f log_batch.txt
```

- Experimentos já concluídos são **pulados** automaticamente (use `--no_skip` para refazê-los).
- Um experimento interrompido é **retomado** da última época salva.
- Para interromper, encerre primeiro o `run-batch.py` e depois o `train-test.py`, para o lote
  não iniciar o próximo experimento.

## Saídas de cada experimento

Cada experimento gera uma pasta
`exp_riverscope/exp_<modelo>_<encoder>_<loss>_<batch>_<lr>_<épocas>_<scheduler>_<da>/` com:

| Arquivo | Conteúdo |
|---|---|
| `general_report.txt` | Versões, argumentos, loss/otimizador/scheduler, melhor época e tempo total |
| `best_model.pt` / `best_model_SMP/` | Pesos da melhor época (menor loss de validação) |
| `training_report.csv` | Loss, IoU, F1, acurácia e LR por época |
| `loss_history`, `smp_history`, `lr_history` (`.png`/`.pdf`) | Curvas de treino |
| `report_smp_(test)_*.csv` | Métricas no teste: globais (`micro`/`macro`) e por imagem (`*-imagewise`) |
| `test_images/` · `test_true/` · `test_pred/` | Imagem de entrada, máscara verdadeira e predição |
| `test_seg_map/` | Mapa de erros por pixel |

Cores do mapa de erros (modo binário):

| Cor | Verdade | Predição | Significado |
|---|---|---|---|
| ⬛ Preto | não rio | não rio | Verdadeiro negativo |
| 🟩 Verde | rio | rio | Verdadeiro positivo |
| 🟥 Vermelho | não rio | rio | **Falso positivo** |
| 🟧 Laranja | rio | não rio | **Falso negativo** |

Além disso, `exp_riverscope/gen_report_(test)_(micro|macro).csv` reúne uma linha por experimento,
facilitando a comparação.

## Resultados

Lote binário (rio vs. resto), conjunto de **teste**, Adam com LR 1e-4, batch 8, scheduler
`plateau`, 1 seed. **IoU** = IoU global do rio; **IoU-img** = média do IoU por imagem.

| Modelo | Encoder | Loss | DA | Melhor época | IoU | F1 | IoU-img |
|---|---|---|---|---:|---:|---:|---:|
| Unet | resnet50 | crossentropy | none | 8 | 0.825 | 0.904 | 0.788 |
| Unet | resnet50 | dice | none | 26 | 0.843 | 0.915 | 0.778 |
| Unet | resnet50 | crossentropy | moderate | 13 | 0.835 | 0.910 | 0.796 |
| Unet | resnet50 | dice | moderate | 50 | 0.843 | 0.915 | 0.801 |
| Unet | efficientnet-b2 | crossentropy | none | 10 | 0.857 | 0.923 | 0.817 |
| Unet | efficientnet-b2 | dice | none | 30 | 0.834 | 0.909 | 0.815 |
| Unet | efficientnet-b2 | crossentropy | moderate | 32 | 0.853 | 0.921 | **0.818** |
| Unet | efficientnet-b2 | dice | moderate | 15 | 0.841 | 0.913 | 0.815 |
| FPN | resnet50 | crossentropy | none | 21 | 0.842 | 0.914 | 0.809 |
| FPN | resnet50 | dice | none | 7 | 0.833 | 0.909 | 0.794 |
| FPN | resnet50 | crossentropy | moderate | 14 | 0.836 | 0.911 | 0.793 |
| FPN | resnet50 | dice | moderate | 5 | 0.825 | 0.904 | 0.788 |
| FPN | efficientnet-b2 | crossentropy | none | 6 | 0.848 | 0.918 | 0.792 |
| FPN | efficientnet-b2 | dice | none | 29 | **0.863** | **0.927** | 0.809 |
| FPN | efficientnet-b2 | crossentropy | moderate | 15 | 0.835 | 0.910 | 0.804 |
| FPN | efficientnet-b2 | dice | moderate | 25 | 0.861 | 0.925 | 0.818 |

Observações:

- As diferenças entre configurações são pequenas (IoU entre 0.825 e 0.863), com **uma única seed** e
  um conjunto de validação pequeno e ruidoso; ainda não é possível afirmar que uma configuração é
  melhor que outra.
- Os modelos começam a sofrer **overfitting** poucas épocas após a melhor época; o *early stopping*
  garante que o modelo avaliado seja o da melhor época de validação.
- **Cerca de 2/3 dos falsos positivos ocorrem em lagos e outros corpos d'água (rótulo 2)**, que no
  modo binário contam como "não rio". O modelo detecta água corretamente, mas tem dificuldade em
  distinguir rio de lago.

## Próximos passos

1. Reportar o IoU por classe e escolher configurações pela **validação**, não pelo teste.
2. **Multiclasse** (terra, rio, outros corpos d'água) para reduzir a confusão rio × lago.
3. Binário **água vs. não água** (rótulos 1 + 2) como referência.
4. **Fusão de dados:** NDWI, bandas do Sentinel-2 (incluindo SWIR) e SWOT `pixc` como canais extras.
5. Repetição das melhores configurações com **várias seeds** (média ± desvio).
6. Análise de erros por tipo de região e teste de generalização para regiões não vistas.

## Créditos

O pipeline é baseado no tutorial *"Getting Started with Semantic Segmentation using PyTorch & SMP"*
(SIBGRAPI 2025), de João Fernando Mari, Leandro Henrique Furtado Pinto Silva,
Mauricio Cunha Escarpinati e André Ricardo Backes, adaptado para o dataset RiverScope
(leitura de GeoTIFFs multibanda, *padding* de tiles, divisões oficiais e mapeamento de rótulos),
com retomada de treino por checkpoint.

Bibliotecas principais: [PyTorch](https://pytorch.org/),
[Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) e
[Albumentations](https://albumentations.ai/).
