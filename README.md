# WSI DeepMIML: Multi-Instance Multi-Label learning on synthetic WSI

This is the second half of the pipeline. The [WSI-Preprocessing-Framework](../WSI-Preprocessing-Framework-main) turns
raw [diffinfinite](https://arxiv.org/abs/2306.13384) output into patched and annotated bags, this repo takes those bags
and runs MIML methods on them.

## Where this sits in the pipeline

The preprocessing framework ends with two things: a folder of patches grouped per sample, and an `annotator.csv` holding
the labels computed from the masks. Both are exactly what we consume here.

```
preprocessing                                   deepmiml
─────────────                                   ────────
diffinfinite macro patches
  ∟ pre-split (1024x1024)
      ∟ patching (256x256)      ──────────>     bags of instances
      ∟ annotator (strategy)    ──────────>     bag-level multi-labels
                                                    ∟ feature extractor
                                                        ∟ MI / ML / MIML method
```

A bag is one pre-split WSI, its instances are the patches cut out of it, and its labels are the classes the annotator
decided are present. Nobody tells us which patch carries which class, and this is what turns the problem into MIML
instead of plain multi-label classification.

## Dataset expectations

`DiffInfiniteDataset` in `utils.py` reads a directory laid out like the preprocessing output:

```
datasets/diffinfinite_tr1/
│
∟ ─ annotator.csv
    patched_presplit_out/
    │
    ∟ ─ patches/
        │
        ∟ ─ {sample_id}/
            │
            ∟ ─ {sample_id}_x_{x_coord}_y_{y_coord}.jpg
```

> Note: the masks folder produced by the preprocessing step is not read here, we only need the patches plus the csv

Bags are matched to their labels through `sample_id`, so the folder name has to be the same id used in the csv. Both
label conventions of the annotator are available:

| `use_strategy` | columns read | meaning |
|---|---|---|
| `True` (default) | `Unknown`, `Carcinoma`, ... | labels after the annotator strategy (TopK, TopKThr, ...) |
| `False` | `ABS_Unknown`, `ABS_Carcinoma`, ... | every class physically present in the mask, no filtering |

The `ABS_` columns are much denser, since nearly every mask contains a few pixels of nearly every class, so training on
them means training on almost-all-ones targets. This is why the strategy columns are the default.

## Feature extraction

Patches get encoded before any method sees them. Extractors live behind a common `FeatureExtractor` interface and are
built by `extractor_factory.py`, so swapping one for another is a single line in the config.

| `using_extractor` | backbone | pretraining | `dimensionality` |
|---|---|---|---|
| `dino` | ViT-S/16, ViT-B/16, ResNet50 | DINO self-distillation, optionally histology weights | 384 / 768 / 2048 |
| `simclr_v2` | ResNet50, ResNet101 | SimCLR v2 contrastive | 2048 |
| `plip` | CLIP ViT-B/32 | pathology image and text pairs | 512 |

With `custom_weights: True` the DINO extractor pulls the DAS-MIL histology checkpoints instead of the ImageNet ones,
picked by `dataset_name` and `scale_level`.

> Every `compute_features` returns one tensor of shape `(n_instances, dimensionality)`, keep that contract if you add a
> new extractor

## Methods

Three families, all reachable through the same `Baseline.run(trainset, testset)` entry point and built by
`miml_factory.py`:

| `method` | family | what it does |
|---|---|---|
| `dsmil` | MIML | dual-stream attention MIL, our reimplementation |
| `og_dsmil` | MIML | the original DSMIL, adapted to our bags |
| `fastmiml` | MIML | shared subspace with sub concepts, SGD on label ranking |
| `misvm` | MI | MI-SVM over binary relevance decomposed bags |
| `milr` | MI | multi-instance logistic regression, several bag aggregation functions |
| `cknn` | MI | Citation-kNN with minimal Hausdorff bag distance |
| `mlknn` | ML | ML-kNN, bayesian on the label counts of the k neighbours |
| `brknn` | ML | binary relevance kNN, both the `a` and `b` variants |

The MI methods cannot eat a multi-label bag directly, so `binary_relevance_transformation` in `utils.py` splits each bag
into one binary problem per class first.

> Only `dsmil`, `og_dsmil` and `mlknn` are wired to the feature extractor at the moment, the others work on the
> flattened bags they receive

## Configuration

Everything is driven by `config.yml`, there are no command line flags apart from the config path:

```yaml
using_extractor: "dino"
method: "mlknn"
trainset_path: "datasets/diffinfinite_tr1/"
testset_path: "datasets/diffinfinite_tr2/"
checkpoint_path: "checkpoints/"
```

Per-extractor settings go under `extractors`, per-method hyperparameters under `miml_methods`. Only the block of the
active extractor and of the active method gets read, so the others can stay as they are.

## Running

```bash
conda env create -f env.yml
conda activate deepmiml
python main.py --config_path config.yml
```

`main.py` builds the extractor, builds the method, wraps them in a `MILWrapper` and hands it the two dataset paths. The
device is picked automatically between cuda, mps and cpu.

## Extending

#### Adding a method

Subclass `Baseline`, implement `run(trainset, testset)`, register it in the `method_factory` dict of `miml_factory.py`
and add its hyperparameters to `miml_methods` in the config. If it needs features instead of raw bags, add its name to
the branch in `mil_wrapper.py` so the extractor gets passed along.

#### Adding an extractor

Subclass `FeatureExtractor`, implement `_load_weights` and `compute_features`, register it in `ext_factory` in
`extractor_factory.py` and describe its versions under `extractors` in the config.

## Environment notes

`brknn` depends on `scikit-multilearn`, which has not been released since 2018 and calls `NearestNeighbors(self.k)`
positionally. Recent scikit-learn made that argument keyword only, so the classifier fails to fit. Either pin
`scikit-learn<1.3` or move to the maintained `scikit-multilearn-ng` fork.

# Credits

The synthetic dataset has been obtained through diffinfinite, Aversa et al., *DiffInfinite: Large Mask-Image Synthesis
via Parallel Random Patch Diffusion in Histopathology*, NeurIPS 2023,
[paper](https://arxiv.org/abs/2306.13384)

The dual-stream MIL work is based on DSMIL, Li, Li and Eliceiri, *Dual-stream Multiple Instance Learning Network for
Whole Slide Image Classification with Self-supervised Contrastive Learning*, CVPR 2021,
[code](https://github.com/binli123/dsmil-wsi). `og_dsmil.py` is their model, `train_diffinfinite.py` is their training
loop adapted to our bags

DINO, Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021,
[code](https://github.com/facebookresearch/dino). The histology checkpoints are the ones released with DAS-MIL,
Bontempo et al., *DAS-MIL: Distilling Across Scales for MIL Classification of Histological WSIs*, MICCAI 2023,
[code](https://github.com/aimagelab/mil4wsi)

SimCLR v2, Chen et al., *Big Self-Supervised Models are Strong Semi-Supervised Learners*, NeurIPS 2020, through the
PyTorch conversion by Separius, [code](https://github.com/Separius/SimCLRv2-Pytorch)

PLIP, Huang et al., *A visual-language foundation model for pathology image analysis using medical Twitter*, Nature
Medicine 2023, [model](https://huggingface.co/vinid/plip)

MI-SVM and mi-SVM, Andrews, Tsochantaridis and Hofmann, *Support Vector Machines for Multiple-Instance Learning*,
NeurIPS 2002, through the `misvm` package by Gary Doran, [code](https://github.com/garydoranjr/misvm)

Citation-kNN, Wang and Zucker, *Solving the Multiple-Instance Problem: A Lazy Learning Approach*, ICML 2000

ML-kNN, Zhang and Zhou, *ML-KNN: A Lazy Learning Approach to Multi-Label Learning*, Pattern Recognition 2007

BRkNN, Spyromitros, Tsoumakas and Vlahavas, *An Empirical Study of Lazy Multilabel Classification Algorithms*, SETN
2008, through scikit-multilearn by Szymański and Kajdanowicz, [code](http://scikit.ml)

MIMLfast, Huang, Gao and Zhou, *Fast Multi-Instance Multi-Label Learning*, AAAI 2014, ported from the original MATLAB
implementation

The multi-instance logistic regression follows the bag aggregation functions discussed in Xu and Frank, *Logistic
Regression and Boosting for Labeled Bags of Instances*, PAKDD 2004, and Ray and Craven, *Supervised Versus Multiple
Instance Learning: An Empirical Comparison*, ICML 2005
