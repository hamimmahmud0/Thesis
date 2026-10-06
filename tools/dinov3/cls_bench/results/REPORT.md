# DINOv3 frozen-backbone classifiers on the annotated traffic crops

## Setup

- **Data**: 1812 hand-reviewed crops from `class_annotator_from_coco/cls_dataset` (decisions.json sha256 `e796a0b72197…`), 14 classes. Split **1252 train / 270 val / 290 test**, grouped by source image (90/18/16 images) and stratified by class, so no frame is in two splits.
- **Label changes**: `Autorickshaw` merged into `Rickshaw` (63 + 182). Dropped (too few to split): `Pushcart` (1), `police_car` (3). `not_an_object` is kept as a real (negative) class.
- **Backbones**: frozen (never trained). Features: CLS token, or CLS + mean of patch tokens (register tokens excluded); crops either stretched to 224×224 (model's own processor) or padded to a square first.
- **Protocol**: variant + hyper-parameters chosen on **val macro-F1**; refit on train; **test scored once**. Because test is small, the same selection is also scored with **5-fold grouped CV over all crops** (mean ± std); CV uses val-selected settings, so it is slightly optimistic but far less noisy.
- **MLP head** (pre-registered): only run for a backbone whose best classical classifier has val macro-F1 < 0.85. LoRA was not run (it would train the backbone; excluded by the brief).
- Primary metric: **macro-F1** (class sizes range from 18 to 455).

### Split sizes per class

| class | train | val | test |
|---|---|---|---|
| Bicycle | 24 | 5 | 3 |
| Bus | 41 | 9 | 9 |
| CNG | 34 | 6 | 2 |
| Car | 101 | 16 | 9 |
| Cyclevan | 26 | 3 | 3 |
| Leguna | 39 | 2 | 5 |
| Motorcycle | 220 | 37 | 43 |
| Pedestrian | 81 | 56 | 26 |
| Rickshaw | 176 | 30 | 39 |
| Truck | 77 | 16 | 18 |
| Van | 85 | 11 | 15 |
| ambulance | 47 | 14 | 11 |
| not_an_object | 287 | 63 | 105 |
| pickup | 14 | 2 | 2 |

With only 2–9 test crops in many classes, **differences of 1–3 points between combinations are within noise**; use the CV column to rank.

## Backbones

| model | HF id | params | dim | extraction (1812 crops × 2 preproc.) |
|---|---|---|---|---|
| ViT-B/16 LVD-1689M | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 0.09 B | 768 | 48s (float32) |
| ViT-L/16 LVD-1689M | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 0.30 B | 1024 | 182s (float32) |
| ViT-L/16 SAT-493M | `facebook/dinov3-vitl16-pretrain-sat493m` | 0.30 B | 1024 | 180s (float32) |
| ViT-7B/16 LVD-1689M | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | 6.72 B | 4096 | 584s (float16) |
| ViT-7B/16 SAT-493M | `facebook/dinov3-vit7b16-pretrain-sat493m` | 6.72 B | 4096 | 579s (float16) |

## Results: every model × classifier (test)

Each cell: **test macro-F1** / test accuracy / CV macro-F1 (mean ± std). Variant = chosen feature/preprocessing.

| classifier | ViT-B/16 LVD-1689M | ViT-L/16 LVD-1689M | ViT-L/16 SAT-493M | ViT-7B/16 LVD-1689M | ViT-7B/16 SAT-493M |
|---|---|---|---|---|---|
| LogReg | **0.644** / 0.745 / 0.650±0.051<br><sub>cls/pad</sub> | **0.694** / 0.741 / 0.681±0.033<br><sub>cls+mean/stretch</sub> | **0.513** / 0.583 / 0.494±0.059<br><sub>cls+mean/stretch</sub> | **0.758** / 0.797 / 0.759±0.034<br><sub>cls+mean/stretch</sub> | **0.592** / 0.662 / 0.537±0.035<br><sub>cls+mean/stretch</sub> |
| LinearSVC | **0.653** / 0.748 / 0.667±0.049<br><sub>cls+mean/pad</sub> | **0.713** / 0.714 / 0.683±0.027<br><sub>cls+mean/pad</sub> | **0.488** / 0.614 / 0.527±0.036<br><sub>cls+mean/stretch</sub> | **0.734** / 0.783 / 0.717±0.022<br><sub>cls+mean/stretch</sub> | **0.586** / 0.634 / 0.538±0.048<br><sub>cls+mean/pad</sub> |
| kNN-cosine | **0.473** / 0.631 / 0.511±0.026<br><sub>cls+mean/pad</sub> | **0.593** / 0.655 / 0.582±0.031<br><sub>cls+mean/pad</sub> | **0.348** / 0.507 / 0.314±0.027<br><sub>cls/pad</sub> | **0.605** / 0.690 / 0.630±0.026<br><sub>cls+mean/pad</sub> | **0.304** / 0.455 / 0.304±0.025<br><sub>cls/pad</sub> |
| NearestCentroid | **0.503** / 0.545 / 0.535±0.040<br><sub>cls/pad</sub> | **0.551** / 0.572 / 0.586±0.034<br><sub>cls/pad</sub> | **0.327** / 0.355 / 0.313±0.023<br><sub>cls+mean/stretch</sub> | **0.615** / 0.652 / 0.662±0.030<br><sub>cls+mean/pad</sub> | **0.387** / 0.417 / 0.342±0.023<br><sub>cls+mean/stretch</sub> |
| RBF-SVM | **0.605** / 0.676 / 0.632±0.023<br><sub>cls/pad</sub> | **0.738** / 0.752 / 0.675±0.037<br><sub>cls+mean/pad</sub> | **0.442** / 0.597 / 0.503±0.020<br><sub>cls+mean/stretch</sub> | **0.735** / 0.776 / 0.753±0.047<br><sub>cls+mean/stretch</sub> | **0.525** / 0.548 / 0.514±0.017<br><sub>cls+mean/stretch</sub> |
| MLP | **0.608** / 0.724 / 0.614±0.035<br><sub>cls/stretch</sub> | **0.665** / 0.717 / 0.656±0.051<br><sub>cls+mean/pad</sub> | **0.425** / 0.590 / 0.454±0.028<br><sub>cls+mean/stretch</sub> | **0.720** / 0.772 / 0.687±0.043<br><sub>cls+mean/stretch</sub> | **0.502** / 0.600 / 0.465±0.072<br><sub>cls+mean/stretch</sub> |

![heatmap](heatmap.png)

## Best per model (ranked by CV macro-F1)

| model | best classifier | variant | hyper-params | val F1 | test macro-F1 | test acc | CV macro-F1 |
|---|---|---|---|---|---|---|---|
| ViT-B/16 LVD-1689M | LinearSVC | cls+mean/pad | `{'C': 0.001}` | 0.687 | 0.653 | 0.748 | 0.667±0.049 |
| ViT-L/16 LVD-1689M | LinearSVC | cls+mean/pad | `{'C': 0.001}` | 0.738 | 0.713 | 0.714 | 0.683±0.027 |
| ViT-L/16 SAT-493M | LinearSVC | cls+mean/stretch | `{'C': 0.001}` | 0.513 | 0.488 | 0.614 | 0.527±0.036 |
| ViT-7B/16 LVD-1689M | LogReg | cls+mean/stretch | `{'C': 0.01}` | 0.772 | 0.758 | 0.797 | 0.759±0.034 |
| ViT-7B/16 SAT-493M | LinearSVC | cls+mean/pad | `{'C': 0.001}` | 0.526 | 0.586 | 0.634 | 0.538±0.048 |

## Overall best: ViT-7B/16 LVD-1689M + LogReg

| class | test F1 | n test |
|---|---|---|
| Bicycle | 0.80 | 3 |
| Bus | 0.90 | 9 |
| CNG | 1.00 | 2 |
| Car | 0.78 | 9 |
| Cyclevan | 0.33 | 3 |
| Leguna | 0.55 | 5 |
| Motorcycle | 0.79 | 43 |
| Pedestrian | 0.67 | 26 |
| Rickshaw | 0.79 | 39 |
| Truck | 0.76 | 18 |
| Van | 0.93 | 15 |
| ambulance | 1.00 | 11 |
| not_an_object | 0.81 | 105 |
| pickup | 0.50 | 2 |

![confusion](confusion_best.png)

## Ablation: feature / preprocessing variant (val macro-F1 of the best hyper-parameters)

| model | classifier | cls+mean/pad | cls+mean/stretch | cls/pad | cls/stretch |
|---|---|---|---|---|---|
| ViT-B/16 LVD-1689M | LinearSVC | 0.687 | 0.666 | 0.684 | 0.669 |
| ViT-L/16 LVD-1689M | LinearSVC | 0.738 | 0.718 | 0.711 | 0.705 |
| ViT-L/16 SAT-493M | LinearSVC | 0.481 | 0.513 | 0.467 | 0.472 |
| ViT-7B/16 LVD-1689M | LogReg | 0.726 | 0.772 | 0.732 | 0.719 |
| ViT-7B/16 SAT-493M | LinearSVC | 0.526 | 0.505 | 0.504 | 0.486 |

## MLP head

- ViT-B/16 LVD-1689M: best classical val macro-F1 0.687 → MLP run
- ViT-L/16 LVD-1689M: best classical val macro-F1 0.739 → MLP run
- ViT-L/16 SAT-493M: best classical val macro-F1 0.513 → MLP run
- ViT-7B/16 LVD-1689M: best classical val macro-F1 0.781 → MLP run
- ViT-7B/16 SAT-493M: best classical val macro-F1 0.526 → MLP run