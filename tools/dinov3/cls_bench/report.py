#!/usr/bin/env python3
"""results.json (+ dataset meta.json) -> REPORT.md and PNG figures."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["vitb16_lvd", "vitl16_lvd", "vitl16_sat", "vit7b16_lvd", "vit7b16_sat"]
NAMES = {"vitb16_lvd": "ViT-B/16 LVD-1689M", "vitl16_lvd": "ViT-L/16 LVD-1689M", "vitl16_sat": "ViT-L/16 SAT-493M",
         "vit7b16_lvd": "ViT-7B/16 LVD-1689M", "vit7b16_sat": "ViT-7B/16 SAT-493M"}
CLFS = ["LogReg", "LinearSVC", "kNN-cosine", "NearestCentroid", "RBF-SVM", "MLP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlp-threshold", type=float, default=0.85)
    a = ap.parse_args()
    R = json.load(open(a.results)); M = json.load(open(a.meta)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = [m for m in ORDER if m in R]
    L = []
    w = L.append
    w("# DINOv3 frozen-backbone classifiers on the annotated traffic crops\n")
    w("## Setup\n")
    t = M["total"]
    w(f"- **Data**: {sum(t.values())} hand-reviewed crops from `class_annotator_from_coco/cls_dataset` "
      f"(decisions.json sha256 `{M['decisions_sha256'][:12]}…`), {len(M['per_class'])} classes. "
      f"Split **{t['train']} train / {t['val']} val / {t['test']} test**, grouped by source image "
      f"({M['groups']['train']}/{M['groups']['val']}/{M['groups']['test']} images) and stratified by class, so no frame is in two splits.")
    w(f"- **Label changes**: `Autorickshaw` merged into `Rickshaw` (63 + 182). Dropped (too few to split): "
      + ", ".join(f"`{k}` ({v})" for k, v in M["dropped_min_count"].items()) + ". `not_an_object` is kept as a real (negative) class.")
    w("- **Backbones**: frozen (never trained). Features: CLS token, or CLS + mean of patch tokens (register tokens excluded); "
      "crops either stretched to 224×224 (model's own processor) or padded to a square first.")
    w("- **Protocol**: variant + hyper-parameters chosen on **val macro-F1**; refit on train; **test scored once**. "
      "Because test is small, the same selection is also scored with **5-fold grouped CV over all crops** (mean ± std); "
      "CV uses val-selected settings, so it is slightly optimistic but far less noisy.")
    w(f"- **MLP head** (pre-registered): only run for a backbone whose best classical classifier has val macro-F1 < {a.mlp_threshold}. "
      "LoRA was not run (it would train the backbone; excluded by the brief).")
    w("- Primary metric: **macro-F1** (class sizes range from 18 to 455).\n")
    w("### Split sizes per class\n")
    w("| class | train | val | test |\n|---|---|---|---|")
    for c, d in M["per_class"].items():
        w(f"| {c} | {d['train']} | {d['val']} | {d['test']} |")
    w("\nWith only 2–9 test crops in many classes, **differences of 1–3 points between combinations are within noise**; "
      "use the CV column to rank.\n")
    w("## Backbones\n")
    w("| model | HF id | params | dim | extraction (1812 crops × 2 preproc.) |\n|---|---|---|---|---|")
    for m in models:
        i = R[m]["info"]
        w(f"| {NAMES[m]} | `{i['model']}` | {i['params_b']:.2f} B | {i['dim']} | {i['extract_s']:.0f}s ({i['dtype']}) |")
    w("")
    w("## Results: every model × classifier (test)\n")
    w("Each cell: **test macro-F1** / test accuracy / CV macro-F1 (mean ± std). Variant = chosen feature/preprocessing.\n")
    w("| classifier | " + " | ".join(NAMES[m] for m in models) + " |\n|---|" + "---|" * len(models))
    heat = np.full((len(CLFS), len(models)), np.nan)
    for i, c in enumerate(CLFS):
        if not any(c in R[m]["classifiers"] for m in models):
            continue
        cells = []
        for j, m in enumerate(models):
            r = R[m]["classifiers"].get(c)
            if r is None:
                cells.append("n/a (not needed)"); continue
            heat[i, j] = r["test_macro_f1"]
            cells.append(f"**{r['test_macro_f1']:.3f}** / {r['test_acc']:.3f} / {r['cv_macro_f1_mean']:.3f}±{r['cv_macro_f1_std']:.3f}<br><sub>{r['variant']}</sub>")
        w(f"| {c} | " + " | ".join(cells) + " |")
    w("")
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4))
    im = ax.imshow(heat, vmin=np.nanmin(heat) - 0.02, vmax=np.nanmax(heat) + 0.01, cmap="viridis")
    ax.set_xticks(range(len(models)), [NAMES[m].replace(" ", "\n", 1) for m in models], fontsize=8)
    ax.set_yticks(range(len(CLFS)), CLFS)
    for (i, j), v in np.ndenumerate(heat):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_title("Test macro-F1"); plt.colorbar(im); plt.tight_layout(); plt.savefig(out / "heatmap.png", dpi=140); plt.close()
    w("![heatmap](heatmap.png)\n")
    w("## Best per model (ranked by CV macro-F1)\n")
    w("| model | best classifier | variant | hyper-params | val F1 | test macro-F1 | test acc | CV macro-F1 |\n|---|---|---|---|---|---|---|---|")
    best = {}
    for m in models:
        c, r = max(R[m]["classifiers"].items(), key=lambda kv: kv[1]["cv_macro_f1_mean"])
        best[m] = (c, r)
        w(f"| {NAMES[m]} | {c} | {r['variant']} | `{r['hp']}` | {r['val_macro_f1']:.3f} | {r['test_macro_f1']:.3f} | {r['test_acc']:.3f} | {r['cv_macro_f1_mean']:.3f}±{r['cv_macro_f1_std']:.3f} |")
    w("")
    om = max(best, key=lambda m: best[m][1]["cv_macro_f1_mean"]); oc, orr = best[om]
    w(f"## Overall best: {NAMES[om]} + {oc}\n")
    classes = R[om]["classes"]
    w("| class | test F1 | n test |\n|---|---|---|")
    ntest = {c: R[om]["test_labels"].count(c) for c in classes}
    for c in classes:
        w(f"| {c} | {orr['per_class_f1'][c]:.2f} | {ntest[c]} |")
    cm = np.array(orr["confusion"]); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for (i, j), v in np.ndenumerate(cm):
        if v:
            ax.text(j, i, v, ha="center", va="center", fontsize=7, color="w" if cmn[i, j] > .5 else "k")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion (test counts): {NAMES[om]} + {oc}")
    plt.tight_layout(); plt.savefig(out / "confusion_best.png", dpi=140); plt.close()
    w("\n![confusion](confusion_best.png)\n")
    w("## Ablation: feature / preprocessing variant (val macro-F1 of the best hyper-parameters)\n")
    vk = sorted({v for m in models for c in R[m]["val_ablation"].values() for v in c})
    w("| model | classifier | " + " | ".join(vk) + " |\n|---|---|" + "---|" * len(vk))
    for m in models:
        c = best[m][0]
        ab = R[m]["val_ablation"][c]
        w(f"| {NAMES[m]} | {c} | " + " | ".join(f"{ab.get(v, float('nan')):.3f}" for v in vk) + " |")
    w("")
    w("## MLP head\n")
    for m in models:
        w(f"- {NAMES[m]}: best classical val macro-F1 {R[m]['best_classical_val']:.3f} → MLP {'run' if R[m]['mlp_run'] else 'skipped (threshold met)'}")
    (out / "REPORT.md").write_text("\n".join(L))
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
