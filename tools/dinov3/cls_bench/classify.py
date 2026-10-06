#!/usr/bin/env python3
"""Classifiers on frozen DINOv3 features (CPU). Reads feats_<model>.npz written by extract.py.

For every model x classifier the preprocessing / feature variant and the hyper-parameters are chosen on the VALIDATION
split (macro-F1), the classifier is refit on TRAIN and scored once on TEST. The same selection is then scored with
5-fold grouped, stratified CV over all crops (mean +- std macro-F1) because the test split is small and noisy.
Pre-registered rule: the MLP head is only run for a model whose best classical classifier has val macro-F1 < --mlp-threshold.
"""
import argparse, json, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

CLASSICAL = ["LogReg", "LinearSVC", "kNN-cosine", "NearestCentroid", "RBF-SVM"]
GRID = {
    "LogReg": [{"C": c} for c in (0.001, 0.01, 0.1, 1, 10)],
    "LinearSVC": [{"C": c} for c in (0.0001, 0.001, 0.01, 0.1, 1)],
    "kNN-cosine": [{"k": k} for k in (1, 3, 5, 10, 20)],
    "NearestCentroid": [{}],
    "RBF-SVM": [{"C": c, "g": g} for c in (1, 10, 100) for g in (0.5, 1, 2)],
    "MLP": [{"alpha": a, "h": h} for a in (1e-3, 1e-1) for h in (256, 1024)],
}


def make(name, hp, d):
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=3000))
    if name == "LinearSVC":
        return make_pipeline(StandardScaler(), LinearSVC(C=hp["C"], class_weight="balanced", dual="auto", max_iter=20000))
    if name == "kNN-cosine":
        return make_pipeline(Normalizer(), KNeighborsClassifier(hp["k"], metric="cosine", weights="distance"))
    if name == "NearestCentroid":
        return make_pipeline(Normalizer(), NearestCentroid())
    if name == "RBF-SVM":
        return make_pipeline(StandardScaler(), SVC(C=hp["C"], gamma=hp["g"] / d, class_weight="balanced"))
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier((hp["h"],), alpha=hp["alpha"], max_iter=400, early_stopping=True,
                                                            n_iter_no_change=15, random_state=0))
    raise ValueError(name)


def metrics(y, p, classes):
    return {"macro_f1": f1_score(y, p, labels=classes, average="macro", zero_division=0),
            "acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "weighted_f1": f1_score(y, p, labels=classes, average="weighted", zero_division=0)}


def variants(z):
    out = {}
    for pre in ("stretch", "pad"):
        out[f"cls/{pre}"] = z[f"cls_{pre}"]
        out[f"cls+mean/{pre}"] = np.hstack([z[f"cls_{pre}"], z[f"mean_{pre}"]])
    return out


def fit_val(name, hp, vkey, X, y, sp):
    warnings.filterwarnings("ignore")
    tr, va = sp == "train", sp == "val"
    m = make(name, hp, X.shape[1]).fit(X[tr], y[tr])
    return name, vkey, hp, f1_score(y[va], m.predict(X[va]), average="macro", zero_division=0)


def cv_scores(name, hp, X, y, g, seed=0):
    warnings.filterwarnings("ignore")
    f = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        p = make(name, hp, X.shape[1]).fit(X[tr], y[tr]).predict(X[te])
        f.append(f1_score(y[te], p, average="macro", zero_division=0))
    return f


def run_model(path, args):
    z = np.load(path, allow_pickle=True)
    info = json.loads(str(z["info"]))
    names_y, sp, g = z["label"].astype(str), z["split"].astype(str), z["group"].astype(str)
    classes_n = sorted(set(names_y))
    y = np.array([classes_n.index(v) for v in names_y])   # integer labels: sklearn's MLP early stopping chokes on strings
    classes = list(range(len(classes_n))); V = variants(z)
    tag = Path(path).stem.replace("feats_", "")
    t0 = time.time()

    def select(names):
        jobs = [(n, vk, hp) for n in names for vk in V for hp in GRID[n]]
        res = Parallel(n_jobs=args.jobs)(delayed(fit_val)(n, hp, vk, V[vk], y, sp) for n, vk, hp in jobs)
        best = {}
        for n, vk, hp, s in res:
            if n not in best or s > best[n][2]:
                best[n] = (vk, hp, s)
        return best, res

    best, allres = select(CLASSICAL)
    top_val = max(b[2] for b in best.values())
    mlp_run = top_val < args.mlp_threshold
    if mlp_run:
        b2, r2 = select(["MLP"]); best.update(b2); allres += r2

    def final(n):
        vk, hp, vs = best[n]
        X = V[vk]; tr, te = sp == "train", sp == "test"
        m = make(n, hp, X.shape[1]).fit(X[tr], y[tr]); p = m.predict(X[te])
        cv = cv_scores(n, hp, X, y, g)
        return n, {"variant": vk, "hp": hp, "val_macro_f1": vs, **{f"test_{k}": v for k, v in metrics(y[te], p, classes).items()},
                   "cv_macro_f1_mean": float(np.mean(cv)), "cv_macro_f1_std": float(np.std(cv)),
                   "per_class_f1": dict(zip(classes_n, f1_score(y[te], p, labels=classes, average=None, zero_division=0).tolist())),
                   "confusion": confusion_matrix(y[te], p, labels=classes).tolist(), "test_pred": [classes_n[i] for i in p]}

    out = dict(final(n) for n in best)
    # ablation: val macro-F1 of the best hp of every (classifier, variant)
    abl = {}
    for n, vk, hp, s in allres:
        abl.setdefault(n, {})[vk] = max(abl.get(n, {}).get(vk, 0), s)
    print(f"{tag}: done in {time.time() - t0:.0f}s, best classical val {top_val:.3f}, MLP {'run' if mlp_run else 'skipped'}", flush=True)
    return tag, {"info": info, "classes": classes_n, "test_labels": names_y[sp == "test"].tolist(), "classifiers": out,
                 "val_ablation": abl, "mlp_run": mlp_run, "best_classical_val": top_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats", nargs="+")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--mlp-threshold", type=float, default=0.85)
    args = ap.parse_args()
    res = dict(run_model(p, args) for p in args.feats)
    Path(args.out).write_text(json.dumps(res))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
