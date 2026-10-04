"""Old LR (trained on the original 40 crops) vs new LR (trained on curated set); confusion matrices."""
import json
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
O = "outputs"; CL = ["Rickshaw", "Motorcycle"]
def load(n):
    m = json.load(open(f"{O}/final/{n}_meta.json")); return np.load(f"{O}/final/{n}_features.npy"), np.array([CL.index(s["cat"]) for s in m]), m
def fit(X, y):
    g = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced"), {"C": [0.1, 1, 10, 100, 1000]},
                     cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy").fit(X, y)
    return g.best_estimator_, g.best_params_["C"], g.best_score_
F = np.load(f"{O}/features.npy"); M = json.load(open(f"{O}/samples.json"))
keep = [i for i, m in enumerate(M) if m["cat"] in CL]
old, oldC, oldcv = fit(F[keep], np.array([CL.index(M[i]["cat"]) for i in keep]))
Xc, yc, mc = load("curated"); Xr, yr, mr = load("random")
new, newC, newcv = fit(Xc, yc)
out = {"old_model": {"train_n": len(keep), "C": oldC, "cv_acc": oldcv}, "new_model": {"train_n": len(yc), "C": newC, "cv_acc": newcv}}
fig, axs = plt.subplots(2, 2, figsize=(9, 8))
for r, (name, mdl) in enumerate((("old (40 crops)", old), ("new (curated)", new))):
    for c, (dn, X, y) in enumerate((("curated set", Xc, yc), ("1000/class random", Xr, yr))):
        if name.startswith("new") and dn == "curated set": axs[r, c].axis("off"); axs[r, c].text(.5, .5, f"training data\n5-fold CV acc {newcv:.3f}", ha="center", va="center"); continue
        p = mdl.predict_proba(X)[:, 1]; pr = (p >= .5).astype(int); cm = confusion_matrix(y, pr, labels=[0, 1])
        acc = (pr == y).mean(); bacc = np.mean([cm[0, 0] / cm[0].sum(), cm[1, 1] / cm[1].sum()])
        out[f"{name.split()[0]}_on_{'curated' if c == 0 else 'random'}"] = {"acc": acc, "balanced_acc": bacc, "auc": roc_auc_score(y, p), "f1_motorcycle": f1_score(y, pr), "confusion": cm.tolist()}
        ax = axs[r, c]; ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/cm[i].sum():.0%})", ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
        ax.set_xticks([0, 1], CL); ax.set_yticks([0, 1], CL); ax.set_xlabel("pred"); ax.set_ylabel("true")
        ax.set_title(f"{name} on {dn}\nacc {acc:.3f}  bal-acc {bacc:.3f}  AUC {roc_auc_score(y, p):.3f}", fontsize=9)
plt.tight_layout(); plt.savefig(f"{O}/final/confusion.png", dpi=130)
json.dump(out, open(f"{O}/final/results.json", "w"), indent=1); print(json.dumps(out, indent=1))
