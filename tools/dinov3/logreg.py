"""Rickshaw vs Motorcycle logistic regression on DINOv3 embeddings; test on fresh crops."""
import json
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
O = "outputs"; CL = ["Rickshaw", "Motorcycle"]
F = np.load(f"{O}/features.npy"); M = json.load(open(f"{O}/samples.json"))
keep = [i for i, m in enumerate(M) if m["cat"] in CL]
Xtr = F[keep]; ytr = np.array([CL.index(M[i]["cat"]) for i in keep])
Xte = np.load(f"{O}/test_features.npy"); Mte = json.load(open(f"{O}/test_samples.json"))
yte = np.array([CL.index(m["cat"]) for m in Mte])

gs = GridSearchCV(LogisticRegression(max_iter=5000), {"C": [0.1, 1, 10, 100, 1000]},
                  cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy").fit(Xtr, ytr)
clf = gs.best_estimator_
print(f"train n={len(ytr)} | best C={gs.best_params_['C']} | 5-fold CV acc={gs.best_score_:.3f} | train acc={clf.score(Xtr, ytr):.3f}")
p = clf.predict_proba(Xte)[:, 1]; pred = (p >= .5).astype(int)
print(f"TEST n={len(yte)} | acc={np.mean(pred == yte):.3f} | AUC={roc_auc_score(yte, p):.3f}")
cm = confusion_matrix(yte, pred, labels=[0, 1]); print("confusion (rows=true, cols=pred) [Rickshaw, Motorcycle]\n", cm)
rows = []
for m, t, pr, pm in zip(Mte, yte, pred, p):
    ok = "OK " if t == pr else "ERR"; print(f"{ok} {m['tag']:<22} true={CL[t]:<10} pred={CL[pr]:<10} P(motorcycle)={pm:.3f}  {m['file']}")
    rows.append((m, t, pr, pm))
np.save(f"{O}/logreg_test_probs.npy", p)

fig, axs = plt.subplots(4, 5, figsize=(12, 10.5))
for ax, (m, t, pr, pm) in zip(axs.ravel(), sorted(rows, key=lambda r: (r[1], r[3]))):
    ax.imshow(Image.open(f"{O}/crops/{m['tag']}.png")); ax.axis("off")
    ax.set_title(f"true {CL[t]}\npred {CL[pr]}  P(M)={pm:.2f}", fontsize=8, color="g" if t == pr else "r")
fig.suptitle(f"Held-out test (green=correct): acc {np.mean(pred==yte):.0%}, AUC {roc_auc_score(yte,p):.2f}"); plt.tight_layout()
plt.savefig(f"{O}/logreg_test.png", dpi=110)
