"""Accuracy of old/new LR on the random 2000 by detection score and by box size."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
O = "outputs"; CL = ["Rickshaw", "Motorcycle"]
F = np.load(f"{O}/features.npy"); M = json.load(open(f"{O}/samples.json")); keep = [i for i, m in enumerate(M) if m["cat"] in CL]
old = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1).fit(F[keep], [CL.index(M[i]["cat"]) for i in keep])
mc = json.load(open(f"{O}/final/curated_meta.json")); Xc = np.load(f"{O}/final/curated_features.npy"); yc = [CL.index(s["cat"]) for s in mc]
new = LogisticRegression(max_iter=5000, class_weight="balanced", C=1).fit(Xc, yc)
mr = json.load(open(f"{O}/final/random_meta.json")); Xr = np.load(f"{O}/final/random_features.npy"); yr = np.array([CL.index(s["cat"]) for s in mr])
sc = np.array([s["score"] for s in mr]); area = np.array([s["bbox"][2] * s["bbox"][3] for s in mr])
def rep(title, masks):
    print(title)
    for name, m in masks:
        if m.sum() < 5: continue
        po, pn = old.predict(Xr[m]), new.predict(Xr[m])
        print(f"  {name:<14} n={m.sum():4d} (R {int((yr[m]==0).sum())}/M {int((yr[m]==1).sum())})  old acc {np.mean(po==yr[m]):.3f}  new acc {np.mean(pn==yr[m]):.3f}")
rep("by score", [(f"{a:.1f}-{b:.1f}", (sc >= a) & (sc < b)) for a, b in ((0, .5), (.5, .7), (.7, .85), (.85, .9), (.9, 1.01))])
q = np.quantile(area, [0, .25, .5, .75, 1.0])
rep("by box area quartile", [(f"{int(q[i])}-{int(q[i+1])}px2", (area >= q[i]) & (area <= q[i + 1])) for i in range(4)])
print("score quantiles R", np.quantile(sc[yr == 0], [.1, .5, .9]).round(2), "M", np.quantile(sc[yr == 1], [.1, .5, .9]).round(2))
