"""Per-pair similarity (Rickshaw↔Rickshaw, Rickshaw↔Motorcycle) with crop snapshots."""
import csv, json
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
OUT = "outputs"
F = np.load(f"{OUT}/features.npy"); meta = json.load(open(f"{OUT}/samples.json"))
idx = lambda c: [i for i, m in enumerate(meta) if m["cat"] == c]
R, M = idx("Rickshaw"), idx("Motorcycle")
S = F @ F.T

def thumb(i, h=64):
    im = Image.open(f"{OUT}/crops/{meta[i]['tag']}.png").convert("RGB"); w = max(1, int(im.width * h / im.height))
    return im.resize((min(w, 128), h))

def heat(rows, cols, name, title):
    A = S[np.ix_(rows, cols)]; n, m = A.shape
    fig = plt.figure(figsize=(1.0 * m + 3, 1.0 * n + 3)); ax = fig.add_axes([0.14, 0.04, 0.82, 0.80])
    ax.imshow(A, cmap="viridis", vmin=0, vmax=1, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for i in range(n):
        for j in range(m):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=7, color="w" if A[i, j] < .6 else "k")
    W, H = fig.get_size_inches() * fig.dpi
    for j, c in enumerate(cols):   # column snapshots on top
        t = ax.inset_axes([j / m, 1.01, 1 / m, 0.9 * 1.0 / n * 1.6], transform=ax.transAxes); t.imshow(thumb(c)); t.axis("off")
        t.set_title(meta[c]["tag"].split("_")[1], fontsize=7, pad=1)
    for i, r in enumerate(rows):   # row snapshots on left
        t = ax.inset_axes([-0.12, 1 - (i + 1) / n, 0.11, 1 / n], transform=ax.transAxes); t.imshow(thumb(r)); t.axis("off")
        ax.text(-0.125, 1 - (i + .5) / n, meta[r]["tag"].split("_")[1], transform=ax.transAxes, ha="right", va="center", fontsize=7)
    fig.suptitle(title, y=0.995); fig.savefig(f"{OUT}/{name}.png", dpi=110); plt.close(fig)

heat(R, R, "pairs_rickshaw_rickshaw", "Rickshaw ↔ Rickshaw (cosine)")
heat(R, M, "pairs_rickshaw_motorcycle", "Rickshaw (rows) ↔ Motorcycle (cols) (cosine)")

with open(f"{OUT}/pairs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "b", "type", "cosine", "a_file", "b_file", "a_score", "b_score"])
    for i in R:
        for j in R + M:
            if j <= i and j in R: continue
            w.writerow([meta[i]["tag"], meta[j]["tag"], "R-R" if j in R else "R-M", f"{S[i,j]:.4f}",
                        meta[i]["file"], meta[j]["file"], meta[i]["score"], meta[j]["score"]])

rr = np.array([S[i, j] for a, i in enumerate(R) for j in R[a + 1:]]); rm = S[np.ix_(R, M)]
print(f"R-R: mean {rr.mean():.3f} min {rr.min():.3f} max {rr.max():.3f}")
print(f"R-M: mean {rm.mean():.3f} min {rm.min():.3f} max {rm.max():.3f}")
# per-rickshaw: how much more like other rickshaws than motorcycles
for i in R:
    a = np.mean([S[i, j] for j in R if j != i]); b = S[i, M].mean()
    print(f"{meta[i]['tag']}: R {a:.3f}  M {b:.3f}  {'<-- closer to Motorcycle' if b > a else ''}")
