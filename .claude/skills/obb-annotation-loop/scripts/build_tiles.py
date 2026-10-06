"""Build a tiled YOLO-OBB dataset from b1_small_annotated with a frame-block split.
python build_tiles.py SRC_DIR OUT_DIR [--val DJI_0614,DJI_0615,DJI_0616] [--tile 1024] [--overlap 0.25] [--empty-keep 0.1]
SRC_DIR has images/<hash>__DJI_xxxx.JPG, labels/<same>.txt (cls x1 y1 .. x4 y4 normalized), classes.txt.
An object is kept in a tile if >= MIN_FRAC of its (parallelogram) area lies inside the tile; corners are clipped to the tile.
Objects with a smaller non-zero overlap are dropped (they are slivers)."""
import argparse, glob, json, os, random
import numpy as np
from PIL import Image

MIN_FRAC = 0.6
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('out')
ap.add_argument('--val', default='DJI_0614,DJI_0615,DJI_0616')
ap.add_argument('--tile', type=int, default=1024); ap.add_argument('--overlap', type=float, default=0.25)
ap.add_argument('--empty-keep', type=float, default=0.1); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)
val_ids = set(a.val.split(','))
cls = [l.strip() for l in open(f'{a.src}/classes.txt') if l.strip()]
stride = int(a.tile * (1 - a.overlap))
G = np.linspace(0.05, 0.95, 10)
UU, VV = np.meshgrid(G, G); UU, VV = UU.ravel(), VV.ravel()

def starts(n):
    if n <= a.tile: return [0]
    s = list(range(0, n - a.tile, stride)); s.append(n - a.tile); return s

def frac_inside(p, x0, y0, x1, y1):
    p = np.asarray(p); pts = p[0] + UU[:, None] * (p[1] - p[0]) + VV[:, None] * (p[3] - p[0])
    return float(((pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)).mean())

stats = {'train': 0, 'val': 0}; inst = {'train': {}, 'val': {}}; split_files = {'train': [], 'val': []}
for sp in ('train', 'val'):
    os.makedirs(f'{a.out}/images/{sp}', exist_ok=True); os.makedirs(f'{a.out}/labels/{sp}', exist_ok=True)
for f in sorted(glob.glob(f'{a.src}/images/*.JPG')):
    stem = os.path.splitext(os.path.basename(f))[0]; dji = stem.split('__')[-1]; sp = 'val' if dji in val_ids else 'train'
    split_files[sp].append(dji)
    im = Image.open(f).convert('RGB'); W, H = im.size; objs = []
    for l in open(f'{a.src}/labels/{stem}.txt'):
        v = l.split()
        if len(v) < 9: continue
        objs.append((int(v[0]), [(float(v[1 + 2 * j]) * W, float(v[2 + 2 * j]) * H) for j in range(4)]))
    for y0 in starts(H):
        for x0 in starts(W):
            x1, y1 = min(x0 + a.tile, W), min(y0 + a.tile, H); lines = []
            for c, p in objs:
                fr = frac_inside(p, x0, y0, x1, y1)
                if fr < MIN_FRAC: continue
                q = [(min(max(x - x0, 0), x1 - x0) / a.tile, min(max(y - y0, 0), y1 - y0) / a.tile) for x, y in p]
                lines.append(f"{c} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in q)); inst[sp][cls[c]] = inst[sp].get(cls[c], 0) + 1
            if not lines and (sp == 'val' or random.random() > a.empty_keep): continue
            name = f'{dji}_{x0}_{y0}'
            im.crop((x0, y0, x1, y1)).save(f'{a.out}/images/{sp}/{name}.jpg', quality=95)
            open(f'{a.out}/labels/{sp}/{name}.txt', 'w').write("\n".join(lines) + ("\n" if lines else "")); stats[sp] += 1
open(f'{a.out}/data.yaml', 'w').write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(cls)))
json.dump({'tiles': stats, 'frames': split_files, 'tile_instances': inst}, open(f'{a.out}/split.json', 'w'), indent=1)
print('tiles', stats); print('frames', {k: len(v) for k, v in split_files.items()})
for sp in ('train', 'val'): print(sp, 'instances/class (tile-level, overlap duplicates):', inst[sp])
