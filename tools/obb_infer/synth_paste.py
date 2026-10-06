"""Copy-paste synthetic OBB tiles.  python synth_paste.py SRC_DIR TILES_REAL OUT_DIR [--ratio 1.0] [--seed 0]
SRC_DIR: b1_small_annotated (full frames + labels + classes.txt).  TILES_REAL: output of build_tiles.py (split.json lists train frames).
Sources: OBB instances cropped from TRAIN frames only. Backgrounds: real labelled TRAIN tiles only (all real labels kept, so pasting never hides
an unlabelled object).  OUT_DIR = real train tiles (symlinked) + synthetic tiles, val symlinked unchanged.
Placement: near existing objects (traffic prior), long axis aligned with the nearest real vehicle (+-20 deg), scale 0.9-1.1, <10% overlap with
anything already in the tile, LAB colour matched to the surrounding ring, feathered alpha."""
import argparse, glob, json, math, os, random
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('tiles'); ap.add_argument('out')
ap.add_argument('--ratio', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--min-paste', type=int, default=5); ap.add_argument('--max-paste', type=int, default=10)
ap.add_argument('--reuse-cap', type=int, default=25); ap.add_argument('--tile', type=int, default=1024)
a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
cls = [l.strip() for l in open(f'{a.src}/classes.txt') if l.strip()]
split = json.load(open(f'{a.tiles}/split.json')); train_frames = set(split['frames']['train'])
PED_LIKE = {cls.index(n) for n in ('Pedestrian', 'Bicycle') if n in cls}

# ---- 1. source patches from train frames
patches = []  # dict(c, img(h,w,3), w, h)
for f in sorted(glob.glob(f'{a.src}/images/*.JPG')):
    stem = os.path.splitext(os.path.basename(f))[0]
    if stem.split('__')[-1] not in train_frames: continue
    im = cv2.imread(f); H, W = im.shape[:2]
    for l in open(f'{a.src}/labels/{stem}.txt'):
        v = l.split()
        if len(v) < 9: continue
        p = np.array([(float(v[1 + 2 * j]) * W, float(v[2 + 2 * j]) * H) for j in range(4)], np.float32)
        w = float(np.linalg.norm(p[1] - p[0])); h = float(np.linalg.norm(p[3] - p[0]))
        if w < 6 or h < 6: continue
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        M = cv2.getPerspectiveTransform(p, dst)
        patch = cv2.warpPerspective(im, M, (int(round(w)), int(round(h))), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        patches.append({'c': int(v[0]), 'img': patch, 'w': w, 'h': h, 'used': 0})
by_cls = {}
for i, p in enumerate(patches): by_cls.setdefault(p['c'], []).append(i)
counts = {c: len(v) for c, v in by_cls.items()}
weights = {c: (1.0 / n) ** 0.7 for c, n in counts.items()}
print('source patches per class:', {cls[c]: n for c, n in sorted(counts.items())})

def poly_of(cx, cy, w, h, phi):
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    return (np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T) + [cx, cy]

def road_mask(tile):
    # gray, mid-brightness, low-saturation pixels = asphalt; roofs (bright), facades (dark), vegetation (saturated) excluded
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV); m = ((hsv[..., 1] < 50) & (hsv[..., 2] > 70) & (hsv[..., 2] < 165)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m); keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 6000: keep[lab == i] = 1
    return keep > 0

def poly_road_frac(poly, road):
    pm = np.zeros(road.shape, np.uint8); cv2.fillPoly(pm, [poly.astype(np.int32)], 1); a_ = pm.sum()
    return 0.0 if a_ == 0 else float((road & (pm > 0)).sum()) / a_

def aabb(p): return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
def overlap_ratio(p, q):
    ax0, ay0, ax1, ay1 = aabb(p); bx0, by0, bx1, by1 = aabb(q)
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    return 0.0 if iw <= 0 or ih <= 0 else iw * ih / max((ax1 - ax0) * (ay1 - ay0), 1e-6)

def long_axis_angle(p):
    e1, e2 = p[1] - p[0], p[3] - p[0]; e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    return math.atan2(e[1], e[0])

def paste(tile, patch, cx, cy, phi, scale, existing_ring_mask):
    T = a.tile; w, h = patch['w'] * scale, patch['h'] * scale
    src = cv2.resize(patch['img'], (max(2, int(round(w))), max(2, int(round(h)))), interpolation=cv2.INTER_CUBIC)
    sh, sw = src.shape[:2]
    alpha = np.zeros((sh, sw), np.float32); m = max(1, int(0.04 * min(sh, sw)))
    alpha[m:sh - m, m:sw - m] = 1.0; alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.8, 0.06 * min(sh, sw)))
    ca, sa = math.cos(phi), math.sin(phi)
    Mat = np.array([[ca, -sa, cx - (ca * sw / 2 - sa * sh / 2)], [sa, ca, cy - (sa * sw / 2 + ca * sh / 2)]], np.float32)
    layer = cv2.warpAffine(src, Mat, (T, T), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    am = cv2.warpAffine(alpha, Mat, (T, T), flags=cv2.INTER_LINEAR, borderValue=0)
    inside = am > 0.5
    ring = cv2.dilate(inside.astype(np.uint8), np.ones((15, 15), np.uint8)) - cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8))
    ring = (ring > 0) & existing_ring_mask
    if ring.sum() > 50 and inside.sum() > 20:
        tl = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB).astype(np.float32); ll = cv2.cvtColor(layer, cv2.COLOR_BGR2LAB).astype(np.float32)
        mr, sr = tl[ring].mean(0), tl[ring].std(0) + 1e-3; mp, sp_ = ll[inside].mean(0), ll[inside].std(0) + 1e-3
        adj = (ll - mp) / sp_ * np.clip(sr, 0.7 * sp_, 1.3 * sp_) + mr
        ll = 0.6 * adj + 0.4 * ll  # partial transfer keeps the object's own colours
        layer = cv2.cvtColor(np.clip(ll, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    am3 = am[..., None]
    return (tile.astype(np.float32) * (1 - am3) + layer.astype(np.float32) * am3).astype(np.uint8)

# ---- 2. build output
os.makedirs(f'{a.out}/images/train', exist_ok=True); os.makedirs(f'{a.out}/labels/train', exist_ok=True)
for sp in ('val',):
    for sub in ('images', 'labels'):
        if not os.path.exists(f'{a.out}/{sub}/{sp}'): os.symlink(os.path.abspath(f'{a.tiles}/{sub}/{sp}'), f'{a.out}/{sub}/{sp}')
real = sorted(glob.glob(f'{a.tiles}/labels/train/*.txt'))
for f in real:
    n = os.path.basename(f)[:-4]
    for sub, ext in (('images', 'jpg'), ('labels', 'txt')):
        d = f'{a.out}/{sub}/train/{n}.{ext}'
        if not os.path.exists(d): os.symlink(os.path.abspath(f'{a.tiles}/{sub}/train/{n}.{ext}'), d)
bg = []
for f in real:  # tiles with traffic AND a usable road area
    n_ = os.path.basename(f)[:-4]; t_ = cv2.imread(f'{a.tiles}/images/train/{n_}.jpg')
    if sum(1 for l in open(f) if l.strip()) >= 3 and t_ is not None and road_mask(t_).mean() >= 0.15: bg.append(f)
print('background tiles with road:', len(bg), 'of', len(real))
n_syn = int(round(a.ratio * len(real))); added = {}; T = a.tile
for k in range(n_syn):
    f = random.choice(bg); n = os.path.basename(f)[:-4]; tile = cv2.imread(f'{a.tiles}/images/train/{n}.jpg'); th, tw = tile.shape[:2]
    objs = []
    for l in open(f):
        v = l.split()
        if len(v) >= 9: objs.append((int(v[0]), np.array([(float(v[1 + 2 * j]) * T, float(v[2 + 2 * j]) * T) for j in range(4)])))
    if tile.shape[0] != T or tile.shape[1] != T: continue
    lines = [open(f).read().strip()] if objs else []
    newobjs = []
    vehicles = [(c, p) for c, p in objs if c not in PED_LIKE] or objs
    road = road_mask(tile); ys, xs = np.nonzero(road)
    for _ in range(random.randint(a.min_paste, a.max_paste)):
        avail = [c for c in by_cls if any(patches[i]['used'] < a.reuse_cap for i in by_cls[c])]
        if not avail: break
        c = random.choices(avail, [weights[x] for x in avail])[0]
        pi = random.choice([i for i in by_cls[c] if patches[i]['used'] < a.reuse_cap]); pt = patches[pi]
        sc = random.uniform(0.9, 1.1)
        for _try in range(30):
            # traffic prior: queue along the long axis of a real vehicle (stays on its road), pedestrians near any real object
            if c in PED_LIKE or not vehicles:
                rp = random.choice(objs)[1]; ctr = rp.mean(0) + np.random.normal(0, 60, 2)
            else:
                rp = random.choice(vehicles)[1]; al = long_axis_angle(rp); t = random.choice([-1, 1]) * random.uniform(40, 220)
                ctr = rp.mean(0) + t * np.array([math.cos(al), math.sin(al)]) + np.random.normal(0, 8, 2) * np.array([-math.sin(al), math.cos(al)])
            if c in PED_LIKE: phi = random.uniform(0, 2 * math.pi)
            else:
                nv = min(vehicles, key=lambda cp: np.linalg.norm(cp[1].mean(0) - ctr)); tgt = long_axis_angle(nv[1]) + math.radians(random.uniform(-20, 20))
                phi = tgt - (0.0 if pt['w'] >= pt['h'] else math.pi / 2) + (math.pi if random.random() < 0.5 else 0)
            poly = poly_of(ctr[0], ctr[1], pt['w'] * sc, pt['h'] * sc, phi)
            if poly.min() < 2 or poly.max() > T - 2: continue
            if poly_road_frac(poly, road) < 0.9: continue
            if any(overlap_ratio(poly, q) > 0.1 or overlap_ratio(q, poly) > 0.1 for _, q in objs + newobjs): continue
            break
        else: continue
        tile = paste(tile, pt, ctr[0], ctr[1], phi, sc, np.ones((T, T), bool)); road = road_mask(tile) if False else road
        newobjs.append((c, poly)); pt['used'] += 1; added[cls[c]] = added.get(cls[c], 0) + 1
    name = f'syn{k:04d}_{n}'
    cv2.imwrite(f'{a.out}/images/train/{name}.jpg', tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = [open(f).read().strip()] if objs else []
    out += [f"{c} " + " ".join(f"{x / T:.6f} {y / T:.6f}" for x, y in p) for c, p in newobjs]
    open(f'{a.out}/labels/train/{name}.txt', 'w').write("\n".join(o for o in out if o) + "\n")
open(f'{a.out}/data.yaml', 'w').write(open(f'{a.tiles}/data.yaml').read().replace(os.path.abspath(a.tiles), os.path.abspath(a.out)))
json.dump({'real_tiles': len(real), 'synthetic_tiles': n_syn, 'pasted': added}, open(f'{a.out}/synth.json', 'w'), indent=1)
print('real', len(real), 'synthetic', n_syn, 'pasted objects per class', added)
