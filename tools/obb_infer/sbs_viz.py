"""python sbs_viz.py DATASET_DIR OUT_DIR PRED1.json:LABEL1 [PRED2.json:LABEL2 ...]
Panels: ORIGINAL (green, from DATASET_DIR/labels + classes.txt) then one panel per prediction file (red)."""
import glob, json, os, sys
from PIL import Image, ImageDraw
D, OUT = sys.argv[1], sys.argv[2]
preds = [(p.split(':')[0], p.split(':')[1]) for p in sys.argv[3:]]
P = [(json.load(open(f)), lab) for f, lab in preds]
cls = [l.strip() for l in open(f'{D}/classes.txt') if l.strip()]
os.makedirs(OUT, exist_ok=True); S = 0.4

def draw(im, polys, col, labels):
    d = ImageDraw.Draw(im)
    for p, l in zip(polys, labels):
        d.polygon([(x * S, y * S) for x, y in p], outline=col, width=2)
        d.text((p[0][0] * S, p[0][1] * S - 10), l, fill=col)

for f in sorted(glob.glob(f'{D}/images/*.JPG')):
    n = os.path.splitext(os.path.basename(f))[0]; im = Image.open(f).convert('RGB'); W, H = im.size
    base = im.resize((int(W * S), int(H * S))); panels = []
    gt, gl = [], []
    for l in open(f'{D}/labels/{n}.txt'):
        v = l.split()
        if len(v) < 9: continue
        gt.append([(float(v[1 + 2 * j]) * W, float(v[2 + 2 * j]) * H) for j in range(4)]); gl.append(cls[int(v[0])])
    a = base.copy(); draw(a, gt, (0, 255, 0), gl); ImageDraw.Draw(a).text((8, 8), f'ORIGINAL ({len(gt)})', fill=(255, 255, 0)); panels.append(a)
    for preds_, lab in P:
        dets = preds_.get(n, []); b = base.copy()
        draw(b, [d['poly'] for d in dets], (255, 0, 0), [f"{d['name']} {d['conf']:.2f}" for d in dets])
        ImageDraw.Draw(b).text((8, 8), f'{lab} ({len(dets)})', fill=(255, 255, 0)); panels.append(b)
    c = Image.new('RGB', (sum(p.width for p in panels) + 10 * (len(panels) - 1), base.height))
    x = 0
    for p in panels: c.paste(p, (x, 0)); x += p.width + 10
    c.save(f'{OUT}/{n}.jpg', quality=88)
    print(n, len(gt), [len(p.get(n, [])) for p, _ in P], flush=True)
