"""Panels for every image in a folder: [ORIGINAL (if annotated)] + one panel per model.
python viz_full.py IMAGES_DIR OUT_DIR GT_DIR TRAIN_IDS VAL_IDS LABEL=pred.json [...]   (ids comma separated DJI ids)"""
import glob, json, os, sys
from PIL import Image, ImageDraw
IM, OUT, GT, TR, VA = sys.argv[1:6]; TR, VA = set(TR.split(',')), set(VA.split(','))
models = [(a.split('=')[0], json.load(open(a.split('=', 1)[1]))) for a in sys.argv[6:]]
cls = [l.strip() for l in open(f'{GT}/classes.txt') if l.strip()]; os.makedirs(OUT, exist_ok=True); S = 0.25
gtmap = {os.path.splitext(os.path.basename(f))[0].split('__')[-1]: os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{GT}/images/*.JPG')}
def draw(im, polys, col, labels):
    d = ImageDraw.Draw(im)
    for p, l in zip(polys, labels):
        d.polygon([(x * S, y * S) for x, y in p], outline=col, width=2); d.text((p[0][0] * S, p[0][1] * S - 10), l, fill=col)
for f in sorted(glob.glob(f'{IM}/*.JPG')):
    n = os.path.splitext(os.path.basename(f))[0]; im = Image.open(f).convert('RGB'); W, H = im.size
    base = im.resize((int(W * S), int(H * S))); panels = []
    tag = 'TRAIN' if n in TR else 'VAL' if n in VA else 'UNSEEN'
    if n in gtmap:
        gt, gl = [], []
        for l in open(f'{GT}/labels/{gtmap[n]}.txt'):
            v = l.split()
            if len(v) >= 9: gt.append([(float(v[1 + 2 * j]) * W, float(v[2 + 2 * j]) * H) for j in range(4)]); gl.append(cls[int(v[0])])
        a = base.copy(); draw(a, gt, (0, 255, 0), gl); ImageDraw.Draw(a).text((8, 8), f'{n} {tag}  ORIGINAL ({len(gt)})', fill=(255, 255, 0)); panels.append(a)
    for lab, pj in models:
        dets = pj.get(n, []); b = base.copy()
        draw(b, [d['poly'] for d in dets], (255, 60, 60), [f"{d['name']} {d['conf']:.2f}" for d in dets])
        ImageDraw.Draw(b).text((8, 8), f'{n} {tag}  {lab} ({len(dets)})', fill=(255, 255, 0)); panels.append(b)
    c = Image.new('RGB', (sum(p.width for p in panels) + 8 * (len(panels) - 1), base.height)); x = 0
    for p in panels: c.paste(p, (x, 0)); x += p.width + 8
    c.save(f'{OUT}/{n}.jpg', quality=85)
print('done')
