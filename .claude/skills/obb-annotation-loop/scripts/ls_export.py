"""Export human OBB annotations from ALL Label Studio projects whose title starts with a prefix (default 'BEDOBB') into one dataset.
python ls_export.py URL REFRESH_TOKEN_FILE OUT_DIR [--prefix BEDOBB] [--merge-map map.json] [--drop A,B] [--keep-empty]
                    [--exclude-projects 5,9] [--limit N] [--dry-run]
OUT_DIR gets images/<h8>__<orig>.JPG, labels/<same>.txt (cls x1 y1 .. x4 y4 normalized), classes.txt, manifest.json - the layout build_tiles.py expects.

Rules (and why):
- Every project matching the prefix is used - annotators work in several projects, so a single project under-counts the data.
- Only human annotations count (predictions are model output). Cancelled/skipped annotations are ignored; per task the latest one wins.
- The same image may sit in several projects: the annotation with the latest updated_at wins; its project is recorded in manifest.json.
- Annotations that were started from a model prediction (parent_prediction set) are counted and flagged: an annotator who pressed Submit
  without editing leaves model output labelled as ground truth. Use --skip-unedited-prediction to leave those out.
- Classes: union of label names; --merge-map {"SUV": "Private-Passenger-Car"} renames, --drop removes classes; classes with 0 instances are removed.
"""
import argparse, collections, hashlib, io, json, os, re, sys
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ls_common import LS, poly_from_rect

ap = argparse.ArgumentParser()
ap.add_argument('url'); ap.add_argument('token_file'); ap.add_argument('out')
ap.add_argument('--prefix', default='BEDOBB'); ap.add_argument('--merge-map', default=''); ap.add_argument('--drop', default='')
ap.add_argument('--exclude-projects', default=''); ap.add_argument('--keep-empty', action='store_true')
ap.add_argument('--skip-unedited-prediction', action='store_true'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--dry-run', action='store_true')
A = ap.parse_args()
ls = LS(A.url, A.token_file)
merge = json.load(open(A.merge_map)) if A.merge_map else {}
drop = {x for x in A.drop.split(',') if x}
excl = {int(x) for x in A.exclude_projects.split(',') if x}

projs = [p for p in ls.projects() if p['title'].startswith(A.prefix) and p['id'] not in excl]
print(f"projects starting with '{A.prefix}': {len(projs)}")
for p in projs:
    print(f"  {p['id']:>4}  {p['title']!r}  tasks={p.get('task_number')} annotations={p.get('total_annotations_number')} predictions={p.get('total_predictions_number')}")
if not projs: sys.exit('no matching projects')

best = {}  # orig image name -> candidate
n_cancel = n_empty = n_from_pred = n_nonrect = 0
per_proj = collections.Counter()
for p in projs:
    r = ls.call('GET', f"/api/projects/{p['id']}/export?exportType=JSON&download_all_tasks=false", timeout=600)
    r.raise_for_status()
    for t in r.json():
        img = t['data'].get('image', '')
        orig = re.sub(r'^[0-9a-f]{8}-', '', os.path.basename(img.split('?')[0]))
        anns = [a for a in t.get('annotations', []) if not a.get('was_cancelled')]
        n_cancel += sum(1 for a in t.get('annotations', []) if a.get('was_cancelled'))
        if not anns: continue
        a = max(anns, key=lambda x: x.get('updated_at') or '')
        regs = [x for x in a['result'] if x.get('type') == 'rectanglelabels']
        n_nonrect += sum(1 for x in a['result'] if x.get('type') not in ('rectanglelabels',))
        if not regs and not A.keep_empty: n_empty += 1; continue
        from_pred = a.get('parent_prediction') is not None
        if from_pred and A.skip_unedited_prediction: continue
        cand = {'project': p['id'], 'task': t['id'], 'updated_at': a.get('updated_at') or '', 'by': a.get('completed_by'), 'from_prediction': from_pred,
                'image': img, 'regions': regs}
        if orig not in best or cand['updated_at'] > best[orig]['updated_at']: best[orig] = cand
    per_proj[p['id']] = sum(1 for c in best.values() if c['project'] == p['id'])
print(f'images with a usable human annotation: {len(best)} (cancelled ignored: {n_cancel}, empty skipped: {n_empty}, non-rectangle regions ignored: {n_nonrect})')
print('winning project per image:', dict(per_proj))
print('annotations that started from a model prediction:', sum(1 for c in best.values() if c['from_prediction']), '- check these were really reviewed')
if A.dry_run: sys.exit(0)

names = sorted({(merge.get(g['value']['rectanglelabels'][0], g['value']['rectanglelabels'][0])) for c in best.values() for g in c['regions']} - drop)
os.makedirs(f'{A.out}/images', exist_ok=True); os.makedirs(f'{A.out}/labels', exist_ok=True)
counts = collections.Counter(); manifest = {}
items = sorted(best.items())[: A.limit or None]
for orig, c in items:
    img = c['image']
    resp = ls.call('GET', img) if img.startswith('/') else requests.get(img, timeout=300)
    resp.raise_for_status()
    im = Image.open(io.BytesIO(resp.content)); W, H = im.size
    stem = f"{hashlib.md5(orig.encode()).hexdigest()[:8]}__{os.path.splitext(orig)[0]}"
    if im.format == 'JPEG': open(f'{A.out}/images/{stem}.JPG', 'wb').write(resp.content)
    else: im.convert('RGB').save(f'{A.out}/images/{stem}.JPG', quality=95)
    lines = []
    for g in c['regions']:
        v = g['value']; nm = merge.get(v['rectanglelabels'][0], v['rectanglelabels'][0])
        if nm in drop: continue
        poly = poly_from_rect(v, W, H)
        lines.append(f"{names.index(nm)} " + " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}" for x, y in poly)); counts[nm] += 1
    open(f'{A.out}/labels/{stem}.txt', 'w').write("\n".join(lines) + ("\n" if lines else ""))
    manifest[stem] = {k: c[k] for k in ('project', 'task', 'updated_at', 'by', 'from_prediction')} | {'orig': orig, 'regions': len(lines), 'size': [W, H]}
keep = [n for n in names if counts[n] > 0]
if keep != names:  # remove classes with 0 instances and renumber
    remap = {names.index(n): keep.index(n) for n in keep}
    for f in os.listdir(f'{A.out}/labels'):
        L = [l.split() for l in open(f'{A.out}/labels/{f}') if l.strip()]
        for v in L: v[0] = str(remap[int(v[0])])
        open(f'{A.out}/labels/{f}', 'w').write("\n".join(" ".join(v) for v in L) + ("\n" if L else ""))
open(f'{A.out}/classes.txt', 'w').write("\n".join(keep) + "\n")
json.dump({'prefix': A.prefix, 'projects': {p['id']: p['title'] for p in projs}, 'merge': merge, 'drop': sorted(drop), 'images': manifest}, open(f'{A.out}/manifest.json', 'w'), indent=1)
print(f'wrote {len(manifest)} images, {sum(counts.values())} objects, {len(keep)} classes -> {A.out}')
print({n: counts[n] for n in keep})
