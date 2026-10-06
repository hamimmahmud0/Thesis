"""Import frames into a Label Studio project as OBB (rotated-rectangle) tasks.
python ls_import.py URL PROJECT_ID REFRESH_TOKEN_FILE IMAGES_DIR GT_DIR PRED_JSON OUT_MAP.json [--model-version V] [--only DJI_0566,..] [--threads 4]
- frames with manual labels in GT_DIR (images/<hash>__DJI_x.JPG + labels/*.txt + classes.txt, normalized 4-corner polygons) become ANNOTATIONS
- all other frames get PREDICTIONS from PRED_JSON ({stem: [{name, conf, poly}]}, full-image pixel coords)
Rotated rectangle convention (Label Studio): x,y = top-left corner (percent), width/height (percent of image w/h), rotation in degrees
clockwise about (x,y). Every conversion is round-trip checked against the source polygon."""
import argparse, glob, json, math, os, random, string, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

ap = argparse.ArgumentParser()
for a in ('url', 'project', 'token_file', 'images', 'gt', 'pred', 'out'): ap.add_argument(a)
ap.add_argument('--model-version', default='new-yolo26l-obb'); ap.add_argument('--only', default=''); ap.add_argument('--threads', type=int, default=4)
A = ap.parse_args(); URL = A.url.rstrip('/'); REFRESH = open(A.token_file).read().strip()
_lock = threading.Lock(); _tok = {'v': None, 't': 0}

def access(force=False):
    with _lock:
        if force or _tok['v'] is None or time.time() - _tok['t'] > 150:
            r = requests.post(f'{URL}/api/token/refresh', json={'refresh': REFRESH}, timeout=30); r.raise_for_status()
            _tok['v'], _tok['t'] = r.json()['access'], time.time()
        return _tok['v']

def call(method, path, **kw):
    for attempt in range(4):
        h = {'Authorization': f'Bearer {access(attempt > 0 and attempt == 1)}'}
        try:
            r = requests.request(method, URL + path, headers=h, timeout=300, **kw)
        except requests.RequestException as e:
            time.sleep(2 * (attempt + 1)); err = e; continue
        if r.status_code == 401: access(True); continue
        if r.status_code >= 500: time.sleep(2 * (attempt + 1)); continue
        return r
    raise RuntimeError(f'{method} {path} failed')

def rect_from_poly(poly, W, H):
    p = [tuple(map(float, q)) for q in poly]
    e1 = (p[1][0] - p[0][0], p[1][1] - p[0][1]); e2 = (p[3][0] - p[0][0], p[3][1] - p[0][1])
    if e1[0] * e2[1] - e1[1] * e2[0] < 0: p = [p[0], p[3], p[2], p[1]]; e1 = (p[1][0] - p[0][0], p[1][1] - p[0][1]); e2 = (p[3][0] - p[0][0], p[3][1] - p[0][1])
    w, h = math.hypot(*e1), math.hypot(*e2); th = math.atan2(e1[1], e1[0])
    c, s = math.cos(th), math.sin(th)
    rec = [(p[0][0] + c * dx - s * dy, p[0][1] + s * dx + c * dy) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
    err = max(min(math.hypot(a - b[0], bb - b[1]) for b in p) for a, bb in rec)
    return {'x': p[0][0] / W * 100, 'y': p[0][1] / H * 100, 'width': w / W * 100, 'height': h / H * 100, 'rotation': math.degrees(th)}, err, (w, h)

def region(name, poly, W, H, score=None):
    v, err, (w, h) = rect_from_poly(poly, W, H)
    if w < 2 or h < 2: return None, err
    v['rectanglelabels'] = [name]
    r = {'id': ''.join(random.choices(string.ascii_letters + string.digits, k=10)), 'type': 'rectanglelabels', 'from_name': 'label', 'to_name': 'image',
         'original_width': W, 'original_height': H, 'image_rotation': 0, 'value': v}
    if score is not None: r['score'] = round(float(score), 4)
    return r, err

classes = [l.strip() for l in open(f'{A.gt}/classes.txt') if l.strip()]
gt_by_dji = {}
for f in glob.glob(f'{A.gt}/images/*.JPG'):
    stem = os.path.splitext(os.path.basename(f))[0]; gt_by_dji[stem.split('__')[-1]] = stem
preds = json.load(open(A.pred)); only = set(x for x in A.only.split(',') if x)
files = sorted(glob.glob(f'{A.images}/*.JPG')); files = [f for f in files if not only or os.path.splitext(os.path.basename(f))[0] in only]
stats = {'max_roundtrip_err_px': 0.0, 'dropped_degenerate': 0}; mapping = {}

def work(f):
    dji = os.path.splitext(os.path.basename(f))[0]; W, H = Image.open(f).size
    with open(f, 'rb') as fh:
        r = call('POST', f'/api/projects/{A.project}/import?return_task_ids=true', files={'file': (os.path.basename(f), fh, 'image/jpeg')})
    if r.status_code >= 300: raise RuntimeError(f'{dji} upload {r.status_code} {r.text[:200]}')
    tid = r.json()['task_ids'][0]; regs = []; kind = None
    if dji in gt_by_dji:
        kind = 'annotation'
        for l in open(f'{A.gt}/labels/{gt_by_dji[dji]}.txt'):
            v = l.split()
            if len(v) < 9: continue
            poly = [(float(v[1 + 2 * j]) * W, float(v[2 + 2 * j]) * H) for j in range(4)]
            rg, err = region(classes[int(v[0])], poly, W, H); stats['max_roundtrip_err_px'] = max(stats['max_roundtrip_err_px'], err)
            if rg: regs.append(rg)
            else: stats['dropped_degenerate'] += 1
        if regs:
            rr = call('POST', f'/api/tasks/{tid}/annotations', json={'result': regs, 'was_cancelled': False, 'ground_truth': False})
            if rr.status_code >= 300: raise RuntimeError(f'{dji} annotation {rr.status_code} {rr.text[:200]}')
    else:
        kind = 'prediction'
        for d in preds.get(dji, []):
            rg, err = region(d['name'], d['poly'], W, H, d['conf']); stats['max_roundtrip_err_px'] = max(stats['max_roundtrip_err_px'], err)
            if rg: regs.append(rg)
            else: stats['dropped_degenerate'] += 1
        if regs:
            sc = sum(x['score'] for x in regs) / len(regs)
            rr = call('POST', '/api/predictions', json={'task': tid, 'result': regs, 'score': round(sc, 4), 'model_version': A.model_version})
            if rr.status_code >= 300: raise RuntimeError(f'{dji} prediction {rr.status_code} {rr.text[:200]}')
    mapping[dji] = {'task_id': tid, 'kind': kind, 'regions': len(regs)}
    print(dji, tid, kind, len(regs), flush=True)

if files:  # first file alone so a size/permission problem aborts early
    work(files[0])
with ThreadPoolExecutor(A.threads) as ex: list(ex.map(work, files[1:]))
json.dump({'stats': stats, 'tasks': mapping}, open(A.out, 'w'), indent=1)
print('DONE', len(mapping), 'tasks', stats, 'annotations:', sum(1 for v in mapping.values() if v['kind'] == 'annotation'), 'predictions:', sum(1 for v in mapping.values() if v['kind'] == 'prediction'))
