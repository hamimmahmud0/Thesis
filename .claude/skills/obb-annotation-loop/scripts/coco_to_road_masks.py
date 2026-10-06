"""Turn a sam31 COCO result (text prompt 'road') into per-frame road mask PNGs for synth_paste.py --road-masks.
python coco_to_road_masks.py INSTANCES.json OUT_DIR [--category road] [--dilate 0]
Frame id = image file stem.split('__')[-1] (same id build_tiles.py / synth_paste.py use), so run SAM on the SAME frame files the dataset was built from.
Needs pycocotools (pip install pycocotools). Output: OUT_DIR/<frame id>.png, 255 = road, same size as the frame.
NOTE: this path has not been exercised end to end yet - after the first run, overlay a mask on one frame and look at it before generating synthetic data."""
import argparse, json, os
import numpy as np
from PIL import Image
from pycocotools import mask as mu

ap = argparse.ArgumentParser()
ap.add_argument('coco'); ap.add_argument('out'); ap.add_argument('--category', default='road'); ap.add_argument('--dilate', type=int, default=0)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
d = json.load(open(A.coco))
cat = {c['id'] for c in d['categories'] if A.category.lower() in c['name'].lower()}
assert cat, f"no category containing '{A.category}' in {[c['name'] for c in d['categories']]}"
imgs = {i['id']: i for i in d['images']}; masks = {}
for a in d['annotations']:
    if a['category_id'] not in cat: continue
    im = imgs[a['image_id']]; h, w = im['height'], im['width']; seg = a['segmentation']
    rle = mu.merge(mu.frPyObjects(seg, h, w)) if isinstance(seg, list) else (mu.frPyObjects(seg, h, w) if isinstance(seg['counts'], list) else seg)
    m = mu.decode(rle).astype(bool)
    masks[a['image_id']] = masks.get(a['image_id'], np.zeros((h, w), bool)) | m
for iid, m in masks.items():
    if A.dilate:
        import cv2
        m = cv2.dilate(m.astype(np.uint8), np.ones((A.dilate, A.dilate), np.uint8)) > 0
    frame = os.path.splitext(os.path.basename(imgs[iid]['file_name']))[0].split('__')[-1]
    Image.fromarray((m * 255).astype(np.uint8)).save(f'{A.out}/{frame}.png')
print(f'wrote {len(masks)} road masks of {len(imgs)} images -> {A.out}')
