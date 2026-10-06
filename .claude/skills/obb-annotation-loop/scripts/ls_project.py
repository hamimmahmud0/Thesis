"""Label Studio project helper.
python ls_project.py URL TOKEN_FILE list [--prefix BEDOBB]
python ls_project.py URL TOKEN_FILE create TITLE --labels-from PROJECT_ID | --classes classes.txt [--description D] [--model-version V]
python ls_project.py URL TOKEN_FILE add-label PROJECT_ID NAME [--color #ff1493]
Notes: titles are limited to 50 chars (longer -> 'Validation error'). New review projects should START WITH the prefix (BEDOBB) so the next
export picks them up. The config is rotated-rectangle labelling (RectangleLabels canRotate) on an Image."""
import argparse, colorsys, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ls_common import LS

ap = argparse.ArgumentParser()
ap.add_argument('url'); ap.add_argument('token_file'); ap.add_argument('cmd', choices=['list', 'create', 'add-label'])
ap.add_argument('args', nargs='*'); ap.add_argument('--prefix', default=''); ap.add_argument('--labels-from', default=''); ap.add_argument('--classes', default='')
ap.add_argument('--description', default=''); ap.add_argument('--model-version', default=''); ap.add_argument('--color', default='#ff1493')
A = ap.parse_args(); ls = LS(A.url, A.token_file)

def config(names):
    cols = ['#%02x%02x%02x' % tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(len(names), 1), 0.85, 0.95)) for i in range(len(names))]
    return ('<View>\n  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
            '  <RectangleLabels name="label" toName="image" canRotate="true" strokeWidth="2" opacity="0.2">\n'
            + ''.join(f'    <Label value="{n}" background="{c}"/>\n' for n, c in zip(names, cols)) + '  </RectangleLabels>\n</View>')

if A.cmd == 'list':
    for p in ls.projects():
        if p['title'].startswith(A.prefix):
            print(f"{p['id']:>4}  {p['title']!r}  tasks={p.get('task_number')} ann={p.get('total_annotations_number')} pred={p.get('total_predictions_number')} sampling={p.get('sampling')}")
elif A.cmd == 'create':
    title = A.args[0]
    if len(title) > 50: sys.exit(f'title is {len(title)} chars; Label Studio allows 50')
    if A.labels_from: cfg = ls.call('GET', f'/api/projects/{A.labels_from}').json()['label_config']
    else: cfg = config([l.strip() for l in open(A.classes) if l.strip()])
    r = ls.call('POST', '/api/projects', json={'title': title, 'description': A.description, 'label_config': cfg, 'model_version': A.model_version})
    d = r.json(); print(r.status_code, {k: d.get(k) for k in ('id', 'title', 'detail', 'validation_errors')})
else:
    pid, name = A.args[0], A.args[1]
    p = ls.call('GET', f'/api/projects/{pid}').json(); cfg = p['label_config']
    if f'value="{name}"' in cfg: sys.exit('label already present')
    new = cfg.replace('  </RectangleLabels>', f'    <Label value="{name}" background="{A.color}"/>\n  </RectangleLabels>')
    assert new != cfg, 'could not find </RectangleLabels> to extend'
    r = ls.call('PATCH', f'/api/projects/{pid}', json={'label_config': new}); print(r.status_code, 'labels now:', len(re.findall('<Label ', r.json().get('label_config', ''))))
