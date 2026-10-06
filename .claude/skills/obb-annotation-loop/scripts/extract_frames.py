"""Sample frames from MP4s stored in a Hugging Face dataset repo WITHOUT downloading them (ffmpeg input seeking over HTTP range requests).
python extract_frames.py OUT_DIR --repo hamimmahmud0/DRINF --include "Doyel Chattor,Polashi Bazaar" --step 35 [--limit-per-video N] [--threads 4]
--include: comma separated substrings; a video is used if its repo path contains any of them (case-insensitive). --step: seconds between frames.
Writes OUT_DIR/<site>_<clip>_t<seconds>.JPG (upper-case extension on purpose: the other scripts glob *.JPG) and OUT_DIR/../frames_manifest.json.
Needs: huggingface_hub, ffmpeg + ffprobe, HF_TOKEN in the environment for private repos."""
import argparse, json, os, re, subprocess, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--repo', required=True); ap.add_argument('--include', default='')
ap.add_argument('--step', type=float, default=35.0); ap.add_argument('--limit-per-video', type=int, default=0)
ap.add_argument('--threads', type=int, default=4); ap.add_argument('--repo-type', default='dataset')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
tok = os.environ.get('HF_TOKEN')
inc = [x.strip().lower() for x in A.include.split(',') if x.strip()]
files = [f for f in HfApi().list_repo_files(A.repo, repo_type=A.repo_type, token=tok) if f.lower().endswith('.mp4') and (not inc or any(i in f.lower() for i in inc))]
print('videos:', *files, sep='\n  ')
base = f'https://huggingface.co/{"datasets/" if A.repo_type == "dataset" else ""}{A.repo}/resolve/main/'
hdr = ['-headers', f'Authorization: Bearer {tok}\r\n'] if tok else []
slug = lambda s: re.sub(r'[^a-z0-9]+', '', s.lower())[:12]
jobs, meta = [], {}
for f in files:
    u = base + urllib.parse.quote(f)
    d = float(json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', *hdr, u], capture_output=True, text=True).stdout)['format']['duration'])
    parts = f.split('/'); site = slug(parts[-3] if len(parts) >= 3 else parts[0]); clip = re.sub(r'_merged$', '', os.path.splitext(parts[-1])[0])
    k = 0
    while k * A.step < d - 0.5 and (not A.limit_per_video or k < A.limit_per_video):
        jobs.append((u, k * A.step, f'{A.out}/{site}_{clip}_t{int(k * A.step):05d}s.JPG')); k += 1
    meta[f'{site}_{clip}'] = {'path': f, 'duration_s': d}
print('frames to extract:', len(jobs), flush=True)

def go(j):
    u, t, out = j
    if os.path.exists(out) and os.path.getsize(out) > 10000: return True
    for _ in range(3):
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.3f}', *hdr, '-i', u, '-frames:v', '1', '-q:v', '2', out], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 10000: return True
    print('FAILED', out, r.stderr[:120], flush=True); return False

with ThreadPoolExecutor(A.threads) as ex: ok = list(ex.map(go, jobs))
json.dump({'repo': A.repo, 'step_s': A.step, 'videos': meta, 'frames': [j[2] for j, o in zip(jobs, ok) if o]}, open(os.path.join(os.path.dirname(os.path.abspath(A.out)), 'frames_manifest.json'), 'w'), indent=1)
print('DONE ok', sum(ok), 'of', len(jobs))
