# Gotchas (all hit in round 1)

## Kaggle VM / MCP
- A fresh VM has no `~/work`: `mkdir -p ~/work` before `upload` or `scp` fails.
- With two or more VMs running, every call needs an explicit `vm_id`.
- `run_command` returns after ~120 s and moves the call to the background; anything longer must be started detached
  (`setsid nohup cmd > log 2>&1 < /dev/null &`) and polled with `sleep <=110`.
- Tool output must be ASCII. Progress bars contain multi-byte characters; cutting them mid-character breaks the SSH layer and every later call returns
  empty. Filter with `LC_ALL=C tr -cd '[:print:]\n'` before any `cut`/`tail -c`.
- Do not use `pkill -f` or `pgrep -f` with a pattern that also appears in your own command line: it matches (and kills / waits on) its own shell. List PIDs
  with `ps -eo pid,args | grep ... | grep -v grep` and `kill <pid>`; wait for a process with `while kill -0 <pid>; do sleep 10; done`.
- An interrupted `upload` can leave a partial copy and the retry may not overwrite it. After uploading a dir, compare file counts; if short,
  `rm -rf` the target and upload again.
- Directory downloads of >100 MB time out (300 s without progress). `tar -cf x.tar dir` on the VM and download the single file.
- Training saves to `runs/obb/runs/<name>/` (Ultralytics prepends `obb/`): a checkpoint-sync loop pointed at `runs/<name>` syncs nothing.
- Training holds ~14 GB on each T4: never start inference on the same GPUs at the same time (OOM). Queue it.
- Kill a stale run by PID, and delete its `runs/` dir before relaunching, or the old process keeps writing the same log.
- Early stopping with a 3-frame validation set triggers on noise (round 1 best epochs were 13 and 29 of ~70-80). Expect that; do not read much into it.
- Stop VMs when outputs are saved. Delete token files first.

## Label Studio
- Auth is JWT: a pasted *refresh* token must be exchanged for a short-lived access token (`POST /api/token/refresh`, `{"refresh": ...}`); `ls_common.LS`
  does this and re-mints on 401. Login by `admin`/password failed because login is by email - prefer a token.
- Project title max 50 characters; longer gives an unhelpful `Validation error`.
- `RectangleLabels canRotate="true"`: value = x, y (top-left, % of image), width/height (% of image), rotation degrees clockwise about (x, y). Confirmed by
  exporting a test task with `exportType=YOLO_OBB` and comparing to the source polygons (exact).
- Predictions vs annotations: model output goes in as predictions (shown as pre-annotations, with `score`); human labels as annotations. Annotators who
  submit an unedited prediction create an annotation with `parent_prediction` set - `ls_export.py` counts these.
- Image import: `POST /api/projects/<id>/import?return_task_ids=true` with a multipart `file`; 5 MB JPEGs pass the proxy. Import the first file alone so a
  limit/permission problem aborts early. Images are served at `/data/upload/<project>/<hash8>-<name>` with the same Bearer token.
- Deleting test tasks (`DELETE /api/tasks/<id>`) is fine in a project you just created; never delete in a shared one.
- Projects get renamed and annotated by others: re-list projects before every round instead of trusting ids from memory.
- A frame with ~500 boxes is slow to render. If annotators struggle, consider cutting such frames into 2x2 crops as separate tasks (not done yet).

## Modelling
- Pretrained DOTA `yolo26l-obb.pt` is the right start; the DIOR-R / VEDAI fine-tunes are domain-shifted and muddy comparisons.
- Satellite models see few drone objects (VEDAI found 3% of held-out objects, DIOR-R 33%, a model fine-tuned on 20 frames 69% at IoU 0.5, class-agnostic)
  and DIOR-R adds large false `dam`/`bridge` boxes on buildings: do not pre-annotate with them.
- Cross-tile NMS: ProbIoU 0.3 can suppress side-by-side vehicles; 0.5-0.6 still removes duplicates (ProbIoU ~1).
- Rare classes: with 8-14 instances each, copy-paste reuses the same crops (cap 25x). Gains on them cannot be measured on 3 val frames.

## Data sources
- DRINF (`hamimmahmud0/DRINF`, dataset repo): 4K, 23.976 fps, 11-15 min clips under `Nadirs/<site>/<date>/DJI_xxxx_merged.mp4`; 1 frame per 35 s gives
  ~117 frames for Doyel Chattor + Polashi Bazaar. Bakshi bazaar clips exist but were not requested.
- b1 frames: `hf://buckets/hamimmahmud0/SAM_COCO_b1/b1/images` (174 JPGs, 4000x3000); `hf buckets sync` pulls them in about a minute.
- Frame ids: scripts key frames by `image stem.split('__')[-1]`; `ls_export.py` names files `<md5 8>__<original name>` so this works for any source.
