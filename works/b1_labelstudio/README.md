# b1 -> Label Studio import copy
Copy of works/b1 (original untouched). Boxes are pseudo-labels, imported as **predictions** (not annotations).

1. Serve with local storage:
   `export LOCAL_FILES_SERVING_ENABLED=true LOCAL_FILES_DOCUMENT_ROOT=/home/hamim-mahmud/Workspace/Thesis/works && label-studio start`
2. Create project, Labeling Interface -> Code -> paste `label_config.xml`.
3. Settings -> Cloud Storage -> Add Source -> Local files, absolute path `.../works/b1_labelstudio/images` (no need to sync) — or just skip.
4. Import `tasks.json` (image URLs are `/data/local-files/?d=b1_labelstudio/images/<name>`).
5. Use "Accept prediction" per task to turn them into editable annotations.

Regenerate: `python3 make_tasks.py`.
