# class_annotator_from_coco

Build a **classification dataset** from a **detection dataset** (COCO format) by reviewing every detected
object by hand, fast, with the keyboard.

The detections (here: SAM-annotated drone traffic frames) are not fully accurate, so `review.py` shows you one
object at a time and you either **confirm** it, **change its class**, **reject** it, or **flag** it for later.
Confirmed objects are cropped and saved into one folder per class.

```
dataset/  (COCO detections)  ──►  review.py (you, keyboard)  ──►  cls_dataset/<Class>/*.png
```

## Contents

1. [Setup](#1-setup)
2. [Prepare the `dataset/` directory](#2-prepare-the-dataset-directory)
3. [Run the reviewer](#3-run-the-reviewer)
4. [Keyboard shortcuts](#4-keyboard-shortcuts)
5. [Classes and `classes.yaml`](#5-classes-and-classesyaml)
6. [Output layout](#6-output-layout)
7. [Command-line options](#7-command-line-options)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Setup

Python 3.10+ (tested on 3.13), Linux desktop session.

```bash
cd tools/utils/class_annotator_from_coco

# (optional) virtual environment
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

Tkinter provides the review window and is **not** a pip package:

```bash
sudo apt install python3-tk        # Ubuntu / Debian (conda's Python already has it)
```

Check everything is in place:

```bash
python3 -c "import cv2, numpy, PIL, yaml, tkinter; print('ok')"
hf --version
```

> Install `opencv-python-headless`, **not** `opencv-python`. Having both wheels installed breaks `import cv2`.
> The tool does not use OpenCV windows, so the headless build is all it needs.

The UI text uses the Ubuntu font if `/usr/share/fonts/truetype/ubuntu/Ubuntu-{R,B}.ttf` exists
(default on Ubuntu) and falls back to DejaVu Sans otherwise.

---

## 2. Prepare the `dataset/` directory

`review.py` reads a COCO dataset from `dataset/` with this layout:

```
dataset/
├── images/
│   ├── DJI_0266_merged_frame_0001.png
│   └── ...
└── annotations/
    └── instances.json
```

Two rules matter:

- `images[].file_name` in `instances.json` is a **bare file name** that exists in `dataset/images/`
  (no `annotate/images/` prefix).
- Every annotation needs `id`, `image_id`, `category_id`, `bbox` (`[x, y, w, h]` in pixels), and ideally `score`.

The full detection dataset lives in the Hugging Face **bucket** `hamimmahmud0/SAM_COCO_v1_b2_3024`
(about 416 files, ~10 MB per 3840×2160 PNG, plus a 69 MB `instances.json`). You do not need all of it. The steps
below download only **30 images whose name contains `merged`** and cut the annotations down to those images.
That gives a ~280 MB dataset with about 15 000 objects.

### 2.1 Log in to Hugging Face

```bash
hf auth login            # paste a token with read access to the bucket
hf auth whoami           # should print your username
```

### 2.2 Look at what is in the bucket

```bash
BUCKET=hamimmahmud0/SAM_COCO_v1_b2_3024

hf buckets list $BUCKET -R > bucket_files.txt     # -R = recursive; columns: size, date, time, path
head -3 bucket_files.txt
grep -c merged bucket_files.txt                   # number of 'merged' images (264 at the time of writing)
```

The bucket layout is:

```
annotate/annotations/instances.json
annotate/images/DJI_0102_frame_0001.png          # other videos
annotate/images/DJI_0266_merged_frame_0001.png   # <- the ones we want ("merged")
...
```

### 2.3 Choose 30 `merged` images

Take images spread evenly over the whole sorted list, so the subset covers different videos and moments rather
than 30 consecutive, nearly identical frames. Change `COUNT` for a bigger or smaller subset.

```bash
COUNT=30

python3 - "$COUNT" <<'EOF' > selected.txt
import sys
count = int(sys.argv[1])
names = sorted(line.split()[-1] for line in open("bucket_files.txt")
               if line.split()[-1].startswith("annotate/images/") and "merged" in line.split()[-1])
step = len(names) / count
print("\n".join(names[int(i * step)] for i in range(count)))
EOF

wc -l selected.txt        # should equal COUNT
cat selected.txt
```

### 2.4 Download only those images and the annotation file

```bash
mkdir -p dataset/images dataset/annotations

# the single annotation file (69 MB)
hf buckets cp hf://buckets/$BUCKET/annotate/annotations/instances.json full_instances.json

# the selected images, 8 downloads in parallel
xargs -P 8 -I{} sh -c 'hf buckets cp hf://buckets/'$BUCKET'/{} dataset/images/$(basename {}) >/dev/null' < selected.txt

ls dataset/images | wc -l       # should equal COUNT
```

### 2.5 Cut the annotations down to the downloaded images

```bash
python3 - <<'EOF'
import json, os

d = json.load(open("full_instances.json"))
have = set(os.listdir("dataset/images"))

d["images"] = [i for i in d["images"] if os.path.basename(i["file_name"]) in have]
ids = {i["id"] for i in d["images"]}
for i in d["images"]:
    i["file_name"] = os.path.basename(i["file_name"])        # drop the 'annotate/images/' prefix
d["annotations"] = [a for a in d["annotations"] if a["image_id"] in ids]

json.dump(d, open("dataset/annotations/instances.json", "w"))
print(len(d["images"]), "images,", len(d["annotations"]), "annotations")
print([c["name"] for c in d["categories"]])
EOF
```

With `COUNT=30` and the default selection you should see roughly `30 images, 15268 annotations` and the categories
`Pedestrian, Vehicle, Car, Bus, Truck, Motorcycle, Bicycle, Rickshaw`.

### 2.6 Clean up (optional)

The intermediate files are no longer needed:

```bash
rm bucket_files.txt selected.txt full_instances.json
```

### Using your own COCO dataset instead

Put it in the same layout (`dataset/images/`, `dataset/annotations/instances.json`) or point the script at it with
`--dataset path/to/dir`. Then edit [`classes.yaml`](classes.yaml) so its class names match your COCO category
names.

---

## 3. Run the reviewer

```bash
python review.py
```

A window opens showing the first object:

- **left**: a close-up crop of the object
- **right**: the wider scene with the object's box in yellow (so you can judge it in context)
- **top**: predicted class, detector score, box size, progress in the current class and overall
- **bottom**: shortcuts and the class → key list

Work through the objects with the keyboard. Every decision is **saved immediately** to
`cls_dataset/decisions.json`, so you can quit (`Q`) any time and run `python review.py` again to resume where you
stopped. Crops are written when you quit or finish.

The queue goes **one predicted class at a time** (all Pedestrians, then all Cars, ...) and keeps objects from the
same image together, which makes consecutive objects easy to compare and keeps image loading fast.

Typical sessions:

```bash
python review.py                              # everything, in classes.yaml order
python review.py --max-per-class 200          # stop each class after 200 accepted crops
python review.py --min-score 0.5 --min-size 24   # skip weak / tiny detections
python review.py --class-order Rickshaw Car   # review these classes first
python review.py --only-flagged               # come back to what you flagged
python review.py --history                    # start in history mode: recheck / edit earlier decisions
python review.py --export-only                # rebuild the crops from saved decisions
```

---

## 4. Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` / `Enter` / `C` | **Confirm** the predicted class |
| *class key* (`1`, `2`, ... `a`, `b`, ...) | **Change class** to that class (and confirm). Keys are shown in the footer |
| `X` / `Delete` | **Reject**: leave the object out of the dataset |
| `F` | **Flag** for later inspection (several objects in one box, inaccurate box, ...) |
| `R` | **Re-annotate**: open the [box editor](#re-annotate-flagged-or-wrong-detections-box-editor) (fix box and class, one or several objects) |
| `V` | Open the per-image [viewer](#image-viewer-find-missing-annotations) |
| `S` | **Skip** for now; it comes back next session |
| `U` / `Backspace` | **Undo** the last decision |
| `Tab` / `Shift+Tab` | Jump to the **next / previous class** that still has objects to label |
| `H` | Toggle **history mode** (see [History](#history-recheck-and-edit-earlier-decisions)) |
| `Left` / `Right` | *(history mode)* go to the **older / newer** decision (`PgUp` / `PgDn` = 10 at a time) |
| `Tab` / `Shift+Tab` | *(history mode)* cycle the **class filter** |
| `N` | **New class**: type a name, `Enter` to create it and apply it to this object, `Esc` to cancel |
| `Q` / `Esc` | Save and quit |

Command keys `c x s u q n f h r v` are never used as class shortcuts. Class shortcuts are digits first, then letters,
then `Shift`+letters, so more than 10 classes is fine.

### History: recheck and edit earlier decisions

Made a mistake a few objects ago? Press **`H`** to open history mode (the top bar turns purple). It lists every
object you have already decided, **newest first when you enter**, and shows its current decision at the top right
(`confirmed Car`, `changed to Bus`, `rejected`, `flagged`).

| Key | In history mode |
|---|---|
| `Left` / `Right` | Step to the older / newer decision (`Home` = oldest, `End` = newest, `PgUp` / `PgDn` = 10) |
| `Tab` / `Shift+Tab` | Next / previous **filter**: `all`, then each class that has decisions, then `[rejected]`, `[flagged]`, `[redrawn]` |
| `R` | Re-annotate this object in the box editor |
| `V` | Open the image viewer on this object's image |
| `Space` / `Enter` / `C` | Set it back to the **predicted** class |
| class key | **Change** it to that class |
| `X` / `Delete` | **Reject** it |
| `F` | **Flag** it |
| `N` | Create a new class and apply it |
| `H` | Back to the review queue where you left off |
| `Q` / `Esc` | Save and quit |

- An edit **overwrites the saved decision in place** and you stay on the same object, so you can see the new
  status and keep stepping back. It is saved immediately, like every other decision.
- `S`, `U` and `Backspace` do nothing in history mode (there is nothing to skip, and an edit replaces a decision, so
  there is no undo).
- `--max-per-class` is respected: an object can keep its own class, but you cannot move it into a full class.
- When the queue runs out (or is already empty) the tool opens history mode instead of closing, so you get a last
  look before quitting. Start in it directly with `python review.py --history`.
- The crops are rebuilt from the saved decisions when you quit, so edits made here are reflected in
  `cls_dataset/` (an object you moved from `Car` to `Bus` is moved to the `Bus` folder).

#### Check annotations by class

The history filter matches the **decided** class (what ends up in the dataset folder), so it is the way to audit a class:

```bash
python review.py --recheck Car Bus    # only objects decided as Car or Bus, oldest first
python review.py --recheck flagged    # also: rejected, redrawn
```

Inside history mode `Tab` / `Shift+Tab` cycles `all` -> one class at a time -> `[rejected]` / `[flagged]` / `[redrawn]`;
the purple bar shows `HISTORY 3 / 40 [Car]`. If an edit moves an object out of the current filter (a `Car` you change to
`Bus`), you jump to the next object that still matches.

### Re-annotate flagged or wrong detections (box editor)

Press **`R`** on any object (queue or history) to open the box editor: a large, zoomed view of the detection.
The original box is dashed yellow, neighbouring detections thin grey, your boxes coloured by class. Use it when a box
covers two objects, the box is off, or the class is wrong **and** the box is off. Typical use: `python review.py
--only-flagged`, press `R` on each flagged object.

| Input | Action |
|---|---|
| drag on empty space | draw a new box (new boxes get the last class you picked) |
| click a box | select it; drag inside = **move**, drag a white handle = **resize** |
| class key | set the selected box's class |
| `Delete` / `Backspace` | remove the selected box |
| `Tab` / `Shift+Tab` | select next / previous box |
| arrow keys (`Shift` = 10 px) | nudge the selected box |
| `R` | reset to the single original box |
| `N` | create a new class |
| mouse wheel / middle- or right-drag | zoom around the cursor / pan |
| `Enter` | **save** (no boxes at all = the detection is rejected) |
| `Esc` | cancel, keep the old decision |

The result is stored as `{"action": "redraw", "boxes": [{"bbox": [x, y, w, h], "class": "Car"}, ...]}` (original image
pixels). Every box becomes a crop `<image>_<id>_<k>.png` in its class folder and counts toward `--max-per-class`.

### Image viewer: find missing annotations

Press **`V`** (or start with `python review.py --view`) to see **all** boxes of one image at once, so you can spot
objects that have no box. Boxes are outlined in their class colour once accepted; the legend (right) lets you
show / hide each status and each class and shows per-image counts:

| Status | Look |
|---|---|
| confirmed / class changed / re-annotated / added by you | solid, class colour, class label |
| not reviewed yet | thin yellow |
| flagged / rejected | dashed orange / red |
| ignored (category or filtered by `--min-score` / `--min-size`, e.g. `Vehicle`) | dotted grey |
| auto-removed pedestrian (`removed_pedestrians.json`) | dotted dark grey, hidden by default |

The ignored and auto-removed boxes are drawn so that an object that is *deliberately* not annotated is not mistaken for
a missing one.

| Input | Action |
|---|---|
| `Left` / `Right` (or `PgUp` / `PgDn`, `Up` / `Down`), `Home` / `End` | previous / next image, first / last |
| image list, **go to** box (number or part of the file name) | jump to an image; each row shows `accepted/total` and `?n` still to review |
| mouse wheel / middle- or right-drag | zoom around the cursor / pan |
| hover | class, score, size of the box under the cursor |
| click a box | select it; then **class key** = change class, `C` / `Space` = confirm, `X` = reject, `F` = flag, `R` = re-annotate |
| drag on empty space | draw a box for a **missing object**, then press its **class key** (`Esc` = discard) |
| `Tab` | hide / show all boxes (look at the bare image) |
| `V` / `Esc` | back to the review queue; `Q` quits |

Selecting a box you added and pressing `X` deletes it. Added boxes are saved in `<out>/added.json`
(`{"id": "n1", "image_id", "class", "bbox"}`) and exported as crops `<image>_n1.png` and into
`corrected_instances.json`; they are an extension, never written into `instances.json`.

**Reject vs `no_vehicle` vs flag**

- `X` reject: the object is dropped, nothing is saved.
- `0` (`no_vehicle`): the detection is wrong (not a vehicle), but you want to **keep it as a negative example**.
  It is exported like any other class.
- `F` flag: the object is not usable yet (box covers more than one object, or the box is off). It goes to a
  separate bin for later inspection.

With `--max-per-class N`, a class stops appearing in the queue once it has `N` accepted crops and the footer shows
`Class accepted/N`. Pressing the key of a full class shows a red "is full" message.

---

## 5. Classes and `classes.yaml`

[`classes.yaml`](classes.yaml) lists the default classes, their order and their shortcut keys:

```yaml
classes:
  - {name: Pedestrian, key: "1"}
  - {name: Car,        key: "2"}
  ...
  - {name: Leguna,     key: "8"}
  - {name: no_vehicle, key: "0"}
```

- A class whose `name` matches a **COCO category name** is *reviewed*: its detections are queued.
- A class with no matching COCO category (`no_vehicle`, `Leguna`, `CNG`, ...) is *label-only*: you can assign it,
  but nothing is queued for it.
- COCO categories **not listed** are ignored completely (for example `Vehicle` is not annotated).
- `key` is optional (auto-assigned if missing) and must be one character, not one of `c x s u q n f h`.
- Use another file with `--classes other.yaml`.

Classes created at run time with `N` are stored in `<out>/classes.json` and come back automatically on the next
session. If a saved class's key later clashes with a key in `classes.yaml`, the yaml wins and the saved class is
moved to a free key (the script prints a note). Add a class permanently by putting it in `classes.yaml`.

---

## 6. Output layout

```
cls_dataset/                       # --out
├── decisions.json                 # every decision {annotation_id: {action, class}}  (resume state)
├── classes.json                   # classes added at run time with N
├── added.json                     # objects drawn in the viewer
├── corrected_instances.json       # COCO file with the verified objects (see below)
├── Pedestrian/
│   └── DJI_0266_merged_frame_0001_27771.png     # <image>_<annotation id>.png
├── Car/
├── no_vehicle/
└── ...

cls_dataset_flagged/               # <out>_flagged, kept OUTSIDE the dataset on purpose
├── flagged.json                   # annotation id, image, bbox, predicted class, score
└── Pedestrian/                    # grouped by predicted class, with extra context around the box
```

- `cls_dataset/` can be loaded directly with e.g. `torchvision.datasets.ImageFolder`.
  Flagged crops live in a sibling folder so they are never mistaken for a class.
- `corrected_instances.json` is rebuilt on every export. It holds the confirmed / re-classed originals (with their
  segmentation), re-annotated boxes (`"source": "redraw"`, `parent_id` = original id) and added boxes
  (`"source": "added"`); rejected, flagged and undecided detections are left out. `instances.json` is never modified.
- Crops are cut from the full-resolution image with 8 px of padding (`--pad`).
- The export is rebuilt from `decisions.json` every time. Do not hand-edit the class folders. To change a decision,
  re-run the reviewer or edit `decisions.json` and run `--export-only`.

`decisions.json` entries look like:

```json
{"27771": {"action": "confirm", "class": "Car"},
 "27772": {"action": "change",  "class": "Truck"},
 "27773": {"action": "reject"},
 "27774": {"action": "flag"},
 "27775": {"action": "redraw", "boxes": [{"bbox": [10, 20, 30, 40], "class": "Car"}]}}
```

---

## 7. Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--dataset DIR` | `dataset` | COCO dataset directory (`images/`, `annotations/instances.json`) |
| `--out DIR` | `cls_dataset` | Where decisions and crops go (flagged crops go to `DIR_flagged`) |
| `--classes FILE` | `classes.yaml` | Default classes and shortcut keys |
| `--class-order A B ...` | yaml order | Review these predicted classes first, in this order |
| `--max-per-class N` | off | Stop each class after `N` accepted crops. Also shuffles image order (see `--seed`) so picks spread across frames |
| `--seed N` | `0` | Shuffle seed used with `--max-per-class` (keep it the same between sessions) |
| `--min-score X` | `0` | Skip detections with a lower score |
| `--min-size PX` | `0` | Skip boxes whose shorter side is below `PX` |
| `--pad PX` | `8` | Extra pixels around the box in exported crops |
| `--history` | off | Start in history mode (recheck / edit earlier decisions) |
| `--recheck CLASS...` | off | Start in history mode, filtered to these decided classes (also `rejected`, `flagged`, `redrawn`) |
| `--view` | off | Start in the per-image viewer |
| `--ui-scale X` | `1.6` | Size of text and widgets in the box editor and viewer (`1` = small, `2` = very large) |
| `--only-flagged` | off | Re-review only the objects flagged earlier; a new decision replaces the flag |
| `--export-only` | off | Rebuild the crops from `decisions.json` and exit (no window) |

---

## 8. Troubleshooting

**`cv2.error: ... The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support`**
You are running an old version of the script that used OpenCV windows. The current `review.py` uses Tkinter.
Update the script; keep `opencv-python-headless`.

**`ModuleNotFoundError: No module named 'tkinter'`**
`sudo apt install python3-tk`.

**`TclError: no display name and no $DISPLAY environment variable`**
You are on a machine without a desktop (plain SSH). Run the reviewer on your desktop, or use X forwarding
(`ssh -X`). The tool needs a graphical session.

**Keys do nothing**
Click the window once so it has keyboard focus. On Wayland, Tk runs through XWayland and focus rules can be strict.

**Text looks different from Ubuntu**
The Ubuntu fonts were not found at `/usr/share/fonts/truetype/ubuntu/`; DejaVu Sans is used instead.
Install `fonts-ubuntu` to get them.

**Image loading feels slow**
The source frames are 3840×2160 PNGs (~10 MB). The queue keeps objects from the same image together so each
image is decoded once per run of objects. Use `--min-score` / `--min-size` to cut down the number of objects.

**`hf: command not found`** or **`hf buckets` is unknown**
`pip install -U "huggingface_hub>=1.26"` (the `hf` CLI ships with it; bucket commands need a recent version).

**`401` / `403` when downloading**
Run `hf auth login` with a token that can read `hamimmahmud0/SAM_COCO_v1_b2_3024`.

**Start over**
Delete `cls_dataset/` (and `cls_dataset_flagged/`). Nothing else holds state.
