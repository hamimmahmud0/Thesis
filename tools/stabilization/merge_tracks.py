#!/usr/bin/env python3
"""Merge per-window CoTracker3 track npz chunks into one tracks npz.

Used by the windowed-grid pipeline (motion_from_tracks.py --window): the
grid/query points are RE-DEFINED every --window frames (default 100), so
each window is tracked separately with queries at its anchor frame. Chunks
SHARE the boundary anchor frame (arrangement 0-99, 99-199, 199-299, ...).
This tool stitches chunk npz files into a single (T, N_total, 2) npz where
each chunk's points are visible only within its window span.
motion_from_tracks.py then re-defines its correspondence grid at every
anchor (points visible at the anchor frame) and chains the per-window
motion through the shared anchor frames into one global trajectory.

Produce the chunks (pick one):
  a) cut the video into overlapping chunks that share anchor frames
     (ffmpeg -frames:v / -ss), track each with the query grid at its
     LOCAL frame 0 (COT3 default), or
  b) track the full video once per anchor with grid_query_frame=anchor
     and keep only [anchor, next_anchor] frames per run.

Chunk offsets: default assumes SHARED anchor frames, i.e.
offset_{i+1} = offset_i + len_i - 1 (0-99, 99-199, ...). Override with
--offsets f0 f1 ... or by putting "frame_offset" in each chunk's meta.

All chunks must report the same width/height (meta). Output npz keys:
tracks, visibility, query_points (anchor-frame positions), meta
(width/height/fps + query_frames list). Compatible with
motion_from_tracks.py and stabilize.py (same schema as a single-run npz).
"""

import argparse
import json
import sys

import numpy as np


def load_chunk(path):
    d = np.load(path, allow_pickle=True)
    tracks = np.asarray(d["tracks"], dtype=np.float32)
    if tracks.ndim == 4 and tracks.shape[0] == 1:
        tracks = tracks[0]
    vis = np.asarray(d["visibility"], dtype=bool)
    if vis.ndim == 3 and vis.shape[0] == 1:
        vis = vis[0]
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        sys.exit(f"{path}: tracks must be (T,N,2), got {tracks.shape}")
    if vis.shape != tracks.shape[:2]:
        sys.exit(f"{path}: visibility {vis.shape} does not match "
                 f"tracks {tracks.shape}")
    meta = {}
    if "meta" in d.files:
        try:
            meta = json.loads(str(d["meta"]))
        except Exception:
            meta = {}
    qp = None
    if "query_points" in d.files:
        qp = np.asarray(d["query_points"], dtype=np.float32)
    return tracks, vis, qp, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chunks", nargs="+",
                    help="chunk npz files, chronological order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", type=int, nargs="+", default=None,
                    help="global start frame of each chunk (default: shared "
                         "anchor frames, offset_{i+1} = offset_i + len_i - 1; "
                         "or per-chunk meta frame_offset)")
    args = ap.parse_args()

    chunks = [load_chunk(p) for p in args.chunks]
    meta0 = chunks[0][3]
    for p, (_, _, _, m) in zip(args.chunks, chunks):
        if (m.get("width"), m.get("height")) != \
                (meta0.get("width"), meta0.get("height")):
            sys.exit(f"{p}: meta {m.get('width')}x{m.get('height')} differs "
                     f"from {meta0.get('width')}x{meta0.get('height')}")

    n = len(chunks)
    offsets = args.offsets
    if offsets is None:
        offsets = [m.get("frame_offset") for _, _, _, m in chunks]
        if all(o is not None for o in offsets):
            offsets = [int(o) for o in offsets]
        else:
            offsets = [0]
            for i in range(1, n):
                offsets.append(offsets[-1] + chunks[i - 1][0].shape[0] - 1)
    if len(offsets) != n:
        sys.exit(f"--offsets needs {n} values, got {len(offsets)}")
    for i in range(1, n):
        if offsets[i] <= offsets[i - 1]:
            sys.exit("chunk offsets must be strictly increasing")
        if offsets[i] > offsets[i - 1] + chunks[i - 1][0].shape[0]:
            sys.exit(f"gap between chunk {i - 1} and {i} "
                     f"(missing frames) - chunks must tile the video")

    T = max(o + c[0].shape[0] for o, c in zip(offsets, chunks))
    N = sum(c[0].shape[1] for c in chunks)
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    vis = np.zeros((T, N), dtype=bool)
    qp = np.zeros((N, 2), dtype=np.float32)
    j = 0
    for o, (tr, vv, q, m) in zip(offsets, chunks):
        k = tr.shape[1]
        tracks[o:o + tr.shape[0], j:j + k] = tr
        vis[o:o + tr.shape[0], j:j + k] = vv
        # Keep finite (but invisible) positions outside each chunk's span.
        if o > 0:
            tracks[:o, j:j + k] = tr[0]
        if o + tr.shape[0] < T:
            tracks[o + tr.shape[0]:, j:j + k] = tr[-1]
        if q is not None and q.shape == (k, 2):
            qp[j:j + k] = q
        else:
            qp[j:j + k] = tr[0]
        j += k

    meta = dict(
        width=meta0.get("width"), height=meta0.get("height"),
        fps=meta0.get("fps"), query_frames=offsets, merged=True, n_chunks=n,
    )
    np.savez(args.out, tracks=tracks, visibility=vis, query_points=qp,
             meta=json.dumps(meta))
    for p, o, c in zip(args.chunks, offsets, chunks):
        print(f"chunk {p}: offset {o}, {c[0].shape[0]} frames, "
              f"{c[0].shape[1]} points")
    print(f"merged: T={T}, N={N} -> {args.out}")


if __name__ == "__main__":
    main()
