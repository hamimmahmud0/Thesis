#!/usr/bin/env python3
"""Frozen DINOv3 features for every crop in <data>/split.csv  ->  <out>.npz  (no training of the backbone).

Per preprocessing variant (stretch = the model's own processor, i.e. resize to 224x224; pad = pad to a square with
the mean colour first, keeps the aspect ratio) it stores  cls_<v>  (CLS token)  and  mean_<v>  (mean of the patch tokens,
register tokens excluded), both float32 and not normalised. Uses the model's own AutoImageProcessor (the sat493m
checkpoints use other mean/std than lvd1689m).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def pad_square(im, fill):
    s = max(im.size)
    c = Image.new("RGB", (s, s), fill)
    c.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return c


@torch.no_grad()
def embed(model, proc, paths, pre, bs, dtype, n_reg):
    fill = tuple(int(round(m * 255)) for m in proc.image_mean)
    cls, mean = [], []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        if pre == "pad":
            ims = [pad_square(im, fill) for im in ims]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(model.device, dtype)
        h = model(pixel_values=x).last_hidden_state.float()
        cls.append(h[:, 0].cpu()); mean.append(h[:, 1 + n_reg:].mean(1).cpu())
    return torch.cat(cls).numpy(), torch.cat(mean).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--pre", nargs="+", default=["stretch", "pad"])
    ap.add_argument("--check-fp16", action="store_true", help="also compare fp16 against fp32 features on 64 crops")
    a = ap.parse_args()
    df = pd.read_csv(Path(a.data) / "split.csv")
    paths = [str(Path(a.data) / p) for p in df.path]
    dtype = getattr(torch, a.dtype)
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype=dtype, device_map="auto" if a.dtype == "float16" else None)
    if a.dtype == "float32":
        model.to("cuda")
    model.eval()
    n_reg = getattr(model.config, "num_register_tokens", 0)
    t_load = time.time() - t0
    print(f"loaded {a.model} dtype={a.dtype} register_tokens={n_reg} hidden={model.config.hidden_size} in {t_load:.0f}s", flush=True)
    out, t1 = {}, time.time()
    for pre in a.pre:
        c, m = embed(model, proc, paths, pre, a.bs, dtype, n_reg)
        assert np.isfinite(c).all() and np.isfinite(m).all(), "NaN/inf in features"
        out[f"cls_{pre}"], out[f"mean_{pre}"] = c, m
        print(f"{pre}: cls {c.shape}  |cls| mean {np.linalg.norm(c, axis=1).mean():.1f}", flush=True)
    t_ext = time.time() - t1
    info = {"model": a.model, "dtype": a.dtype, "dim": model.config.hidden_size, "register_tokens": n_reg,
            "load_s": round(t_load, 1), "extract_s": round(t_ext, 1), "n": len(paths),
            "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3)}
    if a.check_fp16:
        del model; torch.cuda.empty_cache()
        m16 = AutoModel.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
        c16, _ = embed(m16, proc, paths[:64], "stretch", a.bs, torch.float16, n_reg)
        c32 = out["cls_stretch"][:64]
        cos = (c16 * c32).sum(1) / (np.linalg.norm(c16, axis=1) * np.linalg.norm(c32, axis=1))
        info["fp16_vs_fp32_cos_min"] = float(cos.min()); print("fp16 vs fp32 cosine min", cos.min(), flush=True)
    np.savez_compressed(a.out, ids=df.path.values, label=df.label.values, split=df.split.values, group=df.group.values,
                        info=json.dumps(info), **out)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
