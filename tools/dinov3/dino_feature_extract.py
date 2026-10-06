"""Extract DINOv3 features for one or more images and print the feature matrix (N x D, CLS token)."""
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--model", default="facebook/dinov3-vit7b16-pretrain-sat493m")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(args.model)
model = AutoModel.from_pretrained(args.model).to(dev).eval()

imgs = [Image.open(p).convert("RGB") for p in args.images]
inputs = proc(images=imgs, return_tensors="pt").to(dev)
with torch.no_grad():
    feats = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()

np.set_printoptions(precision=4, suppress=True, threshold=200, linewidth=140)
print(f"device={dev} model={args.model}")
print(f"feature matrix shape: {feats.shape}")
print(feats)
