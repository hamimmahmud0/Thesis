"""Static overlay renders of the sampled b1 frames (runs on remote)."""
import json, os
from PIL import Image, ImageDraw, ImageFont
P = "webapp/public/b1"; OUT = "/root/work/b1_overlays"; os.makedirs(OUT, exist_ok=True)
COL = {1: "#ff5252", 2: "#b388ff", 3: "#40c4ff", 4: "#ffab40", 5: "#69f0ae", 6: "#ffeb3b", 7: "#ff80ab", 8: "#18ffff"}
D = json.load(open(f"{P}/data.json")); font = ImageFont.load_default(size=14)
for f in D["frames"]:
    im = Image.open(P + f["img"][3:]).convert("RGBA"); ov = Image.new("RGBA", im.size); d = ImageDraw.Draw(ov)
    for a in f["anns"]:
        c = COL[a["c"]]; pts = [tuple(p) for p in a["p"]] if a["p"] else None
        if pts: d.polygon(pts, fill=c + "55", outline=c)
        else: d.rectangle([a["b"][0], a["b"][1], a["b"][0] + a["b"][2], a["b"][1] + a["b"][3]], outline=c)
    im = Image.alpha_composite(im, ov).convert("RGB"); d = ImageDraw.Draw(im)
    x = 10
    for cid, n in D["categories"].items():
        k = sum(1 for a in f["anns"] if str(a["c"]) == cid)
        t = f"{n} {k}"; d.rectangle([x, 8, x + d.textlength(t, font) + 26, 32], fill="black"); d.rectangle([x + 4, 14, x + 16, 26], fill=COL[int(cid)])
        d.text((x + 22, 12), t, fill="white", font=font); x += int(d.textlength(t, font)) + 36
    im.save(f"{OUT}/{os.path.splitext(f['file'])[0]}_overlay.jpg", quality=88)
print(os.listdir(OUT))
