"""Contact sheets (5x5, ID-labelled) + CSV for outputs/sel100."""
import csv, json
from PIL import Image, ImageDraw, ImageFont
D = "outputs/sel100"; sel = json.load(open(f"{D}/selection.json"))
font = ImageFont.load_default(size=26); small = ImageFont.load_default(size=15)
C, G, COLS = 230, 8, 5
sheets = []
for pre, name in (("R", "Rickshaw"), ("M", "Motorcycle")):
    items = [s for s in sel if s["id"].startswith(pre)]
    for s0 in range(0, len(items), 25):
        part = items[s0:s0 + 25]; rows = (len(part) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (C + G) + G, rows * (C + G) + G), (30, 30, 30)); d = ImageDraw.Draw(sheet)
        for k, s in enumerate(part):
            im = Image.open(f"{D}/crops/{s['id']}.png"); k_ = min(C / im.width, C / im.height, 3.0); im = im.resize((max(1, int(im.width * k_)), max(1, int(im.height * k_))), Image.LANCZOS)
            x, y = G + (k % COLS) * (C + G), G + (k // COLS) * (C + G)
            sheet.paste(im, (x + (C - im.width) // 2, y + (C - im.height) // 2))
            d.rectangle([x, y, x + 78, y + 32], fill=(0, 0, 0)); d.text((x + 4, y + 1), s["id"], fill=(255, 235, 59), font=font)
            d.text((x + 4, y + C - 20), f"{s['score']:.2f}", fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
        p = f"{D}/sheet_{pre}_{s0 // 25 + 1}.png"; sheet.save(p); sheets.append((p, f"{name} {part[0]['id']}–{part[-1]['id']}"))
with open(f"{D}/selection.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "class", "score", "file", "bbox_xywh", "ann_id"])
    for s in sel: w.writerow([s["id"], s["cat"], s["score"], s["file"], s["bbox"], s["ann_id"]])
json.dump(sheets, open(f"{D}/sheets.json", "w"))
print(len(sheets), "sheets")
