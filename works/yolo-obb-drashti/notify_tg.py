"""Watch an Ultralytics run's results.csv and send per-epoch metrics to Telegram.

Credentials come from env vars (never stored in files): TG_TOKEN, TG_CHAT_ID.
Usage (on kaggle-vm, detached):
  TG_TOKEN=... TG_CHAT_ID=... nohup python notify_tg.py --run /kaggle/working/runs/l1280 > notify.log 2>&1 &
Sends one message per new epoch (mAP50, mAP50-95, losses) and a final one when --done-after seconds pass with no new epoch.
"""
import argparse
import csv
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


def send(text):
    url = f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": os.environ["TG_CHAT_ID"], "text": text}).encode()
    try:
        urllib.request.urlopen(url, data, timeout=20)
    except Exception as e:
        print("send failed:", e, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir containing results.csv")
    ap.add_argument("--tag", default=None, help="label in messages (default: run dir name)")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--done-after", type=int, default=3600, help="seconds without a new epoch => send 'stalled/finished'")
    a = ap.parse_args()
    run, tag = Path(a.run), a.tag or Path(a.run).name
    seen, last_new, best = 0, time.time(), 0.0
    send(f"[{tag}] notifier started")
    while True:
        f = run / "results.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            for r in rows[seen:]:
                r = {k.strip(): v for k, v in r.items()}
                m5, m = float(r["metrics/mAP50(B)"]), float(r["metrics/mAP50-95(B)"])
                star = " *best*" if m > best else ""
                best = max(best, m)
                send(f"[{tag}] epoch {r['epoch']}: mAP50={m5:.4f} mAP50-95={m:.4f} "
                     f"P={float(r['metrics/precision(B)']):.3f} R={float(r['metrics/recall(B)']):.3f} "
                     f"train box/cls/ang={float(r['train/box_loss']):.3f}/{float(r['train/cls_loss']):.3f}/"
                     f"{float(r.get('train/angle_loss', 0)):.3f} t={float(r['time'])/3600:.2f}h{star}")
            if len(rows) > seen:
                last_new = time.time()
            seen = len(rows)
        if time.time() - last_new > a.done_after:
            send(f"[{tag}] no new epoch for {a.done_after}s - finished or stalled ({seen} epochs, best mAP50-95 {best:.4f})")
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
