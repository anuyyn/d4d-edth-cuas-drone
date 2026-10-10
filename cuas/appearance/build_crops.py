import argparse
import glob
import hashlib
import io
import os
import random

from PIL import Image

CTX = 2.0
MIN_SIDE = 12
BIRD_PER_IMG = 3


def split_of(key):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 75 else ("val" if h < 87 else "test")


def base_name(path):
    n = os.path.basename(path)
    return n.split("_jpg.rf.")[0].split("_png.rf.")[0].split(".rf.")[0]


def square(cx, cy, side, W, H):
    side = max(MIN_SIDE, min(side, W, H))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, x0 + int(side), y0 + int(side)


def overlaps(b, boxes):
    return any(not (b[2] <= x0 or b[0] >= x1 or b[3] <= y0 or b[1] >= y1) for x0, y0, x1, y1 in boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/cuas_data"))
    ap.add_argument("--out", default=os.path.expanduser("~/cuas_data/crops"))
    a = ap.parse_args()
    rnd = random.Random(0)
    n = {}

    def save(img, split, cls, name):
        d = os.path.join(a.out, split, cls)
        os.makedirs(d, exist_ok=True)
        img.convert("RGB").save(os.path.join(d, name + ".jpg"), quality=92)
        n[(split, cls)] = n.get((split, cls), 0) + 1

    imgs = sorted(glob.glob(f"{a.root}/dronebird/*/images/*"))
    for ip in imgs:
        lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        try:
            im = Image.open(ip)
            im.load()
        except Exception:
            continue
        W, H = im.size
        sp = split_of(base_name(ip))
        stem = os.path.basename(ip).rsplit(".", 1)[0][:60]
        boxes, rows = [], []
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            bx = (x * W, y * H, w * W, h * H)
            boxes.append((bx[0] - bx[2] / 2, bx[1] - bx[3] / 2, bx[0] + bx[2] / 2, bx[1] + bx[3] / 2))
            rows.append((c, bx))
        birds = [r for r in rows if r[0] == 0]
        rnd.shuffle(birds)
        keep = birds[:BIRD_PER_IMG] + [r for r in rows if r[0] == 1]
        sides = []
        for i, (c, (cx, cy, w, h)) in enumerate(keep):
            side = max(w, h) * CTX
            sides.append(side)
            save(im.crop(square(cx, cy, side, W, H)), sp, "bird" if c == 0 else "drone", f"{stem}_{i}")
        side = rnd.choice(sides) if sides else 64
        for _ in range(20):
            cx, cy = rnd.uniform(0, W), rnd.uniform(0, H)
            b = square(cx, cy, side, W, H)
            if not overlaps(b, boxes):
                save(im.crop(b), sp, "none", f"{stem}_bg")
                break

    import pyarrow.parquet as pq
    t = pq.read_table(glob.glob(f"{a.root}/balloon/data/*.parquet")[0])
    col = t.column("image").to_pylist()
    for i, rec in enumerate(col):
        im = Image.open(io.BytesIO(rec["bytes"]))
        save(im, split_of(f"balloon{i}"), "balloon", f"imnet_balloon_{i}")

    for sp in ["train", "val", "test"]:
        print(sp, {c: n.get((sp, c), 0) for c in ["bird", "balloon", "drone", "none"]})


if __name__ == "__main__":
    main()
