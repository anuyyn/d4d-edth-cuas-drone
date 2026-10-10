import json
import os
import random

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mydrone_photos")
OUT = os.path.join(HERE, "mydrone_crops")
VAL_IDX = {2, 8, 13, 22, 30}
OUT_PX = 160


def crop(img, cx, cy, w, h):
    H, W = img.shape[:2]
    w, h = min(w, W), min(h, H)
    x0 = int(min(max(cx - w / 2, 0), W - w))
    y0 = int(min(max(cy - h / 2, 0), H - h))
    return cv2.resize(img[y0:y0 + int(h), x0:x0 + int(w)], (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA), (x0, y0, x0 + w, y0 + h)


def inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def main():
    rnd = random.Random(0)
    boxes = json.load(open(os.path.join(HERE, "mydrone_boxes.json"), encoding="utf-8"))
    n = {}
    for i, name in enumerate(sorted(boxes)):
        split = "val" if i in VAL_IDX else "train"
        img = cv2.imread(os.path.join(SRC, name))
        H, W = img.shape[:2]
        x0, y0, x1, y1 = boxes[name]
        bw, bh = x1 - x0, y1 - y0
        bcx, bcy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(bw, bh)
        stem = os.path.splitext(name)[0]
        for k in range(40 if split == "train" else 12):
            s = side * rnd.uniform(1.1, 2.6)
            w, h = s, s
            if rnd.random() < 0.25:
                w, h = (s * 4 / 3, s) if rnd.random() < 0.5 else (s, s * 4 / 3)
            cx = bcx + rnd.uniform(-0.15, 0.15) * bw
            cy = bcy + rnd.uniform(-0.15, 0.15) * bh
            c, _ = crop(img, cx, cy, w, h)
            d = os.path.join(OUT, split, "drone"); os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{stem}_{k}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n[(split, "drone")] = n.get((split, "drone"), 0) + 1
        got, tries = 0, 0
        while got < (25 if split == "train" else 8) and tries < 2000:
            tries += 1
            s = W * rnd.uniform(0.12, 0.6)
            w, h = (s * 4 / 3, s) if rnd.random() < 0.2 else (s, s)
            c, r = crop(img, rnd.uniform(0, W), rnd.uniform(0, H), w, h)
            if inter(r, (x0, y0, x1, y1)) > 0.05 * bw * bh:
                continue
            d = os.path.join(OUT, split, "none"); os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{stem}_bg{got}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            got += 1
            n[(split, "none")] = n.get((split, "none"), 0) + 1
    print(n)


if __name__ == "__main__":
    main()
