import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mydrone_photos")
OUT = os.path.join(HERE, "mydrone_boxes.json")


def box_of(img):
    H, W = img.shape[:2]
    s = 400 / W
    sm = cv2.resize(img, (400, int(H * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] < 75) & (hsv[..., 1] < 120)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    best, bs = None, -1
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 300:
            continue
        d = np.hypot((cen[i][0] - w / 2) / w, (cen[i][1] - h / 2) / h)
        score = a * (1.2 - d)
        if score > bs:
            best, bs = (x, y, x + bw, y + bh), score
    if best is None:
        return [int(W * .2), int(H * .25), int(W * .8), int(H * .75)]
    return [int(v / s) for v in best]


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
    old = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    boxes = {}
    tiles = []
    for f in files:
        n = os.path.basename(f)
        img = cv2.imread(f)
        b = old.get(n) or box_of(img)
        boxes[n] = b
        t = img.copy()
        cv2.rectangle(t, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 12)
        t = cv2.resize(t, (240, 320))
        cv2.putText(t, str(len(tiles)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        tiles.append(t)
    json.dump(boxes, open(OUT, "w", encoding="utf-8"), indent=1)
    while len(tiles) % 9:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)]
    cv2.imwrite(os.path.join(HERE, "mydrone_boxes_check.jpg"), np.vstack(rows))
    print(len(boxes), "박스 →", OUT)


if __name__ == "__main__":
    main()
