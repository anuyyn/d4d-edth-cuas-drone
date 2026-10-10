import argparse
import glob
import math
import os
import random

import cv2
import numpy as np

S2 = 256
BALLOON_C = [(240, 240, 240), (228, 230, 232), (215, 215, 220), (200, 200, 205), (232, 236, 242),
             (170, 160, 190), (150, 140, 175), (190, 190, 195)]


def sky(rnd):
    top = np.array([rnd.uniform(150, 245), rnd.uniform(140, 235), rnd.uniform(110, 225)])
    bot = top + rnd.uniform(-35, 35)
    g = np.linspace(0, 1, S2)[:, None, None]
    return np.repeat(top * (1 - g) + bot * g, S2, axis=1).astype(np.float32)


def blend(img, mask, color, shade=None, alpha=1.0):
    col = np.array(color, np.float32)[None, None]
    if shade is not None:
        col = np.clip(col * shade[..., None], 0, 255)
    a = (np.clip(mask, 0, 1) * alpha)[..., None]
    img[:] = img * (1 - a) + col * a


def balloon(img, cx, cy_top, w, L, color, rnd):
    m = np.zeros(img.shape[:2], np.float32)
    r = w / 2
    pts = [(cx + r * math.cos(t), cy_top + r - r * math.sin(t)) for t in np.linspace(0, math.pi, 24)]
    side = [(cx - r * (1 - u) ** 1.3, cy_top + r + (L - r) * u) for u in np.linspace(0, 1, 16)]
    side_r = [(cx + r * (1 - u) ** 1.3, cy_top + r + (L - r) * u) for u in np.linspace(1, 0, 16)]
    poly = np.array(pts + side + side_r, np.float32)
    cv2.fillPoly(m, [(poly * 8).astype(np.int32)], 1.0, cv2.LINE_AA, 3)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    shade = 1.0 + 0.12 * np.clip(1 - np.hypot(xx - (cx - r * 0.4), yy - (cy_top + r * 0.6)) / (r * 1.2), 0, 1) \
        - 0.22 * np.clip((xx - cx) / r, 0, 1) - 0.1 * np.clip((yy - cy_top - r) / L, 0, 1)
    blend(img, m, color, shade, 1.0 if rnd.random() < 0.8 else rnd.uniform(0.6, 0.9))
    return cx, cy_top + L


def bag(img, cx, cy_top, bw, bh, rnd, dark):
    m = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(m, (int(cx * 8), int((cy_top + bh * 0.62) * 8)), (int(bw / 2 * 8), int(bh * 0.42 * 8)), 0, 0, 360, 1.0,
                -1, cv2.LINE_AA, 3)
    cv2.rectangle(m, (int(cx - bw * 0.42), int(cy_top + bh * 0.18)), (int(cx + bw * 0.42), int(cy_top + bh * 0.65)), 1.0,
                  -1)
    knot = np.array([[cx - bw * 0.15, cy_top + bh * 0.2], [cx + bw * 0.15, cy_top + bh * 0.2], [cx, cy_top]], np.float32)
    cv2.fillPoly(m, [(knot * 8).astype(np.int32)], 1.0, cv2.LINE_AA, 3)
    if dark:
        v = rnd.uniform(30, 95)
        blend(img, m, (v, v * rnd.uniform(0.9, 1.1), v * rnd.uniform(0.85, 1.15)))
    else:
        v = rnd.uniform(195, 245)
        blend(img, m, (v, v, v * rnd.uniform(0.95, 1.02)), alpha=rnd.uniform(0.75, 0.95))
        for _ in range(rnd.randint(3, 9)):
            px = cx + rnd.uniform(-0.35, 0.35) * bw
            py = cy_top + rnd.uniform(0.35, 0.9) * bh
            g = rnd.uniform(90, 170)
            sm = np.zeros_like(m)
            cv2.ellipse(sm, (int(px), int(py)), (max(1, int(bw * rnd.uniform(0.05, 0.15))),
                                                 max(1, int(bh * rnd.uniform(0.04, 0.1)))), rnd.uniform(0, 180), 0,
                        360, 1.0, -1, cv2.LINE_AA)
            blend(img, sm * m, (g, g, g), alpha=rnd.uniform(0.25, 0.55))
        edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        blend(img, edge, (v * 0.75, v * 0.75, v * 0.75), alpha=0.6)


def device(img, cx, cy, s, rnd):
    v = rnd.uniform(200, 240)
    cv2.rectangle(img, (int(cx - s / 2), int(cy - s * 0.6)), (int(cx + s / 2), int(cy)), (v, v, v), -1, cv2.LINE_AA)
    cv2.rectangle(img, (int(cx - s / 2), int(cy - s * 0.6)), (int(cx + s / 2), int(cy)), (v * 0.7,) * 3, 1, cv2.LINE_AA)


def scene(bg, rnd, kind):
    img = bg.copy()
    if kind == "falling":
        bw = S2 * rnd.uniform(0.18, 0.45)
        bag(img, S2 / 2 + rnd.uniform(-0.1, 0.1) * S2, S2 / 2 - bw * 0.7, bw, bw * rnd.uniform(1.0, 1.5), rnd,
            dark=rnd.random() < 0.35)
        M = cv2.getRotationMatrix2D((S2 / 2, S2 / 2), rnd.uniform(-60, 60), 1.0)
        img = cv2.warpAffine(img, M, (S2, S2), borderMode=cv2.BORDER_REFLECT)
        return img
    n = rnd.choice([1, 2, 2, 3])
    total = S2 * (rnd.uniform(0.35, 0.9) if kind != "bagfocus" else rnd.uniform(1.6, 3.0))
    has_bag = kind != "plain"
    L = total * (0.55 if has_bag else 0.85)
    w = L / rnd.uniform(1.6, 2.3)
    cx0 = S2 / 2 + rnd.uniform(-0.1, 0.1) * S2
    if kind == "bagfocus":
        top = S2 * 0.55 - total + rnd.uniform(-0.1, 0.1) * S2
    else:
        top = S2 / 2 - total / 2
    color = rnd.choice(BALLOON_C)
    necks = []
    for i in range(n):
        dx = (i - (n - 1) / 2) * w * rnd.uniform(0.7, 1.0)
        necks.append(balloon(img, cx0 + dx, top + rnd.uniform(-0.08, 0.08) * L, w, L,
                             color if rnd.random() < 0.8 else rnd.choice(BALLOON_C), rnd))
    line_c = (rnd.uniform(60, 160),) * 3
    if not has_bag:
        if rnd.random() < 0.5:
            nx, ny = necks[0]
            cv2.line(img, (int(nx), int(ny)), (int(nx + rnd.uniform(-10, 10)), int(ny + total * 0.3)), line_c, 1, cv2.LINE_AA)
        return img
    hang = total * rnd.uniform(0.12, 0.25)
    lx, ly = cx0 + rnd.uniform(-0.1, 0.1) * w, max(n_[1] for n_ in necks) + hang
    for nx, ny in necks:
        cv2.line(img, (int(nx), int(ny)), (int(lx), int(ly)), line_c, 1, cv2.LINE_AA)
    dark = rnd.random() < 0.35
    nb = rnd.choice([1, 2, 2, 3])
    bw = total * rnd.uniform(0.1, 0.17)
    device(img, lx, ly + bw * 0.3, bw * 0.6, rnd)
    for j in range(nb):
        ox = lx + (j - (nb - 1) / 2) * bw * 0.9
        bag(img, ox, ly + bw * 0.3, bw, bw * rnd.uniform(1.1, 1.6), rnd, dark)
    return img


def finish(img, rnd):
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), rnd.uniform(0.3, 0.9))
    img = img + np.random.normal(0, rnd.uniform(1, 4), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=os.path.expanduser("~/cuas_data/crops"))
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    rnd = random.Random(1)
    np.random.seed(1)
    for split in ("train", "val"):
        for old in glob.glob(f"{a.crops}/{split}/balloon/synth_*.jpg") + glob.glob(f"{a.crops}/{split}/trash/synth_*.jpg"):
            os.remove(old)
    for split, n in [("train", a.n), ("val", a.n // 8)]:
        bgs = glob.glob(f"{a.crops}/{split}/none/*.jpg")
        os.makedirs(f"{a.crops}/{split}/trash", exist_ok=True)
        cnt = {"trash": 0, "balloon": 0}
        for i in range(n):
            bg = cv2.resize(cv2.imread(rnd.choice(bgs)), (S2, S2)).astype(np.float32) if bgs and rnd.random() < 0.6 \
                else sky(rnd)
            kind = rnd.choices(["full", "bagfocus", "falling", "plain"], [0.4, 0.2, 0.15, 0.25])[0]
            cls = "balloon" if kind == "plain" else "trash"
            cv2.imwrite(f"{a.crops}/{split}/{cls}/synth_{kind}_{i:05d}.jpg", finish(scene(bg, rnd, kind), rnd),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            cnt[cls] += 1
        print(split, cnt)


if __name__ == "__main__":
    main()
