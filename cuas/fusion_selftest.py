import argparse
import glob
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_id as M

W, H, FPS, DUR = 640, 480, 30.0, 5.0


def sprite_from_crop(path):
    im = cv2.imread(path)
    if im is None:
        return None
    g = cv2.GaussianBlur(im, (3, 3), 0).astype(np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = np.median(border, axis=0)
    d = np.linalg.norm(g - bg, axis=2)
    thr = max(20.0, 3 * float(np.median(np.linalg.norm(border - bg, axis=1))) + 10)
    m = (d > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    c = np.array([lab.shape[1] / 2, lab.shape[0] / 2])
    best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA] / (1 + np.hypot(*(np.array(
        [st[i, 0] + st[i, 2] / 2, st[i, 1] + st[i, 3] / 2]) - c)) / 10))
    x, y, w, h = st[best, :4]
    if w * h < 9:
        return None
    alpha = cv2.GaussianBlur((lab == best).astype(np.float32), (3, 3), 0)
    return im[y:y + h, x:x + w], alpha[y:y + h, x:x + w]


def paste(frame, spr, alpha, cx, cy, ext, squash=1.0):
    h, w = spr.shape[:2]
    s = ext / max(w, h)
    nw, nh = max(2, int(round(w * s))), max(2, int(round(h * s * squash)))
    sp = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    al = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)[..., None]
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + nw), min(H, y0 + nh)
    if xa >= xb or ya >= yb:
        return
    f = frame[ya:yb, xa:xb]
    a = al[ya - y0:yb - y0, xa - x0:xb - x0]
    frame[ya:yb, xa:xb] = f * (1 - a) + sp[ya - y0:yb - y0, xa - x0:xb - x0] * a


def path(kind, ext, rng):
    if kind in ("glide_bird", "flap_bird"):
        U = rng.uniform(3, 6) * ext
        hd0, turn = rng.uniform(0, 2 * math.pi), np.cumsum(rng.normal(0, 1.2, int(DUR * FPS) + 2)) / FPS * 2.5
        pts, p = [], np.array([W / 2, H / 2])
        for i in range(len(turn)):
            hd = hd0 + turn[i]
            p = p + np.array([math.cos(hd), math.sin(hd)]) * U / FPS
            if not (60 < p[0] < W - 60):
                hd0 += math.pi
            if not (60 < p[1] < H - 60):
                hd0 += math.pi
            pts.append(p.copy())
        return pts
    pts, p, hd = [], np.array([W / 2, H / 2]) + rng.uniform(-40, 40, 2), rng.uniform(0, 2 * math.pi)
    plan = [(1.2, 4), (1.3, 0), (1.2, 4), (1.3, 0)]
    for d, u in plan:
        for _ in range(int(d * FPS)):
            p = p + np.array([math.cos(hd), math.sin(hd)]) * u * ext / FPS
            p = np.clip(p, 40, [W - 40, H - 40])
            pts.append(p + rng.normal(0, 0.03 * ext, 2))
        if u:
            hd += math.pi / 2
    return pts


def clip(kind, ext, sprite, seed):
    rng = np.random.default_rng(seed)
    spr, al = sprite
    sky = np.zeros((H, W, 3), np.float32)
    top, bot = np.array([215, 170, 120]), np.array([235, 215, 195])
    for y in range(H):
        sky[y] = top + (bot - top) * y / H
    pts = path(kind, ext, rng)
    fl = rng.uniform(4.5, 7.0)
    for i in range(int(DUR * FPS)):
        t = i / FPS
        f = sky.copy()
        squash = 0.35 + 0.65 * abs(math.cos(math.pi * fl * t)) if kind == "flap_bird" else 1.0
        paste(f, spr, al, *pts[min(i, len(pts) - 1)], ext, squash)
        f += rng.normal(0, 2.0, f.shape).astype(np.float32)
        yield t, np.clip(f, 0, 255).astype(np.uint8)


_APP = None


def job(args):
    global _APP
    kind, ext, sp_path, seed, use_app = args
    if use_app and _APP is None:
        from appearance_id import AppearanceClassifier
        _APP = AppearanceClassifier()
    spr = sprite_from_crop(sp_path)
    labels = []

    def on(t, res, g, trs, off):
        if res:
            tr = max(res, key=lambda q: len(q.t))
            labels.append((t, tr.last[0], tr.last[2]))

    M.run(clip(kind, ext, spr, seed), FPS, None, on_frame=on, appearance=_APP if use_app else None)
    tail = [(c, w) for t, c, w in labels if t >= DUR - 2.0]
    final = Counter(c for c, _ in tail).most_common(1)[0][0] if tail else "none"
    why = Counter(w for c, w in tail if c == final).most_common(1)[0][0] if tail else ""
    return args, (final, why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=r"C:\Users\user\cuas_ext\crops_test")
    ap.add_argument("--n", type=int, default=8, help="크기·종류별 사진 수")
    a = ap.parse_args()
    rnd = random.Random(0)
    pools = {}
    for cls in ("bird", "drone"):
        fs = sorted(glob.glob(os.path.join(a.crops, cls, "*.jpg")))
        rnd.shuffle(fs)
        ok = []
        for f in fs:
            s = sprite_from_crop(f)
            if s is not None and max(s[0].shape[:2]) >= 12:
                ok.append(f)
        pools[cls] = ok
    print("쓸 수 있는 시험 사진:", {k: len(v) for k, v in pools.items()})
    kinds = [("glide_bird", "bird"), ("flap_bird", "bird"), ("drone", "drone")]
    sizes = [8, 12, 16, 24, 32, 48]
    jobs = []
    for kind, cls in kinds:
        for ext in sizes:
            for j in range(a.n):
                f = pools[cls][j % len(pools[cls])]
                for use_app in (False, True):
                    jobs.append((kind, ext, f, 1000 * ext + j, use_app))
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        full = dict(ex.map(job, jobs))
    done = {k: v[0] for k, v in full.items()}
    truth = {"glide_bird": "bird", "flap_bird": "bird", "drone": "drone"}
    print(f"\n크기(최대폭)별 맞힌 수 / {a.n}   [움직임만 → 움직임+외형]")
    print(f"{'':12}" + "".join(f"{s:>9}px" for s in sizes))
    for kind, _ in kinds:
        row = f"{kind:12}"
        for ext in sizes:
            r = []
            for use_app in (False, True):
                r.append(sum(done[(kind, ext, pools[truth[kind]][j % len(pools[truth[kind]])], 1000 * ext + j, use_app)]
                             == truth[kind] for j in range(a.n)))
            row += f"{r[0]:>5}→{r[1]:<4}"
        print(row)
        wrong = Counter(v for (k, e, f, s, u), v in done.items() if k == kind and u and v != truth[kind])
        print(f"{'':12}  (외형 포함 오답: {dict(wrong)})")
    print("\n외형 포함 오답 상세:")
    for (k, e, f, sd, u), (v, w) in sorted(full.items()):
        if u and v != truth[k]:
            print(f"  {k:10} {e:>2}px {os.path.basename(f)[:34]:34} -> {v:8} {w}")


if __name__ == "__main__":
    main()
