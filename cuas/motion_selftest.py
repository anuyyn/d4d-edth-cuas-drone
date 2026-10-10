import argparse
import math
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_id as M

W, H, FPS, DUR = 640, 480, 30.0, 6.0
SS = 4


def sky(rng):
    g = np.linspace(205, 165, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    return g


def stamp(frame, x, y, draw, size):
    r = int(size * 1.2 + 4)
    x0, y0 = int(math.floor(x)) - r, int(math.floor(y)) - r
    n = 2 * r + 1
    big = np.zeros((n * SS, n * SS), np.uint8)
    cx, cy = (x - x0) * SS, (y - y0) * SS
    draw(big, cx, cy)
    cov = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
    if xa >= xb or ya >= yb:
        return
    c = cov[ya - y0:yb - y0, xa - x0:xb - x0]
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - c) + 60.0 * c


def ell(cx, cy, ax, ay, ang=0):
    def f(img):
        cv2.ellipse(img, (int(cx * 16), int(cy * 16)), (max(1, int(ax * 16)), max(1, int(ay * 16))),
                    ang, 0, 360, 255, -1, cv2.LINE_AA, 4)
    return f


def scenario(kind, S, rng, wind):
    wx, wy = wind
    x0, y0 = W * 0.5 + rng.uniform(-60, 60), H * 0.5 + rng.uniform(-40, 40)

    if kind == "balloon":
        U = rng.uniform(4, 12) * S
        x0 -= wx * U * DUR / 2; y0 += wy * U * DUR / 2
        ph = rng.uniform(0, 6.28)

        def at(t):
            g = 1 + 0.05 * math.sin(0.7 * t + ph)
            x = x0 + wx * U * t * g
            y = y0 - wy * U * t * g + 0.3 * S * math.sin(2 * math.pi * 0.3 * t + ph)
            sx = SS * S / 2
            return x, y, lambda img, cx, cy: ell(cx, cy, sx, sx * 1.1)(img)
        return at

    if kind in ("bird", "glide_bird"):
        U = rng.uniform(8, 20) * S
        hd0 = rng.uniform(0, 2 * math.pi)
        fl = rng.uniform(4.5, 7.0)
        turn = np.cumsum(rng.normal(0, 0.9, int(DUR * FPS) + 2)) / FPS * 2.0
        pos = [(x0 - math.cos(hd0) * U * DUR / 2, y0 - math.sin(hd0) * U * DUR / 2)]
        for i in range(len(turn)):
            hd = hd0 + turn[i]
            px, py = pos[-1]
            pos.append((px + math.cos(hd) * U / FPS, py + math.sin(hd) * U / FPS))

        def at(t):
            i = min(len(pos) - 1, int(t * FPS))
            x, y = pos[i]
            hd = math.degrees(hd0 + turn[min(i, len(turn) - 1)])
            span = 1.0 if kind == "glide_bird" else 0.35 + 0.65 * abs(math.cos(math.pi * fl * t))

            def draw(img, cx, cy):
                ell(cx, cy, SS * S * 0.18, SS * S * 0.1, hd)(img)
                ell(cx, cy, SS * S * 0.06, SS * S * 0.5 * span, hd)(img)
            return x, y, draw
        return at

    if kind in ("drone", "drone_cruise"):
        segs = []
        hd = rng.uniform(0, 2 * math.pi)
        if kind == "drone":
            plan = [(1.5, rng.uniform(10, 25)), (1.2, 0.0), (1.5, rng.uniform(10, 25)), (1.8, 0.0)]
        else:
            hd = math.atan2(wy, -wx)
            plan = [(DUR, rng.uniform(6, 12))]
        t_acc, x, y = 0.0, x0, y0
        for i, (d, u) in enumerate(plan):
            segs.append((t_acc, t_acc + d, x, y, hd, u * S))
            x += math.cos(hd) * u * S * d; y += math.sin(hd) * u * S * d
            t_acc += d
            if u > 0 and kind == "drone":
                hd += math.pi / 2 * rng.choice([-1, 1])
        jit = rng.normal(0, 0.08 * S, (int(DUR * FPS) + 2, 2))

        def at(t):
            for a, b, sx, sy, h, u in segs:
                if a <= t < b or b >= DUR - 1e-9:
                    break
            k = t - a
            i = min(len(jit) - 1, int(t * FPS))
            x = sx + math.cos(h) * u * k + jit[i, 0]
            y = sy + math.sin(h) * u * k + jit[i, 1]

            def draw(img, cx, cy):
                ell(cx, cy, SS * S * 0.5, SS * S * 0.2)(img)
            return x, y, draw
        return at
    raise ValueError(kind)


def cam_offset(t):
    return 25 * math.sin(0.9 * t) + 8 * t, 12 * math.sin(1.7 * t)


def make_clip(kind, S, seed, wind, pan=False):
    rng = np.random.default_rng(seed)
    at = scenario(kind, S, rng, wind)
    base = sky(rng)
    noise = [rng.normal(0, 2.0, (H, W)).astype(np.float32) for _ in range(12)]
    xs = [at(t)[0] for t in np.arange(0, DUR, 0.5)]
    ys = [at(t)[1] for t in np.arange(0, DUR, 0.5)]
    dx, dy = W / 2 - (min(xs) + max(xs)) / 2, H / 2 - (min(ys) + max(ys)) / 2
    for i in range(int(DUR * FPS)):
        t = i / FPS
        x, y, draw = at(t)
        f = base.copy()
        stamp(f, x + dx, y + dy, draw, S)
        if pan:
            ox, oy = cam_offset(t)
            f = cv2.warpAffine(f, np.float32([[1, 0, ox], [0, 1, oy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        f += noise[int(rng.integers(len(noise)))]
        yield t, np.clip(f, 0, 255).astype(np.uint8)


def gyro_offset(seed):
    rng = np.random.default_rng(seed + 7)
    n = int(DUR * FPS) + 2
    drift = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0)
    q = rng.normal(0, 0.9, (n, 2))

    def f(t):
        i = min(n - 1, int(round(t * FPS)))
        ox, oy = cam_offset(t)
        return ox + drift[i, 0] + q[i, 0], oy + drift[i, 1] + q[i, 1]
    return f


def evaluate(kind, S, seed, wind_known, pan=None, save=None):
    wd = 30.0
    r = math.radians(wd)
    wind = (math.cos(r), math.sin(r))
    labels = []
    first_ok = [None]
    truth = {"glide_bird": "bird", "drone_cruise": "drone"}.get(kind, kind)
    vw = None
    if save:
        vw = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H), False)

    def on_frame(t, res, gray, tracks, off):
        if vw is not None:
            vw.write(gray)
        if not res:
            return
        tr = max(res, key=lambda q: len(q.t))
        labels.append((t, tr.last[0]))
        if first_ok[0] is None and tr.last[0] == truth:
            first_ok[0] = t

    ego = gyro_offset(seed) if pan == "gyro" else pan
    M.run(make_clip(kind, S, seed, wind, bool(pan)), FPS, wind if wind_known else None, ego=ego, on_frame=on_frame)
    if vw is not None:
        vw.release()
    tail = [c for t, c in labels if t >= DUR - 2.0]
    final = Counter(tail).most_common(1)[0][0] if tail else "none"
    return truth, final, first_ok[0]


def _job(args):
    return args, evaluate(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--save", default=None, help="시험 영상 저장 폴더")
    a = ap.parse_args()
    sizes = [3, 5, 8, 15]
    kinds = ["bird", "balloon", "drone"]
    hard = ["glide_bird", "drone_cruise"]
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for k in kinds + hard:
            evaluate(k, 8, 1, True, save=os.path.join(a.save, f"{k}_8px.mp4"))
        print("저장:", a.save)

    conds = [(True, None, "바람방향 앎, 고정 카메라"), (False, None, "바람방향 모름, 고정 카메라"),
             (True, "gyro", "바람방향 앎, 카메라 흔들림 + FC 자이로 보정(잡음·표류 포함)"),
             (True, "image", "바람방향 앎, 카메라 흔들림 + 화면 보정(빈 하늘)")]
    jobs = [(k, S, 100 * S + sd, wk, pan) for wk, pan, _ in conds for k in kinds + hard
            for S in sizes for sd in range(a.seeds)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        done = dict(ex.map(_job, jobs))
    ok_main = True
    for wk, pan, title in conds:
        print(f"\n=== {title} | 크기별 맞힌 수 / {a.seeds}  (괄호: 첫 정답까지 평균 초) ===")
        print(f"{'표적':14}" + "".join(f"{s:>4}px      " for s in sizes))
        for k in kinds + hard:
            row = f"{k:14}"
            for S in sizes:
                res = [done[(k, S, 100 * S + sd, wk, pan)] for sd in range(a.seeds)]
                n_ok = sum(1 for tr_, fin, _ in res if fin == tr_)
                tt = [f for _, fin, f in res if f is not None]
                wrong = Counter(fin for tr_, fin, _ in res if fin != tr_)
                row += f"{n_ok}/{a.seeds}({np.mean(tt):.1f}s)" if tt else f"{n_ok}/{a.seeds}(-)    "
                row += " " if not wrong else "*"
                if k in kinds and pan is None and n_ok < a.seeds * 0.8 and S >= 5:
                    ok_main = False
                if wrong:
                    row += "".join(f"{w[:2]}{c}" for w, c in wrong.items()) + " "
                else:
                    row += "    "
            print(row)
    print("\n* 뒤 = 틀린 판정(bi=새 ba=풍선 dr=무인기 un=미상 no=판정없음)과 횟수. glide_bird·drone_cruise는 한계 확인용(틀려도 됨).")
    print("selftest", "통과" if ok_main else "실패", "(기준: 5px 이상 기본 3종 80% 이상, 흔들림 없는 경우)")
    return 0 if ok_main else 1


if __name__ == "__main__":
    sys.exit(main())
