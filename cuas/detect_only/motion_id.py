import argparse
import csv
import math
import socket
import sys
import time

import cv2
import numpy as np

P = dict(
    bg_kernel=31,
    k_sigma=5.0,
    abs_thr=12.0,
    min_area=1, max_area=60000,
    gate_px=25.0,
    max_miss=6,
    window_s=2.0,
    min_track_s=1.0,
    flap_fmin=2.0, flap_fmax=15.0,
    flap_r=0.40,
    flap_depth=0.12,
    hover_bl=0.6,
    move_bl=2.0,
    fast_bl=60.0,
)


def detect(img, roi_mask=None):
    f = img.astype(np.float32)
    k = P["bg_kernel"]
    small = cv2.resize(f, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    up = lambda z: cv2.resize(z, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)

    def diff(bg_small):
        d = np.abs(f - up(bg_small))
        return d.max(axis=2) if d.ndim == 3 else d

    def thresh(d):
        sub = d[::4, ::4]
        sigma = 1.4826 * float(np.median(np.abs(sub - np.median(sub)))) + 1e-3
        return max(P["abs_thr"], P["k_sigma"] * sigma)

    d = diff(cv2.blur(small, (k, k)))
    thr = thresh(d)
    ms = cv2.resize(cv2.dilate((d > thr).astype(np.uint8), np.ones((9, 9), np.uint8)), (small.shape[1], small.shape[0]),
                    interpolation=cv2.INTER_AREA)
    w = (ms < 0.01).astype(np.float32)
    wb = cv2.blur(w, (k, k))
    if small.ndim == 3:
        bg2 = cv2.blur(small * w[..., None], (k, k)) / np.maximum(wb, 1e-3)[..., None]
        bg2 = np.where((wb > 0.05)[..., None], bg2, cv2.blur(small, (k, k)))
    else:
        bg2 = cv2.blur(small * w, (k, k)) / np.maximum(wb, 1e-3)
        bg2 = np.where(wb > 0.05, bg2, cv2.blur(small, (k, k)))
    d = diff(bg2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if roi_mask is not None:
        m &= roi_mask
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    mass = np.bincount(lab.ravel(), weights=d.ravel(), minlength=n)
    W_ = min(m.shape)
    long_px = max(20, int(0.12 * W_))
    lines = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) > 0.6 * W_ or (max(bw, bh) > long_px and a / (bw * bh) < 0.08):
            lines.append(i)
    near_line = None
    if lines:
        lm = np.isin(lab, lines).astype(np.uint8)
        lm = cv2.dilate(lm, np.ones((7, 7), np.uint8))
        near_line = np.bincount(lab.ravel(), weights=lm.ravel().astype(np.float64), minlength=n)
    lum = f.max(axis=2) if f.ndim == 3 else f
    sat = np.bincount(lab.ravel(), weights=(lum >= 250).ravel().astype(np.float64), minlength=n)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if a >= 6 and sat[i] / a > 0.3:
            continue
        if i in lines:
            continue
        if near_line is not None and near_line[i] / a > 0.25:
            continue
        x0_, y0_ = int(st[i, 0]), int(st[i, 1])
        if a > 12 and (x0_ <= 1 or y0_ <= 1 or x0_ + bw >= m.shape[1] - 1 or y0_ + bh >= m.shape[0] - 1):
            continue
        if P["min_area"] <= a <= P["max_area"]:
            out.append([float(cen[i, 0]), float(cen[i, 1]), a, float(mass[i]),
                        [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])]])
    out.sort(key=lambda d: -d[3])
    keep = []
    for d in out:
        if d[2] > 4 or all(math.hypot(d[0] - k[0], d[1] - k[1]) > 1.5 * math.sqrt(k[2]) + k_halo(k) or d[3] > 0.5 * k[3]
                           for k in keep):
            keep.append(d)
    return group_hanging(drop_line_chains(keep))


def drop_line_chains(dets):
    small = [d for d in dets if d[2] <= 400]
    bad = set()
    for i, d in enumerate(small):
        r = 4 * max(4.0, math.sqrt(d[2])) + 10
        nb = [q for j, q in enumerate(small) if j != i and math.hypot(q[0] - d[0], q[1] - d[1]) < r
              and 0.25 < q[2] / max(d[2], 1) < 4]
        found = False
        for a_ in range(len(nb)):
            for b_ in range(a_ + 1, len(nb)):
                v1 = (nb[a_][0] - d[0], nb[a_][1] - d[1])
                v2 = (nb[b_][0] - d[0], nb[b_][1] - d[1])
                c = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2) + 1e-9)
                if c < -0.94:
                    found = True
                    bad.update({id(d), id(nb[a_]), id(nb[b_])})
                    break
            if found:
                break
    return [d for d in dets if id(d) not in bad]


def group_hanging(dets):
    dets = sorted(dets, key=lambda d: -d[2])
    used = set()
    out = []
    for i, d in enumerate(dets):
        if i in used:
            continue
        x0, y0, x1, y1 = d[4]
        w, h = x1 - x0, y1 - y0
        box = [x0, y0, x1, y1]
        hang = 0
        if d[2] >= 30:
            for j in range(i + 1, len(dets)):
                if j in used:
                    continue
                q = dets[j]
                if (y1 - 0.1 * h <= q[1] <= y1 + 1.5 * h and abs(q[0] - (x0 + x1) / 2) < 0.6 * w
                        and 0.01 * d[2] <= q[2] <= 0.4 * d[2]):
                    used.add(j)
                    hang += 1
                    box = [min(box[0], q[4][0]), min(box[1], q[4][1]), max(box[2], q[4][2]), max(box[3], q[4][3])]
        out.append(((box[0] + box[2]) / 2 if hang else d[0], (box[1] + box[3]) / 2 if hang else d[1], d[2], d[3],
                    max(box[2] - box[0], box[3] - box[1]), hang))
    return out


def k_halo(k):
    return P["bg_kernel"] / 2 if k[2] >= 4 else 2.0


class Track:
    _next = 1

    def __init__(self, t, x, y, a, m, ext=0, hang=0):
        self.id = Track._next
        Track._next += 1
        self.t, self.x, self.y, self.a, self.m, self.e, self.h = [t], [x], [y], [a], [m], [ext], [hang]
        self.miss = 0
        self.max_disp = 0.0
        self.last = ("unknown", 0.0, "")

    def predict(self):
        if len(self.x) >= 2:
            return 2 * self.x[-1] - self.x[-2], 2 * self.y[-1] - self.y[-2]
        return self.x[-1], self.y[-1]

    def add(self, t, x, y, a, m, ext=0, hang=0):
        self.t.append(t); self.x.append(x); self.y.append(y); self.a.append(a); self.m.append(m); self.e.append(ext)
        self.h.append(hang)
        self.max_disp = max(self.max_disp, math.hypot(x - self.x[0], y - self.y[0]))

    def recent_move(self, n):
        x0, y0 = self.x[-n] if len(self.x) >= n else self.x[0], self.y[-n] if len(self.y) >= n else self.y[0]
        return max(math.hypot(x - x0, y - y0) for x, y in zip(self.x[-n:], self.y[-n:]))

    def hanging(self):
        h = [v for v, a in zip(self.h[-30:], self.a[-30:]) if a > 0]
        return sum(1 for v in h if v > 0) / len(h) if h else 0.0

    def extent_px(self):
        e = [v for v in self.e[-30:] if v > 0]
        return float(np.percentile(e, 75)) if e else self.size_px()

    def size_px(self):
        a = [v for v in self.a[-60:] if v > 0]
        return max(1.5, math.sqrt(float(np.median(a)))) if a else 1.5


class Tracker:
    def __init__(self):
        self.tracks = []

    def step(self, t, dets):
        pairs = []
        for ti, tr in enumerate(self.tracks):
            px, py = tr.predict()
            for di, (x, y, *_r) in enumerate(dets):
                dist = math.hypot(x - px, y - py)
                if dist < P["gate_px"] + 2 * tr.size_px():
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            self.tracks[ti].add(t, *dets[di])
            self.tracks[ti].miss = 0
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                px, py = tr.predict()
                tr.add(t, px, py, 0, 0.0)
                tr.miss += 1
        self.tracks = [tr for tr in self.tracks if tr.miss <= P["max_miss"]]
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(t, *d))
        return self.tracks


def periodicity(sig, fps):
    x = np.asarray(sig, dtype=float)
    if len(x) < 8 or x.mean() <= 0:
        return 0.0, 0.0, 0.0
    x = x / x.mean()
    depth = float(np.std(x))
    w = max(3, int(round(fps / P["flap_fmin"])) | 1)
    trend = np.convolve(np.pad(x, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    y = x - trend[: len(x)]
    den = float(np.dot(y, y)) + 1e-9
    kmin = max(2, int(math.floor(fps / min(P["flap_fmax"], fps / 2.2))))
    kmax = min(len(y) // 2, int(math.ceil(fps / P["flap_fmin"])))
    best, bk = 0.0, 0
    r = [float(np.dot(y[:-k], y[k:])) / den * len(y) / (len(y) - k) for k in range(1, kmax + 2)]
    for k in range(kmin, kmax + 1):
        rk = r[k - 1]
        if rk > best and rk >= r[k - 2] and rk >= r[k]:
            best, bk = rk, k
    return best, (fps / bk if bk else 0.0), depth


def features(tr, fps, wind=None):
    n = int(round(P["window_s"] * fps))
    t = np.array(tr.t[-n:]); x = np.array(tr.x[-n:]); y = np.array(tr.y[-n:])
    m = np.array(tr.m[-n:])
    L = tr.size_px()
    f = {"L": L, "dur": float(t[-1] - t[0])}
    f["flap_r"], f["flap_hz"], f["depth"] = periodicity(m, fps)

    step = max(1, int(round(fps / 5)))
    nb = len(x) // step
    if nb < 4:
        return None
    sl = slice(len(x) - nb * step, None)
    xs = x[sl].reshape(nb, step).mean(1)
    ys = y[sl].reshape(nb, step).mean(1)
    ts = t[sl].reshape(nb, step).mean(1)
    if len(xs) < 4:
        return None
    vx, vy = np.diff(xs) / np.diff(ts), np.diff(ys) / np.diff(ts)
    sp = np.hypot(vx, vy) / L
    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    vbar = np.array([vx.mean(), vy.mean()])
    vb = float(np.hypot(*vbar)) / L
    f["speed"] = vb
    f["straight"] = disp / path if path > 1e-6 else 0.0
    f["cv"] = float(np.mean(np.hypot(vx - vbar[0], vy - vbar[1]))) / L / max(vb, 1e-6)
    moving = sp > P["move_bl"]
    turns = [0.0]
    for i in range(1, len(vx)):
        if moving[i] and moving[i - 1]:
            c = (vx[i] * vx[i - 1] + vy[i] * vy[i - 1]) / (math.hypot(vx[i], vy[i]) * math.hypot(vx[i - 1], vy[i - 1]))
            turns.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    f["turn"] = max(turns)
    smax = float(sp.max())
    f["hover"] = float(np.mean(sp < P["hover_bl"])) if smax > P["move_bl"] else 0.0
    f["still"] = smax < P["hover_bl"] * 2
    f["spd_range"] = (smax - float(sp.min())) / max(float(sp.mean()), 1e-6)
    if wind is not None and vb > P["hover_bl"]:
        f["wind_cos"] = float((vbar[0] * wind[0] - vbar[1] * wind[1]) / (np.hypot(*vbar) + 1e-9))
    else:
        f["wind_cos"] = None
    return f


def clip01(v):
    return max(0.0, min(1.0, v))


def classify(f):
    if f is None:
        return "unknown", 0.0, "항적 짧음"
    if f["flap_r"] >= P["flap_r"] and f["depth"] >= P["flap_depth"]:
        conf = 0.5 + 0.5 * clip01((f["flap_r"] - P["flap_r"]) / 0.4)
        return "bird", round(conf, 2), f"날갯짓 {f['flap_hz']:.1f}Hz r={f['flap_r']:.2f}"
    ev_d = {
        "정지비행": f["hover"] >= 0.25,
        "급선회": clip01((f["turn"] - 35) / 40),
        "급가감속": clip01((f["spd_range"] - 0.8) / 0.8),
        "고속": clip01((f["speed"] - P["fast_bl"]) / P["fast_bl"]),
    }
    if f["wind_cos"] is not None:
        ev_d["역풍"] = clip01(-f["wind_cos"] / 0.5)
    ev_d = {k: float(v) for k, v in ev_d.items()}
    d_key = max(ev_d, key=ev_d.get)
    drone = ev_d[d_key]
    ev_b = [
        clip01((f["straight"] - 0.80) / 0.15),
        clip01(1 - (f["cv"] - 0.15) / 0.35),
        clip01(1 - (f["turn"] - 20) / 40),
        clip01(1 - (f["speed"] - P["fast_bl"] / 2) / (P["fast_bl"] / 2)),
        clip01((f["speed"] - P["hover_bl"]) / 1.0),
    ]
    if f["wind_cos"] is not None:
        ev_b.append(clip01((f["wind_cos"] - 0.6) / 0.3))
    balloon = 0.5 * float(np.mean(ev_b)) + 0.5 * float(np.min(ev_b))
    if drone >= 0.5 and drone >= balloon:
        return "drone", round(0.5 + 0.45 * drone, 2), f"{d_key}"
    if balloon >= 0.6 and drone < 0.4:
        why = f"등속직선 곧음={f['straight']:.2f} 변동={f['cv']:.2f}"
        if f["wind_cos"] is not None:
            why += f" 바람일치={f['wind_cos']:.2f}"
        return "balloon", round(0.5 + 0.45 * balloon, 2), why
    return "unknown", round(max(drone, balloon), 2), f"애매(무인기 {drone:.2f} / 풍선 {balloon:.2f})"


APP_MIN_PX = 6
APP_FULL_PX = 24
TARGETS = ("bird", "balloon", "drone", "trash")


WEAK_DRONE = ("급선회", "급가감속", "고속")
STRONG_DRONE = ("정지비행", "역풍")


def fuse(motion, app_p, size_px, recon=False):
    m_cls, m_conf, m_why = motion
    if recon and app_p is not None and size_px >= APP_MIN_PX and app_p.get("none", 0) >= 0.5:
        return "none", round(app_p["none"], 2), "외형상 표적 없음"
    if app_p is None or app_p.get("none", 0) > 0.6 or size_px < APP_MIN_PX:
        return motion
    cls_set = [c for c in TARGETS if c in app_p]
    w = 0.7 * clip01((size_px - APP_MIN_PX) / (APP_FULL_PX - APP_MIN_PX))
    if m_cls == "drone" and m_why in WEAK_DRONE:
        w = min(0.85, w * 1.6 + 0.15)
    if max(app_p[c] for c in cls_set) >= 0.9:
        w = max(w, 0.6)
    if recon:
        w = max(w, 0.85)
    elif m_cls == "drone" and m_why in STRONG_DRONE:
        w = min(w, 0.45)
    tot = sum(app_p[c] for c in cls_set) or 1.0
    pa = {c: app_p[c] / tot for c in cls_set}
    k = len(cls_set)
    if m_cls == "balloon" and "trash" in cls_set:
        pm = {c: (m_conf / 2 if c in ("balloon", "trash") else (1 - m_conf) / (k - 2)) for c in cls_set}
    elif m_cls in cls_set:
        pm = {c: (m_conf if c == m_cls else (1 - m_conf) / (k - 1)) for c in cls_set}
    else:
        pm = {c: 1 / k for c in cls_set}
        w = max(w, 0.5) if size_px >= APP_FULL_PX / 2 else w
    pf = {c: w * pa[c] + (1 - w) * pm[c] for c in cls_set}
    cls = max(pf, key=pf.get)
    a_cls = max(pa, key=pa.get)
    why = f"{m_why} | 외형 {a_cls} {pa[a_cls]:.2f} (비중 {w:.2f})"
    if pf[cls] < 0.45:
        return "unknown", round(pf[cls], 2), why
    return cls, round(pf[cls], 2), why


def payload_rule(last, hang_frac, app_p=None):
    cls, conf, why = last
    lift = app_p is not None and app_p.get("balloon", 0) + app_p.get("trash", 0) >= 0.6
    if hang_frac >= 0.3 and (cls in ("balloon", "trash") or (cls == "unknown" and lift)):
        return "trash", round(max(conf, 0.5 + 0.4 * hang_frac), 2), f"{why} | 매달린 물체 {hang_frac:.0%}"
    return last


def confirmed(tr, fps, recon=False):
    if recon:
        return tr.t[-1] - tr.t[0] >= 0.5 and sum(1 for a in tr.a[-int(fps):] if a > 0) >= 0.5 * fps
    return (tr.t[-1] - tr.t[0] >= P["min_track_s"] and tr.max_disp > max(3.0, 1.5 * tr.size_px())
            and sum(1 for a in tr.a[-int(fps):] if a > 0) >= 0.3 * fps)


class SimpleDetector:

    def __init__(self, appearance, hist=5, min_ext=4, none_thr=0.6, min_conf=0.6, show_balloon=False):
        from collections import deque
        self.show_balloon = show_balloon
        self.app, self.min_ext, self.none_thr, self.min_conf = appearance, min_ext, none_thr, min_conf
        self.hist = deque(maxlen=hist)

    def step(self, img):
        W_ = img.shape[1]
        cand = [d for d in detect(img) if self.min_ext <= d[4] <= 0.3 * W_]
        if not cand:
            self.hist.append(None)
            return None
        x, y, a, m, ext, hang = max(cand, key=lambda d: d[3] * (1 + 0.5 * (d[5] > 0)))
        p = self.app.crop_probs(img, x, y, ext)
        if p.get("none", 0) >= self.none_thr:
            self.hist.append(None)
            return None
        cls = max((c for c in p if c in TARGETS), key=p.get)
        conf = p[cls]
        if hang and cls in ("balloon", "trash", "bird"):
            cls, conf = "trash", max(conf, p.get("balloon", 0) + p.get("trash", 0), 0.7)
        self.hist.append((cls, conf))
        seen = [h for h in self.hist if h]
        from collections import Counter
        maj = Counter(c for c, _ in seen).most_common(1)[0][0]
        mconf = float(np.mean([cf for c, cf in seen if c == maj]))
        if mconf < self.min_conf:
            return None
        if maj == "balloon" and not self.show_balloon:
            return None
        return (int(x - ext / 2), int(y - ext / 2), int(x + ext / 2), int(y + ext / 2)), maj, round(mconf, 2)


LAST_FRAME = None


class Pipeline:

    def __init__(self, fps, wind=None, ego=None, roi=None, appearance=None, app_every=5, recon=False,
                 scene_every=3, scene_n=5, scene_thr=0.6, max_classify=0):
        self.fps, self.wind, self.ego, self.roi = fps, wind, ego, roi
        self.scene_every, self.scene_thr = scene_every, scene_thr
        from collections import deque
        self.scene_hist = deque(maxlen=scene_n)
        self.scene = None
        self.max_classify = max_classify
        self.n = 0
        self.appearance, self.app_every, self.recon = appearance, app_every, recon
        self.trk = Tracker()
        self.off = np.zeros(2)
        self.prev = None
        self.roi_mask = None
        self.gray = None

    def step(self, t, img, hi=None):
        global LAST_FRAME
        LAST_FRAME = img if hi is None else hi
        hk = 1.0 if hi is None else hi.shape[1] / img.shape[1]
        src = img if hi is None else hi
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        if self.roi is not None and self.roi_mask is None:
            self.roi_mask = np.zeros(gray.shape, np.uint8)
            x0, y0, w, h = self.roi
            self.roi_mask[y0:y0 + h, x0:x0 + w] = 1
        dets = detect(img if img.ndim == 3 else gray, self.roi_mask)
        if self.ego == "image":
            small = cv2.resize(gray, None, fx=0.25, fy=0.25).astype(np.float32)
            if self.prev is not None:
                (sx, sy), _ = cv2.phaseCorrelate(self.prev, small)
                self.off += np.array([sx, sy]) * 4
            self.prev = small
        elif callable(self.ego):
            self.off = np.asarray(self.ego(t), dtype=float)
        off = self.off
        if self.ego:
            dets = [(x - off[0], y - off[1], *r) for x, y, *r in dets]
        tracks = self.trk.step(t, dets)
        res = []
        app, recon = self.appearance, self.recon
        big = None
        if self.max_classify:
            live = [q for q in tracks if q.miss == 0 and confirmed(q, self.fps, recon)]
            big = set(id(q) for q in sorted(live, key=lambda q: -q.extent_px())[:self.max_classify])
        for tr in tracks:
            if big is not None and id(tr) not in big:
                continue
            if confirmed(tr, self.fps, recon) and tr.miss == 0:
                tr.last = classify(features(tr, self.fps, self.wind))
                ext = tr.extent_px() * hk
                if app is not None and ext >= APP_MIN_PX:
                    if len(tr.t) % self.app_every == 0 or getattr(tr, "app_p", None) is None:
                        tr.app_p = app.crop_probs(src, (tr.x[-1] + off[0]) * hk, (tr.y[-1] + off[1]) * hk, ext)
                    tr.last = fuse(tr.last, tr.app_p, ext, recon)
                tr.last = payload_rule(tr.last, tr.hanging(), getattr(tr, "app_p", None))
                if tr.last[0] == "none" or (recon and (app is None or ext < APP_MIN_PX or tr.last[1] < 0.6)):
                    continue
                if recon and ext < 12 and tr.recent_move(int(1.5 * self.fps)) * hk < 3:
                    continue
                res.append(tr)
        self.n += 1
        if res or app is None or not recon:
            self.scene_hist.clear()
            self.scene = None
        elif self.n % self.scene_every == 0:
            self.scene_hist.append(app.probs(src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)))
            avg = {c: float(np.mean([h[c] for h in self.scene_hist])) for c in self.scene_hist[0]}
            best = max((c for c in avg if c in TARGETS), key=avg.get)
            ok = len(self.scene_hist) >= 3 and avg[best] >= self.scene_thr and avg[best] > avg.get("none", 0)
            self.scene = (best, round(avg[best], 2)) if ok else None
        return res

    @property
    def tracks(self):
        return self.trk.tracks

    def box(self, tr):
        x, y, e = tr.x[-1] + self.off[0], tr.y[-1] + self.off[1], tr.extent_px()
        return int(x - e / 2), int(y - e / 2), int(x + e / 2), int(y + e / 2)


def run(frames, fps, wind=None, ego=None, roi=None, on_frame=None, appearance=None, app_every=5, recon=False):
    p = Pipeline(fps, wind, ego, roi, appearance, app_every, recon)
    for t, img in frames:
        res = p.step(t, img)
        if on_frame:
            on_frame(t, res, p.gray, p.tracks, p.off)
    return p.trk


def video_frames(src, fps_hint, shrink, max_s):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"영상 열기 실패: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not (1 < fps < 1000):
        fps = fps_hint
    live = src.isdigit() or "://" in src

    def gen():
        i, t0 = 0, time.monotonic()
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if shrink != 1.0:
                f = cv2.resize(f, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
            t = time.monotonic() - t0 if live else i / fps
            if max_s and t > max_s:
                break
            yield t, f
            i += 1
    return gen(), fps


COLORS = {"bird": (0, 200, 0), "balloon": (0, 200, 255), "drone": (0, 0, 255), "trash": (255, 0, 255),
          "unknown": (180, 180, 180)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="영상 파일 / 카메라 번호 / tcp://… 스트림")
    ap.add_argument("--fps", type=float, default=30.0, help="영상에 fps 정보가 없을 때")
    ap.add_argument("--shrink", type=float, default=1.0, help="프레임 축소 배율(먼 표적 흉내)")
    ap.add_argument("--wind-dir", type=float, default=None, help="화면 기준 바람 방향(도), 0=오른쪽 90=위")
    ap.add_argument("--roi", default=None, help="x,y,w,h")
    ap.add_argument("--appearance", default=None, help="외형 분류 ONNX (appearance/out/cuas_cls.onnx) — 표적이 크면 외형도 반영")
    ap.add_argument("--recon", action="store_true", help="정찰드론이 표적을 따라가며 찍는 경우: 외형 위주(움직임은 카메라 움직임이 섞여 보조)")
    ap.add_argument("--ego", action="store_const", const="image", default=None,
                    help="카메라 흔들림을 화면 무늬로 보정(빈 하늘에선 안 됨 → 탑재 시엔 FC 자이로 연동 필요)")
    ap.add_argument("--udp", default=None, help="host:port 로 판정+캡처 이미지(JSON) 송신 → ground_station(9500)")
    ap.add_argument("--csv", default=None, help="프레임별 판정 기록")
    ap.add_argument("--show", action="store_true", help="판정 화면 표시 (q 종료)")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from motion_selftest import main as st
        sys.exit(st())
    if not a.src:
        ap.error("--src 또는 --selftest")

    wind = None
    if a.wind_dir is not None:
        r = math.radians(a.wind_dir)
        wind = (math.cos(r), math.sin(r))
    roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
    frames, fps = video_frames(a.src, a.fps, a.shrink, a.seconds)

    sender = None
    if a.udp:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from detect_msg import Sender
        sender = Sender(a.udp, src="motion")
    wr = None
    if a.csv:
        fcsv = open(a.csv, "w", newline="", encoding="utf-8-sig")
        wr = csv.writer(fcsv)
        wr.writerow(["t", "track", "cls", "conf", "why", "x", "y", "size_px"])
    last_print = {}

    def on_frame(t, res, gray, tracks, off):
        if res:
            main_tr = max(res, key=lambda tr: (tr.last[0] != "unknown", len(tr.t)))
            cls, conf, why = main_tr.last
            if sender:
                args = (cls, conf, LAST_FRAME, main_tr.x[-1] + off[0], main_tr.y[-1] + off[1], main_tr.extent_px())
                sender.maybe_send(*args)
                sender.send_live(*args)
            for tr in res:
                if wr:
                    wr.writerow([f"{t:.3f}", tr.id, *tr.last, f"{tr.x[-1]:.1f}", f"{tr.y[-1]:.1f}", f"{tr.size_px():.1f}"])
                if last_print.get(tr.id) != tr.last[0]:
                    last_print[tr.id] = tr.last[0]
                    print(f"[{t:7.2f}s] #{tr.id:<3} {tr.last[0]:8} {tr.last[1]:.2f}  {tr.last[2]}  ({tr.size_px():.0f}px)")
        if a.show:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for tr in res:
                cls, conf, why = tr.last
                x, y = int(tr.x[-1] + off[0]), int(tr.y[-1] + off[1])
                r = int(tr.size_px() * 2 + 6)
                cv2.rectangle(vis, (x - r, y - r), (x + r, y + r), COLORS[cls], 1)
                cv2.putText(vis, f"{cls} {conf:.2f}", (x - r, y - r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[cls], 1)
            cv2.imshow("motion_id", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    try:
        app = None
        if a.appearance:
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from appearance_id import AppearanceClassifier
            app = AppearanceClassifier(a.appearance)
        run(frames, fps, wind, a.ego, roi, on_frame, appearance=app, recon=a.recon)
    except KeyboardInterrupt:
        pass
    if wr:
        fcsv.close()


if __name__ == "__main__":
    main()
