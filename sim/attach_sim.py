#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import HOVER_THR, MAX_ANGLE, SimLink, wrap

G = 9.81
ATTACH_BIT = 0x20000
TIP_X, TIP_Z = 0.62, -0.06
ROPE_TOP, ROPE_BOT = -1.0, -2.6
ROPE_MID = 0.5 * (ROPE_TOP + ROPE_BOT)
DRAG = 0.35


def run_episode(link, seed, a, rec=None):
    o = link.recv()
    link.send(o["seq"], reset=True, seed=seed | ATTACH_BIT)
    o = link.recv()
    home = o["pos"].copy()
    bal_prev, vb = None, np.zeros(3)
    phase, t_phase = "LAUNCH", o["t"]
    u = None
    eng_s = 0.0
    tries = 1
    t_attach = None
    ok_since = None
    rows = []
    while True:
        t = o["t"]
        p, v = o["pos"], o["vel"]
        roll, pitch, yaw = o["att"]
        bal = o["bal"]
        if bal_prev is not None and t > tprev:
            vb = 0.7 * vb + 0.3 * (bal - bal_prev) / (t - tprev)
        bal_prev, tprev = bal.copy(), t
        rope = bal + np.array([0, 0, ROPE_MID])
        if u is None:
            d = bal - home
            u = np.array([d[0], d[1], 0.0]) / max(1e-6, math.hypot(d[0], d[1]))
        attached = bool(o.get("flags", 0) & 2)
        hit = bool(o.get("flags", 0) & 1)

        q_tip = np.array([TIP_X * math.cos(yaw) * math.cos(pitch), TIP_X * math.sin(yaw) * math.cos(pitch),
                          -TIP_X * math.sin(pitch) + TIP_Z])
        tip = p + q_tip
        dh = math.hypot(tip[0] - bal[0], tip[1] - bal[1])
        along = float(np.dot((tip - rope)[:2], u[:2]))

        zt = rope[2] - TIP_Z
        vt = vb.copy()
        target = None
        if phase == "LAUNCH":
            target = np.array([p[0], p[1], zt])
            vt = np.zeros(3)
            if p[2] > zt - 0.4:
                phase, t_phase = "APPROACH", t
        elif phase == "APPROACH":
            target = rope - u * a.stand + np.array([0, 0, -TIP_Z])
            err = np.linalg.norm(target - p)
            if err < 0.25 and np.linalg.norm(v - vb) < 0.3:
                ok_since = t if ok_since is None else ok_since
                if t - ok_since > 0.8 and t - t_phase > 1.0:
                    phase, t_phase, eng_s = "ENGAGE", t, 0.0
            else:
                ok_since = None
        elif phase == "ENGAGE":
            eng_s = min(a.stand + 0.25 - TIP_X, eng_s + a.creep / 9.0)
            target = rope - u * (a.stand - eng_s) + np.array([0, 0, -TIP_Z])
            vt = vb + u * a.creep
            if attached:
                phase, t_phase, t_attach = "ATTACHED", t, t
            elif along > 0.35:
                phase, t_phase = "APPROACH", t
                tries += 1
        elif phase == "ATTACHED":
            target = p.copy()
            target[2] = zt
            if t - t_phase > 0.8:
                phase, t_phase = "RETREAT", t
        elif phase == "RETREAT":
            target = rope - u * 4.0 + np.array([0, 0, 0.5])
            if t - t_phase > 4.0:
                phase, t_phase = "RETURN", t
        elif phase == "RETURN":
            target = home + np.array([0, 0, 4.0]) + u * 4.0
            vt = np.zeros(3)

        e = target - p
        vmax = 2.5 if phase in ("APPROACH", "RETURN", "RETREAT") else 1.2
        vd = vt + np.clip(a.kp * e, -vmax, vmax)
        acc = a.kv * (vd - v) + DRAG * vd
        acc[:2] = np.clip(acc[:2], -4.0, 4.0)
        fwd = acc[0] * math.cos(yaw) + acc[1] * math.sin(yaw)
        right = acc[0] * math.sin(yaw) - acc[1] * math.cos(yaw)
        p_c = float(np.clip(math.atan2(fwd, G) / MAX_ANGLE, -0.6, 0.6))
        r_c = float(np.clip(math.atan2(right, G) / MAX_ANGLE, -0.6, 0.6))
        az = float(np.clip(2.0 * e[2] + 2.5 * (vd[2] - v[2]), -4, 4))
        thr = HOVER_THR * (G + az) / G / max(0.5, math.cos(roll) * math.cos(pitch))
        yaw_d = math.atan2(u[1], u[0])
        y_c = float(np.clip(-1.2 * wrap(yaw_d - yaw) / math.radians(400), -0.4, 0.4))
        link.send(o["seq"], r_c, p_c, y_c, float(np.clip(thr, 0.08, 0.95)), True)

        rows.append(dict(seed=seed, t=round(t, 3), phase=phase, tries=tries, dh=round(dh, 3),
                         along=round(along, 3), dz=round(tip[2] - rope[2], 3), attached=int(attached),
                         hit=int(hit), x=round(p[0], 2), y=round(p[1], 2), z=round(p[2], 2),
                         dist=round(float(np.linalg.norm(rope - p)), 2)))
        if rec is not None:
            rec(o, rows[-1])
        done = (phase == "RETURN" and t - t_phase > a.tail) or t > a.timeout or hit
        if done:
            break
        o = link.recv()
    ok = t_attach is not None and not hit
    return ok, (t_attach - rows[0]["t"]) if ok else None, tries, hit, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-5", help="예: 1-10 또는 3")
    ap.add_argument("--stand", type=float, default=3.0, help="대기점: 줄 앞 거리 [m]")
    ap.add_argument("--creep", type=float, default=0.6, help="전진 속도(풍선 기준) [m/s]")
    ap.add_argument("--kp", type=float, default=0.9)
    ap.add_argument("--kv", type=float, default=2.2)
    ap.add_argument("--timeout", type=float, default=70.0)
    ap.add_argument("--tail", type=float, default=4.0, help="RETURN 후 녹화 유지 [s]")
    ap.add_argument("--record", default="", help="이름: videos/<이름>_cam.mp4, _chase.mp4, logs/<이름>.csv")
    a = ap.parse_args()
    if "-" in a.seeds:
        s0, s1 = map(int, a.seeds.split("-"))
        seeds = list(range(s0, s1 + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]

    link = SimLink()
    rec = None
    writers = {}
    if a.record:
        import cv2

        def rec(o, row):
            cam = cv2.resize(cv2.cvtColor(np.ascontiguousarray(o["rgb"]), cv2.COLOR_RGB2BGR), (640, 480),
                             interpolation=cv2.INTER_LINEAR)
            if "cam" not in writers:
                writers["cam"] = cv2.VideoWriter(f"videos/{a.record}_cam.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                                 9, (640, 480))
            writers["cam"].write(cam)
            ch = o["chase"]
            ch = (np.zeros((540, 960, 3), np.uint8) if ch is None or ch.shape[:2] != (540, 960)
                  else cv2.cvtColor(np.ascontiguousarray(ch), cv2.COLOR_RGB2BGR))
            if "chase" not in writers:
                writers["chase"] = cv2.VideoWriter(f"videos/{a.record}_chase.mp4",
                                                   cv2.VideoWriter_fourcc(*"mp4v"), 9, (960, 540))
            writers["chase"].write(ch)

    allrows, res = [], []
    for sd in seeds:
        ok, t_ok, tries, hit, rows = run_episode(link, sd, a, rec)
        allrows += rows
        res.append(ok)
        print(f"seed{sd}: {'ATTACHED' if ok else 'FAIL'}" + (f" {t_ok:.1f}s, 시도 {tries}회" if ok else "")
              + (" (풍선 충돌)" if hit else ""), flush=True)
    print(f"== 부착 {sum(res)}/{len(res)}", flush=True)
    for k, w in writers.items():
        if hasattr(w, "release"):
            w.release()
    if a.record:
        with open(f"logs/{a.record}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(allrows[0]))
            wr.writeheader()
            wr.writerows(allrows)


if __name__ == "__main__":
    main()
