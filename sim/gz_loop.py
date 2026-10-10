#!/usr/bin/env python3
import argparse
import csv
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track3 import PRESETS, Confidence, YawController, detect_uv

MAGIC = 0x42574159
HDR = struct.Struct("<IIdfffHH")
CMD = struct.Struct("<IIfII")


def rgb_to_yuv420(rgb):
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) * 0.564 + 128
    v = (r - y) * 0.713 + 128
    h, w = y.shape
    u = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    c = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    return c(y), c(u), c(v)


class PolicyCtl:

    def __init__(self, path, conf_on, max_cmd):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.conf_on, self.max_cmd = conf_on, max_cmd
        self.cmd = 0.0

    def step(self, dx, conf):
        if dx is None or conf < self.conf_on:
            self.cmd = 0.0
            return 0.0, False
        x = np.array(policy_obs(dx, conf, self.cmd), np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = np.tanh(x @ W + b)
        self.cmd = float(np.clip(x[0], -1, 1) * self.max_cmd)
        return self.cmd, True

    def reset(self):
        self.cmd = 0.0


def show(rgb, det, conf, cmd, active, err, ctl, ep, k, window=True, writer=None):
    import cv2
    S = 4
    img = cv2.resize(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), None, fx=S, fy=S,
                     interpolation=cv2.INTER_NEAREST)
    H, W = img.shape[:2]
    cv2.line(img, (W // 2, 0), (W // 2, H), (200, 200, 200), 1)
    if det is not None:
        x0, y0, x1, y1 = det[2]
        c = (0, 220, 0) if active else (0, 160, 255)
        cv2.rectangle(img, (x0 * 2 * S, y0 * 2 * S), ((x1 + 1) * 2 * S, (y1 + 1) * 2 * S), c, 2)
    cx = W // 2
    cv2.rectangle(img, (cx, H - 22), (int(cx + cmd * W / 2), H - 8), (0, 220, 0) if active else (120, 120, 120), -1)
    txt = [f"{ctl.upper()}  ep{ep} #{k}", f"{'AUTO' if active else 'PILOT'}  conf {conf:.2f}",
           f"cmd {cmd:+.2f}   true err {err:+.1f}deg"]
    for i, t in enumerate(txt):
        cv2.putText(img, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, t, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if writer is not None:
        writer.write(img)
    if window:
        cv2.imshow("drone camera (track3 detect)", img)
        cv2.waitKey(1)


def policy_obs(dx, conf, prev_cmd):
    return [dx, conf, prev_cmd]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctl", default="p", choices=["p", "rl", "none"])
    ap.add_argument("--policy", default="policy.npz")
    ap.add_argument("--target", default="red", choices=sorted(PRESETS))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=540, help="에피소드당 프레임 (9Hz x 60s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delay", type=int, default=1, help="명령 적용 지연 [프레임]")
    ap.add_argument("--kp", type=float, default=0.2)
    ap.add_argument("--deadband", type=float, default=0.08)
    ap.add_argument("--max-cmd", type=float, default=0.4)
    ap.add_argument("--slew", type=float, default=0.1)
    ap.add_argument("--conf-on", type=float, default=0.35)
    ap.add_argument("--min-px", type=int, default=6)
    ap.add_argument("--full-px", type=float, default=15)
    ap.add_argument("--pilot-delay", type=int, default=4, help="조종사 반응 지연 [프레임] (~0.45s)")
    ap.add_argument("--pilot-gain", type=float, default=0.15, help="조종사 yaw 게인 (오차 31도당 명령)")
    ap.add_argument("--pilot-max", type=float, default=0.3)
    ap.add_argument("--realtime", action="store_true", help="9Hz 실시간 속도로 (눈으로 볼 때)")
    ap.add_argument("--show", action="store_true", help="드론 카메라 창 (검출 상자·신뢰도·명령)")
    ap.add_argument("--record", default="", help="드론 카메라 화면을 mp4 로 저장 (9fps = 실시간 재생)")
    ap.add_argument("--calib", action="store_true", help="검출 대신 프레임 전체 중 가장 붉은 영역의 U/V 출력")
    ap.add_argument("--save-frame", default="", help="첫 프레임을 .npy 로 저장")
    ap.add_argument("--log", default="")
    ap.add_argument("--port-in", type=int, default=9101)
    ap.add_argument("--port-out", type=int, default=9102)
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", a.port_in))
    sock.settimeout(30)
    dst = ("127.0.0.1", a.port_out)

    p = PRESETS[a.target]
    log = a.log or f"gz_{a.ctl}_d{a.delay}_{time.strftime('%m%d_%H%M%S')}.csv"
    f = open(log, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["ep", "k", "sim_t", "err_true_deg", "dist", "dx", "n", "conf", "cmd", "active"])

    summary = []
    writer = None
    print(f"ctl={a.ctl} delay={a.delay}f episodes={a.episodes} frames={a.frames} log={log}", flush=True)
    t_wall = time.time()
    for ep in range(a.episodes):
        conf_est = Confidence(full_px=a.full_px)
        if a.ctl == "rl":
            ctl = PolicyCtl(a.policy, a.conf_on, a.max_cmd)
        else:
            ctl = YawController(a.kp, a.deadband, a.max_cmd, a.slew, a.conf_on)
        queue = [0.0] * a.delay
        seen = [0.0] * a.pilot_delay
        rng = np.random.default_rng(a.seed + 100 + ep)
        pilot_frames = 0
        errs, act_frames, vis_frames = [], 0, 0
        k = 0
        reset_pending = True
        t_frame = time.time()
        while k < a.frames:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                print("Gazebo 응답 없음 (gazebo 가 떠 있는지, 플러그인 로드됐는지 확인)", flush=True)
                return
            magic, seq, sim_t, err, dist, yrate, w, h = HDR.unpack_from(data)
            if magic != MAGIC:
                continue
            rgb = np.frombuffer(data, np.uint8, offset=HDR.size).reshape(h, w, 3)
            if a.save_frame:
                np.save(a.save_frame, rgb)
                a.save_frame = ""
            y, u, v = rgb_to_yuv420(rgb)

            if reset_pending:
                sock.sendto(CMD.pack(MAGIC, seq, 0.0, 1, a.seed + ep), dst)
                reset_pending = False
                continue

            if a.calib:
                score = v.astype(int) - u.astype(int)
                i = np.unravel_index(np.argmax(score), score.shape)
                print(f"most-red U={u[i]} V={v[i]} Y={y[2 * i[0], 2 * i[1]]}  err_true={np.degrees(err):+.1f}deg",
                      flush=True)
                sock.sendto(CMD.pack(MAGIC, seq, 0.0, 0, 0), dst)
                k += 1
                continue

            det, n = detect_uv(y, u, v, p, a.min_px)
            conf = conf_est.update(det)
            dx = det[0] if det is not None else None
            seen.append(float(np.degrees(err)))
            e_seen = seen.pop(0)
            pilot = 0.0
            if abs(e_seen) > 10:
                pilot = float(np.clip(a.pilot_gain * e_seen / 31.1 + rng.normal(0, 0.03),
                                      -a.pilot_max, a.pilot_max))
            if a.ctl == "none":
                cmd, active = pilot, False
            else:
                cmd, active = ctl.step(dx, conf)
                if not active:
                    cmd = pilot
            pilot_frames += int(not active)
            queue.append(cmd)
            applied = queue.pop(0)
            sock.sendto(CMD.pack(MAGIC, seq, applied, 0, 0), dst)

            e = float(np.degrees(err))
            if a.show or a.record:
                if a.record and writer is None:
                    import cv2
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 9, (w * 4, h * 4))
                show(rgb, det, conf, cmd, active, e, a.ctl, ep, k, window=a.show, writer=writer)
            if a.realtime:
                dt = 1 / 9 - (time.time() - t_frame)
                if dt > 0:
                    time.sleep(dt)
                t_frame = time.time()
            errs.append(e)
            act_frames += int(active)
            vis_frames += int(abs(e) < 31.1)
            wr.writerow([ep, k, f"{sim_t:.3f}", f"{e:.2f}", f"{dist:.2f}",
                         "" if dx is None else f"{dx:.3f}", n, f"{conf:.2f}", f"{cmd:.3f}", int(active)])
            k += 1

        if not a.calib:
            e = np.abs(np.array(errs))
            s = dict(ep=ep, rms=float(np.sqrt((e ** 2).mean())), med=float(np.median(e)),
                     in_fov=vis_frames / len(e), in_5deg=float((e < 5).mean()), auto=act_frames / len(e))
            summary.append(s)
            print(f"ep{ep}: RMS {s['rms']:.1f}deg  median {s['med']:.1f}deg  "
                  f"FOV안 {s['in_fov'] * 100:.0f}%  5도안 {s['in_5deg'] * 100:.0f}%  AUTO {s['auto'] * 100:.0f}%",
                  flush=True)
    f.close()
    if writer is not None:
        writer.release()
    if summary:
        m = {k: np.mean([s[k] for s in summary]) for k in summary[0] if k != "ep"}
        print(f"== {a.ctl} delay={a.delay}: RMS {m['rms']:.1f}deg  median {m['med']:.1f}deg  "
              f"FOV안 {m['in_fov'] * 100:.0f}%  5도안 {m['in_5deg'] * 100:.0f}%  AUTO {m['auto'] * 100:.0f}%  "
              f"(wall {time.time() - t_wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
