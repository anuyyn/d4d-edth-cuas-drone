#!/usr/bin/env python3
import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from track3 import PRESETS, Confidence, detect_uv

MAGIC = 0x534D5241
DRONE_BIT = 0x10000
HDR = struct.Struct("<IId3f3f3f3fIIHH")
CMD = struct.Struct("<II4fIII")
MAGIC_CHASE = 0x45534843
CHASE = struct.Struct("<IIHHHHII")
HFOV = 1.0856
VFOV = 2 * math.atan(math.tan(HFOV / 2) * 0.75)
CAM_TILT = math.radians(20)
FULL_RATE = math.radians(400)
MAX_ANGLE = math.radians(35)
HOVER_THR = 0.5


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


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class SimLink:
    def __init__(self, port_in=9201, port_out=9202):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port_in))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.settimeout(30)
        self.dst = ("127.0.0.1", port_out)
        self.chase = None
        self._cbuf, self._cseq, self._cgot = None, -1, set()

    def _chase_chunk(self, data):
        magic, seq, idx, n, w, h, off, ln = CHASE.unpack_from(data)
        if seq != self._cseq:
            self._cbuf, self._cseq, self._cgot = bytearray(w * h * 3), seq, set()
        self._cbuf[off:off + ln] = data[CHASE.size:CHASE.size + ln]
        self._cgot.add(idx)
        if len(self._cgot) == n:
            self.chase = np.frombuffer(bytes(self._cbuf), np.uint8).reshape(h, w, 3)

    def recv(self):
        while True:
            data, _ = self.sock.recvfrom(65535)
            magic = struct.unpack_from("<I", data)[0]
            if magic == MAGIC_CHASE:
                self._chase_chunk(data)
                continue
            if magic != MAGIC:
                continue
            v = HDR.unpack_from(data)
            o = dict(seq=v[1], t=v[2], pos=np.array(v[3:6]), vel=np.array(v[6:9]),
                     att=np.array(v[9:12]), bal=np.array(v[12:15]), hit=bool(v[15] & 1), armed=bool(v[16]), flags=int(v[15]))
            w, h = v[17], v[18]
            o["rgb"] = np.frombuffer(data, np.uint8, offset=HDR.size).reshape(h, w, 3)
            o["chase"] = self.chase
            return o

    def send(self, seq, roll=0.0, pitch=0.0, yaw=0.0, thr=0.0, armed=False, reset=False, seed=0):
        self.sock.sendto(CMD.pack(MAGIC, seq, roll, pitch, yaw, thr, int(armed), int(reset), seed), self.dst)


class Mission:

    COLORS = {"DISARMED": "#666666", "ARMED": "#b58900", "LAUNCH": "#2f6fd6", "SEARCH": "#e07b00",
              "TRACK": "#d33a3a", "TERMINAL": "#a0006e", "HIT": "#1f9d55", "KILLED": "#111111",
              "FOLLOW": "#1565c0",
              "MANUAL": "#555599"}

    def __init__(self, a):
        self.a = a
        self.guidance = a.guidance
        self.auto = True
        self.mission = a.mission
        self.tgt_cls = a.target
        self.reset_state()

    def reset_state(self):
        self.state = "DISARMED"
        self.armed = False
        self.lock_t0 = None
        self.lost_t = 0.0
        self.good = 0
        self.search_dir = 1.0
        self.prev_los = None
        self.los_rate = np.zeros(2)
        self.last = dict(ax=0.0, ay=0.0)
        self.cmd = dict(roll=0.0, pitch=0.0, yaw=0.0, thr=0.0)
        self.t_launch = None
        self.t_hit = None
        self.t_follow = None
        self.target = "drone (1.8 m)" if self.tgt_cls == "drone" else "balloon (2 m)"

    def arm(self):
        if self.state in ("DISARMED", "KILLED"):
            self.state, self.armed = "ARMED", True
        elif self.state == "ARMED":
            self.state, self.armed = "DISARMED", False

    def launch(self, t):
        if self.state == "DISARMED":
            self.arm()
        if self.state == "ARMED":
            self.state, self.t_launch = "LAUNCH", t

    def kill(self):
        self.state, self.armed = "KILLED", False

    def step(self, o, det, conf, dt):
        a = self.a
        roll, pitch, yaw = o["att"]
        z, vz = o["pos"][2], o["vel"][2]
        t = o["t"]
        ok = det is not None and conf >= a.conf_on
        self.good = self.good + 1 if ok else 0

        ax = ay = None
        if det is not None:
            ax = det[0] * HFOV / 2
            ay = -det[1] * VFOV / 2
            los = np.array([wrap(-yaw + ax), ay + CAM_TILT - pitch])
            if self.prev_los is not None and dt > 0:
                d = np.array([wrap(los[0] - self.prev_los[0]), los[1] - self.prev_los[1]]) / dt
                self.los_rate = a.los_alpha * d + (1 - a.los_alpha) * self.los_rate
            self.prev_los = los
        else:
            self.prev_los = None

        vz_sp, r_c, p_c, y_c = 0.0, 0.0, 0.0, 0.0
        s = self.state
        if o["hit"] and s not in ("HIT", "KILLED", "DISARMED"):
            self.state = s = "HIT"
            self.t_hit = t
        if s in ("DISARMED", "KILLED"):
            self.cmd = dict(roll=0.0, pitch=0.0, yaw=0.0, thr=0.0)
            return self.cmd
        if s == "ARMED":
            self.cmd = dict(roll=0.0, pitch=0.0, yaw=0.0, thr=0.05)
            return self.cmd
        if s == "LAUNCH":
            vz_sp = 2.5
            if z > a.alt:
                self.state = "SEARCH"
        elif s == "SEARCH":
            vz_sp = 1.2 * (a.alt - z)
            y_c = self.search_dir * 0.12 if self.a.scan else 0.0
            if self.good >= 2:
                self.state, self.lock_t0, self.lost_t = "TRACK", t, 0.0
        elif s in ("TRACK", "TERMINAL", "FOLLOW"):
            if ok:
                self.lost_t = 0.0
                self.last = dict(ax=ax, ay=ay)
                if self.mission == "track":
                    if s == "TRACK" and det[3] > a.follow_px:
                        self.state = s = "FOLLOW"
                        self.t_follow = t
                elif det[3] > a.term_px:
                    self.state = s = "TERMINAL"
            else:
                self.lost_t += dt
                if self.lost_t > a.lost_s:
                    self.state = "SEARCH"
                    self.search_dir = 1.0 if self.last["ax"] >= 0 else -1.0
                    self.lock_t0 = None
            axu, el = self.last["ax"], self.last["ay"] + CAM_TILT - pitch
            if self.a.direct:
                vz_sp, y_c, p_c, r_c = 2.0, 0.0, 0.0, 0.0
            else:
                if self.guidance == "PN":
                    yaw_rate = a.N * self.los_rate[0] + a.k_bias * axu
                    vz_sp = a.N * a.vc * self.los_rate[1] + a.k_el * el * a.vc
                else:
                    yaw_rate = a.k_az * axu
                    vz_sp = a.k_el * 2.0 * el * a.vc
                y_c = float(np.clip(yaw_rate / FULL_RATE, -0.5, 0.5))
                fwd = a.p_term if s == "TERMINAL" else a.p_track
                fwd *= max(0.0, 1.0 - abs(axu) / math.radians(25))
                p_c = fwd * (1.0 if ok else 0.5)
                if s == "FOLLOW":
                    size = det[3] if ok else a.follow_px
                    p_c = float(np.clip(a.k_follow * (1.0 - size / a.follow_px), -0.25, a.p_track))
                r_c = float(np.clip(math.atan(a.vc * yaw_rate / 9.81) / MAX_ANGLE, -0.4, 0.4))
            if not self.auto:
                y_c = p_c = r_c = 0.0
                vz_sp = 1.2 * (a.alt - z)
        elif s == "HIT":
            vz_sp = -0.3 * vz

        vz_sp = float(np.clip(vz_sp, -3.0, 3.0))
        thr = (HOVER_THR + 0.12 * (vz_sp - vz)) / max(0.5, math.cos(roll) * math.cos(pitch))
        self.cmd = dict(roll=r_c, pitch=p_c, yaw=y_c, thr=float(np.clip(thr, 0.08, 0.95)))
        return self.cmd

    def lock_time(self, t):
        return 0.0 if self.lock_t0 is None else t - self.lock_t0

    def label(self):
        return "MANUAL" if (not self.auto and self.state in ("TRACK", "TERMINAL", "FOLLOW")) else self.state


def draw_view(rgb, det, conf, mission, o, px, S=4):
    import cv2
    img = cv2.resize(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), None, fx=S, fy=S,
                     interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    st = mission.label()
    col = tuple(int(Mission.COLORS.get(st, "#ffffff")[i:i + 2], 16) for i in (5, 3, 1))
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), col, 3)
    cv2.drawMarker(img, (W // 2, H // 2), (230, 230, 230), cv2.MARKER_CROSS, 24, 1)

    def put(t, x, y, sc, c, th=2):
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

    put(st, 10, 44, 1.4, col, 3)
    ex = det[0] if det is not None else 0.0
    ey = det[1] if det is not None else 0.0
    put(f"err  x={ex:+.2f} y={ey:+.2f}", 10, 76, 0.6, (80, 230, 230), 1)
    c = mission.cmd
    put(f"cmd  roll={c['roll']:+.2f} pitch={c['pitch']:+.2f} yaw={c['yaw']:+.2f} thr={c['thr']:.2f}",
        10, 100, 0.55, (120, 120, 255), 1)
    put(f"{mission.guidance}  alt {o['pos'][2]:.1f}m  lock {mission.lock_time(o['t']):.1f}s", 10, 124, 0.55,
        (230, 230, 230), 1)
    put(f"P {px}", 12, H - 16, 0.8, (255, 220, 60), 2)
    if o["hit"]:
        put("TARGET DESTROYED", W // 2 - 170, H // 2 + 60, 1.0, (60, 255, 60), 3)
    elif mission.state == "FOLLOW":
        put("SHADOWING - NO KILL", W // 2 - 180, H // 2 + 60, 1.0, (255, 200, 80), 3)
    if det is not None:
        x0, y0, x1, y1 = det[2]
        bx0, by0, bx1, by1 = x0 * 2 * S, y0 * 2 * S, (x1 + 1) * 2 * S, (y1 + 1) * 2 * S
        bc = (60, 220, 60) if conf >= mission.a.conf_on else (0, 170, 255)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), bc, 2)
        put(f"{conf:.2f}", bx1 + 4, by0 + 14, 0.6, bc, 1)
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        cv2.arrowedLine(img, (W // 2, H // 2), (cx, cy), (0, 200, 255), 2, tipLength=0.15)
        pad = max(bx1 - bx0, by1 - by0) // 2 + 12
        crop = img[max(0, cy - pad):min(H, cy + pad), max(0, cx - pad):min(W, cx + pad)]
        if crop.size:
            iw = 150
            ins = cv2.resize(crop, (iw, iw), interpolation=cv2.INTER_NEAREST)
            img[H - iw - 12:H - 12, W - iw - 12:W - 12] = ins
            cv2.rectangle(img, (W - iw - 12, H - iw - 12), (W - 12, H - 12), (60, 220, 60), 2)
    return img


class Composer:
    W, H = 1720, 960

    def __init__(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.fig = Figure(figsize=(5.2, 9.6), dpi=100)
        self.cv = FigureCanvasAgg(self.fig)
        self.a1, self.a2 = self.fig.add_subplot(211), self.fig.add_subplot(212)
        self.l1 = [self.a1.plot([], [], label=l)[0] for l in ("error_x", "error_y")]
        self.l2 = [self.a2.plot([], [], label=l)[0] for l in ("cmd roll", "cmd pitch", "cmd yaw")]
        self.a1.set_ylim(-1, 1)
        self.a2.set_ylim(-0.7, 0.7)
        for ax, t in ((self.a1, "/arms/mission_state/error"), (self.a2, "/arms/control_debug")):
            ax.axhline(0, color="#999", lw=0.6)
            ax.set_title(t, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("sim time [s]", fontsize=8)
        self.fig.tight_layout(pad=1.5)

    def plots(self, hist):
        if hist:
            h = np.array(hist)
            t = h[:, 0]
            for i, ln in enumerate(self.l1):
                ln.set_data(t, h[:, 1 + i])
            for i, ln in enumerate(self.l2):
                ln.set_data(t, h[:, 3 + i])
            for ax in (self.a1, self.a2):
                ax.set_xlim(max(0, t[-1] - 10), max(10, t[-1]))
        self.cv.draw()
        return np.asarray(self.cv.buffer_rgba())[:, :, 2::-1].copy()

    @staticmethod
    def panel(m, v, w=660, h=470):
        import cv2
        img = np.full((h, w, 3), (48, 34, 29), np.uint8)

        def put(t, x, y, sc, c=(230, 230, 230), th=1):
            cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

        def btn(t, x0, y0, x1, y1, col):
            cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
            (tw, th_), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            put(t, (x0 + x1 - tw) // 2, (y0 + y1 + th_) // 2, 0.6, (255, 255, 255), 2)

        put("A.R.M.S. Control Panel", 20, 30, 0.6, (200, 200, 200))
        put("MISSION STATE", w // 2 - 70, 70, 0.6, (200, 204, 216))
        st = m.label()
        col = tuple(int(Mission.COLORS.get(st, "#666666")[i:i + 2], 16) for i in (5, 3, 1))
        cv2.rectangle(img, (w // 2 - 150, 85), (w // 2 + 150, 150), col, -1)
        (tw, _), _ = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        put(st, w // 2 - tw // 2, 132, 1.3, (255, 255, 255), 3)
        ex = "--" if math.isnan(v["ex"]) else f"{v['ex']:+.2f}"
        ey = "--" if math.isnan(v["ey"]) else f"{v['ey']:+.2f}"
        put(f"error: {ex}, {ey}   lock: {v['lock']:.1f}s   P: {v['px']}", 120, 180, 0.55)
        put(f"conf {v['conf']:.2f}   alt {v['alt']:.1f} m   range(truth) {v['dist']:.1f} m", 120, 205, 0.55)
        btn("MISSION: " + ("TRACK & SHADOW (no kill)" if m.mission == "track" else "INTERCEPT"),
            40, 225, w - 40, 270, (192, 101, 21) if m.mission == "track" else (58, 58, 211))
        btn("GUIDANCE: " + ("PN (Proportional Navigation)" if m.guidance == "PN" else "PP (Pure Pursuit)"),
            40, 280, w - 40, 325, (201, 63, 106))
        bw = (w - 80 - 30) // 4
        xs = [40 + i * (bw + 10) for i in range(4)]
        btn("MODE: " + ("AUTO" if m.auto else "MANUAL"), xs[0], 345, xs[0] + bw, 390,
            (91, 158, 46) if m.auto else (119, 119, 119))
        btn("* ARMED" if m.armed else "ARM", xs[1], 345, xs[1] + bw, 390, (43, 57, 192))
        btn("KILL", xs[2], 345, xs[2] + bw, 390, (80, 62, 44))
        btn("RESET", xs[3], 345, xs[3] + bw, 390, (85, 85, 85))
        put(f"t = {v['t']:.1f} s    target: {m.target}    camera: 160x120 @ 9 Hz", 40, 430, 0.5,
            (170, 170, 170))
        return img

    def compose(self, cam, chase, m, v, hist):
        import cv2
        out = np.full((self.H, self.W, 3), 245, np.uint8)
        pl = self.plots(hist)
        hh, ww = min(pl.shape[0], self.H), min(pl.shape[1], 520)
        out[:hh, :ww] = pl[:hh, :ww]
        cx = 520
        out[10:10 + cam.shape[0], cx + 10:cx + 10 + cam.shape[1]] = cam
        pn = self.panel(m, v)
        out[self.H - pn.shape[0]:, cx:cx + pn.shape[1]] = pn
        rx = 1180
        cv2.rectangle(out, (rx, 0), (self.W - 1, 44), (60, 60, 60), -1)
        cv2.putText(out, "Gazebo Sim (chase view)", (rx + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (240, 240, 240), 1, cv2.LINE_AA)
        if chase is not None:
            cw = self.W - rx
            ch = min(int(chase.shape[0] * cw / chase.shape[1]), self.H - 44)
            im = cv2.resize(cv2.cvtColor(np.ascontiguousarray(chase), cv2.COLOR_RGB2BGR), (cw, ch))
            out[44:44 + ch, rx:rx + cw] = im
        return out


class Runner:

    def __init__(self, a):
        self.a = a
        self.link = SimLink()
        self.m = Mission(a)
        self.lock = threading.Lock()
        self.view = None
        self.hist = deque(maxlen=int(12 * 9))
        self.req = deque()
        self.stop = False
        self.seed = a.seed
        self.writer = None
        self.results = []
        self.logf = None
        self.launch_after_reset = False
        self.reply, self.log_id, self._last_state = None, "", None
        if a.cmd_port:
            threading.Thread(target=self.cmd_listener, daemon=True).start()
        if a.log:
            import csv
            self.logf = open(a.log, "w", newline="")
            self.logw = csv.writer(self.logf)
            self.logw.writerow(["seed", "t", "state", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
                                "dist", "true_az_deg", "true_el_deg", "det", "ex", "ey", "px", "conf",
                                "c_roll", "c_pitch", "c_yaw", "c_thr"])

    def cmd_listener(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", self.a.cmd_port))
        self.cmd_sock = s
        while not self.stop:
            data, addr = s.recvfrom(1024)
            parts = data.decode("utf-8", "ignore").strip().split(",")
            if parts[0] in ("intercept", "track"):
                self.reply = addr
                self.log_id = parts[2] if len(parts) > 2 else ""
                cls = parts[1].lower() if len(parts) > 1 else "balloon"
                self.m.mission = parts[0]
                self.m.tgt_cls = "drone" if cls == "drone" else "balloon"
                self.m.target = f"{cls.upper()} (log #{self.log_id})"
                self.req.append("engage")
            elif parts[0] in ("launch", "reset", "kill"):
                self.req.append(parts[0])
            elif parts[0] == "ping":
                s.sendto(b"pong", addr)

    def report_state(self, t):
        st = self.m.state
        if (self.log_id and self.m.t_launch is not None and self.m.t_follow is None
                and st not in ("HIT", "KILLED", "DISARMED")
                and t - self.m.t_launch > self.a.engage_s):
            self.m.kill()
            st = "MISS"
            self.m.state = "KILLED"
            if self.reply is not None:
                try:
                    self.cmd_sock.sendto(f"state,MISS,{self.log_id}".encode(), self.reply)
                except OSError:
                    pass
            self._last_state = "KILLED"
            return
        if st != self._last_state and self.reply is not None:
            try:
                self.cmd_sock.sendto(f"state,{st},{self.log_id}".encode(), self.reply)
            except OSError:
                pass
        self._last_state = st

    def handle_requests(self, o):
        reset = False
        while self.req:
            r = self.req.popleft()
            if r == "arm":
                self.m.arm()
            elif r == "launch":
                self.m.launch(o["t"])
            elif r == "engage":
                self.launch_after_reset = True
                self.next_seed = self.seed + 1 if self.a.vary_seed else self.a.seed
                reset = True
            elif r == "kill":
                self.m.kill()
            elif r == "guid":
                self.m.guidance = "PP" if self.m.guidance == "PN" else "PN"
            elif r == "mode":
                self.m.auto = not self.m.auto
            elif r == "reset":
                reset = True
        return reset

    def run(self, episodes=None):
        a = self.a
        conf_est = Confidence(full_px=a.full_px)
        p = PRESETS["red"]
        ep, t_prev, pending_reset = 0, None, True
        t_wall = time.time()
        while not self.stop:
            try:
                o = self.link.recv()
            except socket.timeout:
                print("Gazebo 응답 없음 (arms 월드가 떠 있는지 확인)", flush=True)
                return
            if pending_reset:
                self.link.send(o["seq"], reset=True,
                               seed=self.seed | (DRONE_BIT if self.m.tgt_cls == "drone" else 0))
                target = self.m.target
                self.m.reset_state()
                self.m.target = target
                conf_est = Confidence(full_px=a.full_px)
                pending_reset, t_prev = False, None
                if a.auto or self.launch_after_reset:
                    self.req.append("launch")
                    self.launch_after_reset = False
                continue
            self.next_seed = None
            if self.handle_requests(o):
                self.seed = self.next_seed if self.next_seed is not None else self.seed + 1
                pending_reset = True
                self.link.send(o["seq"])
                continue

            dt = 0.0 if t_prev is None else o["t"] - t_prev
            t_prev = o["t"]
            y, u, v = rgb_to_yuv420(o["rgb"])
            det, px = detect_uv(y, u, v, p, a.min_px)
            conf = conf_est.update(det)
            cmd = self.m.step(o, det, conf, dt)
            self.report_state(o["t"])
            self.link.send(o["seq"], cmd["roll"], cmd["pitch"], cmd["yaw"], cmd["thr"], self.m.armed)

            dist = float(np.linalg.norm(o["bal"] - o["pos"]))
            ex = det[0] if det is not None else float("nan")
            ey = det[1] if det is not None else float("nan")
            self.hist.append((o["t"], ex, ey, cmd["roll"], cmd["pitch"], cmd["yaw"], cmd["thr"]))
            if self.logf:
                d = o["bal"] - o["pos"]
                taz = math.degrees(wrap(-math.atan2(d[1], d[0]) + o["att"][2]))
                tel = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
                self.logw.writerow([self.seed, f"{o['t']:.2f}", self.m.state, *[f"{x:.2f}" for x in o["pos"]],
                                    *[f"{x:.2f}" for x in o["vel"]], *[f"{math.degrees(x):.1f}" for x in o["att"]],
                                    f"{dist:.2f}", f"{taz:.1f}", f"{tel:.1f}", int(det is not None),
                                    f"{ex:.2f}", f"{ey:.2f}", px, f"{conf:.2f}",
                                    *[f"{cmd[k]:.2f}" for k in ("roll", "pitch", "yaw", "thr")]])

            if not a.nogui or a.record:
                img = draw_view(o["rgb"], det, conf, self.m, o, px)
                view = dict(img=img, state=self.m.label(), guidance=self.m.guidance,
                            auto=self.m.auto, armed=self.m.armed, ex=ex, ey=ey, px=px, conf=conf,
                            lock=self.m.lock_time(o["t"]), dist=dist, alt=o["pos"][2], t=o["t"])
                if a.record:
                    import cv2
                    if self.writer is None:
                        self.comp = Composer()
                        self.writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 9,
                                                      (Composer.W, Composer.H))
                    self.writer.write(self.comp.compose(img, o["chase"], self.m, view, self.hist))
                with self.lock:
                    self.view = view

            followed = (self.m.mission == "track" and self.m.state == "FOLLOW"
                        and o["t"] - self.m.t_follow > a.follow_ok)
            done = (self.m.state == "HIT" or followed or o["t"] > a.timeout
                    or self.m.state == "KILLED" and a.nogui)
            if episodes is not None and done:
                ok_ = followed if self.m.mission == "track" else self.m.state == "HIT"
                t_ok = self.m.t_follow if self.m.mission == "track" else self.m.t_hit
                res = dict(ep=ep, seed=self.seed, hit=ok_, collide=self.m.state == "HIT",
                           t_hit=(t_ok - self.m.t_launch) if ok_ else None)
                self.results.append(res)
                word = "FOLLOW" if self.m.mission == "track" else "HIT"
                print(f"ep{ep} seed{self.seed}: {word if res['hit'] else 'FAIL'}"
                      + (f" {res['t_hit']:.1f}s" if res["hit"] else "")
                      + (" (충돌)" if self.m.mission == "track" and res["collide"] else ""), flush=True)
                ep += 1
                if ep >= episodes:
                    break
                self.seed += 1
                pending_reset = True
            elif a.record and a.auto and (o["t"] > a.timeout or
                                          (self.m.state == "HIT" and o["t"] - self.m.t_hit > 3) or
                                          (self.m.state == "FOLLOW" and o["t"] - self.m.t_follow > a.follow_rec)):
                break

            if not a.nogui and a.realtime:
                time.sleep(max(0.0, 1 / 9 - (time.time() - getattr(self, "_tw", 0))))
                self._tw = time.time()
        if self.writer is not None:
            self.writer.release()
        if self.logf:
            self.logf.close()
        if self.results:
            hits = [r for r in self.results if r["hit"]]
            ts = [r["t_hit"] for r in hits]
            print(f"== {self.a.guidance}: 명중 {len(hits)}/{len(self.results)} "
                  f"({100 * len(hits) / len(self.results):.0f}%)  명중시간 중앙값 "
                  f"{np.median(ts) if ts else float('nan'):.1f}s  (wall {time.time() - t_wall:.0f}s)", flush=True)


def gui(runner):
    import tkinter as tk
    from PIL import Image, ImageTk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("A.R.M.S. Control Panel")
    root.configure(bg="#1d2230")
    root.geometry("600x330+660+560")
    tk.Label(root, text="MISSION STATE", fg="#c8ccd8", bg="#1d2230", font=("Helvetica", 10)).pack(pady=(10, 2))
    st = tk.Label(root, text="DISARMED", fg="white", bg="#666666", font=("Helvetica", 24, "bold"), width=12)
    st.pack(pady=4)
    info = tk.Label(root, text="", fg="#e8e8e8", bg="#1d2230", font=("Courier", 10), justify="center")
    info.pack(pady=2)
    q = runner.req
    tk.Button(root, text="LAUNCH (→ TRACK)", bg="#d33a3a", fg="white", activebackground="#e05555",
              font=("Helvetica", 12, "bold"), command=lambda: q.append("launch")).pack(fill="x", padx=30, pady=3)
    gb = tk.Button(root, text="", bg="#6a3fc9", fg="white", activebackground="#8058e0",
                   font=("Helvetica", 11, "bold"), command=lambda: q.append("guid"))
    gb.pack(fill="x", padx=30, pady=3)
    row = tk.Frame(root, bg="#1d2230")
    row.pack(pady=6)
    mb = tk.Button(row, text="", bg="#2e9e5b", fg="white", width=10, command=lambda: q.append("mode"))
    mb.pack(side="left", padx=4)
    ab = tk.Button(row, text="ARM", bg="#c0392b", fg="white", width=10, command=lambda: q.append("arm"))
    ab.pack(side="left", padx=4)
    tk.Button(row, text="KILL", bg="#2c3e50", fg="white", width=10, command=lambda: q.append("kill")).pack(
        side="left", padx=4)
    tk.Button(row, text="RESET", bg="#555", fg="white", width=10, command=lambda: q.append("reset")).pack(
        side="left", padx=4)
    root.bind("<space>", lambda e: q.append("launch"))
    root.bind("k", lambda e: q.append("kill"))

    cam = tk.Toplevel(root)
    cam.title("A.R.M.S.")
    cam.geometry("+650+0")
    cv = tk.Label(cam, bg="black")
    cv.pack()

    pw = tk.Toplevel(root)
    pw.title("A.R.M.S. plots")
    pw.geometry("630x1000+0+0")
    if not runner.a.plots:
        pw.withdraw()
    fig = Figure(figsize=(6.3, 9.6), dpi=100)
    a1, a2 = fig.add_subplot(211), fig.add_subplot(212)
    fig.tight_layout(pad=2.0)
    canvas = FigureCanvasTkAgg(fig, master=pw)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    lines1 = [a1.plot([], [], label=l)[0] for l in ("error_x", "error_y")]
    lines2 = [a2.plot([], [], label=l)[0] for l in ("cmd roll", "cmd pitch", "cmd yaw")]
    a1.set_ylim(-1, 1); a2.set_ylim(-0.7, 0.7)
    a1.axhline(0, color="#999", lw=0.6); a2.axhline(0, color="#999", lw=0.6)
    a1.set_title("/arms/mission_state/error", fontsize=9); a2.set_title("/arms/control_debug", fontsize=9)
    a1.legend(loc="upper right", fontsize=8); a2.legend(loc="upper right", fontsize=8)
    photo = {"p": None}
    tick = {"n": 0}

    def refresh():
        with runner.lock:
            v = runner.view
        if v is not None:
            im = Image.fromarray(v["img"][:, :, ::-1])
            photo["p"] = ImageTk.PhotoImage(im)
            cv.configure(image=photo["p"])
            st.configure(text=v["state"], bg=Mission.COLORS.get(v["state"], "#666"))
            ex = "  --" if math.isnan(v["ex"]) else f"{v['ex']:+.2f}"
            ey = "  --" if math.isnan(v["ey"]) else f"{v['ey']:+.2f}"
            info.configure(text=f"error: {ex}, {ey}   lock: {v['lock']:.1f}s   P: {v['px']}\n"
                                f"conf {v['conf']:.2f}   alt {v['alt']:.1f}m   거리(참값) {v['dist']:.1f}m")
            gb.configure(text="유도: PN(비례항법)" if v["guidance"] == "PN" else "유도: PP(추적법)")
            mb.configure(text="모드: 자동" if v["auto"] else "모드: 수동", bg="#2e9e5b" if v["auto"] else "#777")
            ab.configure(text="● ARMED" if v["armed"] else "ARM")
        tick["n"] += 1
        if runner.a.plots and tick["n"] % 2 == 0 and runner.hist:
            h = np.array(runner.hist)
            t = h[:, 0]
            for i, ln in enumerate(lines1):
                ln.set_data(t, h[:, 1 + i])
            for i, ln in enumerate(lines2):
                ln.set_data(t, h[:, 3 + i])
            for ax_ in (a1, a2):
                ax_.set_xlim(max(0, t[-1] - 10), max(10, t[-1]))
            canvas.draw_idle()
        root.after(100, refresh)

    def on_close():
        runner.stop = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guidance", default="PN", choices=["PN", "PP"])
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--episodes", type=int, default=0, help="평가 에피소드 수 (0=무한, GUI 조작)")
    ap.add_argument("--auto", action="store_true", help="에피소드 시작 시 자동 LAUNCH")
    ap.add_argument("--record", default="")
    ap.add_argument("--log", default="", help="프레임별 CSV")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--direct", action="store_true", help="9/22 영상 방식(정면 고정 상승, 유도 없음)")
    ap.add_argument("--vary-seed", action="store_true", help="요격 명령마다 다른 상황(기본: --seed 고정 재현)")
    ap.add_argument("--scan", type=int, default=1, help="SEARCH 에서 yaw 회전 탐색")
    ap.add_argument("--engage-s", type=float, default=30.0, help="요격 명령 후 이 시간(sim s) 안에 못 맞히면 MISS")
    ap.add_argument("--plots", type=int, default=1, help="그래프 창 (0=숨김)")
    ap.add_argument("--cmd-port", type=int, default=9510, help="관제 로그 요격 명령 UDP (0=끔)")
    ap.add_argument("--mission", default="intercept", choices=["intercept", "track"],
                    help="시작 임무 (관제 명령이 오면 그 명령으로 바뀜)")
    ap.add_argument("--target", default="balloon", choices=["balloon", "drone"], help="시작 표적 모양")
    ap.add_argument("--follow-px", type=int, default=200, help="추적 임무: 표적이 이 픽셀 이상이면 FOLLOW(거리 유지)")
    ap.add_argument("--k-follow", type=float, default=0.6, help="추적 거리 유지 게인")
    ap.add_argument("--follow-ok", type=float, default=8.0, help="평가: FOLLOW 를 이만큼 유지하면 성공 [s]")
    ap.add_argument("--follow-rec", type=float, default=10.0, help="녹화: FOLLOW 후 이만큼 찍고 종료 [s]")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--realtime", type=int, default=1)
    ap.add_argument("--alt", type=float, default=5.0, help="탐색 고도 [m]")
    ap.add_argument("--conf-on", type=float, default=0.35)
    ap.add_argument("--full-px", type=float, default=15)
    ap.add_argument("--min-px", type=int, default=6)
    ap.add_argument("--term-px", type=int, default=300, help="이 픽셀 이상이면 TERMINAL(돌입)")
    ap.add_argument("--lost-s", type=float, default=1.0)
    ap.add_argument("--N", type=float, default=3.0, help="PN 항법상수")
    ap.add_argument("--vc", type=float, default=6.0, help="접근속도 가정 [m/s] (온보드엔 거리·속도 없음)")
    ap.add_argument("--k-bias", type=float, default=1.0, help="PN 에 더하는 조준 유지항")
    ap.add_argument("--los-alpha", type=float, default=0.3, help="시선각속도 필터 (작을수록 부드러움)")
    ap.add_argument("--cmd-alpha", type=float, default=0.5, help="명령 평활 계수")
    ap.add_argument("--k-az", type=float, default=2.0, help="PP yaw 게인 [1/s]")
    ap.add_argument("--k-el", type=float, default=0.5)
    ap.add_argument("--p-track", type=float, default=0.30)
    ap.add_argument("--p-term", type=float, default=0.45)
    a = ap.parse_args()
    if a.nogui:
        a.realtime = 0

    r = Runner(a)
    eps = a.episodes if a.episodes > 0 else None
    if a.nogui:
        r.run(eps)
    else:
        th = threading.Thread(target=r.run, args=(eps,), daemon=True)
        th.start()
        gui(r)


if __name__ == "__main__":
    main()
