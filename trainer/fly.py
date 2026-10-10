import argparse
import csv
import ctypes
import os
import socket
import sys
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "sync_work"))
from track_render import R_IN, R_OUT, R_MID, HALF_STRAIGHT, lateral_error, track_heading_ccw

SPEED, HEIGHT = 6.0, 8.0
YAW_GAIN = np.radians(86.0)
TAU, DT = 0.10, 1 / 30
HALF_W = (R_OUT - R_IN) / 2
HFOV, DOWN = 110.0, 40.0
RW, RH = 480, 360
TEX = os.path.join(ROOT, "sim", "models", "running_track", "materials", "textures", "track.png")
X0, X1, Y0, Y1 = -110.0, 110.0, -60.0, 60.0
OUTER_BGR = np.array([70, 120, 92], np.uint8)
SKY_TOP, SKY_HZN = np.array([215, 160, 95], np.float32), np.array([235, 215, 190], np.float32)
FAR = 300.0


class FPVRenderer:

    def __init__(self):
        img = cv2.imread(TEX)
        if img is None:
            raise SystemExit(f"트랙 텍스처 없음: {TEX} (sim/make_track_model.py 로 생성)")
        for bgr in ((220, 60, 40), (40, 210, 230)):
            img[np.all(np.abs(img.astype(int) - bgr) < 8, axis=2)] = OUTER_BGR
        self.tex = img
        self.th, self.tw = img.shape[:2]
        f = (RW / 2) / np.tan(np.radians(HFOV) / 2)
        u = (np.arange(RW) + 0.5 - RW / 2) / f
        v = (np.arange(RH) + 0.5 - RH / 2) / f
        uu, vv = np.meshgrid(u, v)
        c, s = np.cos(np.radians(DOWN)), np.sin(np.radians(DOWN))
        rf, rl, ru = c - s * vv, -uu, -s - c * vv
        hit = ru < -1e-6
        t = np.where(hit, HEIGHT / np.maximum(-ru, 1e-6), np.inf)
        self.ground = hit & (t < FAR)
        self.gxb = (rf * t)[self.ground].astype(np.float32)
        self.gyb = (rl * t)[self.ground].astype(np.float32)
        fog = np.clip((t[self.ground] - 60.0) / (FAR - 60.0), 0, 1) ** 1.5
        self.fog_i = np.nonzero(fog > 0.02)[0]
        self.fog = fog[self.fog_i, None].astype(np.float32)
        self.gidx = np.flatnonzero(self.ground)
        self.tex_flat = np.vstack([img.reshape(-1, 3), OUTER_BGR[None]])
        el = np.clip((ru + 0.05) / 0.6, 0, 1)[..., None]
        self.sky = (SKY_HZN * (1 - el) + SKY_TOP * el).astype(np.uint8)
        self.haze = SKY_HZN

    def render(self, x, y, yaw):
        cy, sy = np.float32(np.cos(yaw)), np.float32(np.sin(yaw))
        gx = np.float32(x) + cy * self.gxb - sy * self.gyb
        gy = np.float32(y) + sy * self.gxb + cy * self.gyb
        r = ((X1 - gx) * np.float32(self.th / (X1 - X0))).astype(np.int32)
        c = ((Y1 - gy) * np.float32(self.tw / (Y1 - Y0))).astype(np.int32)
        idx = r * self.tw + c
        idx[(r < 0) | (r >= self.th) | (c < 0) | (c >= self.tw)] = self.th * self.tw
        col = self.tex_flat.take(idx, axis=0)
        f = col[self.fog_i].astype(np.float32)
        col[self.fog_i] = (f + (self.haze - f) * self.fog).astype(np.uint8)
        out = self.sky.copy().reshape(-1, 3)
        out[self.gidx] = col
        return out.reshape(RH, RW, 3)


user32 = ctypes.windll.user32 if os.name == "nt" else None
VK = dict(left=0x25, right=0x27)


def key_down(vk):
    return bool(user32 and (user32.GetAsyncKeyState(vk) & 0x8000))


class KeyboardInput:
    name = "keyboard"

    def __init__(self, a):
        self.v = 0.0

    def read(self, state):
        d = key_down(VK["right"]) - key_down(VK["left"])
        if d:
            self.v = float(np.clip(self.v + d * 4.0 * DT, -1, 1))
        else:
            self.v -= np.sign(self.v) * min(abs(self.v), 8.0 * DT)
        return self.v

    def set_neutral(self):
        pass


class JOYINFOEX(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in
                ("dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos", "dwUpos", "dwVpos",
                 "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1", "dwReserved2")]


JOY_AXES = ["x", "y", "z", "r", "u", "v"]


def joy_read_raw(jid=0):
    ji = JOYINFOEX()
    ji.dwSize = ctypes.sizeof(JOYINFOEX)
    ji.dwFlags = 0xFF
    if ctypes.windll.winmm.joyGetPosEx(jid, ctypes.byref(ji)) != 0:
        return None
    return {k: getattr(ji, "dw" + k.upper() + "pos") for k in JOY_AXES}


class JoyInput:
    name = "joy"

    def __init__(self, a):
        self.axis, self.inv, self.dead, self.jid = a.axis, -1.0 if a.invert else 1.0, a.deadband_stick, a.joy_id
        if joy_read_raw(self.jid) is None:
            raise SystemExit("조이스틱 없음: TX12 를 USB 로 꽂고 'USB Joystick (HID)' 선택 후 다시 실행")
        self.center = 32767.5

    def read(self, state):
        raw = joy_read_raw(self.jid)
        if raw is None:
            return 0.0
        v = (raw[self.axis] - self.center) / 32767.5 * self.inv
        v = 0.0 if abs(v) < self.dead else np.sign(v) * (abs(v) - self.dead) / (1 - self.dead)
        return float(np.clip(v, -1, 1))

    def set_neutral(self):
        raw = joy_read_raw(self.jid)
        if raw is not None:
            self.center = float(raw[self.axis])


class IMUInput:
    name = "imu"

    def __init__(self, a):
        self.idx = {"roll": 0, "pitch": 1, "yaw": 2}[a.imu_axis]
        self.inv = -1.0 if a.invert else 1.0
        self.dead, self.full = a.deadband_deg, a.full_deg
        self.last, self.last_t, self.neutral = None, 0.0, 0.0
        self.recent = []
        self.ser = self.sock = None
        if a.serial:
            import serial
            self.ser = serial.Serial(a.serial, 115200, timeout=0)
            self.buf = b""
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", a.udp_port))
            self.sock.setblocking(False)

    def _lines(self):
        if self.sock:
            while True:
                try:
                    pkt, _ = self.sock.recvfrom(256)
                except (BlockingIOError, OSError):
                    return
                yield pkt
        else:
            self.buf += self.ser.read(4096)
            *lines, self.buf = self.buf.split(b"\n")
            yield from lines

    def poll(self):
        for ln in self._lines():
            try:
                vals = [float(t) for t in ln.decode(errors="ignore").strip().split(",")[-3:]]
            except ValueError:
                continue
            if len(vals) == 3:
                self.last, self.last_t = vals, time.perf_counter()
                self.recent.append(vals[self.idx])
                self.recent = self.recent[-30:]

    def angle(self):
        self.poll()
        return None if self.last is None else self.last[self.idx]

    def alive(self):
        return self.last is not None and time.perf_counter() - self.last_t < 0.5

    def read(self, state):
        ang = self.angle()
        if ang is None or not self.alive():
            return 0.0
        d = (ang - self.neutral) * self.inv
        if abs(d) < self.dead:
            return 0.0
        return float(np.clip(np.sign(d) * (abs(d) - self.dead) / (self.full - self.dead), -1, 1))

    def set_neutral(self):
        self.poll()
        if self.recent:
            self.neutral = float(np.mean(self.recent))


class GeomInput:
    name = "geom"

    def __init__(self, a):
        pass

    def read(self, s):
        x, y, yaw = s
        e = lateral_error(x, y)
        he = np.angle(np.exp(1j * (track_heading_ccw(x, y) - yaw)))
        ff = SPEED / R_MID if abs(x) > HALF_STRAIGHT else 0.0
        return float(np.clip(-(ff + 1.5 * he + 0.15 * e) / YAW_GAIN, -1, 1))

    def set_neutral(self):
        pass


INPUTS = dict(keyboard=KeyboardInput, joy=JoyInput, imu=IMUInput, geom=GeomInput)


def hud(view, t, T, stick, e, offs, trial, n_trials, pid, inp, warn=None, banner=None):
    v = cv2.resize(view, (RW * 2, RH * 2), interpolation=cv2.INTER_LINEAR)
    W, H = v.shape[1], v.shape[0]
    cv2.rectangle(v, (0, 0), (W, 44), (0, 0, 0), -1)
    cv2.putText(v, f"{pid}  input={inp}  trial {trial}/{n_trials}   t {t:5.1f}/{T:.0f}s   off-track {offs}",
                (12, 30), 0, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    bx, by, bw = W // 2 - 200, H - 40, 400
    cv2.rectangle(v, (bx, by), (bx + bw, by + 18), (40, 40, 40), -1)
    cv2.line(v, (bx + bw // 2, by - 4), (bx + bw // 2, by + 22), (200, 200, 200), 1)
    sx = int(bx + bw / 2 + stick * bw / 2)
    cv2.rectangle(v, (min(sx, bx + bw // 2), by), (max(sx, bx + bw // 2), by + 18), (0, 200, 255), -1)
    cv2.putText(v, f"direction {stick:+.2f}", (bx + bw + 12, by + 16), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    gx, gy, gw = 20, H - 40, 220
    cv2.rectangle(v, (gx, gy), (gx + gw, gy + 18), (40, 40, 40), -1)
    ex = int(gx + gw / 2 + np.clip(e / HALF_W, -1, 1) * gw / 2)
    colr = (0, 220, 0) if abs(e) < HALF_W * 0.5 else ((0, 200, 255) if abs(e) < HALF_W * 0.85 else (0, 0, 255))
    cv2.circle(v, (ex, gy + 9), 9, colr, -1)
    cv2.putText(v, f"lane offset {e:+.1f} m", (gx, gy - 8), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if warn:
        cv2.putText(v, warn, (W // 2 - 220, 90), 0, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    if banner:
        lines = banner.split("\n")
        y0 = H // 2 - 30 * len(lines)
        cv2.rectangle(v, (W // 2 - 380, y0 - 40), (W // 2 + 380, y0 + 52 * len(lines)), (0, 0, 0), -1)
        for i, ln in enumerate(lines):
            cv2.putText(v, ln, (W // 2 - 360, y0 + 52 * i), 0, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return v


def nearest_center(x, y):
    cx = np.clip(x, -HALF_STRAIGHT, HALF_STRAIGHT)
    dx, dy = x - cx, y
    d = np.hypot(dx, dy) or 1.0
    px, py = cx + dx / d * R_MID, dy / d * R_MID
    return px, py, track_heading_ccw(px, py)


def run_trial(a, dev, ren, trial, writer_dir, headless=False, vw=None):
    x, y, yaw, rate = -30.0, -R_MID, 0.0, 0.0
    log, offs, frozen = [], 0, 0
    laps_ang, prev_ang = 0.0, np.arctan2(y, x)
    n = int(round(a.seconds / DT))
    t_next = time.perf_counter()
    k = 0
    while k < n:
        st = (x, y, yaw)
        stick = 0.0
        if frozen > 0:
            frozen -= 1
            if frozen == 0:
                x, y, yaw = nearest_center(x, y)
                rate = 0.0
        else:
            stick = dev.read(st)
            rate += (-stick * YAW_GAIN - rate) * DT / TAU
            yaw += rate * DT
            x += SPEED * np.cos(yaw) * DT
            y += SPEED * np.sin(yaw) * DT
        e = float(lateral_error(x, y))
        if frozen == 0 and abs(e) > HALF_W:
            offs += 1
            frozen = int(round(a.penalty / DT))
        ang = np.arctan2(y, x)
        laps_ang += np.angle(np.exp(1j * (ang - prev_ang)))
        prev_ang = ang
        t = k * DT
        log.append((round(t, 4), round(x, 3), round(y, 3), round(yaw, 4), round(stick, 4), round(e, 3), int(frozen > 0)))
        if not headless or vw is not None:
            warn = "OFF TRACK" if frozen > 0 else None
            if isinstance(dev, IMUInput) and not dev.alive():
                warn = "NO IMU SIGNAL"
            frame = hud(ren.render(x, y, yaw), t, a.seconds, stick, e, offs, trial, a.trials, a.pid, dev.name, warn)
            if vw is not None:
                vw.write(frame)
            if not headless:
                cv2.imshow("FPV trainer", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    return None
                t_next += DT
                time.sleep(max(0.0, t_next - time.perf_counter()))
        k += 1
    arr = np.array(log)
    flying = arr[arr[:, 6] == 0]
    res = dict(
        pid=a.pid, input=dev.name, trial=trial, seconds=a.seconds,
        mean_abs_offset_m=round(float(np.mean(np.abs(flying[:, 5]))), 3) if len(flying) else None,
        on_track_pct=round(100.0 * len(flying) / len(arr), 1),
        off_track_count=offs,
        laps=round(float(laps_ang / (2 * np.pi)), 3),
        stick_rms=round(float(np.sqrt(np.mean(flying[:, 4] ** 2))), 3) if len(flying) else None,
        stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    path = os.path.join(writer_dir, f"{a.pid}_{dev.name}_t{trial}.csv")
    np.savetxt(path, arr, delimiter=",", header="t,x,y,yaw,stick,offset_m,penalty", comments="", fmt="%g")
    return res


def wait_screen(ren, a, dev, trial, text):
    x, y, yaw = -30.0, -R_MID, 0.0
    while True:
        stick = dev.read((x, y, yaw))
        frame = hud(ren.render(x, y, yaw), 0, a.seconds, stick, 0.0, 0, trial, a.trials, a.pid, dev.name,
                    "NO IMU SIGNAL" if isinstance(dev, IMUInput) and not dev.alive() else None, text)
        cv2.imshow("FPV trainer", frame)
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            return False
        if k == ord("c"):
            dev.set_neutral()
        if k == ord(" "):
            break
    t0 = time.perf_counter()
    while (el := time.perf_counter() - t0) < 3.0:
        dev.read((x, y, yaw))
        frame = hud(ren.render(x, y, yaw), 0, a.seconds, 0.0, 0.0, 0, trial, a.trials, a.pid, dev.name, None,
                    f"{3 - int(el)}\nhold neutral")
        cv2.imshow("FPV trainer", frame)
        cv2.waitKey(15)
    dev.set_neutral()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="TEST", help="참가자 ID (예: P01)")
    ap.add_argument("--input", default="keyboard", choices=list(INPUTS))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--penalty", type=float, default=1.0, help="이탈 시 정지 시간(초)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--record", action="store_true", help="시도별 mp4 저장")
    ap.add_argument("--headless", action="store_true", help="화면 없이 최고 속도로 (geom 자가 시험용)")
    ap.add_argument("--axis", default="r", choices=JOY_AXES, help="조이스틱 축 (--list-joy 로 확인)")
    ap.add_argument("--joy-id", type=int, default=0)
    ap.add_argument("--deadband-stick", type=float, default=0.02)
    ap.add_argument("--list-joy", action="store_true", help="조이스틱 축 값을 계속 출력 (스틱을 움직여 축 찾기)")
    ap.add_argument("--imu-axis", default="roll", choices=["roll", "pitch", "yaw"])
    ap.add_argument("--deadband-deg", type=float, default=3.0)
    ap.add_argument("--full-deg", type=float, default=35.0)
    ap.add_argument("--udp-port", type=int, default=9400)
    ap.add_argument("--serial", default=None, help="USB 시리얼 포트 (예: COM5). 없으면 Wi-Fi UDP")
    ap.add_argument("--invert", action="store_true", help="방향 반대로")
    a = ap.parse_args()

    if a.list_joy:
        while True:
            print(joy_read_raw(a.joy_id), flush=True)
            time.sleep(0.2)

    os.makedirs(a.out, exist_ok=True)
    dev = INPUTS[a.input](a)
    ren = FPVRenderer()
    summary = os.path.join(a.out, "results.csv")
    new = not os.path.exists(summary)
    for trial in range(1, a.trials + 1):
        if not a.headless:
            txt = f"trial {trial}/{a.trials}  -  SPACE to start\nkeep the drone on the red track" + \
                  ("\nc = re-center IMU" if a.input == "imu" else "")
            if not wait_screen(ren, a, dev, trial, txt):
                break
        vw = None
        if a.record:
            vw = cv2.VideoWriter(os.path.join(a.out, f"{a.pid}_{dev.name}_t{trial}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (RW * 2, RH * 2))
        res = run_trial(a, dev, ren, trial, a.out, a.headless, vw)
        if vw is not None:
            vw.release()
        if res is None:
            break
        with open(summary, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res))
            if new:
                w.writeheader()
                new = False
            w.writerow(res)
        print("결과:", res, flush=True)
        if not a.headless:
            msg = (f"trial {trial} done\nmean offset {res['mean_abs_offset_m']} m\n"
                   f"on track {res['on_track_pct']}%   off-track {res['off_track_count']}x\nSPACE = next")
            frame = hud(ren.render(-30.0, -R_MID, 0.0), a.seconds, a.seconds, 0, 0, res["off_track_count"],
                        trial, a.trials, a.pid, dev.name, None, msg)
            cv2.imshow("FPV trainer", frame)
            while (cv2.waitKey(50) & 0xFF) not in (ord(" "), 27, ord("q")):
                pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
