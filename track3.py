#!/usr/bin/env python3
import argparse
import csv
import struct
import time

import numpy as np

PRESETS = {
    "yellow": dict(u=(0, 110), v=(128, 175), ymin=90),
    "red":    dict(u=(80, 135), v=(165, 255), ymin=40),
    "blue":   dict(u=(150, 255), v=(0, 120), ymin=30),
    "green":  dict(u=(0, 120), v=(0, 120), ymin=40),
}


def split_yuv420(buf, w, h, stride):
    y = buf[:h, :w]
    uv = buf[h:h + h // 2].reshape(-1)
    half = (stride // 2) * (h // 2)
    u = uv[:half].reshape(h // 2, stride // 2)[:, :w // 2]
    v = uv[half:2 * half].reshape(h // 2, stride // 2)[:, :w // 2]
    return y, u, v


def detect_uv(y, u, v, p, min_px):
    ys = y[::2, ::2]
    mask = ((u >= p["u"][0]) & (u <= p["u"][1]) &
            (v >= p["v"][0]) & (v <= p["v"][1]) &
            (ys >= p["ymin"]))
    rows, cols = np.nonzero(mask)
    n = cols.size
    if n < min_px:
        return None, n
    mh, mw = mask.shape
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    fill = n / float((x1 - x0 + 1) * (y1 - y0 + 1))
    cx = (cols.mean() + 0.5) / mw * 2 - 1
    cy = (rows.mean() + 0.5) / mh * 2 - 1
    return (cx, cy, (x0, y0, x1, y1), n, fill), n


class Confidence:

    def __init__(self, full_px, max_jump=0.5, lost_frames=3):
        self.full_px = full_px
        self.max_jump = max_jump
        self.lost_frames = lost_frames
        self.prev = None
        self.miss = 0

    def update(self, det):
        if det is None:
            self.miss += 1
            if self.miss >= self.lost_frames:
                self.prev = None
            return 0.0
        cx, cy, _, n, fill = det
        s_size = min(1.0, n / self.full_px)
        s_fill = min(1.0, fill / 0.6)
        if self.prev is None:
            s_time = 0.5
        else:
            jump = np.hypot(cx - self.prev[0], cy - self.prev[1])
            s_time = max(0.0, 1.0 - jump / self.max_jump)
        self.prev = (cx, cy)
        self.miss = 0
        return float(s_size * s_fill * s_time)


class YawController:

    def __init__(self, kp, deadband, max_cmd, slew, conf_on):
        self.kp, self.deadband, self.max_cmd = kp, deadband, max_cmd
        self.slew, self.conf_on = slew, conf_on
        self.cmd = 0.0

    def step(self, dx, conf):
        if conf < self.conf_on or dx is None:
            target = 0.0
        else:
            e = 0.0 if abs(dx) < self.deadband else dx
            target = float(np.clip(self.kp * e, -self.max_cmd, self.max_cmd))
        self.cmd += float(np.clip(target - self.cmd, -self.slew, self.slew))
        return self.cmd, conf >= self.conf_on

    def reset(self):
        self.cmd = 0.0


MSP_API_VERSION = 1
MSP_RC = 105
MSP_SET_RAW_RC = 200


def msp_frame(cmd, payload=b""):
    hdr = bytes([len(payload), cmd])
    cs = 0
    for b in hdr + payload:
        cs ^= b
    return b"$M<" + hdr + payload + bytes([cs])


class MspLink:

    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = bytearray()
        self.last = {}
        self.errors = 0

    def send(self, cmd, payload=b""):
        self.ser.write(msp_frame(cmd, payload))

    def poll(self):
        n = self.ser.in_waiting
        if n:
            self.buf += self.ser.read(n)
        self.feed()

    def feed(self):
        b = self.buf
        while True:
            i = b.find(b"$M")
            if i < 0:
                del b[:max(0, len(b) - 1)]
                return
            del b[:i]
            if len(b) < 6:
                return
            size, cmd = b[3], b[4]
            if len(b) < 6 + size:
                return
            payload = bytes(b[5:5 + size])
            cs = 0
            for x in b[3:5 + size]:
                cs ^= x
            if b[2:3] == b">" and cs == b[5 + size]:
                self.last[cmd] = (time.monotonic(), payload)
            elif b[2:3] == b"!":
                self.last[("err", cmd)] = (time.monotonic(), payload)
            else:
                self.errors += 1
            del b[:6 + size]

    def rc(self, max_age=0.5):
        v = self.last.get(MSP_RC)
        if v is None or time.monotonic() - v[0] > max_age:
            return None
        p = v[1]
        return list(struct.unpack(f"<{len(p) // 2}H", p))

    def close(self):
        self.ser.close()


class OutNone:
    def engaged(self):
        return True

    def send(self, dx, dy, conf, cmd, active):
        pass

    def close(self):
        pass


class OutUart:

    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0)

    def engaged(self):
        return True

    def send(self, dx, dy, conf, cmd, active):
        dx = 0.0 if dx is None else dx
        dy = 0.0 if dy is None else dy
        body = f"TRK,{dx:.3f},{dy:.3f},{conf:.2f},{cmd:.3f},{int(active)}"
        cs = 0
        for ch in body.encode():
            cs ^= ch
        self.ser.write(f"${body}*{cs:02X}\r\n".encode())

    def close(self):
        self.ser.close()


class OutMsp:

    def __init__(self, port, baud, yaw_range, yaw_idx=3, sw_aux=4, sw_min=1700):
        self.link = MspLink(port, baud)
        self.yaw_range = yaw_range
        self.yaw_idx = yaw_idx
        self.sw_idx = 3 + sw_aux
        self.sw_min = sw_min
        self.sw_us = None

    def engaged(self):
        self.link.poll()
        rc = self.link.rc()
        self.link.send(MSP_RC)
        if rc is None or len(rc) <= self.sw_idx:
            self.sw_us = None
            return False
        self.sw_us = rc[self.sw_idx]
        return self.sw_us >= self.sw_min

    def send(self, dx, dy, conf, cmd, active):
        if not active:
            return
        yaw = int(1500 + cmd * self.yaw_range)
        ch = [1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000]
        ch[self.yaw_idx] = yaw
        self.link.send(MSP_SET_RAW_RC, struct.pack("<8H", *ch))

    def close(self):
        self.link.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="red", choices=sorted(PRESETS))
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--h", type=int, default=120)
    ap.add_argument("--out", default="none", choices=["none", "uart", "msp"])
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--kp", type=float, default=0.2)
    ap.add_argument("--deadband", type=float, default=0.08)
    ap.add_argument("--max-cmd", type=float, default=0.4)
    ap.add_argument("--slew", type=float, default=0.1, help="프레임당 최대 명령 변화")
    ap.add_argument("--conf-on", type=float, default=0.35, help="이 값 미만이면 조종사에게 넘김")
    ap.add_argument("--min-px", type=int, default=6)
    ap.add_argument("--full-px", type=float, default=15, help="이 픽셀 수(U/V 1/4 해상도)면 크기 점수 1. 30cm 풍선@6m ~20px")
    ap.add_argument("--yaw-range", type=int, default=200, help="MSP yaw ±us")
    ap.add_argument("--yaw-idx", type=int, default=3, help="MSP 페이로드의 yaw 위치 (BF map 기준, AETR=3)")
    ap.add_argument("--sw-aux", type=int, default=4, help="MSP OVERRIDE 를 건 AUX 번호 (AUX1=5채널)")
    ap.add_argument("--sw-min", type=int, default=1700, help="이 us 이상이면 스위치 ON (Modes 탭 범위와 맞출 것)")
    ap.add_argument("--preview", action="store_true", help="DRM 프리뷰(고글) 켬")
    ap.add_argument("--overlay", action="store_true", help="검출 상자 오버레이 (fps 비교용)")
    ap.add_argument("--calib", action="store_true", help="중앙 16x16 U/V 중앙값만 출력")
    ap.add_argument("--log", default=f"track_{time.strftime('%m%d_%H%M%S')}.csv")
    ap.add_argument("--seconds", type=float, default=0, help="0=무한")
    a = ap.parse_args()

    from picamera2 import Picamera2, Preview

    W, H = a.w, a.h
    DW, DH = 320, 240
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (DW, DH), "format": "XRGB8888"},
        lores={"size": (W, H), "format": "YUV420"},
        buffer_count=4)
    picam2.configure(cfg)
    stride = picam2.camera_config["lores"]["stride"]
    if a.preview or a.overlay:
        picam2.start_preview(Preview.DRM)
    picam2.start()
    time.sleep(1.0)

    p = PRESETS[a.target]
    conf_est = Confidence(full_px=a.full_px)
    ctl = YawController(a.kp, a.deadband, a.max_cmd, a.slew, a.conf_on)
    out = {"none": OutNone,
           "uart": lambda: OutUart(a.port, a.baud),
           "msp": lambda: OutMsp(a.port, a.baud, a.yaw_range, a.yaw_idx, a.sw_aux, a.sw_min)}[a.out]()
    overlay = np.zeros((DH, DW, 4), np.uint8) if a.overlay else None

    f = open(a.log, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["t", "lat_ms", "dx", "dy", "n", "fill", "conf", "cmd", "sw", "active"])
    print(f"target={a.target} out={a.out} lores={W}x{H} stride={stride} "
          f"preview={a.preview} overlay={a.overlay} log={a.log}", flush=True)

    t0 = time.monotonic()
    last = t0
    frames = 0
    lat_sum = 0.0
    try:
        while True:
            req = picam2.capture_request()
            buf = req.make_array("lores")
            ts = req.get_metadata().get("SensorTimestamp")
            req.release()
            y, u, v = split_yuv420(buf, W, H, stride)

            if a.calib:
                cu, cv_ = u.shape[1] // 2, u.shape[0] // 2
                pu = u[cv_ - 8:cv_ + 8, cu - 8:cu + 8]
                pv = v[cv_ - 8:cv_ + 8, cu - 8:cu + 8]
                py = y[2 * cv_ - 16:2 * cv_ + 16, 2 * cu - 16:2 * cu + 16]
                print(f"center Y={int(np.median(py))} U={int(np.median(pu))} "
                      f"V={int(np.median(pv))}", flush=True)
                time.sleep(0.5)
                continue

            det, n = detect_uv(y, u, v, p, a.min_px)
            conf = conf_est.update(det)
            dx = dy = None
            fill = 0.0
            if det is not None:
                dx, dy, box, n, fill = det
            sw = out.engaged()
            cmd, active = ctl.step(dx, conf)
            if not sw:
                ctl.reset()
                cmd, active = 0.0, False
            out.send(dx, dy, conf, cmd, active)

            now_ns = time.monotonic_ns()
            lat_ms = (now_ns - ts) / 1e6 if ts else float("nan")
            wr.writerow([f"{now_ns / 1e9:.4f}", f"{lat_ms:.1f}",
                         "" if dx is None else f"{dx:.3f}",
                         "" if dy is None else f"{dy:.3f}",
                         n, f"{fill:.2f}", f"{conf:.2f}", f"{cmd:.3f}", int(sw), int(active)])

            if overlay is not None:
                overlay[:] = 0
                if det is not None:
                    sx, sy = DW / (W // 2), DH / (H // 2)
                    x0, y0, x1, y1 = box
                    x0, x1 = int(x0 * sx), min(DW - 1, int((x1 + 1) * sx))
                    y0, y1 = int(y0 * sy), min(DH - 1, int((y1 + 1) * sy))
                    c = np.array([0, 255, 0, 255] if active else [0, 160, 255, 255], np.uint8)
                    overlay[y0:y0 + 2, x0:x1] = c
                    overlay[y1 - 2:y1, x0:x1] = c
                    overlay[y0:y1, x0:x0 + 2] = c
                    overlay[y0:y1, x1 - 2:x1] = c
                picam2.set_overlay(overlay)

            frames += 1
            if lat_ms == lat_ms:
                lat_sum += lat_ms
            t = time.monotonic()
            if t - last >= 2.0:
                print(f"fps={frames / (t - t0):.1f} lat_avg={lat_sum / max(frames, 1):.0f}ms "
                      f"px={n} conf={conf:.2f} cmd={cmd:+.2f} sw={'ON' if sw else 'OFF'} "
                      f"{'AUTO' if active else 'PILOT'}",
                      flush=True)
                last = t
            if a.seconds and t - t0 >= a.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        f.close()
        picam2.stop()


if __name__ == "__main__":
    main()
