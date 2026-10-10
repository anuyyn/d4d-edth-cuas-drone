import argparse
import math
import os
import struct
import time

MSP_API_VERSION = 1
MSP_RC = 105
MSP_SET_RAW_RC = 200
NAMES = ["ROLL", "PITCH", "YAW", "THR"] + [f"AUX{i}" for i in range(1, 13)]


def msp_frame(cmd, payload=b""):
    hdr = bytes([len(payload), cmd])
    cs = 0
    for b in hdr + payload:
        cs ^= b
    return b"$M<" + hdr + payload + bytes([cs])


def open_serial(port, baud):
    import termios
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    speed = getattr(termios, f"B{baud}")
    att = termios.tcgetattr(fd)
    att[0] = 0
    att[1] = 0
    att[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    att[3] = 0
    att[4] = att[5] = speed
    att[6][termios.VMIN] = 0
    att[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, att)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


class Msp:

    def __init__(self, port, baud):
        self.fd = open_serial(port, baud)
        self.buf = bytearray()
        self.last = {}

    def send(self, cmd, payload=b""):
        try:
            os.write(self.fd, msp_frame(cmd, payload))
        except (BlockingIOError, OSError):
            pass

    def poll(self):
        while True:
            try:
                d = os.read(self.fd, 512)
            except BlockingIOError:
                break
            if not d:
                break
            self.buf += d
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
            cs = 0
            for x in b[3:5 + size]:
                cs ^= x
            if b[2:3] == b">" and cs == b[5 + size]:
                self.last[cmd] = (time.monotonic(), bytes(b[5:5 + size]))
            del b[:6 + size]

    def rc(self, max_age=0.5):
        v = self.last.get(MSP_RC)
        if v is None or time.monotonic() - v[0] > max_age:
            return None
        return list(struct.unpack(f"<{len(v[1]) // 2}H", v[1]))

    def close(self):
        os.close(self.fd)


class FcLink:

    def __init__(self, port="/dev/serial0", baud=115200, sw_aux=4, sw_min=1700, yaw_idx=3, yaw_range=200,
                 kp=0.2, deadband=0.08, max_cmd=0.4, slew=0.1, width=640, height=480,
                 target="any", follow="yaw", pitch_idx=1, pitch_range=100, kp_pitch=0.8, pitch_fwd=0.5,
                 pitch_back=0.3, size_want=0.25, lost_s=1.0):
        self.sw_idx = 3 + sw_aux
        self.sw_min, self.yaw_idx, self.yaw_range = sw_min, yaw_idx, yaw_range
        self.kp, self.deadband, self.max_cmd, self.slew = kp, deadband, max_cmd, slew
        self.width, self.height = width, height
        self.targets = None if target in ("any", "", None) else set(target.split(","))
        self.follow = follow
        self.pitch_idx, self.pitch_range = pitch_idx, pitch_range
        self.kp_pitch, self.pitch_fwd, self.pitch_back, self.size_want = kp_pitch, pitch_fwd, pitch_back, size_want
        self.lost_s = lost_s
        self.cmd = self.pcmd = 0.0
        self.lock = None
        self.sw = False
        self.sw_us = None
        self.err = None
        try:
            self.msp = Msp(port, baud)
            self.ok = True
        except Exception as e:
            self.msp, self.ok, self.err = None, False, f"{port} 열기 실패: {e}"

    def linked(self):
        return self.ok and self.msp.rc() is not None

    def pick(self, boxes):
        c = [d for d in boxes if self.targets is None or d[1] in self.targets]
        if not c:
            return None
        now = time.monotonic()
        ctr = lambda d: ((d[0][0] + d[0][2]) / 2, (d[0][1] + d[0][3]) / 2)
        if self.lock and now - self.lock[2] < self.lost_s:
            d = min(c, key=lambda d: math.hypot(ctr(d)[0] - self.lock[0], ctr(d)[1] - self.lock[1]))
            if math.hypot(ctr(d)[0] - self.lock[0], ctr(d)[1] - self.lock[1]) < 0.25 * self.width:
                self.lock = (*ctr(d), now)
                return d
        d = max(c, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
        self.lock = (*ctr(d), now)
        return d

    def update(self, boxes):
        if not self.ok:
            return False, 0.0, None
        self.msp.poll()
        self.msp.send(MSP_RC)
        rc = self.msp.rc()
        self.sw_us = rc[self.sw_idx] if rc is not None and len(rc) > self.sw_idx else None
        self.sw = self.sw_us is not None and self.sw_us >= self.sw_min
        if not self.sw:
            self.cmd = self.pcmd = 0.0
            self.lock = None
            return False, 0.0, None
        tgt = self.pick(boxes)
        want, pwant = 0.0, 0.0
        if tgt is not None:
            x1, y1, x2, y2 = tgt[0]
            dx = ((x1 + x2) / 2 - self.width / 2) / (self.width / 2)
            e = 0.0 if abs(dx) < self.deadband else dx
            want = max(-self.max_cmd, min(self.max_cmd, self.kp * e))
            if self.follow == "yaw+pitch":
                size = (y2 - y1) / self.height
                pe = (self.size_want - size) / self.size_want
                pwant = max(-self.pitch_back, min(self.pitch_fwd, self.kp_pitch * pe))
                if abs(dx) > 0.3 and pwant > 0:
                    pwant = 0.0
        self.cmd += max(-self.slew, min(self.slew, want - self.cmd))
        self.pcmd += max(-0.05, min(0.05, pwant - self.pcmd))
        ch = [1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000]
        ch[self.yaw_idx] = int(1500 + self.cmd * self.yaw_range)
        ch[self.pitch_idx] = int(1500 + self.pcmd * self.pitch_range)
        self.msp.send(MSP_SET_RAW_RC, struct.pack("<8H", *ch))
        return True, self.cmd, tgt

    def close(self):
        if self.msp:
            self.msp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sw-aux", type=int, default=4)
    ap.add_argument("--sw-min", type=int, default=1700)
    ap.add_argument("--yaw-idx", type=int, default=3)
    ap.add_argument("--wiggle", action="store_true")
    ap.add_argument("--seconds", type=float, default=0)
    a = ap.parse_args()
    m = Msp(a.port, a.baud)
    m.send(MSP_API_VERSION)
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end and MSP_API_VERSION not in m.last:
        m.poll()
        time.sleep(0.01)
    if MSP_API_VERSION not in m.last:
        print("FC 응답 없음. 확인: 파이 GPIO14(TX)→FC R6, GPIO15(RX)→FC T6, GND 공통 / "
              "BF Ports 탭 UART6 MSP 켬 115200 / 파이 config.txt enable_uart=1, cmdline 의 console=serial0 제거")
        return
    p = m.last[MSP_API_VERSION][1]
    print(f"MSP OK: protocol={p[0]} API={p[1]}.{p[2]}", flush=True)
    sw_idx = 3 + a.sw_aux
    t0 = last = time.monotonic()
    try:
        while not a.seconds or time.monotonic() - t0 < a.seconds:
            m.poll()
            m.send(MSP_RC)
            rc = m.rc()
            now = time.monotonic()
            on = rc is not None and len(rc) > sw_idx and rc[sw_idx] >= a.sw_min
            if a.wiggle and on:
                ch = [1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000]
                ch[a.yaw_idx] = int(1500 + 150 * math.sin((now - t0) * 2 * math.pi / 4.0))
                m.send(MSP_SET_RAW_RC, struct.pack("<8H", *ch))
            if rc is not None and now - last >= 0.3:
                s = " ".join(f"{NAMES[i]}={v}" for i, v in enumerate(rc[:12]))
                print(f"{s}  SW(AUX{a.sw_aux})={'ON' if on else 'off'}" + ("  -> yaw 흔드는 중" if a.wiggle and on else ""),
                      flush=True)
                last = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        m.close()


if __name__ == "__main__":
    main()
