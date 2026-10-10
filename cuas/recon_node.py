import argparse
import json
import socket
import sys
import threading
import time

MSP_SET_NAME = 11
OSD_TEXT = {"bird": "BIRD", "balloon": "BALLOON", "drone": "DRONE", "trash": "TRASH"}
CLASSES = tuple(OSD_TEXT)


def msp_frame(cmd, payload=b""):
    hdr = bytes([len(payload), cmd])
    cs = 0
    for b in hdr + payload:
        cs ^= b
    return b"$M<" + hdr + payload + bytes([cs])


class FcSerial:
    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0)

    def set_name(self, text):
        self.ser.write(msp_frame(MSP_SET_NAME, text.encode("ascii")[:16]))
        self.ser.read(64)


class FcMock:
    def __init__(self):
        self.sent = []

    def set_name(self, text):
        f = msp_frame(MSP_SET_NAME, text.encode("ascii")[:16])
        self.sent.append(f)
        print(f"[mock-fc] OSD <- {text!r:20} frame {f.hex(' ')}")


class Stabilizer:

    def __init__(self, stable=3, thr=0.6, cooldown=5.0):
        self.stable, self.thr, self.cooldown = stable, thr, cooldown
        self.cand, self.n = None, 0
        self.current, self.last_emit = None, -1e9
        self.confs = []

    def feed(self, cls, conf, now):
        if cls not in CLASSES or conf < self.thr:
            self.cand, self.n, self.confs = None, 0, []
            return None
        if cls == self.cand:
            self.n += 1
            self.confs.append(conf)
        else:
            self.cand, self.n, self.confs = cls, 1, [conf]
        if self.n < self.stable:
            return None
        avg = sum(self.confs[-self.stable:]) / self.stable
        new = cls != self.current or now - self.last_emit > self.cooldown
        if new:
            self.current, self.last_emit = cls, now
        return cls, avg, new


def source_udp(port, q):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    while True:
        data, _ = s.recvfrom(1024)
        parts = data.decode("utf-8", "ignore").strip().split(",")
        try:
            q.append((parts[0].strip().lower(), float(parts[1]) if len(parts) > 1 else 1.0))
        except ValueError:
            pass


def source_keys(q):
    keymap = {"b": "bird", "l": "balloon", "d": "drone", "t": "trash"}
    print("b=새 l=풍선 d=무인기 t=오물 (Enter). 한 번 입력이 프레임 여러 개로 들어간다.")
    for line in sys.stdin:
        cls = keymap.get(line.strip()[:1])
        if cls:
            q.extend([(cls, 0.95)] * 5)


def source_cycle(q):
    seq = ["balloon"] * 6 + ["none"] * 3 + ["drone", "bird", "drone"] + ["drone"] * 5 + ["bird"] * 4
    while True:
        for c in seq:
            q.append((c, 0.9))
            time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mock-fc", action="store_true")
    ap.add_argument("--console", default="127.0.0.1", help="노트북 IP")
    ap.add_argument("--console-port", type=int, default=9500)
    ap.add_argument("--source", choices=["udp", "keys", "cycle"], default="udp")
    ap.add_argument("--listen", type=int, default=9501, help="--source udp 일 때 분류기 결과 받는 포트")
    ap.add_argument("--stable", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--name", default="recon1")
    ap.add_argument("--seconds", type=float, default=0, help="0=무한, 시험용")
    a = ap.parse_args()

    fc = FcMock() if a.mock_fc else FcSerial(a.port, a.baud)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    q = []
    target = {"udp": lambda: source_udp(a.listen, q), "keys": lambda: source_keys(q),
              "cycle": lambda: source_cycle(q)}[a.source]
    threading.Thread(target=target, daemon=True).start()

    st = Stabilizer(a.stable, a.thr, a.cooldown)
    fc.set_name("SCANNING")
    osd_text, last_seen = "SCANNING", time.monotonic()
    t0 = time.monotonic()
    while not a.seconds or time.monotonic() - t0 < a.seconds:
        if not q:
            if osd_text != "SCANNING" and time.monotonic() - last_seen > 2.0:
                fc.set_name("SCANNING")
                osd_text = "SCANNING"
            time.sleep(0.02)
            continue
        cls, conf = q.pop(0)
        now = time.monotonic()
        r = st.feed(cls, conf, now)
        if r is None:
            continue
        cls, avg, new = r
        last_seen = now
        text = f"{OSD_TEXT[cls]} {int(avg * 100)}"
        if text.split()[0] != osd_text.split()[0]:
            fc.set_name(text)
            osd_text = text
        if new:
            msg = json.dumps({"cls": cls, "conf": round(avg, 3), "src": a.name})
            out.sendto(msg.encode(), (a.console, a.console_port))
            print(f"[event] {msg}")


if __name__ == "__main__":
    main()
