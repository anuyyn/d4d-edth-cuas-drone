#!/usr/bin/env python3
import argparse
import csv
import socket
import struct
import time

from track3 import MSP_RC, MSP_SET_RAW_RC, MspLink

AXIS = {"roll": 0, "pitch": 1, "yaw": 2}


def tilt_to_cmd(d, dead, full):
    if abs(d) < dead:
        return 0.0
    s = 1.0 if d > 0 else -1.0
    return max(-1.0, min(1.0, s * (abs(d) - dead) / (full - dead)))


class MockLink:

    def __init__(self):
        self.t0 = time.monotonic()
        self.sent = []

    def poll(self):
        pass

    def send(self, cmd, payload=b""):
        if cmd == MSP_SET_RAW_RC:
            self.sent.append((time.monotonic() - self.t0, struct.unpack("<8H", payload)))

    def rc(self, max_age=0.5):
        el = time.monotonic() - self.t0
        aux4 = 1900 if 2.0 < el < 6.0 else 1000
        return [1500, 1500, 1500, 1200, 1000, 1000, 1000, aux4]

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--udp-port", type=int, default=9400)
    ap.add_argument("--imu-axis", default="roll", choices=list(AXIS), help="어느 기울기를 쓸지 (손 좌우 = roll)")
    ap.add_argument("--map", default="yaw", choices=["yaw", "roll", "both"], help="IMU 가 덮어쓸 채널")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--deadband-deg", type=float, default=3.0)
    ap.add_argument("--full-deg", type=float, default=35.0)
    ap.add_argument("--range", type=int, default=200, help="1500 기준 ±us (처음엔 작게)")
    ap.add_argument("--yaw-idx", type=int, default=3, help="MSP 페이로드 yaw 위치 (AETR=3)")
    ap.add_argument("--roll-idx", type=int, default=0)
    ap.add_argument("--sw-aux", type=int, default=4)
    ap.add_argument("--sw-min", type=int, default=1700)
    ap.add_argument("--stale", type=float, default=0.2, help="IMU 가 이 초 동안 안 오면 송신 중단 (+BF 300ms 뒤 교관 복귀)")
    ap.add_argument("--hz", type=float, default=50)
    ap.add_argument("--log", default=f"imu_msp_{time.strftime('%m%d_%H%M%S')}.csv")
    ap.add_argument("--seconds", type=float, default=0, help="0=무한")
    ap.add_argument("--mock", action="store_true", help="FC 없이 가짜 FC 로 시험")
    a = ap.parse_args()

    link = MockLink() if a.mock else MspLink(a.port, a.baud)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", a.udp_port))
    sock.setblocking(False)
    ai, inv, sw_idx = AXIS[a.imu_axis], (-1.0 if a.invert else 1.0), 3 + a.sw_aux

    imu, imu_t, recent = None, 0.0, []
    neutral, was_on = 0.0, False
    period, t0 = 1.0 / a.hz, time.monotonic()
    nxt = t0
    fh = open(a.log, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["t", "imu_roll", "imu_pitch", "imu_yaw", "imu_age", "sw_us", "engaged", "cmd", "yaw_us", "roll_us"])
    print(f"UDP {a.udp_port} 대기, 스위치 AUX{a.sw_aux}>={a.sw_min} 일 때 {a.map} 덮어씀, 로그 {a.log}", flush=True)
    try:
        while not a.seconds or time.monotonic() - t0 < a.seconds:
            while True:
                try:
                    pkt, _ = sock.recvfrom(256)
                except (BlockingIOError, OSError):
                    break
                try:
                    v = [float(x) for x in pkt.decode(errors="ignore").strip().split(",")[-3:]]
                except ValueError:
                    continue
                if len(v) == 3:
                    imu, imu_t = v, time.monotonic()
                    recent = (recent + [v[ai]])[-15:]
            now = time.monotonic()
            age = now - imu_t if imu else 99.0
            link.poll()
            rc = link.rc()
            link.send(MSP_RC)
            sw = rc[sw_idx] if rc and len(rc) > sw_idx else None
            on = sw is not None and sw >= a.sw_min
            ok = imu is not None and age < a.stale
            if on and not was_on and recent:
                neutral = sum(recent) / len(recent)
                print(f"[{now - t0:6.1f}s] 권한 받음, 중립 {neutral:+.1f}도", flush=True)
            if was_on and not on:
                print(f"[{now - t0:6.1f}s] 교관이 회수", flush=True)
            was_on = on
            cmd, yaw_us, roll_us = 0.0, "", ""
            if on and ok:
                cmd = tilt_to_cmd((imu[ai] - neutral) * inv, a.deadband_deg, a.full_deg)
                ch = [1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000]
                us = int(1500 + cmd * a.range)
                if a.map in ("yaw", "both"):
                    ch[a.yaw_idx] = yaw_us = us
                if a.map in ("roll", "both"):
                    ch[a.roll_idx] = roll_us = us
                link.send(MSP_SET_RAW_RC, struct.pack("<8H", *ch))
            w.writerow([f"{now - t0:.3f}"] + ([f"{x:.2f}" for x in imu] if imu else ["", "", ""]) +
                       [f"{min(age, 99):.3f}", sw if sw is not None else "", int(on and ok), f"{cmd:.3f}", yaw_us, roll_us])
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        link.close()
        if a.mock:
            s = link.sent
            print(f"[mock] MSP_SET_RAW_RC {len(s)}개, 시각 {s[0][0]:.2f}~{s[-1][0]:.2f}s" if s else "[mock] 보낸 것 없음")
            if s:
                ys = [p[a.yaw_idx] for _, p in s]
                print(f"[mock] yaw us 범위 {min(ys)}~{max(ys)}, 처음 {ys[0]}, 마지막 {ys[-1]}")


if __name__ == "__main__":
    main()
