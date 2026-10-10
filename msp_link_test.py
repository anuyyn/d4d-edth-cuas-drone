#!/usr/bin/env python3
import argparse
import math
import struct
import time

from track3 import MSP_API_VERSION, MSP_RC, MSP_SET_RAW_RC, MspLink

NAMES = ["ROLL", "PITCH", "YAW", "THR"] + [f"AUX{i}" for i in range(1, 13)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--wiggle", action="store_true")
    ap.add_argument("--sw-aux", type=int, default=4)
    ap.add_argument("--sw-min", type=int, default=1700)
    ap.add_argument("--yaw-idx", type=int, default=3)
    a = ap.parse_args()

    link = MspLink(a.port, a.baud)
    link.send(MSP_API_VERSION)
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end and MSP_API_VERSION not in link.last:
        link.poll()
        time.sleep(0.01)
    if MSP_API_VERSION not in link.last:
        print("응답 없음. 확인: TX<->RX 교차 배선, GND 공통, Ports 탭 해당 UART MSP ON, "
              "보드레이트, 파이 시리얼 콘솔 OFF / dtoverlay=disable-bt")
        return
    p = link.last[MSP_API_VERSION][1]
    print(f"MSP OK: protocol={p[0]} API={p[1]}.{p[2]}")

    sw_idx = 3 + a.sw_aux
    t0 = time.monotonic()
    last_print = 0.0
    try:
        while True:
            link.poll()
            link.send(MSP_RC)
            rc = link.rc()
            now = time.monotonic()
            on = rc is not None and len(rc) > sw_idx and rc[sw_idx] >= a.sw_min
            if a.wiggle and on:
                yaw = int(1500 + 150 * math.sin((now - t0) * 2 * math.pi / 4.0))
                ch = [1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000]
                ch[a.yaw_idx] = yaw
                link.send(MSP_SET_RAW_RC, struct.pack("<8H", *ch))
            if rc is not None and now - last_print >= 0.3:
                s = " ".join(f"{NAMES[i]}={v}" for i, v in enumerate(rc[:8]))
                tag = f"  SW(AUX{a.sw_aux})={'ON' if on else 'off'}"
                if a.wiggle and on:
                    tag += "  -> yaw 흔드는 중"
                print(s + tag, flush=True)
                last_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


if __name__ == "__main__":
    main()
