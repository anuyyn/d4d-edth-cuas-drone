import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_id as M
from appearance_id import AppearanceClassifier
from detect_msg import Sender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--udp", default="127.0.0.1:9500")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--width", type=int, default=320, help="파이 분석 해상도 흉내(폭)")
    a = ap.parse_args()
    app = AppearanceClassifier()
    tx = Sender(a.udp, src="fake_pi")
    while True:
        for v in a.videos:
            cap = cv2.VideoCapture(v)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            p = M.Pipeline(fps, appearance=app, recon=True)
            t0, i = time.time(), 0
            print("송출:", os.path.basename(v), flush=True)
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                t = i / fps
                i += 1
                h, w = f.shape[:2]
                f = cv2.resize(f, (a.width, int(h * a.width / w)), interpolation=cv2.INTER_AREA)
                res = p.step(t, f)
                dets = [(p.box(q), q.last[0], q.last[1]) for q in res]
                if not dets and p.scene:
                    dets = [(None, p.scene[0], p.scene[1])]
                tx.send_frame(f, dets)
                if res:
                    q = max(res, key=lambda r: r.extent_px())
                    x1, y1, x2, y2 = p.box(q)
                    args = (q.last[0], q.last[1], f, (x1 + x2) / 2, (y1 + y2) / 2, q.extent_px())
                    tx.maybe_send(*args)
                    tx.send_live(*args)
                time.sleep(max(0.0, t - (time.time() - t0)))
        if not a.loop:
            break


if __name__ == "__main__":
    main()
