import argparse
import collections
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion_id as M
from appearance_id import AppearanceClassifier

COL = {"trash": (255, 0, 255), "balloon": (0, 200, 255), "bird": (0, 200, 0), "drone": (0, 0, 255),
       "unknown": (160, 160, 160)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--split-s", type=float, default=0)
    ap.add_argument("--normal", action="store_true", help="정찰 모드 대신 일반(고정 카메라) 모드")
    ap.add_argument("--udp", default=None, help="관제 로그로 판정+캡처 송신 (예: 127.0.0.1:9500)")
    ap.add_argument("--realtime", action="store_true", help="영상 속도대로 천천히(관제 화면 시연용)")
    a = ap.parse_args()
    app = AppearanceClassifier()
    cap = cv2.VideoCapture(a.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    Hc = frames[0].shape[0] - a.crop_bottom
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    before, after = collections.Counter(), collections.Counter()
    main_cls = collections.Counter()
    shown = [0]
    sender = None
    if a.udp:
        from detect_msg import Sender
        sender = Sender(a.udp, src="video")
    t_start = [None]

    def on(t, res, g, trs, off):
        i = shown[0]
        shown[0] += 1
        f = frames[i].copy()
        main_cls[max(res, key=lambda q: q.extent_px()).last[0] if res else "없음"] += 1
        if res and sender:
            mt = max(res, key=lambda q: q.extent_px())
            args = (mt.last[0], mt.last[1], frames[i], mt.x[-1] + off[0], mt.y[-1] + off[1], mt.extent_px())
            sender.maybe_send(*args)
            sender.send_live(*args)
        if a.realtime:
            import time
            t_start[0] = t_start[0] or time.time() - t
            time.sleep(max(0.0, t - (time.time() - t_start[0])))
        for tr in res:
            c, cf, why = tr.last
            (after if a.split_s and t >= a.split_s else before)[c] += 1
            x, y, e = tr.x[-1] + off[0], tr.y[-1] + off[1], tr.extent_px()
            cv2.rectangle(f, (int(x - e / 2), int(y - e / 2)), (int(x + e / 2), int(y + e / 2)), COL[c], 2)
            cv2.putText(f, f"{c.upper()} {cf:.2f}", (int(x - e / 2), max(20, int(y - e / 2) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL[c], 2)
        vw.write(f)

    M.run(((i / fps, f[:Hc]) for i, f in enumerate(frames)), fps, None, on_frame=on, appearance=app,
          recon=not a.normal)
    vw.release()
    n = sum(main_cls.values())
    print(f"장면별 가장 큰 표적 판정 ({n}장면):", ", ".join(f"{k} {v}({v / n:.0%})" for k, v in main_cls.most_common()))
    print("앞 구간 판정(트랙×프레임):", dict(before))
    if a.split_s:
        print("뒤 구간 판정(트랙×프레임):", dict(after))
    print("저장:", a.out)


if __name__ == "__main__":
    main()
