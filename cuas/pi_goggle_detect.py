import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FALLBACK = ["rpicam-hello", "-t", "0", "--framerate", "30", "--width", "640", "--height", "480"]
COL = {"trash": (255, 0, 255), "balloon": (255, 200, 0), "bird": (0, 200, 0), "drone": (255, 0, 0),
       "unknown": (160, 160, 160)}


def fallback(why):
    print("탐지 없이 영상만 송출로 전환:", why, flush=True)
    os.execvp(FALLBACK[0], FALLBACK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "appearance", "out", "cuas_cls.onnx"))
    ap.add_argument("--udp", default="172.20.10.15:9500",
                    help="관제 노트북으로 판정+캡처(JSON). 기본 = 팀 핫스팟 브로드캐스트(노트북 IP 몰라도 됨), 끄려면 none")
    ap.add_argument("--every", type=int, default=1, help="분석할 프레임 간격(lores 기준)")
    ap.add_argument("--bench", type=float, default=0)
    a = ap.parse_args()
    try:
        import cv2
        import numpy as np
        from picamera2 import Picamera2, Preview
        import motion_id as M
        from appearance_id import AppearanceClassifier
    except Exception as e:
        if a.bench:
            raise
        fallback(f"모듈 없음: {e}")

    app = AppearanceClassifier(a.model)
    cv2.setNumThreads(4)
    cam = Picamera2()
    cfg = cam.create_preview_configuration(main={"size": (640, 480)}, lores={"size": (320, 240), "format": "YUV420"},
                                           buffer_count=4, controls={"FrameRate": 30})
    cam.configure(cfg)
    if not a.bench:
        try:
            cam.start_preview(Preview.DRM, x=0, y=0, width=720, height=480)
        except Exception as e:
            cam.close()
            fallback(f"DRM 미리보기 실패: {e}")
    cam.start()

    sender = None
    if a.udp and a.udp != "none":
        from detect_msg import Sender
        sender = Sender(a.udp, src="pi")

    stats = {"n": 0, "t0": time.time(), "last_print": time.time()}

    def frames():
        i = 0
        while True:
            yuv = cam.capture_array("lores")
            i += 1
            if i % a.every:
                continue
            yield time.time(), cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            if a.bench and time.time() - stats["t0"] > a.bench:
                return

    P = M.Pipeline(30.0 / a.every, appearance=app, recon=True)

    def on_frame(t, res, gray, tracks, off):
        stats["n"] += 1
        if not a.bench:
            ov = np.zeros((480, 640, 4), np.uint8)
            for tr in res:
                c, cf, _ = tr.last
                x, y, e = (tr.x[-1] + off[0]) * 2, (tr.y[-1] + off[1]) * 2, tr.extent_px() * 2
                col = COL.get(c, (255, 255, 255)) + (255,)
                cv2.rectangle(ov, (int(x - e / 2), int(y - e / 2)), (int(x + e / 2), int(y + e / 2)), col, 3)
                cv2.putText(ov, f"{c.upper()} {cf:.2f}", (int(x - e / 2), max(24, int(y - e / 2) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            if not res and P.scene:
                c, cf = P.scene
                col = COL.get(c, (255, 255, 255)) + (255,)
                cv2.putText(ov, f"{c.upper()} {cf:.2f} (FULL VIEW)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            cam.set_overlay(ov)
        if sender:
            sender.send_frame(M.LAST_FRAME, [(((q.x[-1] + off[0]) - q.extent_px() / 2, (q.y[-1] + off[1]) - q.extent_px() / 2,
                                                (q.x[-1] + off[0]) + q.extent_px() / 2, (q.y[-1] + off[1]) + q.extent_px() / 2),
                                               q.last[0], q.last[1]) for q in res]
                              or ([(None, P.scene[0], P.scene[1])] if P.scene else []))
        if res and sender:
            tr = max(res, key=lambda q: q.extent_px())
            args = (tr.last[0], tr.last[1], M.LAST_FRAME, tr.x[-1] + off[0], tr.y[-1] + off[1], tr.extent_px())
            sender.maybe_send(*args)
            sender.send_live(*args)
        now = time.time()
        if now - stats["last_print"] > 5:
            fps = stats["n"] / (now - stats["t0"])
            labels = ", ".join(f"{q.last[0]} {q.last[1]:.2f}" for q in res) or "-"
            print(f"[{now - stats['t0']:6.1f}s] 분석 {fps:.1f} fps | {labels}", flush=True)
            stats["last_print"] = now

    try:
        for t, img in frames():
            res = P.step(t, img)
            on_frame(t, res, P.gray, P.tracks, P.off)
    finally:
        fps = stats["n"] / max(1e-6, time.time() - stats["t0"])
        print(f"끝: 분석 {stats['n']} 프레임, 평균 {fps:.1f} fps", flush=True)
        cam.close()


if __name__ == "__main__":
    main()
