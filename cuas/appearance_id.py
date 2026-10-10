import argparse
import json
import os
import socket
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "appearance", "out", "cuas_cls.onnx")


class AppearanceClassifier:
    def __init__(self, onnx_path=DEFAULT_MODEL):
        meta = json.load(open(os.path.splitext(onnx_path)[0] + ".json", encoding="utf-8"))
        self.classes = meta["classes"]
        self.size = meta["size"]
        self.mean = np.array(meta["mean"], np.float32)
        self.std = np.array(meta["std"], np.float32)
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

    def probs(self, bgr):
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        x = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        self.net.setInput(x.transpose(2, 0, 1)[None])
        z = self.net.forward()[0].astype(np.float64)
        e = np.exp(z - z.max())
        return dict(zip(self.classes, (e / e.sum()).tolist()))

    def crop_probs(self, frame, cx, cy, target_px, ctx=2.0):
        H, W = frame.shape[:2]
        side = int(max(12, min(target_px * ctx, W, H)))
        x0 = int(min(max(cx - side / 2, 0), W - side))
        y0 = int(min(max(cy - side / 2, 0), H - side))
        return self.probs(frame[y0:y0 + side, x0:x0 + side])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="카메라 번호 / 영상 / tcp://… / 사진 파일")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--roi", default=None, help="x,y,w,h (모니터 화면 부분만)")
    ap.add_argument("--every", type=int, default=3, help="몇 프레임마다 분류")
    ap.add_argument("--smooth", type=int, default=5, help="확률 평균낼 판정 수")
    ap.add_argument("--udp", default=None, help="host:port 로 'cls,conf,image' (none 은 안 보냄)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    clf = AppearanceClassifier(a.model)
    if a.src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        p = clf.probs(cv2.imread(a.src))
        print(", ".join(f"{k} {v:.2f}" for k, v in sorted(p.items(), key=lambda kv: -kv[1])))
        return
    cap = cv2.VideoCapture(int(a.src) if a.src.isdigit() else a.src)
    if not cap.isOpened():
        sys.exit(f"열기 실패: {a.src}")
    sock, dst = None, None
    if a.udp:
        h, pt = a.udp.rsplit(":", 1)
        sock, dst = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (h, int(pt))
    hist = deque(maxlen=a.smooth)
    i, last = 0, None
    t0 = time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        view = f
        if a.roi:
            x, y, w, h = map(int, a.roi.split(","))
            view = f[y:y + h, x:x + w]
        if i % a.every == 0:
            hist.append(clf.probs(view))
            avg = {c: float(np.mean([p[c] for p in hist])) for c in clf.classes}
            cls = max(avg, key=avg.get)
            conf = avg[cls]
            if sock and cls != "none":
                sock.sendto(f"{cls},{conf:.2f},image".encode(), dst)
            if cls != last:
                print(f"[{time.time() - t0:7.2f}s] {cls:8} {conf:.2f}   " +
                      " ".join(f"{k}={v:.2f}" for k, v in avg.items()))
                last = cls
        if a.show:
            vis = f.copy()
            if a.roi:
                x, y, w, h = map(int, a.roi.split(","))
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
            if last:
                cv2.putText(vis, f"{last} {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("appearance_id", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1


if __name__ == "__main__":
    main()
