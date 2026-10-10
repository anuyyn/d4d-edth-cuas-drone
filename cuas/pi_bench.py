import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
cv2.setNumThreads(4)


def bench(name, fn, n=20):
    fn()
    t = time.time()
    for _ in range(n):
        fn()
    ms = (time.time() - t) / n * 1000
    print(f"{name:34} {ms:7.1f} ms  ({1000 / ms:5.1f} 회/초)", flush=True)
    return ms


def main():
    print("OpenCV", cv2.__version__, "| 코어", os.cpu_count())
    import motion_id as M
    from appearance_id import AppearanceClassifier
    frame = (np.random.rand(240, 320, 3) * 30 + 150).astype(np.uint8)
    cv2.circle(frame, (160, 120), 12, (60, 60, 60), -1)
    bench("검출 motion_id.detect 320x240", lambda: M.detect(frame))
    app = AppearanceClassifier(os.path.join(HERE, "appearance", "out", "cuas_cls.onnx"))
    crop = frame[80:160, 120:200]
    bench("우리 분류기 MobileNetV3-S 128px", lambda: app.probs(crop))
    yp = os.path.join(HERE, "appearance", "out", "yolov8n_320_bench.onnx")
    if os.path.exists(yp):
        net = cv2.dnn.readNetFromONNX(yp)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (320, 320)), 1 / 255.0, (320, 320), swapRB=True)

        def yolo():
            net.setInput(blob)
            net.forward()
        bench("YOLOv8n 320px (속도 비교용)", yolo, n=10)


if __name__ == "__main__":
    main()
