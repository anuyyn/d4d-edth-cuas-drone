import sys
import time

import os

import cv2
import numpy as np

cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30
t0, i = time.time(), 0
out = sys.stdout.buffer
while True:
    ok, f = cap.read()
    if not ok:
        break
    sc = float(os.environ.get("FAKE_SCALE", "1"))
    if sc < 1:
        canvas = np.full((480, 640, 3), 90, np.uint8)
        w, h = int(640 * sc), int(480 * sc)
        canvas[(480 - h) // 2:(480 - h) // 2 + h, (640 - w) // 2:(640 - w) // 2 + w] = cv2.resize(f, (w, h))
        f = canvas
    else:
        f = cv2.resize(f, (640, 480))
    out.write(cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).tobytes())
    out.flush()
    i += 1
    time.sleep(max(0.0, i / fps - (time.time() - t0)))
