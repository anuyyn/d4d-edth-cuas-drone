import os
import sys
from collections import Counter

import cv2

from clip_cls import ClipClassifier
from detect_gui import LocalContrastDetector, class_from_name, default_items


def crop_with_margin(frame, box, k=0.8):
    x1, y1, x2, y2 = box
    m = int(k * max(x2 - x1, y2 - y1)) + 16
    h, w = frame.shape[:2]
    return frame[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]


def main():
    paths = sys.argv[1:]
    items = [(p, class_from_name(p)) for p in paths] if paths else default_items()
    clf = ClipClassifier()
    det = LocalContrastDetector()
    total = {"crop": [0, 0], "frame": [0, 0]}
    for path, truth in items:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(fps * 0.5), 1)
        res = {"crop": Counter(), "frame": Counter()}
        n = 0
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            i += 1
            if i % step:
                continue
            d = det.detect(fr)
            res["frame"][clf.classify(fr)[0]] += 1
            if d:
                res["crop"][clf.classify(crop_with_margin(fr, d[0]["box"]))[0]] += 1
            n += 1
        print(f"\n{os.path.basename(path)}  (정답: {truth}, 표본 {n}장)")
        for mode, c in res.items():
            tot = sum(c.values())
            ok_n = c.get(truth, 0)
            total[mode][0] += ok_n
            total[mode][1] += tot
            dist = ", ".join(f"{k} {v}" for k, v in c.most_common())
            print(f"  {'잘라낸 물체' if mode == 'crop' else '화면 전체'}: 정답률 {ok_n}/{tot} "
                  f"({100.0 * ok_n / max(tot, 1):.0f}%)   분포: {dist}")
    print("\n=== 전체 ===")
    for mode, (a, b) in total.items():
        print(f"  {'잘라낸 물체' if mode == 'crop' else '화면 전체'}: {a}/{b} ({100.0 * a / max(b, 1):.0f}%)")


if __name__ == "__main__":
    main()
