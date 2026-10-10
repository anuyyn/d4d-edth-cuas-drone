import argparse
import collections
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motion_id as M
from appearance_id import AppearanceClassifier

D = r"C:\Users\user\OneDrive\바탕 화면"
VIDEOS = [
    ("오물풍선", "trash", os.path.join(D, "trash_01.mp4"), 0, 99),
    ("새-갈매기(하늘)", "bird", os.path.join(D, "bird_01.mp4"), 0, 99),
    ("새-왜가리(풀숲)", "bird", os.path.join(D, "bird_02.mp4"), 0, 99),
    ("무인기-고정익(숲)", "drone", os.path.join(D, "drone_01.mp4"), 0, 99),
    ("표적없음-전깃줄", "none", r"C:\Users\user\Videos\화면 녹화\화면 녹화 중 2026-10-10 220115.mp4", 3.3, 99),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "appearance", "out", "cuas_cls.onnx"))
    ap.add_argument("--no-scene", action="store_true", help="화면 전체 판정 끄기(비교용)")
    a = ap.parse_args()
    app = AppearanceClassifier(a.model)
    for name, truth, v, s0, s1 in VIDEOS:
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        p = M.Pipeline(fps, appearance=app, recon=True, scene_every=10 ** 9 if a.no_scene else 3)
        c = collections.Counter()
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if not (s0 <= t < s1):
                continue
            h, w = f.shape[:2]
            if "화면 녹화" in v:
                f = f[:600]
                h = 600
            f = cv2.resize(f, (320, int(h * 320 / w)))
            res = p.step(t, f)
            if res:
                c[max(res, key=lambda q: q.extent_px()).last[0]] += 1
            elif p.scene:
                c["(화면)" + p.scene[0]] += 1
            else:
                c["없음"] += 1
        n = sum(c.values())
        right = c[truth] + c["(화면)" + truth]
        judged = n - c["없음"]
        if truth == "none":
            print(f"{name:14} 헛경보 {judged}/{n} ({judged / n:.0%})  {dict(c.most_common(5))}")
        else:
            print(f"{name:14} 정답 {right}/{judged} ({right / max(judged, 1):.0%})  판정없음 {c['없음'] / n:.0%}  "
                  f"{dict(c.most_common(5))}")


if __name__ == "__main__":
    main()
