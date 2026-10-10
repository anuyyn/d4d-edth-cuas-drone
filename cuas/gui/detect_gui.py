import argparse
import glob
import json
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
CUAS = os.path.dirname(HERE)
sys.path.insert(0, CUAS)
OUT = HERE
CLASSES = ["새", "오물풍선", "무인기"]
INFO = {
    "새": ("자연 비행체(새)", "이것은 새입니다. 아무것도 하지 않아도 됩니다.", "조치 없음 — Enter = 상황 종료"),
    "오물풍선": ("오물풍선 (풍선 + 매달린 오물·쓰레기봉투)", "이것은 오물풍선입니다. 드론 GPS를 오물에 부착해주세요.",
             "부착형 추적 드론 출동 — Enter = 가제보 실행"),
    "무인기": ("무인기 (소형 드론)", "이것은 무인기입니다. 드론을 무인기에 요격하세요.", "요격 드론 출동 — Enter = 가제보 실행"),
    "풍선": ("풍선 (오물 안 보임)", "관찰하세요. 오물이 보이면 '오물풍선'으로 정정하세요.", "관찰 — Enter = 관찰 기록"),
    "미확인": ("분류 안 됨", "사람이 직접 확인 필요", "-"),
}
COLORS = {"새": "#4fc3f7", "오물풍선": "#ce93d8", "풍선": "#ffb74d", "무인기": "#ef5350", "미확인": "#bdbdbd"}
EN2KO = {"bird": "새", "trash": "오물풍선", "drone": "무인기", "balloon": "풍선", "unknown": "미확인"}
KO2EN = {v: k for k, v in EN2KO.items()}
MISSION = {"오물풍선": "track", "무인기": "intercept"}
BG, FG, PANEL = "#1e1f24", "#eceff1", "#2b2d33"
FONT = "Malgun Gothic"
TTF = r"C:\Windows\Fonts\malgun.ttf"


def pil_font(size):
    try:
        return ImageFont.truetype(TTF, size)
    except Exception:
        return ImageFont.load_default()


def class_from_name(path):
    s = path.lower()
    for key, cls in (("bird", "새"), ("새", "새"), ("gull", "새"),
                     ("ballon", "오물풍선"), ("balloon", "오물풍선"), ("풍선", "오물풍선"),
                     ("small_dron", "무인기"), ("drone", "무인기"), ("dron", "무인기"), ("uav", "무인기"), ("무인기", "무인기")):
        if key in s:
            return cls
    return "미확인"


def map_yolo_name(name):
    n = str(name).lower()
    if any(k in n for k in ("bird", "새", "gull")):
        return "새"
    if any(k in n for k in ("trash", "오물")):
        return "오물풍선"
    if any(k in n for k in ("balloon", "풍선", "kite")):
        return "풍선"
    if any(k in n for k in ("drone", "uav", "무인기", "airplane", "aeroplane")):
        return "무인기"
    return "미확인"


class LocalContrastDetector:

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = 480.0 / w
        small = cv2.resize(frame, (480, int(h * s)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bg = cv2.GaussianBlur(gray, (0, 0), 22)
        diff = cv2.absdiff(gray, bg)
        thr = max(22.0, float(diff.mean() + 4.0 * diff.std()))
        mask = (diff > thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, area_total = None, small.shape[0] * small.shape[1]
        sh, sw = small.shape[:2]
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 0.0004 * area_total or a > 0.12 * area_total:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + bw >= sw - 2 or y + bh >= sh - 2:
                continue
            patch = diff[y:y + bh, x:x + bw]
            if float(patch.max()) < 40.0:
                continue
            score = float(np.sqrt(a)) * float(patch.mean())
            if best is None or score > best[0]:
                best = (score, x, y, bw, bh)
        if best is None:
            return []
        _, x, y, bw, bh = best
        return [dict(box=(int(x / s), int(y / s), int((x + bw) / s), int((y + bh) / s)), cls=None, conf=None)]


class ClipDetector:

    MIN_CONF = 0.45

    def __init__(self, sync=False):
        from clip_cls import ClipClassifier
        import threading
        self.clf = ClipClassifier()
        self.loc = LocalContrastDetector()
        self.sync = sync
        self.latest = ("기타", 0.0)
        self._frame, self._stop = None, False
        self._lock = threading.Lock()
        self._n = 0
        if not sync:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    def _loop(self):
        while not self._stop:
            with self._lock:
                f, self._frame = self._frame, None
            if f is None:
                time.sleep(0.02)
                continue
            cls, conf, _ = self.clf.classify(f)
            self.latest = (cls, conf)
            time.sleep(0.2)

    def detect(self, frame):
        self._n += 1
        if self.sync:
            if self._n % 4 == 1:
                cls, conf, _ = self.clf.classify(frame)
                self.latest = (cls, conf)
        else:
            with self._lock:
                self._frame = frame
        cls, conf = self.latest
        if cls == "기타":
            return []
        boxes = self.loc.detect(frame)
        box = boxes[0]["box"] if boxes else None
        return [dict(box=box, cls=cls if conf >= self.MIN_CONF else "미확인", conf=conf)]


class CuasDetector:

    def __init__(self, fps=30.0):
        import motion_id as M
        from appearance_id import AppearanceClassifier
        self.M, self.app = M, AppearanceClassifier()
        self.reset(fps)

    def reset(self, fps=None):
        self.fps = fps or getattr(self, "fps", 30.0)
        self.p = self.M.Pipeline(self.fps, appearance=self.app, recon=True)
        self.t = 0.0

    def detect(self, frame):
        h, w = frame.shape[:2]
        k = min(1.0, 640.0 / w)
        small = cv2.resize(frame, (int(w * k), int(h * k))) if k < 1 else frame
        self.t += 1.0 / self.fps
        res = self.p.step(self.t, small)
        out = []
        for tr in sorted(res, key=lambda q: -q.extent_px()):
            x1, y1, x2, y2 = self.p.box(tr)
            cls, conf, why = tr.last
            out.append(dict(box=(int(x1 / k), int(y1 / k), int(x2 / k), int(y2 / k)), cls=EN2KO.get(cls, "미확인"),
                            conf=float(conf), why=why))
        if not out and self.p.scene:
            cls, conf = self.p.scene
            out.append(dict(box=None, cls=EN2KO.get(cls, "미확인"), conf=float(conf), why="화면 전체 판정"))
        return out[:1]


class PiSource:

    def __init__(self, port=9500):
        import socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.port = port
        self.name, self.hint, self.fps = f"정찰드론 실시간 (UDP {port})", "미확인", 10.0
        self.frame_no, self.dets = 0, []
        self.t0, self.last_rx = time.time(), 0.0

    def read(self):
        import base64
        latest = None
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except OSError:
                break
            try:
                d = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if d.get("type") == "frame":
                latest = d
        if latest is None:
            return None, False
        fr = cv2.imdecode(np.frombuffer(base64.b64decode(latest["img"]), np.uint8), cv2.IMREAD_COLOR)
        if fr is None:
            return None, False
        self.dets = [dict(box=tuple(b["box"]) if b.get("box") else None, cls=EN2KO.get(b["cls"], "미확인"),
                          conf=b["conf"])
                     for b in sorted(latest.get("dets", []),
                                     key=lambda b: -((b["box"][2] - b["box"][0]) if b.get("box") else 0))][:1]
        self.frame_no += 1
        self.last_rx = time.time()
        return fr, False

    @property
    def t(self):
        return time.time() - self.t0

    @property
    def alive(self):
        return time.time() - self.last_rx < 2.0


class YoloDetector:
    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(path)

    def detect(self, frame):
        res = self.model.predict(frame, imgsz=640, verbose=False)[0]
        out = []
        for b in res.boxes:
            name = res.names[int(b.cls)]
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            out.append(dict(box=(x1, y1, x2, y2), cls=map_yolo_name(name), conf=float(b.conf)))
        out.sort(key=lambda d: -(d["conf"] or 0))
        return out[:1]


class Source:
    def __init__(self, items, webcam=False):
        self.items = items
        self.idx = -1
        self.cap = None
        self.webcam = webcam
        self.name, self.hint, self.fps = "", "미확인", 30.0
        self.frame_no = 0
        self._next()

    def _next(self):
        if self.cap is not None:
            self.cap.release()
        if self.webcam:
            self.cap = cv2.VideoCapture(0)
            self.name, self.hint = "웹캠", "미확인"
        else:
            self.idx = (self.idx + 1) % len(self.items)
            path, self.hint = self.items[self.idx]
            self.cap = cv2.VideoCapture(path)
            self.name = os.path.basename(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_no = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self._next()
            ok, frame = self.cap.read()
            if not ok:
                return None, True
            self.frame_no += 1
            return frame, True
        self.frame_no += 1
        return frame, False

    @property
    def t(self):
        return self.frame_no / self.fps


def collect(paths):
    items = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "**", "*.mp4"), recursive=True)):
                items.append((f, class_from_name(f)))
        elif os.path.isfile(p):
            items.append((p, class_from_name(p)))
    return items


def default_items():
    cands = [r"C:\dron\bird_2_trim.mp4", r"C:\dron\bird_1_long.mp4", r"C:\dron\ballon_1_long.mp4",
             r"C:\dron\small_dron_1_15s.mp4"]
    return [(p, class_from_name(p)) for p in cands if os.path.isfile(p)]


class EventStore:
    def __init__(self, save_birds):
        os.makedirs(os.path.join(OUT, "snap"), exist_ok=True)
        self.events = []
        self.save_birds = save_birds
        self.path = os.path.join(OUT, "events.json")
        self.txt_path = os.path.join(OUT, "event_log.txt")
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")

    def add(self, cls, conf, src, t, frame, box):
        e = dict(id=len(self.events) + 1, date=time.strftime("%Y-%m-%d"), wall=time.strftime("%H:%M:%S"),
                 t=round(t, 1), source=src,
                 cls=cls, conf=conf, status="미확인", box=list(box) if box else [], snap=None)
        h, w = frame.shape[:2]
        if box:
            x1, y1, x2, y2 = box
            m = int(0.6 * max(x2 - x1, y2 - y1)) + 20
            crop = frame[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
        else:
            crop = frame
        e["_crop"] = crop.copy()
        e["_full"] = frame.copy()
        logged = not (cls == "새" and not self.save_birds)
        if cls == "새" and not self.save_birds:
            e["status"] = "자동 종결(새, 저장 안 함)"
        if logged:
            fn = os.path.join(OUT, "snap", f"event_{e['id']:04d}_{cls}.png")
            cv2.imencode(".png", e["_full"])[1].tofile(fn)
            e["snap"] = fn
        self.events.append(e)
        self.flush()
        return e

    def flush(self):
        logged = [e for e in self.events if e["snap"] is not None]
        rows = [{k: v for k, v in e.items() if not k.startswith("_")} for e in logged]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        lines = [f"탐지 이벤트 로그  (프로그램 시작 {self.started})", "=" * 72]
        for e in logged:
            conf = "-" if e["conf"] is None else f"{e['conf'] * 100:.0f}%"
            lines.append(f"[{e['date']} {e['wall']}] #{e['id']:03d} | 객체: {e['cls']} ({conf}) | "
                         f"소스: {e['source']} {e['t']}초 | 상태: {e['status']}")
            lines.append(f"      사진: {os.path.relpath(e['snap'], OUT)}")
        if not logged:
            lines.append("(아직 기록된 이벤트가 없습니다)")
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


class EventTrigger:

    def __init__(self, need=5, gap=2.5):
        self.need, self.gap = need, gap
        self.run, self.last, self.run_cls = 0, {}, None

    def update(self, found, cls, t):
        if found and cls == self.run_cls:
            self.run += 1
        else:
            self.run, self.run_cls = (1, cls) if found else (0, None)
        if not found or self.run < self.need:
            return False
        if t - self.last.get(cls, -1e9) < self.gap:
            return False
        self.last[cls] = t
        return True


class App:
    def __init__(self, root, source, detector, save_birds, arms_host=None, arms_port=9510):
        self.root, self.src, self.det = root, source, detector
        import socket
        self.arms = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.arms.setblocking(False)
        self.arms_dst = (arms_host, arms_port) if arms_host else None
        self.arms_pong = 0.0
        self.store = EventStore(save_birds)
        self.trig = EventTrigger()
        self.paused = False
        self.selected = None
        self.save_birds_var = tk.BooleanVar(value=save_birds)
        self.last_t = time.time()
        self.photo_refs = {}
        self.demo = False
        self.follow_live = True
        self.live_t, self.live_draw_t, self.live_on = 0.0, 0.0, False
        self._build()
        self.root.after(10, self.tick)

    def _build(self):
        r = self.root
        r.title("탐지 상황 GUI")
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{min(1260, sw - 20)}x{min(800, sh - 70)}+0+0")
        r.configure(bg=BG)
        top = tk.Frame(r, bg=PANEL)
        top.pack(fill="x")
        for text, cmd in (("영상 열기", self.open_file), ("폴더 열기", self.open_dir), ("웹캠", self.open_cam),
                          ("일시정지/재생", self.toggle_pause)):
            tk.Button(top, text=text, command=cmd, font=(FONT, 10), bg="#3a3d45", fg=FG, relief="flat",
                      padx=10).pack(side="left", padx=4, pady=6)
        tk.Checkbutton(top, text="새도 로그에 저장", variable=self.save_birds_var, command=self._birds_toggle,
                       font=(FONT, 10), bg=PANEL, fg=FG, selectcolor="#3a3d45", activebackground=PANEL,
                       activeforeground=FG).pack(side="left", padx=12)
        mode = ("YOLO 모델" if isinstance(self.det, YoloDetector) else
                "우리 탐지기(검출·추적·외형 ONNX·오물봉투 확인)" if isinstance(self.det, CuasDetector) else
                "CLIP 제로샷(화면을 보고 분류, 파일 이름 안 씀)")
        mj = os.path.join(CUAS, "appearance", "out", "cuas_cls.onnx")
        mtime = time.strftime("%m/%d %H:%M", time.localtime(os.path.getmtime(mj))) if os.path.exists(mj) else "?"
        if isinstance(self.src, PiSource):
            mode = f"정찰드론 실시간 · 드론에서 탐지 · 학습 모델({mtime})"
        elif isinstance(self.det, CuasDetector):
            mode = f"학습 모델 cuas_cls.onnx({mtime} 학습) + 오물봉투 확인"
        self.arms_lbl = tk.Label(top, font=(FONT, 10), bg=PANEL, fg="#aaa", text="가제보: -")
        self.arms_lbl.pack(side="right", padx=8)
        self.mode_lbl = tk.Label(top, font=(FONT, 10, "bold"), bg=PANEL, fg="#81c784", text=f"분류 방식: {mode}")
        self.mode_lbl.pack(side="right", padx=12)

        main = tk.Frame(r, bg=BG)
        main.pack(fill="both", expand=True)
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", padx=8, pady=8)
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        tk.Label(left, text="송출 화면", font=(FONT, 12, "bold"), bg=BG, fg=FG).pack(anchor="w")
        self.video = tk.Canvas(left, width=640, height=360, bg="black", highlightthickness=0)
        self.video.pack()
        self.src_lbl = tk.Label(left, text="", font=(FONT, 10), bg=BG, fg="#90a4ae", anchor="w")
        self.src_lbl.pack(fill="x")

        hdr = tk.Frame(left, bg=BG)
        hdr.pack(fill="x", pady=(10, 0))
        tk.Label(hdr, text="객체 정보 (실시간 캡처)", font=(FONT, 12, "bold"), bg=BG, fg=FG).pack(side="left")
        self.live_lbl = tk.Label(hdr, text="", font=(FONT, 11, "bold"), bg=BG, fg="#ff5252")
        self.live_lbl.pack(side="left", padx=8)
        tk.Button(hdr, text="실시간 보기", command=self.go_live, font=(FONT, 9), bg="#3a3d45", fg=FG, relief="flat",
                  padx=8).pack(side="right")
        info = tk.Frame(left, bg=PANEL)
        info.pack(fill="x")
        self.big = tk.Canvas(info, width=250, height=160, bg="black", highlightthickness=0)
        self.big.pack(side="left", padx=6, pady=6)
        txt = tk.Frame(info, bg=PANEL)
        txt.pack(side="left", fill="both", expand=True, padx=6)
        self.cls_lbl = tk.Label(txt, text="-", font=(FONT, 20, "bold"), bg=PANEL, fg=FG, anchor="w")
        self.cls_lbl.pack(fill="x", pady=(2, 0))
        self.desc_lbl = tk.Label(txt, text="탐지를 기다리는 중", font=(FONT, 14, "bold"), bg=PANEL,
                                 fg=FG, anchor="w", justify="left", wraplength=340)
        self.desc_lbl.pack(fill="x", pady=2)
        self.resp_lbl = tk.Label(txt, text="", font=(FONT, 10, "bold"), bg=PANEL, fg="#ffd54f", anchor="w",
                                 justify="left", wraplength=340)
        self.resp_lbl.pack(fill="x")
        self.meta_lbl = tk.Label(txt, text="", font=(FONT, 9), bg=PANEL, fg="#90a4ae", anchor="w", justify="left",
                                 wraplength=340)
        self.meta_lbl.pack(fill="x", pady=2)

        tk.Label(right, text="이벤트로 찍힌 객체 (최근 4컷, 사진을 누르면 선택)", font=(FONT, 12, "bold"),
                 bg=BG, fg=FG).pack(anchor="w")
        self.slots = []
        grid = tk.Frame(right, bg=BG)
        grid.pack(fill="x")
        for i in range(4):
            c = tk.Canvas(grid, width=250, height=158, bg="#15161a", highlightthickness=1,
                          highlightbackground="#444")
            c.grid(row=i // 2, column=i % 2, padx=4, pady=4)
            self.slots.append(c)
            self._draw_empty(c, i)

        tk.Label(right, text="이벤트 로그", font=(FONT, 12, "bold"), bg=BG, fg=FG).pack(anchor="w", pady=(8, 0))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=24,
                        font=(FONT, 10))
        style.configure("Treeview.Heading", font=(FONT, 10, "bold"))
        cols = ("id", "시각", "영상시간", "소스", "객체", "신뢰도", "상태")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=7)
        for c, w in zip(cols, (36, 70, 64, 150, 56, 56, 140)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree)
        btns = tk.Frame(right, bg=BG)
        btns.pack(fill="x", pady=6)
        tk.Label(btns, text="사람 검증:", font=(FONT, 10, "bold"), bg=BG, fg=FG).pack(side="left")
        tk.Button(btns, text="맞음 ✔", command=lambda: self.verify(None), font=(FONT, 10), bg="#2e7d32", fg="white",
                  relief="flat", padx=10).pack(side="left", padx=4)
        tk.Label(btns, text="정정(직접 입력):", font=(FONT, 10), bg=BG, fg=FG).pack(side="left", padx=(10, 2))
        self.fix_var = tk.StringVar()
        self.fix_entry = tk.Entry(btns, textvariable=self.fix_var, font=(FONT, 11), width=16, bg="#3a3d45", fg=FG,
                                  insertbackground=FG, relief="flat")
        self.fix_entry.pack(side="left", padx=2, ipady=3)
        self.fix_entry.bind("<Return>", lambda ev: self.apply_fix())
        tk.Button(btns, text="정정 적용", command=self.apply_fix, font=(FONT, 10), bg="#3a3d45", fg=FG,
                  relief="flat", padx=8).pack(side="left", padx=4)
        act = tk.Frame(right, bg=BG)
        act.pack(fill="x", pady=(0, 6))
        self.act_btn = tk.Button(act, text="대응 실행 (Enter)", command=self.act, font=(FONT, 12, "bold"),
                                 bg="#c62828", fg="white", relief="flat", padx=14, pady=4)
        self.act_btn.pack(side="left")
        tk.Label(act, text="정정 버튼:", font=(FONT, 10), bg=BG, fg=FG).pack(side="left", padx=(12, 2))
        for name in ("새", "오물풍선", "무인기"):
            tk.Button(act, text=name, command=lambda n=name: self.verify(n), font=(FONT, 10), bg="#3a3d45", fg=FG,
                      relief="flat", padx=8).pack(side="left", padx=2)
        self.act_lbl = tk.Label(act, text="", font=(FONT, 10, "bold"), bg=BG, fg="#ffd54f")
        self.act_lbl.pack(side="left", padx=10)
        self.root.bind("<Return>", lambda ev: None if ev.widget is self.fix_entry else self.act())
        self.root.after(1000, self._ping)

    def act(self):
        e = self.selected
        if e is None:
            return
        cls = e["cls"]
        if cls in MISSION:
            if self.arms_dst is None:
                e["status"] = "출동 명령 (가제보 미연결)"
            else:
                self.arms.sendto(f"{MISSION[cls]},{KO2EN[cls]},{e['id']}".encode(), self.arms_dst)
                e["status"] = "부착형 추적 드론 출동" if cls == "오물풍선" else "요격 드론 출동"
                self.active = e
        elif cls == "새":
            e["status"] = "상황 종료 (조치 없음)"
        elif cls == "풍선":
            e["status"] = "관찰"
        else:
            return
        self.act_lbl.config(text=f"#{e['id']} {e['status']}")
        self._update_row(e)

    def _update_row(self, e):
        if str(e["id"]) in self.tree.get_children():
            vals = list(self.tree.item(str(e["id"]), "values"))
            vals[4], vals[6] = e["cls"], e["status"]
            self.tree.item(str(e["id"]), values=vals)
        self.store.flush()
        self._refresh_thumbs()
        if e is self.selected:
            self.select(e)

    def _poll_arms(self):
        KO = {"LAUNCH": "발진", "SEARCH": "탐색", "TRACK": "추적", "TERMINAL": "돌입", "HIT": "요격 성공",
              "MISS": "실패(다시 Enter)", "FOLLOW": "GPS 부착·추적 중", "KILLED": "중지"}
        while True:
            try:
                data, _ = self.arms.recvfrom(256)
            except OSError:
                break
            parts = data.decode("utf-8", "ignore").split(",")
            if parts[0] == "pong":
                self.arms_pong = time.time()
            elif parts[0] == "state" and len(parts) >= 3 and parts[1] in KO:
                e = next((x for x in self.store.events if str(x["id"]) == parts[2]), None)
                if e is not None:
                    e["status"] = KO[parts[1]]
                    self.act_lbl.config(text=f"#{e['id']} {e['status']}")
                    self._update_row(e)

    def _ping(self):
        if self.arms_dst:
            try:
                self.arms.sendto(b"ping", self.arms_dst)
            except OSError:
                pass
            ok = time.time() - self.arms_pong < 4
            self.arms_lbl.config(text=f"가제보 {self.arms_dst[0]} " + ("● 연결됨" if ok else "○ 응답 없음"),
                                 fg="#81c784" if ok else "#ff8a65")
        self.root.after(2000, self._ping)

    def _birds_toggle(self):
        self.store.save_birds = self.save_birds_var.get()

    def toggle_pause(self):
        self.paused = not self.paused

    def _set_source(self, items, webcam=False):
        if items or webcam:
            self.src = Source(items, webcam)
            self.trig = EventTrigger()
            if hasattr(self.det, "reset"):
                self.det.reset(self.src.fps)

    def open_file(self):
        p = filedialog.askopenfilenames(title="영상 선택", filetypes=[("영상", "*.mp4 *.avi *.mov *.mkv")])
        self._set_source(collect(list(p)))

    def open_dir(self):
        d = filedialog.askdirectory(title="영상 폴더 선택")
        if d:
            self._set_source(collect([d]))

    def open_cam(self):
        self._set_source([], webcam=True)

    def tick(self):
        self._poll_arms()
        now = time.time()
        if self.paused or self.src is None:
            self.root.after(30, self.tick)
            return
        due = int((now - self.last_t) * self.src.fps)
        if due < 1:
            self.root.after(5, self.tick)
            return
        self.last_t = now
        frame, switched = self.src.read()
        if frame is None:
            if isinstance(self.src, PiSource) and not self.src.alive:
                self.src_lbl.config(text=f"소스: {self.src.name}   ○ 신호 없음 — 드론(파이)·핫스팟 확인")
            self.root.after(30, self.tick)
            return
        if switched:
            self.trig = EventTrigger()
            if hasattr(self.det, "reset"):
                self.det.reset(self.src.fps)
        dets = self.src.dets if isinstance(self.src, PiSource) else self.det.detect(frame)
        box, cls, conf = None, None, None
        if dets:
            d = dets[0]
            box, cls, conf = d["box"], d["cls"], d["conf"]
        found = cls is not None
        if self.trig.update(found, cls, self.src.t):
            e = self.store.add(cls, conf, self.src.name, self.src.t, frame, box)
            self._after_event(e)
        self._show_video(frame, box, cls, conf)
        self._update_live(frame, dets[0] if dets else None)
        self.root.after(1, self.tick)

    def _fit(self, img, w, h):
        ih, iw = img.shape[:2]
        k = min(w / iw, h / ih)
        return cv2.resize(img, (max(1, int(iw * k)), max(1, int(ih * k)))), k

    def _show_video(self, frame, box, cls, conf):
        img, k = self._fit(frame, 640, 360)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        if cls is not None:
            col = COLORS.get(cls, "#fff")
            label = cls + (f" {conf * 100:.0f}%" if conf is not None else "")
            f = pil_font(18)
            tw = dr.textlength(label, font=f)
            if box is not None:
                x1, y1, x2, y2 = [int(v * k) for v in box]
                dr.rectangle((x1, y1, x2, y2), outline=col, width=3)
            else:
                x1, y1 = 8, 34
            ty = max(0, y1 - 26)
            dr.rectangle((x1, ty, x1 + tw + 10, ty + 26), fill=col)
            dr.text((x1 + 5, ty + 1), label, font=f, fill="black")
        ph = ImageTk.PhotoImage(pil)
        self.photo_refs["video"] = ph
        self.video.delete("all")
        self.video.create_image(320, 180, image=ph)
        self.src_lbl.config(text=f"소스: {self.src.name}   영상 시간 {self.src.t:5.1f}초   "
                                 f"이벤트 {len(self.store.events)}건")

    def _crop(self, frame, box):
        h, w = frame.shape[:2]
        if not box:
            return frame
        x1, y1, x2, y2 = box
        m = int(0.6 * max(x2 - x1, y2 - y1)) + 20
        return frame[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]

    def _show_info(self, img, cls, conf, meta, live):
        fit, _ = self._fit(img, 246, 156)
        ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fit, cv2.COLOR_BGR2RGB)))
        self.photo_refs["big"] = ph
        self.big.delete("all")
        self.big.create_image(125, 80, image=ph)
        if live:
            self.big.create_rectangle(0, 0, 70, 20, fill="#d32f2f", outline="")
            self.big.create_text(35, 10, text="● LIVE", fill="white", font=(FONT, 9, "bold"))
        name, desc, resp = INFO.get(cls, (f"사람이 입력한 이름: {cls}", "사람이 직접 정정한 객체입니다.", "-"))
        conf_t = "" if conf is None else f"  {conf * 100:.0f}%"
        self.cls_lbl.config(text=f"{cls}{conf_t}", fg=COLORS.get(cls, FG))
        self.desc_lbl.config(text=desc, fg=COLORS.get(cls, FG) if cls in ("무인기", "오물풍선") else FG)
        self.resp_lbl.config(text=f"대응: {resp}")
        self.meta_lbl.config(text=f"{name}  |  {meta}")

    def _update_live(self, frame, d):
        now = time.time()
        if d is not None and d.get("cls") not in (None, "미확인"):
            self.live_t = now
            if self.follow_live and now - self.live_draw_t >= 0.1:
                self.live_draw_t = now
                self.live_on = True
                self.live_lbl.config(text="● 실시간")
                where = "화면 전체 판정" if not d.get("box") else "탐지 위치 확대"
                self._show_info(self._crop(frame, d.get("box")), d["cls"], d.get("conf"),
                                f"실시간 · {where} · {self.src.name}", live=True)
        elif self.live_on and now - self.live_t > 1.0:
            self.live_on = False
            self.live_lbl.config(text="")
            if self.selected is not None and self.follow_live:
                self.select(self.selected)

    def go_live(self):
        self.follow_live = True
        self.live_draw_t = 0.0

    def pick(self, e):
        self.follow_live = False
        self.live_lbl.config(text="(지난 이벤트 보는 중 — '실시간 보기')")
        self.select(e)

    def _after_event(self, e):
        self.follow_live = True
        self._refresh_thumbs()
        conf = "-" if e["conf"] is None else f"{e['conf'] * 100:.0f}%"
        self.tree.insert("", 0, iid=str(e["id"]),
                         values=(e["id"], e["wall"], f"{e['t']}초", e["source"], e["cls"], conf, e["status"]))
        self.select(e)

    def _draw_empty(self, c, i):
        c.delete("all")
        c.create_text(125, 79, text=f"빈칸 {i + 1}\n(이벤트 대기)", fill="#5f6672", font=(FONT, 11), justify="center")

    def _refresh_thumbs(self):
        evs = self.store.events[-4:][::-1]
        for i, c in enumerate(self.slots):
            if i >= len(evs):
                self._draw_empty(c, i)
                c.unbind("<Button-1>")
                continue
            e = evs[i]
            c.delete("all")
            img, _ = self._fit(e["_crop"], 244, 118)
            ph = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self.photo_refs[("slot", i)] = ph
            col = COLORS.get(e["cls"], "#cfd8dc")
            c.create_rectangle(0, 0, 250, 26, fill=col, outline=col)
            conf = "" if e["conf"] is None else f"  {e['conf'] * 100:.0f}%"
            c.create_text(8, 13, anchor="w", text=f"{e['cls']}{conf}", fill="black", font=(FONT, 12, "bold"))
            c.create_image(125, 28 + 59, image=ph)
            c.create_text(6, 148, anchor="w", fill="#cfd8dc", font=(FONT, 9),
                          text=f"#{e['id']}  {e['t']}초  {e['status'][:12]}")
            c.bind("<Button-1>", lambda ev, e=e: self.pick(e))

    def select(self, e):
        self.selected = e
        if not (self.follow_live and self.live_on):
            self._show_info(e["_crop"], e["cls"], e["conf"],
                            f"이벤트 #{e['id']} · {e['source']} {e['t']}초 · 상태: {e['status']}", live=False)
        if str(e["id"]) in self.tree.get_children() and self.tree.selection() != (str(e["id"]),):
            self.tree.selection_set(str(e["id"]))

    def _on_tree(self, _):
        sel = self.tree.selection()
        if sel:
            e = next((x for x in self.store.events if str(x["id"]) == sel[0]), None)
            if e and e is not self.selected:
                self.pick(e)

    def apply_fix(self):
        name = self.fix_var.get().strip()
        if not name:
            return
        self.verify(name)
        self.fix_var.set("")

    def verify(self, cls):
        e = self.selected
        if e is None:
            return
        if cls is None:
            e["status"] = "확인(맞음)"
        else:
            e["status"] = f"정정: {e['cls']}→{cls}"
            e["cls"] = cls
            if e["snap"] is None and cls != "새":
                fn = os.path.join(OUT, "snap", f"event_{e['id']:04d}_{cls}.png")
                cv2.imencode(".png", e["_full"])[1].tofile(fn)
                e["snap"] = fn
        if str(e["id"]) in self.tree.get_children():
            vals = list(self.tree.item(str(e["id"]), "values"))
            vals[4], vals[6] = e["cls"], e["status"]
            self.tree.item(str(e["id"]), values=vals)
        self.store.flush()
        self._refresh_thumbs()
        self.select(e)


def grab_own_window(root, path):
    import ctypes
    from ctypes import wintypes
    u, g = ctypes.windll.user32, ctypes.windll.gdi32
    root.update_idletasks()
    hwnd = u.GetParent(root.winfo_id()) or root.winfo_id()
    rc = wintypes.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(rc))
    w, h = rc.right - rc.left, rc.bottom - rc.top
    hdc = u.GetWindowDC(hwnd)
    mdc = g.CreateCompatibleDC(hdc)
    bmp = g.CreateCompatibleBitmap(hdc, w, h)
    g.SelectObject(mdc, bmp)
    u.PrintWindow(hwnd, mdc, 2)

    class BIH(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]
    bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    g.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")
    img.save(path)
    g.DeleteObject(bmp)
    g.DeleteDC(mdc)
    u.ReleaseDC(hwnd, hdc)


def selftest(items, det):
    store = EventStore(save_birds=False)
    for path, hint in items:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        trig, n, found_n, ev, counts = EventTrigger(), 0, 0, 0, {}
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            d = det.detect(fr)
            cls = d[0]["cls"] if d else None
            found_n += 1 if d else 0
            if trig.update(bool(d), cls, n / fps):
                store.add(cls, d[0]["conf"], os.path.basename(path), n / fps, fr, d[0]["box"])
                ev += 1
            if d:
                counts[cls] = counts.get(cls, 0) + 1
        print(f"{os.path.basename(path):<28} 프레임 {n:5d}  분류됨 {found_n:5d}  이벤트 {ev:3d}건  "
              f"종류별 프레임: {counts}   (채점용 정답: {hint})")
    print("이벤트 기록:", store.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--yolo")
    ap.add_argument("--pi", action="store_true", help="정찰드론(파이) 실시간 수신 (인자 없으면 기본)")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--clip", action="store_true", help="팀 원래 방식(CLIP 제로샷, torch 필요)")
    ap.add_argument("--arms-host", default="auto", help="가제보 A.R.M.S. 주소 (auto = WSL IP, none = 끔)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save-birds", action="store_true", help="새도 로그/사진으로 저장")
    ap.add_argument("--shot", help="실행 후 --shot-after 초 뒤 이 창만 PNG로 저장하고 종료(확인용)")
    ap.add_argument("--shot-after", type=float, default=25.0)
    a = ap.parse_args()

    use_pi = a.pi or (not a.paths and not a.webcam and not a.selftest)
    items = collect(a.paths) if a.paths else ([] if use_pi else default_items())
    if not items and not a.webcam and not use_pi:
        sys.exit("재생할 영상을 찾지 못했습니다. 영상 경로를 인자로 주세요.")
    if a.yolo:
        det = YoloDetector(a.yolo)
    elif a.clip:
        try:
            det = ClipDetector(sync=a.selftest)
        except Exception as ex:
            sys.exit(f"분류 모델을 불러오지 못했습니다: {ex}\n"
                     "  pip install open_clip_torch  로 설치하거나, --yolo 학습모델.pt 를 지정하세요.")
    else:
        det = CuasDetector()
    arms_host = None
    if a.arms_host == "auto":
        try:
            from ground_station import wsl_ip
            arms_host = wsl_ip()
        except Exception:
            arms_host = None
    elif a.arms_host != "none":
        arms_host = a.arms_host
    if a.selftest:
        selftest(items, det)
        return
    root = tk.Tk()
    src = PiSource(a.port) if use_pi else Source(items, a.webcam)
    app = App(root, src, det, a.save_birds, arms_host)
    if a.shot:
        t0 = time.time()

        def _hb():
            print(f"[hb] {time.time() - t0:5.1f}s 프레임 {app.src.frame_no} 이벤트 {len(app.store.events)}",
                  file=sys.stderr, flush=True)
            root.after(5000, _hb)
        root.after(5000, _hb)

        def _snap():
            try:
                grab_own_window(root, a.shot)
            except Exception as ex:
                print("창 캡처 실패:", ex, file=sys.stderr)
            root.destroy()
        root.after(int(a.shot_after * 1000), _snap)
    root.mainloop()


if __name__ == "__main__":
    main()
