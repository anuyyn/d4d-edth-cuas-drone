import argparse
import csv
import json
import os
import random
import socket
import subprocess
import time

CLASSES = {"bird": "새", "balloon": "풍선", "drone": "무인기", "trash": "오물"}
CLS_COLOR = {"bird": "#2e7d32", "balloon": "#1565c0", "drone": "#c62828", "trash": "#8e24aa"}
STATE_KO = {"LAUNCH": "발진", "SEARCH": "탐색", "TRACK": "추적", "TERMINAL": "돌입",
            "HIT": "요격 성공", "MISS": "미명중", "KILLED": "중지", "ARMED": "대기",
            "FOLLOW": "부착·추적 중(GPS)"}
GUIDE = {
    "bird": ("새입니다", "아무 조치도 필요 없습니다.", None),
    "trash": ("오물풍선입니다", "드론으로 GPS를 오물에 부착하세요.  (Enter = 추적 드론 발진)", "track"),
    "drone": ("무인기입니다", "드론으로 무인기를 요격하세요.  (Enter = 요격 드론 발진)", "intercept"),
    "balloon": ("풍선입니다 (오물 안 보임)", "관찰만 하세요. 오물이 보이면 4 로 정정하세요.", None),
}
MERGE_S = 10.0


def parse_packet(data):
    try:
        s = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not s:
        return None
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            return None
        cls, conf, src = d.get("cls"), d.get("conf", 0.0), d.get("src", "recon")
        img = d.get("img")
    else:
        img = None
        parts = [p.strip() for p in s.split(",")]
        cls = parts[0]
        conf = parts[1] if len(parts) > 1 else 0.0
        src = parts[2] if len(parts) > 2 else "recon"
    cls = str(cls).lower()
    if cls not in CLASSES:
        return None
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    live = s.startswith("{") and d.get("type") == "live"
    return {"cls": cls, "conf": conf, "src": str(src), "img": img if isinstance(img, str) else None, "live": live}


class EventLog:

    def __init__(self):
        self.rows = []

    def add(self, cls, conf, src, now, img=None):
        if self.rows:
            r = self.rows[-1]
            if r["cls"] == cls and r["src"] == src and r["op"] is None and now - r["t1"] < MERGE_S:
                r["t1"], r["n"], r["conf"] = now, r["n"] + 1, max(r["conf"], conf)
                if img:
                    r["img"] = img
                return r, False
        r = dict(id=len(self.rows) + 1, t0=now, t1=now, src=src, cls=cls, conf=conf, n=1,
                 op=None, status="확인 대기", img=img)
        self.rows.append(r)
        return r, True

    def decide(self, r, op):
        if r["op"] is not None:
            return False
        r["op"] = op
        if op == "bird":
            r["status"] = "상황 종료"
        elif op == "balloon":
            r["status"] = "관찰"
        elif op == "trash":
            r["status"] = "추적 드론 명령"
            return True
        elif op == "drone":
            r["status"] = "요격 명령"
            return True
        else:
            r["status"] = "오탐 기록"
        return False

    def stats(self):
        done = [r for r in self.rows if r["op"] is not None]
        agree = sum(r["op"] == r["cls"] for r in done)
        by = {k: sum(r["cls"] == k for r in self.rows) for k in CLASSES}
        txt = f"탐지 {len(self.rows)}건  오물 {by['trash']} · 무인기 {by['drone']} · 풍선 {by['balloon']} · 새 {by['bird']}"
        if done:
            txt += f"\n관리자 확인 {len(done)}건 중 정찰드론 판정 일치 {agree}건, 정정 {len(done) - agree}건"
        return txt


def wsl_ip():
    try:
        out = subprocess.run(["wsl.exe", "-d", "Ubuntu-22.04", "--", "hostname", "-I"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return out[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def run_gui(port, arms_host, arms_port, demo, logdir, geometry):
    import tkinter as tk
    from tkinter import ttk

    F = "Malgun Gothic"
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, time.strftime("events_%Y%m%d_%H%M%S.csv"))
    log_f = open(log_path, "w", newline="", encoding="utf-8-sig")
    log_w = csv.writer(log_f)
    log_w.writerow(["시각", "id", "출처", "정찰드론판정", "신뢰도", "관리자판정", "상태"])

    def write_log(r):
        log_w.writerow([time.strftime("%H:%M:%S"), r["id"], r["src"], r["cls"], f"{r['conf']:.3f}",
                        r["op"] or "", r["status"]])
        log_f.flush()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", port))
    rx.setblocking(False)
    arms = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    arms.setblocking(False)
    arms_dst = (arms_host, arms_port) if arms_host else None
    link = {"last_pong": 0.0, "active": None}

    log = EventLog()
    root = tk.Tk()
    root.title("관제 로그 — 정찰드론 탐지")
    root.geometry(geometry)
    root.configure(bg="#1d2230")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", font=(F, 11), rowheight=30, background="#262c3d",
                    fieldbackground="#262c3d", foreground="#e8e8e8")
    style.configure("Treeview.Heading", font=(F, 10, "bold"), background="#323a50", foreground="#c8ccd8")
    style.map("Treeview", background=[("selected", "#4a5a8a")])

    tk.Label(root, text="정찰드론 실시간 탐지 로그", font=(F, 15, "bold"), fg="white",
             bg="#1d2230").pack(anchor="w", padx=12, pady=(10, 0))
    link_var = tk.StringVar()
    link_lbl = tk.Label(root, textvariable=link_var, font=(F, 10), fg="#aaa", bg="#1d2230")
    link_lbl.pack(anchor="w", padx=12)

    stats_var = tk.StringVar()
    tk.Label(root, textvariable=stats_var, font=(F, 10), fg="#e8e8e8", bg="#1d2230",
             justify="left").pack(side="bottom", anchor="w", padx=12, pady=(2, 10))
    tk.Label(root, text="Enter 실행(안내대로)  ·  1 새 / 2 풍선 / 3 무인기 / 4 오물로 정정  ·  Del 오탐",
             font=(F, 10), fg="#c8ccd8", bg="#1d2230").pack(side="bottom", anchor="w", padx=12, pady=(6, 0))
    banner = tk.Label(root, text="대기 중", font=(F, 17, "bold"), fg="white", bg="#444", pady=10)
    banner.pack(side="bottom", fill="x", padx=10)

    det = tk.Frame(root, bg="#262c3d", height=230)
    det.pack(side="bottom", fill="x", padx=10, pady=(6, 4))
    det.pack_propagate(False)
    img_lbl = tk.Label(det, text="캡처 없음", font=(F, 10), fg="#888", bg="#11141c", width=200, height=200)
    img_lbl.pack(side="left", padx=10, pady=12)
    gtitle = tk.Label(det, text="", font=(F, 20, "bold"), fg="white", bg="#262c3d", anchor="w", justify="left")
    gtitle.pack(anchor="w", padx=6, pady=(28, 4))
    gtext = tk.Label(det, text="탐지를 기다리는 중", font=(F, 13), fg="#e8e8e8", bg="#262c3d", anchor="w",
                     justify="left", wraplength=360)
    gtext.pack(anchor="w", padx=6)
    gmeta = tk.Label(det, text="", font=(F, 10), fg="#aab", bg="#262c3d", anchor="w", justify="left")
    gmeta.pack(anchor="w", padx=6, pady=(10, 0))
    photo = {"ref": None}
    live = {"img": None, "t": 0.0, "cls": None, "conf": 0.0}

    def set_image(b64, caption=""):
        try:
            import base64
            import cv2
            import numpy as np
            im = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (200, 200))
            if caption:
                cv2.rectangle(im, (0, 182), (200, 200), (0, 0, 0), -1)
                cv2.putText(im, caption, (6, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)
            ok, png = cv2.imencode(".png", im)
            photo["ref"] = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
            img_lbl.config(image=photo["ref"], text="")
            return True
        except Exception:
            return False

    def live_fresh():
        return live["img"] is not None and time.time() - live["t"] < 1.5

    def show_detail(r):
        if r is None:
            return
        cls = r["op"] if r["op"] in GUIDE else r["cls"]
        title, todo, _ = GUIDE[cls]
        gtitle.config(text=f"#{r['id']}  {title}", fg=CLS_COLOR[cls] if cls != "bird" else "#7bd17f")
        if cls == "trash":
            gtitle.config(fg="#e080ff")
        elif cls == "drone":
            gtitle.config(fg="#ff7b7b")
        elif cls == "balloon":
            gtitle.config(fg="#7fb2ff")
        gtext.config(text=todo)
        gmeta.config(text=f"정찰드론 판정 신뢰도 {r['conf']:.0%} · 출처 {r['src']} · 상태 {r['status']}")
        newest = log.rows and r is log.rows[-1]
        if newest and live_fresh() and set_image(live["img"], "LIVE"):
            return
        if r.get("img") and set_image(r["img"], time.strftime("%H:%M:%S", time.localtime(r["t1"]))):
            return
        photo["ref"] = None
        img_lbl.config(image="", text="캡처 없음")

    def save_capture(r):
        if r.get("img"):
            import base64
            fn = os.path.join(logdir, f"cap_{time.strftime('%H%M%S')}_{r['id']}_{r['cls']}.jpg")
            with open(fn, "wb") as f:
                f.write(base64.b64decode(r["img"]))

    cols = ("t", "id", "cls", "conf", "src", "st")
    heads = ("시각", "#", "판정", "신뢰도", "출처", "상태")
    widths = (80, 40, 90, 60, 75, 130)
    frm = tk.Frame(root, bg="#1d2230")
    frm.pack(fill="both", expand=True, padx=10, pady=6)
    tree = ttk.Treeview(frm, columns=cols, show="headings", selectmode="browse")
    for c, h, w in zip(cols, heads, widths):
        tree.heading(c, text=h)
        tree.column(c, width=w, anchor="center", stretch=(c == "st"))
    tree.tag_configure("bird", foreground="#7bd17f")
    tree.tag_configure("balloon", foreground="#7fb2ff")
    tree.tag_configure("drone", foreground="#ff7b7b")
    tree.tag_configure("trash", foreground="#e080ff")
    tree.tag_configure("done", foreground="#8a8f9c")
    sb = ttk.Scrollbar(frm, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def row_values(r):
        if r["op"] in (None, r["cls"]):
            cls_txt = CLASSES[r["cls"]]
        else:
            cls_txt = f"{CLASSES[r['cls']]}→{CLASSES.get(r['op'], '오탐')}"
        n = f" ×{r['n']}" if r["n"] > 1 else ""
        return (time.strftime("%H:%M:%S", time.localtime(r["t1"])), r["id"], cls_txt + n,
                f"{r['conf']:.2f}", r["src"], r["status"])

    def refresh_row(r):
        if r["op"] is None or r is link["active"]:
            tag = r["op"] if r["op"] in CLASSES else r["cls"]
        else:
            tag = "done"
        tree.item(str(r["id"]), values=row_values(r), tags=(tag,))
        stats_var.set(log.stats())

    def on_detect(cls, conf, src, img=None):
        r, new = log.add(cls, conf, src, time.time(), img)
        if img and (new or r.get("saved") is None):
            save_capture(r)
            r["saved"] = True
        if new:
            tree.insert("", "end", iid=str(r["id"]), values=row_values(r), tags=(cls,))
            tree.see(str(r["id"]))
            sel = tree.selection()
            if not sel or log.rows[int(sel[0]) - 1]["op"] is not None:
                tree.selection_set(str(r["id"]))
            if link["active"] is None:
                banner.config(text=f"#{r['id']} {GUIDE[cls][0]} — {GUIDE[cls][1].split('  (')[0]}",
                              bg=CLS_COLOR[cls])
            write_log(r)
        refresh_row(r)
        sel = selected()
        if sel is r:
            show_detail(r)

    def selected():
        sel = tree.selection()
        return log.rows[int(sel[0]) - 1] if sel else None

    def decide(op):
        r = selected()
        if r is None:
            return
        if r["op"] in ("drone", "trash") and r["status"] in ("미명중", "중단(새 요격)") and op == "AI":
            op, fire = r["op"], True
            r["status"] = "요격 명령" if op == "drone" else "추적 드론 명령"
        elif r["op"] is not None:
            return
        else:
            if op == "AI":
                op = r["cls"]
            fire = log.decide(r, op)
        if fire:
            mission = GUIDE[op][2]
            if arms_dst is None:
                r["status"] += " (시뮬 미연결)"
                banner.config(text=f"#{r['id']} 명령 — A.R.M.S. 연결 안 됨", bg="#8a6d00")
            else:
                arms.sendto(f"{mission},{op},{r['id']}".encode(), arms_dst)
                prev = link["active"]
                link["active"] = r
                if prev is not None and prev is not r:
                    prev["status"] = "중단(새 요격)"
                    refresh_row(prev)
                banner.config(text=f"#{r['id']} 오물풍선 — 추적 드론 발진 (GPS 부착)" if op == "trash"
                              else f"#{r['id']} 무인기 — 요격 드론 발진", bg="#8e24aa" if op == "trash" else "#c62828")
        elif op == "bird":
            banner.config(text=f"#{r['id']} 새로 확인 — 조치 없음, 상황 종료", bg="#2e7d32")
        elif op == "balloon":
            banner.config(text=f"#{r['id']} 풍선(오물 없음) — 관찰", bg="#1565c0")
        else:
            banner.config(text=f"#{r['id']} 오탐으로 기록", bg="#555")
        write_log(r)
        refresh_row(r)
        show_detail(r)
        for x in log.rows:
            if x["op"] is None:
                tree.selection_set(str(x["id"]))
                tree.see(str(x["id"]))
                break

    def poll():
        while True:
            try:
                data, _ = rx.recvfrom(65535)
            except OSError:
                break
            d = parse_packet(data)
            if d and d.get("live"):
                live.update(img=d["img"], t=time.time(), cls=d["cls"], conf=d["conf"])
                sel = selected()
                if sel is not None and sel is log.rows[-1]:
                    set_image(d["img"], "LIVE")
            elif d:
                on_detect(d["cls"], d["conf"], d["src"], d.get("img"))
        while True:
            try:
                data, _ = arms.recvfrom(256)
            except OSError:
                break
            parts = data.decode("utf-8", "ignore").split(",")
            if parts[0] == "pong":
                link["last_pong"] = time.time()
            elif parts[0] == "state" and len(parts) >= 3:
                r = link["active"]
                if r is None or str(r["id"]) != parts[2]:
                    continue
                st = parts[1]
                if st in ("DISARMED", "ARMED") and r["status"] == "요격 명령":
                    continue
                r["status"] = STATE_KO.get(st, st)
                if st == "HIT":
                    banner.config(text=f"#{r['id']} {CLASSES[r['op']]} 요격 성공", bg="#1f9d55")
                    link["active"] = None
                elif st == "FOLLOW":
                    banner.config(text=f"#{r['id']} 추적 드론이 오물에 붙어 따라가는 중 (GPS 부착)", bg="#1f9d55")
                elif st == "MISS":
                    banner.config(text=f"#{r['id']} 실패 — 그 줄에서 Enter 로 다시 출동", bg="#b8860b")
                    link["active"] = None
                elif st in ("TRACK", "TERMINAL"):
                    banner.config(text=f"#{r['id']} 요격기 {STATE_KO[st]} 중", bg="#d33a3a")
                write_log(r)
                refresh_row(r)
                if selected() is r:
                    show_detail(r)
        root.after(50, poll)

    def ping():
        if arms_dst:
            try:
                arms.sendto(b"ping", arms_dst)
            except OSError:
                pass
            ok = time.time() - link["last_pong"] < 4
            link_var.set(f"요격 시뮬 {arms_dst[0]}:{arms_dst[1]} " + ("● 연결됨" if ok else "○ 응답 없음"))
            link_lbl.config(fg="#7bd17f" if ok else "#ff9b6b")
        else:
            link_var.set("요격 시뮬 주소 없음 (--arms-host)")
        root.after(2000, ping)

    def demo_tick():
        on_detect(random.choice(list(CLASSES)), random.uniform(0.6, 0.97), "demo")
        root.after(random.randint(4000, 8000), demo_tick)

    tree.bind("<<TreeviewSelect>>", lambda e: show_detail(selected()))
    root.bind("<Return>", lambda e: decide("AI"))
    root.bind("1", lambda e: decide("bird"))
    root.bind("2", lambda e: decide("balloon"))
    root.bind("3", lambda e: decide("drone"))
    root.bind("4", lambda e: decide("trash"))
    root.bind("<Delete>", lambda e: decide("false_alarm"))
    root.bind("<F1>", lambda e: on_detect("bird", 1.0, "manual"))
    root.bind("<F2>", lambda e: on_detect("balloon", 1.0, "manual"))
    root.bind("<F3>", lambda e: on_detect("drone", 1.0, "manual"))
    root.bind("<F4>", lambda e: on_detect("trash", 1.0, "manual"))

    def on_close():
        log_f.close()
        rx.close()
        arms.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    stats_var.set(log.stats())
    tree.focus_set()
    poll()
    ping()
    if demo:
        root.after(1000, demo_tick)
    root.mainloop()


def selftest():
    ok = True
    cases = [(b'{"cls":"balloon","conf":0.87,"src":"recon1"}', "balloon"), (b"drone,0.6", "drone"),
             (b"bird", "bird"), (b"cat,0.9", None), (b"{bad", None), (b"\xff\xfe", None)]
    for raw, want in cases:
        got = parse_packet(raw)
        got = got and got["cls"]
        ok &= got == want
        print(f"parse {raw!r:50} -> {got}  {'OK' if got == want else 'FAIL'}")
    L = EventLog()
    L.add("balloon", 0.8, "r", 0.0)
    L.add("balloon", 0.9, "r", 5.0)
    L.add("drone", 0.7, "r", 6.0)
    L.add("drone", 0.7, "r", 20.0)
    merged = [(r["cls"], r["n"]) for r in L.rows]
    print("병합:", merged)
    ok &= merged == [("balloon", 2), ("drone", 1), ("drone", 1)]
    fire = [L.decide(L.rows[0], "trash"), L.decide(L.rows[1], "bird"), L.decide(L.rows[2], "false_alarm"),
            L.decide(L.rows[0], "drone")]
    print("출동 명령 여부:", fire, [r["status"] for r in L.rows])
    ok &= fire == [True, False, False, False]
    L2 = EventLog()
    L2.add("balloon", 0.8, "r", 0.0)
    L2.add("drone", 0.8, "r", 1.0)
    f2 = [L2.decide(L2.rows[0], "balloon"), L2.decide(L2.rows[1], "drone")]
    print("풍선=관찰, 무인기=요격:", f2, [r["status"] for r in L2.rows])
    ok &= f2 == [False, True]
    d = parse_packet(b'{"cls":"trash","conf":0.9,"src":"pi","img":"AAAA"}')
    ok &= d is not None and d["img"] == "AAAA" and not d["live"]
    dl = parse_packet(b'{"type":"live","cls":"bird","conf":0.8,"src":"pi","img":"BBBB"}')
    ok &= dl is not None and dl["live"]
    print("실시간 캡처 구분:", dl and dl["live"])
    print("이미지 포함 수신:", d and (d["cls"], d["img"]))
    print(L.stats())
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9500, help="정찰드론 탐지 수신")
    ap.add_argument("--arms-host", default="auto", help="arms.py 주소 (auto = WSL IP)")
    ap.add_argument("--arms-port", type=int, default=9510)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--geometry", default="640x1040+0+0")
    ap.add_argument("--logdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    host = wsl_ip() if a.arms_host == "auto" else a.arms_host
    run_gui(a.port, host, a.arms_port, a.demo, a.logdir, a.geometry)
