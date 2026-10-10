import argparse
import multiprocessing as mp
import os
import queue
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W, H = 640, 480
FALLBACK = ["rpicam-hello", "-t", "0", "--framerate", "30", "--width", "640", "--height", "480"]
COL = {"trash": (255, 0, 255), "balloon": (0, 200, 255), "bird": (0, 220, 0), "drone": (0, 0, 255)}
LABEL = {"trash": "TRASH", "balloon": "BALLOON", "bird": "BIRD", "drone": "DRONE"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fallback(why):
    log("탐지 없이 영상만 송출로 전환:", why)
    if os.environ.get("CUAS_FB"):
        sys.exit(1)
    os.execvp(FALLBACK[0], FALLBACK)


def fb_open(np):
    path = os.environ.get("CUAS_FB", "/dev/fb0")
    if "CUAS_FB_SIZE" in os.environ:
        fw, fh, bpp = map(int, os.environ["CUAS_FB_SIZE"].split("x"))
        stride = fw * bpp // 8
        if not os.path.exists(path) or os.path.getsize(path) < stride * fh:
            with open(path, "wb") as f:
                f.write(b"\0" * stride * fh)
    else:
        base = "/sys/class/graphics/fb0"
        fw, fh = map(int, open(base + "/virtual_size").read().strip().split(","))
        bpp = int(open(base + "/bits_per_pixel").read())
        stride = int(open(base + "/stride").read()) if os.path.exists(base + "/stride") else fw * bpp // 8
    mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=(fh, stride))
    return mm, fw, fh, bpp, stride


def quiet_console():
    subprocess.run(["systemctl", "stop", "getty@tty1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        open("/sys/class/graphics/fbcon/cursor_blink", "w").write("0")
    except OSError:
        pass
    try:
        with open("/dev/tty1", "w") as t:
            t.write("\033[?25l\033[2J")
    except OSError:
        pass


def net_diag():
    for cmd in (["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "dev"],
                ["nmcli", "-t", "-f", "NAME,TYPE,AUTOCONNECT", "con", "show"],
                ["nmcli", "-t", "-f", "SSID,CHAN,FREQ,SIGNAL,SECURITY", "dev", "wifi", "list", "--rescan", "no"],
                ["ip", "-4", "-br", "addr"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:
            out = f"(실패 {e})"
        log("[net]", " ".join(cmd[:4]), "|", out.replace("\n", " ; ")[:900])


def tmatch(cv2, gray, tmpl, box):
    x1, y1, x2, y2 = box
    th, tw = tmpl.shape
    m = max(10, int(0.8 * max(tw, th)))
    H_, W_ = gray.shape
    sx1, sy1 = max(0, int(x1 - m)), max(0, int(y1 - m))
    sx2, sy2 = min(W_, int(x2 + m)), min(H_, int(y2 + m))
    region = gray[sy1:sy2, sx1:sx2]
    if region.shape[0] < th or region.shape[1] < tw:
        return box, 0.0
    r = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    nx, ny = sx1 + loc[0], sy1 + loc[1]
    return (nx, ny, nx + tw, ny + th), float(mx)


class Follower:
    PAD = 3

    def __init__(self, cv2):
        from collections import deque
        self.cv2 = cv2
        self.ring = deque(maxlen=24)
        self.items = []
        self.last_fid = None

    def push(self, fid, gray):
        self.ring.append((fid, gray))

    def reset(self, fid, dets):
        frames = list(self.ring)
        src = next((g for f, g in frames if f == fid), None)
        later = [g for f, g in frames if f > fid]
        if src is None:
            src, later = (frames[-1][1], []) if frames else (None, [])
        self.items = []
        if src is None:
            return
        H_, W_ = src.shape
        for d in dets:
            (x1, y1, x2, y2), c, cf = d[0], d[1], d[2]
            b = [int(x1 / 2) - self.PAD, int(y1 / 2) - self.PAD, int(x2 / 2) + self.PAD, int(y2 / 2) + self.PAD]
            b = [max(0, b[0]), max(0, b[1]), min(W_, b[2]), min(H_, b[3])]
            if b[2] - b[0] < 6 or b[3] - b[1] < 6:
                continue
            tmpl = src[b[1]:b[3], b[0]:b[2]].copy()
            box = tuple(b)
            for g in later:
                nb, sc = tmatch(self.cv2, g, tmpl, box)
                if sc >= 0.4:
                    box = nb
            self.items.append(dict(box=box, tmpl=tmpl, cls=c, conf=cf, t_ok=time.time()))
        self.last_fid = fid

    def step(self, gray):
        now = time.time()
        keep = []
        for it in self.items:
            nb, sc = tmatch(self.cv2, gray, it["tmpl"], it["box"])
            if sc >= 0.4:
                it["box"], it["t_ok"] = nb, now
            if now - it["t_ok"] < 0.6:
                keep.append(it)
        self.items = keep

    def boxes(self):
        P_ = self.PAD
        return [(((it["box"][0] + P_) * 2, (it["box"][1] + P_) * 2, (it["box"][2] - P_) * 2, (it["box"][3] - P_) * 2),
                 it["cls"], it["conf"]) for it in self.items]


def put_latest(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def analyzer(model, udp, q_in, q_out, scene_every, det_w=320, mode="recon", max_hz=10.0):
    import cv2
    import motion_id as M
    from appearance_id import AppearanceClassifier
    cv2.setNumThreads(2)
    app = AppearanceClassifier(model)
    P = M.Pipeline(10.0, appearance=app, recon=True, scene_every=scene_every, scene_thr=0.8, max_classify=2)
    S = M.SimpleDetector(app)
    sender = None
    if udp != "none":
        try:
            from detect_msg import Sender
            sender = Sender(udp, src="drone")
        except OSError as e:
            log("UDP 송출 끔:", e)
    n, t0 = 0, time.monotonic()
    t_last = 0.0
    while True:
        item = q_in.get()
        if item is None:
            return
        fid, small = item
        wait = 1.0 / max_hz - (time.time() - t_last)
        if wait > 0:
            time.sleep(wait)
        t_last = time.time()
        if mode == "simple":
            r = S.step(small)
            k, res, scene = 1.0, [], None
            dets = [r] if r else []
            n += 1
            out = ([((x1 * 2, y1 * 2, x2 * 2, y2 * 2), c, cf, (0.0, 0.0), time.time()) for (x1, y1, x2, y2), c, cf in dets],
                   None, n / (time.monotonic() - t0), fid)
            put_latest(q_out, out)
            if sender:
                try:
                    sender.send_frame(small, dets)
                    if r:
                        (x1, y1, x2, y2), c, cf = r
                        args = (c, cf, small, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1))
                        sender.maybe_send(*args)
                        sender.send_live(*args)
                except OSError:
                    pass
            continue
        if det_w < 320:
            k = 320 / det_w
            tiny = cv2.resize(small, (det_w, det_w * 3 // 4), interpolation=cv2.INTER_AREA)
            res = P.step(time.time(), tiny, small)
        else:
            k = 1.0
            res = P.step(time.time(), small)
        dets = [(tuple(v * k for v in P.box(q)), q.last[0], q.last[1]) for q in res]
        now_a = time.time()
        vel = []
        for q in res:
            j = max(0, len(q.t) - 4)
            dt = q.t[-1] - q.t[j]
            vel.append(((q.x[-1] - q.x[j]) * k * 2 / dt, (q.y[-1] - q.y[j]) * k * 2 / dt) if dt > 0 else (0.0, 0.0))
        scene = P.scene if not res else None
        n += 1
        out = ([((x1 * 2, y1 * 2, x2 * 2, y2 * 2), c, cf, v, now_a) for ((x1, y1, x2, y2), c, cf), v in zip(dets, vel)],
               scene, n / (time.monotonic() - t0), fid)
        put_latest(q_out, out)
        if sender:
            try:
                sender.send_frame(small, dets or ([(None, scene[0], scene[1])] if scene else []))
                if res:
                    q = max(res, key=lambda r: r.extent_px())
                    x1, y1, x2, y2 = (v * k for v in P.box(q))
                    args = (q.last[0], q.last[1], small, (x1 + x2) / 2, (y1 + y2) / 2, q.extent_px() * k)
                    sender.maybe_send(*args)
                    sender.send_live(*args)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "appearance", "out", "cuas_cls.onnx"))
    ap.add_argument("--udp", default="255.255.255.255:9500",
                    help="노트북 GUI 로 송출(기본 = 같은 와이파이 전체 브로드캐스트), 끄려면 none")
    ap.add_argument("--fps", type=int, default=30, help="카메라·고글 표시 프레임률(NTSC 출력과 맞춰 30)")
    ap.add_argument("--scene-every", type=int, default=6, help="표적 못 찾을 때 화면 전체 판정 간격(분석 프레임)")
    ap.add_argument("--max-hz", type=float, default=10.0, help="분석 최대 초당 횟수(넘으면 쉼 → 고글 영상이 안 밀림)")
    ap.add_argument("--mode", choices=["simple", "recon"], default="recon",
                    help="recon=추적·확정 거름 있는 정찰 모드(기본, 헛경보 적음) / simple=테니스공 방식(가장 두드러진 것 1개 바로 박스)")
    ap.add_argument("--det-width", type=int, default=320,
                    help="찾기 해상도 폭. 160=빠름(표적이 클 때) / 320=작은 표적도(기본, 10/11 실기에서 160은 화면 재촬영 표적을 놓침)")
    ap.add_argument("--fc", default="/dev/serial0", help="Betaflight FC UART(MSP). none=끔(스위치 없이 항상 탐지 표시)")
    ap.add_argument("--fc-baud", type=int, default=115200)
    ap.add_argument("--sw-aux", type=int, default=4, help="BF Modes 탭에서 MSP OVERRIDE 를 건 AUX 번호")
    ap.add_argument("--sw-min", type=int, default=1700, help="이 us 이상이면 스위치 ON (Modes 탭 범위와 맞출 것)")
    ap.add_argument("--seconds", type=float, default=0, help="시험용: N초 뒤 종료")
    ap.add_argument("--target", default="any",
                    help="따라갈 종류: balloon / drone / trash / bird, 여러 개는 쉼표(balloon,trash), any=전부")
    ap.add_argument("--follow", choices=["yaw", "yaw+pitch"], default="yaw",
                    help="yaw=제자리 회전만 / yaw+pitch=회전+다가가기(FC 마스크 10, ANGLE 모드 필수)")
    ap.add_argument("--size-want", type=float, default=0.25, help="yaw+pitch: 표적이 화면 높이의 이 비율이 되면 멈춤")
    ap.add_argument("--pitch-fwd", type=float, default=0.5, help="yaw+pitch: 최대 전진(1=피치 스틱 +100us)")
    extra = os.environ.get("CUAS_ARGS_FILE", "/boot/firmware/cuas_pkg/args.txt")
    argv = sys.argv[1:]
    if os.path.exists(extra):
        argv = open(extra, encoding="utf-8").read().split() + argv
    a = ap.parse_args(argv)
    try:
        import cv2
        import numpy as np
        import motion_id
        import appearance_id
    except Exception as e:
        fallback(f"모듈 없음: {e}")
    try:
        fb, fbw, fbh, bpp, stride = fb_open(np)
    except Exception as e:
        fallback(f"fb0 열기 실패: {e}")
    log(f"fb {fbw}x{fbh} {bpp}bpp stride {stride} | OpenCV {cv2.__version__} | v6 {a.mode} 모드·분석 {a.max_hz:g}Hz + 매 장면 템플릿 추적·조명 필터·카메라 {a.fps}fps")
    if not os.environ.get("CUAS_FB"):
        quiet_console()
        import threading

        def _diag_once():
            time.sleep(20)
            subprocess.run(["iw", "dev", "wlan0", "set", "power_save", "off"], capture_output=True)
            net_diag()
        threading.Thread(target=_diag_once, daemon=True).start()
    cv2.setNumThreads(2)

    ctx = mp.get_context("fork")
    q_in, q_out = ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)
    q_in.cancel_join_thread()
    def start_analyzer():
        pr = ctx.Process(target=analyzer, args=(a.model, a.udp, q_in, q_out, a.scene_every, a.det_width, a.mode, a.max_hz), daemon=True)
        pr.start()
        return pr
    proc = start_analyzer()
    fc = None
    if a.fc != "none":
        from fc_link import FcLink
        fc = FcLink(a.fc, a.fc_baud, sw_aux=a.sw_aux, sw_min=a.sw_min, width=W, height=H, target=a.target,
                    follow=a.follow, size_want=a.size_want, pitch_fwd=a.pitch_fwd)
        log(f"FC 링크 {a.fc} {a.fc_baud}: " + ("열림" if fc.ok else fc.err) + f" | 스위치 AUX{a.sw_aux} >= {a.sw_min}us"
            f" | 목표 {a.target} | {a.follow}")

    cam_cmd = os.environ.get("CUAS_CAM_CMD")
    cmd = shlex.split(cam_cmd) if cam_cmd else ["rpicam-vid", "-t", "0", "-n", "--codec", "yuv420", "--width", str(W),
                                                "--height", str(H), "--framerate", str(a.fps), "--flush", "-o", "-"]
    try:
        cam = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    except OSError as e:
        proc.terminate()
        fallback(f"카메라 실행 실패: {e}")
    nbytes = W * H * 3 // 2

    ox, oy = max(0, (fbw - W) // 2), max(0, (fbh - H) // 2)
    cw, ch = min(W, fbw), min(H, fbh)
    buf = bytearray(nbytes)
    view = memoryview(buf)
    dets, scene, fps_a = [], None, 0.0
    fol = Follower(cv2)
    fid = 0
    t_start, nshow, last_log = time.monotonic(), 0, time.monotonic()
    try:
        while True:
            got = 0
            while got < nbytes:
                r = cam.stdout.readinto(view[got:])
                if not r:
                    raise EOFError("카메라 스트림 끝")
                got += r
            yuv = np.frombuffer(buf, np.uint8).reshape(H * 3 // 2, W)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            fid += 1
            small = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
            gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            fol.push(fid, gray_s)
            if q_in.empty():
                try:
                    q_in.put_nowait((fid, small))
                except queue.Full:
                    pass
            fresh = False
            try:
                dets, scene, fps_a, fid_a = q_out.get_nowait()
                fol.reset(fid_a, dets)
                fresh = True
            except queue.Empty:
                pass
            if not fresh:
                fol.step(gray_s)
            boxes = fol.boxes()
            sw, ycmd, tgt = fc.update(boxes) if fc else (False, 0.0, None)
            fc_live = fc is not None and fc.linked()
            show = sw or not fc_live
            for (x1, y1, x2, y2), c, cf in (boxes if show else []):
                col = COL.get(c, (255, 255, 255))
                cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
                txt = f"{LABEL.get(c, c.upper())} {cf:.2f}"
                (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                ty = max(th_ + 6, int(y1) - 6)
                cv2.rectangle(bgr, (int(x1), ty - th_ - 6), (int(x1) + tw + 8, ty + 4), col, -1)
                cv2.putText(bgr, txt, (int(x1) + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            if fc_live:
                if sw:
                    st = f"AUTO {a.target.upper()}" + (" LOCK" if tgt is not None else " SEARCH")
                    cv2.putText(bgr, st, (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 6)
                    cv2.putText(bgr, st, (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if a.follow == "yaw+pitch":
                        by = int(H / 2 - fc.pcmd * 150)
                        cv2.line(bgr, (W - 30, H // 2), (W - 30, by), (0, 0, 255), 6)
                        cv2.line(bgr, (W - 40, H // 2), (W - 20, H // 2), (255, 255, 255), 2)
                    bx = int(W / 2 + ycmd * 150)
                    cv2.line(bgr, (W // 2, H - 30), (bx, H - 30), (0, 0, 255), 6)
                    cv2.line(bgr, (W // 2, H - 40), (W // 2, H - 20), (255, 255, 255), 2)
                    if tgt is not None:
                        (x1, y1, x2, y2) = tgt[0]
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        cv2.drawMarker(bgr, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
                else:
                    cv2.putText(bgr, "MANUAL", (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 6)
                    cv2.putText(bgr, "MANUAL", (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if show and not fol.items and scene:
                c, cf = scene
                cv2.putText(bgr, f"{LABEL.get(c, c.upper())} {cf:.2f} (FULL VIEW)", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL.get(c, (255, 255, 255)), 2)
            cv2.putText(bgr, "CUAS", (W - 90, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if bpp == 16:
                px = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGR565)
                fb[oy:oy + ch, ox * 2:(ox + cw) * 2] = px[:ch, :cw].reshape(ch, cw * 2)
            else:
                px = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                fb[oy:oy + ch, ox * 4:(ox + cw) * 4] = px[:ch, :cw].reshape(ch, cw * 4)
            nshow += 1
            now = time.monotonic()
            if nshow % 60 == 0 and not proc.is_alive():
                log("분석 프로세스 종료됨 - 다시 시작", proc.exitcode)
                proc = start_analyzer()
            if now - last_log > 5:
                labels = ", ".join(f"{d[1]} {d[2]:.2f}" for d in dets) or (f"(화면){scene[0]}" if scene else "-")
                health = ""
                if not os.environ.get("CUAS_FB") and int(now - t_start) % 30 < 6:
                    try:
                        th = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout.strip()
                        tp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=3).stdout.strip()
                        health = f" | {th} {tp}"
                    except Exception:
                        pass
                log(f"표시 {nshow / (now - t_start):.1f} fps | 분석 {fps_a:.1f} fps | {labels}{health}")
                if health:
                    try:
                        os.sync()
                    except OSError:
                        pass
                last_log = now
            if a.seconds and now - t_start > a.seconds:
                break
    except Exception as e:
        log("오류:", e)
        cam.kill()
        proc.terminate()
        if not a.seconds:
            raise
    cam.kill()
    proc.terminate()
    if hasattr(fb, "flush"):
        fb.flush()


if __name__ == "__main__":
    main()
