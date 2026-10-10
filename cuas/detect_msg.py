import base64
import json
import socket
import time

import cv2


COL = {"trash": (255, 0, 255), "balloon": (0, 200, 255), "bird": (0, 200, 0), "drone": (0, 0, 255)}


def crop_jpeg_b64(frame, x, y, ext, out_px=200, quality=70, box=None):
    H, W = frame.shape[:2]
    side = int(max(64, min(ext * 2.2, W, H)))
    x0 = int(min(max(x - side / 2, 0), W - side))
    y0 = int(min(max(y - side / 2, 0), H - side))
    crop = frame[y0:y0 + side, x0:x0 + side]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    crop = cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA)
    if box:
        k = out_px / side
        cx, cy, e = (x - x0) * k, (y - y0) * k, ext * k
        c = COL.get(box[0], (200, 200, 200))
        cv2.rectangle(crop, (int(cx - e / 2), int(cy - e / 2)), (int(cx + e / 2), int(cy + e / 2)), c, 2)
        cv2.rectangle(crop, (0, 0), (out_px, 22), (0, 0, 0), -1)
        cv2.putText(crop, f"{box[0].upper()} {box[1]:.2f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


class Sender:
    def __init__(self, dst, src="recon", interval=2.0, live_hz=5.0):
        h, p = dst.rsplit(":", 1)
        self.dst = (h, int(p))
        self.src, self.interval = src, interval
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.last_cls, self.last_t = None, 0.0
        self.live_dt, self.last_live = 1.0 / live_hz if live_hz > 0 else 1e9, 0.0
        self.last_frame = 0.0

    def send_frame(self, frame, dets, max_hz=10.0, width=320, quality=60):
        now = time.time()
        if frame is None or now - self.last_frame < 1.0 / max_hz:
            return False
        h, w = frame.shape[:2]
        k = width / w if w > width else 1.0
        small = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        msg = {"type": "frame", "src": self.src, "t": round(now, 3), "w": small.shape[1], "h": small.shape[0],
               "img": base64.b64encode(buf.tobytes()).decode("ascii"),
               "dets": [{"box": [int(v * k) for v in bx] if bx is not None else None, "cls": c, "conf": round(float(cf), 3)}
                        for bx, c, cf in dets]}
        data = json.dumps(msg).encode()
        if len(data) > 62000:
            return False
        self.sock.sendto(data, self.dst)
        self.last_frame = now
        return True

    def send_live(self, cls, conf, frame, x, y, ext):
        now = time.time()
        if frame is None or now - self.last_live < self.live_dt:
            return False
        img = crop_jpeg_b64(frame, x, y, ext, quality=60, box=(cls, conf))
        if not img:
            return False
        msg = {"type": "live", "cls": cls, "conf": round(float(conf), 3), "src": self.src, "img": img}
        self.sock.sendto(json.dumps(msg).encode(), self.dst)
        self.last_live = now
        return True

    def maybe_send(self, cls, conf, frame=None, x=0, y=0, ext=0):
        if cls in ("unknown", "none"):
            return False
        now = time.time()
        if cls == self.last_cls and now - self.last_t < self.interval:
            return False
        msg = {"cls": cls, "conf": round(float(conf), 3), "src": self.src}
        if frame is not None:
            img = crop_jpeg_b64(frame, x, y, ext)
            if img and len(img) < 55000:
                msg["img"] = img
        self.sock.sendto(json.dumps(msg).encode(), self.dst)
        self.last_cls, self.last_t = cls, now
        return True
