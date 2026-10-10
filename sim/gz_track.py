import argparse
import json
import os
import socket
import struct
import sys
import time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "..", "sync_work")
sys.path.insert(0, SW)
from track_mask import track_mask
from bc_features import center
from track_render import (Camera, R_MID, R_IN, R_OUT, HALF_STRAIGHT,
                          lateral_error, track_heading_ccw)

SPEED, HEIGHT = 6.0, 8.0
YAW_GAIN = np.radians(86.0)
TAU, DT = 0.10, 1 / 30
HALF_W = (R_OUT - R_IN) / 2
MAGIC, MAGIC_CHASE, MAGIC_IMG = 0x4B435254, 0x45534843, 0x43474D49
HDR = struct.Struct("<IId6fHH")
CMD = struct.Struct("<II6fI")
CHS = struct.Struct("<IIHHHHII")


class RidgePolicy:
    def __init__(self, path):
        p = np.load(path)
        self.coef, self.b = p["coef"], float(p["intercept"])
        self.mu, self.sd, self.lag = p["mu"], p["sd"], int(p["lag_frames"])
        self.hist = []

    def __call__(self, mask, state):
        f = center(mask[None])[0]
        self.hist.append(f)
        prev = self.hist[-1 - self.lag] if len(self.hist) > self.lag else f
        x = (np.concatenate([f, f - prev]) - self.mu) / self.sd
        return float(np.clip(x @ self.coef + self.b, -1, 1))


def geom_policy(mask, s):
    x, y, yaw = s
    e = lateral_error(x, y)
    he = np.angle(np.exp(1j * (track_heading_ccw(x, y) - yaw)))
    ff = SPEED / R_MID if abs(x) > HALF_STRAIGHT else 0.0
    return float(np.clip(-(ff + 1.5 * he + 0.15 * e) / YAW_GAIN, -1, 1))


def panel(fpv_rgb, chase_rgb, mask, hist, t, stick, off, name):
    fpv = cv2.resize(cv2.cvtColor(fpv_rgb, cv2.COLOR_RGB2BGR), (480, 360))
    ch = cv2.resize(cv2.cvtColor(chase_rgb, cv2.COLOR_RGB2BGR), (480, 360)) if chase_rgb is not None \
        else np.zeros((360, 480, 3), np.uint8)
    cv2.putText(fpv, "drone camera (Gazebo)", (8, 22), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(ch, "chase view", (8, 22), 0, 0.6, (255, 255, 255), 2)
    bot = np.zeros((200, 960, 3), np.uint8)
    mk = cv2.resize((mask * 255).astype(np.uint8), (256, 192), interpolation=cv2.INTER_NEAREST)
    bot[4:196, 4:260] = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
    cv2.putText(bot, "AI input: track mask", (8, 190), 0, 0.45, (0, 255, 255), 1)
    gx0, gy0, gw, gh = 280, 20, 660, 130
    cv2.rectangle(bot, (gx0, gy0), (gx0 + gw, gy0 + gh), (70, 70, 70), 1)
    cv2.line(bot, (gx0, gy0 + gh // 2), (gx0 + gw, gy0 + gh // 2), (90, 90, 90), 1)
    h = np.array(hist[-180:])
    if len(h) > 1:
        xs = gx0 + gw - (h[-1, 0] - h[:, 0]) / 6.0 * gw
        ys = gy0 + gh / 2 - np.clip(h[:, 1], -0.4, 0.4) / 0.4 * (gh / 2 - 4)
        cv2.polylines(bot, [np.stack([xs, ys], 1).astype(np.int32)], False, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(bot, f"policy={name}  t={t:5.1f}s  direction stick {stick:+.2f}  offset from lane center {off:+.1f} m",
                (gx0, gy0 + gh + 25), 0, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([np.hstack([fpv, ch]), bot])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ridge", choices=["ridge", "zero", "geom"])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--out", default=os.path.join(SW, "sim_gazebo"))
    ap.add_argument("--ridge", default=os.path.join(SW, "sim_offline", "ridge_policy.npz"))
    ap.add_argument("--stop-off-track", action="store_true", default=True)
    ap.add_argument("--delay", type=float, default=0.0, help="정책 출력 지연(초): 조종사 반응 시간 흉내")
    ap.add_argument("--tag", default="", help="출력 파일 이름 뒤에 붙일 표시")
    ap.add_argument("--snap", nargs=6, type=float, default=None,
                    help="디버그: 이 자세(x y z roll pitch yaw[도])로 옮겨 한 장만 찍고 끝")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = RidgePolicy(a.ridge) if a.policy == "ridge" else (geom_policy if a.policy == "geom" else (lambda m, s: 0.0))
    nd, q = int(round(a.delay / DT)), []

    def pol(m, s_):
        q.append(base(m, s_))
        return q[-1 - nd] if len(q) > nd else 0.0
    name = a.policy + a.tag

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    sock.bind(("127.0.0.1", 9301))
    sock.settimeout(180.0)
    peer = ("127.0.0.1", 9302)

    x, y, yaw, rate = -30.0, -R_MID, 0.0, 0.0
    vw = cv2.VideoWriter(os.path.join(a.out, f"gz_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 560))
    chase, chunks, imgs = None, {}, {}
    log, hist, k, off_t = [], [], 0, None
    laps_ang, prev_ang = 0.0, np.arctan2(y, x)
    cam_ref = Camera(hfov_deg=110.0, down_deg=40.0, height=HEIGHT)
    t_wall = time.time()
    warm, dropped = 0, 0
    while k * DT < a.seconds:
        pkt, _ = sock.recvfrom(1 << 21)
        magic = struct.unpack_from("<I", pkt)[0]
        if magic == MAGIC_CHASE:
            _, seq, idx, n, w, h, off, ln = CHS.unpack_from(pkt)
            buf = chunks.setdefault(seq, bytearray(w * h * 3))
            buf[off:off + ln] = pkt[CHS.size:CHS.size + ln]
            if idx == n - 1:
                chase = np.frombuffer(bytes(buf), np.uint8).reshape(h, w, 3)
                chunks.pop(seq, None)
            continue
        if magic == MAGIC_IMG:
            _, seq, idx, n, w, h, off, ln = CHS.unpack_from(pkt)
            ent = imgs.setdefault(seq, [bytearray(w * h * 3), 0])
            ent[0][off:off + ln] = pkt[CHS.size:CHS.size + ln]
            ent[1] += ln
            continue
        if magic != MAGIC:
            continue
        hd = HDR.unpack_from(pkt)
        seq, w, h = hd[1], hd[9], hd[10]
        ent = imgs.pop(seq, None)
        for old_seq in [s_ for s_ in imgs if s_ < seq]:
            imgs.pop(old_seq)
        if ent is None or ent[1] < w * h * 3:
            dropped += 1
            sock.sendto(CMD.pack(MAGIC, seq, x, y, HEIGHT, 0, 0, yaw, 0), peer)
            continue
        img = np.frombuffer(bytes(ent[0]), np.uint8).reshape(h, w, 3)
        px, py, pyaw = hd[3], hd[4], hd[8]
        if a.snap is not None:
            sx, sy, sz, sr, sp, syw = a.snap
            if warm >= 5:
                cv2.imwrite(os.path.join(a.out, "snap.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if chase is not None:
                    cv2.imwrite(os.path.join(a.out, "snap_chase.png"), cv2.cvtColor(chase, cv2.COLOR_RGB2BGR))
                print("snap 저장, 자세", [round(v, 2) for v in hd[3:9]], flush=True)
                sock.sendto(CMD.pack(MAGIC, seq, sx, sy, sz, *np.radians([sr, sp, syw]), 0), peer)
                break
            warm += 1
            sock.sendto(CMD.pack(MAGIC, seq, sx, sy, sz, *np.radians([sr, sp, syw]), 0), peer)
            continue
        if warm < 5:
            warm += 1
            sock.sendto(CMD.pack(MAGIC, seq, x, y, HEIGHT, 0, 0, yaw, 0), peer)
            continue
        mask = track_mask(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sock.settimeout(20.0)
        if k == 0:
            ref = cam_ref.mask(px, py, pyaw)
            inter, uni = (mask * ref).sum(), np.clip(mask + ref, 0, 1).sum()
            iou = float(inter / max(uni, 1))
            cv2.imwrite(os.path.join(a.out, "render_check.png"), np.hstack([
                cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (256, 192)),
                cv2.cvtColor(cv2.resize((mask * 255).astype(np.uint8), (256, 192), interpolation=0), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(cv2.resize((ref * 255).astype(np.uint8), (256, 192), interpolation=0), cv2.COLOR_GRAY2BGR)]))
            print(f"[대조] Gazebo 마스크 vs 간이 렌더러 IoU {iou:.2f} (자세 {px:.1f},{py:.1f},{np.degrees(pyaw):.0f}°)", flush=True)
        stick = pol(mask, (px, py, pyaw))
        rate += (-stick * YAW_GAIN - rate) * DT / TAU
        yaw += rate * DT
        x += SPEED * np.cos(yaw) * DT
        y += SPEED * np.sin(yaw) * DT
        sock.sendto(CMD.pack(MAGIC, seq, x, y, HEIGHT, 0, 0, yaw, 0), peer)

        t = k * DT
        e = float(lateral_error(px, py))
        ang = np.arctan2(py, px)
        laps_ang += np.angle(np.exp(1j * (ang - prev_ang)))
        prev_ang = ang
        hist.append((t, stick))
        log.append((t, px, py, pyaw, stick, e, float(mask.mean())))
        vw.write(panel(img, chase, mask, hist, t, stick, e, name))
        k += 1
        if k % 150 == 0:
            print(f"  t={t:5.1f}s 위치 ({px:6.1f},{py:6.1f}) 스틱 {stick:+.2f} 중앙선오차 {e:+.2f}m "
                  f"[실시간 대비 {t / (time.time() - t_wall):.2f}배]", flush=True)
        if abs(e) > HALF_W and off_t is None:
            off_t = t
            print(f"  트랙 이탈 t={t:.1f}s", flush=True)
            if a.stop_off_track:
                for _ in range(45):
                    vw.write(panel(img, chase, mask, hist, t, stick, e, name))
                break
    vw.release()
    np.savetxt(os.path.join(a.out, f"gz_{name}.csv"), np.array(log), delimiter=",",
               header="t,x,y,yaw,stick,offset_m,mask_frac", comments="", fmt="%.4f")
    print(f"유실로 버린 화면 {dropped}개", flush=True)
    res = dict(policy=name, delay_s=a.delay, ridge_file=os.path.basename(a.ridge), dropped_frames=dropped, seconds=round(k * DT, 1), off_track_s=off_t, laps=round(laps_ang / (2 * np.pi), 2),
               mean_abs_offset_m=round(float(np.mean([abs(r[5]) for r in log])), 2) if log else None)
    json.dump(res, open(os.path.join(a.out, f"gz_{name}.json"), "w"), indent=1)
    print("결과:", res, flush=True)


if __name__ == "__main__":
    main()
