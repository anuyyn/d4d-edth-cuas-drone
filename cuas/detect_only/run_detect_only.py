import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PI = "172.20.10.13"
FILES = ["pi_fb_detect.py", "fc_link.py", "motion_id.py", "appearance_id.py", "detect_msg.py",
         "appearance/out/cuas_cls.onnx", "appearance/out/cuas_cls.json"]
PYLIB = os.environ.get("CUAS_PYLIB", "")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="cp949", errors="replace").stdout


def ensure_wifi():
    if "cuas-hotspot" not in sh(["netsh", "wlan", "show", "interfaces"]):
        print("노트북 와이파이 → cuas-hotspot")
        sh(["netsh", "wlan", "connect", "name=cuas-hotspot"])
        time.sleep(6)


def wait_pi(sec=180):
    t0 = time.time()
    while time.time() - t0 < sec:
        if "TTL=" in sh(["ping", "-n", "1", "-w", "1500", PI]):
            return True
    return False


def ssh():
    sys.path.insert(0, PYLIB)
    import paramiko
    for _ in range(6):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(PI, username=os.environ.get("CUAS_PI_USER", "pi"), password=os.environ["CUAS_PI_PASS"], timeout=15, allow_agent=False, look_for_keys=False)
            return c
        except Exception as e:
            print("SSH 재시도:", e)
            time.sleep(5)
    sys.exit("파이 SSH 실패")


def run(c, cmd, t=90):
    _, o, e = c.exec_command(cmd, timeout=t)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace").replace("[sudo] password for pi: ", "").strip()
    return out + (("\n[stderr] " + err) if err else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()
    S = "echo '" + os.environ["CUAS_PI_PASS"] + "' | sudo -S "
    ensure_wifi()
    print("파이 기다리는 중 …")
    if not wait_pi():
        sys.exit("파이가 핫스팟(cuas-hotspot)에 안 보임 — 파이 전원·핫스팟 확인")
    c = ssh()
    if a.stop:
        print(run(c, S + "systemctl stop cuas-detect-only; " + S + "systemctl start fpv-detect; systemctl is-active fpv-detect"))
        return
    sf = c.open_sftp()
    run(c, "mkdir -p /home/pi/cuas_detect/appearance/out")
    for f in FILES:
        sf.put(os.path.join(HERE, f.replace("/", os.sep)), "/home/pi/cuas_detect/" + f)
    sf.close()
    print(run(c, S + "systemctl stop fpv-detect fpv-camera cuas-detect-only 2>/dev/null; "
              + S + "systemctl reset-failed cuas-detect-only 2>/dev/null; "
              + S + "systemd-run --unit=cuas-detect-only --working-directory=/home/pi/cuas_detect "
              "-p StandardOutput=append:/boot/firmware/cuas_pkg/detect.log -p StandardError=append:/boot/firmware/cuas_pkg/detect.log "
              "/usr/bin/python3 -u /home/pi/cuas_detect/pi_fb_detect.py --fc none; "
              "sleep 12; systemctl is-active cuas-detect-only; grep -a fps /boot/firmware/cuas_pkg/detect.log | tail -2"))
    c.close()
    busy = sh(["powershell", "-NoProfile", "-Command", "(Get-NetUDPEndpoint -LocalPort 9500 -ErrorAction SilentlyContinue).OwningProcess"]).strip()
    if busy:
        print("GUI 이미 켜져 있음(9500 사용 중, PID", busy + ") — 그대로 사용")
    elif not a.no_gui:
        py = os.path.join(os.environ["USERPROFILE"], "anaconda3", "python.exe")
        subprocess.Popen([py, "-X", "utf8", "detect_gui.py", "--pi"], cwd=os.path.join(HERE, "gui"),
                         creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        print("GUI 켬 (드론 실시간 수신)")


if __name__ == "__main__":
    main()
