import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WHEELS = r"C:\Users\user\cuas_ext\pi_wheels"
FILES = ["motion_id.py", "appearance_id.py", "detect_msg.py", "pi_fb_detect.py", "fc_link.py",
         "appearance/out/cuas_cls.onnx", "appearance/out/cuas_cls.json"]
HOOK = " systemd.run=/boot/firmware/cuas_pkg/install.sh systemd.run_success_action=reboot systemd.unit=kernel-command-line.target"

INSTALL = r'''#!/bin/bash
LOG=/boot/firmware/cuas_pkg/install.log
exec >>"$LOG" 2>&1
echo "==== $(date) 설치 시작"
sed -i 's| systemd.run=[^ ]*||; s| systemd.run_success_action=[^ ]*||; s| systemd.unit=kernel-command-line.target||' /boot/firmware/cmdline.txt
echo "cmdline: $(cat /boot/firmware/cmdline.txt)"
SRC=/boot/firmware/cuas_pkg
mkdir -p /home/pi/cuas
cp -r "$SRC/cuas/." /home/pi/cuas/
chown -R 1000:1000 /home/pi/cuas
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
DST=/usr/local/lib/python$PYV/dist-packages
echo "python $PYV -> $DST"
mkdir -p "$DST"
if python3 -c "import cv2, numpy" 2>/dev/null; then
  echo "OpenCV 이미 설치됨 - 패키지 설치 건너뜀"
else
python3 - "$SRC/wheels" "$DST" <<'PY'
import glob, os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
for w in sorted(glob.glob(os.path.join(src, "*.whl"))):
    zipfile.ZipFile(w).extractall(dst)
    print("unpacked", os.path.basename(w))
PY
find "$DST" -name "*.so*" -exec chmod 755 {} \;
fi
python3 -c "import cv2, numpy; print('check OK cv2', cv2.__version__, 'numpy', numpy.__version__)"
cd /home/pi/cuas && python3 -c "from appearance_id import AppearanceClassifier; import numpy as np; a=AppearanceClassifier(); print('model OK', sorted(a.probs(np.zeros((64,64,3),np.uint8)).items(), key=lambda kv: -kv[1])[0])"
echo "netplan: $(ls /etc/netplan 2>&1 | tr '\n' ' ')"
echo "NM 연결 파일: $(ls /etc/NetworkManager/system-connections 2>&1 | tr '\n' ' ')"
if [ -f "$SRC/wifi.nmconnection" ]; then
  install -m 600 -o root -g root "$SRC/wifi.nmconnection" /etc/NetworkManager/system-connections/cuas-wifi.nmconnection
  rm -f "$SRC/wifi.nmconnection"
  echo "와이파이 키파일 설치: $(grep '^ssid=' /etc/NetworkManager/system-connections/cuas-wifi.nmconnection)"
fi
cp "$SRC/fpv-detect.service" /etc/systemd/system/fpv-detect.service
systemctl disable fpv-camera.service
systemctl enable fpv-detect.service
ls -l /etc/systemd/system/multi-user.target.wants/ | grep fpv
echo "==== $(date) 설치 끝 (재부팅)"
sync
exit 0
'''

SERVICE = '''[Unit]
Description=CUAS detection, boxes drawn onto composite output (goggles)
After=multi-user.target
StartLimitIntervalSec=120
StartLimitBurst=4
OnFailure=fpv-camera.service

[Service]
WorkingDirectory=/home/pi/cuas
ExecStart=/usr/bin/python3 -u /home/pi/cuas/pi_fb_detect.py
Restart=on-failure
RestartSec=3
StandardOutput=append:/boot/firmware/cuas_pkg/detect.log
StandardError=append:/boot/firmware/cuas_pkg/detect.log

[Install]
WantedBy=multi-user.target
'''


WIFI = """[connection]
id=cuas-wifi
type=wifi
autoconnect=true
autoconnect-priority=10

[wifi]
mode=infrastructure
ssid={ssid}

[wifi-security]
key-mgmt=wpa-psk
psk={psk}

[ipv4]
method=auto

[ipv6]
method=auto
"""


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "D:/"
    wifi = sys.argv[2] if len(sys.argv) > 2 else None
    if not (os.path.exists(os.path.join(D, "cmdline.txt")) and os.path.exists(os.path.join(D, "bcm2710-rpi-zero-2-w.dtb"))):
        sys.exit(f"{D} 는 파이 SD카드 부트 파티션이 아님")
    pkg = os.path.join(D, "cuas_pkg")
    os.makedirs(os.path.join(pkg, "cuas", "appearance", "out"), exist_ok=True)
    os.makedirs(os.path.join(pkg, "wheels"), exist_ok=True)
    for f in FILES:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(pkg, "cuas", f))
    for w in os.listdir(WHEELS):
        if w.endswith(".whl"):
            shutil.copyfile(os.path.join(WHEELS, w), os.path.join(pkg, "wheels", w))
    if wifi:
        ssid, psk = wifi.split(":", 1)
        with open(os.path.join(pkg, "wifi.nmconnection"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(WIFI.format(ssid=ssid, psk=psk))
        print("와이파이 키파일 넣음:", ssid)
    args = os.path.join(pkg, "args.txt")
    if not os.path.exists(args):
        with open(args, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("--target any --follow yaw\n")
    for name, txt in (("install.sh", INSTALL), ("fpv-detect.service", SERVICE)):
        with open(os.path.join(pkg, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    cmd = os.path.join(D, "cmdline.txt")
    line = open(cmd, encoding="utf-8").read().strip().splitlines()[0]
    bak = os.path.join(D, "cmdline.txt.bak_before_cuas")
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    if "systemd.run=" not in line:
        line += HOOK
    line = " ".join(t for t in line.split() if not t.startswith(("console=serial0", "console=ttyAMA0", "console=ttyS0")))
    cfg = os.path.join(D, "config.txt")
    txt = open(cfg, encoding="utf-8").read()
    add = [k for k in ("enable_uart=1", "dtoverlay=disable-bt") if k not in txt]
    if add:
        if not os.path.exists(cfg + ".bak_before_cuas"):
            shutil.copyfile(cfg, cfg + ".bak_before_cuas")
        with open(cfg, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n[all]\n# CUAS: FC UART(MSP)\n" + "\n".join(add) + "\n")
        print("config.txt 추가:", add)
    with open(cmd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print("cmdline.txt:", open(cmd, encoding="utf-8").read().strip())
    for r, _, fs in os.walk(pkg):
        for f in fs:
            p = os.path.join(r, f)
            print(f"{os.path.getsize(p):>10}  {p}")


if __name__ == "__main__":
    main()
