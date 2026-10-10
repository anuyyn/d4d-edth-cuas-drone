import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["pi_fb_detect.py", "fc_link.py", "motion_id.py", "appearance_id.py", "detect_msg.py",
         "appearance/out/cuas_cls.onnx", "appearance/out/cuas_cls.json"]
HOOK = " systemd.run=/boot/firmware/cuas_pkg/install.sh systemd.run_success_action=reboot systemd.unit=kernel-command-line.target"

INSTALL = r'''#!/bin/bash
LOG=/boot/firmware/cuas_pkg/install.log
exec >>"$LOG" 2>&1
echo "==== $(date) 탐지 고정본 설치 시작"
sed -i 's| systemd.run=[^ ]*||; s| systemd.run_success_action=[^ ]*||; s| systemd.unit=kernel-command-line.target||' /boot/firmware/cmdline.txt
SRC=/boot/firmware/cuas_pkg
mkdir -p /home/pi/cuas
cp -r "$SRC/cuas/." /home/pi/cuas/
chown -R 1000:1000 /home/pi/cuas
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
DST=/usr/local/lib/python$PYV/dist-packages
if python3 -c "import cv2, numpy" 2>/dev/null; then
  echo "OpenCV 이미 설치됨"
else
  mkdir -p "$DST"
  python3 - "$SRC/wheels" "$DST" <<'PY'
import glob, os, sys, zipfile
for w in sorted(glob.glob(os.path.join(sys.argv[1], "*.whl"))):
    zipfile.ZipFile(w).extractall(sys.argv[2]); print("unpacked", os.path.basename(w))
PY
  find "$DST" -name "*.so*" -exec chmod 755 {} \;
fi
python3 -c "import cv2, numpy; print('check OK cv2', cv2.__version__)"
mkdir -p /var/log/journal
systemd-tmpfiles --create --prefix /var/log/journal 2>/dev/null
echo "journald 영구 저장 켬"
nmcli -g 802-11-wireless.powersave con show cuas-wifi 2>/dev/null
ls /etc/NetworkManager/system-connections/
cp "$SRC/fpv-detect.service" /etc/systemd/system/fpv-detect.service
systemctl disable fpv-camera.service 2>/dev/null
systemctl enable fpv-detect.service
echo "==== $(date) 설치 끝 (재부팅)"
sync
exit 0
'''

SERVICE = '''[Unit]
Description=CUAS detection only (no flight commands), boxes onto composite output (goggles)
After=multi-user.target
StartLimitIntervalSec=120
StartLimitBurst=4
OnFailure=fpv-camera.service

[Service]
WorkingDirectory=/home/pi/cuas
ExecStart=/usr/bin/python3 -u /home/pi/cuas/pi_fb_detect.py --fc none
Restart=on-failure
RestartSec=3
StandardOutput=append:/boot/firmware/cuas_pkg/detect.log
StandardError=append:/boot/firmware/cuas_pkg/detect.log

[Install]
WantedBy=multi-user.target
'''


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "D:/"
    if not (os.path.exists(os.path.join(D, "cmdline.txt")) and os.path.exists(os.path.join(D, "bcm2710-rpi-zero-2-w.dtb"))):
        sys.exit(f"{D} 는 파이 SD카드 부트 파티션이 아님")
    pkg = os.path.join(D, "cuas_pkg")
    os.makedirs(os.path.join(pkg, "cuas", "appearance", "out"), exist_ok=True)
    for f in FILES:
        shutil.copyfile(os.path.join(HERE, f.replace("/", os.sep)), os.path.join(pkg, "cuas", f.replace("/", os.sep)))
    if not os.path.isdir(os.path.join(pkg, "wheels")):
        print("주의: SD에 wheels 폴더가 없음(파이에 OpenCV 가 이미 있으면 괜찮음)")
    for name, txt in (("install.sh", INSTALL), ("fpv-detect.service", SERVICE)):
        with open(os.path.join(pkg, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    cmd = os.path.join(D, "cmdline.txt")
    line = open(cmd, encoding="utf-8").read().strip().splitlines()[0]
    if "systemd.run=" not in line:
        line += HOOK
    with open(cmd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print("SD 기록 완료 — 탐지 고정본(--fc none), 설치 훅 추가")


if __name__ == "__main__":
    main()
