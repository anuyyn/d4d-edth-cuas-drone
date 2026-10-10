#!/bin/bash

cd "$(dirname "$0")"
source ~/arms-venv/bin/activate

echo '[*] 가상 X 디스플레이 시작...'
Xvfb :99 -screen 0 1280x720x24 > /dev/null 2>&1 &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

echo '[*] Gazebo GUI 시작...'
gazebo worlds/arms.world &
GZ_PID=$!
sleep 6

echo '[*] FFmpeg 녹화 시작...'
ffmpeg -f x11grab -video_size 1280x720 -framerate 30 -i :99 \
  -c:v libx264 -preset ultrafast -crf 20 \
  videos/arms_GUI_final.mp4 > /tmp/ffmpeg.log 2>&1 &
FFMPEG_PID=$!
sleep 2

echo '[*] A.R.M.S. 실행 중 (풍선 명중 대기)...'
timeout 40 python3 arms.py --auto --episodes 1 > /tmp/arms.log 2>&1

sleep 3

echo '[*] 프로세스 종료...'
kill $FFMPEG_PID $GZ_PID $XVFB_PID 2>/dev/null
wait 2>/dev/null

sleep 1

echo '[+] 완료!'
if [ -f videos/arms_GUI_final.mp4 ]; then
  ls -lh videos/arms_GUI_final.mp4
  echo '[✓] 풍선 명중 장면 포함 MP4 생성 완료!'
else
  echo '[-] 파일 생성 실패'
fi
