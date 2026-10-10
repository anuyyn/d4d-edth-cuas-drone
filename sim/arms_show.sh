#!/bin/bash
cd "$(dirname "$0")"
source /usr/share/gazebo/setup.sh
pgrep -x gzserver >/dev/null || { echo "gzserver 없음: 다른 창에서 bash arms_server.sh 먼저"; exit 1; }
[ -f ~/.gazebo/gui.ini.bak ] || cp ~/.gazebo/gui.ini ~/.gazebo/gui.ini.bak 2>/dev/null
printf "[geometry]
x=1290
y=0
width=630
height=1000
" > ~/.gazebo/gui.ini
pgrep -x gzclient >/dev/null || { nohup gzclient > /tmp/gzclient.log 2>&1 & sleep 5; }
gz camera -c gzclient_camera -f interceptor >/dev/null 2>&1 &
exec ~/arms-venv/bin/python arms.py --guidance ${1:-PN}
