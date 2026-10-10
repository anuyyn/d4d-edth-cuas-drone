#!/bin/bash
cd "$(dirname "$0")"
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
pgrep -x gzserver >/dev/null || { echo "gzserver 없음: 다른 창에서 bash server.sh 먼저"; exit 1; }
pgrep -x gzclient >/dev/null || { nohup gzclient > /tmp/gzclient.log 2>&1 & sleep 4; }
python3 gz_loop.py --ctl ${1:-p} --kp ${2:-0.2} --episodes 20 --frames 540 --show --realtime --log logs/show.csv
