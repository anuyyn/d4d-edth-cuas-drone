#!/bin/bash
cd "$(dirname "$0")"
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
pgrep -x gzserver >/dev/null || { gzserver worlds/arms.world > /tmp/gzserver.log 2>&1 & sleep 6; }
[ -f ~/.gazebo/gui.ini.bak ] || cp ~/.gazebo/gui.ini ~/.gazebo/gui.ini.bak 2>/dev/null
printf "[geometry]\nx=1290\ny=0\nwidth=630\nheight=1000\n" > ~/.gazebo/gui.ini
pgrep -x gzclient >/dev/null || { nohup gzclient > /tmp/gzclient.log 2>&1 & sleep 5; }
gz camera -c gzclient_camera -f interceptor >/dev/null 2>&1 &
echo "WSL IP: $(hostname -I)   (관제 로그는 자동으로 찾음)"
exec ~/arms-venv/bin/python arms.py --guidance PN --plots 0 --auto "$@"
