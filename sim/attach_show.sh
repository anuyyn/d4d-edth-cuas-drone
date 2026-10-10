#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
pkill -x gzclient; pkill -x gzserver; sleep 1
gzserver --verbose worlds/attach.world > /tmp/gzs_attach.log 2>&1 &
for i in $(seq 60); do grep -aq 'arms\] loaded' /tmp/gzs_attach.log && break; sleep 1; done
[ -f ~/.gazebo/gui.ini.bak ] || cp ~/.gazebo/gui.ini ~/.gazebo/gui.ini.bak 2>/dev/null
printf "[geometry]\nx=0\ny=0\nwidth=1600\nheight=900\n" > ~/.gazebo/gui.ini
gzclient > /tmp/gzclient.log 2>&1 &
sleep 8
gz camera -c gzclient_camera -f interceptor >/dev/null 2>&1 &
~/arms-venv/bin/python attach_sim.py --seeds ${1:-1-10} --tail 6
echo "끝. Gazebo 창은 그대로 둠 (닫으려면 창 닫기)"
wait
