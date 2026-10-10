#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
pkill -x gzserver; sleep 1
gzserver worlds/arms.world > /tmp/gzs_rec.log 2>&1 &
sleep 6
timeout 200 ~/arms-venv/bin/python arms.py --nogui --auto --seed ${1:-1} --cmd-port 0 --record videos/demo_arms_raw.mp4 --log logs/demo_arms.csv
pkill -x gzserver
