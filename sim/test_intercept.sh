#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
pkill -x gzserver; sleep 1
gzserver worlds/arms.world > /tmp/gzs_test.log 2>&1 &
sleep 6
timeout ${1:-120} ~/arms-venv/bin/python arms.py --nogui --episodes 0 --timeout 999 ${2}
pkill -x gzserver
