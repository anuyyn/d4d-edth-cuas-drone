#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
pkill -x gzserver; sleep 1
gzserver --verbose worlds/attach.world > /tmp/gzs_attach.log 2>&1 &
for i in $(seq 60); do grep -aq 'arms\] loaded' /tmp/gzs_attach.log && break; sleep 1; done
sleep 3
timeout 900 ~/arms-venv/bin/python attach_sim.py "$@"
grep -a "TAG ATTACHED\|HIT" /tmp/gzs_attach.log | tail -5
pkill -x gzserver
