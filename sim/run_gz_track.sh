#!/bin/bash
POL=${1:-ridge}; SEC=${2:-120}; shift 2
cd "$(dirname "$0")"
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=$PWD/models:$GAZEBO_MODEL_PATH
pkill -x gzserver; sleep 1
gzserver worlds/running_track.world > /tmp/gz_track_server.log 2>&1 &
sleep 6
~/arms-venv/bin/python gz_track.py --policy "$POL" --seconds "$SEC" "$@"
pkill -x gzserver
grep -i -E "error|track\]" /tmp/gz_track_server.log | head -5
