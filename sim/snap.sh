#!/bin/bash
cd "$(dirname "$0")"
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=$PWD/models:$GAZEBO_MODEL_PATH
pkill -x gzserver; sleep 1
gzserver worlds/running_track.world > /tmp/gz_snap.log 2>&1 &
~/arms-venv/bin/python gz_track.py --policy zero --snap "$@"
pkill -x gzserver
