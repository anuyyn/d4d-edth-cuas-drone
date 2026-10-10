#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
export GAZEBO_MODEL_PATH=$PWD/models:$GAZEBO_MODEL_PATH
pkill -x gzserver; sleep 1
exec gzserver worlds/running_track.world
