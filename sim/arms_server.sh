#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
pkill -x gzserver; sleep 1
exec gzserver worlds/arms.world
