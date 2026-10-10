#!/bin/bash
source /usr/share/gazebo/setup.sh
export GAZEBO_PLUGIN_PATH=$HOME/drone_sim_build:$GAZEBO_PLUGIN_PATH
cd "$(dirname "$0")"
exec gzserver worlds/balloon_track.world
