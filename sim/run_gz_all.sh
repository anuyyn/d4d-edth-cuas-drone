#!/bin/bash
D="$(dirname "$0")"
bash "$D/run_gz_track.sh" ridge 180
bash "$D/run_gz_track.sh" zero 30
bash "$D/run_gz_track.sh" geom 60
