#!/bin/bash
cd "$(dirname "$0")" && mkdir -p videos
C=${1:-p}; S=${2:-60}
python3 gz_loop.py --ctl $C --episodes 1 --frames $((S*9)) --seed ${3:-2} --record videos/$C.mp4 --log logs/rec_$C.csv | tail -1
