#!/bin/bash
cd "$(dirname "$0")" && mkdir -p videos
G=${1:-PN}; S=${2:-1}
~/arms-venv/bin/python arms.py --nogui --auto --guidance $G --seed $S --record videos/arms_${G}_seed${S}.mp4 --log logs/rec_arms_${G}_${S}.csv
ls -la videos/arms_${G}_seed${S}.mp4
