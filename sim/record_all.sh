#!/bin/bash
cd "$(dirname "$0")"
pkill -f "^python3 gz_loop"; sleep 4
pgrep -x gzserver >/dev/null || { echo "gzserver 없음"; exit 1; }
bash record.sh none 60 2
bash record.sh p 60 2
ls -la videos
