#!/bin/bash
cd "$(dirname "$0")" && mkdir -p logs
EP=${1:-3}; FR=${2:-540}; D=${3:-1}
for c in none p; do
  python3 gz_loop.py --ctl $c --episodes $EP --frames $FR --delay $D --log logs/gz_${c}_d${D}.csv | tail -n $((EP+1))
done
