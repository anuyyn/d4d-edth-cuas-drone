#!/bin/bash
cd "$(dirname "$0")" && mkdir -p logs
D=${2:-1}; EP=${3:-5}
for kp in $1; do
  python3 gz_loop.py --ctl p --kp $kp --episodes $EP --frames 540 --delay $D --log logs/gz_p_kp${kp}_d${D}.csv | tail -n 1
done
