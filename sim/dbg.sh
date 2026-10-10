#!/bin/bash
cd "$(dirname "$0")" && mkdir -p logs
~/arms-venv/bin/python arms.py --nogui --auto --episodes ${2:-1} --guidance ${1:-PN} --log logs/arms_dbg.csv | tail -3
