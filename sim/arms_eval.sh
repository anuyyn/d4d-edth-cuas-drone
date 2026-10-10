#!/bin/bash
cd "$(dirname "$0")"
for g in PN PP; do ~/arms-venv/bin/python arms.py --nogui --auto --episodes ${1:-10} --guidance $g | tail -n $(( ${1:-10} + 1 )); done
