#!/bin/bash
set -e
SRC="$(cd "$(dirname "$0")/plugins" && pwd)"
mkdir -p ~/drone_sim_build && cd ~/drone_sim_build
cmake "$SRC" -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j4 2>&1 | grep -E "error|warning: unused|Built target" || true
ls -la ~/drone_sim_build/*.so
