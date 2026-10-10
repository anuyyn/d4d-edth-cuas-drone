#!/bin/bash
D="$(dirname "$0")"
bash "$D/run_gz_track.sh" ridge 180 --delay 0.3 --ridge ../sync_work/sim_offline/ridge_policy_aug.npz --tag _aug_d03
bash "$D/run_gz_track.sh" ridge 180 --delay 0.3 --tag _d03
