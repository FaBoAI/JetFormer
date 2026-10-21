#!/bin/bash
# V-drive: 10/10 の AI の記録 8 本 × 乱れ方で、変更前の木と今の木を回す(torch なし・装置なし)
set -u
OLD="$1"; OUT="$2"; V=/home/jetson/jetformer/experiments/20261021_bench_prep/verify_drive
PY=/home/jetson/miniconda/envs/jetracer/bin/python
NEW=/home/jetson/jetformer
mkdir -p "$OUT"
cd "$V"
LOGS="20261010-172910 20261010-173445 20261010-203924 20261010-204208 20261010-204605 20261010-171639 20261010-172505 20261010-181504"
for log in $LOGS; do
  for var in plain perturb dry; do
    $PY -I v_replay.py --tree "$OLD" --log $log --variant $var --out "$OUT/old_${log}_${var}.pkl"
    $PY -I v_replay.py --tree "$NEW" --log $log --variant $var --zone 0 --out "$OUT/new_${log}_${var}_z0.pkl"
  done
  $PY -I v_replay.py --tree "$NEW" --log $log --variant dry --zone 1 --out "$OUT/new_${log}_dry_z1.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant plain --zone 1 --out "$OUT/new_${log}_plain_z1.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant perturb --zone 1 --out "$OUT/new_${log}_perturb_z1.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant toggle --zone 0 --out "$OUT/new_${log}_toggle.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant perturb --zone 1 --zone-fail-at 300 --out "$OUT/new_${log}_perturb_z1_zfail.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant perturb --zone 1 --imu-fail-at 2000 --out "$OUT/new_${log}_perturb_z1_ifail.pkl"
done
echo ALLDONE
