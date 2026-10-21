#!/bin/bash
# Fix-drive: V-drive の記録の再生(verify_drive/v_replay.py・v_compare.py、変えずに使う)を 2 本(007・101)で回し直す。
# 変更前の木(git archive 30b71e8)と、直したあとの今の木を比べる。torch なし・装置なし
#   jf-heavy --lock none --mem 1500M -- bash replay_subset.sh <変更前の木> <置き場>
set -u
OLD="$1"; OUT="$2"; V=/home/jetson/jetformer/experiments/20261021_bench_prep/verify_drive
PY=/home/jetson/miniconda/envs/jetracer/bin/python
NEW=/home/jetson/jetformer
mkdir -p "$OUT"
cd "$V"
for log in 20261010-173445 20261010-204605; do
  for var in plain perturb; do
    $PY -I v_replay.py --tree "$OLD" --log $log --variant $var --out "$OUT/old_${log}_${var}.pkl"
    $PY -I v_replay.py --tree "$NEW" --log $log --variant $var --zone 0 --out "$OUT/new_${log}_${var}_z0.pkl"
    $PY -I v_replay.py --tree "$NEW" --log $log --variant $var --zone 1 --out "$OUT/new_${log}_${var}_z1.pkl"
  done
  $PY -I v_replay.py --tree "$NEW" --log $log --variant toggle --zone 0 --out "$OUT/new_${log}_toggle.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant plain --zone 1 --pca-fail 200,400,600,800,1000,1200,1400,1600 --out "$OUT/new_${log}_plain_z1_pcafail.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant perturb --zone 0 --zone-fail-at 300 --out "$OUT/new_${log}_perturb_z0_zfail.pkl"
  $PY -I v_replay.py --tree "$NEW" --log $log --variant perturb --zone 1 --zone-fail-at 300 --out "$OUT/new_${log}_perturb_z1_zfail.pkl"
  for var in plain perturb; do
    echo "== off $log $var"; $PY -I v_compare.py off "$OUT/old_${log}_${var}.pkl" "$OUT/new_${log}_${var}_z0.pkl"
    echo "== on $log $var"; $PY -I v_compare.py on "$OUT/new_${log}_${var}_z0.pkl" "$OUT/new_${log}_${var}_z1.pkl"
  done
  echo "== toggle $log"; $PY -I v_compare.py toggle "$OUT/old_${log}_plain.pkl" "$OUT/new_${log}_toggle.pkl"
done
echo ALLDONE
