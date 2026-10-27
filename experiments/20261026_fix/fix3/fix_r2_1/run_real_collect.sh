#!/bin/bash
# 直す 2 回目・1 組目(2026-10-27): 本物の記録 10/23・10/24 で ⑤ collect.py を、直す前(before/collect.py.orig)と後で流して比べる。読むだけ。
#   bash run_real_collect.sh <作業場所(消してよい)>
# 作業場所にだけ書く(data/raw・data/labels・diag_launch/out には書かない。--out で作業場所へ)。GPU なし。
set -u
W=$1
JF=/home/jetson/jetformer
B=$JF/experiments/20261026_fix/fix3/fix_r2_1/before
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
NEW=$JF/experiments/20261026_fix/diag_launch/collect.py
mkdir -p "$W/old" "$W/out_old" "$W/out_new"
cp "$B/collect.py.orig" "$W/old/collect.py"
for tag in old new; do
  if [ $tag = old ]; then COL=$W/old/collect.py; else COL=$NEW; fi
  O=$W/out_$tag
  for d in 20261023 20261024; do
    $H $PY -I "$COL" --day $d --out "$O/launches_$d.json" > "$O/launches_$d.stdout" 2> "$O/launches_$d.stderr"; echo "$tag $d rc=$?"
    $H $PY -I "$COL" --day $d --first-only --out "$O/launches_${d}_first.json" > "$O/launches_${d}_first.stdout" 2> "$O/launches_${d}_first.stderr"; echo "$tag $d first rc=$?"
  done
  $H $PY -I "$COL" --out "$O/launches.json" > "$O/launches.stdout" 2> "$O/launches.stderr"; echo "$tag default rc=$?"
done
for d in 20261023 20261024; do
  $H $PY -I "$NEW" --day $d --ignore-gain --out "$W/out_new/launches_${d}_ig.json" > "$W/out_new/launches_${d}_ig.stdout" 2>&1; echo "new $d ignore-gain rc=$?"
done
