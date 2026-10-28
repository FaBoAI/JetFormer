#!/bin/bash
# fix8 係 tools 直す 8-2 回目: 本物の記録 10/10・10/23・10/24 で ⑤ を直す前と今で流す(読むだけ)。
#   bash run_real.sh old|new
set -u
TAG=$1
HERE=/home/jetson/jetformer/experiments/20261026_fix/fix8/fix_tools_r2/real
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
cd "$HERE" || exit 1
for d in 20261010 20261023 20261024; do
  PYTHONDONTWRITEBYTECODE=1 $H $PY -B real_one.py "$TAG" "${d}_plain" --day $d
  PYTHONDONTWRITEBYTECODE=1 $H $PY -B real_one.py "$TAG" "${d}_first-only" --day $d --first-only
  PYTHONDONTWRITEBYTECODE=1 $H $PY -B real_one.py "$TAG" "${d}_ignore-gain" --day $d --ignore-gain
  PYTHONDONTWRITEBYTECODE=1 $H $PY -B real_one.py "$TAG" "${d}_ignore-gain_first-only" --day $d --ignore-gain --first-only
done
PYTHONDONTWRITEBYTECODE=1 $H $PY -B real_one.py "$TAG" default
echo "$TAG done"
