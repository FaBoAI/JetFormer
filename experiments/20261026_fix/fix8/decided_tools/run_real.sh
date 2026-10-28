#!/bin/bash
# fix8 係 tools-decided: 本物の記録 10/10・10/23・10/24 で ⑤③④② を直す前と今で流す。読むだけ。
#   bash run_real.sh old|new
set -u
TAG=$1
HERE=/home/jetson/jetformer/experiments/20261026_fix/fix8/decided_tools
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
cd "$HERE" || exit 1
for d in 20261010 20261023 20261024; do
  for v in "" "--first-only" "--ignore-gain"; do
    n=${v#--}; n=${n:-plain}
    PYTHONDONTWRITEBYTECODE=1 $H $PY real_one.py "$TAG" c5 "${d}_$n" --day $d $v
  done
  PYTHONDONTWRITEBYTECODE=1 $H $PY real_one.py "$TAG" s3qp $d
  PYTHONDONTWRITEBYTECODE=1 $H $PY real_one.py "$TAG" runs $d
done
PYTHONDONTWRITEBYTECODE=1 $H $PY real_one.py "$TAG" c5 default
echo "$TAG done"
