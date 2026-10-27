#!/bin/bash
# fix7 係 tools 直す 7-1 回目(2026-10-27): 本物の記録 10/10・10/23・10/24 で ⑤ collect.py を、直す前(before/collect.py.orig)と今で流して
# 比べる。読むだけ。書くのは作業場所(out/real_<old|new>)だけ(data/raw・data/labels・diag_launch/out には書かない)。GPU なし。
#   bash run_real.sh <old|new>
set -u
TAG=$1
JF=/home/jetson/jetformer
HERE=$JF/experiments/20261026_fix/fix7/tools_r1
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
O=$HERE/out/real_$TAG
mkdir -p "$O"
if [ "$TAG" = old ]; then COL=$HERE/before/collect.py.orig; else COL=$JF/experiments/20261026_fix/diag_launch/collect.py; fi
cd "$O" || exit 1
for d in 20261010 20261023 20261024; do
  for v in "" "--first-only" "--ignore-gain"; do
    n=${v#--}; n=${n:-plain}
    PYTHONDONTWRITEBYTECODE=1 $H $PY -I -B "$COL" --day $d $v --out "$O/launches_${d}_$n.json" > "$O/launches_${d}_$n.stdout" 2> "$O/launches_${d}_$n.stderr"
    echo "$TAG collect $d $n rc=$?"
  done
done
PYTHONDONTWRITEBYTECODE=1 $H $PY -I -B "$COL" --out "$O/launches.json" > "$O/launches.stdout" 2> "$O/launches.stderr"; echo "$TAG collect default rc=$?"
echo "$TAG done"
