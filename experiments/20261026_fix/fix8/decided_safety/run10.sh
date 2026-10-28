#!/bin/bash
# fix8 係 safety-decided: exp10 のグリッドを 2 本の列で流す(どれも jf-heavy --lock none --mem 1500M)
# 使い方: run10.sh A|B
set -u
cd /home/jetson/jetformer/experiments/20261026_fix/fix8/decided_safety
PY=/home/jetson/miniconda/envs/jetracer/bin/python
export PYTHONDONTWRITEBYTECODE=1
case "$1" in
  A) jobs="plain drop:0/2 glitch tof";;
  B) jobs="drop:1/2 vskip burst fuzz8101 fuzz8102";;
  *) echo "A か B"; exit 2;;
esac
for j in $jobs; do
  g=${j%%:*}; part=0/1
  [ "$g" != "$j" ] && part=${j#*:}
  tag="ds_${1}_${g}"
  echo "== $(date +%T) $g $part"
  DS_TAG=$tag /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $PY -I -B exp10.py "$g" "$part" > "out/exp10_${g}_${part//\//of}.log" 2>&1
  tail -1 "out/exp10_${g}_${part//\//of}.log"
done
echo "== $(date +%T) end $1"
