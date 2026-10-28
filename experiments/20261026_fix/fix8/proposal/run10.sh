#!/bin/bash
# fix8 係 safety-proposal: exp10.py のくみを P 本ずつ(既定 3。xargs の順番待ち)流す。
# 引数 = "くみ:形の並び"(形を省くと既定 = '-')。出力 out/exp10_<くみ>.json・.log
cd /home/jetson/jetformer
export D=experiments/20261026_fix/fix8/proposal
export PY=/home/jetson/miniconda/envs/jetracer/bin/python
export PYTHONDONTWRITEBYTECODE=1
P=${P:-3}
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c '
  kv="{}"; k=${kv%%:*}; v=${kv#*:}; [ "$v" = "$kv" ] && v=-
  /home/jetson/bin/jf-heavy --lock none --mem 1500M -- $PY -I $D/exp10.py $k $v > $D/out/exp10_$k.log 2>&1
  echo "done $k $(tail -1 $D/out/exp10_$k.log | cut -c1-200)"'
echo all done "$@"
