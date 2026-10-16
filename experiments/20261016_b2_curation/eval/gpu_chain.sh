#!/bin/bash
# B2 の評価の GPU の段を順に動かす(crosscheck が済むのを待って → 予測 → エンジン)。もう一度動かせば続きから
cd /home/jetson/jetformer
PY=/home/jetson/miniconda/envs/jetracer/bin/python
GPU="/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10"
while pgrep -f "b2_eval.py cross[c]heck" >/dev/null || pgrep -f "train_b2.py ru[n]" >/dev/null; do sleep 20; done
echo "[$(date +%T)] 予測を始めます"
$GPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py predict || exit 1
echo "[$(date +%T)] エンジンを始めます"
$GPU $PY -u experiments/20261016_b2_curation/eval/b2_eval.py engine || exit 1
echo "[$(date +%T)] 済み"
