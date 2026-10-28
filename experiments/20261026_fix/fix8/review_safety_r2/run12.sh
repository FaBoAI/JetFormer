#!/bin/bash
# 反証 8-2 回目・係 safety の台本を 4 つの列で流す(10/28 08:40 の Jetson の再起動のあと。どれも 1 組 1 行で out/*.jsonl に足すので、
# 止まっても流し直せば済んだ組は飛ばして続ける)。装置に触れない・GPU なし・API なし(jf-heavy が始める前に GET で走行中かを見るだけ)。
#   bash experiments/20261026_fix/fix8/review_safety_r2/run12.sh   (~/jetformer で)
cd /home/jetson/jetformer || exit 1
D=experiments/20261026_fix/fix8/review_safety_r2
PY="/home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B"
export PYTHONDONTWRITEBYTECODE=1
w() {   # 列の名前, 台本と引数 ...
  local name=$1; shift
  while [ $# -gt 0 ]; do
    local script=$1 args=$2; shift 2
    local tag
    tag=$(echo "$script $args" | sed 's#\.py##; s#/#of#g; s# #_#g')
    $PY $D/$script $args >> $D/out/run_$tag.log 2>&1
  done
  echo "worker $name finished" >> $D/out/run12_workers.log
}
w W1 exp12.py "tb 0/3" idold12.py "0/2" exp12.py "hmh 0/2" &
w W2 exp12.py "tb 1/3" idold12.py "1/2" exp12.py "hmh 1/2" &
w W3 exp12.py "tb 2/3" exp12.py "hb 0/1" &
w W4 exp12.py "hm 0/1" exp12c.py "0/1" &
wait
echo "all finished $(date)" >> $D/out/run12_workers.log
