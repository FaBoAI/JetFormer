#!/bin/bash
# C-DIV の種 1 の学習が実際に GPU で動き始めてから(lock big を取ってから)、Q2 の予測(GPU)を lock big の順番待ちに入れる。
# 学習が始まる前に入れると、学習の道具(diverse_train.py)が「GPU を使う処理がある」と見て種 1 を始めないため。
# 学習が終わると lock が空き、この予測が動く(道具は q2_eval の処理が終わるまで次の S-MOD を始めない)。
# 種 1 が済んでいれば(result.json が done)すぐに予測する。予測が全部そろうまでくり返す(学習が途中で止まって続きからになったときも)。
set -u
REPO=/home/jetson/jetformer
RUN=$REPO/data/runs/simdiv_20261015-CDIV-s1
Z=$REPO/data/zoo/sim_diverse_train_20261015
LOG=/home/jetson/jetformer_reports/sim_diverse_train_20261015/q2/logs/gpu_s1.log
PY=/home/jetson/miniconda/envs/jetracer/bin/python
done_all() {
  for f in pred/CDIV_s1.npz pred/CDIV_s1_last.npz pert/CDIV_s1.npz sim/CDIV_s1.npz sim5/CDIV_s1.npz; do
    [ -s "$Z/$f" ] || return 1
  done
  return 0
}
echo "[$(date +%T)] 種 1 の学習が GPU で動き始めるのを待ちます" >> "$LOG"
n=0
while ! done_all; do
  if grep -q '"state": "done"' "$RUN/result.json" 2>/dev/null \
     || pgrep -f "^$PY -u tools/train.py .*simdiv_20261015-CDIV-s1" >/dev/null; then
    echo "[$(date +%T)] 種 1 の学習が動いている(または済んだ)ので、予測を lock big の順番待ちに入れます" >> "$LOG"
    cd "$REPO" && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 \
      "$PY" -u "$REPO/experiments/20261015_sim_diverse_train/q2/q2_eval.py" gpu >> "$LOG" 2>&1
    echo "[$(date +%T)] 予測の終了コード $?" >> "$LOG"
    n=$((n + 1))
    [ $n -ge 6 ] && { echo "[$(date +%T)] 6 回やってもそろわないので止めます" >> "$LOG"; exit 1; }
    sleep 30
  else
    sleep 30
  fi
done
echo "[$(date +%T)] 種 1 の予測がそろいました" >> "$LOG"
