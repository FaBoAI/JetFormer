#!/bin/bash
# 段 3(v2h-aug2-e60-s0〜s3)の予測と C3 を回す(2026-10-26 02:2x。D3 の係)。
# GPU の仕事は 1 回ずつ jf-heavy --lock big(別の係の run_c3_all_20261026.sh と順番に分け合う。あいだ 60 秒)。
# ① eval_v2.py predict(H1・介入の best・last。キューが出した予測があれば読み直して確かめるだけ)を 1 本ずつ
# ② desk_v2.py predict --what shift(D3 の robust_dl1_us)→ report --out(別の場所。desk/desk_v2.json には書かない)
# ③ desk_v2.py predict --what tab → ④ trt 2 回 → report --out
# report の出口: $OUTD/desk_v2.json → /home/jetson/jetformer_reports/train_v2_20261025/desk_v2_stage3.json に写す
#   cd ~/jetformer && setsid nohup bash experiments/20261025_train_v2/c3/run_stage3_d3_20261026.sh \
#       > experiments/20261025_train_v2/c3/logs/run_stage3_d3_20261026.log 2>&1 < /dev/null &
cd /home/jetson/jetformer || exit 1
C=experiments/20261025_train_v2/c3
P=/home/jetson/miniconda/envs/jetracer/bin/python
L=$C/logs
REP=/home/jetson/jetformer_reports/train_v2_20261025
OUTD=${OUTD:-/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/desk_stage3_out}
DONE=$L/run_stage3_d3_20261026.done
rm -f "$DONE"
N=(v2_20261025-v2h-aug2-e60-s0 v2_20261025-v2h-aug2-e60-s1 v2_20261025-v2h-aug2-e60-s2 v2_20261025-v2h-aug2-e60-s3)
ts() { date '+%F %T'; }
G=(/home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 "$P" -u)
report() {   # $1 = 段の名前
  echo "[$(ts)] report --out $OUTD ($1)"
  /home/jetson/bin/jf-heavy --lock none --mem 2500M --need-mb 1500 -- "$P" -I $C/desk_v2.py report --out "$OUTD" >> $L/report_stage3_d3.log 2>&1
  local rc=$?
  echo "[$(ts)] report rc=$rc $(tail -1 $L/report_stage3_d3.log)"
  if [ $rc -eq 0 ]; then
    cp "$OUTD/desk_v2.json" "$REP/desk_v2_stage3.json.tmp" && mv "$REP/desk_v2_stage3.json.tmp" "$REP/desk_v2_stage3.json"
    cp "$OUTD/desk_v2.md" "$REP/desk_v2_stage3.md.tmp" && mv "$REP/desk_v2_stage3.md.tmp" "$REP/desk_v2_stage3.md"
    echo "[$(ts)] 写しました: $REP/desk_v2_stage3.json($1)"
    date '+%F %T' > "$L/run_stage3_d3_20261026.$1.done"
  fi
}
for n in "${N[@]}"; do
  echo "[$(ts)] eval_v2 predict 開始: $n"
  "${G[@]}" experiments/20261025_train_v2/eval_v2.py predict --runs "$n" >> "$L/evalpred_$n.log" 2>&1
  echo "[$(ts)] eval_v2 predict rc=$? $(tail -2 "$L/evalpred_$n.log" | tr '\n' ' ')"
  sleep 60
done
bash $C/run_c3_gpu.sh shift "${N[@]}"
report shift
bash $C/run_c3_gpu.sh tab "${N[@]}"
report tab
bash $C/run_c3_gpu.sh trt "${N[@]}"
report all
echo "[$(ts)] 段 3 の C3 おわり"
date '+%F %T' > "$DONE"
