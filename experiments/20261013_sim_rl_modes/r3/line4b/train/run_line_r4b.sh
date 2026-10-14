#!/bin/bash
# 経路いろいろ r4b の学習(2026-10-14 夜、line r4 の段のやり直し。protocol_line_r4b.json training のとおり)。
#   /home/jetson/bin/jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15 bash /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line4b/train/run_line_r4b.sh
# lock rl を取ってから動く(jf-heavy の中で nice 15。子プロセスも nice 15)。止めたあと(電源断を含む)も同じコマンドで last.pt から続く。
# 50 万ステップごとに検証の組 LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-VG30R40-R4B + LN-FAM6
# (高速版・シールド v5b、eval_line_r4b.mjs、2 本)を裏で評価する。早めの終わりは使わない(400 万まで学ぶ)。
# 学習のあと、評価の残ったチェックポイントと、選びの候補の r4 の 300 万・400 万(r3/line_r4/train/main/policies)を同じ検証の組で評価する。
set -u
REP=/home/jetson/jetformer_reports/sim_rl_modes_20261013
OUT=$REP/r3/line_r4b/train
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
L=$EXP/r3/line4b
STEPS=${STEPS:-4000000}
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 JF_TMP=/tmp
mkdir -p "$OUT/logs"
log() { echo "[$(date -Is)] $*" >> "$OUT/logs/chain.log"; }
log "lock rl を取りました(pid $$、nice $(nice))"
( cd "$REP/r3" && sha256sum -c protocol_line_r4b.sha256 ) >> "$OUT/logs/chain.log" 2>&1 || { log "protocol_line_r4b.json が固めたものと違うので止めます"; exit 3; }
if [ ! -f "$OUT/main/done.json" ]; then
  python -u "$L/train/train_line_r4b.py" --run line_r4b --steps "$STEPS" --out "$OUT/main" --seed 47 --init line_r4_4M --eval --tag line-r4b \
    --lr-start 3e-5 --lr-end 1.5e-5 --ckpt-every 500000 --save-every 250000 >> "$OUT/logs/main.log" 2>&1
  log "学習 終了コード $?"
else
  log "学習はもう終わっています"
fi
# 評価の残ったチェックポイントを評価する(1 つずつ、高速版、2 本)
mkdir -p "$OUT/main/evals/line" "$OUT/r4_ckpt/evals"
for p in "$OUT"/main/policies/step_*.json; do
  [ -f "$p" ] || continue
  e="$OUT/main/evals/line/$(basename "$p")"
  if [ ! -s "$e" ]; then
    log "残りの検証: $(basename "$p")"
    node "$L/eval_line_r4b.mjs" --set validation --core fast --jobs 2 --policy "$p" --out "$e" > "${e%.json}.log" 2>&1 || log "検証が失敗: $p"
  fi
done
# 選びの候補の r4 のチェックポイント(protocol_line_r4b.json selection.candidates)
for s in 003014656 004014080; do
  p=$REP/r3/line_r4/train/main/policies/step_$s.json
  e="$OUT/r4_ckpt/evals/r4_step_$s.json"
  if [ ! -s "$e" ]; then
    log "r4 のチェックポイントの検証: step_$s"
    node "$L/eval_line_r4b.mjs" --set validation --core fast --jobs 2 --policy "$p" --out "$e" > "${e%.json}.log" 2>&1 || log "検証が失敗: $p"
  fi
done
log "終わり"
