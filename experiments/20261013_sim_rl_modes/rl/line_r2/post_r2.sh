#!/bin/bash
# 経路いろいろ r2 の学習のあと(2026-10-14、学習の段 line r2): 選び方 → 試験の組(1 回)→ 報告だけの組 → 第 1 回の方策の比べる値 → 図の軌跡 → 図。
#   /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 600 -- nice -n 15 bash rl/line_r2/post_r2.sh
set -eu
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/line/r2
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
log() { echo "[$(date -Is)] $*" | tee -a "$OUT/logs/post.log"; }
( cd /home/jetson/jetformer_reports/sim_rl_modes_20261013 && sha256sum -c protocol_line_r2.sha256 )
python3 rl/line_r2/select_line_r2.py "$OUT/main" --out "$OUT/selection.json" | tee -a "$OUT/logs/post.log"
POL=$(python3 -c "import json;print(json.load(open('$OUT/selection.json'))['selected']['policy'])")
log "選んだ方策: $POL"
# 試験の組(1 回だけ)。既にあれば走らせない(選び直さない)
if [ ! -f "$OUT/eval_test_r2.json" ]; then
  node rl/eval_modes.mjs --mode line --protocol line_r2 --set test --core original --jobs 2 --policy "$POL" --out "$OUT/eval_test_r2.json" 2>>"$OUT/logs/post.log" | tee -a "$OUT/logs/post.log"
fi
# 報告だけの組(新しい開始 120 回)
[ -f "$OUT/eval_fresh_r2.json" ] || node rl/eval_modes.mjs --mode line --protocol line_r2 --set fresh --core original --jobs 2 --policy "$POL" --out "$OUT/eval_fresh_r2.json" 2>>"$OUT/logs/post.log" | tee -a "$OUT/logs/post.log"
# 比べる値: 第 1 回の方策(meta.line_profile = v2 にした写し)を同じ試験の組で(選び方には使わない)
[ -f "$OUT/eval_test_r1policy_v2.json" ] || node rl/eval_modes.mjs --mode line --protocol line_r2 --set test --core original --jobs 2 --policy "$OUT/analysis/r1policy_as_v2.json" --out "$OUT/eval_test_r1policy_v2.json" 2>>"$OUT/logs/post.log" | tee -a "$OUT/logs/post.log"
# 図の軌跡(試験の組の全部、元のコア)
[ -f "$OUT/traj/traj_test_r2.json" ] || node rl/line_r2/line_r2_traj.mjs --policy "$POL" --out "$OUT/traj/traj_test_r2.json" --set test --all 2>/dev/null
[ -f "$OUT/traj/traj_test_r1policy_v2.json" ] || node rl/line_r2/line_r2_traj.mjs --policy "$OUT/analysis/r1policy_as_v2.json" --out "$OUT/traj/traj_test_r1policy_v2.json" --set test --all 2>/dev/null
STEP=$(python3 -c "import json;print(json.load(open('$OUT/selection.json'))['selected']['step'])")
python rl/line_r2/line_r2_figs.py curves "$OUT/main" --baseline "$OUT/analysis/val_r1policy_as_v2.json" --selected "$STEP"
python rl/line_r2/line_r2_figs.py target
python rl/line_r2/line_r2_figs.py traj "$OUT/traj/traj_test_r2.json"
python rl/line_r2/line_r2_figs.py start "$OUT/eval_test_r2.json" "$OUT/eval_test_r1policy_v2.json"
python rl/line_r2/line_r2_figs.py by_s "$OUT/traj/traj_test_r2.json" "$OUT/traj/traj_test_r1policy_v2.json"
log "終わり"
