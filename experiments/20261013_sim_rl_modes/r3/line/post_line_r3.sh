#!/bin/bash
# 経路いろいろ r3 の学習のあと(2026-10-14、r3 の学習の段 line): 選び方 → 試験の組(1 回)→ 報告だけの組 → 比べる値 → 代わりの方法(不合格のとき)→ 書き出し → 図。
#   /home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 700 -- nice -n 15 bash /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line/post_line_r3.sh
# 結果のファイルがあればその手順は飛ばす(試験の組を 2 回走らせない・選び直さない)。
set -eu
REP=/home/jetson/jetformer_reports/sim_rl_modes_20261013
O=$REP/r3/line_r3
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
L=$EXP/r3/line
JOBS=${JOBS:-3}
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export JF_TMP=/tmp
log() { echo "[$(date -Is)] $*" | tee -a "$O/logs_post.log"; }
( cd "$REP/r3" && grep ' protocol_line_r3.json$' protocol_r3.sha256 | sha256sum -c )
[ -f "$O/selection.json" ] || python3 "$L/select_line_r3.py" "$O/train/main" --out "$O/selection.json" | tee -a "$O/logs_post.log"
POL=$(python3 -c "import json;print(json.load(open('$O/selection.json'))['selected']['policy'])")
log "選んだ方策: $POL"
# 試験の組(1 回だけ。既にあれば走らせない)
if [ ! -f "$O/eval_test_r3.json" ]; then
  log "試験の組を始めます(元の sim-core.js、$JOBS 本)"
  node "$L/eval_line_r3.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$POL" --out "$O/eval_test_r3.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
fi
PASS=$(python3 -c "import json;print(json.load(open('$O/eval_test_r3.json'))['all_judged_pass'])")
log "試験の組: all_judged_pass=$PASS"
# 報告だけの組(新しい開始 240 回)
[ -f "$O/eval_fresh_r3.json" ] || node "$L/eval_line_r3.mjs" --set fresh --core original --jobs "$JOBS" --policy "$POL" --out "$O/eval_fresh_r3.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
# 比べる値(判定・選び方には使わない): 最速の方策を同じ 60 の開始で、r2 の方策(シールドあり・なし)を同じ試験の組で
[ -f "$O/compare/eval_fastest_p60.json" ] || node "$L/eval_line_r3.mjs" --set fastest --core original --jobs "$JOBS" --keep-path --out "$O/compare/eval_fastest_p60.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
[ -f "$O/compare/eval_test_r2policy_shield.json" ] || node "$L/eval_line_r3.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$EXP/candidates/line_r2/policy.json" --out "$O/compare/eval_test_r2policy_shield.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
[ -f "$O/compare/eval_test_r2policy_noshield.json" ] || node "$L/eval_line_r3.mjs" --set test --core original --jobs "$JOBS" --no-shield --keep-path --policy "$EXP/candidates/line_r2/policy.json" --out "$O/compare/eval_test_r2policy_noshield.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
[ -f "$O/compare/eval_test_r3_noshield.json" ] || node "$L/eval_line_r3.mjs" --set test --core original --jobs "$JOBS" --no-shield --policy "$POL" --out "$O/compare/eval_test_r3_noshield.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
# 代わりの方法(不合格のときだけ。protocol_line_r3.json fallback)
if [ "$PASS" != "True" ] && [ ! -f "$O/eval_fallback_r3.json" ]; then
  node "$L/eval_line_r3.mjs" --set fallback --core original --jobs "$JOBS" --out "$O/eval_fallback_r3.json" 2>>"$O/logs_post.log" | tee -a "$O/logs_post.log"
fi
# 書き出し(candidates/line_r3/。bundle/ は変えない)
python3 "$L/export_line_r3.py" --policy "$POL" --eval "$O/eval_test_r3.json" --selection "$O/selection.json" | tee -a "$O/logs_post.log"
# 図
python "$L/line_r3_figs.py" all | tee -a "$O/logs_post.log"
log "終わり"
