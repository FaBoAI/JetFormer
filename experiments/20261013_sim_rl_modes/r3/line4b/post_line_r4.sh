#!/bin/bash
# 経路いろいろ r4 の学習のあと(2026-10-14、line r4 の段): 選び方 → 試験の組(1 回)→ 報告だけの組 → 確認 r3 の崩れた組 → 比べる値 → 書き出し → 図。
#   /home/jetson/bin/jf-heavy --mem 1800M --lock none --need-mb 900 -- nice -n 15 bash /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line4/post_line_r4.sh
# 結果のファイルがあればその手順は飛ばす(試験の組を 2 回走らせない・選び直さない)。
set -eu
REP=/home/jetson/jetformer_reports/sim_rl_modes_20261013
O=$REP/r3/line_r4
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
L=$EXP/r3/line4
JOBS=${JOBS:-3}
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate jetracer
cd "$EXP"
export JF_TMP=/tmp
mkdir -p "$O/compare"
log() { echo "[$(date -Is)] $*" | tee -a "$O/logs/post.log"; }
( cd "$REP/r3" && sha256sum -c protocol_line_r4.sha256 )
[ -f "$O/selection.json" ] || python3 "$L/select_line_r4.py" "$O/train/main" --out "$O/selection.json" | tee -a "$O/logs/post.log"
POL=$(python3 -c "import json;print(json.load(open('$O/selection.json'))['selected']['policy'])")
log "選んだ方策: $POL"
# 試験の組(1 回だけ。既にあれば走らせない)
if [ ! -f "$O/eval_test_r4.json" ]; then
  log "試験の組を始めます(元の sim-core.js、$JOBS 本)"
  node "$L/eval_line_r4.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$POL" --out "$O/eval_test_r4.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
fi
PASS=$(python3 -c "import json;print(json.load(open('$O/eval_test_r4.json'))['all_judged_pass'])")
log "試験の組: all_judged_pass=$PASS"
# 報告だけの組(新しい一様な開始 300 回)と、確認 r3 の崩れた組(下調べで見たので報告だけ)
[ -f "$O/eval_fresh_r4.json" ] || node "$L/eval_line_r4.mjs" --set fresh --core original --jobs "$JOBS" --policy "$POL" --out "$O/eval_fresh_r4.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
[ -f "$O/eval_regression_r4.json" ] || node "$L/eval_line_r4.mjs" --set regression --core original --jobs "$JOBS" --keep-path --policy "$POL" --out "$O/eval_regression_r4.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
# 比べる値(判定・選び方には使わない。試験の判定のあと): 最速を同じ 60 の開始で、選んだ方策のシールドなし、r3 の方策 + シールド v5、r3 の方策 + r3 のシールド(v4 の動き)
[ -f "$O/compare/eval_fastest_p60.json" ] || node "$L/eval_line_r4.mjs" --set fastest --core original --jobs "$JOBS" --keep-path --out "$O/compare/eval_fastest_p60.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
[ -f "$O/compare/eval_test_r4_noshield.json" ] || node "$L/eval_line_r4.mjs" --set test --core original --jobs "$JOBS" --no-shield --keep-path --policy "$POL" --out "$O/compare/eval_test_r4_noshield.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
R3POL=$EXP/r3/candidates/line_r3/policy.json
[ -f "$O/compare/eval_test_r3policy_v5.json" ] || node "$L/eval_line_r4.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$R3POL" --out "$O/compare/eval_test_r3policy_v5.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
[ -f "$O/compare/eval_test_r3policy_r3shield.json" ] || node "$L/compare_line_r4.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$R3POL" --shield-config "$L/dev_cfg/sh_r3.json" --out "$O/compare/eval_test_r3policy_r3shield.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
# 書き出し(candidates/line_r4/。bundle/ は変えない)
python3 "$L/export_line_r4.py" --policy "$POL" --eval "$O/eval_test_r4.json" --selection "$O/selection.json" | tee -a "$O/logs/post.log"
# 図
python "$L/line_r4_figs.py" all | tee -a "$O/logs/post.log"
log "終わり"
