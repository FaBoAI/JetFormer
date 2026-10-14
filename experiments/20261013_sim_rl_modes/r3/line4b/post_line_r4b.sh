#!/bin/bash
# 経路いろいろ r4b の学習のあと(2026-10-14 夜、line r4 の段のやり直し): 選び方 → 試験の組(1 回)→ 報告だけの組 → 確認の段の崩れた組 → 比べる値 →
# 書き出し → ブラウザの F-ID → 図。
#   /home/jetson/bin/jf-heavy --mem 1800M --lock none --need-mb 900 -- nice -n 15 bash /home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line4b/post_line_r4b.sh
# 結果のファイルがあればその手順は飛ばす(試験の組を 2 回走らせない・選び直さない。電源断のあとも同じコマンドで続く)。
set -eu
REP=/home/jetson/jetformer_reports/sim_rl_modes_20261013
O=$REP/r3/line_r4b
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
L=$EXP/r3/line4b
JOBS=${JOBS:-3}
PY=/home/jetson/miniconda/envs/jetracer/bin/python
cd "$EXP"
export JF_TMP=/tmp
mkdir -p "$O/compare" "$O/logs"
log() { echo "[$(date -Is)] $*" | tee -a "$O/logs/post.log"; }
ok_json() { [ -s "$1" ] && python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$1" 2>/dev/null; }
( cd "$REP/r3" && sha256sum -c protocol_line_r4b.sha256 ) | tee -a "$O/logs/post.log"
ok_json "$O/selection.json" || python3 "$L/select_line_r4b.py" --out "$O/selection.json" | tee -a "$O/logs/post.log"
POL=$(python3 -c "import json;print(json.load(open('$O/selection.json'))['selected']['policy'])")
log "選んだ方策: $POL"
# 試験の組(1 回だけ。結果があれば走らせない。書きかけで止まったときだけ消して走らせ直す)
if ! ok_json "$O/eval_test_r4b.json"; then
  [ -e "$O/eval_test_r4b.json" ] && { log "試験の結果が書きかけなので走らせ直します"; rm -f "$O/eval_test_r4b.json"; }
  log "試験の組を始めます(元の sim-core.js、$JOBS 本)"
  node "$L/eval_line_r4b.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$POL" --out "$O/eval_test_r4b.json" 2>>"$O/logs/post.log" | tee -a "$O/logs/post.log"
fi
PASS=$(python3 -c "import json;print(json.load(open('$O/eval_test_r4b.json'))['all_judged_pass'])")
log "試験の組: all_judged_pass=$PASS"
run() { # 出力 説明 コマンド...
  local out=$1 what=$2; shift 2
  if ok_json "$out"; then return; fi
  log "$what"
  "$@" --out "$out" 2>>"$O/logs/post.log" | tail -c 3000 >> "$O/logs/post.log" || log "失敗: $what"
}
# 報告だけの組と、確認の段の崩れた組(下調べで見たので報告だけ)
run "$O/eval_fresh_r4b.json" "報告だけの組(LN-F300-R4 + LN-FG30W60-R4B)" node "$L/eval_line_r4b.mjs" --set fresh --core original --jobs "$JOBS" --keep-path --policy "$POL"
run "$O/eval_regression_r4b.json" "確認の段の崩れた組(REG-*)" node "$L/eval_line_r4b.mjs" --set regression --core original --jobs "$JOBS" --keep-path --policy "$POL"
# 比べる値(判定・選び方には使わない。試験の判定のあと)
R3POL=$EXP/r3/candidates/line_r3/policy.json
R4POL=$REP/r3/line_r4/train/main/policies/step_004014080.json
run "$O/compare/eval_fastest_p60.json" "最速を LN-P60-R4 の開始で" node "$L/eval_line_r4b.mjs" --set fastest --core original --jobs "$JOBS" --keep-path
run "$O/compare/eval_test_noshield.json" "選んだ方策のシールドなし(試験の組)" node "$L/eval_line_r4b.mjs" --set test --core original --jobs "$JOBS" --no-shield --keep-path --policy "$POL"
run "$O/compare/eval_fresh_noshield.json" "選んだ方策のシールドなし(報告だけの組)" node "$L/eval_line_r4b.mjs" --set fresh --core original --jobs "$JOBS" --no-shield --policy "$POL"
run "$O/compare/eval_r3pol_noshield_g30_box.json" "r3 の方策のシールドなし(G30R120・BOX120)" node "$L/eval_line_r4b.mjs" --set test --core original --jobs "$JOBS" --no-shield --keep-path --policy "$R3POL" --only LN-G30R120-R4B,LN-BOX120-R4
run "$O/compare/eval_r4init_noshield_g30_box.json" "r4 の 401 万(r4b の初期化元)のシールドなし(G30R120・BOX120)" node "$L/eval_line_r4b.mjs" --set test --core original --jobs "$JOBS" --no-shield --keep-path --policy "$R4POL" --only LN-G30R120-R4B,LN-BOX120-R4
run "$O/compare/eval_test_r3pol_v5b.json" "r3 の方策 + シールド v5b(試験の組)" node "$L/compare_line_r4b.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$R3POL"
run "$O/compare/eval_test_r3pol_r3shield.json" "r3 の方策 + r3 のシールドの動き(試験の組)" node "$L/compare_line_r4b.mjs" --set test --core original --jobs "$JOBS" --keep-path --policy "$R3POL" --shield-config "$L/dev_cfg/sh_r3.json"
# 書き出し(candidates/line_r4/。bundle/ は変えない)
if [ ! -s "$EXP/r3/candidates/line_r4/sha256.txt" ]; then
  log "書き出し"
  python3 "$L/export_line_r4b.py" --policy "$POL" --eval "$O/eval_test_r4b.json" --selection "$O/selection.json" | tee -a "$O/logs/post.log"
fi
# ブラウザ(Firefox、http://192.168.55.1:8080 のページの中で候補の走行中の仕組みを読み込み、F-ID を再生)
if ! ok_json "$O/browser/browser_fid.json"; then
  log "ブラウザの F-ID(lock big の順番待ち)"
  /home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 1500 -- $PY -u "$L/browser_fid_ln4b.py" --cand "$EXP/r3/candidates/line_r4" --out "$O/browser" 2>&1 | tee -a "$O/logs/post.log" || log "ブラウザの F-ID が失敗"
fi
# 図
$PY "$L/line_r4b_figs.py" all 2>&1 | tee -a "$O/logs/post.log" || log "図が失敗"
log "終わり"
