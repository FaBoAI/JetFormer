#!/bin/bash
# 経路いろいろ(line)の代わりの方法(protocol acceptance.fallbacks.line「線の混ぜ合わせ」)に使えるベストラップのチェックポイントを確かめる(2026-10-14、学習の段 line)。
# 条件: F-REC60 60/60 有効・最小すき間 ≥ 0.03 m(rl/eval_modes.mjs --mode fastest --only F-REC60、試験の組 = REC60 の 60 開始、元の sim-core.js)。
#   jf-heavy --mem 1500M --lock none -- nice -n 15 bash rl/line_fallback_qualify.sh
set -u
EXP=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
BL=/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/line/fallback
cd "$EXP"
for c in main-s1/step_010010624 main-s1/step_006012928 main-s2/step_010010624 r2-dr-s11/step_001015808 r3-s21/step_001015808 r3-s22/step_004014080 main-s0/step_006012928; do
  run=${c%%/*}; st=${c##*/}
  f="$OUT/qualify_${run}_${st}.json"
  [ -f "$f" ] && continue
  node rl/eval_modes.mjs --mode fastest --only F-REC60 --set test --core original --policy "$BL/$run/policies/$st.json" --out "$f" > /dev/null 2>> "$OUT/qualify.log"
  python3 -c "
import json; d=json.load(open('$f')); rows=d['tests']['F-REC60']
v=sum(1 for r in rows if r['valid_episode']); mc=min(r['min_clearance'] for r in rows if r['valid_episode']) if v else None
print('$c', f'{v}/{len(rows)}', mc, 'OK' if v==len(rows)==60 and mc is not None and mc>=0.03 else 'NG')" >> "$OUT/qualify.txt"
done
