#!/bin/bash
# 離脱復活 r3 の試験の組(1 回だけ)と報告だけの組(2026-10-14、学習の段 recover r3)。lock none・nice 15・3 本ずつ。
#   r3/rc/run_test_rc3.sh <選んだ policy.json>
# 1. RC-P100-R3 + RC-KICK-R3 + RC-STD + RC-REC60 + RC-U60-R3(判定)→ eval_test.json(もうあれば止まる = 試験は 1 回だけ)
# 2. 新しい開始 150 回(報告)→ eval_fresh.json   3. 同じ開始を元の乱し v1 で(報告)→ eval_v1.json
# 4. 最速の方策の RC-P100-R3・RC-KICK-R3(報告)→ eval_fastest_test.json
# 5. 判定が不合格なら protocol fallbacks.recover(強さ 弱・中、乱し v2 のまま。1 回だけ)→ eval_fallback.json
set -u
POL=$1
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc3
H="/home/jetson/bin/jf-heavy --mem 2G --lock none --need-mb 900 -- nice -n 15"
cd $E/r3/rc
run() { local out=$1; shift; if [ -f $R/$out ]; then echo "[$(date +%T)] $out はもうあります(走らせない)"; return 0; fi
  echo "[$(date +%T)] $out を始めます"; $H node eval_rc3.mjs "$@" --jobs 3 --out $R/$out > $R/${out%.json}.log 2>&1; echo "[$(date +%T)] $out 終わり(終了コード $?)"; tail -1 $R/${out%.json}.log | cut -c1-400; }
run eval_test.json --set test --policy $POL --label selected
run eval_fresh.json --set fresh --policy $POL --label selected
run eval_v1.json --set v1 --policy $POL --label selected
run eval_fastest_test.json --set test --only RC-P100-R3,RC-KICK-R3 --policy $E/r3/rc/ref/fastest_policy.json --label fastest
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/eval_test.json'))['all_judged_pass'] else 1)"; then
  run eval_fallback.json --set fallback --policy $POL --label selected_weak_medium
fi
echo "[$(date +%T)] 全部終わり"
