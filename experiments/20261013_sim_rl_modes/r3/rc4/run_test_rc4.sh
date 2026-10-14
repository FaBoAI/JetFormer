#!/bin/bash
# 離脱復活 r4 の試験の組(1 回だけ)と報告だけの組(2026-10-14、学習の段 recover r4)。lock none・nice 15・3 本ずつ。
#   r3/rc4/run_test_rc4.sh <選んだ policy.json>
# 1. RC-P100-R4 + RC-KICK-R4 + RC-KICKR-R4 + RC-STD + RC-REC60 + RC-U60-R4(判定)→ eval_test.json(もうあれば走らせない = 試験は 1 回だけ)
# 2. 新しい開始 150 回(報告)→ eval_fresh.json   3. 同じ開始を元の乱し v1 で(報告)→ eval_v1.json
# 4. 最速の方策の RC-P100-R4・RC-KICK-R4・RC-KICKR-R4(報告)→ eval_fastest_test.json
# 5. r3 の候補(8d686076…)を同じ試験の組で(報告。r3 からの変化)→ eval_r3_test.json
set -u
POL=$1
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
R=/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc4
H="/home/jetson/bin/jf-heavy --mem 2G --lock none --need-mb 900 -- nice -n 15"
cd $E/r3/rc4
sha256sum eval_rc4.mjs episode_rc4.mjs rt/mode_runtime.mjs rt/perturb_v2.mjs rt/shield.mjs rt/recoverable_table.mjs > $R/code_sha256_before_test.txt
run() { local out=$1; shift; if [ -f $R/$out ]; then echo "[$(date +%T)] $out はもうあります(走らせない)"; return 0; fi
  echo "[$(date +%T)] $out を始めます"; $H node eval_rc4.mjs "$@" --jobs 3 --out $R/$out > $R/${out%.json}.log 2>&1; echo "[$(date +%T)] $out 終わり(終了コード $?)"; tail -1 $R/${out%.json}.log | cut -c1-600; }
run eval_test.json --set test --policy $POL --label selected
run eval_fresh.json --set fresh --policy $POL --label selected
run eval_v1.json --set v1 --policy $POL --label selected
run eval_fastest_test.json --set test --only RC-P100-R4,RC-KICK-R4,RC-KICKR-R4 --policy $E/r3/rc4/ref/fastest_policy.json --label fastest
run eval_r3_test.json --set test --policy $E/r3/rc4/ref/recover_r3_policy.json --label r3_candidate
echo "[$(date +%T)] 全部終わり"
