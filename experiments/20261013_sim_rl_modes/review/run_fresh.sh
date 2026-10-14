#!/bin/bash
# 審査の段(2026-10-14): 各モードの合否の試験を新しい種 2 つ(A・B)で走らせ直す。判定は rl/eval_modes.mjs と同じ(写しの review/eval_fresh.mjs)。
#   bash review/run_fresh.sh [A|B ...]
# 方策は登録簿(data/sim_rl/modes/<key>/policy.json = 画面で使うもの)。出していない line・obstacle は candidates/<mode>_r2・obstacle の候補。
set -u
E=/home/jetson/jetformer/experiments/20261013_sim_rl_modes
OUT=/home/jetson/jetformer_reports/sim_rl_modes_20261013/review/fresh
REG=/home/jetson/jetformer/data/sim_rl/modes
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 700 -- nice -n 15"
cd "$E" || exit 1
TAGS=${*:-A B}
for tag in $TAGS; do
  case $tag in
    A) F='{"tag":"A","seed_base":5100000,"halton_from":40001,"kick_t0":1.1}';;
    B) F='{"tag":"B","seed_base":5200000,"halton_from":50001,"kick_t0":1.4}';;
    *) echo "unknown tag $tag"; exit 2;;
  esac
  mkdir -p "$OUT/$tag"
  for m in fastest no_wall recover speed line obstacle; do
    o="$OUT/$tag/$m.json"; [ -f "$o" ] && continue
    case $m in
      fastest|recover) args=(--policy "$REG/$m/policy.json");;
      no_wall) args=(--policy "$REG/$m/policy.json" --only NW-N60,NW-STD,NW-G,NW-R200,NW-FC10);;
      speed) args=(--policy "$REG/$m/policy.json" --protocol speed_r2);;
      line) args=(--policy "$E/candidates/line_r2/policy.json" --protocol line_r2);;
      obstacle) args=(--policy "$E/candidates/obstacle/policy.json");;
    esac
    echo "$(date +%T) start $tag $m" >> "$OUT/run.log"
    JF_FRESH="$F" $H node review/eval_fresh.mjs --mode $m "${args[@]}" --set test --core original --jobs 2 --out "$o" >> "$OUT/run.log" 2>&1 \
      || echo "$(date +%T) FAIL $tag $m" >> "$OUT/run.log"
    echo "$(date +%T) end $tag $m" >> "$OUT/run.log"
  done
done
echo "$(date +%T) all done" >> "$OUT/run.log"
