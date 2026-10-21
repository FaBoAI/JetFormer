#!/bin/bash
# bench_nightly.sh: 実車の日の夜の 1 コマンド(2026-10-21、段 S3)
#   その日の記録 → ラベル → 区間の成績表 → B の記録で G1(上限の再生) → ブレーキの表 → ベンチマークレポート
#
#   bash tools/bench_nightly.sh 20261023                       # 1 日目の夜
#   bash tools/bench_nightly.sh 20261024 --day1 20261023       # 2 日目の夜(レポートは 2 日分。2 日目の判定は 10/24 の周だけ)
#   bash tools/bench_nightly.sh 20261010 --assign A.json --bench B.json --out DIR --mock   # 模擬(10/10 の記録)
#
# 選択肢:
#   --day1 YYYYMMDD   1 日目(2 日目の夜に渡す。省くと bench_candidates.json の days から決める:
#                     その日が days[1] 以降なら days[0]。その日が days に無ければ注意を出して 1 日目として数える)
#   --bench FILE      候補の表(既定 data/bench_candidates.json)
#   --assign FILE     {ログ: 説明欄の文} の JSON(説明欄の書き忘れの直し・模擬)
#   --labels DIR      ラベルの場所(既定 data/labels。無いログだけ tools/label_events.py で作る)
#   --relabel         ラベルがあっても作り直す
#   --out DIR         途中の出力(既定 ~/jetformer_reports/<bench>/nightly_<日>)
#   --report DIR      レポート(既定 ~/jetformer_reports/<bench>)
#   --mock            模擬の印(bench_report の --mock)。ほかの bench_report の選択肢は -- のあとに並べる
#
# 読み取りと軽い計算だけ(numpy・matplotlib)。GPU は使わない。どの段も jf-heavy --lock none --mem 1500M で動かす。
# data/raw は変えない。走行中(10/23・24 の昼)は動かさない。jetformer-web は再起動しない。
set -u
JF=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/jetson/miniconda/envs/jetracer/bin/python}
HEAVY=${HEAVY:-/home/jetson/bin/jf-heavy}
DAY=${1:-}
[ -n "$DAY" ] || { sed -n '2,20p' "$0"; exit 2; }
shift
DAY=${DAY//-/}
DAY1=""; BENCH="$JF/data/bench_candidates.json"; ASSIGN=""; LABELS="$JF/data/labels"; RELABEL=0; OUT=""; REPORT=""; MOCK=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --day1) DAY1=${2//-/}; shift 2;;
    --bench) BENCH="$2"; shift 2;;
    --assign) ASSIGN="$2"; shift 2;;
    --labels) LABELS="$2"; shift 2;;
    --relabel) RELABEL=1; shift;;
    --out) OUT="$2"; shift 2;;
    --report) REPORT="$2"; shift 2;;
    --mock) MOCK="--mock"; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "知らない選択肢: $1" >&2; exit 2;;
  esac
done
BNAME=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('bench','real_bench'))" "$BENCH" 2>/dev/null || echo real_bench)
# 1 日目の自動の決め方(V-tools 4 章 11: 2 日目の夜に --day1 を忘れると、2 日目の初めてのブロックの K が 3 になる)
if [ -z "$DAY1" ]; then
  D1=$("$PY" - "$BENCH" "$DAY" <<'PYEOF' 2>/dev/null
import json, sys
try:
    b = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception:
    print('NOBENCH'); sys.exit(0)
day = sys.argv[2]
day = '%s-%s-%s' % (day[:4], day[4:6], day[6:8])
dates = [d.get('date') for d in b.get('days', []) if d.get('date')]
if day in dates:
    i = dates.index(day)
    print('' if i == 0 else dates[0].replace('-', ''))
else:
    print('NOTINPLAN')
PYEOF
)
  case "$D1" in
    NOBENCH|NOTINPLAN|"") ;;
    *) DAY1=$D1; AUTO_DAY1=1;;
  esac
  if [ "$D1" = "NOTINPLAN" ]; then
    echo "注意: $DAY は $BENCH の days に無い(1 日目として数える。2 日目なら --day1 を渡す)" >&2
  fi
fi
OUT=${OUT:-/home/jetson/jetformer_reports/$BNAME/nightly_$DAY}
REPORT=${REPORT:-/home/jetson/jetformer_reports/$BNAME}
mkdir -p "$OUT"
LOG="$OUT/nightly.log"
run() {  # 段の名前 コマンド…
  local name="$1"; shift
  echo "== $(date '+%H:%M:%S') $name" | tee -a "$LOG"
  "$HEAVY" --lock none --mem 1500M --need-mb 800 -- nice -n 10 "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "   → 終了コード $rc" | tee -a "$LOG"
  return $rc
}
cd "$JF" || exit 1
echo "# bench_nightly $DAY(1 日目 ${DAY1:-なし}${AUTO_DAY1:+、bench_candidates.json の days から})bench=$BNAME $(date '+%F %T')" | tee "$LOG"
mapfile -t LOGS < <(ls -d data/raw/"$DAY"-* 2>/dev/null | xargs -n1 basename 2>/dev/null | grep -E '^[0-9]{8}-[0-9]{6}$')
[ ${#LOGS[@]} -gt 0 ] || { echo "data/raw/$DAY-* がありません" | tee -a "$LOG"; exit 1; }
echo "記録 ${#LOGS[@]} 本: ${LOGS[*]}" | tee -a "$LOG"

# 1. ラベル(無いものだけ)
NEED=()
for lg in "${LOGS[@]}"; do
  if [ $RELABEL -eq 1 ] || [ ! -f "$LABELS/$lg.json" ] || [ ! -f "$LABELS/$lg.npz" ]; then NEED+=("$lg"); fi
done
if [ ${#NEED[@]} -gt 0 ]; then
  run "1 ラベル(${#NEED[@]} 本)" "$PY" tools/label_events.py "${NEED[@]}" --out "$LABELS" || echo "   ラベルの段が失敗(続ける)" | tee -a "$LOG"
  # 作れなかったものを 1 本ずつ作り直す(label_events.py は 10/21 からログごとに飛ばして続け、失敗があれば終了コード 1)
  LEFT=()
  for lg in "${NEED[@]}"; do
    if [ ! -f "$LABELS/$lg.json" ] || [ ! -f "$LABELS/$lg.npz" ] || { [ $RELABEL -eq 1 ] && [ "$LABELS/$lg.json" -ot "$LOG" ]; }; then LEFT+=("$lg"); fi
  done
  for lg in "${LEFT[@]}"; do
    run "1 ラベル(1 本ずつ)$lg" "$PY" tools/label_events.py "$lg" --out "$LABELS" || echo "   $lg のラベルが作れない(このログは成績表・G1・ブレーキの表で「読めなかったログ」に出る)" | tee -a "$LOG"
  done
else
  echo "== 1 ラベル: 全部あるので作らない" | tee -a "$LOG"
fi
AS=(); [ -n "$ASSIGN" ] && AS=(--assign "$ASSIGN")
# 2. 区間の成績表(その日の AI の記録全部。候補・電池・ブロックごと)
run "2 区間の成績表" "$PY" tools/section_scorecard.py --day "$DAY" --labels "$LABELS" --bench "$BENCH" "${AS[@]}" --out "$OUT/scorecard" || true
# 3. B の記録で G1(上限の再生。人の記録も R1 の検出の誤りに使う)
run "3 G1(B の記録)" "$PY" tools/cap_replay.py --day "$DAY" --labels "$LABELS" --cand B "${AS[@]}" --out "$OUT/cap_replay" || true
# 4. ブレーキの表
run "4 ブレーキの表" "$PY" tools/brake_table.py --day "$DAY" --labels "$LABELS" --out "$OUT/brake_table" || true
# 5. ベンチマークレポート
DAYS=(--day "$DAY"); [ -n "$DAY1" ] && DAYS=(--day "$DAY1" --day "$DAY")
CR=(); [ -f "$OUT/cap_replay/cap_replay.json" ] && CR=(--cap-replay "$OUT/cap_replay/cap_replay.json")
BR=(); [ -f "$OUT/brake_table/brake_table.json" ] && BR=(--brake "$OUT/brake_table/brake_table.json")
if [ -n "$ASSIGN" ] && [ -n "$DAY1" ]; then
  echo "注意: --assign は 2 日分の記録の説明欄をまとめて上書きする JSON にすること" | tee -a "$LOG"
fi
run "5 ベンチマークレポート" "$PY" tools/bench_report.py "${DAYS[@]}" --labels "$LABELS" --bench "$BENCH" "${AS[@]}" "${CR[@]}" "${BR[@]}" $MOCK --out "$REPORT" "${EXTRA[@]}"
RC=$?
echo "# 終わり $(date '+%T')。成績表 $OUT/scorecard/scorecard.md・G1 $OUT/cap_replay/cap_replay.md・ブレーキ $OUT/brake_table/brake_table.md・レポート $REPORT/README.md" | tee -a "$LOG"
exit $RC
