#!/bin/bash
# bench_nightly.sh: 実車の日の夜の 1 コマンド(2026-10-21、段 S3)
#   その日の記録 → ラベル → 区間の成績表 → B の記録で G1(上限の再生) → ブレーキの表 → DE の揺れの表 → ベンチマークレポート
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
#   --report DIR      レポート(既定 ~/jetformer_reports/<bench>/report_<日>。2026-10-26 までの既定は <bench> そのもので、
#                     10/23 の README.md・report.json を上書きした)
#   --overwrite       既定の出力先に前の結果があっても上書きする(無ければ止まる。--out・--report で渡した場所は今まで通り上書き)
#   --mock            模擬の印(bench_report の --mock)。ほかの bench_report の選択肢は -- のあとに並べる
#
# 既定の出力先(--out・--report を省いたとき)が空でなければ、何も書かずに終了コード 3 で止まる(2026-10-26 fix2)。
# 2026-10-26 fix3: 記録が無い日は、既定の出力先に何も作らずに終了コード 1(前は nightly.log だけが残り、次の回が 3 で止まった)。
#   段 3・4(G1・ブレーキの表)が 0 以外で終わったら、前の回の JSON が残っていてもレポートに渡さない(nightly.log に注意)。
#   既定の report_<日> を新しく作るときは、前の報告(<bench>/report_<前の日>/ か <bench>/)の deviations.md の
#   「## 人が書いたこと」を写す(bench_report がそのまま残す)。--report で渡した場所には写さない(今まで通り)。
# 読み取りと軽い計算だけ(numpy・matplotlib)。GPU は使わない。どの段も jf-heavy --lock none --mem 1500M で動かす。
# data/raw は変えない。走行中(10/23・24 の昼)は動かさない。jetformer-web は再起動しない。
# 環境変数(試験用): JF_REPORTS(既定の出力先の根。既定 /home/jetson/jetformer_reports)・PY・HEAVY
set -u
JF=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/jetson/miniconda/envs/jetracer/bin/python}
HEAVY=${HEAVY:-/home/jetson/bin/jf-heavy}
DAY=${1:-}
[ -n "$DAY" ] || { sed -n '2,29p' "$0"; exit 2; }
shift
DAY=${DAY//-/}
DAY1=""; BENCH="$JF/data/bench_candidates.json"; ASSIGN=""; LABELS="$JF/data/labels"; RELABEL=0; OUT=""; REPORT=""; MOCK=""
OVERWRITE=0
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
    --overwrite) OVERWRITE=1; shift;;
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
# 既定の出力先(2026-10-26 fix2): レポートも日付つきの新しい場所にし、前の結果がある既定の場所には書かない
REPORTS_ROOT=${JF_REPORTS:-/home/jetson/jetformer_reports}
OUT_DEF=0; REPORT_DEF=0
[ -n "$OUT" ] || { OUT_DEF=1; OUT="$REPORTS_ROOT/$BNAME/nightly_$DAY"; }
[ -n "$REPORT" ] || { REPORT_DEF=1; REPORT="$REPORTS_ROOT/$BNAME/report_$DAY"; }
# 記録(2026-10-26 fix3、反証 1 回目 code:F4): 既定の出力先のときは、何かを作る前に確かめる(前は nightly.log だけを書いてから
# 止まり、その dir が「前の結果」になって、記録が来たあとの次の既定の回が終了コード 3 で止まった)
mapfile -t LOGS < <(cd "$JF" && ls -d data/raw/"$DAY"-* 2>/dev/null | xargs -n1 basename 2>/dev/null | grep -E '^[0-9]{8}-[0-9]{6}$')
if [ ${#LOGS[@]} -eq 0 ] && [ $OUT_DEF -eq 1 ]; then
  echo "data/raw/$DAY-* がありません(既定の出力先 $OUT には何も作らない)"
  exit 1
fi
# この台本が用意した deviations.md(人の注を写しただけで、bench_report がまだ書いていない)の頭の行(2026-10-26 fix3)
SEED_HEAD='# 外れたこと(bench_nightly.sh が用意した。上は bench_report が書き直す)'
seed_only() {  # dir の中身が、この台本が用意した deviations.md だけ(前の回が段 5 の前に落ちた)
  [ "$(ls -A "$1" 2>/dev/null)" = "deviations.md" ] && [ "$(head -n 1 "$1/deviations.md" 2>/dev/null)" = "$SEED_HEAD" ]
}
if [ $OVERWRITE -eq 0 ]; then
  BUSY=()
  [ $OUT_DEF -eq 1 ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && BUSY+=("$OUT")
  [ $REPORT_DEF -eq 1 ] && [ -n "$(ls -A "$REPORT" 2>/dev/null)" ] && ! seed_only "$REPORT" && BUSY+=("$REPORT")
  if [ ${#BUSY[@]} -gt 0 ]; then
    echo "止めます: 既定の出力先に前の結果があります(上書きしない): ${BUSY[*]}" >&2
    echo "  別の場所に出すなら --out DIR --report DIR、上書きしてよいなら --overwrite を付ける" >&2
    exit 3
  fi
fi
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
echo "出力 $OUT$([ $OUT_DEF -eq 1 ] && echo '(既定)')・レポート $REPORT$([ $REPORT_DEF -eq 1 ] && echo '(既定)')" | tee -a "$LOG"
[ ${#LOGS[@]} -gt 0 ] || { echo "data/raw/$DAY-* がありません" | tee -a "$LOG"; exit 1; }     # --out を渡したとき(今まで通り)
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
run "3 G1(B の記録)" "$PY" tools/cap_replay.py --day "$DAY" --labels "$LABELS" --cand B "${AS[@]}" --out "$OUT/cap_replay"
RC3=$?
# 4. ブレーキの表
run "4 ブレーキの表" "$PY" tools/brake_table.py --day "$DAY" --labels "$LABELS" --out "$OUT/brake_table"
RC4=$?
# 4b. DE の揺れの表(2026-10-24。GM の出口の行き過ぎと DE の切り返しを設定ごとに。レポートには入れない)
run "4b DE の揺れの表" "$PY" tools/de_swing.py --day "$DAY" --labels "$LABELS" "${AS[@]}" --out "$OUT/de_swing" || true
# 5. ベンチマークレポート
DAYS=(--day "$DAY"); [ -n "$DAY1" ] && DAYS=(--day "$DAY1" --day "$DAY")
# 段 3・4 が 0 以外で終わったら、その JSON は前の回のもの(同じ --out で回し直したとき)かもしれないので渡さない
# (2026-10-26 fix3、反証 1 回目 code:F5。前は [ -f ] だけを見て、前の回の G1・ブレーキの表をこの回のレポートに使った)
CR=(); BR=()
if [ -f "$OUT/cap_replay/cap_replay.json" ]; then
  if [ $RC3 -eq 0 ]; then CR=(--cap-replay "$OUT/cap_replay/cap_replay.json")
  else echo "注意: 段 3(G1)が終了コード $RC3。$OUT/cap_replay/cap_replay.json は前の回のものかもしれないので、レポートに渡さない" | tee -a "$LOG"; fi
fi
if [ -f "$OUT/brake_table/brake_table.json" ]; then
  if [ $RC4 -eq 0 ]; then BR=(--brake "$OUT/brake_table/brake_table.json")
  else echo "注意: 段 4(ブレーキの表)が終了コード $RC4。$OUT/brake_table/brake_table.json は前の回のものかもしれないので、レポートに渡さない" | tee -a "$LOG"; fi
fi
# 既定の report_<日> を新しく作るときは、前の報告の「## 人が書いたこと」を写す(2026-10-26 fix3、反証 1 回目 code:F3。
# 前の既定(<bench> そのもの)では同じ deviations.md に人の注が残ったが、日付つきの場所では空で始まった)
if [ $REPORT_DEF -eq 1 ] && { [ ! -e "$REPORT/deviations.md" ] || [ "$(head -n 1 "$REPORT/deviations.md" 2>/dev/null)" = "$SEED_HEAD" ]; }; then
  mkdir -p "$REPORT"
  "$PY" - "$REPORTS_ROOT/$BNAME" "$DAY" "$REPORT/deviations.md" "$SEED_HEAD" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys
import time
from pathlib import Path
root, day, dst, head = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
MARK = '\n## 人が書いたこと\n'
TAG = '- (bench_nightly.sh が写した)'


def notes(p):
    """deviations.md の「## 人が書いたこと」のあとの行(写したことの行は除く)。無い・空なら None"""
    try:
        s = p.read_text(encoding='utf-8')
    except (OSError, ValueError):
        return None
    if MARK not in s:
        return None
    body = [x for x in s[s.index(MARK) + len(MARK):].split('\n') if not x.startswith(TAG)]
    return body if any(x.strip() for x in body) else None


cands = [root / 'deviations.md'] + [p for _, p in sorted((p.parent.name[len('report_'):], p) for p in root.glob('report_*/deviations.md')
                                                          if p.parent.name[len('report_'):] < day)]
src = None
for p in cands:                       # いちばん新しい、人の注のある報告(日付つきの報告は前の報告から写すので、全部を引き継ぐ)
    b = notes(p)
    if b is not None:
        src = (p, b)
if src is None:
    print('人が書いたこと: 前の報告に無い(空で始める)')
else:
    p, body = src
    while body and not body[0].strip():
        body = body[1:]
    dst.write_text(head + '\n' + MARK + '\n'
                   + '%s %s から(%s)\n' % (TAG, p, time.strftime('%Y-%m-%d %H:%M')) + '\n'.join(body), encoding='utf-8')
    print('人が書いたこと: %s から %s に写した(%d 行)' % (p, dst, sum(1 for x in body if x.strip())))
PYEOF
fi
if [ -n "$ASSIGN" ] && [ -n "$DAY1" ]; then
  echo "注意: --assign は 2 日分の記録の説明欄をまとめて上書きする JSON にすること" | tee -a "$LOG"
fi
run "5 ベンチマークレポート" "$PY" tools/bench_report.py "${DAYS[@]}" --labels "$LABELS" --bench "$BENCH" "${AS[@]}" "${CR[@]}" "${BR[@]}" $MOCK --out "$REPORT" "${EXTRA[@]}"
RC=$?
echo "# 終わり $(date '+%T')。成績表 $OUT/scorecard/scorecard.md・G1 $OUT/cap_replay/cap_replay.md・ブレーキ $OUT/brake_table/brake_table.md・DE の揺れ $OUT/de_swing/de_swing.md・レポート $REPORT/README.md" | tee -a "$LOG"
exit $RC
