#!/bin/bash
# 指摘 rehearsal-s3-silent の確かめ(打ち消しの票 A)。手順書 8.1 の塊の ③・④ の行を文から抜き出し、日付だけ本物の記録のある
# 20261024(--since 1830)に替えて流す(data/raw・data/labels は読むだけ。出力は作り物の家の R の下)。
set -u
HERE=/home/jetson/jetformer/experiments/20261026_fix/fix8/refute_rehearsal_r1_A
RB=/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md
FAKE=$HERE/tmp_f5/home; rm -rf "$HERE/tmp_f5"; mkdir -p "$FAKE"
L3=$(grep -n '^python3 -I tools/standing_3lap.py' "$RB" | head -1)
L4=$(grep -n '^python3 -I tools/quick_eval_by_params.py' "$RB" | head -1)
echo "手順書 8.1 の ③: $L3"
echo "手順書 8.1 の ④: $L4"
C3=$(echo "${L3#*:}" | sed 's/--day 20261103/--day 20261024 --since 1830/')
C4=$(echo "${L4#*:}" | sed 's/--day=20261103/--day=20261024 --since 1830/')
echo "流す ③: $C3"
echo "流す ④: $C4"
R=$FAKE/jetformer_reports/bench_20261103_tune; mkdir -p "$R"
cd /home/jetson/jetformer
export PYTHONDONTWRITEBYTECODE=1
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- env HOME="$FAKE" R="$R" bash -c "$C3" > "$HERE/tmp_f5/s3_stdout.txt" 2> "$HERE/tmp_f5/s3_stderr.txt"
echo "③ の終了コード $? ・ 標準出力 $(wc -l < "$HERE/tmp_f5/s3_stdout.txt") 行 $(wc -c < "$HERE/tmp_f5/s3_stdout.txt") バイト ・ 標準エラー $(wc -l < "$HERE/tmp_f5/s3_stderr.txt") 行"
head -3 "$HERE/tmp_f5/s3_stderr.txt" | sed "s#$FAKE#<作り物の家>#g"
echo "③ の md: $(wc -l < "$R/standing_3lap.md") 行。見出し:"
grep -n '^#' "$R/standing_3lap.md" | head -10
echo "③ の md の「動いてから」の字: $(grep -c '動いてから' "$R/standing_3lap.md") 行"
/home/jetson/bin/jf-heavy --lock none --mem 1500M -- env HOME="$FAKE" R="$R" bash -c "$C4" > "$HERE/tmp_f5/s4_stdout.txt" 2> "$HERE/tmp_f5/s4_stderr.txt"
echo "④ の終了コード $? ・ 標準出力 $(wc -l < "$HERE/tmp_f5/s4_stdout.txt") 行。見出し:"
grep -n '^#' "$HERE/tmp_f5/s4_stdout.txt" | head -10
echo "--- ④ の画面の 3 章の表 と ③ の md の止まってから 3 周の表(R1 まで行った発進)の行を比べる"
awk '/^## 3\. 止まってから 3 周/{f=1;next} /^## /{f=0} f && /^\|/' "$HERE/tmp_f5/s4_stdout.txt" > "$HERE/tmp_f5/s4_t3.txt"
grep '^|' "$R/standing_3lap.md" > "$HERE/tmp_f5/s3_rows.txt"
echo "④ 3 章の表の行 $(wc -l < "$HERE/tmp_f5/s4_t3.txt")・そのうち ③ の md にそのままある行 $(grep -cxFf "$HERE/tmp_f5/s3_rows.txt" "$HERE/tmp_f5/s4_t3.txt")"
echo "④ 3 章の表の頭:"; head -2 "$HERE/tmp_f5/s4_t3.txt" | cut -c1-260
echo "--- 手順書で standing_3lap.md を開く・見るコマンド(cat・sed・less・head)"
grep -n 'standing_3lap\.md' "$RB" | grep -E '(cat|sed|less|head|more) ' || echo "(無い)"
echo "--- 手順書 8.1 の表の ③ の「出力」の欄"
grep -n '^| ③ | `tools/standing_3lap.py`' "$RB" | awk -F'|' '{print "  出力の欄:" $5}'
rm -rf "$HERE/tmp_f5"
