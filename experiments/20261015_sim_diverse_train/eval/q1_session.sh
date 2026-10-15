#!/bin/bash
# Q1 の閉ループ評価のセッション(sim_diverse_train_20261015、2026-10-15)。
#   bash q1_session.sh <jobs> [--a2-if-different]
# 1) train/PAUSE を置く(学習の道具が次の学習を始めずに待つ。置いたのが自分のときだけ、終わりに消す)
# 2) run_eval.sh <jobs>(ブリッジ 8821 + Firefox。lock big を待つ)
# 3) --a2-if-different: same_as_yesterday.py で前日との同一性を確かめ、違えば A2-s0 の E0〜E11 を同じ手順で走らせ直す
# 4) PAUSE を消す(自分が置いたものだけ)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
PAUSE=$R/train/PAUSE
MARK="q1-eval $$ $(date +%FT%T)"
MINE=0
if [ ! -e "$PAUSE" ]; then echo "$MARK" > "$PAUSE"; MINE=1; echo "PAUSE を置きました: $MARK"; else echo "PAUSE はすでにあります: $(cat $PAUSE)"; fi
release() { if [ $MINE = 1 ] && [ -e "$PAUSE" ] && [ "$(cat $PAUSE)" = "$MARK" ]; then rm -f "$PAUSE"; echo "PAUSE を消しました $(date +%T)"; fi; }
trap release EXIT
JOBS="$1"; shift
A2=0; [ "${1:-}" = "--a2-if-different" ] && A2=1
echo "評価の開始 $(date +%T): $JOBS"
bash "$HERE/run_eval.sh" --jobs "$JOBS"
echo "評価の終わり $(date +%T)"
if [ $A2 = 1 ]; then
  python3 -I "$HERE/same_as_yesterday.py"; RC=$?
  if [ $RC = 3 ]; then
    echo "前日と同じとは言えません → A2-s0 の E0〜E11 を走らせ直します $(date +%T)"
    bash "$HERE/run_eval.sh" --jobs "A2-s0:core,A2-s0:ext"
    echo "A2-s0 の終わり $(date +%T)"
  fi
fi
