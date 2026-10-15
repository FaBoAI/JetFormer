#!/bin/bash
# DAgger の状態集め(protocol.json dagger.procedure 1、sim_diverse_train_20261015、2026-10-15): 腕の D0〜D11 の単周を、評価と同じ設定で
# ブラウザの閉ループで走らせる(eval/run_eval.sh、結果は eval/closed_loop_dagger/)。train/PAUSE を置いて学習の道具の次の学習を待たせ、終わったら消す
# (置いたのが自分のときだけ)。
#   bash dstate_session.sh <ラベル(SDIV-s0 など)> [ほかの段 例 SMOD-s0:core,...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
PAUSE=$R/train/PAUSE
LAB=${1:?ラベル}
MARK="q1-dagger-dstates $$ $(date +%FT%T)"
MINE=0
if [ ! -e "$PAUSE" ]; then echo "$MARK" > "$PAUSE"; MINE=1; echo "PAUSE を置きました: $MARK"; else echo "PAUSE はすでにあります: $(cat $PAUSE)"; fi
release() { if [ $MINE = 1 ] && [ -e "$PAUSE" ] && [ "$(cat $PAUSE)" = "$MARK" ]; then rm -f "$PAUSE"; echo "PAUSE を消しました $(date +%T)"; fi; }
trap release EXIT
echo "状態集めの開始 $(date +%T): $LAB"
bash "$HERE/../eval/run_eval.sh" --jobs "$LAB:dstates" --out "$R/eval/closed_loop_dagger" --screens "$R/screens/dagger"
echo "状態集めの終わり $(date +%T)"
