#!/bin/bash
# 腕の学習データ(合わせたフォルダー)を作る(protocol.json の arms_Q1・arms_Q2。CPU だけ・画像は写さない)
#   S-DIV = 5 モードの見本 75 シーン + D1(DART)30 + D2(余裕の線)30(すべての分け)
#   S-MOD = 5 モードの見本 75 シーン(すべての分け)
#   C-DIV = 実機 clean_driving_20261011 の 52 シーン(学習 41・検証 11)+ 5 モードの学習の 12 シーンずつ(60 シーン。検証は実機だけ)
# 出来ているもの(splits.json がある)は飛ばす。電源断で .tmp が残っていたら make_combined.py が消して作り直す。
set -eu
HERE=/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train
DS=/home/jetson/jetformer/data/datasets
PY=/home/jetson/miniconda/envs/jetracer/bin/python
OUT=$HERE/data
MODES="$DS/sim-rl-fastest_20261014-1125 $DS/sim-rl-speed_20261014-1125 $DS/sim-rl-no_wall_20261015-0302 $DS/sim-rl-recover_20261015-0302 $DS/sim-rl-line_20261015-0302"
D1=$DS/jetformer-simulator-rl-dart-fast-10min_20261014-062216_4446c73d
D2=$DS/jetformer-simulator-rl-margin-line-10min_20261014-070934_48d04e6b
REAL=$DS/clean_driving_20261011
mkdir -p "$OUT"
build() {  # 腕 引数...
  local arm=$1; shift
  if [ -f "$OUT/$arm/splits.json" ]; then echo "$arm: 出来ています(飛ばす)"; return 0; fi
  [ -d "$OUT/$arm" ] && rmdir "$OUT/$arm" 2>/dev/null || true
  nice -n 10 "$PY" -I "$HERE/make_combined.py" --out "$OUT/$arm" "$@"
}
build S-DIV --note "sim_diverse_train_20261015 の S-DIV(Q1 の主の腕): 合格した 5 モードの見本 75 シーン + DART(D1)30 + 余裕の線(D2)30 = 135 シーン。シミュレーター専用(ジャンル 8)。シンボリックリンク" \
  -- $MODES $D1 $D2
build S-MOD --note "sim_diverse_train_20261015 の S-MOD(Q1 の 2 つ目の腕): 合格した 5 モードの見本 75 シーンだけ。シミュレーター専用(ジャンル 8)。シンボリックリンク" \
  -- $MODES
CDIV_SIM=""
for m in $MODES; do CDIV_SIM="$CDIV_SIM $m@train"; done
build C-DIV --modes-from "$DS/simreal_mix_20261013/dataset_card.json" \
  --note "sim_diverse_train_20261015 の C-DIV(Q2 の腕): 実機 clean_driving_20261011 の 52 シーン(学習 41・検証 11 = A・C と同じ)+ 合格した 5 モードの見本の学習の 12 シーンずつ = 60 シーン・36,000 フレーム(すべて学習)。検証は実機の 11 シーンだけ。シンボリックリンク" \
  -- $REAL@train,val $CDIV_SIM
du -sh "$OUT"/*
