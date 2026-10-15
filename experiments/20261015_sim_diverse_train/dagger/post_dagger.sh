#!/bin/bash
# DAgger の記録のあと(CPU、lock none。sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 4〜6):
#   bash post_dagger.sh <記録の結果のフォルダー> <まとめの名前> <元の腕 S-DIV|S-MOD>
# 1) 合格したシーンを 1 つの集約 ZIP に(build_collection.py。アプリと同じ処理、取り置きだけ 5 GiB)
# 2) 取り込み(import_dataset_dagger.py。ジャンル 8・シーンのタブに展開しない)
# 3) 確かめ(verify_dagger_scenes.py: G2 の位置 ≤ 2 cm・info.dart・jf なし・600 フレーム・画像がすべて違う)
# 4) 付け替え check(relabel_dataset_dagger.py --noise dagger。G3 の R2・R3)→ 合格なら write
# 5) A4 の学習データと a4_plan.json(make_a4.py)
# 6) ブリッジ側の記録(sim/runtime/datasets)を消す(取り込んだあと。protocol disk_budget)
# どの段も、前に済んでいれば飛ばす(電源断のあとの続き)。だめな段で止まる。
set -eu
OUT=${1:?結果のフォルダー}
NAME=${2:?まとめの名前}
BASE=${3:?元の腕}
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/jetson/jetformer_reports/sim_diverse_train_20261015
J="/home/jetson/bin/jf-heavy --mem 2G --lock none --need-mb 1000 --"
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/post_dagger.log"; }
log "開始"
if [ ! -f "$OUT/collection.json" ]; then
  $J python -I "$HERE/build_collection.py" --result "$OUT/result.json" --name "$NAME" --out "$OUT/collection.json" > "$OUT/collection.log" 2>&1 \
    || { log "まとめが失敗(collection.log)"; exit 1; }
fi
ZIP=$(python -I -c "import json;print(json.load(open('$OUT/collection.json'))['zip'])")
log "まとめ $ZIP"
if [ ! -f "$OUT/import.json" ]; then
  $J python -I "$HERE/import_dataset_dagger.py" "$ZIP" --out "$OUT/import.json" > "$OUT/import.log" 2>&1 || { log "取り込みが失敗(import.log)"; exit 1; }
fi
DSNAME=$(python -I -c "import json;print(json.load(open('$OUT/import.json'))['name'])")
DS=/home/jetson/jetformer/data/datasets/$DSNAME
log "取り込み $DSNAME"
$J python -I "$HERE/verify_dagger_scenes.py" --dataset "$DS" --out "$OUT/verify.json" > "$OUT/verify.log" 2>&1 || true
python -I -c "import json,sys;sys.exit(0 if json.load(open('$OUT/verify.json'))['all_pass'] else 1)" || { log "確かめが不合格(verify.json)"; exit 1; }
log "確かめ 合格(G2 ほか)"
TEACHER=$(python -I -c "
t=open('/home/jetson/jetformer/experiments/20261015_sim_diverse_train/web/dagger_plan.mjs').read();import json
print(json.loads(t[t.index('Object.freeze(')+14:t.rindex(');')])['teacher'])")
if [ ! -f "$OUT/relabel_check.json" ]; then
  $J nice -n 10 python3 -I "$HERE/relabel_dataset_dagger.py" --dataset-dir "$DS" --teacher $TEACHER --noise dagger --mode check --jobs 3 \
    --reference-scenes 1 --reference-stride 37 --report "$OUT/relabel_check.json" > "$OUT/relabel_check.log" 2>&1 || { log "付け替え check が失敗"; exit 1; }
fi
python -I -c "
import json,sys;r=json.load(open('$OUT/relabel_check.json'));c=r['checks']
ok=c['R2_replay'] and c['R3_noise_frames'] and c['pwm_recon_0us'] and c['reference_equal'] in (True,None)
print('付け替え check', c);sys.exit(0 if ok else 1)" || { log "G3(R2・R3)が不合格"; exit 1; }
log "付け替え check 合格(G3 の R2・R3)"
if [ ! -f "$OUT/relabel_write.json" ]; then
  $J nice -n 10 python3 -I "$HERE/relabel_dataset_dagger.py" --dataset-dir "$DS" --teacher $TEACHER --noise dagger --mode write --jobs 3 --reuse-work \
    --report "$OUT/relabel_write.json" > "$OUT/relabel_write.log" 2>&1 || { log "付け替え write が失敗"; exit 1; }
fi
log "付け替え write 済み"
$J python3 -I "$HERE/make_a4.py" --base-arm "$BASE" --dagger-dataset "$DSNAME" > "$OUT/make_a4.log" 2>&1 || { log "A4 のデータが作れない(make_a4.log)"; exit 1; }
log "A4 のデータと a4_plan.json"
# ブリッジ側の記録(シーンごとの記録と集約 ZIP)を消す
du -sh $R/sim/runtime/datasets 2>/dev/null | tee -a "$OUT/post_dagger.log" || true
rm -rf $R/sim/runtime/datasets
log "ブリッジ側の記録を消した。終わり"
