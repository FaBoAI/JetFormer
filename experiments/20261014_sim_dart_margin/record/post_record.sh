#!/bin/bash
# 記録のあとの処理(記録の段、2026-10-14、sim_dart_20261014): 検証 → 取り込み → 付け替え(check → write)→ ずれの統計。CPU だけ(lock none)。
#   bash post_record.sh D1|D2|D3
# 途中の段がだめなら止まる(set -e)。付け替えの write は check の R2・R3・基準の一致が合格したときだけ(R4 は書くだけで止めない)。
# D2 は乱しなしなので付け替えない(check だけ)。ただし check で「先生が立て直せない」フレームが出たら(終わりの外し。build_check_r1 の指摘)、
# 同じ道具の write で future_valid だけを外す(正解は元と同じ。差 0 µs を check で確かめる)。
set -eu
D=${1:?D1・D2・D3}
E=/home/jetson/jetformer/experiments/20261014_sim_dart_margin
R=/home/jetson/jetformer_reports/sim_dart_20261014
OUT=$R/recording/$D
HEAVY=/home/jetson/bin/jf-heavy
case $D in D1) T=fast; N=N_fast; REF=2;; D2) T=margin; N=none; REF=0;; D3) T=margin; N=N_margin; REF=1;; esac
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $OUT/post_record.log; }
log "開始 $D"
# 1) 検証
if [ ! -f $OUT/verify/collection_verify.json ]; then
  $HEAVY --mem 1500M --lock none -- nice -n 10 python -I $E/record/verify_dart_collection.py --dataset $D --result $OUT/result.json \
    --out-dir $OUT/verify --workers 3 --reuse > $OUT/verify.log 2>&1 || { log "検証が不合格(verify.log)"; exit 1; }
fi
log "検証 合格"
# 2) 取り込み
if [ ! -f $OUT/import.json ]; then
  # 名前の形を protocol(jetformer-simulator-<collection_name>、A0 と同じ形)に合わせる(記録の段で追加。deviations.md 12)
  CN=$(python -I -c "import json,os;p='$OUT/collection.json';print(json.load(open(p))['request']['name'] if os.path.exists(p) else '')")
  case "$CN" in jetformer-simulator-*) ;; *)
    [ -f $OUT/collection.json ] && mv $OUT/collection.json $OUT/collection.unprefixed.json
    PN=jetformer-simulator-$(python -I -c "import json;print(json.load(open('$R/protocol.json'))['recording']['datasets']['$D']['collection_name'])")
    $HEAVY --mem 2G --lock none -- python -I $E/record/build_collection.py --result $OUT/result.json --name $PN --out $OUT/collection.json \
      > $OUT/collection_prefixed.log 2>&1 || { log "集約 ZIP の作り直しが失敗(collection_prefixed.log)"; exit 1; }
    log "集約 ZIP を $PN の名前で作った";;
  esac
  ZIP=$(python -I -c "
import json,os
r=json.load(open('$OUT/result.json'))
c=r.get('collection') or {}
p='$OUT/collection.json'
print(json.load(open(p))['zip'] if os.path.exists(p) else c.get('zip',''))")
  [ -f "$ZIP" ] || { log "集約 ZIP が見つかりません: $ZIP"; exit 1; }
  # 集約 ZIP の形式(ベストラップの check_collection_zip.py の写し: マニフェストの sha256・NPZ の形が元の 20 分データと同じ・PWM の作り直し)
  $HEAVY --mem 1500M --lock none -- nice -n 10 python -I $E/record/check_collection_zip.py $ZIP --out $OUT/collection_zip_check.json \
    > $OUT/collection_zip_check.log 2>&1 || { log "集約 ZIP の検査が不合格(collection_zip_check.log)"; exit 1; }
  log "集約 ZIP の検査 合格"
  $HEAVY --mem 2G --lock none -- python -I $E/record/import_dataset.py $ZIP --dataset $D --out $OUT/import.json > $OUT/import.log 2>&1 \
    || { log "取り込みが失敗(import.log)"; exit 1; }
fi
NAME=$(python -I -c "import json;print(json.load(open('$OUT/import.json'))['name'])")
DS=/home/jetson/jetformer/data/datasets/$NAME
log "取り込み $NAME"
# 3) 付け替え check(基準の作り方との比べつき)
if [ ! -f $OUT/relabel_check.json ]; then
  $HEAVY --mem 1500M --lock none -- nice -n 10 python3 -I $E/relabel/relabel_dataset.py --dataset-dir $DS --teacher $T --noise $N \
    --mode check --jobs 3 --reference-scenes $REF --reference-stride 37 --report $OUT/relabel_check.json > $OUT/relabel_check.log 2>&1 \
    || { log "付け替え check が失敗(relabel_check.log)"; exit 1; }
fi
OKW=$(python -I -c "
import json
r=json.load(open('$OUT/relabel_check.json'));c=r['checks'];s=r['summary']
ok=c['R2_replay'] and c['R3_noise_frames'] and c['pwm_recon_0us'] and c['reference_equal'] in (True,None)
if '$N'=='none': ok=ok and s['frames_diff_gt_1us_used']==0
print('yes' if ok else 'no', s['invalid_teacher_frames'])")
log "付け替え check: 書いてよいか・外すフレーム = $OKW"
set -- $OKW
if [ "$1" != yes ]; then log "check が不合格なので書きません"; exit 1; fi
# 4) write(D1・D3。D2 は外すフレームがあるときだけ future_valid を外す)
if [ "$N" != none ] || [ "$2" != 0 ]; then
  if ! python -I -c "import json,sys;sys.exit(0 if json.load(open('$DS/dataset_card.json')).get('relabel') else 1)"; then
    $HEAVY --mem 1500M --lock none -- nice -n 10 python3 -I $E/relabel/relabel_dataset.py --dataset-dir $DS --teacher $T --noise $N \
      --mode write --jobs 3 --reuse-work --reference-scenes $REF --reference-stride 37 --report $OUT/relabel.json > $OUT/relabel_write.log 2>&1 \
      || { log "付け替え write が失敗(relabel_write.log)"; exit 1; }
  fi
  log "付け替え write 済み"
fi
# 5) ずれの統計(図の元)
$HEAVY --mem 1500M --lock none -- nice -n 10 node $E/record/deviation_stats.mjs --dataset-dir $DS --teacher $T --work $E/relabel/work/$NAME \
  --examples 3 --out $OUT/deviation.json > $OUT/deviation.log 2>&1 || { log "ずれの統計が失敗(deviation.log)"; exit 1; }
log "終わり $D"
