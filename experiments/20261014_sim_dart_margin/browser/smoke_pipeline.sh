#!/bin/bash
# 準備の段の試し(ブラウザの記録のあと、CPU だけ): 検証 → 一時の JetFormer の data に取り込み → 付け替え → validate_dataset → 一時の data を消す
# (2026-10-14、sim_dart_20261014)。本番の data/ には何も書かない。
#   jf-heavy --mem 1500M --lock none -- bash smoke_pipeline.sh <run_browser_smoke.sh の結果のフォルダー>
set -u
OUT=${1:?結果のフォルダー}
E=/home/jetson/jetformer/experiments/20261014_sim_dart_margin
R=/home/jetson/jetformer_reports/sim_dart_20261014
TMP=$(mktemp -d /tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad/dart_build/smoke_jfdata.XXXX)
source /home/jetson/miniconda/etc/profile.d/conda.sh && conda activate jetracer
echo "一時の data: $TMP" > "$OUT/pipeline.txt"
for D in D1 D2 D3; do
  RES=$OUT/$D/result.json
  [ -f "$RES" ] || { echo "$D: result.json が無い" >> "$OUT/pipeline.txt"; continue; }
  nice -n 10 python -I $E/record/verify_dart_collection.py --dataset $D --result $RES --out-dir $OUT/$D/verify --workers 1 > $OUT/$D/verify.log 2>&1
  echo "$D 検証 $?" >> "$OUT/pipeline.txt"
  ZIP=$(python -I - "$RES" <<'EOF'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
c = r.get('collection') or {}
cid = str(c.get('download_url', '')).rsplit('/', 1)[-1]
z = Path('/home/jetson/jetformer_reports/sim_dart_20261014/sim/runtime/datasets') / cid / c.get('filename', '')
print(z if z.is_file() else '')
EOF
)
  if [ -z "$ZIP" ]; then
    # アプリのまとめが空き容量の取り置き(20 GiB)で止まったときは、取り置きだけ小さくした同じ処理でまとめる(record/build_collection.py)
    case $D in D1) CN=smoke-rl-dart-fast;; D2) CN=smoke-rl-margin-line;; D3) CN=smoke-rl-margin-dart;; esac
    python -I $E/record/build_collection.py --result $RES --name $CN --out $OUT/$D/collection.json > $OUT/$D/collection.log 2>&1
    echo "$D まとめ(build_collection) $?" >> "$OUT/pipeline.txt"
    ZIP=$(python -I -c "import json;print(json.load(open('$OUT/$D/collection.json'))['zip'])" 2>/dev/null)
  fi
  [ -n "$ZIP" ] || { echo "$D: 集約 ZIP が無い" >> "$OUT/pipeline.txt"; continue; }
  echo "$D ZIP $ZIP" >> "$OUT/pipeline.txt"
  python -I $E/record/import_dataset.py "$ZIP" --dataset $D --out $OUT/$D/import.json --temp-data-dir $TMP --to-scenes no > $OUT/$D/import.log 2>&1
  echo "$D 取り込み $?" >> "$OUT/pipeline.txt"
  NAME=$(python -I -c "import json;print(json.load(open('$OUT/$D/import.json'))['name'])")
  DS=$TMP/datasets/$NAME
  case $D in D1) T=fast; N=N_fast; MODE=write;; D2) T=margin; N=none; MODE=check;; D3) T=margin; N=N_margin; MODE=write;; esac
  # 基準の作り方(O(N²))とも比べる。--states replay では fork と基準が同じ状態から始まるので差 0 のはず
  nice -n 10 python -I $E/relabel/relabel_dataset.py --dataset-dir $DS --teacher $T --noise $N --mode check --states replay --jobs 1 \
    --reference-scenes 1 --reference-stride 7 --work $OUT/$D/relabel_ref_work --report $OUT/$D/relabel_reference_replay.json > $OUT/$D/relabel_ref.log 2>&1
  echo "$D 基準との比べ(replay) $?" >> "$OUT/pipeline.txt"
  nice -n 10 python -I $E/relabel/relabel_dataset.py --dataset-dir $DS --teacher $T --noise $N --mode $MODE --jobs 1 \
    --reference-scenes 1 --reference-stride 7 --work $OUT/$D/relabel_work --backup $OUT/$D/relabel_backup --report $OUT/$D/relabel.json > $OUT/$D/relabel.log 2>&1
  echo "$D 付け替え($MODE) $?" >> "$OUT/pipeline.txt"
  JETFORMER_DATA=$TMP python -I - "$DS" "$OUT/$D/validate_after.json" <<'EOF' >> "$OUT/pipeline.txt" 2>&1
import json, sys
from pathlib import Path
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import config
from jetformer.dataset_import import validate_dataset
assert str(config.DATA_DIR).startswith('/tmp/'), config.DATA_DIR
e, w, s = validate_dataset(Path(sys.argv[1]))
card = json.loads((Path(sys.argv[1]) / 'dataset_card.json').read_text())
Path(sys.argv[2]).write_text(json.dumps({'errors': e, 'warnings': w, 'summary': s, 'relabel': card.get('relabel'), 'label_mask': card.get('label_mask')}, ensure_ascii=False, indent=1))
print('validate_dataset', sys.argv[1].rsplit('/', 1)[-1], 'errors', e, 'summary', s)
EOF
done
du -sh $TMP >> "$OUT/pipeline.txt"
rm -rf "$TMP"
echo "一時の data を消した: $TMP $( [ -e $TMP ] && echo 残っている || echo 消えた)" >> "$OUT/pipeline.txt"
