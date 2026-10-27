#!/bin/bash
# 直す 5 回目・係 tools(2026-10-27): 本物の記録 10/23・10/24 で ⑤ collect.py・③ standing_3lap・④ quick_eval_by_params・car_check を、
# 直す前(fix5/tools/before/*.orig)と後で流す。読むだけ。作業場所にだけ書く(data/raw・data/labels・data/reports・diag_launch/out には
# 書かない)。GPU なし。   bash run_real.sh <作業場所(消してよい)> <old|new>
set -u
W=$1; TAG=$2
JF=/home/jetson/jetformer
BF=$JF/experiments/20261026_fix/fix5/tools/before
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
O=$W/out_$TAG
mkdir -p "$O"
if [ "$TAG" = old ]; then
  mkdir -p "$W/old_root/tools"
  for f in $JF/tools/*.py; do ln -sf "$f" "$W/old_root/tools/$(basename $f)"; done
  for n in car_check.py standing_3lap.py quick_eval_by_params.py; do rm -f "$W/old_root/tools/$n"; cp "$BF/$n.orig" "$W/old_root/tools/$n"; done
  ln -sfn $JF/jetformer "$W/old_root/jetformer"
  ln -sfn $JF/data "$W/old_root/data"
  cp "$BF/collect.py.orig" "$W/old_root/collect.py"
  # collect.py は /home/jetson/jetformer/tools を読む → 写しの tools を先に読むように書き換える(写しだけ)
  sed -i "s#sys.path.insert(0, '/home/jetson/jetformer/tools')#sys.path.insert(0, '/home/jetson/jetformer/tools'); sys.path.insert(0, '$W/old_root/tools')#" "$W/old_root/collect.py"
  R=$W/old_root; COL=$W/old_root/collect.py
else
  R=$JF; COL=$JF/experiments/20261026_fix/diag_launch/collect.py
fi
cd "$O" || exit 1
for d in 20261023 20261024; do
  for v in "" "--first-only" "--ignore-gain"; do
    n=${v#--}; n=${n:-plain}
    $H $PY -I "$COL" --day $d $v --out "$O/launches_${d}_$n.json" > "$O/launches_${d}_$n.stdout" 2> "$O/launches_${d}_$n.stderr"; echo "$TAG collect $d $n rc=$?"
  done
  $H $PY -I "$R/tools/standing_3lap.py" --day $d --raw $JF/data/raw --labels $JF/data/labels --json "$O/standing_$d.json" --md "$O/standing_$d.md" > "$O/standing_$d.stdout" 2>&1; echo "$TAG standing $d rc=$?"
  for ig in none gain; do
    a=""; [ $ig = gain ] && a="--ignore gain"
    $H $PY -I "$R/tools/quick_eval_by_params.py" 0000 --day=$d $a --raw $JF/data/raw --labels $JF/data/labels --out "$O/by_params_${d}_$ig" > "$O/by_params_${d}_$ig.stdout" 2>&1; echo "$TAG by_params $d $ig rc=$?"
  done
  $H $PY -I "$R/tools/car_check.py" --latest --day $d --raw $JF/data/raw --labels $JF/data/labels --no-json > "$O/cc_latest_$d.stdout" 2> "$O/cc_latest_$d.stderr"; echo "rc=$?" > "$O/cc_latest_$d.rc"
done
$H $PY -I "$COL" --out "$O/launches.json" > "$O/launches.stdout" 2> "$O/launches.stderr"; echo "$TAG collect default rc=$?"
$H $PY -I $JF/experiments/20261026_fix/fix5/tools/cc_all.py "$R" "$O/cc" > "$O/cc_all.stdout" 2> "$O/cc_all.stderr"; echo "$TAG cc_all rc=$?"
echo "$TAG done"
