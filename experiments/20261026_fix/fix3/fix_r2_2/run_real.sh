#!/bin/bash
# 直す 2 回目・2 組目(2026-10-27): 本物の記録 10/23・10/24 で ⑤ collect.py と car_check を、直す前(before/*.orig)と後で流す。読むだけ。
#   bash run_real.sh <作業場所(消してよい)>
# 作業場所にだけ書く(data/raw・data/labels・data/reports・diag_launch/out には書かない。--out・--json で作業場所へ)。
# GPU なし(jf-heavy --lock none --mem 1500M)。
set -u
W=$1
JF=/home/jetson/jetformer
B=$JF/experiments/20261026_fix/fix3/fix_r2_2/before
PY=/home/jetson/miniconda/envs/jetracer/bin/python
H="/home/jetson/bin/jf-heavy --lock none --mem 1500M --"
mkdir -p "$W/old_root/tools" "$W/out_old" "$W/out_new"
# 直す前の根: tools は本物への symlink、car_check だけ写し(car_check は自分の dir の section_scorecard・label_events を読む)
for f in $JF/tools/*.py; do ln -sf "$f" "$W/old_root/tools/$(basename $f)"; done
rm -f "$W/old_root/tools/car_check.py"; cp "$B/car_check.py.orig" "$W/old_root/tools/car_check.py"
ln -sfn $JF/jetformer "$W/old_root/jetformer"
cp "$B/collect.py.orig" "$W/old_root/collect.py"
run() {   # run <old|new> <root> <collect>
  local tag=$1 R=$2 COL=$3 O=$W/out_$1
  cd "$O" || exit 1
  for d in 20261023 20261024; do
    for v in "" "--first-only" "--ignore-gain"; do
      n=${v#--}; n=${n:-plain}
      $H $PY -I "$COL" --day $d $v --out "$O/launches_${d}_$n.json" > "$O/launches_${d}_$n.stdout" 2> "$O/launches_${d}_$n.stderr"; echo "$tag collect $d $n rc=$?"
    done
  done
  $H $PY -I "$COL" --out "$O/launches.json" > "$O/launches.stdout" 2> "$O/launches.stderr"; echo "$tag collect default rc=$?"
  mkdir -p "$O/car_check"
  : > "$O/car_check/rc.txt"
  for lg in $(ls $JF/data/raw | grep -E '^2026102[34]-' ); do
    [ -f "$JF/data/raw/$lg/log.json" ] || continue
    $H $PY -I "$R/tools/car_check.py" "$lg" --json "$O/car_check/$lg.json" > "$O/car_check/$lg.txt" 2>&1; echo "$lg $?" >> "$O/car_check/rc.txt"
  done
  echo "$tag car_check done $(wc -l < $O/car_check/rc.txt)"
}
run old "$W/old_root" "$W/old_root/collect.py"
run new "$JF" "$JF/experiments/20261026_fix/diag_launch/collect.py"
