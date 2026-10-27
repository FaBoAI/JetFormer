#!/bin/bash
# fix6 係 safety 直す 6-2 回目: grid2.py の組を 5 本ずつ並べて流す(jf-heavy --lock none --mem 1500M。読むだけ)
cd "$(dirname "$0")"
PY=/home/jetson/miniconda/envs/jetracer/bin/python
jobs=("$@")
printf '%s\n' "${jobs[@]}" | xargs -P 5 -I{} bash -c 'set -- {}; R2_TAG="$1_${2//\//-}" /home/jetson/bin/jf-heavy --lock none --mem 1500M --need-mb 800 -- '"$PY"' -I grid2.py "$1" "$2" > "out/log_${R2_OUT:-grid2}_$1_${2//\//-}.txt" 2>&1'
