#!/bin/bash
# 反証 8-2 rehearsal: 手順書 7 章「起こし直した」「走った値が計画と違った」の、形の違う ON の本(3-6 = 左 4.2 のまま・1 周目の R1 4.5G)を
# 「シーン」のタブの道で「試し」にする前と後で、⑤ の回 3 の組を流す(作り物の家 form_trial)。シーンの説明の直しは scene_path.py で確かめた
# 画面の PATCH と同じ形を、本物の dataset.update_scene で書く
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FH=/home/jetson/jetformer/experiments/20261026_fix/fix8/tmp_rehearsal_r2/form_trial/home
export HOME=$FH PATH=/home/jetson/miniconda/envs/jetracer/bin:$PATH TZ=Asia/Tokyo PYTHONDONTWRITEBYTECODE=1
R=~/jetformer_reports/bench_20261103_tune; mkdir -p $R
five() { ( cd ~/jetformer/experiments/20261026_fix/diag_launch && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I collect.py --day 20261103 --ignore-gain --out $R/l.json "$@" 2>/dev/null | sed -n '/11\/03 回3 B/,/11\/03 回4/p;/まとめから外した/p' | cut -c1-600 ); }
echo "## 試しにする前"; five
python3 -I -B - "$FH" <<'PY'
import sys, json
sys.path.insert(0, '/home/jetson/jetformer')
from pathlib import Path
from jetformer import config
config.RAW_DIR = Path(sys.argv[1]) / 'jetformer/data/raw'
from jetformer import dataset
nm = '20261103-093400-0001'
s = json.loads((config.RAW_DIR / nm[:15] / nm / 'scene.json').read_text(encoding='utf-8'))
d = s['description']; i = d.index('memo=')
r = dataset.update_scene(nm, d[:i + 5] + '試し(左 4.2 のまま)。' + d[i + 5:], s['tags'], None)
print('## 「シーン」のタブで 3-6 の %s に「試し(左 4.2 のまま)。」を足した' % nm)
PY
echo "## 試しにしたあと"; five
echo "## (比べ)main session が --skip 20261103-093400 で外したとき"; five --skip 20261103-093400
