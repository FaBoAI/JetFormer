"""fix8 係 safety 直す 8-1 回目(rs8-1-backearly-rear-faster)の試験の値を選ぶ台本: tests/test_rec_fix7_safety.py の閉じたループ
(場面 brake = 10/23 の形。本物の drive._loop・偽物の ESC と車)で、後ろの物を止まった所から d [m] に置き、見張り・残す ON(1・1)と
OFF(0・0)の最初の後ろの当たりの速さ・物なしの下がる距離・書いた値の並びを出す。ESC は既定の模型(中立 0.1 秒・後退側から
0.08 秒)と (0.12, 0.10)。車両の値は 100 Hz。
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/fix_safety_r1/brake_rear12.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_rec_fix8_safety_r1 as T  # noqa: E402

TMP = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_safety_r1' / 'brake'
DS = [round(0.1 + 0.05 * k, 2) for k in range(35)]


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    out = {'ds': DS, 'esc': {}}
    try:
        for esc in (None, (0.12, 0.10)):
            key = 'default' if esc is None else '%s_%s' % esc
            with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
                tp = Path(td)
                sim1, rec1, pca1 = T._brake_run(mp, tp, 1, esc=esc)
                sim0, rec0, pca0 = T._brake_run(mp, tp, 0, esc=esc)
                x0 = T._stop_x(sim1)
                x0b = T._stop_x(sim0)
                row = {'stop_x_on': x0, 'stop_x_off': x0b, 'seq_on': T._phases(pca1, 1.5)[1], 'seq_off': T._phases(pca0, 1.5)[1],
                       'back_on': round(x0 - min(x for t, x, v, th in sim1.log if t > 2.0), 3),
                       'back_off': round(x0b - min(x for t, x, v, th in sim0.log if t > 2.0), 3),
                       'vmax_on': round(-min(v for t, x, v, th in sim1.log), 3),
                       'vmax_off': round(-min(v for t, x, v, th in sim0.log), 3),
                       'guards_on': [e[1]['what'] for e in rec1.events if e[0] == 'recovery_guard'], 'hits': {}}
                for d in DS:
                    rear = round(2.4 - (x0 - d), 4)
                    on = T._rear_hit(T._brake_run(mp, tp, 1, rear=rear, esc=esc)[0])
                    off = T._rear_hit(T._brake_run(mp, tp, 0, rear=rear, esc=esc)[0])
                    row['hits'][str(d)] = [on, off]
                out['esc'][key] = row
                print(key, {k: v for k, v in row.items() if k != 'hits'}, flush=True)
                for d in DS:
                    print('   d %.2f  ON %s  OFF %s' % (d, *row['hits'][str(d)]), flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    (HERE / 'out' / 'brake_rear12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
