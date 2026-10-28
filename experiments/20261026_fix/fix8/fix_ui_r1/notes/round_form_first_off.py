"""気になること(直していない)の確かめ: ② runs_1103 の回の形の印(round_form。計画が null の項目 = その日に決める)は、今も回の最初の本と
比べる。回 3 の最初の本が起き直しのまま(左 4.1)で、3-2 から入れ直した(左 4.2)日に印がどの本に付くか。作り物の log.json だけ。
出力 out_round_form_first_off.txt"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/tests')
import test_ui_fix8_r1 as T  # noqa: E402

HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp(dir=HERE))
G42 = dict(T.G, ceil_left_mps=4.2, speed_ceiling=1)
G41 = dict(T.G, ceil_left_mps=4.1, speed_ceiling=1)
lines = []
for name, recs in (('3-1 が起き直しのまま(左 4.1・見張り OFF)・3-2 から入れ直し', [('3-1', {'ceil_left_mps': 4.1, 'speed_ceiling': 1}), ('3-2', G42), ('3-3', G42)]),
                   ('3-1 だけ左を入れ忘れ(4.1)', [('3-1', G41), ('3-2', G42), ('3-3', G42)])):
    raw = tmp / ('raw_%d' % len(lines))
    T._write_day(raw, recs)
    table, rows = T._runs(T.RUNS, raw, 'rf_%d' % len(lines))
    lines.append('## ' + name)
    for r in rows:
        lines.append('  %s: %s' % (r['run_id'], '・'.join(r['diffs']) or 'なし'))
    lines += ['  ' + x for x in table if ' 本・計画と違う ' in x]
import shutil  # noqa: E402
shutil.rmtree(tmp)
(HERE / 'out_round_form_first_off.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
