"""直す 2 回目・2 組目(r2-rec2-2)の確かめ: 本物の 10/23・10/24(読むだけ)で、発進のフレームの場所で決めた LH の入口
(_lh_self)と、周で決めた LH の入口(x_lh_m・v_launch_at_lh)がどれだけ合うか。JSON は tmp にだけ書く(消してよい場所)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix3/fix_r2_2/probe_lh_real.py <tmp>
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
COL = JF / 'experiments/20261026_fix/diag_launch/collect.py'
tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('collect_probe_lh', str(COL))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
got = []
orig = m.launch_rows


def wrap(d, laps_ref, kind):
    rs = orig(d, laps_ref, kind)
    got.extend(rs)
    return rs


m.launch_rows = wrap
import contextlib, io  # noqa: E401,E402
buf = io.StringIO()
with contextlib.redirect_stdout(buf):      # 1 日ずつ(2 日を一度に渡すと引数なしと同じ = 10/23・10/24 の前の読み方)
    m.main(['--day', '20261023', '--out', str(tmp / 'l23.json')])
    m.main(['--day', '20261024', '--out', str(tmp / 'l24.json')])
rows = [r for r in got if r.get('to_r1_s') is not None]
both = [r for r in rows if r.get('v_launch_at_lh') is not None and r.get('_lh_self')]
none_ = [r for r in rows if r.get('fly_ref') == 'none']
dx = [r['_lh_self']['x_m'] - r['x_lh_m'] for r in both if r.get('x_lh_m') is not None]
dv = [r['_lh_self']['v_mps'] - r['v_launch_at_lh'] for r in both]
out = {'rows_r1': len(rows), 'both': len(both), 'self_missing': sum(1 for r in rows if not r.get('_lh_self')),
       'fly_none': len(none_), 'fly_none_with_self': sum(1 for r in none_ if r.get('lh_self')),
       'dx_m': {'median': statistics.median(dx), 'min': min(dx), 'max': max(dx)} if dx else None,
       'dv_mps': {'median': statistics.median(dv), 'min': min(dv), 'max': max(dv),
                  'abs_le_0p1': sum(1 for x in dv if abs(x) <= 0.1), 'abs_le_0p2': sum(1 for x in dv if abs(x) <= 0.2)} if dv else None,
       'rows': [{'log': r['log'], 't_ai': r['t_ai'], 'fly_ref': r.get('fly_ref'), 'x_lh_m': r.get('x_lh_m'),
                 'v_lap': r.get('v_launch_at_lh'), 'self': r.get('_lh_self'), 'dist_m': r.get('dist_m')} for r in rows]}
(JF / 'experiments/20261026_fix/fix3/fix_r2_2/out/probe_lh_real.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False))
for r in out['rows']:
    print(r['log'], r['t_ai'], r['fly_ref'], 'lap x', r['x_lh_m'], 'v', r['v_lap'], '| self', r['self'], 'dist', r['dist_m'])
