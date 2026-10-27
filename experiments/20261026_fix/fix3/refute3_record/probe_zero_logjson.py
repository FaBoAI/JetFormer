"""反証 3 回目・係 record: 電源断の本物の形(10/23 の 20261023-204825: log.json も scene.json も npz も 0 バイト)の ON の本を、
⑤ collect.py --day --ignore-gain が「読めない記録」として出し「満たさない」にするか(r2-rec2-5 が本当に直ったか)。読むだけ・tmp だけ。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/refute3_record/probe_zero_logjson.py <tmp の場所>

回 3: OFF 3 本・ON 3 本(3-1〜3-6)。3-4(ON)は発進のあと 3.0 秒に 9G で当たり、そのあと Jetson が落ちた。
  A: npz だけ消す(r2-rec2-5 の試験と同じ形。log.json は残る)
  B: 本物の 10/23 の形: log.json・scene.json・npz が全部 0 バイト
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def build(tmp, name, shape):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    D.add(12, T.PRM)
    D.add(13, T.ON)
    D.add(14, T.PRM)
    bad, bd = D.add(15, T.ON, hits=[(3.0, 9.0)])
    D.add(16, T.PRM)
    D.add(17, T.ON)
    for f in bd.rglob('*'):
        if f.is_file() and f.suffix in ('.npz', '.json'):
            if shape == 'npz_only' and f.name == 'log.json':
                continue
            if shape == 'npz_only':
                f.unlink()
            else:
                f.write_bytes(b'')
    j, out, _, _ = D.run(name='z_' + name, args=['--ignore-gain'])
    keep = [x for x in out.splitlines() if 'ON − OFF' in x or '読めない' in x or x.startswith('skip') or x.startswith('logs ')]
    return {'bad_log': bad, 'lines': keep, 'unreadable_json': j.get('unreadable'),
            'compare': [g.get('compare') for g in j['summary']]}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    res = {'A_npz_removed_logjson_kept': build(tmp, 'a', 'npz_only'),
           'B_real_power_loss_shape_all_zero_bytes': build(tmp, 'b', 'zero')}
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_zero_logjson.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k, v['bad_log'])
        for x in v['lines']:
            print('  ', x)
        print('   unreadable(JSON):', v['unreadable_json'])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
