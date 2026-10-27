"""直す 6-1 回目・係 tools: ⑤ collect.py が、AI 側にしたシーンごと書けずに落ちた本をどう読むか(w1-tools-r6-1-tools-1)。

  python3 -I probe_lost_ai.py before|after
    before = 直す前の写し(../before/collect.py.orig)、after = 今の diag_launch/collect.py。出力は ../out/probe_lost_ai_<版>.json

作り物の回 3(反証の p1 と同じ: OFF 4 本・ON 3 本(起点まで 2.8・2.7・2.6 秒)・3-8 は ON で 25 秒に AI 側・31 秒に 9G)。
  A   : 3-8 の記録がそろっている
  B   : 3-8 のシーン 0002(20〜40 秒)に npz が無く CAM_FRONT/ だけ(反証の B)
  B0  : 3-8 のシーン 0002 の AI_CMD.npz だけ 0 バイト(ほかの npz はある。AI 側の 25 秒からのフレームが無い)
  B2  : B と同じ。ただし log.json の launch_boost は 0 で、events.jsonl の発進の前進の start だけが ON を示す(途中で ON にした形)
  C   : 3-8 は 1 回目の発進(1.4 秒・人側 20 秒)が読める。2 回目の発進(45 秒に AI 側・発進の前進の start)の入ったシーン 0003(40〜60 秒)
        に npz が無い(2 回目の発進が黙って消えるか)
  D   : 3-8 は AI 側にしていない本(人だけ)で、シーン 0002 の VEHICLE.npz が無い(AI_CMD はあり select はすべて False)
  E   : B と同じ形で、3-8 が OFF の本
作り物は fix6/tmp_tools/r1_probe の下に作って最後に消す。本物の data/raw・labels は読まない。
"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
import test_fix3_r3_2_tools as T  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402
from test_lap_anchor import T0_US  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix6/tmp_tools/r1_probe'
PATHS = {'before': HERE.parent / 'before' / 'collect.py.orig',
         'after': JF / 'experiments/20261026_fix/diag_launch/collect.py'}


def lose(D, log, scene, only=None, zero=False):
    sc = D.raw / log / ('%s-%04d' % (log, scene))
    for p in sc.glob('*.npz'):
        if only and p.stem not in only:
            continue
        if zero:
            p.write_bytes(b'')
        else:
            p.unlink()
    if not only and not zero:
        (sc / 'CAM_FRONT').mkdir()


def add_event(D, log, t, event='start'):
    with open(D.raw / log / 'events.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'launch_boost', 'event': event, 'us': 250.0, 'cam_t_us': T0_US + int(t * 1e6),
                            't_us': T0_US + int(t * 1e6) + 500}) + '\n')


def day(root, case):
    D = T.RealDay(root)
    on = dict(T.G, launch_boost=1)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, on, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, on, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, on, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    if case == 'C':
        log, _ = D.add_real(19, on, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        add_event(D, log, 45.0)
        lose(D, log, 3)
        return D, log
    if case == 'D':
        log, _ = D.add_real(19, on, T.HAND, dur=40.0, t_ai=99.0, t_move=99.0)
        lose(D, log, 2, only=('VEHICLE',))
        return D, log
    p = on if case != 'E' else T.G
    if case == 'B2':
        p = T.G
    kw = dict(dur=40.0, t_ai=25.0, t_move=25.1, first_anchor=28.0, hits=[(31.0, 9.0)])
    if case != 'E':
        kw['launch'] = (25.0, 0.5, 1)
    log, _ = D.add_real(19, p, T.HAND, **kw)
    if case == 'E':
        pass
    if case in ('B', 'B2', 'E'):
        lose(D, log, 2)
    elif case == 'B0':
        lose(D, log, 2, only=('AI_CMD',), zero=True)
    return D, log


def run(D, path, name, args):
    m = load_mod(name, path)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('%s.json' % name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue()


def main():
    ver = sys.argv[1]
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {}
    try:
        for case in ('A', 'B', 'B0', 'B2', 'C', 'D', 'E'):
            for args in (['--ignore-gain'], ['--ignore-gain', '--first-only']):
                key = case + ('_first' if '--first-only' in args else '')
                root = TMP / key
                root.mkdir()
                D, log = day(root, case)
                j, o = run(D, PATHS[ver], 'p_%s_%s' % (ver, key.replace('-', '_')), args)
                g = j['summary'][0] if j.get('summary') else {}
                c = g.get('compare') or {}
                res[key] = {
                    'log': log, 'rows_of_log': [(r['t_ai'], r.get('launch_state')) for r in j['launches'] if r['log'] == log],
                    'on_n': (g.get('by_launch', {}).get('on') or {}).get('n'),
                    'off_n': (g.get('by_launch', {}).get('off') or {}).get('n'),
                    'unreadable_json': j.get('unreadable'), 'group_unreadable': g.get('unreadable'),
                    'problems_logs': sorted(j.get('problems') or {}), 'checks': c.get('checks'), 'all_ok': c.get('all_ok'),
                    'lines': [x for x in o.splitlines() if x.startswith(('logs', 'skip', '記録が欠けている本', '  読めない', '  → 読めない'))
                              or 'ON − OFF' in x],
                    'log_named': [x for x in o.splitlines() if log in x and not x.startswith(log + ' No.')]}
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    out = HERE.parent / 'out' / ('probe_lost_ai_%s.json' % ver)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k, 'rows', v['rows_of_log'], 'on', v['on_n'], 'off', v['off_n'], 'all_ok', v['all_ok'])
        for x in v['lines']:
            print('   ', x[:260])


if __name__ == '__main__':
    main()
