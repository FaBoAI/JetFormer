"""直す 6-2 回目・係 tools: 反証 6-2 回目 w2-tools-r62-1 の確かめ(⑤ collect.py。電源断で events.jsonl の start も消えた本)。

  jf-heavy --lock none --mem 1500M -- python3 -I probe_powerloss.py before|after

作り物の回 3(OFF 4 本・ON 3 本。fix6/tools_r1/probes/probe_lost_ai.py・反証の p1_powerloss_c.py と同じ)の 3-8 を次の形にする:
  C_power      ON の 80 秒の本。1 回目の発進 1.4 秒(読める・最初の周はきれい)・20 秒に人側で止める・60 秒に 2 回目の発進・68 秒に
               電源断。シーン 0002・0003 の npz・scene.json が 0 バイト、0004 は CAM_FRONT/ だけ。events.jsonl は 1.4 秒の start だけ
  C_power_nul  C_power と同じで、60 秒の start の行が NUL バイトで残った形(ext4 の i_disksize だけ伸びた)
  C_power_ev   C_power と同じで、60 秒の start の行が残った形(前からの fix6 の決まりで読めない記録になる)
  C_power_off  C_power と同じ形の OFF の本(log.json・計画とも OFF・events.jsonl の出来事なし)→ 前と同じのはず
  C_proc       fix6 の C(jetformer-web の落ち: 45 秒の start が残り、シーン 0003 が無い)→ 前と同じのはず
  C_run        ON の 60 秒の本。1 回目の発進 1.4 秒・AI 側のまま走り、45 秒に電源断の形(シーン 0003 が CAM_FRONT/ だけ・0002 は 0 バイト)
               = 最後の読めるフレーム(20 秒)は AI 側で走っている。最初の周は記録の中
  C_mid        ON の 60 秒の本。1 回目の発進 1.4 秒・人側 20 秒。シーン 0002 だけ npz が無い(真ん中。0003 は読める・start なし)
               → 前と同じのはず(電源断の形ではない。欠けたシーンの間の start が無ければ前からの決まりのまま)
  C_skip       C_power で、events.jsonl に 60 秒の skip(見送り)だけが残った形
  C_midrun     C_power と同じ形で、log.json の launch_boost は 0・events.jsonl の params で 30 秒に launch_boost 0 → 1(途中で ON)
各形を --ignore-gain と --ignore-gain --first-only で流す。作り物は fix6/tmp_tools_r2/probe の下に作って最後に消す。
本物の data/raw・labels は読まない。出力 ../out/probe_powerloss_<before|after>.json
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
TMP = JF / 'experiments/20261026_fix/fix6/tmp_tools_r2/probe'
PATHS = {'before': HERE.parent / 'before' / 'collect.py.orig',
         'after': JF / 'experiments/20261026_fix/diag_launch/collect.py'}
ON = dict(T.G, launch_boost=1)


def add_event(D, log, t, event='start'):
    with open(D.raw / log / 'events.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'launch_boost', 'event': event, 'us': 250.0, 'cam_t_us': T0_US + int(t * 1e6),
                            't_us': T0_US + int(t * 1e6) + 500}) + '\n')


def power_cut(D, log, lost, zero):
    sc = lambda k: D.raw / log / ('%s-%04d' % (log, k))      # noqa: E731
    for p in sc(lost).glob('*.npz'):       # 今のシーン(書く前に落ちた): CAM_FRONT/ だけ
        p.unlink()
    (sc(lost) / 'CAM_FRONT').mkdir(exist_ok=True)
    for k in zero:                         # 落ちる前 30 秒の中に保存したシーン: scene.json まで 0 バイト
        for p in list(sc(k).glob('*.npz')) + [sc(k) / 'scene.json']:
            if p.exists():
                p.write_bytes(b'')


def day(root, case):
    D = T.RealDay(root)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, ON, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, ON, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, ON, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    if case == 'C_proc':
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        add_event(D, log, 45.0)
        power_cut(D, log, 3, ())
    elif case == 'C_run':
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, **T.on_kw(42))
        power_cut(D, log, 3, (2,))
    elif case == 'C_mid':
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        power_cut(D, log, 2, ())
    elif case == 'C_power_off':
        log, _ = D.add_real(19, T.G, T.HAND, dur=80.0, human_after=20.0, stop_after=20.0, t_ai=1.4, t_move=1.5)
        power_cut(D, log, 4, (2, 3))
    else:
        p = dict(ON, launch_boost=0) if case == 'C_midrun' else ON
        log, _ = D.add_real(19, p, T.HAND, dur=80.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        if case == 'C_midrun':            # 1.4 秒の start は消して、30 秒に params で 0 → 1(途中で ON)だけ
            (D.raw / log / 'events.jsonl').write_text(json.dumps(
                {'kind': 'params', 'changed': {'launch_boost': [0, 1]}, 'epoch_us': T0_US + 30_000_000}) + '\n', encoding='utf-8')
        if case in ('C_power_ev', 'C_power_nul'):
            add_event(D, log, 60.0)
        if case == 'C_skip':
            add_event(D, log, 60.0, 'skip')
        if case == 'C_power_nul':
            ef = D.raw / log / 'events.jsonl'
            ls_ = ef.read_bytes().split(b'\n')
            ef.write_bytes(ls_[0] + b'\n' + b'\x00' * (len(ls_[1]) + 1))
        power_cut(D, log, 4, (2, 3))
    return D, log


def run(D, name, path, args):
    m = load_mod(name, path)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('%s.json' % name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue(), out.read_text(encoding='utf-8')


CASES = ('C_power', 'C_power_nul', 'C_power_ev', 'C_power_off', 'C_proc', 'C_run', 'C_mid', 'C_skip', 'C_midrun')


def main(which):
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {}
    try:
        for case in CASES:
            for args in (['--ignore-gain'], ['--ignore-gain', '--first-only']):
                tag = case + ('_first' if '--first-only' in args else '')
                root = TMP / tag
                root.mkdir()
                D, log = day(root, case)
                j, o, txt = run(D, 'pp_%s_%s' % (which, tag), PATHS[which], args)
                g = j['summary'][0] if j.get('summary') else {}
                c = g.get('compare') or {}
                res[tag] = {'log': log, 'all_ok': c.get('all_ok'), 'checks': c.get('checks'),
                            'on_n': (g.get('by_launch', {}).get('on') or {}).get('n'),
                            'off_n': (g.get('by_launch', {}).get('off') or {}).get('n'),
                            'unreadable': [{k: u.get(k) for k in ('log', 'why', 'launch_state', 'lost_scenes', 'launch_starts',
                                                                    'lost_after_s', 'launch_on_from')}
                                           for u in j.get('unreadable') or []],
                            'lines': [x for x in o.splitlines()
                                      if 'ON − OFF' in x or x.startswith(('記録が欠けている本', '  読めない', 'skip', '  → 読めない'))],
                            'json_sha': __import__('hashlib').sha256(txt.encode()).hexdigest(),
                            'stdout_sha': __import__('hashlib').sha256(
                                '\n'.join(x for x in o.splitlines() if not x.startswith('出力 ')).encode()).hexdigest()}
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        if TMP.parent.exists() and not any(TMP.parent.iterdir()):
            TMP.parent.rmdir()
    out = HERE.parent / 'out' / ('probe_powerloss_%s.json' % which)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k, 'on', v['on_n'], 'off', v['off_n'], 'all_ok', v['all_ok'], 'unreadable', [u['log'] for u in v['unreadable']])
        for x in v['lines']:
            print('   ', x[:300])


if __name__ == '__main__':
    main(sys.argv[1])
