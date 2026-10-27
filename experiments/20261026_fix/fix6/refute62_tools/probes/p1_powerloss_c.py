"""反証 6-2 回目・係 tools: fix6 の C(読めた本の 2 回目の発進のシーンごと落ちた)を、電源断の本物の形で読む。

  python3 -I p1_powerloss_c.py

作り物の回 3 は fix6/tools_r1/probes/probe_lost_ai.py と同じ(OFF 4 本・ON 3 本・3-8 は ON。1 回目の発進 1.4 秒・人側 20 秒)。
3-8 の形:
  C_proc   fix6 の C そのまま(jetformer-web の落ち = ページキャッシュは残る): 45 秒の start が events.jsonl にある・シーン 0003 に npz が無い
  C_power  電源断の本物の形(10/24 の 20261024-162305: 落ちる 10 秒前・30 秒前に保存したシーンは scene.json まで 0 バイト・今のシーンは
           CAM_FRONT/ だけ)。80 秒の本・2 回目の発進 60 秒・その最初の周の 68 秒に落ちた: シーン 0002・0003 の npz・scene.json が
           0 バイト、0004 に npz が無い。events.jsonl の 60 秒の start は、同じ書き戻しの窓(dirty_expire 30 秒・add_event は fsync
           しない)なので残っていない(1 回目の 1.4 秒の start だけ)
  C_power_ev  C_power と同じで、60 秒の start だけ残った形(比べ: 違いは events.jsonl の 1 行だけ)
出力は ../out/p1_powerloss_c.json。作り物は fix6/tmp_refute62_tools の下に作って最後に消す。本物の data/raw・labels は読まない。
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
TMP = JF / 'experiments/20261026_fix/fix6/tmp_refute62_tools/p1'
COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'


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
    if case == 'C_proc':                   # fix6 の C そのまま
        log, _ = D.add_real(19, on, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        add_event(D, log, 45.0)
        lost, zero = 3, ()
    else:                                  # 電源断: 2 回目の発進 60 秒・最初の周の 68 秒に落ちた。0001(21 秒に保存 = 47 秒前)は残る、
        # 0002(41 秒 = 27 秒前)・0003(61 秒 = 7 秒前)は 0 バイト、0004 は CAM_FRONT/ だけ(本物の 20261024-162305 の 0007〜0009 と同じ形)
        log, _ = D.add_real(19, on, T.HAND, dur=80.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        if case == 'C_power_ev':
            add_event(D, log, 60.0)
        lost, zero = 4, (2, 3)
    sc = lambda k: D.raw / log / ('%s-%04d' % (log, k))      # noqa: E731
    for p in sc(lost).glob('*.npz'):       # 今のシーン(書く前に落ちた): CAM_FRONT/ だけ
        p.unlink()
    (sc(lost) / 'CAM_FRONT').mkdir()
    for k in zero:                         # 落ちる前 30 秒の中に保存したシーン: scene.json まで 0 バイト
        for p in list(sc(k).glob('*.npz')) + [sc(k) / 'scene.json']:
            if p.exists():
                p.write_bytes(b'')
    return D, log


def run(D, name, args):
    m = load_mod(name, COLLECT)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('%s.json' % name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue()


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {}
    try:
        for case in ('C_proc', 'C_power', 'C_power_ev'):
            root = TMP / case
            root.mkdir()
            D, log = day(root, case)
            ev = (D.raw / log / 'events.jsonl').read_text(encoding='utf-8').splitlines()
            j, o = run(D, 'p1_%s' % case, ['--ignore-gain'])
            g = j['summary'][0] if j.get('summary') else {}
            c = g.get('compare') or {}
            res[case] = {'log': log, 'events_lines': len(ev), 'on_n': (g.get('by_launch', {}).get('on') or {}).get('n'),
                         'off_n': (g.get('by_launch', {}).get('off') or {}).get('n'),
                         'unreadable': j.get('unreadable'), 'checks': c.get('checks'), 'all_ok': c.get('all_ok'),
                         'lines': [x for x in o.splitlines() if 'ON − OFF' in x or x.startswith(('記録が欠けている本', '  読めない', 'skip'))]}
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    out = HERE.parent / 'out' / 'p1_powerloss_c.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    for k, v in res.items():
        print('==', k, 'events', v['events_lines'], 'on', v['on_n'], 'off', v['off_n'], 'all_ok', v['all_ok'])
        for x in v['lines']:
            print('   ', x[:400])


if __name__ == '__main__':
    main()
