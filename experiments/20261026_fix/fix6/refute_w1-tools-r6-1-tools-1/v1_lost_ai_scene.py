"""反証 w1-tools-r6-1-tools-1 の確かめ(独立)。⑤ collect.py が、AI 側のシーンごと npz の無い本を黙って飛ばすか。
(1) 本物の 10/24 20261024-134607(シーン 1 は select すべて False・AI 側は 27.9 秒 = シーン 2)の npz・json だけを tmp に写し、
    A: そのまま / B: シーン 2 の npz を消し CAM_FRONT/ だけ・シーン 3 以降なし(書いている途中で落ちた形)
(2) 作り物の回 3(test_fix3_r3_2_tools.RealDay)で ON の本 3-8 を同じ形にし、比べの答えが変わるか。
本物の data/raw・labels は読むだけ(写しは tmp)。tmp は最後に消す。"""
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

COLLECT = JF / 'experiments/20261026_fix/diag_launch/collect.py'
TMP = JF / 'experiments/20261026_fix/fix6/tmp_w1tools61'
HERE = Path(__file__).resolve().parent
REAL = JF / 'data/raw'


def run(raw, lab, bench, out, name, day, args=()):
    m = load_mod(name, COLLECT)
    m.RAW, m.LAB, m.BENCH, m.OUT = raw, lab, bench, out
    out.mkdir(parents=True, exist_ok=True)
    o = out / (name + '.json')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', day, '--out', str(o), '--ignore-gain'] + list(args))
    return json.loads(o.read_text(encoding='utf-8')), buf.getvalue()


def copy_real(log, dst, lost):
    src = REAL / log
    d = dst / log
    d.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, d / f.name)
    for s in sorted(p for p in src.iterdir() if p.is_dir()):
        idx = int(s.name[-4:])
        if lost and idx >= 3:
            continue                     # 落ちたあとのシーンは無い
        sd = d / s.name
        sd.mkdir()
        (sd / 'CAM_FRONT').mkdir()
        if lost and idx == 2:
            continue                     # 書いている途中: CAM_FRONT/ だけ
        for f in s.iterdir():
            if f.is_file():
                shutil.copy2(f, sd / f.name)


def real_case():
    res = {}
    log = '20261024-134607'
    for case, lost in (('A_intact', False), ('B_lost', True)):
        root = TMP / 'real' / case
        raw = root / 'raw'
        copy_real(log, raw, lost)
        j, o = run(raw, root / 'nolab', T.JF / 'data' / 'bench_candidates.json', root / 'out', 'real_' + case, '20261024')
        res[case] = {'n_launch_rows': len(j['launches']), 'unreadable': j.get('unreadable'),
                     'problems': j.get('problems'), 'summary_groups': len(j.get('summary') or []),
                     'stdout_interesting': [x for x in o.splitlines() if x.startswith(('logs', 'skip', '記録が欠けている本', '読めない'))
                                            or '読めない' in x or log in x][:12]}
    return res


def synth_day(root, lost):
    D = T.RealDay(root)
    on = dict(T.G, launch_boost=1)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, on, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, on, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, on, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    log, _ = D.add_real(19, on, T.HAND, dur=40.0, t_ai=25.0, t_move=25.1, launch=(25.0, 0.5, 1), first_anchor=28.0,
                        hits=[(31.0, 9.0)])
    if lost:
        sc = D.raw / log / ('%s-0002' % log)
        for p in sc.glob('*.npz'):
            p.unlink()
        (sc / 'CAM_FRONT').mkdir()
    return D, log


def synth_case():
    res = {}
    for case, lost in (('A_intact', False), ('B_lost', True)):
        root = TMP / 'synth' / case
        root.mkdir(parents=True)
        D, log = synth_day(root, lost)
        j, o = run(D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out', 'syn_' + case, '20261103')
        g = j['summary'][0]
        on = g['by_launch'].get('on') or {}
        res[case] = {'log': log, 'rows_for_log': [(r['t_ai'], r.get('launch_state')) for r in j['launches'] if r['log'] == log],
                     'on_n': on.get('n'), 'on_first_lap_collisions': on.get('collisions_first_lap'),
                     'unreadable': j.get('unreadable'), 'checks': (g.get('compare') or {}).get('checks'),
                     'all_ok': (g.get('compare') or {}).get('all_ok'),
                     'log_named_in_stdout': [x for x in o.splitlines() if log in x and not x.startswith(log + ' No.')],
                     'stdout_key': [x for x in o.splitlines() if x.startswith(('logs', 'skip', '記録が欠けている本'))
                                    or 'ON − OFF' in x or '読めない' in x]}
    return res


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    try:
        out = {'real': real_case(), 'synth': synth_case()}
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'v1_lost_ai_scene.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
