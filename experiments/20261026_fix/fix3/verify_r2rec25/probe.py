"""r2-rec2-5 の確かめ(作り物の記録だけ。data/raw は読まない): 回 3 の ON の本(3-4)が 3.0 秒に 9G で当たり、
D = 電源断の形(10/23 の 20261023-204825 と同じ: log.json・scene.json・npz が全部 0 バイト、jpg だけ残る)
B = サーバーだけ落ちた形(log.json はある・その本の npz が無い)
で、⑤ collect.py --day と ② runs_1103.py が何を出すか"""
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402

RUNS = JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'


def runs_1103(D, tag):
    spec = importlib.util.spec_from_file_location('runs_' + tag, str(RUNS))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    fake = D.tmp / 'jf'
    (fake / 'data').mkdir(parents=True, exist_ok=True)
    (fake / 'data' / 'bench_candidates.json').write_text(D.bench.read_text(encoding='utf-8'), encoding='utf-8')
    m.JF, m.RAW = fake, D.raw
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ['runs_1103.py', '--day', '20261103']
    try:
        with contextlib.redirect_stdout(buf):
            m.main()
    finally:
        sys.argv = old
    return buf.getvalue()


def make(tmp, name, cut=None):
    d = tmp / name
    d.mkdir(parents=True)
    D = T.Day(d)
    made = []
    for i in range(8):
        kw = dict(hits=[(3.0, 9.0)], stop_after=3.1) if i == 3 else {}
        made.append(D.add(12 + i, T.ON if i % 2 else T.PRM, **kw))
    bad, ld = made[3]
    if cut:
        cut(ld)
    j, out, txt, _ = D.run(name='collect_' + name)
    g = j['summary'][0]
    r2 = runs_1103(D, name)
    return {'bad': bad,
            'files_after_cut': sorted('%s:%d' % (p.relative_to(ld), p.stat().st_size) for p in ld.rglob('*') if p.is_file()
                                      and p.suffix != '.jpg')[:12],
            'collect_by_launch_n': {k: v['n'] for k, v in g['by_launch'].items()},
            'collect_all_ok': (g.get('compare') or {}).get('all_ok'),
            'collect_bad_in_json': bad in txt,
            'collect_stdout_lines_total': len(out.splitlines()),
            'collect_skip_line_no': [i for i, x in enumerate(out.splitlines()) if x.startswith('skip')],
            'collect_stdout_skip': [x for x in out.splitlines() if x.startswith('skip')],
            'collect_verdict': [x.strip()[:160] for x in out.splitlines() if 'ON − OFF' in x],
            'runs1103_bad_rows': [x for x in r2.splitlines() if bad in x],
            'runs1103_tail': r2.splitlines()[-3:]}


def cut_power(ld):
    """電源断: 書いて 30 秒以内のファイルは 0 バイト(10/23 204825 と同じ)。jpg は残す"""
    for p in ld.rglob('*'):
        if p.is_file() and p.suffix != '.jpg':
            p.write_bytes(b'')


def cut_server(ld):
    for p in ld.glob('*/*.npz'):
        p.unlink()


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    res = {'A_kept': make(tmp, 'a'), 'D_power_loss': make(tmp, 'd', cut_power), 'B_server_crash': make(tmp, 'b', cut_server)}
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
