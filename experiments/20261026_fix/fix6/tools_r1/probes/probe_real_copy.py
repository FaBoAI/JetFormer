"""直す 6-1 回目・係 tools: 本物の 10/24 20261024-134607(シーン 0001 は select がすべて False・AI 側は 27.9 秒 = シーン 0002)の
npz・json だけを tmp に写し(読むだけ)、A: そのまま / B: シーン 0002 に npz が無く CAM_FRONT/ だけ・0003 から無い(書いている途中で
落ちた形)を、⑤ collect.py の直す前の写しと今で --day 20261024 --ignore-gain で読む(反証の確かめ v1 の real と同じ形)。
tmp は fix6/tmp_tools/r1_real の下で、最後に消す。出力 ../out/probe_real_copy.json"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
from test_fix3_r1_2 import load_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix6/tmp_tools/r1_real'
REAL = JF / 'data/raw'
LOG = '20261024-134607'
PATHS = {'before': HERE.parent / 'before' / 'collect.py.orig', 'after': JF / 'experiments/20261026_fix/diag_launch/collect.py'}


def copy_real(dst, lost):
    src = REAL / LOG
    d = dst / LOG
    d.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, d / f.name)
    for s in sorted(p for p in src.iterdir() if p.is_dir()):
        idx = int(s.name[-4:])
        if lost and idx >= 3:
            continue
        sd = d / s.name
        sd.mkdir()
        if lost and idx == 2:
            (sd / 'CAM_FRONT').mkdir()
            continue
        for f in s.iterdir():
            if f.is_file():
                shutil.copy2(f, sd / f.name)


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    res = {}
    try:
        for case, lost in (('A_intact', False), ('B_lost', True)):
            raw = TMP / case / 'raw'
            copy_real(raw, lost)
            for ver, path in PATHS.items():
                m = load_mod('real_%s_%s' % (case, ver), path)
                m.RAW, m.LAB, m.BENCH, m.OUT = raw, TMP / case / 'nolab', JF / 'data' / 'bench_candidates.json', TMP / case / 'out'
                o = TMP / case / ('%s.json' % ver)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    m.main(['--day', '20261024', '--out', str(o), '--ignore-gain'])
                j = json.loads(o.read_text(encoding='utf-8'))
                res['%s_%s' % (case, ver)] = {
                    'launch_rows': len(j['launches']), 'unreadable': j.get('unreadable'), 'problems': j.get('problems'),
                    'summary_groups': len(j.get('summary') or []),
                    'group_unreadable': [g.get('unreadable') for g in j.get('summary') or []],
                    'lines': [x for x in buf.getvalue().splitlines() if x.startswith(('logs', 'skip', '記録が欠けている本'))
                              or '読めない' in x or LOG in x][:12]}
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    out = HERE.parent / 'out' / 'probe_real_copy.json'
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
