#!/usr/bin/env python3
"""10/27 fix4(小さな残り): tests/test_fix4_small.py がわざと壊したものを見つけるか(本物のファイルは変えない)。

写しの木(fix4/tmp_mut/tree。jetformer/ と試験の 3 本・写し 2 つ。data は本物への symlink = 読むだけ)に 1 か所ずつ壊しを入れて、
写しの木の tests/test_fix4_small.py だけを流す。対照(壊さない)は全部通る・壊しは 1 つ以上落ちる、を見る。終わったら木を消す。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix4/mutate_small.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
WORK = HERE / 'tmp_mut'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
OUT = HERE / 'out' / 'mutate_small.txt'

BENCH = 'jetformer/web/bench.py'
DRIVE = 'jetformer/drive.py'
APP = 'jetformer/web/static/app.js'
OLD_SW = "SWITCH_KEYS = ('zone_caps', 'speed_ceiling', 'speed_auto', 'launch_boost', 'rec_back_guard', 'rec_hold_keep', 'launch_no_kick')"
MUTANTS = [
    ('control', None, None, None),
    ('bench を直す前に戻す', BENCH, 'COPY', JF / 'tests/fixtures/fix4_small_before/bench.py'),
    ('SWITCH_KEYS から 3 つを外す', BENCH, OLD_SW, "SWITCH_KEYS = ('zone_caps', 'speed_ceiling', 'speed_auto', 'launch_boost')"),
    ('SWITCH_KEYS から rec_hold_keep だけ外す', BENCH, OLD_SW, OLD_SW.replace(", 'rec_hold_keep'", '')),
    ('名前を変える', BENCH, "'rec_back_guard': 'リカバリーの後ろの見張り', 'rec_hold_keep'",
     "'rec_back_guard': 'リカバリーの見張り', 'rec_hold_keep'"),
    ('基準の警告を外す', BENCH, "        if base['params'].get(k):\n            warnings.append('baseline.params の %s が 1 です。画面の「基準の設定」は「%s」",
     "        if False:\n            warnings.append('baseline.params の %s が 1 です。画面の「基準の設定」は「%s」"),
    ('基準で 0 にしてしまう', BENCH, "    for k in OPT_KEYS:          # 試験の選択肢",
     "    for k in OPT_KEYS:\n        base['params'][k] = 0\n    for k in OPT_KEYS:          # 試験の選択肢"),
    ('PLAN_KEYS に足す', BENCH, "             'speed_ceiling', 'speed_auto')\n", "             'speed_ceiling', 'speed_auto', 'rec_back_guard')\n"),
    ('ブロックの値も捨てる(launch_us)', BENCH, "            if k not in drive.PARAM_LIMITS or k in SWITCH_KEYS:",
     "            if k not in drive.PARAM_LIMITS or k in SWITCH_KEYS or k == 'launch_us':"),
    ('drive.py のコードを 1 か所(HOLD_VEH_GAP_S)', DRIVE, 'HOLD_VEH_GAP_S = 0.06', 'HOLD_VEH_GAP_S = 0.07'),
    ('drive.py の既定値を 1 か所(rec_hold_keep 1)', DRIVE, "'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}",
     "'rec_back_guard': 0, 'rec_hold_keep': 1, 'launch_no_kick': 0}"),
    ('drive.py の説明を直す前に戻す', DRIVE, 'COPY', JF / 'tests/fixtures/fix4_small_before/drive.py'),
    ('app.js が基準の設定・ブロックで選択肢を消さない', APP, '    delete f.params[o.key];\n', '\n'),
    ('app.js が走行開始で後ろの見張りを送る', APP, "  p.launch_boost = $('#dr-launch-boost').checked ? 1 : 0;                                  // 発進の前進(2026-10-26)\n  return p;",
     "  p.launch_boost = $('#dr-launch-boost').checked ? 1 : 0;                                  // 発進の前進(2026-10-26)\n  p.rec_back_guard = 1;\n  return p;"),
]


def build():
    if WORK.exists():
        shutil.rmtree(WORK)
    t = WORK / 'tree'
    shutil.copytree(JF / 'jetformer', t / 'jetformer', ignore=shutil.ignore_patterns('__pycache__'))
    (t / 'tests' / 'fixtures').mkdir(parents=True)
    for f in ('tests/test_fix4_small.py', 'tests/test_ui_fix3.py', 'tests/fixtures/app_sim_fix3.js', 'pytest.ini'):
        shutil.copy2(JF / f, t / f)
    for d in ('tests/fixtures/fix4_small_before', 'tests/fixtures/ui_fix3_before'):
        shutil.copytree(JF / d, t / d)
    (t / 'data').symlink_to(JF / 'data')
    return t


def main():
    t = build()
    lines, bad = [], []
    for name, rel, old, new in MUTANTS:
        bak = None
        if rel:
            f = t / rel
            bak = f.read_bytes()
            if old == 'COPY':
                shutil.copy2(new, f)
            else:
                s = f.read_text(encoding='utf-8')
                assert s.count(old) == 1, (name, s.count(old))
                f.write_text(s.replace(old, new), encoding='utf-8')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('JETFORMER_DATA', None)
        chk = subprocess.run([PY, '-c', 'import sys; sys.path.insert(0, "%s"); import jetformer; print(jetformer.__file__)' % t],
                             capture_output=True, text=True, cwd=str(t), env=env)
        assert chk.stdout.strip().startswith(str(t)), chk.stdout
        r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::DeprecationWarning',
                            '--basetemp=%s' % (WORK / 'bt'), 'tests/test_fix4_small.py'],
                           capture_output=True, text=True, cwd=str(t), env=env, timeout=600)
        last = (r.stdout.strip().splitlines() or [''])[-1]
        failed = [x.split('::')[-1].split(' ')[0] for x in r.stdout.splitlines() if x.startswith('FAILED')]
        ok = (r.returncode == 0) if name == 'control' else (r.returncode == 1 and failed)
        if not ok:
            bad.append(name)
        lines.append('%-4s %s: rc=%d %s  落ちた: %s' % ('OK' if ok else 'NG', name, r.returncode, last, ', '.join(failed) or '—'))
        print(lines[-1], flush=True)
        if rel:
            (t / rel).write_bytes(bak)
    shutil.rmtree(WORK)
    lines.append('%d 通り(対照 1・壊し %d): 思いどおりでないもの %s' % (len(MUTANTS), len(MUTANTS) - 1, bad or 'なし'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(lines[-1])
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
