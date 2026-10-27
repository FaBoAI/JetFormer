"""直す 7-1 回目・係 safety: 前からの試験が落ちる理由の切り分け。jetformer/・tests/ を一時の場所に写し、写しの drive.py が
esc_rev を渡さない(s4-a を止める。s4-b はそのまま)形で、落ちた試験ファイルを流す(本物の jetformer/ は変えない)。出力は標準出力"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments' / '20261026_fix' / 'fix7' / 'tmp_safety' / 'probe_old'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'


def main(tests):
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    try:
        shutil.copytree(JF / 'jetformer', TMP / 'jetformer', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(JF / 'tests', TMP / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(JF / 'pytest.ini', TMP / 'pytest.ini')
        f = TMP / 'jetformer' / 'drive.py'
        s = f.read_text(encoding='utf-8')
        old = "                    esc_rev = esc_gate.rev_s(esc_q[0], q_ns) > 0.0"
        assert s.count(old) == 1
        f.write_text(s.replace(old, "                    esc_rev = None"), encoding='utf-8')
        p = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=%s' % (TMP / 'bt')] + tests,
                           cwd=str(TMP), capture_output=True, text=True, timeout=3000)
        out = p.stdout + p.stderr
        for x in sorted(set(re.findall(r'FAILED (\S+)', out))):
            print('FAILED', x)
        print(out.strip().splitlines()[-1])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main(sys.argv[1:])
