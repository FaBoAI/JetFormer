"""fix8 係 safety-decided: 合わせる前の試験 8 本(before_tests/ の写し)を、説明の文を足したあとの drive.py に流すと落ちる(= たどりを足す
必要があった)ことを見る。一時の場所に jetformer/・tests/fixtures/・experiments/・data/ への symlink と、tests/ の .py の symlink を置き、
8 本だけ写しに差し替えて、drive.py の sha256 を見る試験 9 個を流す(本物のファイルは変えない)。
    python -I probe_old_tests10.py   → 標準出力(終わったら一時の場所を消す)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_safety_decided' / 'oldtests'
NODES = ['tests/test_fix3_r3_2_tools.py::test_untouched_files',
         'tests/test_fix3_r3_1.py::test_before_copies_and_defaults_unchanged',
         'tests/test_fix4_small.py::test_drive_only_docstring_changed',
         'tests/test_rec_fix5_safety.py::test_before_copy_defaults_and_changed_parts',
         'tests/test_rec_fix6_safety.py::test_before_copy_defaults_and_changed_parts',
         'tests/test_rec_fix6_safety_r2.py::test_before_copy_defaults_and_changed_parts',
         'tests/test_rec_fix7_safety.py::test_before_copy_defaults_and_changed_parts',
         'tests/test_rec_fix7_safety_r2.py::test_r1_copy_defaults_and_changed_parts']


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    (TMP / 'tests').mkdir(parents=True)
    try:
        for name in ('jetformer', 'experiments', 'data', 'tools', 'pytest.ini'):
            os.symlink(JF / name, TMP / name)
        bt = HERE / 'before_tests'
        olds = {p.name for p in bt.glob('test_*.py')}
        for p in (JF / 'tests').iterdir():
            if p.name == '__pycache__':
                continue
            if p.name in olds:
                shutil.copy2(bt / p.name, TMP / 'tests' / p.name)
            else:
                os.symlink(p, TMP / 'tests' / p.name)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o',
                            'tmp_path_retention_policy=failed', '--basetemp', str(TMP / 'bt'), '-rf'] + NODES,
                           cwd=str(TMP), env=env, capture_output=True, text=True, timeout=1200)
        lines = r.stdout.strip().splitlines()
        print('\n'.join(x for x in lines if x.startswith('FAILED') or ' passed' in x or ' failed' in x))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
