"""ブリッジ用の Python 環境の準備(jetformer/sim/env.py)のテスト。pip・venv は動かさない(コマンドを差し替える)"""

import json
import os
import sys

import pytest

import _sim_testlib as L
from jetformer.sim import env, install, paths


def test_requirements_file_is_pinned_with_hashes():
    reqs = env.parse_requirements(paths.REQUIREMENTS.read_text(encoding='utf-8'))
    names = {n for n, v, h in reqs}
    assert {'numpy', 'opencv-python-headless', 'onnxruntime', 'protobuf', 'flatbuffers', 'packaging'} == names
    for n, v, hashes in reqs:
        assert len(hashes) == 1 and len(hashes[0]) == 64, n
    assert env.pinned()['onnxruntime'] == '1.31.0' and env.pinned()['numpy'] == '2.2.6'
    assert env.pinned()['opencv-python-headless'] == '4.13.0.92'


def test_parse_requirements_rejects_bad_lines():
    with pytest.raises(ValueError):
        env.parse_requirements('numpy>=2\n')
    assert env.parse_requirements('# c\n\nnumpy==1.0 --hash=sha256:' + 'a' * 64 + '\n') == [('numpy', '1.0', ['a' * 64])]


@pytest.fixture
def rel(tmp_path, monkeypatch):
    L.install_fake_release(tmp_path, monkeypatch)
    return install.active_release()


def test_version_mismatch_with_release(rel):
    assert env.version_mismatch(rel) == []
    p = rel.root / 'requirements-model.txt'
    os.chmod(p, 0o644)
    p.write_text('numpy==2.3.0\nopencv-python-headless==4.13.0.92\nonnxruntime==1.31.0\n')
    diff = env.version_mismatch(rel)
    assert len(diff) == 1 and 'numpy' in diff[0]
    with pytest.raises(env.EnvError, match='固定版'):
        env.setup(run=lambda *a, **k: (0, []))


def test_pip_commands_prefer_offline_wheels(tmp_path):
    py = tmp_path / 'py'
    cmds = env.pip_commands(py, tmp_path / 'nowheels')
    assert [n for n, _ in cmds] == ['pypi']
    assert '--require-hashes' in cmds[0][1] and '--only-binary=:all:' in cmds[0][1] and '--no-deps' in cmds[0][1]
    assert cmds[0][1][cmds[0][1].index('--index-url') + 1] == 'https://pypi.org/simple'
    (tmp_path / 'w').mkdir()
    (tmp_path / 'w' / 'numpy-2.2.6-cp313-cp313-manylinux_2_17_aarch64.whl').write_bytes(b'x')
    cmds = env.pip_commands(py, tmp_path / 'w')
    assert [n for n, _ in cmds] == ['offline', 'pypi']
    assert '--no-index' in cmds[0][1] and str(tmp_path / 'w') in cmds[0][1]


class FakeRun(object):
    """venv・pip・版の確認・自己診断のコマンドを記録して、成功したことにする"""

    def __init__(self, fail_offline=False, packages=None, selftest_rc=0):
        self.calls = []
        self.fail_offline = fail_offline
        self.packages = packages or dict(env.pinned())
        self.selftest_rc = selftest_rc

    def __call__(self, cmd, on_line=None, timeout=None, cwd=None):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[1:3] == ['-m', 'venv']:
            vd = cmd[3]
            os.makedirs(os.path.join(vd, 'bin'), exist_ok=True)
            with open(os.path.join(vd, 'pyvenv.cfg'), 'w') as f:
                f.write('home = %s\n' % os.path.dirname(sys.executable))
            os.symlink(sys.executable, os.path.join(vd, 'bin', 'python'))
            return 0, []
        if '-m' in cmd and 'pip' in cmd:
            if on_line:
                for i in range(5):
                    on_line('Processing x%d' % i)
            if self.fail_offline and '--no-index' in cmd:
                return 1, ['ERROR: no matching distribution']
            return 0, ['Successfully installed']
        if '-c' in cmd:
            return 0, ['noise', json.dumps({'python': '3.13.14', 'packages': self.packages, 'providers': ['CPUExecutionProvider']})]
        if '--self-test' in cmd:
            return self.selftest_rc, ['Ran 13 tests in 0.2s', 'OK' if self.selftest_rc == 0 else 'FAILED']
        raise AssertionError(cmd)


def test_setup_creates_record_and_status_ready(rel, tmp_path):
    (paths.wheels_dir()).mkdir(parents=True)
    (paths.wheels_dir() / 'numpy-2.2.6-cp313-cp313-manylinux_2_17_aarch64.whl').write_bytes(b'x')
    run = FakeRun(fail_offline=True)
    progress = []
    r = env.setup(progress=lambda p, m='': progress.append((p, m)), run=run)
    assert r['ready'] and r['self_test_count'] == 13 and r['source'] == 'pypi'
    pip_calls = [c for c in run.calls if 'pip' in c]
    assert '--no-index' in pip_calls[0] and '--index-url' in pip_calls[1]
    selftest = [c for c in run.calls if '--self-test' in c][0]
    assert selftest[1:3] == ['-I', '-B'] and str(rel.root) in selftest
    assert progress[-1][0] == 1.0 and any('pip' in m for _, m in progress)
    st = env.status()
    assert st['ready'] and st['python_version'] == '3.13.14' and st['packages']['onnxruntime'] == '1.31.0'
    assert env.setup(run=run)['already_ready']          # もう一度押しても作り直さない


def test_setup_rejects_wrong_installed_version(rel):
    pk = dict(env.pinned(), onnxruntime='1.30.0')
    with pytest.raises(env.EnvError, match='onnxruntime'):
        env.setup(run=FakeRun(packages=pk))
    assert not env.status()['ready']


def test_setup_selftest_failure(rel):
    with pytest.raises(env.EnvError, match='自己診断'):
        env.setup(run=FakeRun(selftest_rc=1))
    st = env.status()
    assert not st['ready'] and st['problem']


def test_status_detects_broken_venv(rel):
    env.setup(run=FakeRun())
    assert env.status()['ready']
    # 元の Python が消えた(jetracer env を上げたなど)
    cfg = paths.venv_dir() / 'pyvenv.cfg'
    cfg.write_text('home = /nonexistent/python/bin\n')
    st = env.status()
    assert not st['ready'] and '準備し直して' in st['problem']


def test_status_detects_requirements_change(rel, monkeypatch, tmp_path):
    env.setup(run=FakeRun())
    other = tmp_path / 'req.txt'
    other.write_text(paths.REQUIREMENTS.read_text() + '\n# changed\n')
    monkeypatch.setattr(paths, 'REQUIREMENTS', other)
    assert '固定版が変わりました' in env.status()['problem']


def test_setup_needs_release(tmp_path, monkeypatch):
    from jetformer import config
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    with pytest.raises(env.EnvError, match='導入'):
        env.setup(run=FakeRun())
    assert env.status() == {'ready': False, 'problem': None, 'python_version': None, 'packages': {}, 'created_at': None}
