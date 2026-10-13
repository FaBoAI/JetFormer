"""ブリッジの起動スクリプト(jetformer/sim/bridge_runner.py)のテスト。偽のシミュレーター(標準ライブラリだけのブリッジ)で動かす"""

import http.client
import importlib.util
import json
import signal
import subprocess
import sys

import pytest

import _sim_testlib as L
from jetformer.sim import paths


@pytest.fixture
def sim_dir(tmp_path):
    d = tmp_path / 'release'
    for rel, data in L.release_files().items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return d


def run_runner(sim_dir, runtime, *extra):
    return subprocess.Popen([sys.executable, '-I', '-B', str(paths.RUNNER), '--sim-dir', str(sim_dir), '--runtime-root', str(runtime),
                             '--nice', '0', *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(runtime.parent))


def test_runner_listens_on_loopback_and_stops(sim_dir, tmp_path):
    runtime = tmp_path / 'runtime'
    (runtime / 'uploads' / 'sub').mkdir(parents=True)
    (runtime / 'uploads' / 'old.onnx').write_bytes(b'x' * 10)
    (runtime / 'datasets').mkdir()
    (runtime / 'datasets' / 'keep.txt').write_text('keep')
    p = run_runner(sim_dir, runtime)
    try:
        ev = json.loads(p.stdout.readline())
        assert ev['event'] == 'listening' and ev['host'] == '127.0.0.1' and ev['port'] > 0 and ev['pid'] == p.pid
        c = http.client.HTTPConnection('127.0.0.1', ev['port'], timeout=5)
        c.request('GET', '/api/status')
        r = c.getresponse()
        body = json.loads(r.read())
        assert r.status == 200 and body['runtime_root'] == str(runtime.resolve())
        c = http.client.HTTPConnection('127.0.0.1', ev['port'], timeout=5)
        c.request('GET', '/api/status', headers={'Host': '192.168.5.14:8080'})
        assert c.getresponse().status == 403             # Host の検査は実際のポートで
        assert list((runtime / 'uploads').iterdir()) == []   # 前回のアップロードは消える
        assert (runtime / 'datasets' / 'keep.txt').exists()
        (runtime / 'uploads').mkdir(exist_ok=True)
        (runtime / 'uploads' / 'model.onnx').write_bytes(b'x' * 10)     # 動いている間に受け取ったモデルの写し
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=10) == 0
        assert list((runtime / 'uploads').iterdir()) == []   # 止めるときにも消す(review で追加)
        assert (runtime / 'datasets' / 'keep.txt').exists()
        rest = p.stdout.read()
        assert '"event": "stopped"' in rest
    finally:
        if p.poll() is None:
            p.kill()
    assert not list(sim_dir.rglob('__pycache__'))       # -B: シミュレーターのフォルダーに何も作らない


def test_runner_sigint_stops(sim_dir, tmp_path):
    p = run_runner(sim_dir, tmp_path / 'rt')
    try:
        json.loads(p.stdout.readline())
        p.send_signal(signal.SIGINT)
        assert p.wait(timeout=10) == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_runner_self_test(sim_dir, tmp_path):
    p = run_runner(sim_dir, tmp_path / 'rt', '--self-test')
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 0 and 'Ran 13 tests' in out


def test_runner_missing_bridge(tmp_path):
    (tmp_path / 'empty').mkdir()
    p = run_runner(tmp_path / 'empty', tmp_path / 'rt')
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 2 and 'bridge.py' in out


@pytest.mark.skipif(importlib.util.find_spec('onnxruntime') is None, reason='onnxruntime がない環境(ブリッジ用 venv にだけ入れる)')
def test_ort_threads_limited():
    spec = importlib.util.spec_from_file_location('jf_bridge_runner', str(paths.RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import onnxruntime as ort
    base = ort.InferenceSession
    try:
        assert mod.limit_ort_threads(2)
        assert ort.InferenceSession is not base and ort.InferenceSession._jetformer_limited
    finally:
        ort.InferenceSession = base


def test_ort_limit_without_onnxruntime(monkeypatch):
    spec = importlib.util.spec_from_file_location('jf_bridge_runner2', str(paths.RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    assert mod.limit_ort_threads(4) is False


def test_runner_stops_when_parent_dies(sim_dir, tmp_path):
    """review で追加: --parent-pid の親(JetFormer)が強制終了したら、孤児のブリッジは自分で止まる。
    本物と同じく stdout・stderr は親へのパイプ(親と一緒に閉じる。書くと BrokenPipeError になるのを見張りが越えられること)"""
    import os
    import time
    runtime = tmp_path / 'rt'
    # 中間のプロセス: ブリッジを起動して pid を出し、listening を待ってから自分は終わる(= JetFormer の kill -9)
    code = ('import json, os, subprocess, sys\n'
            'p = subprocess.Popen(sys.argv[1:] + ["--parent-pid", str(os.getpid())], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n'
            'ev = json.loads(p.stdout.readline())\n'
            'print(p.pid, ev["port"], flush=True)\n'
            'os._exit(0)\n')
    mid = subprocess.run([sys.executable, '-c', code, sys.executable, '-I', '-B', str(paths.RUNNER), '--sim-dir', str(sim_dir),
                          '--runtime-root', str(runtime), '--nice', '0'], capture_output=True, text=True, timeout=30)
    pid, port = (int(x) for x in mid.stdout.split())

    def alive(pid):
        try:
            with open('/proc/%d/stat' % pid) as f:
                return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
        except OSError:
            return False
    t0 = time.time()
    while alive(pid) and time.time() - t0 < 10:
        time.sleep(0.1)
    try:
        assert not alive(pid), 'ブリッジが親の終了後も動いている'
        with pytest.raises(OSError):
            http.client.HTTPConnection('127.0.0.1', port, timeout=2).request('GET', '/api/status')
    finally:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
