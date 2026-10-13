"""ブリッジのプロセス管理(jetformer/sim/bridge_proc.py)のテスト。偽のシミュレーターと本物の起動スクリプトを使う(CPU だけ・数秒)"""

import os
import signal
import sys
import time

import pytest

import _sim_testlib as L
from jetformer.sim import bridge_proc, paths


@pytest.fixture
def sim_dir(tmp_path):
    d = tmp_path / 'release'
    for rel, data in L.release_files().items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return d


def make(tmp_path, sim_dir, **kw):
    args = dict(python=sys.executable, runner=paths.RUNNER, sim_dir=sim_dir, runtime_root=tmp_path / 'runtime',
                log_path=tmp_path / 'logs' / 'bridge.log', nice=0, backoff_s=(0.1, 0.1, 0.1), health_timeout_s=10,
                listen_timeout_s=10)
    args.update(kw)
    return bridge_proc.BridgeProcess(**args)


def wait_state(bp, states, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if bp.status()['state'] in states:
            return bp.status()
        time.sleep(0.05)
    raise AssertionError('state %s (expected %s)' % (bp.status(), states))


def test_start_running_stop(tmp_path, sim_dir):
    bp = make(tmp_path, sim_dir)
    st = bp.start()
    assert st['state'] in ('starting', 'running') and bp.running()
    st = wait_state(bp, ('running',))
    assert st['port'] and st['pid'] and bp.port == st['port']
    assert bridge_proc.health_check(bp.port)
    assert bp.start()['state'] == 'running'                    # 2 回押しても 1 つだけ
    pid = st['pid']
    st = bp.stop()
    assert st['state'] == 'stopped' and bp.port is None and not bp.running()
    with pytest.raises(OSError):
        os.kill(pid, 0)                                        # 子プロセスは残らない
    time.sleep(0.3)
    assert bp.status()['state'] == 'stopped'                  # 自分で止めたら起動し直さない
    logs = bp.logs()
    assert any('待ち受けています' in x for x in logs['lines'])
    assert (tmp_path / 'logs' / 'bridge.log').is_file()


def test_crash_restarts_with_backoff_then_fails(tmp_path, sim_dir):
    bp = make(tmp_path, sim_dir, restart_limit=2)
    bp.start()
    seen = set()
    for i in range(3):
        st = wait_state(bp, ('running',))
        seen.add(st['pid'])
        os.kill(st['pid'], signal.SIGKILL)                     # 異常終了
        if i < 2:
            wait_state(bp, ('starting',), timeout=5)
    st = wait_state(bp, ('failed',))
    assert len(seen) == 3 and st['restarts'] == 2 and st['restarted_at']
    assert '繰り返し異常終了' in st['last_error']
    bp.stop()


def test_memory_check(tmp_path, sim_dir):
    bp = make(tmp_path, sim_dir, mem_available=lambda: 200 << 20)
    with pytest.raises(bridge_proc.BridgeError, match='メモリ'):
        bp.start()
    assert bp.status()['state'] == 'stopped'


def test_preflight_errors(tmp_path, sim_dir):
    def pre():
        raise bridge_proc.BridgeError('先に Python 環境を準備してください')
    bp = make(tmp_path, sim_dir, preflight=pre)
    with pytest.raises(bridge_proc.BridgeError, match='準備'):
        bp.start()


def test_listen_timeout(tmp_path, sim_dir):
    silent = tmp_path / 'silent.py'
    silent.write_text('import time\nprint("starting", flush=True)\ntime.sleep(30)\n')
    bp = make(tmp_path, sim_dir, runner=silent, listen_timeout_s=0.5)
    bp.start()
    st = wait_state(bp, ('failed',))
    assert '待ち受け' in st['last_error'] and 'starting' in st['last_error']
    assert st['pid'] is None


def test_health_timeout(tmp_path, sim_dir):
    liar = tmp_path / 'liar.py'
    liar.write_text('import json, time\nprint(json.dumps({"event": "listening", "port": 9, "pid": 1}), flush=True)\ntime.sleep(30)\n')
    bp = make(tmp_path, sim_dir, runner=liar, health_timeout_s=0.5, health_check=lambda port: False)
    bp.start()
    st = wait_state(bp, ('failed',))
    assert '応答しません' in st['last_error']


def test_exit_during_start(tmp_path, sim_dir):
    bad = tmp_path / 'bad.py'
    bad.write_text('import sys\nprint("Traceback: boom", flush=True)\nsys.exit(3)\n')
    bp = make(tmp_path, sim_dir, runner=bad)
    bp.start()
    st = wait_state(bp, ('failed',))
    assert '終了コード 3' in st['last_error'] and 'boom' in st['last_error']


def test_log_ring_suppresses_per_frame_lines(tmp_path, sim_dir):
    noisy = tmp_path / 'noisy.py'
    noisy.write_text(
        'import json, sys, time\n'
        'for i in range(50):\n'
        '    print("[bridge] POST /api/predict -> 200", file=sys.stderr)\n'
        '    print("[bridge] POST /api/datasets/frame -> 200", file=sys.stderr)\n'
        'print("[bridge] POST /api/predict -> 503", file=sys.stderr)\n'
        'sys.stderr.flush()\n'
        'print(json.dumps({"event": "listening", "port": 9, "pid": 1}), flush=True)\n'
        'time.sleep(30)\n')
    bp = make(tmp_path, sim_dir, runner=noisy, health_check=lambda port: True)
    bp.start()
    wait_state(bp, ('running',))
    time.sleep(0.2)
    logs = bp.logs()
    assert logs['suppressed'] == {'predict': 50, 'frame': 50}
    assert any('-> 503' in x for x in logs['lines'])
    assert not any('frame -> 200' in x for x in logs['lines'])
    assert len(bp.logs(1)['lines']) == 1
    bp.stop(timeout=1)


def test_stop_escalates_to_kill(tmp_path, sim_dir):
    stubborn = tmp_path / 'stubborn.py'
    stubborn.write_text('import json, signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                        'print(json.dumps({"event": "listening", "port": 9, "pid": 1}), flush=True)\ntime.sleep(60)\n')
    bp = make(tmp_path, sim_dir, runner=stubborn, health_check=lambda port: True)
    bp.start()
    st = wait_state(bp, ('running',))
    t0 = time.time()
    assert bp.stop(timeout=0.3)['state'] == 'stopped'
    assert time.time() - t0 < 10
    with pytest.raises(OSError):
        os.kill(st['pid'], 0)


def test_child_gets_parent_pid_and_oom_score(tmp_path, sim_dir):
    """review で追加: ブリッジは JetFormer の pid を知り(いなくなったら止まる)、メモリ不足のときは先に止められる"""
    bp = make(tmp_path, sim_dir)
    cmd = bp._command(dict(bp.cfg, port=0))
    i = cmd.index('--parent-pid')
    assert cmd[i + 1] == str(os.getpid()) and cmd[cmd.index('--oom-score-adj') + 1] == '500'
    bp.start()
    st = wait_state(bp, ('running',))
    try:
        with open('/proc/%d/oom_score_adj' % st['pid']) as f:
            assert int(f.read()) == 500
    finally:
        bp.stop()
