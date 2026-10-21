"""段 G2 の gpu_queue.py(GPU キューの手伝い)の試験。torch を読まない・GPU を使わない・本物の data/runs に書かない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q experiments/20261021_bench_prep/train/test_gpu_queue.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gpu_queue as g  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs, logs = tmp_path / 'runs', tmp_path / 'logs'
    runs.mkdir()
    logs.mkdir()
    q = tmp_path / 'gpu_queue.tsv'
    q.write_text('# 注\nbench_x-slow-s0\t/home/jetson/bin/jf-heavy --mem 3G -- python -u tools/train.py --policy jetformer_v0 '
                 '--data data/datasets/clean_driving_20261011 --epochs 3 --lr 3e-4 --stride 3 --batch 64 --no-command-input '
                 '--impact-g 2.5 --seed 0 --out data/runs/bench_x-slow-s0\n', encoding='utf-8')
    monkeypatch.setattr(g, 'RUNS', runs)
    monkeypatch.setattr(g, 'LOGS', logs)
    monkeypatch.setattr(g, 'QUEUE', q)
    monkeypatch.setattr(g, 'STATE', logs / 'state.json')
    monkeypatch.setattr(g, 'SUMMARY', logs / 'summary.json')
    return runs, logs


def make_done(run, epochs=3):
    run.mkdir(parents=True, exist_ok=True)
    for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json'):
        (run / f).write_text('x')
    (run / 'result.json').write_text(json.dumps({'state': 'done', 'epochs_done': epochs, 'best_val_us': 50.0, 'best_epoch': 2,
                                                 'history': [{'val_us': 52.0}, {'val_us': 50.0}, {'val_us': 51.0}]}))


def test_queue_and_trained(env):
    runs, _ = env
    assert [n for n, _ in g.queue()] == ['bench_x-slow-s0']
    assert g.epochs_of('bench_x-slow-s0') == 3
    assert not g.is_trained('bench_x-slow-s0')
    run = runs / 'bench_x-slow-s0'
    make_done(run, epochs=2)                  # エポックが足りない → 済んでいない
    assert not g.is_trained('bench_x-slow-s0')
    make_done(run, epochs=3)
    assert g.is_trained('bench_x-slow-s0')
    (run / 'best.pth').unlink()               # ファイルが欠けている → 済んでいない
    assert not g.is_trained('bench_x-slow-s0')


def test_fails_ignore_interrupted(env):
    n = 'bench_x-slow-s0'
    g.cmd_begin(n, 'train')                   # 電源断で終わりの記録が無い試み → 失敗に数えない
    assert g.fails(n, 'train') == 0
    g.cmd_begin(n, 'train')
    g.cmd_end(n, 'train', 137)                # 終わったのに済んでいない → 失敗
    assert g.fails(n, 'train') == 1
    assert g.last_rc(n, 'train') == 137
    make_done(env[0] / n)
    g.cmd_begin(n, 'train')
    a = g.cmd_end(n, 'train', 0)
    assert a['ok'] and g.fails(n, 'train') == 1


def test_need_engine_and_check(env):
    runs, logs = env
    n = 'bench_x-slow-s0'
    run = runs / n
    make_done(run)
    assert g.need_engine(n)
    eng = run / g.ENGINE
    eng.write_bytes(b'e' * 10)
    os.utime(eng, ns=(time.time_ns() - 10 ** 10, time.time_ns() - 10 ** 10))   # ONNX より古い → 作り直す
    assert g.need_engine(n)
    eng.write_bytes(b'e' * 10)
    assert not g.need_engine(n)
    assert g.check_current(run) is None
    chk = {'run': n, 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 2)), 'all_gates': True,
           'accuracy': {'actions_max_us': 1.1, 'actions_p99_us': 0.8, 'gate_actions_2p5us': True, 'finite': True},
           'latency_ms': {'run_numpy_median': 2.3, 'run_numpy_p99': 2.6, 'gate_run_numpy': True}}
    (logs / ('engine_check_%s.json' % n)).write_text(json.dumps(chk))
    assert g.cmd_enginecheck(n) == 0
    c = g.check_current(run)
    assert c and c['accuracy_ok'] and c['ok'] and c['actions_max_us'] == 1.1 and c['gate_us'] == 2.5
    st = eng.stat()
    assert c['engine'] == {'file': g.ENGINE, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
    eng.write_bytes(b'e' * 11)                # エンジンが変わった → 今の確認ではない
    assert g.check_current(run) is None


def test_enginecheck_accuracy_fail(env):
    runs, logs = env
    n = 'bench_x-slow-s0'
    run = runs / n
    make_done(run)
    (run / g.ENGINE).write_bytes(b'e')
    chk = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 2)), 'all_gates': False,
           'accuracy': {'actions_max_us': 3.4, 'gate_actions_2p5us': False, 'finite': True},
           'latency_ms': {'run_numpy_median': 2.3, 'run_numpy_p99': 2.6, 'gate_run_numpy': True}}
    (logs / ('engine_check_%s.json' % n)).write_text(json.dumps(chk))
    assert g.cmd_enginecheck(n) == 1
    c = g.check_current(run)
    assert c['accuracy_ok'] is False and c['ok'] is False     # drive.model_info はこれで drivable False にする


def test_runjson(env):
    runs, _ = env
    n, cmd = g.queue()[0]
    (runs / n).mkdir()
    assert g.cmd_runjson(n, cmd) == 0
    m = json.loads((runs / n / 'run.json').read_text())
    assert m['name'] == n and m['arm'] == 'SLOW' and m['epochs'] == 3 and m['stride'] == 3 and m['seed'] == 0
    assert m['command_input'] is False and m['policy'] == 'jetformer_v0' and m['impact_g'] == 2.5
    assert m['policy_cfg'] == {'dim': 256, 'layers': 4, 'heads': 8}
    before = (runs / n / 'run.json').read_text()
    assert g.cmd_runjson(n, cmd) == 0         # 既にあれば書き換えない
    assert (runs / n / 'run.json').read_text() == before


def test_summary_rows(env, capsys):
    runs, _ = env
    n = 'bench_x-slow-s0'
    make_done(runs / n)
    assert g.cmd_summary() == 0
    s = json.loads(g.SUMMARY.read_text())
    r = s['runs'][0]
    assert r['train'] == 'done' and r['best_val_us'] == 50.0 and r['last_val_us'] == 51.0 and r['engine'] is False
    assert n in capsys.readouterr().out
