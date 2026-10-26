"""キュー v3 の手伝い(gpu_queue_v3.py)の試験(torch を読まない・GPU を使わない・学習を始めない)。
決まりの答え・lock2_v3・物差しは試験用の別のファイル(GPU_QUEUE_DECISIONS・GPU_QUEUE_LOCK2・GPU_QUEUE_PROBE_DIR)で置く。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q \
        -p no:cacheprovider /home/jetson/jetformer/experiments/20261026_fix/test_gpu_queue_v3.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, 'gpu_queue_v3.py')


def h(*args, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith('GPU_QUEUE_')}
    e.update(env or {})
    r = subprocess.run([sys.executable, '-I', Q] + list(args), capture_output=True, text=True, env=e)
    return r.returncode, r.stdout.strip()


def fake(tmp_path, order=('v3d', 'v3n', 'v3b'), probe=None, lock2=True):
    dec = tmp_path / 'dec.json'
    dec.write_text(json.dumps({'D0': {'order': list(order)}} if order else {}))
    env = {'GPU_QUEUE_DECISIONS': str(dec), 'GPU_QUEUE_LOCK2': str(tmp_path / ('lock2.json' if lock2 else 'none.json')),
           'GPU_QUEUE_PROBE_DIR': str(tmp_path / 'probe')}
    if lock2:
        (tmp_path / 'lock2.json').write_text('{}')
    (tmp_path / 'probe').mkdir(exist_ok=True)
    for name, (st23, stand) in (probe or {}).items():
        (tmp_path / 'probe' / ('%s_best.json' % name)).write_text(json.dumps({'x': {
            'st10_thr_vs_b_us': 0.0, 'st23_thr_vs_b_us': st23, 'stand_brake_side_frac': stand, 'stand_go_frac': 0.2,
            'low10_brake_frac': 0.0}}))
    return env


def test_tsv_rows_match_protocol():
    rc, out = h('templates', env={'GPU_QUEUE_DECISIONS': '/nonexistent'})
    names = out.splitlines()
    assert rc == 0 and len(names) == 22
    assert names[:6] == ['bfast2_20261026-e15-s0', 'bnull_20261026-e15-s0', 'bfast2_20261026-e15-s1', 'bnull_20261026-e15-s1',
                         'bfast2_20261026-e15-s2', 'bnull_20261026-e15-s2']                 # B-FAST2 が先
    assert names[6:10] == ['abl-same-e15-s0', 'abl-noTK-e15-s0', 'abl-noTK_noLB-e15-s0', 'abl-tkB-e15-s0']


def test_bf2_skipped_when_bf2_0_fails(tmp_path):
    rc, out = h('resolve', 'bfast2_20261026-e15-s0', env=fake(tmp_path))
    assert out.startswith('SKIP\tBF2-0 が「学習しない」')


def test_stage0_runs_same_command_as_ablation_short():
    rc, out = h('resolve', 'abl-tkB-e15-s0')
    kind, name, cmd = out.split('\t')
    assert kind == 'RUN' and name == 'abl-tkB-e15-s0'
    assert ('--drop-mask experiments/20261026_fix/diag_data/out/ablation/drop_v2h_same.npz --label-override-npz '
            'experiments/20261026_fix/diag_data/out/ablation/override_tk_thr_B.npz --epochs 15') in cmd
    assert cmd.endswith('--seed 0 --out experiments/20261026_fix/diag_data/out/ablation/runs/abl-tkB-e15-s0')
    assert h('post', 'abl-tkB-e15-s0')[1] == 'probe'                      # 段 0 はエンジンを作らない


def test_stage1_stops_without_lock2_or_d0(tmp_path):
    assert h('resolve', 'v3_20261026-v3n-aug2-e60-s0', env=fake(tmp_path, lock2=False))[1].startswith('STOP\tprotocol_v3_lock2.json')
    assert 'D0.order' in h('resolve', 'v3_20261026-v3n-aug2-e60-s0', env=fake(tmp_path, order=None))[1]


def test_stage1_order_follows_d0(tmp_path):
    names = h('templates', env=fake(tmp_path, order=('v3b', 'v3d', 'v3n')))[1].splitlines()[10:]
    assert names[:3] == ['v3_20261026-v3b-aug2-e60-s0', 'v3_20261026-v3d-aug2-e60-s0', 'v3_20261026-v3n-aug2-e60-s0']
    assert [n[-2:] for n in names] == ['s0'] * 3 + ['s1'] * 3 + ['s2'] * 3 + ['s3'] * 3
    rc, out = h('resolve', 'v3_20261026-v3d-aug2-e60-s0', env=fake(tmp_path))
    kind, name, cmd = out.split('\t')
    assert kind == 'RUN' and '--label-override-npz data/derived/dataset_v3_20261026/override_v3d_tk_thr_B.npz' in cmd
    assert '--action-ref-from-samples --probe-npz experiments/20261026_fix/diag_data/out/probe_set.npz' in cmd


def test_futility_skips_seed2_3_only_when_both_bad(tmp_path):
    p = {'v3_20261026-v3n-aug2-e60-s0': (50.0, 0.01), 'v3_20261026-v3n-aug2-e60-s1': (10.0, 0.30),     # 両方とも外れ
         'v3_20261026-v3d-aug2-e60-s0': (50.0, 0.01), 'v3_20261026-v3d-aug2-e60-s1': (42.4, 0.192)}    # 種 1 は境目(通す)
    env = fake(tmp_path, probe=p)
    assert h('resolve', 'v3_20261026-v3n-aug2-e60-s2', env=env)[1].startswith('SKIP\t途中でやめる決まり')
    assert h('resolve', 'v3_20261026-v3n-aug2-e60-s3', env=env)[1].startswith('SKIP\t途中でやめる決まり')
    assert h('resolve', 'v3_20261026-v3d-aug2-e60-s2', env=env)[1].startswith('RUN')
    assert h('resolve', 'v3_20261026-v3b-aug2-e60-s2', env=env)[1].startswith('RUN')       # 物差しがまだ無い → 始める
    assert h('resolve', 'v3_20261026-v3n-aug2-e60-s1', env=env)[1].startswith('RUN')       # 種 0・1 は止めない


def test_smoke_one_row_separate_output():
    env = {'GPU_QUEUE_SMOKE': '1', 'GPU_QUEUE_DECISIONS': '/nonexistent'}
    assert h('templates', env=env)[1].splitlines() == ['v3_20261026-v3n-aug2-e60-s0']
    kind, name, cmd = h('resolve', 'v3_20261026-v3n-aug2-e60-s0', env=env)[1].split('\t')
    assert kind == 'RUN' and name == '_smoke_v3_20261026-v3n-aug2-e60-s0'
    assert '--epochs 1 ' in cmd and cmd.endswith('--out data/runs/_smoke_v3_20261026-v3n-aug2-e60-s0 --max-steps 5')
    assert h('resolve', 'abl-same-e15-s0', env=env)[1].startswith('SKIP\t試しは 1 行だけ')
    assert h('rundir', name, env=env)[1] == '/home/jetson/jetformer/data/runs/_smoke_v3_20261026-v3n-aug2-e60-s0'


def test_estimate_and_preflight_run():
    rc, out = h('estimate')
    assert rc == 0 and '段 1' in out
    rc, out = h('preflight')
    assert 'OK  gpu_queue_v3.tsv の 22 行が protocol と同じ' in out and 'NG' not in out
