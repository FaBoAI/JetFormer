"""正解の付け替え(relabel/relabel_dataset.py + relabel_scene.mjs)の試験(pytest)。2026-10-14、sim_dart_20261014。

    jf-heavy --mem 1500M --lock none -- nice -n 10 python3 -m pytest -q tests/test_relabel.py --basetemp <一時の場所>

  1. 乱しなしの本物の記録(A0 の 1 シーン、ブラウザで記録したもの): 付け替えた正解 = 元の正解(差 0 µs)・外すフレーム 0(R1 の小さな版)。
     データセットは --mode check で読むだけ。
  2. 乱しなしの Node の記録(D2 の先生、150 フレーム): 一時のデータセットで --mode write。正解は変わらない・future_valid も同じ。
  3. 乱しありの Node の記録(D1 = 最速 + N_fast、D3 = 余裕の線 + N_margin、150 フレーム):
     - 再生が記録と一致(≤ 1e-9)
     - 乱していないフレーム t+1 では future_actions[t, 0] = 記録の vehicle[t+1](≤ 1e-3 µs)、乱したフレームでは違う
     - 窓 t+1..t+1+k に乱しが無ければ future_actions[t, k] = vehicle[t+1+k]、乱しがあればほとんど違う
     - ほかの配列は変わらない・元の配列の写し(backup)がある・dataset_card.json に relabel がある・2 回目の write は断る
  4. PWM の式が収集器(コピーした dataset_recorder.py の action_to_pwm)と同じ
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path('/home/jetson/jetformer/experiments/20261014_sim_dart_margin')
TOOL = EXP / 'relabel' / 'relabel_dataset.py'
A0 = Path('/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0')
A0_SCENE = '20261013-162009-efc847eb-0001'
sys.path.insert(0, str(EXP / 'relabel'))
import relabel_dataset as RD  # noqa: E402

REC = RD.load_recorder()


def make_fixture(dataset, index, frames, out):
    r = subprocess.run(['node', str(EXP / 'tests' / 'make_fixture.mjs'), '--dataset', dataset, '--index', str(index), '--frames', str(frames),
                        '--out', str(out)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def build_dataset(root, scenes):
    """収集器と同じ形の aligned NPZ(付け替えが触る配列と、触らない配列の代表)を、Node の記録から作る"""
    (root / 'aligned').mkdir(parents=True)
    ids = []
    for sid, src in scenes.items():
        shutil.copytree(src, root / 'simulation' / 'scenes' / sid)
        sess = json.loads((src / 'session.json').read_text())
        rows = [json.loads(l) for l in (src / 'frames.jsonl').read_text().splitlines() if l.strip()]
        N = len(rows)
        cal, model = sess['calibration'], sess['model_config']
        veh = np.zeros((N, 4), np.float32)
        for i, r in enumerate(rows):
            veh[i, :2] = REC.action_to_pwm(r['command_si'], cal, model)
            veh[i, 2] = 1500.
            veh[i, 3] = r['state_post_action']['speed'] * cal['erpm_per_mps'] * cal['erpm_forward_sign']
        fa = np.zeros((N, 10, 2), np.float32)
        for t in range(N):
            for k in range(10):
                fa[t, k] = veh[min(N - 1, t + 1 + k), :2]
        fv = np.arange(N) < N - 10
        rng = np.random.default_rng(0)
        np.savez_compressed(root / 'aligned' / f'{sid}.npz', timestamps_us=np.arange(N, dtype=np.int64) * 33333,
                            cam_files=np.array([f'samples/CAM_FRONT/{sid}/{i}.jpg' for i in range(N)]),
                            tof=rng.integers(0, 4000, (N, 42, 54)).astype(np.uint16), imu=rng.normal(size=(N, 10, 6)).astype(np.float32),
                            vehicle=veh, vehicle_valid=np.ones(N, bool), future_actions=fa, future_valid=fv, mode=np.full(N, 8, np.int16))
        ids.append(sid)
    (root / 'splits.json').write_text(json.dumps({'train': ids, 'val': []}))
    (root / 'dataset_card.json').write_text(json.dumps({'name': root.name, 'scenes': len(ids)}))


def run_tool(args):
    r = subprocess.run([sys.executable, '-I', str(TOOL), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-2000:]
    return json.loads(r.stdout[r.stdout.index('{'):])


@pytest.fixture(scope='module')
def node_fixtures(tmp_path_factory):
    base = tmp_path_factory.mktemp('fixtures')
    out = {}
    for name, (d, i) in {'d1': ('D1', 3), 'd2': ('D2', 0), 'd3': ('D3', 2)}.items():
        info = make_fixture(d, i, 150, base / name)
        out[name] = (base / name, info)
    return out


def test_pwm_same_as_recorder():
    cal = {'max_steering_rad': .52, 'steering_sign': 1, 'pwm_min_us': 1000., 'pwm_max_us': 2000., 'max_speed_mps': 7.24, 'max_reverse_speed_mps': .8}
    model = {'steer_center_us': 1442., 'throttle_neutral_us': 1498.}
    lab = [[.3, 4.5, -.2, 3.0] + [0, 0] * 8]
    got = RD.to_pwm(REC, lab, cal, model)
    assert got.dtype == np.float32
    assert np.allclose(got[0, 0], np.float32(REC.action_to_pwm({'steering': .3, 'speed': 4.5}, cal, model)), atol=0)
    assert abs(float(got[0, 0, 0]) - (1442 + .3 / .52 * 558)) < 1e-3 and abs(float(got[0, 1, 0]) - (1442 - .2 / .52 * 442)) < 1e-3
    assert abs(float(got[0, 0, 1]) - (1498 - 4.5 / 7.24 * 498)) < 1e-3


def test_clean_real_A0_scene_identity(tmp_path):
    """乱しなしの本物の記録(ブラウザ): 付け替えても正解は元と同じ(≤ 1 µs、実際は 0)・外すフレーム 0"""
    res = run_tool(['--dataset-dir', str(A0), '--teacher', 'fast', '--noise', 'none', '--mode', 'check', '--scenes', A0_SCENE,
                    '--jobs', '1', '--work', str(tmp_path / 'work')])
    s = res['summary']
    assert s['scenes'] == 1 and s['labelled_frames'] == 590
    assert s['diff_vs_original_max_us_used'] <= 1.0 and s['frames_diff_gt_1us_used'] == 0
    assert s['invalid_teacher_frames'] == 0
    assert s['replay_max_abs_error'] <= 1e-9 and s['pwm_recon_vs_vehicle_max_us'] <= 1e-3
    assert res['checks']['R1_identity'] is True


def test_clean_node_margin_scene_unchanged(node_fixtures, tmp_path):
    src, info = node_fixtures['d2']
    assert info['noisy'] == 0
    root = tmp_path / 'ds_d2'
    build_dataset(root, {'fx-d2-0001': src})
    before = dict(np.load(root / 'aligned' / 'fx-d2-0001.npz'))
    res = run_tool(['--dataset-dir', str(root), '--teacher', 'margin', '--noise', 'none', '--mode', 'write', '--jobs', '1',
                    '--work', str(tmp_path / 'work'), '--backup', str(tmp_path / 'backup')])
    after = dict(np.load(root / 'aligned' / 'fx-d2-0001.npz'))
    assert np.abs(after['future_actions'].astype(np.float64) - before['future_actions']).max() <= 1e-3
    assert np.array_equal(after['future_valid'], before['future_valid'])
    assert res['summary']['replay_max_abs_error'] == 0.0
    assert res['summary']['invalid_teacher_frames'] == 0


@pytest.mark.parametrize('name,teacher,noise', [('d1', 'fast', 'N_fast'), ('d3', 'margin', 'N_margin')])
def test_noisy_scene_labels_differ_only_where_noise(node_fixtures, tmp_path, name, teacher, noise):
    src, info = node_fixtures[name]
    assert info['noisy'] > 10, '乱したフレームが少なすぎる試験の記録'
    sid = f'fx-{name}-0001'
    root = tmp_path / f'ds_{name}'
    build_dataset(root, {sid: src})
    npz = root / 'aligned' / f'{sid}.npz'
    before = dict(np.load(npz))
    res = run_tool(['--dataset-dir', str(root), '--teacher', teacher, '--noise', noise, '--mode', 'write', '--jobs', '1',
                    '--work', str(tmp_path / 'work'), '--backup', str(tmp_path / 'backup')])
    s = res['summary']
    # 再生が記録と一致(Node で作った記録なので差 0)
    assert s['replay_max_abs_error'] <= 1e-9
    assert s['r2_clean_vs_info_max'] <= 1e-9 and s['r2_noise_flag_mismatch'] == 0
    after = dict(np.load(npz))
    rows = [json.loads(l) for l in (src / 'frames.jsonl').read_text().splitlines() if l.strip()]
    noisy = np.array([r['action_info']['dart']['noise'] is not None for r in rows])
    N, T = len(rows), len(rows) - 10
    fa, veh = after['future_actions'].astype(np.float64), before['vehicle'].astype(np.float64)
    # R3: 乱していないフレーム t+1 では最初の正解 = 記録の操作。乱したフレームでは違う
    d0 = np.abs(fa[:T, 0, :] - veh[1:T + 1, :2]).max(axis=1)
    nx = noisy[1:T + 1]
    assert nx.any() and (~nx).any()
    assert d0[~nx].max() <= 1e-3
    assert (d0[nx] > 1.0).mean() > .9
    # R3b: 窓に乱しが無ければ 10 個とも記録の操作と同じ。乱しがあれば(ほとんど)違う
    clean_cells, noisy_cells = [], []
    for t in range(T):
        cum = np.logical_or.accumulate(noisy[t + 1:t + 11])
        for k in range(10):
            dd = float(np.abs(fa[t, k] - veh[t + 1 + k, :2]).max())
            (noisy_cells if cum[k] else clean_cells).append(dd)
    assert max(clean_cells) <= 1e-3
    # 乱しのあとの窓では、先生の舵が両方とも ±0.52 で頭打ち(PWM 1000 / 2000)・速さも上限 4.5 のフレームは同じ値になるので、
    # 「ほとんど違う」は 6 割以上とする(2026-10-14 の試しで D1 88%・乱したフレームの次の 1 個目は 9 割以上が違う)
    assert np.mean(np.array(noisy_cells) > 1.0) > .6
    # 付け替えたフレームの正解は、記録の乱し入りの操作ではなく先生の操作(乱したフレームの次の正解と記録の操作の差)
    assert s['r3_noisy_next_mean_us'] > 5
    # ほかの配列は同じ・最後の 10 フレームは元のまま
    for k in before:
        if k not in ('future_actions', 'future_valid'):
            assert np.array_equal(before[k], after[k]), k
    assert np.array_equal(after['future_actions'][T:], before['future_actions'][T:])
    assert not after['future_valid'][T:].any()
    assert (after['future_valid'][:T] <= before['future_valid'][:T]).all()
    # 元の配列の写しと dataset_card の記録
    bk = np.load(tmp_path / 'backup' / f'{sid}.npz')
    assert np.array_equal(bk['future_actions'], before['future_actions']) and np.array_equal(bk['future_valid'], before['future_valid'])
    card = json.loads((root / 'dataset_card.json').read_text())
    assert card['relabel']['schema'] == 'jf-sim-dart-relabel/v1' and card['relabel']['noise'] == noise and card['relabel']['horizon'] == 10
    # 2 回目の付け替えは断る
    r = subprocess.run([sys.executable, '-I', str(TOOL), '--dataset-dir', str(root), '--teacher', teacher, '--noise', noise, '--mode', 'write',
                        '--work', str(tmp_path / 'work2'), '--backup', str(tmp_path / 'backup2')], capture_output=True, text=True)
    assert r.returncode != 0 and '付け替え済み' in (r.stderr + r.stdout)
