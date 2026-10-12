"""評価の道具が、どのモデルの種類(方策)も走行と同じ手順(runtime.load_run → predict → post.Stage)で評価するかのテスト。
GPU・実データなし(CPU・合成のシーン・偽の予測と偽の後処理)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_policy_eval.py
"""

import json
import os
import sys
import types

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import config  # noqa: E402
from jetformer import model as jm  # noqa: E402
from jetformer.policies import post  # noqa: E402
import compare_models as cm  # noqa: E402
import rl_policies as rp  # noqa: E402

N = 8
FRAME_US = 33333


def _scene(tmp_path):
    cv2.imwrite(str(tmp_path / 'img.jpg'), np.full((16, 16, 3), 100, np.uint8))
    return {'timestamps_us': (np.arange(N) * FRAME_US).astype(np.int64), 'cam_files': np.array(['img.jpg'] * N),
            'tof': np.zeros((N, 42, 54), np.uint16), 'imu': np.zeros((N, 10, 6), np.float32),
            'vehicle': np.tile(np.array([1443, 1498, 1000, -2000], np.float32), (N, 1)), 'mode': np.full(N, 5, np.int16)}


def _fake_predict(calls):
    def predict(model, inputs):
        calls.append({k: np.array(v) for k, v in inputs.items()})
        steer = 0.1 + inputs['state'][0, 0]                        # 自己入力が効くように今の操作値を足す
        a = np.full((1, 10, 2), -0.2, np.float32)
        a[0, :, 0] = steer
        return {'actions': a, 'waypoints': np.zeros((1, 10, 2), np.float32), 'speed_logits': np.zeros((1, 10), np.float32)}
    return predict


class CountingPost(object):
    kind = 'count'
    made = []

    def __init__(self, pp, ref):
        self.ref, self.n, self.resets, self.t = ref, 0, 0, []
        CountingPost.made.append(self)

    def reset(self):
        self.resets += 1

    def step(self, outputs, v_now, params, t_ns):
        self.n += 1
        self.t.append(t_ns)
        pred = outputs['actions'] * 500 + 1500
        pred[:, 1] = self.ref[1] - 10 * self.n                     # 呼ばれた順に前進を強める(時刻の順か確かめる)
        return pred.astype(np.float32), {'mode': params.get('tcp_mode', 0), 'branch': 'count', 'v_hat': v_now,
                                         'v_cmd': v_now, 'v_curv': float('nan'), 'cap': 0.0}


@pytest.fixture
def counting_post(monkeypatch):
    mod = types.ModuleType('jetformer_test_eval_post')
    mod.CountingPost = CountingPost
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(post.KINDS, 'count', (mod.__name__, 'CountingPost'))
    CountingPost.made = []


def test_compare_predict_scene_v0_is_actions(tmp_path):
    a = _scene(tmp_path)
    cfg = {'num_modes': 6, 'command_input': False, 'steer_center_us': 1443.0, 'throttle_neutral_us': 1498.0}
    calls = []
    tf, sf, sub = cm.predict_scene(None, cfg, [4, 5], tmp_path, a, predict=_fake_predict(calls))
    assert tf.shape == (N, 10, 2) and not sub.any()
    assert np.allclose(tf[:, :, 0], 0.1 * 500 + 1500) and np.allclose(tf[:, :, 1], -0.2 * 500 + 1500)
    assert np.allclose(sf, tf)                                     # 今の操作値を入れないモデルは自己入力でも同じ
    assert set(calls[0]) == {'image', 'tof', 'imu', 'state', 'mode'} and calls[0]['mode'][0, 5] == 1


def test_compare_predict_scene_runs_post_in_time_order(tmp_path, counting_post):
    a = _scene(tmp_path)
    cfg = {'num_modes': 6, 'command_input': True, 'steer_center_us': 1443.0, 'throttle_neutral_us': 1498.0,
           'postprocess': {'kind': 'count'}}
    tf, sf, _ = cm.predict_scene(None, cfg, [4, 5], tmp_path, a, post_params={'tcp_mode': 1},
                                 predict=_fake_predict([]))
    tf_post, sf_post = CountingPost.made                           # 教師あり・自己入力で別の後処理
    assert tf_post.n == N and sf_post.n == N
    assert tf_post.t == [int(t) * 1000 for t in a['timestamps_us']]
    assert np.allclose(tf[:, 0, 1], [1498 - 10 * (i + 1) for i in range(N)])
    assert not np.allclose(sf[:, :, 0], tf[:, :, 0])               # 自己入力は後処理の後の予測を今の操作値に戻す


def test_compare_load_model_uses_registry(monkeypatch, tmp_path):
    d = tmp_path / 'runs' / 'r'
    d.mkdir(parents=True)
    torch.manual_seed(0)
    m = jm.JetFormer(num_modes=6, dim=32, layers=1, heads=4)
    torch.save(m.state_dict(), d / 'best.pth')
    (d / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'trained_modes': [5], 'policy': 'jetformer_v0',
                                                     'policy_cfg': {'dim': 32, 'layers': 1, 'heads': 4}}))
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    model, cfg, trained, res = cm.load_model('r', device='cpu')
    assert type(model) is jm.JetFormer and trained == [5] and res == {}


def test_gpu_patterns_cover_new_tools():
    for cmd in (['python', 'tools/benchmark_model.py', '--policy', 'act'], ['python', '-u', 'tools/policy_zoo.py', 'train'],
                ['python', 'tools/policy_zoo.py', 'predict'], ['python', 'tools/policy_zoo.py', 'engine-check']):
        assert rp.is_gpu_cmd(cmd), cmd
    assert not rp.is_gpu_cmd(['python', 'tools/policy_zoo.py', 'evaluate'])
