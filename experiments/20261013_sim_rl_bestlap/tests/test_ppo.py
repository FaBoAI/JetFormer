"""PPO の実装(rl/ppo.py・rl/train_rl.py・rl/policy_forward.mjs)の試験。2026-10-13、RL の学習の段

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/test_ppo.py

確かめること:
  - 収集(numpy)の log 確率が、更新(torch)の log 確率と同じ(同じ重みで ≤ 1e-4)
  - 書き出した JS の方策と、Python(float32 の numpy・torch)の決定的な行動の差が 1 万状態で ≤ 1e-4(protocol algorithm.deploy_policy)
  - 打ち切りの価値の補い・GAE の形、チェックポイントからの再開(ステップ数・正規化・重みが戻る)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
from vec_env import NodeVecEnv  # noqa: E402
from ppo import PPO, PPOConfig, export_policy_json, policy_numpy_forward  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    """少しだけ学習した方策(重みが初期値のままだと出力がほぼ 0 で試験にならないため、2 回更新する)。"""
    torch.set_num_threads(2)
    env = NodeVecEnv(n_workers=2, envs_per_worker=8, seed=11)
    cfg = PPOConfig(rollout_steps=128, minibatch=512)
    agent = PPO(env, cfg, total_steps=10_000, seed=11)
    agent.start(11)
    batches = []
    for _ in range(2):
        b, eps, _ = agent.collect()
        batches.append(b)
        agent.update(b)
    # 方策の出力を大きくして、切る範囲・tanh の非線形の部分も試す
    with torch.no_grad():
        agent.model.actor[-1].weight.mul_(30.0)
    obs_raw = []
    o = env.reset(5)
    for t in range(700):
        a = np.clip(agent.act_deterministic(o) + np.random.default_rng(t).normal(0, 0.3, (16, 2)), -1, 1).astype(np.float32)
        obs_raw.append(o)
        o, *_ = env.step(a)
    yield agent, env, np.concatenate(obs_raw)[:10_000], batches
    env.close()


def test_collect_logp_matches_torch(trained):
    agent, env, _, _ = trained
    b, _, _ = agent.collect()
    o, a = torch.from_numpy(b["obs"]), torch.from_numpy(b["act"])
    with torch.no_grad():
        lp = agent.model.dist(o).log_prob(a).sum(-1).numpy()
        v = agent.model.value(o).numpy()
    assert np.abs(lp - b["logp"]).max() <= 1e-4
    assert np.abs(v - b["val"]).max() <= 1e-4
    assert b["obs"].shape == (agent.cfg.rollout_steps * env.num_envs, 79)
    assert np.isfinite(b["adv"]).all() and np.isfinite(b["ret"]).all()


def test_js_policy_matches_python(trained, tmp_path):
    agent, _, obs, _ = trained
    assert obs.shape[0] == 10_000
    w = export_policy_json(agent.model, agent.rms, agent.cfg, {"test": True})
    wp = tmp_path / "w.json"
    wp.write_text(json.dumps(w))
    (tmp_path / "obs.f32").write_bytes(obs.astype("<f4").tobytes())
    subprocess.run(["/usr/bin/node", str(ROOT / "rl/policy_check.mjs"), str(wp), str(tmp_path / "obs.f32"), str(tmp_path / "act.f64")],
                   check=True, cwd=str(ROOT))
    js = np.frombuffer((tmp_path / "act.f64").read_bytes(), "<f8").reshape(-1, 2)
    py32 = policy_numpy_forward(w, obs, np.float32)
    py64 = policy_numpy_forward(w, obs, np.float64)
    th = agent.act_deterministic(obs)
    assert np.abs(js - py64).max() <= 1e-9          # 同じ重み・float64 なら同じ式
    assert np.abs(js - py32).max() <= 1e-4          # protocol: JS と Python の差 ≤ 1e-4
    assert np.abs(js - th).max() <= 1e-4            # 学習中の torch の決定的な方策とも
    assert (np.abs(js) >= 1.0).mean() > 0.01        # 切る範囲も試せている
    assert np.abs(js).mean() > 0.1


def test_checkpoint_roundtrip(trained, tmp_path):
    agent, env, _, _ = trained
    p = tmp_path / "ck.pt"
    torch.save(agent.state_dict(), p)
    other = PPO(env, agent.cfg, total_steps=agent.total_steps, seed=999)
    other.load_state_dict(torch.load(p, weights_only=False))
    assert other.steps == agent.steps and other.updates == agent.updates
    assert np.allclose(other.rms.mean, agent.rms.mean) and np.allclose(other.rms.var, agent.rms.var)
    for a, b in zip(agent.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)
    x = np.random.default_rng(0).normal(0, 1, (5, 79)).astype(np.float32)
    assert np.array_equal(agent.act_deterministic(x), other.act_deterministic(x))


def test_truncation_bootstrap_and_gae():
    """打ち切り(truncated)だけ価値で補い、終了(terminated)では補わないこと。偽の環境で確かめる。"""

    class FakeEnv:
        num_envs = 2

        def __init__(self):
            self.t = 0

        def reset(self, seed=None):
            self.t = 0
            return np.zeros((2, 79), np.float32)

        def step(self, a):
            self.t += 1
            term = np.array([self.t == 3, False])
            trunc = np.array([False, self.t == 3])
            info = {"final_obs": np.ones((2, 79), np.float32) * (self.t == 3), "episodes": []}
            return np.zeros((2, 79), np.float32), np.ones(2, np.float32), term, trunc, info

    env = FakeEnv()
    cfg = PPOConfig(rollout_steps=4, minibatch=8, gamma=0.5, gae_lambda=1.0)
    agent = PPO(env, cfg, total_steps=8, seed=0)
    with torch.no_grad():
        for p in agent.model.critic.parameters():
            p.zero_()
        agent.model.critic[-1].bias.fill_(2.0)    # 価値はいつも 2
    agent.start(0)
    b, _, _ = agent.collect()
    ret = b["ret"].reshape(4, 2)
    # 環境 0: t=2(3 回目)で終了 → 報酬 1、補いなし。環境 1: t=2 で打ち切り → 1 + 0.5·2 = 2
    assert ret[2, 0] == pytest.approx(1.0)
    assert ret[2, 1] == pytest.approx(2.0)
    # 終了の前: 1 + 0.5·1 = 1.5 / 1 + 0.5·2 = 2
    assert ret[1, 0] == pytest.approx(1.5)
    assert ret[1, 1] == pytest.approx(2.0)
    # 最後の行(t=3)は次の状態の価値 2 で補う: 1 + 0.5·2 = 2
    assert ret[3, 0] == pytest.approx(2.0)
