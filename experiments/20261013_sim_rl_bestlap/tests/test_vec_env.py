"""RL 環境の Python 側(rl/vec_env.py)の試験。2026-10-13、RL 環境の段

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap && python -m pytest -q tests/test_vec_env.py

確かめること: 形と型、リセット・ステップの決定性(ワーカーの分け方によらない)、自動リセットと終了前の観測、
教師の行動を Python から送り返しても README の周回時間になること、衝突・停止での終了、開始の引き直し。
Node のワーカーを数個起動するだけの軽い試験(GPU は使わない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from vec_env import NodeVecEnv, OBS_DIM  # noqa: E402

STD = {"gate": 0, "left_offset_m": 0, "left_yaw_deg": 0, "seed": 1}


def smooth_actions(n_envs: int, steps: int, seed: int = 0) -> np.ndarray:
    """ランダムだが滑らかな行動(すぐ衝突しすぎないように)。"""
    rng = np.random.default_rng(seed)
    a = np.zeros((steps, n_envs, 2), np.float32)
    cur = np.zeros((n_envs, 2), np.float32)
    for t in range(steps):
        cur = np.clip(cur + rng.normal(0, 0.15, (n_envs, 2)).astype(np.float32), -1, 1)
        cur[:, 1] = np.clip(cur[:, 1], -1, -0.2)
        a[t] = cur
    return a


def rollout(env: NodeVecEnv, actions: np.ndarray, seed: int):
    obs = [env.reset(seed)]
    starts = list(env.last_starts)
    rew, term, trunc, eps = [], [], [], []
    for a in actions:
        o, r, te, tr, info = env.step(a)
        obs.append(o); rew.append(r); term.append(te); trunc.append(tr); eps += info["episodes"]
    return np.stack(obs), np.stack(rew), np.stack(term), np.stack(trunc), eps, starts


def test_shapes_and_types():
    with NodeVecEnv(n_workers=2, envs_per_worker=3, seed=5) as env:
        assert env.num_envs == 6 and env.info["obs_dim"] == OBS_DIM == 79
        o = env.reset()
        assert o.shape == (6, 79) and o.dtype == np.float32 and np.isfinite(o).all()
        o, r, te, tr, info = env.step(np.zeros((6, 2), np.float32))
        assert o.shape == (6, 79) and r.shape == (6,) and te.dtype == bool and tr.dtype == bool
        assert info["final_obs"].shape == (6, 79) and info["teacher_action"] is None
        assert len(env.last_starts) == 6 and all(0 <= s["gate"] <= 32 for s in env.last_starts)


def test_determinism_and_worker_layout():
    acts = smooth_actions(8, 250, seed=1)
    with NodeVecEnv(n_workers=2, envs_per_worker=4, seed=42) as a:
        ra = rollout(a, acts, seed=7)
    with NodeVecEnv(n_workers=2, envs_per_worker=4, seed=42) as b:
        rb = rollout(b, acts, seed=7)
    with NodeVecEnv(n_workers=1, envs_per_worker=8, seed=42) as c:
        rc = rollout(c, acts, seed=7)
    for x, y, z in zip(ra[:4], rb[:4], rc[:4]):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, z)
    assert ra[5] == rb[5] == rc[5]
    assert len(ra[4]) > 0  # 何本かのエピソードが終わっている
    # 同じ VecEnv でリセットの種を同じにすれば同じ開始
    with NodeVecEnv(n_workers=1, envs_per_worker=4, seed=0) as d:
        o1 = d.reset(11); s1 = d.last_starts
        d.step(np.zeros((4, 2), np.float32))
        o2 = d.reset(11); s2 = d.last_starts
        np.testing.assert_array_equal(o1, o2)
        assert s1 == s2
        o3 = d.reset(12)
        assert not np.array_equal(o1, o3)


def test_collision_autoreset_and_final_obs():
    with NodeVecEnv(n_workers=1, envs_per_worker=2, seed=0, starts=[STD]) as env:
        o0 = env.reset()
        np.testing.assert_array_equal(o0[0], o0[1])
        acts = np.array([[0, 1], [0, 1]], np.float32)  # 4.5 m/s で直進 → 壁に衝突
        for t in range(200):
            o, r, te, tr, info = env.step(acts)
            if te.any():
                break
        assert te.all() and not tr.any() and t in range(40, 60)
        assert (r < -9.9).all()
        eps = info["episodes"]
        assert len(eps) == 2 and all(e["reason"] == "collision" and e["collisions"] == 1 for e in eps)
        # 終了前の最後の観測(速さ 0、すき間が小さい)と、リセット後の観測(最初と同じ)
        assert np.all(info["final_obs"][:, 0] == 0) and np.all(info["final_obs"][:, 78] < 0.1)
        np.testing.assert_array_equal(o, o0)


def test_stall_terminates_in_train_mode_only():
    with NodeVecEnv(n_workers=1, envs_per_worker=1, seed=0, starts=[STD]) as env:
        env.reset()
        for t in range(200):
            o, r, te, tr, info = env.step(np.array([[0, -1]], np.float32))
            if te.any():
                break
        assert info["episodes"][0]["reason"] == "stall" and info["episodes"][0]["frames"] == 89 and r[0] <= -5
    with NodeVecEnv(n_workers=1, envs_per_worker=1, seed=0, starts=[STD], mode="eval") as env:
        env.reset()
        for t in range(700):
            o, r, te, tr, info = env.step(np.array([[0, -1]], np.float32))
            if te.any() or tr.any():
                break
        assert tr[0] and not te[0] and t == 599 and info["episodes"][0]["reason"] == "collection_duration_complete"


def test_teacher_actions_through_python_reproduce_readme_laps():
    """教師の行動(正規化した a)を float32 で Python から送り返して走らせる。float32 の丸めの分だけずれるが、
    周回時間は README の値と 1e-3 秒(1 フレームの 1/30)以内で一致する。"""
    starts = [STD, {"gate": 0, "left_offset_m": 0.03, "left_yaw_deg": 3, "seed": 2},
              {"gate": 0, "left_offset_m": -0.03, "left_yaw_deg": -3, "seed": 3}, {"gate": 0, "left_offset_m": 0.012, "left_yaw_deg": -1.4, "seed": 4}]
    with NodeVecEnv(n_workers=2, envs_per_worker=2, seed=0, starts=starts, teacher_actions=True) as env:
        env.reset()
        a = env.last_teacher_action
        assert np.isfinite(a).all() and (np.abs(a) <= 1).all()
        eps = []
        for t in range(600):
            o, r, te, tr, info = env.step(a)
            eps += info["episodes"]
            a = info["teacher_action"]
            if len(eps) == 4:
                break
        assert len(eps) == 4
        by = {e["env"]: e for e in eps}
        assert all(e["reason"] == "collection_duration_complete" and e["collisions"] == 0 and e["completed_laps"] == 2 for e in eps)
        lap1 = by[0]["laps"][0]["elapsed"]; lap2 = by[0]["laps"][1]["elapsed"]
        assert abs(lap1 - 8.75841899184614) < 1e-3 and abs(lap2 - 8.370092214503968) < 1e-3
        for e in eps:
            assert 8.70 < e["laps"][0]["elapsed"] < 8.77 and 8.35 < e["laps"][1]["elapsed"] < 8.42
            assert all(l["gates"] == 33 and l["crossings"] == 1 for l in e["laps"])


def test_random_starts_are_valid_and_follow_distribution():
    with NodeVecEnv(n_workers=1, envs_per_worker=64, seed=3) as env:
        env.reset()
        s = env.last_starts
        gates = np.array([x["gate"] for x in s]); offs = np.array([x["left_offset_m"] for x in s]); yaws = np.array([x["left_yaw_deg"] for x in s])
        assert (np.abs(offs) <= 0.12).all() and (np.abs(yaws) <= 8).all()
        assert 15 <= (gates == 0).sum() <= 50 and len(set(gates.tolist())) > 10


@pytest.mark.parametrize("variant,cap", [("main", 4.5), ("open", 7.24)])
def test_variant_speed_cap(variant, cap):
    with NodeVecEnv(n_workers=1, envs_per_worker=1, seed=0, starts=[STD], variant=variant) as env:
        assert env.info["speed_cap_mps"] == cap
        env.reset()
        vmax = 0.0
        for t in range(40):
            o, r, te, tr, info = env.step(np.array([[0, 1]], np.float32))
            vmax = max(vmax, float(o[0, 0]) * 4.5)
            if te.any():
                break
        assert vmax <= cap + 1e-4 and vmax > 3.0
