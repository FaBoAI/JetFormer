"""モードの学習の環境(Node のワーカー + Python)の形の試験(1 ワーカー × 2 環境)。"""
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "rl"))
from vec_env_modes import ModeVecEnv  # noqa: E402


@pytest.mark.parametrize("run,dim", [("no_wall", 79), ("recover", 79), ("cond", 97), ("obstacle", 95)])
def test_shapes_and_injected(run, dim):
    with ModeVecEnv(run, n_workers=1, envs_per_worker=2, seed=3) as env:
        assert env.obs_dim == dim
        o = env.reset()
        assert o.shape == (2, dim) and np.isfinite(o).all()
        env.set_progress(2_000_000)
        inj = 0
        for _ in range(300):
            o, r, term, trunc, info = env.step(np.tile([[0.0, -0.3]], (2, 1)))
            assert o.shape == (2, dim) and r.shape == (2,) and info["injected"].shape == (2,) and info["final_obs"].shape == (2, dim)
            inj += int(info["injected"].sum())
        if run != "recover":
            assert inj == 0
