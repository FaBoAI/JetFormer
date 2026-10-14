"""PPO の写し(観測の次元を引数に・差し込んだ行動を方策の損失から外す・1 層目を 0 で広げる・書き出し)の試験。"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "rl"))
from ppo_modes import ActorCritic, PPOConfig, PPO, RunningMeanStd, widen_state_dict, export_policy_json, policy_numpy_forward  # noqa: E402
import train_modes  # noqa: E402

INIT = Path("/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/main-s1/checkpoints/step_010010624.pt")


@pytest.mark.parametrize("dim", [79, 95, 97])
def test_widen_keeps_output(dim):
    d = torch.load(INIT, map_location="cpu", weights_only=False)
    cfg = PPOConfig()
    old = ActorCritic(cfg, 79)
    old.load_state_dict(d["model"])
    sd, rms = widen_state_dict(d["model"], d["rms"], dim)
    new = ActorCritic(cfg, dim)
    new.load_state_dict(sd)
    r_old, r_new = RunningMeanStd((79,)), RunningMeanStd((dim,))
    r_old.load_state_dict(d["rms"])
    r_new.load_state_dict(rms)
    x = np.random.default_rng(0).normal(size=(32, 79)).astype(np.float32)
    extra = np.random.default_rng(1).normal(size=(32, dim - 79)).astype(np.float32)
    with torch.no_grad():
        a = old.actor(torch.from_numpy(r_old.normalize(x))).numpy()
        b = new.actor(torch.from_numpy(r_new.normalize(np.concatenate([x, extra], 1)))).numpy()
        va = old.value(torch.from_numpy(r_old.normalize(x))).numpy()
        vb = new.value(torch.from_numpy(r_new.normalize(np.concatenate([x, extra], 1)))).numpy()
    assert np.abs(a - b).max() < 1e-6 and np.abs(va - vb).max() < 1e-5
    assert np.allclose(rms["mean"][79:], 0) and np.allclose(rms["var"][79:], 1)


def test_export_matches_js():
    """書き出した重み(jf-sim-rl-mode-policy/v1)を、ブラウザと同じ mode_runtime.mjs の読み方で JS が計算した値と比べる(≤ 1e-4)。"""
    d = torch.load(INIT, map_location="cpu", weights_only=False)
    cfg = PPOConfig()
    sd, rms = widen_state_dict(d["model"], d["rms"], 97)
    m = ActorCritic(cfg, 97)
    m.load_state_dict(sd)
    with torch.no_grad():
        m.actor[0].weight[:, 79:] = torch.randn(256, 18) * 0.05
    r = RunningMeanStd((97,))
    r.load_state_dict(rms)
    w = export_policy_json(m, r, cfg, {"test": True}, "cond97", "cond")
    obs = np.random.default_rng(2).normal(size=(20, 97)).astype(np.float64)
    py = policy_numpy_forward(w, obs, np.float64)
    tmp = EXP / "tests" / "_tmp_policy.json"
    tmp.write_text(json.dumps({"w": w, "obs": obs.tolist()}))
    try:
        js = ("const m = await import('" + str(EXP / "rl/runtime/mode_runtime.mjs") + "'); const fs = await import('node:fs');"
              " const d = JSON.parse(fs.readFileSync(process.argv[1], 'utf8')); const act = m.policyFromWeights(d.w, 97);"
              " process.stdout.write(JSON.stringify(d.obs.map(o => act(Float64Array.from(o), [0, 0]))));")
        out = subprocess.run(["node", "--input-type=module", "-e", js, str(tmp)], capture_output=True, text=True, check=True)
    finally:
        tmp.unlink()
    assert np.abs(np.asarray(json.loads(out.stdout)) - py).max() < 1e-4
    assert w["schema"] == "jf-sim-rl-mode-policy/v1" and w["obs_dim"] == 97 and w["obs_wrapper"] == "cond97"


class _FakeEnv:
    num_envs, obs_dim = 4, 79


def test_policy_loss_ignores_injected_steps():
    """差し込んだ行動のステップ(pmask = 0)は、どんなアドバンテージでも方策の勾配に効かない。"""
    torch.manual_seed(0)
    cfg = PPOConfig(epochs=1, minibatch=64, target_kl=1e9)
    rng = np.random.default_rng(0)
    N = 64
    base = dict(obs=rng.normal(size=(N, 79)).astype(np.float32), act=rng.normal(size=(N, 2)).astype(np.float32),
                logp=np.full(N, -1.0, np.float32), adv=rng.normal(size=N).astype(np.float32), ret=np.zeros(N, np.float32),
                val=np.zeros(N, np.float32), pmask=np.r_[np.ones(48), np.zeros(16)].astype(np.float32))
    grads = []
    for poison in (0.0, 1e6):
        agent = PPO(_FakeEnv(), cfg, 1000, seed=0)
        agent.np_rng = np.random.default_rng(5)
        b = {k: v.copy() for k, v in base.items()}
        b["adv"][48:] = poison
        b["ret"][:] = 0
        agent.opt.zero_grad()
        t = {k: torch.from_numpy(v) for k, v in b.items()}
        pm = t["pmask"]
        adv = t["adv"]
        msum = pm.sum()
        am = (adv * pm).sum() / msum
        asd = (((adv - am) ** 2 * pm).sum() / msum).sqrt()
        adv = (adv - am) / (asd + 1e-8)
        d = agent.model.dist(t["obs"])
        lp = d.log_prob(t["act"]).sum(-1)
        ratio = (lp - t["logp"]).exp()
        pg = -(torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv) * pm).sum() / msum
        pg.backward()
        grads.append(agent.model.actor[0].weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_tuning_ranges():
    train_modes.check_tuning("no_wall", {"clearance_barrier_w": 0.3, "terminal_clearance_m": 0.07})
    with pytest.raises(SystemExit):
        train_modes.check_tuning("no_wall", {"clearance_barrier_w": 0.9})
    with pytest.raises(SystemExit):
        train_modes.check_tuning("cond", {"smooth_steer": -0.05})
    train_modes.check_tuning("cond", {"tracking_w": 0.1})
