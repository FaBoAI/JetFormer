# 離脱復活 r3(2026-10-14): rl/ppo_modes.py の写し(変えていない。共有のファイルが途中で変わっても学習が変わらないように)
"""PPO(ベストラップの rl/ppo.py の写し。2026-10-13、sim_rl_modes)。違い:
  - 観測の次元 obs_dim を引数にする(79 / 95 / 97)
  - 差し込んだ行動(離脱復活の乱し)のステップは、方策の損失・エントロピーから外す(価値の学習には使う)。env の info["injected"]
  - 書き出しの schema は jf-sim-rl-mode-policy/v1(obs_dim・obs_wrapper・mode を持つ。JS の policy_forward.mjs は mode_runtime が読み替える)
  - 初期化元(ベストラップのチェックポイント、79 次元)から始めるとき、新しい入力の 1 層目の重みを 0 で広げる(init_from_checkpoint)
アルゴリズム・既定の値はベストラップと同じ(MLP 256-256、rollout 256 × 64 環境、5 エポック、ミニバッチ 4,096、clip 0.2、γ 0.995、λ 0.95、target_kl 0.03)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

ACT_DIM = 2
MODE_POLICY_SCHEMA = "jf-sim-rl-mode-policy/v1"


@dataclass
class PPOConfig:
    hidden: int = 256
    log_std_init: float = -0.7
    rollout_steps: int = 256
    epochs: int = 5
    minibatch: int = 4096
    lr_start: float = 1e-4          # 続きの学習(r2 と同じ)
    lr_end: float = 3e-5
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    gamma: float = 0.995
    gae_lambda: float = 0.95
    obs_clip: float = 5.0
    obs_eps: float = 1e-8

    def to_dict(self) -> dict:
        return asdict(self)


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        new_mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def normalize(self, x: np.ndarray, clip: float = 5.0, eps: float = 1e-8) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + eps), -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = np.asarray(d["mean"], np.float64), np.asarray(d["var"], np.float64), float(d["count"])


def _mlp(n_in: int, hidden: int, n_out: int, out_gain: float) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2) if m.out_features == hidden else out_gain)
            nn.init.zeros_(m.bias)
    return net


class ActorCritic(nn.Module):
    def __init__(self, cfg: PPOConfig, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.actor = _mlp(obs_dim, cfg.hidden, ACT_DIM, 0.01)
        self.critic = _mlp(obs_dim, cfg.hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(cfg.log_std_init)))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def widen_state_dict(model_sd: dict, rms_sd: dict, obs_dim: int) -> tuple[dict, dict]:
    """79 次元の重み・正規化を obs_dim へ広げる。1 層目の新しい列は 0(初めは初期化元と同じ出力)、正規化は平均 0・分散 1。"""
    sd = {k: v.clone() for k, v in model_sd.items()}
    for net in ("actor", "critic"):
        w = sd[f"{net}.0.weight"]
        old = w.shape[1]
        if old > obs_dim:
            raise ValueError(f"初期化元の観測の次元 {old} が新しい {obs_dim} より大きい")
        if old < obs_dim:
            sd[f"{net}.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], obs_dim - old, dtype=w.dtype)], 1)
    old = len(rms_sd["mean"])
    rms = {"mean": np.concatenate([np.asarray(rms_sd["mean"], np.float64), np.zeros(obs_dim - old)]),
           "var": np.concatenate([np.asarray(rms_sd["var"], np.float64), np.ones(obs_dim - old)]), "count": float(rms_sd["count"])}
    return sd, rms


class PPO:
    def __init__(self, env, cfg: PPOConfig, total_steps: int, seed: int = 0, obs_dim: int | None = None):
        self.env, self.cfg, self.total_steps = env, cfg, int(total_steps)
        self.n = env.num_envs
        self.obs_dim = int(obs_dim or env.obs_dim)
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.model = ActorCritic(cfg, self.obs_dim)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr_start, eps=1e-5)
        self.rms = RunningMeanStd((self.obs_dim,))
        self.steps = 0
        self.updates = 0
        self.raw_obs = None

    def init_from_checkpoint(self, d: dict, log_std_init: float | None = None) -> None:
        sd, rms = widen_state_dict(d["model"], d["rms"], self.obs_dim)
        self.model.load_state_dict(sd)
        self.rms.load_state_dict(rms)
        if log_std_init is not None:
            with torch.no_grad():
                self.model.log_std.fill_(log_std_init)

    def lr_now(self) -> float:
        frac = min(1.0, self.steps / max(1, self.total_steps))
        return self.cfg.lr_start + (self.cfg.lr_end - self.cfg.lr_start) * frac

    def start(self, seed: int | None = None) -> None:
        self.raw_obs = self.env.reset(seed)
        self.rms.update(self.raw_obs)

    def norm(self, raw: np.ndarray) -> np.ndarray:
        return self.rms.normalize(raw, self.cfg.obs_clip, self.cfg.obs_eps)

    def _np_params(self):
        def lin(seq):
            return [(m.weight.detach().numpy().T.astype(np.float32).copy(), m.bias.detach().numpy().astype(np.float32).copy())
                    for m in seq if isinstance(m, nn.Linear)]
        return lin(self.model.actor), lin(self.model.critic), self.model.log_std.detach().numpy().astype(np.float32).copy()

    @staticmethod
    def _np_mlp(P, x):
        h = np.tanh(x @ P[0][0] + P[0][1])
        h = np.tanh(h @ P[1][0] + P[1][1])
        return h @ P[2][0] + P[2][1]

    def collect(self):
        cfg, n, T, D = self.cfg, self.n, self.cfg.rollout_steps, self.obs_dim
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(self.steps)
        A, C, log_std = self._np_params()
        std = np.exp(log_std)
        lp_const = float(-(log_std.sum()) - ACT_DIM * 0.5 * math.log(2 * math.pi))
        obs_buf = np.zeros((T, n, D), np.float32)
        act_buf = np.zeros((T, n, ACT_DIM), np.float32)
        logp_buf = np.zeros((T, n), np.float32)
        rew_buf = np.zeros((T, n), np.float32)
        done_buf = np.zeros((T, n), np.float32)
        val_buf = np.zeros((T, n), np.float32)
        pmask_buf = np.ones((T, n), np.float32)    # 1 = 方策の損失に使う、0 = 差し込んだ行動のステップ
        episodes = []
        last_v = np.zeros(n, np.float32)

        def act(on):
            mu = self._np_mlp(A, on)
            v = self._np_mlp(C, on)[:, 0]
            eps = self.np_rng.standard_normal(mu.shape).astype(np.float32)
            a = (mu + std * eps).astype(np.float32)
            logp = (-0.5 * (eps.astype(np.float64) ** 2).sum(-1) + lp_const).astype(np.float32)
            return a, logp, v

        groups = self.env.worker_groups(2)
        raw = self.raw_obs.copy()
        for ws, sl in groups:
            on = self.norm(raw[sl])
            a, logp, v = act(on)
            obs_buf[0, sl], act_buf[0, sl], logp_buf[0, sl], val_buf[0, sl] = on, a, logp, v
            self.env.step_async_group(ws, np.clip(a, -1.0, 1.0))
        for t in range(T):
            for ws, sl in groups:
                raw_next, r, term, trunc, info = self.env.step_wait_group(ws)
                r = r.astype(np.float32).copy()
                if trunc.any():
                    idx = np.nonzero(trunc & ~term)[0]
                    if idx.size:
                        r[idx] += cfg.gamma * self._np_mlp(C, self.norm(info["final_obs"][idx]))[:, 0]
                rew_buf[t, sl] = r
                done_buf[t, sl] = (term | trunc).astype(np.float32)
                pmask_buf[t, sl] = (~info["injected"]).astype(np.float32)
                episodes += info["episodes"]
                raw[sl] = raw_next
                self.rms.update(raw_next)
                on = self.norm(raw_next)
                if t + 1 < T:
                    a, logp, v = act(on)
                    obs_buf[t + 1, sl], act_buf[t + 1, sl], logp_buf[t + 1, sl], val_buf[t + 1, sl] = on, a, logp, v
                    self.env.step_async_group(ws, np.clip(a, -1.0, 1.0))
                else:
                    last_v[sl] = self._np_mlp(C, on)[:, 0]
        self.raw_obs = raw
        adv = np.zeros((T, n), np.float32)
        gae = np.zeros(n, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nv * nonterm - val_buf[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + val_buf
        self.steps += T * n
        batch = dict(obs=obs_buf.reshape(-1, D), act=act_buf.reshape(-1, ACT_DIM), logp=logp_buf.reshape(-1),
                     adv=adv.reshape(-1), ret=ret.reshape(-1), val=val_buf.reshape(-1), pmask=pmask_buf.reshape(-1))
        stats = dict(rollout_reward_mean=float(rew_buf.mean()), action_abs_mean=np.abs(act_buf).mean((0, 1)).tolist(),
                     action_sat_frac=float((np.abs(act_buf) >= 1.0).mean()), injected_frac=float(1 - pmask_buf.mean()))
        return batch, episodes, stats

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        lr = self.lr_now()
        for g in self.opt.param_groups:
            g["lr"] = lr
        N = batch["obs"].shape[0]
        t = {k: torch.from_numpy(v) for k, v in batch.items()}
        pg_losses, v_losses, kls, clipfracs = [], [], [], []
        stop = False
        epochs_done = 0
        for ep in range(cfg.epochs):
            perm = torch.from_numpy(self.np_rng.permutation(N))
            for i in range(0, N, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                o, a, lp_old, pm = t["obs"][idx], t["act"][idx], t["logp"][idx], t["pmask"][idx]
                adv = t["adv"][idx]
                # アドバンテージの正規化は方策の損失に使うステップだけで
                msum = pm.sum().clamp(min=1.0)
                am = (adv * pm).sum() / msum
                asd = (((adv - am) ** 2 * pm).sum() / msum).sqrt()
                adv = (adv - am) / (asd + 1e-8)
                d = self.model.dist(o)
                lp = d.log_prob(a).sum(-1)
                ratio = (lp - lp_old).exp()
                pg = -(torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv) * pm).sum() / msum
                v = self.model.value(o)
                v_loss = 0.5 * ((v - t["ret"][idx]) ** 2).mean()
                ent = (d.entropy().sum(-1) * pm).sum() / msum
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                with torch.no_grad():
                    log_ratio = lp - lp_old
                    approx_kl = ((((ratio - 1) - log_ratio) * pm).sum() / msum).item()
                    clipfracs.append(((((ratio - 1).abs() > cfg.clip).float() * pm).sum() / msum).item())
                if approx_kl > cfg.target_kl:
                    kls.append(approx_kl)
                    stop = True
                    break
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                pg_losses.append(pg.item()); v_losses.append(v_loss.item()); kls.append(approx_kl)
            epochs_done = ep + 1
            if stop:
                break
        self.updates += 1
        y, yhat = batch["ret"], batch["val"]
        ev = float(1 - np.var(y - yhat) / (np.var(y) + 1e-8))
        return dict(lr=lr, pg_loss=float(np.mean(pg_losses)) if pg_losses else None, v_loss=float(np.mean(v_losses)) if v_losses else None,
                    approx_kl=float(np.mean(kls)) if kls else None, max_kl=float(np.max(kls)) if kls else None,
                    clipfrac=float(np.mean(clipfracs)) if clipfracs else None, kl_early_stop=stop, epochs_done=epochs_done,
                    minibatches=len(pg_losses), explained_variance=ev, log_std=self.model.log_std.detach().tolist())

    @torch.no_grad()
    def act_deterministic(self, raw_obs: np.ndarray) -> np.ndarray:
        o = torch.from_numpy(self.norm(np.asarray(raw_obs, np.float32)))
        return self.model.actor(o).clamp(-1, 1).numpy()

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "rms": self.rms.state_dict(), "steps": self.steps,
                "updates": self.updates, "cfg": self.cfg.to_dict(), "total_steps": self.total_steps, "obs_dim": self.obs_dim,
                "torch_rng": torch.get_rng_state(), "np_rng": self.np_rng.bit_generator.state}

    def load_state_dict(self, d: dict) -> None:
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        self.rms.load_state_dict(d["rms"])
        self.steps, self.updates = int(d["steps"]), int(d["updates"])
        torch.set_rng_state(d["torch_rng"])
        self.np_rng.bit_generator.state = d["np_rng"]


def export_policy_json(model: ActorCritic, rms: RunningMeanStd, cfg: PPOConfig, meta: dict, obs_wrapper: str, mode: str) -> dict:
    """ブラウザ・Node の mode_runtime.mjs(policy_forward.mjs)が読む重み。観測の正規化は固定値。層の重みは行優先(出力 × 入力)。"""
    layers = []
    lin = [m for m in model.actor if isinstance(m, nn.Linear)]
    for k, m in enumerate(lin):
        w = m.weight.detach().to(torch.float32).numpy()
        b = m.bias.detach().to(torch.float32).numpy()
        layers.append({"n_in": int(w.shape[1]), "n_out": int(w.shape[0]), "act": "tanh" if k < len(lin) - 1 else "linear",
                       "w": [float(x) for x in w.reshape(-1)], "b": [float(x) for x in b]})
    std = np.sqrt(rms.var + cfg.obs_eps)
    return {"schema": MODE_POLICY_SCHEMA, "obs_dim": int(model.obs_dim), "act_dim": ACT_DIM, "obs_wrapper": obs_wrapper, "mode": mode,
            "obs_mean": [float(np.float32(x)) for x in rms.mean], "obs_std": [float(np.float32(x)) for x in std],
            "obs_clip": cfg.obs_clip, "action_clip": 1.0, "layers": layers,
            "log_std": [float(x) for x in model.log_std.detach().numpy()], "meta": meta}


def policy_numpy_forward(weights: dict, raw_obs: np.ndarray, dtype=np.float32) -> np.ndarray:
    x = np.asarray(raw_obs, dtype)
    mean = np.asarray(weights["obs_mean"], dtype)
    std = np.asarray(weights["obs_std"], dtype)
    x = np.clip((x - mean) / std, -weights["obs_clip"], weights["obs_clip"])
    for L in weights["layers"]:
        w = np.asarray(L["w"], dtype).reshape(L["n_out"], L["n_in"])
        x = x @ w.T + np.asarray(L["b"], dtype)
        if L["act"] == "tanh":
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0)
