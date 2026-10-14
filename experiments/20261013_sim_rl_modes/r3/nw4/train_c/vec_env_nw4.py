# r4 no_wall(2026-10-14): r3/nw/train/vec_env_nw3.py の写し。違いはワーカーのファイル(env_worker_nw4.mjs)だけ。以下は r3 の説明。
# r3 no_wall(2026-10-14): rl/vec_env_modes.py の写し。違いはワーカーのファイル(env_worker_nw3.mjs)だけ
"""モードの学習の環境をまとめて動かす Python 側(2026-10-13、sim_rl_modes)。ベストラップの rl/vec_env.py の写しで、違いは:
  - 観測の次元はワーカーの返事(obs_dim)に従う(79 / 95 / 97)
  - step の info["injected"](差し込んだ行動なら 1。PPO が方策の損失から外す)
  - set_progress(steps): 学習の総ステップ数をワーカーへ(乱しの強さのカリキュラム)
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE / "env_worker_nw4.mjs"
NODE = os.environ.get("RL_NODE", "/usr/bin/node")
ACT_DIM = 2


class WorkerError(RuntimeError):
    pass


class _Worker:
    def __init__(self, cfg: dict, node: str):
        self.proc = subprocess.Popen([node, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0, cwd=str(HERE))
        self.n = cfg["n_envs"]
        self.out, self.inp = self.proc.stdout, self.proc.stdin
        payload = json.dumps(cfg).encode()
        self.inp.write(b"I" + struct.pack("<I", len(payload)) + payload)
        self.hello, _ = self.read()
        self.D = int(self.hello["obs_dim"])

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.out.readinto(view[got:])
            if not k:
                raise WorkerError(f"env_worker_modes(pid {self.proc.pid})が途中で終わりました(終了コード {self.proc.poll()})")
            got += k
        return bytes(buf)

    def read(self):
        jlen = struct.unpack("<I", self._read_exact(4))[0]
        meta = json.loads(self._read_exact(jlen)) if jlen else {}
        if "error" in meta:
            raise WorkerError("env_worker_modes のエラー: " + meta["error"])
        blen = struct.unpack("<I", self._read_exact(4))[0]
        data = np.frombuffer(self._read_exact(blen), dtype="<f4") if blen else np.zeros(0, np.float32)
        return meta, data

    def close(self):
        if self.proc.poll() is None:
            try:
                self.inp.write(b"Q")
                self.inp.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class ModeVecEnv:
    def __init__(self, run: str, n_workers: int = 4, envs_per_worker: int = 16, seed: int = 0, mode: str = "train", episode_mode: str = "continuous",
                 tuning: dict | None = None, starts="random", time_limit: float = 20.0, node: str = NODE):
        self.run = run
        self.n_workers, self.envs_per_worker = n_workers, envs_per_worker
        self.num_envs = n_workers * envs_per_worker
        self.workers: list[_Worker] = []
        try:
            for w in range(n_workers):
                cfg = dict(n_envs=envs_per_worker, env_offset=w * envs_per_worker, seed=int(seed) & 0xFFFFFFFF, run=run, mode=mode,
                           episode_mode=episode_mode, time_limit=time_limit, tuning=tuning or {}, starts=starts)
                self.workers.append(_Worker(cfg, node))
        except BaseException:
            self.close()
            raise
        self.obs_dim, self.act_dim = self.workers[0].D, ACT_DIM
        self.info = dict(self.workers[0].hello, n_workers=n_workers, envs_per_worker=envs_per_worker, num_envs=self.num_envs)

    def reset(self, seed: int | None = None) -> np.ndarray:
        for w in self.workers:
            w.inp.write(b"R" + struct.pack("<i", -1 if seed is None else int(seed) & 0x7FFFFFFF))
        obs = []
        for w in self.workers:
            meta, data = w.read()
            obs.append(data[: w.n * w.D].reshape(w.n, w.D))
        return np.concatenate(obs).copy()

    def set_progress(self, steps: int) -> None:
        for w in self.workers:
            w.inp.write(b"P" + struct.pack("<d", float(steps)))

    def step_async_group(self, worker_ids, actions) -> None:
        a = np.asarray(actions, dtype=np.float32).reshape(-1, ACT_DIM)
        o = 0
        for k in worker_ids:
            w = self.workers[k]
            w.inp.write(b"S" + np.ascontiguousarray(a[o:o + w.n], dtype="<f4").tobytes())
            o += w.n

    def step_wait_group(self, worker_ids):
        obs, rew, term, trunc, final, inj, episodes = [], [], [], [], [], [], []
        for k in worker_ids:
            w = self.workers[k]
            meta, data = w.read()
            n, D, o = w.n, w.D, 0
            obs.append(data[o:o + n * D].reshape(n, D)); o += n * D
            rew.append(data[o:o + n]); o += n
            term.append(data[o:o + n]); o += n
            trunc.append(data[o:o + n]); o += n
            final.append(data[o:o + n * D].reshape(n, D)); o += n * D
            inj.append(data[o:o + n]); o += n
            episodes += meta["episodes"]
        info = {"final_obs": np.concatenate(final).copy(), "episodes": episodes, "injected": np.concatenate(inj).astype(bool)}
        return (np.concatenate(obs).copy(), np.concatenate(rew).copy(), np.concatenate(term).astype(bool), np.concatenate(trunc).astype(bool), info)

    def worker_groups(self, n_groups: int = 2):
        n_groups = max(1, min(n_groups, self.n_workers))
        bounds = np.linspace(0, self.n_workers, n_groups + 1).round().astype(int)
        out = []
        for g in range(n_groups):
            ws = list(range(bounds[g], bounds[g + 1]))
            lo = sum(self.workers[k].n for k in range(bounds[g]))
            hi = lo + sum(self.workers[k].n for k in ws)
            out.append((ws, slice(lo, hi)))
        return out

    def step(self, actions):
        ids = list(range(self.n_workers))
        self.step_async_group(ids, actions)
        return self.step_wait_group(ids)

    def close(self) -> None:
        for w in self.workers:
            w.close()
        self.workers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
