"""RL の環境をまとめて動かす Python 側(2026-10-13、RL 環境の段)。

Node のワーカー(rl/env_worker.mjs)を W 個起動し、各ワーカーが E 個の環境を持つ(既定 4 × 16 = 64 環境)。
全ワーカーへ先に行動を送ってから返事を読むので、ワーカーどうしは並行に進む。
numpy だけを使う(torch・gymnasium は要らない)。conda の jetracer 環境(Python 3.13)で動かす。

    from vec_env import NodeVecEnv
    with NodeVecEnv(n_workers=4, envs_per_worker=16, seed=0) as env:
        obs = env.reset()                       # (64, 79) float32
        obs, rew, term, trunc, info = env.step(actions)   # actions: (64, 2)、[-1, 1] に切られる
        info["final_obs"]   # (64, 79): 終わった環境の、終了前の最後の観測(打ち切りの価値の補いに使う)
        info["episodes"]    # 終わったエピソードの要約(周回時間・終わった理由・最小すき間など)のリスト
        info["teacher_action"]  # teacher_actions=True のとき (64, 2)。RC 教師の約束の範囲の外は NaN

自動リセット: 終わった環境は、返した obs がすでにリセット後の観測になっている。
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE / "env_worker.mjs"
NODE = os.environ.get("RL_NODE", "/usr/bin/node")
OBS_DIM = 79
ACT_DIM = 2


class WorkerError(RuntimeError):
    pass


class _Worker:
    def __init__(self, cfg: dict, node: str, node_args: list[str]):
        self.proc = subprocess.Popen([node, *node_args, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, bufsize=0, cwd=str(HERE))
        self.n = cfg["n_envs"]
        self.cfg = cfg
        self.out = self.proc.stdout
        self.inp = self.proc.stdin
        payload = json.dumps(cfg).encode()
        self.inp.write(b"I" + struct.pack("<I", len(payload)) + payload)
        self.hello, _ = self.read()

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.out.readinto(view[got:])
            if not k:
                raise WorkerError(f"env_worker(pid {self.proc.pid})が途中で終わりました(終了コード {self.proc.poll()})")
            got += k
        return bytes(buf)

    def read(self):
        jlen = struct.unpack("<I", self._read_exact(4))[0]
        meta = json.loads(self._read_exact(jlen)) if jlen else {}
        if "error" in meta:
            raise WorkerError("env_worker のエラー: " + meta["error"])
        blen = struct.unpack("<I", self._read_exact(4))[0]
        data = np.frombuffer(self._read_exact(blen), dtype="<f4") if blen else np.zeros(0, np.float32)
        return meta, data

    def send_reset(self, seed: int = -1):
        self.inp.write(b"R" + struct.pack("<i", int(seed)))

    def send_step(self, actions: np.ndarray):
        self.inp.write(b"S" + np.ascontiguousarray(actions, dtype="<f4").tobytes())

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


class NodeVecEnv:
    """Node のワーカーで動く 64 環境(既定)。行動・観測は float32。"""

    def __init__(self, n_workers: int = 4, envs_per_worker: int = 16, seed: int = 0, variant: str = "main",
                 mode: str = "train", episode_mode: str = "continuous", core: str = "fast", teacher_actions: bool = False,
                 starts="random", time_limit: float = 20.0, node: str = NODE, node_args: list[str] | None = None):
        if variant not in ("main", "open"):
            raise ValueError("variant は main か open です")
        self.n_workers, self.envs_per_worker = n_workers, envs_per_worker
        self.num_envs = n_workers * envs_per_worker
        self.obs_dim, self.act_dim = OBS_DIM, ACT_DIM
        self.teacher_actions = teacher_actions
        self.seed = seed
        self.workers: list[_Worker] = []
        try:
            for w in range(n_workers):
                cfg = dict(n_envs=envs_per_worker, env_offset=w * envs_per_worker, seed=int(seed) & 0xFFFFFFFF, variant=variant,
                           mode=mode, episode_mode=episode_mode, core=core, teacher_actions=teacher_actions, starts=starts,
                           time_limit=time_limit)
                self.workers.append(_Worker(cfg, node, node_args or []))
        except BaseException:
            self.close()
            raise
        hello = self.workers[0].hello
        if hello.get("obs_dim") != OBS_DIM:
            raise WorkerError(f"観測の次元が違います: {hello}")
        self.info = dict(hello, n_workers=n_workers, envs_per_worker=envs_per_worker, num_envs=self.num_envs)
        self.last_starts: list[dict] = []

    # ---- リセット ----
    def reset(self, seed: int | None = None) -> np.ndarray:
        for w in self.workers:
            w.send_reset(-1 if seed is None else int(seed) & 0x7FFFFFFF)
        obs, teach, starts = [], [], []
        for w in self.workers:
            meta, data = w.read()
            starts += meta["starts"]
            n = w.n
            obs.append(data[: n * OBS_DIM].reshape(n, OBS_DIM))
            if self.teacher_actions:
                teach.append(data[n * OBS_DIM:].reshape(n, 2))
        self.last_starts = starts
        self.last_teacher_action = np.concatenate(teach).copy() if self.teacher_actions else None
        return np.concatenate(obs).copy()

    # ---- 1 フレーム進める ----
    def step_async(self, actions: np.ndarray) -> None:
        a = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, ACT_DIM)
        for k, w in enumerate(self.workers):
            w.send_step(a[k * w.n:(k + 1) * w.n])

    def _wait_workers(self, workers):
        obs, rew, term, trunc, final, teach, episodes = [], [], [], [], [], [], []
        for w in workers:
            meta, data = w.read()
            n, o = w.n, 0
            obs.append(data[o:o + n * OBS_DIM].reshape(n, OBS_DIM)); o += n * OBS_DIM
            rew.append(data[o:o + n]); o += n
            term.append(data[o:o + n]); o += n
            trunc.append(data[o:o + n]); o += n
            final.append(data[o:o + n * OBS_DIM].reshape(n, OBS_DIM)); o += n * OBS_DIM
            if self.teacher_actions:
                teach.append(data[o:o + 2 * n].reshape(n, 2)); o += 2 * n
            episodes += meta["episodes"]
        info = {"final_obs": np.concatenate(final).copy(), "episodes": episodes,
                "teacher_action": np.concatenate(teach).copy() if self.teacher_actions else None}
        return (np.concatenate(obs).copy(), np.concatenate(rew).copy(), np.concatenate(term).astype(bool),
                np.concatenate(trunc).astype(bool), info)

    def step_wait(self):
        out = self._wait_workers(self.workers)
        self.last_teacher_action = out[4]["teacher_action"]
        return out

    # ---- ワーカーの組ごとに進める(PPO の収集で、片方の組が進む間にもう片方の行動を計算するため) ----
    def worker_groups(self, n_groups: int = 2) -> list[tuple[list[int], slice]]:
        """ワーカーを連続した n_groups 組に分ける。戻り値は [(ワーカーの番号, 環境の範囲)]。"""
        n_groups = max(1, min(n_groups, self.n_workers))
        bounds = np.linspace(0, self.n_workers, n_groups + 1).round().astype(int)
        out = []
        for g in range(n_groups):
            ws = list(range(bounds[g], bounds[g + 1]))
            lo = sum(self.workers[k].n for k in range(bounds[g]))
            hi = lo + sum(self.workers[k].n for k in ws)
            out.append((ws, slice(lo, hi)))
        return out

    def step_async_group(self, worker_ids: list[int], actions: np.ndarray) -> None:
        a = np.asarray(actions, dtype=np.float32).reshape(-1, ACT_DIM)
        o = 0
        for k in worker_ids:
            w = self.workers[k]
            w.send_step(a[o:o + w.n]); o += w.n
        if o != a.shape[0]:
            raise ValueError("step_async_group: 行動の数が組の環境の数と違います")

    def step_wait_group(self, worker_ids: list[int]):
        return self._wait_workers([self.workers[k] for k in worker_ids])

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def worker_pids(self) -> list[int]:
        return [w.proc.pid for w in self.workers]

    def close(self) -> None:
        for w in self.workers:
            w.close()
        self.workers = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
