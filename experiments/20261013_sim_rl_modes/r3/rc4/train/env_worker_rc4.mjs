// 離脱復活 r4(2026-10-14): r3/rc/train/env_worker_rc3.mjs の写し。環境は r3/rc4/train/env_rc4.mjs(run 'recover_r4')。
// モードの学習の環境ワーカー(Node)。ベストラップの rl/env_worker.mjs の写し(2026-10-13、sim_rl_modes)。違い:
//   - 環境は rl/env_modes.mjs の ModeEnv(run = no_wall | recover | cond | obstacle | base)。観測の次元はモードで変わる(79/95/97)
//   - 'S' の返事に「差し込んだ行動か」injected(E 個)を足す(PPO が方策の損失から外す)
//   - 'P' + float64 = 学習の総ステップ数(乱しの強さのカリキュラム)
// やり取り(リトルエンディアン):
//   'I' + uint32 長さ + JSON {n_envs, env_offset, seed, run, mode, episode_mode, time_limit, tuning, starts}
//   'R' + int32 種(−1 = 今の乱数を続ける)  → JSON {starts} + 観測 E×D
//   'S' + float32 × E × 2                 → JSON {episodes} + 観測 E×D・報酬 E・終了 E・打ち切り E・終了前の観測 E×D・injected E
//   'P' + float64                          → 返事なし
//   'Q'
import {ModeEnv} from './env_rc4.mjs';
import * as core from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/fast_core_modes.mjs';
import {createRng, mix32} from '../rt/rng.mjs';
import {FEATURES_VERSION, REFERENCE_LOOP_SHA256} from '../rt/features.mjs';

const log = (...a) => process.stderr.write(a.join(' ') + '\n');
let envs = [], rngs = [], cfg = null, nEnvs = 0, D = 79;
let obsBuf, finalBuf, rewBuf, termBuf, truncBuf, injBuf;

function send(json, arrays = []) {
  const j = Buffer.from(JSON.stringify(json));
  let total = 0;
  for (const a of arrays) total += a.byteLength;
  const out = Buffer.allocUnsafe(8 + j.length + total);
  out.writeUInt32LE(j.length, 0); j.copy(out, 4); out.writeUInt32LE(total, 4 + j.length);
  let off = 8 + j.length;
  for (const a of arrays) { Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(out, off); off += a.byteLength; }
  process.stdout.write(out);
}
function resetEnv(i) {
  const env = envs[i];
  if (cfg.starts === 'random') return {tries: env.resetRandom(rngs[i]), start: env.start};
  const list = cfg.starts, base = list[(cfg.env_offset + i) % list.length];
  for (let k = 0; k < 33; k++) {
    const st = {...base, gate: ((base.gate ?? 0) + k) % 33, seed: base.seed ?? rngs[i].uint32()};
    if (env.reset(rngs[i], st)) return {tries: k, start: env.start};
  }
  throw new Error('env_worker_modes: どのゲートからも開始できません');
}
function writeObs(i) { const o = envs[i].obs; for (let j = 0; j < D; j++) obsBuf[i * D + j] = o[j]; }
async function init(json) {
  cfg = {n_envs: 16, env_offset: 0, seed: 0, run: 'base', mode: 'train', episode_mode: 'continuous', time_limit: 20, tuning: {}, starts: 'random', ...json};
  nEnvs = cfg.n_envs;
  envs = Array.from({length: nEnvs}, (_, i) => new ModeEnv({core, run: cfg.run, mode: cfg.mode, episodeMode: cfg.episode_mode, timeLimit: cfg.time_limit,
    tuning: cfg.tuning, envIndex: cfg.env_offset + i}));
  D = envs[0].dim;
  rngs = envs.map((_, i) => createRng(mix32(cfg.seed, cfg.env_offset + i)));
  obsBuf = new Float32Array(nEnvs * D); finalBuf = new Float32Array(nEnvs * D);
  rewBuf = new Float32Array(nEnvs); termBuf = new Float32Array(nEnvs); truncBuf = new Float32Array(nEnvs); injBuf = new Float32Array(nEnvs);
  send({ok: true, obs_dim: D, act_dim: 2, n_envs: nEnvs, run: cfg.run, train_config: envs[0].cfg, features_version: FEATURES_VERSION,
    reference_loop_sha256: REFERENCE_LOOP_SHA256, pid: process.pid, node: process.version});
}
function resetAll(seed) {
  if (seed >= 0) rngs = envs.map((_, i) => createRng(mix32(seed >>> 0, cfg.env_offset + i)));
  const starts = [];
  for (let i = 0; i < nEnvs; i++) { const r = resetEnv(i); starts.push({...r.start, redraws: r.tries}); writeObs(i); }
  send({starts}, [obsBuf]);
}
function stepAll(actions) {
  const episodes = [];
  finalBuf.fill(0);
  for (let i = 0; i < nEnvs; i++) {
    const env = envs[i], r = env.step(actions[2 * i], actions[2 * i + 1]);
    rewBuf[i] = r.reward; termBuf[i] = r.terminated ? 1 : 0; truncBuf[i] = r.truncated ? 1 : 0; injBuf[i] = r.injected ? 1 : 0;
    if (env.done) {
      const o = env.obs;
      for (let j = 0; j < D; j++) finalBuf[i * D + j] = o[j];
      const summary = env.summary(); summary.env = cfg.env_offset + i;
      episodes.push(summary);
      const rr = resetEnv(i); summary.next_start_redraws = rr.tries;
    }
    writeObs(i);
  }
  send({episodes}, [obsBuf, rewBuf, termBuf, truncBuf, finalBuf, injBuf]);
}
let pending = Buffer.alloc(0), busy = false;
async function pump() {
  if (busy) return;
  busy = true;
  try {
    for (;;) {
      if (pending.length < 1) break;
      const op = String.fromCharCode(pending[0]);
      if (op === 'I') {
        if (pending.length < 5) break;
        const len = pending.readUInt32LE(1);
        if (pending.length < 5 + len) break;
        const json = JSON.parse(pending.subarray(5, 5 + len).toString('utf8'));
        pending = pending.subarray(5 + len);
        await init(json);
      } else if (op === 'R') {
        if (pending.length < 5) break;
        const seed = pending.readInt32LE(1); pending = pending.subarray(5);
        resetAll(seed);
      } else if (op === 'S') {
        const need = 1 + nEnvs * 2 * 4;
        if (pending.length < need) break;
        const actions = new Float32Array(nEnvs * 2);
        Buffer.from(actions.buffer).set(pending.subarray(1, need));
        pending = pending.subarray(need);
        stepAll(actions);
      } else if (op === 'P') {
        if (pending.length < 9) break;
        const steps = pending.readDoubleLE(1); pending = pending.subarray(9);
        for (const e of envs) e.setProgress(steps);
      } else if (op === 'Q') { process.exit(0); }
      else throw new Error('env_worker_modes: 知らない命令 ' + JSON.stringify(op));
    }
  } catch (e) {
    log('env_worker_modes エラー:', e.stack ?? e.message);
    send({error: String(e.message ?? e)});
    process.exit(1);
  } finally { busy = false; }
  if (pending.length && !busy) setImmediate(pump);
}
process.stdin.on('data', chunk => { pending = pending.length ? Buffer.concat([pending, chunk]) : chunk; pump(); });
process.stdin.on('end', () => process.exit(0));
