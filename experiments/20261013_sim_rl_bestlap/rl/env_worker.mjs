#!/usr/bin/env node
// RL の環境ワーカー(Node)。1 プロセスで E 個の環境を持ち、Python(rl/vec_env.py)と標準入出力のバイナリでやり取りする(2026-10-13)
//
//   node rl/env_worker.mjs        (起動は vec_env.py が行う。標準出力はやり取り専用。ログは標準エラーへ)
//
// やり取り(すべてリトルエンディアン)
//   Python → ワーカー: 1 バイトの命令 + 中身
//     'I' + uint32 長さ + JSON   初期化 {n_envs, env_offset, seed, variant, mode, episode_mode, core, teacher_actions, starts, dr, reward}
//                                (reward は r3 の追加実験だけ: 'r3' ですき間の約束つきの報酬。null・'default' なら今までと同じ)
//     'R' + int32 種(−1 = 今の乱数を続ける)   全環境をリセット
//     'S' + float32 × E × 2       全環境を 1 フレーム進める(行動 a ∈ [−1, 1]²。範囲外は切る)
//     'Q'                         終わる
//   ワーカー → Python: uint32 JSON の長さ + JSON + uint32 バイナリの長さ + バイナリ(float32)
//     'I' の返事: JSON {ok, obs_dim, ...}、バイナリなし
//     'R' の返事: JSON {starts}、バイナリ = 観測 E×79 [+ 教師の行動 E×2]
//     'S' の返事: JSON {episodes: [終わったエピソードの要約…]}、
//                 バイナリ = 観測 E×79・報酬 E・終了 E・打ち切り E・終了前の最後の観測 E×79 [+ 教師の行動 E×2]
//   自動リセット: 終わった環境は、終了前の最後の観測を「最後の観測」の欄に入れ、観測の欄にはリセット後の観測を入れる。
//   終わっていない環境の「最後の観測」の欄は 0。
// 開始: starts が 'random' なら protocol の分布(半分ゲート 0・半分ランダムなゲート、±0.12 m・±8°)。
//       配列なら環境 i は starts[i % 長さ] を毎回使う(評価・試験用)。invalid_spawn は引き直す(配列なら次のゲートへ)。
// 乱数: 環境 i(通し番号 env_offset + i)の乱数は mix32(種, 通し番号) で作る。ワーカーの分け方によらず同じ結果になる。
import {RlEnv, loadCore, loadRcTeacher, loadBaseWorld, createRng, mix32, resetRandom, REWARDS, rewardVersionOf, DR_SPECS, sampleDr} from './env_core.mjs';
import {OBS_DIM, FEATURES_VERSION, REFERENCE_LOOP_SHA256} from './features.mjs';

const log = (...a) => process.stderr.write(a.join(' ') + '\n');
let envs = [], rngs = [], cfg = null, nEnvs = 0, teacherOn = false, drSpec = null;
let obsBuf, finalBuf, rewBuf, termBuf, truncBuf, teachBuf;

function send(json, floatArrays = []) {
  const j = Buffer.from(JSON.stringify(json));
  let total = 0;
  for (const a of floatArrays) total += a.byteLength;
  const out = Buffer.allocUnsafe(8 + j.length + total);
  out.writeUInt32LE(j.length, 0); j.copy(out, 4);
  out.writeUInt32LE(total, 4 + j.length);
  let off = 8 + j.length;
  for (const a of floatArrays) { Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(out, off); off += a.byteLength; }
  process.stdout.write(out);
}

function resetEnv(i) {
  const env = envs[i];
  // r2(参考の追加実験): dr が名前なら、回ごとに物理のずれを引く(DR_SPECS)。null なら今までと同じ
  if (cfg.starts === 'random' && drSpec) { const perturb = sampleDr(rngs[i], drSpec); return {tries: resetRandom(env, rngs[i], undefined, 100, {perturb}), start: env.start}; }
  if (cfg.starts === 'random') return {tries: resetRandom(env, rngs[i]), start: env.start};
  const list = cfg.starts, base = list[(cfg.env_offset + i) % list.length];
  for (let k = 0; k < 33; k++) {
    // 開始が障害物と重なるときは次の番号のゲートにする(protocol ROB24_invalid_spawn_rule と同じ規則)
    const st = {...base, gate: ((base.gate ?? 0) + k) % 33, seed: base.seed ?? rngs[i].uint32()};
    if (env.reset(st)) return {tries: k, start: env.start};
  }
  throw new Error('env_worker: どのゲートからも開始できません');
}
function writeObs(i) {
  const o = envs[i].obs;
  for (let j = 0; j < OBS_DIM; j++) obsBuf[i * OBS_DIM + j] = o[j];
  if (teacherOn) { teachBuf[2 * i] = envs[i].teacherAction[0]; teachBuf[2 * i + 1] = envs[i].teacherAction[1]; }
}

async function init(json) {
  cfg = {n_envs: 16, env_offset: 0, seed: 0, variant: 'main', mode: 'train', episode_mode: 'continuous', core: 'fast',
    teacher_actions: false, starts: 'random', time_limit: 20, dr: null, reward: null, ...json};
  nEnvs = cfg.n_envs;
  const reward = REWARDS[cfg.reward ?? 'default'];
  if (!reward) throw new Error('env_worker: 知らない reward です: ' + cfg.reward);
  if (cfg.dr && !DR_SPECS[cfg.dr]) throw new Error('env_worker: 知らない dr です: ' + cfg.dr);
  drSpec = cfg.dr ? DR_SPECS[cfg.dr] : null;
  const core = await loadCore(cfg.core);
  const teacherModule = cfg.teacher_actions ? await loadRcTeacher() : null;
  teacherOn = !!cfg.teacher_actions;
  const baseWorld = loadBaseWorld();
  envs = Array.from({length: nEnvs}, () => new RlEnv({core, coreKind: cfg.core, baseWorld, variant: cfg.variant, mode: cfg.mode,
    episodeMode: cfg.episode_mode, timeLimit: cfg.time_limit, teacherModule, teacherActions: teacherOn, reward}));
  rngs = envs.map((_, i) => createRng(mix32(cfg.seed, cfg.env_offset + i)));
  obsBuf = new Float32Array(nEnvs * OBS_DIM); finalBuf = new Float32Array(nEnvs * OBS_DIM);
  rewBuf = new Float32Array(nEnvs); termBuf = new Float32Array(nEnvs); truncBuf = new Float32Array(nEnvs);
  teachBuf = new Float32Array(nEnvs * 2);
  send({ok: true, obs_dim: OBS_DIM, act_dim: 2, n_envs: nEnvs, features_version: FEATURES_VERSION, reference_loop_sha256: REFERENCE_LOOP_SHA256,
    speed_cap_mps: envs[0].speedCap, core: cfg.core, pid: process.pid, node: process.version, reward_version: rewardVersionOf(reward), reward,
    dr: cfg.dr, dr_spec: drSpec});
}

function resetAll(seed) {
  if (seed >= 0) rngs = envs.map((_, i) => createRng(mix32(seed >>> 0, cfg.env_offset + i)));
  const starts = [];
  for (let i = 0; i < nEnvs; i++) { const r = resetEnv(i); starts.push({...r.start, redraws: r.tries}); writeObs(i); }
  send({starts}, teacherOn ? [obsBuf, teachBuf] : [obsBuf]);
}

function stepAll(actions) {
  const episodes = [];
  finalBuf.fill(0);
  for (let i = 0; i < nEnvs; i++) {
    const env = envs[i], r = env.step(actions[2 * i], actions[2 * i + 1]);
    rewBuf[i] = r.reward; termBuf[i] = r.terminated ? 1 : 0; truncBuf[i] = r.truncated ? 1 : 0;
    if (env.done) {
      const o = env.obs;
      for (let j = 0; j < OBS_DIM; j++) finalBuf[i * OBS_DIM + j] = o[j];
      const summary = env.summary(); summary.env = cfg.env_offset + i;
      delete summary.collision_events;
      episodes.push(summary);
      const rr = resetEnv(i); summary.next_start_redraws = rr.tries;
    }
    writeObs(i);
  }
  const arrays = [obsBuf, rewBuf, termBuf, truncBuf, finalBuf];
  if (teacherOn) arrays.push(teachBuf);
  send({episodes}, arrays);
}

// ---- 標準入力の読み取り(命令は 1 つずつ来る。途中で切れた塊もつなぐ) ----
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
      } else if (op === 'Q') { process.exit(0); }
      else throw new Error('env_worker: 知らない命令 ' + JSON.stringify(op));
    }
  } catch (e) {
    log('env_worker エラー:', e.stack ?? e.message);
    send({error: String(e.message ?? e)});
    process.exit(1);
  } finally { busy = false; }
  if (pending.length && !busy) setImmediate(pump);
}
process.stdin.on('data', chunk => { pending = pending.length ? Buffer.concat([pending, chunk]) : chunk; pump(); });
process.stdin.on('end', () => process.exit(0));
