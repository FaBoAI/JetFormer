#!/usr/bin/env node
// 忠実度の再生(2026-10-16、シミュレーター較正 S3・試験 T3)。実車の窓の指令を、較正版のコア(JS)に流す。
//
//   node replay_fidelity.mjs <入出力のフォルダー>
//
// 入力(fidelity.py の export_windows が書く): header.json と、窓の配列(float64、[窓, 格子])の u_s.f64・u_v.f64・v.f64
//   格子 k = −n_hist−n_warm 〜 n_win(10 ms)。k < −n_warm は遅れのための過去の指令、−n_warm ≤ k < 0 が準備、k = 0 が窓の始まり。
// やり方(protocol fidelity.replay と calib_model.replay の離散の決まりのとおり):
//   - dt = 0.01 秒・障害物の無い広い世界。路面は窓の区間(R1 → shortcut、GM → turf、ほか → carpet)
//   - 過去の指令(k < −n_warm)は prefill、準備の間は速さを実際の値に固定(forceSpeed)して舵の状態だけを流す
//   - k = 0 で速さを実際の値にして、そこから 1 秒(または 2 秒)を物理で流す。向き・位置は k = 0 の姿勢からの相対
// 物理: P_default(元の式・遅れ 0)、P_cfg(元の式・設定欄の値・遅れ 1/30 秒)、P_cal(jf-calib-v1・窓ごとの k_batt)、
//       P_cal_k1(k_batt = 1)、P_cal_brake_low / high(ブレーキの弱め・強め)
// 出力: out_<物理>.f64(v・psi・x・y をこの順に [窓, n_win+1])と out.json
import fs from 'node:fs';
import path from 'node:path';
import {loadCalibratedCore, loadProfile, origReplayProfile, sha256File, RUNTIME_PATH} from './calib_core.mjs';

const dir = process.argv[2];
if (!dir) throw new Error('使い方: node replay_fidelity.mjs <フォルダー>');
const H = JSON.parse(fs.readFileSync(path.join(dir, 'header.json'), 'utf8'));
const {n, n_hist: NH, n_warm: NW, n_win: NWIN} = H;
const NALL = NH + NW + NWIN + 1;
function readArr(name) {
  const b = fs.readFileSync(path.join(dir, H.files[name]));
  if (b.length !== n * NALL * 8) throw new Error(`replay_fidelity: ${name} の大きさが違います`);
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
}
const U_S = readArr('u_s'), U_V = readArr('u_v'), V = readArr('v');

const SIM_MU = {shortcut: .7, turf: .6};       // carpet は既定の路面(μ 0.8)
const BASE_CFG = Object.freeze({dt: .01, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6,
  brakeDeceleration: 1.2, timeLimit: 3600, stopOnCollision: false, seed: 1});
const world = surface => ({segments: [], disks: [], route: [], spawn: {x: 0, z: 0, yaw: 0},
  bounds: {minX: -1e5, maxX: 1e5, minZ: -1e5, maxZ: 1e5},
  surfaces: SIM_MU[surface] ? [{id: surface, x: 0, z: 0, w: 1e4, d: 1e4, mu: SIM_MU[surface]}] : []});

const cal = loadProfile('P_cal'), cfgP = loadProfile('P_cfg');
const PHYS = {
  P_default: {profile: origReplayProfile(0, 'P_default'), cfg: {}, kb: false},
  P_cfg: {profile: origReplayProfile(cfgP.profile.actuator_delay_frames / 30, 'P_cfg'),
    cfg: Object.fromEntries(['driveTau', 'driveAccelerationCap', 'brakeDeceleration', 'steerRate', 'maxSteer', 'frictionMultiplier'].map(k => [k, cfgP.profile.sim_config[k]])), kb: false},
  P_cal: {profile: cal.profile, sha: cal.sha256, cfg: {}, kb: true},
  P_cal_k1: {profile: cal.profile, sha: cal.sha256, cfg: {}, kb: false},
  P_cal_brake_low: {profile: cal.profile, sha: cal.sha256, cfg: {}, kb: true, brakeScenario: 'low'},
  P_cal_brake_high: {profile: cal.profile, sha: cal.sha256, cfg: {}, kb: true, brakeScenario: 'high'},
};
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const meta = {schema: 'jf-calib-fidelity-replay/v1', created_at: new Date().toISOString(), n, n_win: NWIN, runtime_sha256: sha256File(RUNTIME_PATH),
  physics: {}};
for (const name of H.physics) {
  const P = PHYS[name];
  if (!P) throw new Error('replay_fidelity: 知らない物理です: ' + name);
  const t0 = performance.now();
  const core = await loadCalibratedCore(P.profile, {fast: true, profileSha256: P.sha ?? null, brakeScenario: P.brakeScenario ?? 'nominal'});
  const sims = {};
  const out = new Float64Array(4 * n * (NWIN + 1));
  const OV = 0, OP = n * (NWIN + 1), OX = 2 * n * (NWIN + 1), OY = 3 * n * (NWIN + 1);
  for (let i = 0; i < n; i++) {
    const surf = H.surface[i];
    if (!sims[surf]) sims[surf] = core.createSimulation(world(surf), {...BASE_CFG, ...P.cfg});
    const sim = sims[surf];
    sim.reset();
    const base = i * NALL;
    const prefill = [];
    for (let k = 0; k < NH; k++) prefill.push({steering: U_S[base + k], speed: U_V[base + k]});
    sim.calibControl({prefill, kBatt: P.kb ? H.k_batt[i] : 1, forceSpeed: V[base + NH]});
    let x0 = 0, z0 = 0, yaw0 = 0, prevYaw = 0, psi = 0;
    for (let f = 0; f < NW + NWIN; f++) {
      const a = base + NH + f;
      if (f < NW) sim.calibControl({forceSpeed: V[a]});
      if (f === NW) {
        sim.calibControl({forceSpeed: null, setSpeed: V[a]});
        const s0 = sim.peek().state;
        x0 = s0.x; z0 = s0.z; yaw0 = s0.yaw; prevYaw = yaw0; psi = 0;
        const o = i * (NWIN + 1);
        out[OV + o] = s0.speed; out[OP + o] = 0; out[OX + o] = 0; out[OY + o] = 0;
      }
      sim.step({steering: U_S[a], speed: U_V[a]}, true);
      if (f >= NW) {
        const s = sim.peek().state;
        psi += wrap(s.yaw - prevYaw); prevYaw = s.yaw;
        const dx = s.x - x0, dz = s.z - z0, c = Math.cos(yaw0), sn = Math.sin(yaw0);
        const o = i * (NWIN + 1) + (f - NW + 1);
        out[OV + o] = s.speed; out[OP + o] = psi;
        out[OX + o] = dx * c - dz * sn;          // 始まりの向きの前
        out[OY + o] = -dx * sn - dz * c;         // 始まりの向きの左
      }
    }
  }
  fs.writeFileSync(path.join(dir, `out_${name}.f64`), Buffer.from(out.buffer));
  const info = Object.values(sims)[0]?.report().model.calibration ?? null;
  meta.physics[name] = {seconds: (performance.now() - t0) / 1000, calibration: info, config_override: P.cfg, k_batt_per_window: P.kb};
  console.log(name, n, 'windows', meta.physics[name].seconds.toFixed(1), 's');
}
fs.writeFileSync(path.join(dir, 'out.json'), JSON.stringify(meta, null, 1));
