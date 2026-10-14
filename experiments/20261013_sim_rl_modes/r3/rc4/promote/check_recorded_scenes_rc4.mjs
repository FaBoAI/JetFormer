#!/usr/bin/env node
// 離脱復活 r4(2026-10-14、学習の段 recover r4): 記録したブラウザのシーンそのもので、止まり・進み・ゲートの外しを確かめ直す(昇格・記録の段の手当て)。
// 確認 recover r3 の指摘: Firefox と Node は閉ループで 50〜108 フレーム目から分かれる(Math の最後の桁)。Node の予想(predict_scenes)で
// 「有効・外しなし」と選んだ開始 42 のうち 3 がブラウザの記録ではゲートの外しあり。→ 取り込む前に、記録したシーンの frames.jsonl と
// evaluation-report.json だけから同じ決まりで数え直し、外れたシーンは取り込まない。
//
//   node r3/rc4/promote/check_recorded_scenes_rc4.mjs <シーンのフォルダー(frames.jsonl のある所)か、その親(simulation/scenes)・データセットの根> … [--out 結果.json]
//
// 決まり(protocol_recover_r4.json definitions_added・protocol_recover_r3.json definitions_added と同じ):
//   - 有効(valid_episode): evaluation-report の終わりが collection_duration_complete・600 フレーム・衝突 0
//   - 止まり: フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上続く。ブレーキの乱し(info.jf.ph = 'perturb' で指令の速さ 0)の区間とそのあと 30 フレームは数えない
//   - 進み: R(reference_loop.json)への射影の s が 20 秒で 26.12 m 以上
//   - ゲートの外し(gate_miss_real): sim のゲートの数えが変わらないまま s が 2.019 m より先へ進んだ。フレームごとのゲートの数えは記録に無いので、
//     軌跡(state_post_action の位置)と sim-core と同じ決まり(次のゲートの面を越えた点の横ずれ ≤ gateHalfWidth)で数え直し、
//     数え直した passed_gates が evaluation-report の metrics.passed_gates と同じことを確かめる(違えば「確かめられない」= 取り込まない)
// 読むだけ(どのファイルも変えない)。
import fs from 'node:fs';
import path from 'node:path';
import {CANONICAL_ROUTE} from '../rt/features.mjs';
import {realGateMisses} from '../gate_miss_rc4.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const REF = JSON.parse(fs.readFileSync(path.join(here, '../rt/reference_loop.json'), 'utf8'));
const RX = REF.x, RZ = REF.z, RS = REF.s, RN = REF.n, LOOP = REF.length_m;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const SEGL = Array.from({length: RN}, (_, i) => i < RN - 1 ? RS[i + 1] - RS[i] : LOOP - RS[RN - 1]);
function projSeg(i, x, z) {
  const j = (i + 1) % RN, dx = RX[j] - RX[i], dz = RZ[j] - RZ[i], l2 = dx * dx + dz * dz;
  const u = Math.max(0, Math.min(1, ((x - RX[i]) * dx + (z - RZ[i]) * dz) / l2));
  const px = RX[i] + dx * u, pz = RZ[i] + dz * u;
  return {d2: (x - px) ** 2 + (z - pz) ** 2, sLocal: RS[i] + u * SEGL[i]};
}
const idxOf = sl => { let lo = 0, hi = RN - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (RS[m] <= sl) lo = m; else hi = m - 1; } return lo; };
// 前のフレームの s(周をまたいだ値)の −0.6〜+1.6 m を探す(確認の段の自前の射影と同じ)
export function project(x, z, prevS = null) {
  let best = null, bestLap = 0;
  if (prevS == null) { for (let i = 0; i < RN; i++) { const r = projSeg(i, x, z); if (!best || r.d2 < best.d2) best = r; } }
  else {
    const a = prevS - .6, b = prevS + 1.6;
    let lap = Math.floor(a / LOOP), i = idxOf(a - lap * LOOP);
    for (let g = 0; g < RN + 2; g++) {
      if (lap * LOOP + RS[i] > b) break;
      const r = projSeg(i, x, z);
      if (!best || r.d2 < best.d2) { best = r; bestLap = lap; }
      i++; if (i === RN) { i = 0; lap++; }
    }
  }
  return best.sLocal + bestLap * LOOP;
}
const readJsonl = p => fs.readFileSync(p, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));

export function checkScene(dir) {
  const out = {scene: path.basename(dir), dir};
  const rows = readJsonl(path.join(dir, 'frames.jsonl')).sort((a, b) => a.frame - b.frame);
  const rep = JSON.parse(fs.readFileSync(path.join(dir, 'evaluation-report.json'), 'utf8'));
  const cfg = rep.config ?? {}, m = rep.metrics ?? {}, fin = rep.final_state ?? rows.at(-1)?.state_post_action ?? {};
  const n = rows.length;
  out.frames = n; out.reason = fin.reason ?? null; out.collisions = m.collisions ?? null;
  out.valid_episode = fin.reason === 'collection_duration_complete' && n === 600 && m.collisions === 0;
  // 位置の列: p[0] = 開始(フレーム 0 の操作の前)、p[f + 1] = フレーム f の操作のあと
  const P = [rows[0].pose_pre_action, ...rows.map(r => r.state_post_action)];
  // 開始のゲート: 開始の位置にいちばん近い経路の点(app.js の reset は route[gate] から左右 0.3 m 以内に置く)
  let k = 0, bd = Infinity;
  CANONICAL_ROUTE.forEach((q, i) => { const d = Math.hypot(q[0] - P[0].x, q[1] - P[0].z); if (d < bd) { bd = d; k = i; } });
  const route = [...CANONICAL_ROUTE.slice(k), ...CANONICAL_ROUTE.slice(0, k)], targets = [...route.slice(1), route[0]];
  out.start_gate = k; out.start_route_distance_m = +bd.toFixed(3);
  const half = cfg.gateHalfWidth ?? .65;
  let idx = 0, passed = 0;
  const gArr = [0], sArr = [];
  let prevS = null;
  for (let f = 0; f < P.length; f++) {
    const s = project(P[f].x, P[f].z, prevS); prevS = s; sArr.push(s);
    if (f === 0) continue;
    const a0 = P[f - 1], b0 = P[f];
    for (let guard = 0; guard < 4; guard++) {   // sim-core の advanceGates と同じ決まり(1 フレームの弦で。ゲートの間は 0.5 m 以上なので 1 フレームに多くて 1〜2 個)
      const target = targets[idx], before = idx ? targets[idx - 1] : route[0];
      const len = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / len, nz = (target[1] - before[1]) / len;
      const sg = p => (p.x - target[0]) * nx + (p.z - target[1]) * nz;
      const a = sg(a0), b = sg(b0);
      if (!(a < 0 && b >= 0)) break;
      const fr = -a / (b - a), x = a0.x + (b0.x - a0.x) * fr, z = a0.z + (b0.z - a0.z) * fr;
      if (Math.abs(-(x - target[0]) * nz + (z - target[1]) * nx) > half) break;
      passed++; idx = (idx + 1) % targets.length;
    }
    gArr.push(passed);
  }
  out.passed_gates_recount = passed; out.passed_gates_report = m.passed_gates ?? null;
  // 1 フレームの弦で数えるので、終わりの数フレームのゲートは sim(小区間で数える・時間で打ち切る)と 1 つずれることがある(既存の記録 192 シーンで 3)。
  // 数えの差がそれだけで説明できる(報告の値が「5 フレーム前までの数え」〜「数え直し + 1」の中)なら同じとみなす
  out.gate_count_consistent = m.passed_gates === passed
    || (Math.abs(m.passed_gates - passed) === 1 && m.passed_gates >= gArr[Math.max(0, gArr.length - 6)] && m.passed_gates <= passed + 1);
  // フレーム f の操作の前の数え = gArr[f]、s = sArr[f](f = 0〜599)
  const gm = realGateMisses(sArr.slice(0, n), gArr.slice(0, n));
  out.gate_miss = gm.misses > 0; out.gate_misses = gm.misses; out.gate_stuck_max_m = +gm.stuck_max_m.toFixed(3);
  out.progress_m = +(sArr[n] - sArr[0]).toFixed(3);
  out.progress_ok = sArr[n] - sArr[0] >= LOOP - 1e-9;
  // 止まり(ブレーキの乱しの区間とそのあと 30 フレームは数えない)
  const excl = new Uint8Array(n), brakeEv = new Set();
  rows.forEach((r, f) => { const jf = r.action_info?.jf; if (jf?.ph === 'perturb' && r.command_si?.speed === 0) brakeEv.add(jf.ev); });
  let evStart = null, evEnd = null, curEv = null;
  const mark = (a, b) => { for (let f = a; f < Math.min(n, b + 30); f++) excl[f] = 1; };
  rows.forEach((r, f) => {
    const jf = r.action_info?.jf, e = jf?.ph === 'perturb' && brakeEv.has(jf.ev) ? jf.ev : null;
    if (e !== curEv) { if (curEv != null) mark(evStart, f); curEv = e; evStart = f; }
  });
  if (curEv != null) mark(evStart, n);
  let run = 0, maxRun = 0;
  for (let f = 30; f < n; f++) {
    if (excl[f]) continue;
    const v = Math.abs(P[f].speed ?? 0);   // フレーム f の操作の前の速さ(= フレーム f − 1 の操作のあと)
    if (v < .2) { run++; maxRun = Math.max(maxRun, run); } else run = 0;
  }
  out.stall = maxRun >= 30; out.stall_max_run = maxRun;
  out.perturb_frames = rows.filter(r => r.action_info?.jf?.ph === 'perturb').length;
  out.shield_frames = rows.filter(r => r.action_info?.jf?.sh === 1).length;
  out.mode = rows[0].action_info?.jf?.m ?? null;
  out.valid_episode_r3 = out.valid_episode && !out.stall && out.progress_ok;
  out.accept = out.valid_episode_r3 && !out.gate_miss && out.gate_count_consistent;
  out.reject_reasons = [!out.valid_episode && 'valid_episode', out.stall && 'stall', !out.progress_ok && 'progress', out.gate_miss && 'gate_miss',
    !out.gate_count_consistent && 'gate_recount_mismatch'].filter(Boolean);
  return out;
}

function sceneDirs(p) {
  if (fs.existsSync(path.join(p, 'frames.jsonl'))) return [p];
  for (const sub of ['simulation/scenes', 'scenes']) {
    const d = path.join(p, sub);
    if (fs.existsSync(d)) return fs.readdirSync(d).map(x => path.join(d, x)).filter(x => fs.existsSync(path.join(x, 'frames.jsonl'))).sort();
  }
  if (fs.existsSync(path.join(p, 'simulation', 'frames.jsonl'))) return [path.join(p, 'simulation')];   // 1 シーンの ZIP の形
  return [];
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const args = process.argv.slice(2), outIdx = args.indexOf('--out');
  const outPath = outIdx >= 0 ? args[outIdx + 1] : null;
  const inputs = args.filter((a, i) => a !== '--out' && args[i - 1] !== '--out');
  const res = [];
  for (const p of inputs) for (const d of sceneDirs(p)) { try { res.push(checkScene(d)); } catch (e) { res.push({dir: d, error: String(e?.message ?? e), accept: false}); } }
  const summary = {schema: 'jf-sim-rl-recorded-scene-check/v1', created_at: new Date().toISOString(), scenes: res.length, accept: res.filter(r => r.accept).length,
    rule: 'valid_episode_r3(止まり・進み)かつ gate_miss_real 0 かつ ゲートの数え直しが evaluation-report と同じ', results: res};
  const txt = JSON.stringify(summary, null, 1);
  if (outPath) fs.writeFileSync(outPath, txt);
  console.log(JSON.stringify({scenes: summary.scenes, accept: summary.accept, rejected: res.filter(r => !r.accept).map(r => ({scene: r.scene ?? r.dir, why: r.reject_reasons ?? r.error}))}));
}
