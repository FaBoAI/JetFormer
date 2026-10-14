// 障害物よけ(obstacle)の確認 r1 の回の一覧(2026-10-14、確認の段)。
//   VREPLAY   : 検証の組(13 配置 × 2 開始 + 障害物なし = 27 回)。開始・種は学習の段の結果の行(main/evals/obstacle/step_002015232.json)から
//               「データとして」読む(同じ条件でないと比べられないため)。配置は rl/obstacle_sets/validation.json。走らせ方・すき間・判定は自前
//   FRESH     : 新しい配置 50(layouts.mjs の作り方 = 試験の組と同じ分布、種 1,930,000〜。学習・検証・試験・記録の種と重ならない。
//               STD の前後 1.8 m に置かない = 試験の組と同じ)× 2 開始(STD と、記録の開始の分布(ゲート U{0..32}・±0.15 m・±10°)の自前の一様乱数で
//               「開始の前後 1.5 m に障害物が無い・開始で重ならない」もの)。種 2,600,000〜。試験の組(OB-T100)の作り方を、新しい種で
//   FRESH_FAST: FRESH と同じ配置・開始で最速の方策(OB-BASE と同じ考え。報告と「最速と違うか」)
//   EMPTY     : 障害物なしの単周(STD と、±3 cm・±3° の 6 つの小さなずれ)を障害物よけ・最速の両方で(O4 とその頑健さ)
//   NOOB      : 障害物なしの連続周回、新しい開始 20(記録の開始の分布)を障害物よけ・最速の両方で(障害物の無いときの走りの違い)
//   REC       : 記録の組 30 配置 × 自前の開始 1(記録の決まり)を障害物よけで(学習の段の「記録のプランで 0/90」の確かめ)
import fs from 'node:fs';
import {startClearance, projectGlobal, activeWorld, BASE_WORLD} from './run_ob.mjs';
import {LOOP_L_OWN} from '../check_ln/own_extra.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const OUT = `${REP}/check_obstacle_r1`;
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
const modL = s => ((s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN;
const dS = (a, b) => { let d = modL(a - b); if (d > LOOP_L_OWN / 2) d -= LOOP_L_OWN; return d; };
// 自前の開始の条件: 開始の位置の R 上の s から前後 1.5 m に障害物(の中心の s)が無い・開始で重ならない
export function startOk(st, obstacles) {
  const {world} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const s = projectGlobal(world.spawn.x, world.spawn.z).s;
  if (!(obstacles ?? []).every(o => Math.abs(dS(projectGlobal(o.x, o.z).s, s)) >= 1.5)) return {ok: false};
  const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg, obstacles);
  return {ok: c > 1e-9, clearance: c, s};
}
export function randomStart(rnd, obstacles, dist = {off: .15, yaw: 10}) {
  for (let k = 0; k < 1000; k++) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -dist.off + 2 * dist.off * rnd(), left_yaw_deg: -dist.yaw + 2 * dist.yaw * rnd()};
    const r = startOk(st, obstacles);
    if (r.ok) return {...st, start_clearance: r.clearance};
  }
  throw new Error('開始が見つかりません');
}
export function loadFresh() { return JSON.parse(fs.readFileSync(`${OUT}/fresh_layouts.json`, 'utf8')).layouts; }

export function buildSet(name) {
  const J = [];
  if (name === 'VREPLAY') {
    const ev = JSON.parse(fs.readFileSync(`${REP}/train/obstacle/main/evals/obstacle/step_002015232.json`, 'utf8'));
    const lays = Object.fromEntries(JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/validation.json`, 'utf8')).layouts.map(l => [l.id, l]));
    for (const r of ev.tests['OB-T100']) {
      const st = {gate: r.start.gate, left_offset_m: r.start.left_offset_m, left_yaw_deg: r.start.left_yaw_deg, seed: r.start.seed};
      J.push({id: r.id, test: 'VREPLAY', spec: {policy: 'obstacle', start: st, layout: lays[r.layout_id]}});
    }
    J.push({id: 'EMPTY', test: 'VREPLAY_EMPTY', spec: {policy: 'obstacle', start: {...STD, seed: 1}, layout: null, single: true}});
  } else if (name === 'FRESH' || name === 'FRESH_FAST') {
    const lays = loadFresh(), rnd = mulberry32(0x0b57ac1e), pol = name === 'FRESH' ? 'obstacle' : 'fastest';
    lays.forEach((lay, i) => {
      const h = randomStart(rnd, lay.obstacles);
      J.push({id: `${lay.id}-STD`, test: name, spec: {policy: pol, start: {...STD, seed: 2600000 + 2 * i}, layout: lay, keepPath: i < 12}});
      J.push({id: `${lay.id}-R`, test: name, spec: {policy: pol, start: {gate: h.gate, left_offset_m: h.left_offset_m, left_yaw_deg: h.left_yaw_deg, seed: 2600000 + 2 * i + 1},
        layout: lay, keepPath: i < 12}});
    });
  } else if (name === 'EMPTY') {
    for (const pol of ['obstacle', 'fastest']) {
      J.push({id: `${pol}-STD`, test: 'EMPTY', spec: {policy: pol, start: {...STD, seed: 1}, layout: null, single: true}});
      let k = 0;
      for (const [off, yaw] of [[.03, 0], [-.03, 0], [0, 3], [0, -3], [.03, -3], [-.03, 3]])
        J.push({id: `${pol}-V${k++}`, test: 'EMPTY_V', spec: {policy: pol, start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 2700000 + k}, layout: null, single: true}});
    }
  } else if (name === 'NOOB') {
    const rnd = mulberry32(0x5eed0b0b);
    for (let i = 0; i < 20; i++) {
      const h = randomStart(rnd, []);
      for (const pol of ['obstacle', 'fastest'])
        J.push({id: `N${i}-${pol}`, test: 'NOOB', spec: {policy: pol, start: {gate: h.gate, left_offset_m: h.left_offset_m, left_yaw_deg: h.left_yaw_deg, seed: 2800000 + i}, layout: null, keepPath: i < 6}});
    }
  } else if (name === 'REC') {
    const lays = JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/recording.json`, 'utf8')).layouts, rnd = mulberry32(0x7ec0ade5);
    lays.forEach((lay, i) => {
      const h = randomStart(rnd, lay.obstacles);
      J.push({id: `${lay.id}-R`, test: 'REC', spec: {policy: 'obstacle', start: {gate: h.gate, left_offset_m: h.left_offset_m, left_yaw_deg: h.left_yaw_deg, seed: 2900000 + i}, layout: lay}});
    });
  } else throw new Error('知らない組: ' + name);
  return J;
}
