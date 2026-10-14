// 障害物よけ(obstacle)の確認 r2 の回の一覧(2026-10-14、確認の段 r2)。
//   SELREPLAY : 学習の段の選びの組 OB-V100-R2(100 回 + OB-EMPTY + EMPTY-VAR6)。開始・種・配置 ID は学習の段の結果の行
//               (train/obstacle/r2/final/selected_selection_original.json)から「データとして」読む。走らせ方・すき間・判定は自前
//   VREPLAY   : protocol の検証の組(13 配置 × 2 + OB-EMPTY)。開始・種は r2b/main/evals/obstacle/step_008011776.json の行から
//   FRESH     : 新しい配置 150(種 4,130,000〜、試験の組と同じ作り方・STD の前後に置かない)× 2 開始(STD と、記録・試験の開始の分布
//               (ゲート U{0..32}・±0.15 m・±10°)の自前の一様乱数で「開始の前後 1.5 m に障害物が無い・開始で重ならない」もの)。種 4,200,000〜
//   FRESH_FAST: FRESH と同じ配置・開始で最速の方策(OB-BASE と同じ考え)
//   EMPTY     : 障害物なしの単周: STD・±3 cm/±3° の 6 つ・±3 cm/±3° の中の自前の乱数 24 個。障害物よけ・最速の両方(O4 とその頑健さ)
//   NOOB      : 障害物なしの連続周回、新しい開始 20 を障害物よけ・最速の両方で
//   REC       : 記録の組 30 配置 × 自前の開始 3(記録の決まりの分布と 1.5 m の検査)。記録のプランの前もっての走り(prefilter)を集計でまねる
import fs from 'node:fs';
import {startClearance, projectGlobal, activeWorld, BASE_WORLD} from './run_ob2.mjs';
import {LOOP_L_OWN} from '../check_ln/own_extra.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const OUT = `${REP}/check_obstacle_r2`;
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
const modL = s => ((s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN;
const dS = (a, b) => { let d = modL(a - b); if (d > LOOP_L_OWN / 2) d -= LOOP_L_OWN; return d; };
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
export function loadFresh() { return JSON.parse(fs.readFileSync(`${OUT}/fresh2_layouts.json`, 'utf8')).layouts; }
const pick = st => ({gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: st.seed});

export function buildSet(name) {
  const J = [];
  if (name === 'SELREPLAY') {
    const ev = JSON.parse(fs.readFileSync(`${REP}/train/obstacle/r2/final/selected_selection_original.json`, 'utf8'));
    const lays = Object.fromEntries(JSON.parse(fs.readFileSync(`${EXP}/rl/ob2/selection_layouts.json`, 'utf8')).layouts.map(l => [l.id, l]));
    for (const r of ev.rows) {
      const single = r.test !== 'OB-V100-R2';
      J.push({id: r.id, test: r.test, spec: {policy: 'obstacle', start: pick(r.start), layout: r.layout_id ? lays[r.layout_id] : null, single}});
    }
  } else if (name === 'VREPLAY') {
    const ev = JSON.parse(fs.readFileSync(`${REP}/train/obstacle/r2/r2b/main/evals/obstacle/step_008011776.json`, 'utf8'));
    const lays = Object.fromEntries(JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/validation.json`, 'utf8')).layouts.map(l => [l.id, l]));
    for (const r of ev.tests['OB-T100']) J.push({id: r.id, test: 'VREPLAY', spec: {policy: 'obstacle', start: pick(r.start), layout: lays[r.layout_id]}});
    for (const r of ev.tests['OB-EMPTY']) J.push({id: 'EMPTY', test: 'VREPLAY_EMPTY', spec: {policy: 'obstacle', start: pick(r.start), layout: null, single: true}});
  } else if (name === 'FRESH' || name === 'FRESH_FAST' || name === 'FRESH_R1') {
    const lays = loadFresh(), rnd = mulberry32(0x0b57ac2e), pol = {FRESH: 'obstacle', FRESH_FAST: 'fastest', FRESH_R1: 'obstacle_r1'}[name];
    lays.forEach((lay, i) => {
      const h = randomStart(rnd, lay.obstacles);
      if (name === 'FRESH_R1' && i >= 50) return;
      J.push({id: `${lay.id}-STD`, test: name, spec: {policy: pol, start: {...STD, seed: 4200000 + 2 * i}, layout: lay, keepPath: i < 12}});
      J.push({id: `${lay.id}-R`, test: name, spec: {policy: pol, start: {gate: h.gate, left_offset_m: h.left_offset_m, left_yaw_deg: h.left_yaw_deg, seed: 4200000 + 2 * i + 1},
        layout: lay, keepPath: i < 12}});
    });
  } else if (name === 'EMPTY') {
    const rnd = mulberry32(0x0e4e4e4e), rv = [];
    for (let k = 0; k < 24; k++) rv.push([-.03 + .06 * rnd(), -3 + 6 * rnd()]);
    for (const pol of ['obstacle', 'fastest']) {
      J.push({id: `${pol}-STD`, test: 'EMPTY', spec: {policy: pol, start: {...STD, seed: 1}, layout: null, single: true}});
      let k = 0;
      for (const [off, yaw] of [[.03, 0], [-.03, 0], [0, 3], [0, -3], [.03, -3], [-.03, 3]])
        J.push({id: `${pol}-V${k++}`, test: 'EMPTY_V', spec: {policy: pol, start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 4300000 + k}, layout: null, single: true}});
      rv.forEach(([off, yaw], j) =>
        J.push({id: `${pol}-RV${j}`, test: 'EMPTY_RV', spec: {policy: pol, start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 4310000 + j}, layout: null, single: true}}));
    }
  } else if (name === 'NOOB') {
    const rnd = mulberry32(0x5eed0b0c);
    for (let i = 0; i < 20; i++) {
      const h = randomStart(rnd, []);
      for (const pol of ['obstacle', 'fastest'])
        J.push({id: `N${i}-${pol}`, test: 'NOOB', spec: {policy: pol, start: {gate: h.gate, left_offset_m: h.left_offset_m, left_yaw_deg: h.left_yaw_deg, seed: 4400000 + i}, layout: null, keepPath: i < 6}});
    }
  } else if (name === 'REC') {
    const lays = JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/recording.json`, 'utf8')).layouts, rnd = mulberry32(0x7ec0ade6);
    lays.forEach((lay, i) => {
      for (let k = 0; k < 3; k++) {
        const h = randomStart(rnd, lay.obstacles);
        J.push({id: `${lay.id}-T${k}`, test: 'REC', spec: {policy: 'obstacle', start: {gate: h.gate, left_offset_m: h.left_offset_m, left_yaw_deg: h.left_yaw_deg, seed: 4500000 + 3 * i + k},
          layout: lay, keepPath: true}});
      }
    });
  } else throw new Error('知らない組: ' + name);
  return J;
}
