// 乱し(離脱復活)の予定と差し込み(2026-10-13、sim_rl_modes)。Node とブラウザで同じファイル(純粋な ESM、乱数は seed からだけ)。
// modes.json common.perturbation_recording(記録)と modes[recover].training.perturbation(学習)のとおり:
//   種類: 舵のパルス(舵を左右どちらかへ |δ| = 0.52·k で n フレーム、速さは方策のまま)・わざと外す(e_tgt = e_y ± d の線を前方 0.8 m の
//   pure pursuit で T 秒、速さは方策のまま。その側の使える幅 < d + 0.05 なら反対側、両側とも足りなければ舵のパルスに替える。d_min < 0.10 で打ち切り)・
//   ブレーキ/踏み込み(速さの指令を 0 か V_cap に n フレーム、舵は方策のまま)・状態のキック(学習だけ。Node の高速版で状態を書き換える)。
//   予定: 最初の出来事 first_event_s、終わってから gap_after_end_s あけて次、no_start_after_s 以降は始めない。
//   始めてよい位置: R 上の s から先 2.0 m の区間の「すき間 5 cm で動ける幅」の最小が width_min 以上で、除く区間に入らない。
//   だめな位置なら許される最初のフレームまで遅らせる。
// 決定性: 乱数は seed から作り、出来事ごとに決まった数だけ引く。同じ snapshot の列なら Node とブラウザで同じ乱しになる。
import {createRng, mix32} from './rng.mjs';
import {tableAt} from './line_profiles.mjs';
import {CATALOGUE} from './catalogue.mjs';
import {loopPointAt} from './features.mjs';

const C = CATALOGUE.common, REC = C.perturbation_recording, Z = C.zones;
const recoverMode = CATALOGUE.modes.find(m => m.key === 'recover');
export const CLASS_NAMES = Object.freeze(['弱', '中', '強']);
const CLASS_ALIASES = {weak: '弱', medium: '中', strong: '強', '弱': '弱', '中': '中', '強': '強'};
const FPS = 30;

// 記録の決まり(ブラウザでもできる行動の差し込みだけ)
export const RECORD_SPEC = Object.freeze({name: 'record', types: Object.fromEntries(Object.entries(REC.types).map(([k, v]) => [k, v.p])), classes: REC.classes,
  first_event_s: REC.schedule.first_event_s, gap_after_end_s: REC.schedule.gap_after_end_s, no_start_after_s: REC.schedule.no_start_after_s,
  zone: {width_min_m: .60, lookahead_m: 2.0, exclude: [Z.black_sheet_zone_s, Z.back_straight_zone_s]}});
// 学習の決まり(状態のキックを含む。幅の条件は 0.45 m、黒シートの区間だけ除く)
const TP = recoverMode.training.perturbation;
export const TRAIN_SPEC = Object.freeze({name: 'train', types: TP.types_p, classes: REC.classes,
  first_event_s: TP.schedule.first_event_s, gap_after_end_s: TP.schedule.gap_after_end_s, no_start_after_s: REC.schedule.no_start_after_s,
  zone: {width_min_m: .45, lookahead_m: 2.0, exclude: [Z.black_sheet_zone_s]},
  kick: {yaw_rad: [.1, .5], lateral_m: [.05, .25], speed_factor: [.5, 1.3]}});

const inZone = (s, z, Lr) => { const m = ((s % Lr) + Lr) % Lr; return m >= z[0] && m <= z[1]; };

// 乱しを始めてよい位置か(spec.zone)。s は追跡器の s
export function perturbAllowed(s, spec = RECORD_SPEC, loopLength = 26.119648724030924) {
  const zone = spec.zone;
  for (let d = 0; d <= zone.lookahead_m + 1e-9; d += .05) {
    const q = s + d;
    if (tableAt('width05', q) < zone.width_min_m) return false;
    for (const z of zone.exclude) if (inZone(q, z, loopLength)) return false;
  }
  return true;
}

const pickInt = (u, [a, b]) => Math.min(b, a + Math.floor(u * (b - a + 1)));
const pickReal = (u, [a, b]) => a + (b - a) * u;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// spec: RECORD_SPEC か TRAIN_SPEC。seed: シーンの種(config.seed)。cls: 固定の強さ(null なら出来事ごとに等確率)。
// scale: 強さの倍率(学習のカリキュラム。継続フレーム・外す幅・キックの大きさに掛ける)
export function createPerturber({spec = RECORD_SPEC, seed = 1, cls = null, scale = 1, maxSteer = .52, wheelbase = .26, loopLength = 26.119648724030924} = {}) {
  // cls: null / 'auto' = 出来事ごとに 3 つから等確率、'弱' など = 固定、['弱', '中'] = その中から等確率(protocol fallbacks.recover)
  const pool = cls == null || cls === 'auto' ? CLASS_NAMES : (Array.isArray(cls) ? cls : [cls]).map(c => CLASS_ALIASES[c]);
  if (!pool.length || pool.some(c => !c)) throw new Error('perturb: 強さは 弱・中・強 か auto です: ' + cls);
  const rng = createRng(mix32(seed >>> 0, spec.name === 'record' ? 0x9e7 : 0x7a1));
  let nextStart = Math.round(pickReal(rng.uniform(), spec.first_event_s) * FPS);
  let active = null, count = 0;
  const events = [], pt = {x: 0, z: 0, tx: 0, tz: 0};
  const typeOf = u => { let acc = 0; for (const [k, p] of Object.entries(spec.types)) { acc += p; if (u < acc) return k; } return Object.keys(spec.types).at(-1); };
  function begin(f, ctx) {
    const u = Array.from({length: 12}, () => rng.uniform());   // 出来事ごとにいつも 12 個
    let type = typeOf(u[0]);
    const cname = pool[Math.min(pool.length - 1, Math.floor(u[1] * pool.length))], K = spec.classes[cname];
    const sign = u[2] < .5 ? 1 : -1;
    const ev = {n: ++count, type, cls: cname, ps: f, pe: null, sign, s: ctx.s, scale, aborted: false};
    if (type === 'excursion') {
      const d = pickReal(u[5], K.excursion.d_m) * scale, T = pickReal(u[6], K.excursion.T_s);
      // その側の使える幅(先 1.5 m の最小)が d + 0.05 m より狭ければ反対側、両側ともだめなら舵のパルス
      const room = side => { let m = Infinity; for (let q = 0; q <= 1.5 + 1e-9; q += .25) m = Math.min(m, side > 0 ? tableAt('rhoL', ctx.s + q) - ctx.ey : tableAt('rhoR', ctx.s + q) + ctx.ey); return m; };
      let side = sign;
      if (room(side) < d + .05) side = -side;
      if (room(side) < d + .05) type = ev.type = 'steer_pulse';
      else { Object.assign(ev, {sign: side, d, T, e_tgt: ctx.ey + side * d, frames: Math.max(1, Math.round(T * FPS))}); }
    }
    if (ev.type === 'steer_pulse') Object.assign(ev, {frames: Math.max(1, Math.round(pickInt(u[3], K.steer_pulse.frames) * scale)), k: Math.min(1, pickReal(u[4], K.steer_pulse.k))});
    if (ev.type === 'speed_pulse') {
      const brake = u[7] < .5;
      Object.assign(ev, {kind: brake ? 'brake' : 'burst', frames: Math.max(1, Math.round(pickInt(u[3], brake ? K.speed_pulse.brake_frames : K.speed_pulse.burst_frames) * scale))});
    }
    if (ev.type === 'state_kick') {
      const kk = spec.kick, which = Math.floor(u[8] * 6);   // 0..2 は 1 つ、3..5 は 2 つ
      const parts = [['yaw'], ['lateral'], ['speed'], ['yaw', 'lateral'], ['yaw', 'speed'], ['lateral', 'speed']][which];
      ev.kick = {};
      if (parts.includes('yaw')) ev.kick.yaw = sign * pickReal(u[9], kk.yaw_rad) * scale;
      if (parts.includes('lateral')) ev.kick.lateral = (u[11] < .5 ? 1 : -1) * pickReal(u[10], kk.lateral_m) * scale;
      if (parts.includes('speed')) ev.kick.speed_factor = 1 + (pickReal(u[11], kk.speed_factor) - 1) * scale;
      ev.frames = 1;
    }
    ev.pe_planned = f + ev.frames;
    events.push(ev);
    return ev;
  }
  function finish(f) {
    active.pe = f;
    nextStart = f + Math.round(pickReal(rng.uniform(), spec.gap_after_end_s) * FPS);
    active = null;
  }
  // フレーム f(この操作を適用するフレーム)の乱し。ctx = {s, ey, dmin, x, z, yaw, speedCap, policy: {steering, speed}}
  // 戻り値 {inj: null | {steering?, speed?}, kick: null | {...}, ev: 出来事の番号(無ければ 0), phase: 'drive' | 'perturb'}
  function step(f, ctx) {
    if (active && active.type === 'excursion' && ctx.dmin < .10 && f > active.ps) { active.aborted = true; finish(f); }
    if (active && f >= active.pe_planned) finish(f);
    if (!active && f >= nextStart && f <= Math.round(spec.no_start_after_s * FPS) && perturbAllowed(ctx.s, spec, loopLength)) active = begin(f, ctx);
    if (!active) return {inj: null, kick: null, ev: 0, phase: 'drive'};
    const ev = active;
    if (ev.type === 'state_kick') {
      const out = {inj: null, kick: f === ev.ps ? ev.kick : null, ev: ev.n, phase: 'kick'};
      return out;
    }
    let inj;
    if (ev.type === 'steer_pulse') inj = {steering: ev.sign * maxSteer * ev.k};
    else if (ev.type === 'speed_pulse') inj = {speed: ev.kind === 'brake' ? 0 : ctx.speedCap};
    else {
      // わざと外す: R を e_tgt ずらした線の前方 0.8 m の点を pure pursuit で追う
      loopPointAt(ctx.s + .8, pt);
      const tx = pt.x + ev.e_tgt * pt.tz, tz = pt.z - ev.e_tgt * pt.tx;
      const dx = tx - ctx.x, dz = tz - ctx.z, alpha = wrap(Math.atan2(-dz, dx) - ctx.yaw), ld = Math.max(Math.hypot(dx, dz), .3);
      inj = {steering: Math.max(-maxSteer, Math.min(maxSteer, Math.atan2(2 * wheelbase * Math.sin(alpha), ld)))};
    }
    return {inj, kick: null, ev: ev.n, phase: 'perturb'};
  }
  return {step, get events() { return events; }, get active() { return active; }, get nextStart() { return nextStart; }};
}

// 学習から外す観測 k(protocol definitions.mask_rule): 学習しない操作 j(l = 0)を正解(k+1〜k+10)に含む k。
// learn[j] = 0/1(フレームの操作)→ mask[k] = true なら外す。乱しの区間 [ps, pe) なら k ∈ [ps − 10, pe − 2]
export function maskFromLearn(learn, horizon = 10) {
  const n = learn.length, mask = new Uint8Array(n);
  for (let j = 0; j < n; j++) if (!learn[j]) for (let k = Math.max(0, j - horizon); k <= j - 1; k++) mask[k] = 1;
  return mask;
}
