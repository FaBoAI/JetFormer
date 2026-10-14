// 乱し v2(jf-perturb-record/v2。2026-10-14、離脱復活 r3、protocol_recover_r3.json perturbation_r3)。Node とブラウザで同じファイル(純粋な ESM)。
// v1(perturb.mjs createPerturber)の種類・強さの中身・予定・幅の決まりはそのまま。足したのは次の 3 つだけ:
//   (1) 立て直せる表(recoverable_table.mjs): マス = (R 上の s の 0.25 m のマス, 速さの 0.5 m/s のマス, 18 通りの組)。サンプル 3 つ以上で
//       立て直せる割合 ≥ min_rate(記録・試験 0.95、学習 0.80)のマスだけ「使える」。サンプルの無いマスは使えない。
//   (2) 替え方: 出来事ごとの 12 個の一様乱数は v1 と同じ引き方で「頼んだ組」を決め、頼んだ組が今の s・速さで使えなければ
//       (1) 向きを逆に(ブレーキ/踏み込みは踏み込み ⇄ ブレーキ)、(2) 同じ強さのわざと外す(頼んだ向き → 逆)、(3) 同じ強さのブレーキ、
//       (4) 1 つ弱い強さで (0)〜(3) をくり返す(弱まで)、の順で最初に使える組(当てた組)にする。わざと外すは、その側の幅(先 1.5 m の最小)が
//       クラスの上限の d + 0.05 m に足りなければ使えない。どれも使えなければそのフレームでは始めず、次のフレームで同じ頼んだ組で調べ直す(乱数は引き直さない)。
//   (3) 始まりを散らす: 使える組があるフレームでも、mix32(mix32(シーンの種, f), 0x5a7e) / 2³² < 0.25 のときだけ始める(乱数ではなく決定的)。
// 当てた組の中身(フレーム数・k・d・T)は v1 と同じく、同じ一様乱数でクラスの範囲から引く(表は上限で作ったので、範囲の中のどの値でも同じかそれより立て直しやすい)。
// 乱数の列: 出来事ごとに 12 個(頼んだときに 1 度だけ)+ 終わったときに 1 個(次までの間)= v1 と同じ数・同じ順。
// 学習(spec.name = 'train')では状態のキック(表に無い)も v1 と同じく引く。キックは表を引かずに、幅の決まり・散らしだけで始める。
import {createRng, mix32} from './rng.mjs';
import {tableAt} from './line_profiles.mjs';
import {loopPointAt} from './features.mjs';
import {perturbAllowed, RECORD_SPEC, CLASS_NAMES} from './perturb.mjs';

export const PERTURB_V2_ID = 'jf-perturb-record/v2';
export const V2_DEFAULTS = Object.freeze({min_rate: .95, min_n: 3, s_bin_m: .25, v_bin_mps: .5, spread_p: .25, spread_salt: 0x5a7e});
const CLASS_ALIASES = {weak: '弱', medium: '中', strong: '強', '弱': '弱', '中': '中', '強': '強'};
const FPS = 30;
const pickInt = (u, [a, b]) => Math.min(b, a + Math.floor(u * (b - a + 1)));
const pickReal = (u, [a, b]) => a + (b - a) * u;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// 組の名前(表の鍵): steer_pulse:弱:L・excursion:中:R・brake:強・burst:弱
export function comboKey(type, cls, sign = 1, kind = null) {
  if (type === 'speed_pulse') return `${kind}:${cls}`;
  return `${type}:${cls}:${sign > 0 ? 'L' : 'R'}`;
}
// 表を引く: 使えるか(サンプル min_n 以上・割合 min_rate 以上)
export function cellOf(table, s, v, key) {
  const L = table.loop_length_m, sm = ((s % L) + L) % L;
  const sb = Math.floor(sm / table.s_bin_m), vb = Math.floor(Math.max(0, v) / table.v_bin_mps);
  return table.cells[`${sb}|${vb}|${key}`] ?? null;
}
export function usable(table, s, v, key, minRate, minN = 3) {
  const c = cellOf(table, s, v, key);
  return !!c && c[0] >= minN && c[1] / c[0] >= minRate - 1e-12;
}
// わざと外すの幅(perturb.mjs の begin と同じ式。先 1.5 m の最小)
function room(side, s, ey) { let m = Infinity; for (let q = 0; q <= 1.5 + 1e-9; q += .25) m = Math.min(m, side > 0 ? tableAt('rhoL', s + q) - ey : tableAt('rhoR', s + q) + ey); return m; }

// 替え方の順番(頼んだ組 → 候補の並び)。戻り値 [{type, cls, sign, kind, step}]
export function fallbackOrder(req, classes = CLASS_NAMES) {
  const out = [], seen = new Set();
  const push = (c, step) => { const k = comboKey(c.type, c.cls, c.sign, c.kind); if (!seen.has(k)) { seen.add(k); out.push({...c, step}); } };
  let ci = classes.indexOf(req.cls);
  if (ci < 0) ci = classes.length - 1;
  for (let i = ci; i >= 0; i--) {
    const cls = classes[i], weaker = i < ci ? 'weaker' + (ci - i) + '+' : '';
    if (req.type === 'speed_pulse') {
      push({type: 'speed_pulse', cls, kind: req.kind, sign: req.sign}, weaker + 'requested');
      push({type: 'speed_pulse', cls, kind: req.kind === 'brake' ? 'burst' : 'brake', sign: req.sign}, weaker + 'reverse');
    } else {
      push({type: req.type, cls, sign: req.sign}, weaker + 'requested');
      push({type: req.type, cls, sign: -req.sign}, weaker + 'reverse');
    }
    push({type: 'excursion', cls, sign: req.sign}, weaker + 'excursion');
    push({type: 'excursion', cls, sign: -req.sign}, weaker + 'excursion_reverse');
    push({type: 'speed_pulse', cls, kind: 'brake', sign: req.sign}, weaker + 'brake');
  }
  return out;
}

// spec: RECORD_SPEC か TRAIN_SPEC(perturb.mjs)。table: recoverable_table.mjs の RECOVERABLE_TABLE。minRate: 記録・試験 0.95、学習 0.80
export function createPerturberV2({spec = RECORD_SPEC, seed = 1, cls = null, scale = 1, maxSteer = .52, wheelbase = .26, loopLength = 26.119648724030924,
  table, minRate = V2_DEFAULTS.min_rate, minN = V2_DEFAULTS.min_n, spreadP = V2_DEFAULTS.spread_p} = {}) {
  if (!table || !table.cells) throw new Error('perturb_v2: 立て直せる表がありません');
  const pool = cls == null || cls === 'auto' ? CLASS_NAMES : (Array.isArray(cls) ? cls : [cls]).map(c => CLASS_ALIASES[c]);
  if (!pool.length || pool.some(c => !c)) throw new Error('perturb_v2: 強さは 弱・中・強 か auto です: ' + cls);
  const rng = createRng(mix32(seed >>> 0, spec.name === 'record' ? 0x9e7 : 0x7a1));
  let nextStart = Math.round(pickReal(rng.uniform(), spec.first_event_s) * FPS);
  let active = null, pending = null, count = 0, waits = 0;
  const events = [], pt = {x: 0, z: 0, tx: 0, tz: 0};
  const typeOf = u => { let acc = 0; for (const [k, p] of Object.entries(spec.types)) { acc += p; if (u < acc) return k; } return Object.keys(spec.types).at(-1); };
  const noStart = Math.round(spec.no_start_after_s * FPS);
  const spreadOk = f => mix32(mix32(seed >>> 0, f >>> 0), V2_DEFAULTS.spread_salt) / 4294967296 < spreadP;
  // 頼んだ組(12 個の一様乱数。v1 の begin と同じ引き方)
  function request(f) {
    const u = Array.from({length: 12}, () => rng.uniform());
    const type = typeOf(u[0]);
    const cname = pool[Math.min(pool.length - 1, Math.floor(u[1] * pool.length))];
    const sign = u[2] < .5 ? 1 : -1;
    return {u, f_req: f, type, cls: cname, sign, kind: type === 'speed_pulse' ? (u[7] < .5 ? 'brake' : 'burst') : null};
  }
  function usableCombo(c, ctx) {
    if (c.type === 'excursion' && room(c.sign, ctx.s, ctx.ey) < spec.classes[c.cls].excursion.d_m[1] * Math.max(1, scale) + .05) return false;
    return usable(table, ctx.s, ctx.v, comboKey(c.type, c.cls, c.sign, c.kind), minRate, minN);
  }
  function assign(req, ctx) {
    if (req.type === 'state_kick') return {type: 'state_kick', cls: req.cls, sign: req.sign, step: 'requested'};
    // 替え方の弱い側は、固定の強さ(pool)のときもその中だけ(代わりの試験 弱・中 なら 中 → 弱)
    const classes = CLASS_NAMES.filter(c => pool.includes(c));
    for (const c of fallbackOrder(req, classes)) if (usableCombo(c, ctx)) return c;
    return null;
  }
  function begin(f, ctx, req, as) {
    const u = req.u, K = spec.classes[as.cls];
    const ev = {n: ++count, type: as.type, cls: as.cls, ps: f, pe: null, sign: as.sign, s: ctx.s, v: ctx.v, scale, aborted: false,
      req: {type: req.type, cls: req.cls, sign: req.sign, kind: req.kind, f: req.f_req}, fallback: as.step, wait_frames: f - req.f_req};
    if (as.type === 'excursion') {
      const d = pickReal(u[5], K.excursion.d_m) * scale, T = pickReal(u[6], K.excursion.T_s);
      Object.assign(ev, {d, T, e_tgt: ctx.ey + as.sign * d, frames: Math.max(1, Math.round(T * FPS))});
    }
    if (as.type === 'steer_pulse') Object.assign(ev, {frames: Math.max(1, Math.round(pickInt(u[3], K.steer_pulse.frames) * scale)), k: Math.min(1, pickReal(u[4], K.steer_pulse.k))});
    if (as.type === 'speed_pulse') {
      const brake = as.kind === 'brake';
      Object.assign(ev, {kind: as.kind, frames: Math.max(1, Math.round(pickInt(u[3], brake ? K.speed_pulse.brake_frames : K.speed_pulse.burst_frames) * scale))});
    }
    if (as.type === 'state_kick') {
      const kk = spec.kick, which = Math.floor(u[8] * 6);
      const parts = [['yaw'], ['lateral'], ['speed'], ['yaw', 'lateral'], ['yaw', 'speed'], ['lateral', 'speed']][which];
      ev.kick = {};
      if (parts.includes('yaw')) ev.kick.yaw = as.sign * pickReal(u[9], kk.yaw_rad) * scale;
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
  // フレーム f の乱し。ctx = {s, ey, dmin, x, z, yaw, v(sim の速さ), speedCap, policy}
  // 戻り値 {inj, kick, ev, phase: 'drive' | 'perturb' | 'kick'}(v1 と同じ形)
  function step(f, ctx) {
    if (active && active.type === 'excursion' && ctx.dmin < .10 && f > active.ps) { active.aborted = true; finish(f); }
    if (active && f >= active.pe_planned) finish(f);
    if (!active && f >= nextStart && f <= noStart && perturbAllowed(ctx.s, spec, loopLength)) {
      if (!pending) pending = request(f);
      const as = assign(pending, ctx);
      if (as && spreadOk(f)) { active = begin(f, ctx, pending, as); pending = null; }
      else waits++;
    }
    if (!active) return {inj: null, kick: null, ev: 0, phase: 'drive'};
    const ev = active;
    if (ev.type === 'state_kick') return {inj: null, kick: f === ev.ps ? ev.kick : null, ev: ev.n, phase: 'kick'};
    let inj;
    if (ev.type === 'steer_pulse') inj = {steering: ev.sign * maxSteer * ev.k};
    else if (ev.type === 'speed_pulse') inj = {speed: ev.kind === 'brake' ? 0 : ctx.speedCap};
    else {
      loopPointAt(ctx.s + .8, pt);
      const tx = pt.x + ev.e_tgt * pt.tz, tz = pt.z - ev.e_tgt * pt.tx;
      const dx = tx - ctx.x, dz = tz - ctx.z, alpha = wrap(Math.atan2(-dz, dx) - ctx.yaw), ld = Math.max(Math.hypot(dx, dz), .3);
      inj = {steering: Math.max(-maxSteer, Math.min(maxSteer, Math.atan2(2 * wheelbase * Math.sin(alpha), ld)))};
    }
    return {inj, kick: null, ev: ev.n, phase: 'perturb'};
  }
  return {step, version: PERTURB_V2_ID, get events() { return events; }, get active() { return active; }, get nextStart() { return nextStart; },
    get pending() { return pending; }, get waits() { return waits; }};
}
