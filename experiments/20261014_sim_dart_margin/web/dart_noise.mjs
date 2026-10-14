// DART の乱しの予定(2026-10-14、sim_dart_20261014)。ブラウザの差し込みモジュール・Node の予想・付け替えの確かめが同じこのファイルを使う(純粋な ESM)。
//
// 決まりは protocol.json dart(N_fast・N_margin・guards・determinism)。設計の段の measure_teachers.mjs の makeNoise(burst_m_gg・burst_xl_gg)と
// 同じ形の「舵のかたより(バースト)」:
//   - 最初の乱しは first_start_s の U 秒後から始めてよい。前の乱しが終わったら gap_after_end_s の U 秒あけて次
//   - 始める判定(そのフレームで、待ち時間が過ぎていれば): 場所(modes の perturbAllowed(s))→ すき間 d_min ≥ start → ゲートの手前 1.2 m で |lateral| ≤ 0.30 m
//     だめならそのフレームは始めず、次のフレームでまた判定する
//   - 始めたら: かたより ±U(steer_offset_rad)(左右 等確率)・長さ U(duration_frames)(丸め)・25% で速さ × U(speed factor)
//   - 打ち切り: ゲートの手前 1.2 m で |lateral| > 0.45 m、または d_min < abort
// 乱数: rng.mjs の createRng(splitmix32 系)を mix32(mix32(シーンの種, 塩), 引き直しの回 k) で初期化。時刻・Math.random は使わない。
// 乱しは (シーンの種, 塩, k) と、フレームごとの状態(場所・すき間・ゲート)だけで決まる。scale は大きさ(かたより)に掛ける倍率(既定 1。5 回引き直してもだめなシーンだけ 0.7)。
import {createRng, mix32} from './rt/rng.mjs';
import {perturbAllowed} from './rt/perturb.mjs';

export const DART_NOISE_VERSION = 'jf-sim-dart-noise/v1';
export const FPS = 30;
export const STEER_LIMIT = .52;

// protocol.json dart.N_fast / N_margin / guards をそのまま写したもの(tests で protocol.json と照合する)
export const NOISE_SPECS = Object.freeze({
  N_fast: Object.freeze({name: 'N_fast', steer_offset_rad: [.06, .15], duration_frames: [6, 15], gap_after_end_s: [.6, 1.6], first_start_s: [.2, .8],
    speed_p: .25, speed_factor: [.70, .95], guard_start_dmin: .12, guard_abort_dmin: .06, gate_window_m: 1.2, gate_start_max_lat: .30, gate_abort_lat: .45}),
  N_margin: Object.freeze({name: 'N_margin', steer_offset_rad: [.10, .25], duration_frames: [8, 20], gap_after_end_s: [.5, 1.4], first_start_s: [.15, .6],
    speed_p: .25, speed_factor: [.65, .95], guard_start_dmin: .10, guard_abort_dmin: .05, gate_window_m: 1.2, gate_start_max_lat: .30, gate_abort_lat: .45}),
});
// データセットの塩(protocol.json dart.determinism)
export const SALTS = Object.freeze({D1: 0xD1A7, D3: 0xD3A7});

const inGateWindow = (ctx, P) => ctx.gate_along > -P.gate_window_m && ctx.gate_along <= 0;

// seed: シーンの種(config.seed)、salt: データセットの塩、k: 引き直しの回(0 = 最初)、scale: かたよりの倍率
// 戻り値 noise(f, ctx): ctx = {s, dmin, gate_along, gate_lateral}(そのフレームの操作の前の状態で先生が測った値)。
//   → {dSteer, speedFactor, ev}(乱している)か null。同じフレームで 2 回呼ばれたら同じ値を返す。フレームが飛んだら例外。
export function createDartNoise(specName, {seed, salt, k = 0, scale = 1} = {}) {
  const P = NOISE_SPECS[specName];
  if (!P) throw new Error('dart_noise: 知らない乱しです: ' + specName);
  if (!Number.isInteger(seed) || !Number.isInteger(salt) || !Number.isInteger(k)) throw new Error('dart_noise: seed・salt・k は整数です');
  const rng = createRng(mix32(mix32(seed >>> 0, salt >>> 0), k >>> 0));
  const U = ([lo, hi]) => lo + (hi - lo) * rng.uniform();
  let next = Math.round(U(P.first_start_s) * FPS), active = null, lastF = -1, memo = null;
  const events = [];
  function endActive(f, aborted) {
    if (aborted) { active.aborted = f; active.end = f; }
    next = f + Math.round(U(P.gap_after_end_s) * FPS);
    active = null;
  }
  function noise(f, ctx) {
    if (f === lastF) return memo && {...memo};
    if (f !== lastF + 1) throw new Error(`dart_noise: フレームが飛びました(${lastF} → ${f})`);
    lastF = f;
    if (active && f >= active.end) endActive(f, false);
    if (!active && f >= next) {
      let ok = perturbAllowed(ctx.s);
      if (ok && !(ctx.dmin >= P.guard_start_dmin)) ok = false;
      if (ok && inGateWindow(ctx, P) && Math.abs(ctx.gate_lateral) > P.gate_start_max_lat) ok = false;
      if (ok) {
        const amp = scale * U(P.steer_offset_rad) * (rng.uniform() < .5 ? -1 : 1);
        const len = Math.round(U(P.duration_frames));
        const sf = rng.uniform() < P.speed_p ? U(P.speed_factor) : 1;
        active = {ev: events.length, start: f, end: f + len, planned_end: f + len, dSteer: amp, speedFactor: sf, aborted: null};
        events.push(active);
      }
    }
    if (active && inGateWindow(ctx, P) && Math.abs(ctx.gate_lateral) > P.gate_abort_lat) endActive(f, true);
    else if (active && ctx.dmin < P.guard_abort_dmin) endActive(f, true);
    memo = active ? {dSteer: active.dSteer, speedFactor: active.speedFactor, ev: active.ev} : null;
    return memo && {...memo};
  }
  noise.events = events;
  noise.spec = P;
  noise.params = {spec: specName, seed: seed >>> 0, salt: salt >>> 0, k, scale};
  return noise;
}

// 乱しを先生の操作に足す(protocol.json dart.applied_and_previous_action):
//   舵 = clamp(先生の舵 + かたより, ±0.52)、速さ = clamp(先生の速さ × 倍率, 0, speedCap)
export function applyNoise(clean, nz, speedCap) {
  if (!nz) return {steering: clean.steering, speed: clean.speed};
  return {steering: Math.max(-STEER_LIMIT, Math.min(STEER_LIMIT, clean.steering + nz.dSteer)),
    speed: Math.max(0, Math.min(speedCap, clean.speed * nz.speedFactor))};
}
