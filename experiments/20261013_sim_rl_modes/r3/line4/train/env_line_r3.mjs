// 経路いろいろ r3 の学習の環境(2026-10-14、r3 の学習の段 line)。protocol_line_r3.json training のとおり。
// 写した env_modes.mjs(r3/line/train/env_modes.mjs、元は rl/env_modes.mjs)の ModeEnv を包み、run 'line_r3' のときだけ次を替える:
//   - 開始: 45% = 記録・試験の開始の分布(ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 12 cm)
//           35% = 大きく横を向いた開始だけ(同じ分布から棄却法、R に対する向きのずれ |Δψ| ≥ 0.8 rad、開始のすき間 ≥ 12 cm)
//           20% = 広め(±0.25 m・±20°、開始のすき間 ≥ 10 cm)
//   - 報酬: r2 の line と同じ(進みに追従の掛け算 σ 0.08・追従 −0.05·min(1,(e/0.15)²)・3 cm 未満で −10 で終える)に、
//           すき間の壁を d_min < 7 cm で −0.3·min(4, ((0.07 − d)/0.03)²) に(r2 は 6 cm・0.2)、
//           止まりの見張り(1.5 秒以降、直近 60 フレームの R の進みが 0.6 m 未満なら −10 で終える。obstacle r2b と同じ 'slow')を足す
//   - 目標の線 v2・観測 cond97・c_v の混ぜ方(80% U(0.9, 1.0)・20% U(0.6, 0.9))・経路の形の割合は r2 と同じ
// run 'line_r3' 以外は元の ModeEnv のまま(この写しでは使わない)。
import {ModeEnv as BaseModeEnv, LINE_R2_DEFAULTS, sampleStartMix} from './env_modes.mjs';
import {activeWorldFor} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {projectWindow, loopPointAt, LOOP_LENGTH} from '../rt4/features.mjs';
import {modeSpec} from '../rt4/mode_runtime.mjs';

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
export const LINE_R3_DEFAULTS = Object.freeze({
  start_mix: Object.freeze([
    Object.freeze({p: .45, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12}),
    Object.freeze({p: .35, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12, min_abs_dpsi_rad: .8}),
    Object.freeze({p: .2, gate: 'U{0..32}', left_offset_m: [-.25, .25], left_yaw_deg: [-20, 20], min_initial_clearance_m: .10})]),
  heading_max_draws: 5000,
  slow_progress: Object.freeze({after_s: 1.5, window_frames: 60, min_progress_m: .6}),
  reward: Object.freeze({kind: 'cond', tracking_w: .05, tracking_scale_m: .15, progress_track_sigma_m: .08,
    clearance_barrier_w: .3, clearance_barrier_start_m: .07, barrier_scale_m: .03, barrier_cap: 4, terminal_clearance_m: .03}),
});

export function trainConfigLineR3() {
  const D = LINE_R3_DEFAULTS, m = modeSpec('line');
  return {run: 'line_r3', mode: 'line', wrapper: 'cond97', virtual: true, gate_miss_terminal: true, line_profile: LINE_R2_DEFAULTS.line_profile,
    cond: {families_p: m.training.conditioning.families_p, c_v_mix: LINE_R2_DEFAULTS.c_v_mix},
    start_mix: D.start_mix, slow_progress: {...D.slow_progress}, reward: {...D.reward}};
}

// 開始の向きと、開始の位置の R の向き(R への射影の s ± 0.05 m の点から)のずれ(protocol_line_r3.json definitions_added.large_heading_start)
const P0 = {x: 0, z: 0, tx: 0, tz: 0}, P1 = {x: 0, z: 0, tx: 0, tz: 0};
export function startDpsi(baseWorld, gate, offset, yawDeg) {
  const {world} = activeWorldFor(baseWorld, gate, offset, yawDeg), sp = world.spawn;
  const s = projectWindow(sp.x, sp.z, 0, LOOP_LENGTH).s;
  loopPointAt(s - .05, P0); loopPointAt(s + .05, P1);
  return {dpsi: wrap(sp.yaw - Math.atan2(-(P1.z - P0.z), P1.x - P0.x)), s};
}

export class ModeEnv extends BaseModeEnv {
  constructor(opts = {}) {
    if (opts.run !== 'line_r3') { super(opts); return; }
    super({...opts, run: 'line'});
    this.run = 'line_r3';
    this.cfg = trainConfigLineR3();
    this.mix = this.cfg.start_mix;
    this.headingDraws = 0;
  }
  // 大きく横を向いた開始の成分は、向きのずれが足りなければ同じ分布から位置・向きだけを引き直す(棄却法)。開始のすき間は元の reset が調べる
  reset(rng, start = null, extra = {}) {
    if (this.run !== 'line_r3' || start) return super.reset(rng, start, extra);
    let st = sampleStartMix(rng, this.mix);
    const comp = this.mix[st.component];
    if (comp.min_abs_dpsi_rad != null) {
      const [o0, o1] = comp.left_offset_m, [y0, y1] = comp.left_yaw_deg;
      let h = startDpsi(this.baseWorld, st.gate, st.left_offset_m, st.left_yaw_deg), k = 0;
      while (Math.abs(h.dpsi) < comp.min_abs_dpsi_rad) {
        if (++k > LINE_R3_DEFAULTS.heading_max_draws) throw new Error('env_line_r3: 大きく横を向いた開始が引けません');
        st = {...st, gate: Math.min(32, Math.floor(rng.uniform() * 33)), left_offset_m: o0 + (o1 - o0) * rng.uniform(), left_yaw_deg: y0 + (y1 - y0) * rng.uniform()};
        h = startDpsi(this.baseWorld, st.gate, st.left_offset_m, st.left_yaw_deg);
      }
      this.headingDraws += k;
      st.dpsi = h.dpsi;
    }
    const ok = super.reset(rng, st, extra);
    if (ok && st.dpsi != null) this.start.dpsi = st.dpsi;
    return ok;
  }
}
