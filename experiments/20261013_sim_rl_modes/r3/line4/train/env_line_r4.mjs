// 経路いろいろ r4 の学習の環境(2026-10-14、line r4 の段)。protocol_line_r4.json training のとおり。
// r3 の env_line_r3.mjs(写し、同じ場所)の ModeEnv を包み、run 'line_r4' のときだけ開始の混ぜ方を替える(報酬・目標の線・観測・c_v は r3 と同じ):
//   35% = 記録・試験の開始の分布(ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 12 cm)
//   20% = 大きく横を向いた開始だけ(同じ分布から棄却法、|Δψ| ≥ 0.8 rad、≥ 12 cm)
//   15% = 角の箱(ゲート U{0..32}・4 つの角を等しく、|左右| U(0.13, 0.22) m・|向き| U(9, 17)°、≥ 10 cm)
//   15% = 崩れた角(ゲート U{15..18}・左右 U(−0.23, −0.10) m・向き U(+6, +18)°、≥ 10 cm。確認 r3 の衝突・3 cm 未満・止まりの所)
//   15% = 広め(±0.25 m・±20°、≥ 10 cm)
// 角の成分は、開始のすき間が足りなければ同じ成分の中で引き直す(上限 40 回。r3 の大きく横を向いた成分と同じ考え方)。
// 確認 r3 の下調べ(probe/feas_corner.mjs): ゲート 17 の角(−0.2 m・+15°)から前へだけで 50 cm 以上空けたまま 1.8 m 進む操作がある = 方策の弱さ。
import {ModeEnv as R3Env, LINE_R3_DEFAULTS, startDpsi, trainConfigLineR3} from './env_line_r3.mjs';
import {sampleStartMix} from './env_modes.mjs';

export const LINE_R4_DEFAULTS = Object.freeze({
  start_mix: Object.freeze([
    Object.freeze({p: .35, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12, name: 'record'}),
    Object.freeze({p: .20, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12, min_abs_dpsi_rad: .8, name: 'large_heading'}),
    Object.freeze({p: .15, gate: 'U{0..32}', left_offset_m: [-.22, .22], left_yaw_deg: [-17, 17], min_initial_clearance_m: .10, corner: Object.freeze({abs_offset_m: [.13, .22], abs_yaw_deg: [9, 17]}), name: 'corner_box'}),
    Object.freeze({p: .15, gate: 'U{15..18}', left_offset_m: [-.23, -.10], left_yaw_deg: [6, 18], min_initial_clearance_m: .10, gates: Object.freeze([15, 16, 17, 18]), name: 'hot_corner'}),
    Object.freeze({p: .15, gate: 'U{0..32}', left_offset_m: [-.25, .25], left_yaw_deg: [-20, 20], min_initial_clearance_m: .10, name: 'wide'})]),
  heading_max_draws: 5000, corner_max_draws: 40,
});

export function trainConfigLineR4() {
  return {...trainConfigLineR3(), run: 'line_r4', start_mix: LINE_R4_DEFAULTS.start_mix};
}

export class ModeEnv extends R3Env {
  constructor(opts = {}) {
    if (opts.run !== 'line_r4') { super(opts); return; }
    super({...opts, run: 'line_r3'});
    this.run = 'line_r4';
    this.cfg = trainConfigLineR4();
    this.mix = this.cfg.start_mix;
    this.headingDraws = 0; this.cornerDraws = 0;
    this.componentCount = new Array(this.mix.length).fill(0);
  }
  // 角の成分の 1 回の引き(ゲート・符号・大きさ)。r は rng
  drawCorner(comp, rng) {
    if (comp.gates) {
      const g = comp.gates[Math.min(comp.gates.length - 1, Math.floor(rng.uniform() * comp.gates.length))];
      const [o0, o1] = comp.left_offset_m, [y0, y1] = comp.left_yaw_deg;
      return {gate: g, left_offset_m: o0 + (o1 - o0) * rng.uniform(), left_yaw_deg: y0 + (y1 - y0) * rng.uniform()};
    }
    const C = comp.corner, g = Math.min(32, Math.floor(rng.uniform() * 33)), so = rng.uniform() < .5 ? -1 : 1, sy = rng.uniform() < .5 ? -1 : 1;
    return {gate: g, left_offset_m: so * (C.abs_offset_m[0] + (C.abs_offset_m[1] - C.abs_offset_m[0]) * rng.uniform()),
      left_yaw_deg: sy * (C.abs_yaw_deg[0] + (C.abs_yaw_deg[1] - C.abs_yaw_deg[0]) * rng.uniform())};
  }
  reset(rng, start = null, extra = {}) {
    if (this.run !== 'line_r4' || start) return super.reset(rng, start, extra);
    let st = sampleStartMix(rng, this.mix);
    const comp = this.mix[st.component];
    this.componentCount[st.component]++;
    if (comp.min_abs_dpsi_rad != null) {
      // 大きく横を向いた成分(r3 と同じ棄却法)
      const [o0, o1] = comp.left_offset_m, [y0, y1] = comp.left_yaw_deg;
      let h = startDpsi(this.baseWorld, st.gate, st.left_offset_m, st.left_yaw_deg), k = 0;
      while (Math.abs(h.dpsi) < comp.min_abs_dpsi_rad) {
        if (++k > LINE_R4_DEFAULTS.heading_max_draws) throw new Error('env_line_r4: 大きく横を向いた開始が引けません');
        st = {...st, gate: Math.min(32, Math.floor(rng.uniform() * 33)), left_offset_m: o0 + (o1 - o0) * rng.uniform(), left_yaw_deg: y0 + (y1 - y0) * rng.uniform()};
        h = startDpsi(this.baseWorld, st.gate, st.left_offset_m, st.left_yaw_deg);
      }
      this.headingDraws += k;
      st.dpsi = h.dpsi;
      // 元の ModeEnv.reset に開始を渡す(r3 と同じ。start があると super(r3).reset はそのまま元の reset へ渡す)
      const ok = R3Env.prototype.reset.call(this, rng, st, extra);
      if (ok) { this.start.dpsi = st.dpsi; this.start.component_name = comp.name; }
      return ok;
    }
    if (comp.corner || comp.gates) {
      for (let k = 0; k < LINE_R4_DEFAULTS.corner_max_draws; k++) {
        const c = this.drawCorner(comp, rng);
        this.cornerDraws++;
        const ok = R3Env.prototype.reset.call(this, rng, {...st, ...c}, extra);
        if (ok) { this.start.component_name = comp.name; return true; }
      }
      return false;
    }
    const ok = R3Env.prototype.reset.call(this, rng, st, extra);
    if (ok) this.start.component_name = comp.name;
    return ok;
  }
}
