// 経路いろいろ r4b の学習の環境(2026-10-14 夜、line r4 の段のやり直し)。protocol_line_r4b.json training のとおり。
// r4 の env_line_r4.mjs(写し、同じ場所)の ModeEnv を包み、run 'line_r4b' のときだけ開始の混ぜ方と、一部の成分の形・c_v を替える
// (報酬・目標の線・観測は r3・r4 と同じ):
//   30% = 記録・試験の開始の分布(ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 12 cm)。形・c_v はいつもの混ぜ方
//   15% = 大きく横を向いた開始だけ(同じ分布から棄却法、|Δψ| ≥ 0.8 rad、≥ 12 cm)
//   12% = 角の箱(ゲート U{0..32}・4 つの角を等しく、|左右| U(0.13, 0.22) m・|向き| U(9, 17)°、≥ 10 cm)
//   13% = 崩れた角(ゲート U{15..18}・左右 U(−0.23, −0.10) m・向き U(+6, +18)°、≥ 10 cm。確認 r3 の衝突・3 cm 未満・止まりの所)
//   15% = ゲート 30 の角 × 右寄りの形(新。確認 line r3 の再確認の指摘 1): ゲート U{28..32}・左右 U(+0.06, +0.23) m・向き U(+4, +18)°(≥ 10 cm)、
//         形は const c0(半分 U(−0.9, −0.8)・半分 U(−0.8, −0.5))、c_v U(0.9, 1.0)
//    7% = 角の箱 × 形の端(新): 角の箱と同じ開始に、形を範囲の端(const c0 ±0.9・inside/outside/late_apex/early_apex a 0.9)の 6 つ + random
//         から等しく、c_v U(0.9, 1.0)
//    8% = 広め(±0.25 m・±20°、≥ 10 cm)
// 形・c_v を決める成分は、引いた値を元の reset の extra.line・extra.c_v に渡す(元の reset が引く乱数の数は変わらない)。
// 角の成分は、開始のすき間が足りなければ同じ成分の中で引き直す(上限 40 回。r4 と同じ)。
import {ModeEnv as R4Env, LINE_R4_DEFAULTS, trainConfigLineR4} from './env_line_r4.mjs';
import {ModeEnv as R3Env, startDpsi} from './env_line_r3.mjs';
import {sampleStartMix} from './env_modes.mjs';

const F = Object.freeze;
export const LINE_R4B_DEFAULTS = F({
  start_mix: F([
    F({p: .30, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12, name: 'record'}),
    F({p: .15, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12, min_abs_dpsi_rad: .8, name: 'large_heading'}),
    F({p: .12, gate: 'U{0..32}', left_offset_m: [-.22, .22], left_yaw_deg: [-17, 17], min_initial_clearance_m: .10, corner: F({abs_offset_m: [.13, .22], abs_yaw_deg: [9, 17]}), name: 'corner_box'}),
    F({p: .13, gate: 'U{15..18}', left_offset_m: [-.23, -.10], left_yaw_deg: [6, 18], min_initial_clearance_m: .10, gates: F([15, 16, 17, 18]), name: 'hot_corner'}),
    F({p: .15, gate: 'U{28..32}', left_offset_m: [.06, .23], left_yaw_deg: [4, 18], min_initial_clearance_m: .10, gates: F([28, 29, 30, 31, 32]),
      shape: F({kind: 'right', c0_lo: F([-.9, -.8]), c0_hi: F([-.8, -.5]), c_v: F([.9, 1.0])}), name: 'g30_right'}),
    F({p: .07, gate: 'U{0..32}', left_offset_m: [-.22, .22], left_yaw_deg: [-17, 17], min_initial_clearance_m: .10, corner: F({abs_offset_m: [.13, .22], abs_yaw_deg: [9, 17]}),
      shape: F({kind: 'ext', c_v: F([.9, 1.0])}), name: 'corner_ext'}),
    F({p: .08, gate: 'U{0..32}', left_offset_m: [-.25, .25], left_yaw_deg: [-20, 20], min_initial_clearance_m: .10, name: 'wide'})]),
  heading_max_draws: 5000, corner_max_draws: 40,
});
const EXT6 = [{family: 'const', c0: .9}, {family: 'const', c0: -.9}, {family: 'inside', a: .9}, {family: 'outside', a: .9}, {family: 'late_apex', a: .9}, {family: 'early_apex', a: .9}];

export function trainConfigLineR4b() {
  return {...trainConfigLineR4(), run: 'line_r4b', start_mix: LINE_R4B_DEFAULTS.start_mix};
}

// 成分の形・c_v(引く数はいつも 3 つ: 種類・値・c_v)
function shapeFor(sh, rng) {
  const u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform(), seed = rng.uint32();
  const c_v = sh.c_v[0] + (sh.c_v[1] - sh.c_v[0]) * u3;
  if (sh.kind === 'right') {
    const [a, b] = u1 < .5 ? sh.c0_lo : sh.c0_hi;
    return {line: {family: 'const', c0: a + (b - a) * u2}, c_v};
  }
  const k = Math.min(6, Math.floor(u1 * 7));
  return {line: k < 6 ? {...EXT6[k]} : {family: 'random', seed}, c_v};
}

export class ModeEnv extends R4Env {
  constructor(opts = {}) {
    if (opts.run !== 'line_r4b') { super(opts); return; }
    super({...opts, run: 'line_r4'});
    this.run = 'line_r4b';
    this.cfg = trainConfigLineR4b();
    this.mix = this.cfg.start_mix;
    this.headingDraws = 0; this.cornerDraws = 0;
    this.componentCount = new Array(this.mix.length).fill(0);
  }
  reset(rng, start = null, extra = {}) {
    if (this.run !== 'line_r4b' || start) return super.reset(rng, start, extra);
    let st = sampleStartMix(rng, this.mix);
    const comp = this.mix[st.component];
    this.componentCount[st.component]++;
    let ex = extra;
    if (comp.shape) { const s = shapeFor(comp.shape, rng); ex = {...extra, line: s.line, c_v: s.c_v}; }
    const done = ok => { if (ok) this.start.component_name = comp.name; return ok; };
    if (comp.min_abs_dpsi_rad != null) {
      const [o0, o1] = comp.left_offset_m, [y0, y1] = comp.left_yaw_deg;
      let h = startDpsi(this.baseWorld, st.gate, st.left_offset_m, st.left_yaw_deg), k = 0;
      while (Math.abs(h.dpsi) < comp.min_abs_dpsi_rad) {
        if (++k > LINE_R4B_DEFAULTS.heading_max_draws) throw new Error('env_line_r4b: 大きく横を向いた開始が引けません');
        st = {...st, gate: Math.min(32, Math.floor(rng.uniform() * 33)), left_offset_m: o0 + (o1 - o0) * rng.uniform(), left_yaw_deg: y0 + (y1 - y0) * rng.uniform()};
        h = startDpsi(this.baseWorld, st.gate, st.left_offset_m, st.left_yaw_deg);
      }
      this.headingDraws += k;
      st.dpsi = h.dpsi;
      const ok = R3Env.prototype.reset.call(this, rng, st, ex);
      if (ok) this.start.dpsi = st.dpsi;
      return done(ok);
    }
    if (comp.corner || comp.gates) {
      for (let k = 0; k < LINE_R4B_DEFAULTS.corner_max_draws; k++) {
        const c = this.drawCorner(comp, rng);
        this.cornerDraws++;
        const ok = R3Env.prototype.reset.call(this, rng, {...st, ...c}, ex);
        if (ok) return done(true);
      }
      return false;
    }
    return done(R3Env.prototype.reset.call(this, rng, st, ex));
  }
}
