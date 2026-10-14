// 安全シールド(「絶対壁にぶつからない」の走行中の見張り)と、その中で使う車の式の写し(2026-10-13、sim_rl_modes)。
// Node とブラウザで同じファイル(純粋な ESM。乱数・時刻を使わない。snapshot とフレームの値だけで決まる)。modes.json common.shield のとおり:
//   提案の指令を 1 フレーム、そのあと予備の操縦で H = 24 フレーム(0.8 秒)先まで、sim-core と同じ式で進め、小区間ごとの正確なすき間
//   (車体の長方形と全障害物。features.mjs の clearance)を測る。標準の模型ですき間 ≥ 0.07 m、悲観の模型(舵の速さ 0.85 rad/s・遅延 1 フレーム・
//   ブレーキ 0.8 m/s²)ですき間 ≥ 0.05 m なら合格。だめなら候補(予備の操縦そのもの、舵 {−0.52, −0.26, 0, 0.26, 0.52} × 速さ {0, 提案})のうち
//   合格するもので提案に一番近いもの(舵の差を主に)、どれも合格しなければ予想の最小すき間が一番大きいもの。
// 2026-10-14 no_wall の学習の段で直した(deviations.md 2。検証の組だけで決めた):
//   1. 予備の操縦を 3 つの組にした。どれか 1 つで 0.8 秒の安全が続けば合格(「安全に続けられる操縦がある」の意味は同じ):
//      (a) 使える幅の表の中点の線を前方 0.6 + 0.2·|v| m で追う pure pursuit + ブレーキ、(b) 前方 0.6 m の pure pursuit + ブレーキ(元の予備の操縦)、
//      (c) 提案の舵をそのまま保ってブレーキ。元の (b) だけでは 4 m/s 近くで舵の遅れから蛇行して壁に寄り、ヘアピン(s 5〜7)では回り切れず、
//      安全な指令まで「危ない」と判定していた(パイロットの方策で 12.8% のフレーム)。替えた指令で線を外れ、衝突・ゲートの外し・停止を起こしていた。
//   2. 提案が標準の模型では合格し悲観の模型だけで不合格のときは、提案を替えない(fallback 'proposal_nominal'、sh = 0)。
//      提案が標準の模型でも不合格のときの替え方は、両方に合格する候補 → 標準の模型だけ合格する候補(提案に近い順)→ 予想のすき間が一番大きい候補。
//      悲観の模型ではヘアピンを回り切れない速さでも標準の模型では安全なことが多く、両方に合格する「まっすぐ」の指令に替えるとコースを外れていた。
//   予備の操縦 (a): 使える幅の表の中点の線(最大すき間の線)を前方 0.6 + 0.2·|v| m で追う pure pursuit の舵 + 速さの指令 0(ブレーキ)。
// 2026-10-14 no_wall の作り直し(r2)で直した(jf-shield/v3、deviations.md に記録。確認 r1 の指摘: 遅い舵 + 遅延 1 の物理では
//   標準の模型が外れ、v2 の「標準の模型で合格なら替えない」が衝突を通していた・悲観の模型(舵 0.85)が DR-T の下限 0.8 より甘い):
//   3. 走りながらの見積もり(identify): snapshot の実際の舵角 δ と、自分が返した指令の列から、舵の速さ・遅延(0/1/2 フレーム)・舵のゲイン・ブレーキを見積もり、
//      「標準の模型」の代わりに「見積もった模型」で検査する(keep_nominal の「標準」も見積もった模型)。sim の舵は 1 フレームで高々 舵の速さ × dt しか動かず、
//      届かなかったフレーム(stopped short)ではちょうど 舵の速さ × dt 動くので、届かなかったフレームの |Δδ|/dt がそのまま舵の速さになる。
//      遅延の仮説は、指令の列と δ の動きが合わなくなったら捨てる(届く前に止まった・行き過ぎた・逆へ動いた)。残った仮説のうち一番悪い(遅い)ものを使う。
//      見積もりの前(走り出しの数フレーム)は config の値(標準の物理)。標準の物理なら見積もりは標準の値になり、v2 と同じ判定になる。
//   4. 悲観の模型を DR-T の悪い端にした: 舵の速さ min(0.8, 見積もり)・遅延 max(1, 見積もり)・ブレーキ min(0.7, 見積もり)・μ × 0.85。
// 計算を減らす工夫(判定は変わらない): フレームの両端のすき間 c0・c1 と、そのフレームで車体の点が動ける距離の上限 D から、
//   max(c0, c1) − D ≥ しきい値 ならそのフレームの小区間はどれもしきい値以上(すき間は剛体の動きに対して 1-リプシッツ)。そうでなければ小区間ごとに測る。
// 2026-10-14 no_wall の第 3 回(r3、jf-shield/v4。protocol_no_wall_r3.json runtime.shield。確認 r2 の指摘: DR-T の範囲の中の隅
//   (速い舵 1.8・ブレーキ 0.7・μ × 0.85・摩擦円・前進のゲイン 1.04)で v3 があっても衝突。v3 の模型は摩擦円・μ・前進のゲインを持たず、
//   見積もった模型で合格なら替えなかった):
//   5. 模型 Q(「見積もり + 見積もれない値の悪い端」): 舵の速さ・遅延・舵のゲインは見積もりのまま、ブレーキは見積もり(まだ減速を見ていなければ 0.7)、
//      μ × 0.85・摩擦円あり・前進のゲイン 1.04。先読み 45 フレーム、しきい値 0.07 m。keep_nominal(替えない)にも E と Q の両方の合格が要る。
//   6. 速さの見張り: ブレーキが標準より弱いと見積もったら、v3 の上限にさらに 0.92 を掛ける(weak_extra)。
//   7. c_nominal 0.07 → 0.08(N2 のしきい値と同じ)。悲観の模型・予備の操縦・候補・決定性は v3 のまま。すくみをほどく(unstick)は使わない(既定で切)。
// 2026-10-14 r3 の学習の段で直した(jf-shield/v4.1、deviations.md に記録。protocol_no_wall_r3.json runtime.shield の外。決めたのは試験の前、
//   検証の組と下調べの組だけを見て): 検証の組の残りの衝突(隅: 舵 1.8・遅延 1・ブレーキ 0.7・μ 0.85・摩擦円・前進のゲイン 1.04)を調べると、
//   (a) v4 の速さの見張りは「舵の速さが分かる(known)」までかからず、ブレーキ 0.7 が分かっても(フレーム 25)上限がかからなかった(known はフレーム 50)、
//   (b) 走り出しから方策がブレーキを使わないあいだはブレーキが見積もれず、4.5 m/s の指令のまま 3.9 m/s まで出て、模型 Q の 45 フレームの先読みで
//       危なさが見えたとき(フレーム 44)にはどの候補も合格しなかった(弱いブレーキでは止まるのに 5 秒かかる = 先読みより長い)。そこで:
//   8. ブレーキの部分は舵の速さの見積もりを待たない(brake_without_known)。ブレーキの値は、残った遅延の仮説のうちブレーキを見たものの最小(小さいほど上限がきびしい)
//   9. ブレーキの見積もりの探り(brake_probe): まだどの仮説もブレーキを見ておらず、速さが v_min(2.0 m/s)を越えたら、1 フレームだけ速さの指令を
//      今の速さ × 0.85 にする(sh = 1、fallback 'brake_probe')。そのフレームはブレーキいっぱいで減速するので、ブレーキがそのまま見積もれる(標準の物理で
//      速さが約 0.04 m/s 落ちるだけ)。見えなければ every(15)フレームごとにくり返す。これで弱いブレーキの物理では走り出しの 2 m/s のころから
//      学習と同じ上限(4.5·√(b/1.2)·0.92)がかかる(学習の速さの見張りはフレーム 3 から本当の値でかかっている)。
//   v4 と同じ動きにするには governor: {brake_without_known: false, brake_probe: {enabled: false}} を渡す。
//   既定の値が v4 の値(protocol_no_wall_r3.json の値と同じ)。v3 と同じ動きにするには
//   {c_nominal: 0.07, worst: {enabled: false}, governor: {weak_extra: 1}} を渡す(r3/nw/test_shield_v4.mjs で web/rl の v3 と操作が全く同じことを試験)。
import {clearance, surfaceMu, projectWindow, loopPointAt, G} from './features.mjs';
import {tableAt} from './line_profiles.mjs';

export const SHIELD_VERSION = 'jf-shield/v4.1';
export const SHIELD_DEFAULTS = Object.freeze({version: SHIELD_VERSION, H: 24, c_nominal: .08, c_pessimistic: .05,
  // 悲観の模型(v3: DR-T の悪い端。見積もりがもっと悪ければそちら)。v2 は 舵 0.85・遅延 1・ブレーキ 0.8
  pessimistic: Object.freeze({steerRate: .8, delay: 1, brakeDeceleration: .7, frictionMultiplier: .85}),
  // 走りながらの見積もり(v3)。gain = 舵のゲインの仮説の幅、delays = 遅延の仮説、tol = 数値の許し
  identify: Object.freeze({enabled: true, gain: Object.freeze([.9, 1.1]), delays: Object.freeze([0, 1, 2]), tol: 1e-7}),
  // 速さの見張り(v3): 舵の速さ r と遅延 d が分かったら、舵の応答の時間 T = d·dt + θ/r が標準(θ/r_標準)より長いぶん、速さの指令の上限を下げる
  //   上限 = v_max · T_標準 / T(標準の物理では T = T_標準 なので上限なし)。0.8 秒の先読みでは、遅い舵で速く走り始めてからでは止まれないため
  governor: Object.freeze({enabled: true, theta_rad: .35, v_max: 4.5, min_frame: 3, brake: true, weak_extra: .92,
    brake_without_known: true, brake_probe: Object.freeze({enabled: true, v_min: 2.0, factor: .85, every: 15})}),
  // 予備の操縦の組(la + k·|v| m 先を追う pure pursuit + ブレーキ、hold = 提案の舵を保ってブレーキ)。どれか 1 つで合格すればよい
  backups: Object.freeze([Object.freeze({la: .6, k: .2}), Object.freeze({la: .6, k: 0}), Object.freeze({hold: true})]),
  keep_nominal: true,   // 提案が標準の模型(v3 では見積もった模型)で合格なら替えない
  // v4(r3 の下調べ、2026-10-14): 見積もれない値(μ・摩擦円・前進のゲイン)を DR-T の悪い端にした「見積もり + 悪い端」の模型 Q。
  //   見積もった値(舵の速さ・遅延・舵のゲイン)はそのまま、ブレーキはまだ見ていなければ悪い端。先読み H_q フレーム、しきい値 c。
  //   enabled のとき、keep_nominal の「替えない」には E(見積もった模型)と Q の両方の合格が要る(v3 は E だけ)
  worst: Object.freeze({enabled: true, frictionCircle: true, frictionMultiplier: .85, speedGain: 1.04, brakeUnknown: .7, H: 45, c: .07}),
  // v4: 悲観の模型 P にも摩擦円・前進のゲインを入れる
  pess_fc: false, pess_speedGain: 1,
  // v4: すくみをほどく(unstick)。止まっていて(|v| < 0.05)合格が「速さ 0」だけのフレームが after フレーム続いたら、
  //   予備の操縦の舵と舵の組 × 速さ speeds の「はう」候補を、ゆるめたしきい値 c(見積もった模型・Q とも)で調べ、合格の中で予備の操縦の舵に一番近いものを使う
  unstick: Object.freeze({enabled: false, after: 15, speeds: Object.freeze([.5, 1.0]), c: .05, hold: 90, exit_margin: .01}),
  steer_set: Object.freeze([-.52, -.26, 0, .26, .52])});
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const approach = (v, t, d) => v + clamp(t - v, -d, d);
const finite = v => typeof v === 'number' && Number.isFinite(v);

// sim-core の step() の 1 フレーム分(小区間の積分)。衝突・ゲートは見ない(すき間だけを呼ぶ側が測る)
// config は sim-core の normalizedConfig と同じ値(snapshot.config)
export function createCarModel(geometry, config) {
  const c = {...config};
  const wb = Math.min(c.wheelbase ?? .26, c.vehicleLength ?? .42), dt = c.dt ?? 1 / 30, maxSpeed = c.maxSpeed, maxSteer = c.maxSteer;
  const substep = Math.min(.01, dt), L = c.vehicleLength, W = c.vehicleWidth;
  const vehicleRadius = Math.hypot(L, W) / 2, maxRate = maxSpeed + vehicleRadius * maxSpeed / wb * Math.tan(maxSteer);
  const realDrive = c.longitudinalModel === 'real-drive-response-estimate-v1';
  const fm = c.frictionMultiplier ?? 1;
  // st = {x, z, yaw, speed, steering}(書き換える)。m = {steerRate, brakeDeceleration, steerGain, frictionMultiplier}(上書き)。poses があれば小区間の姿勢を x,z,yaw の順に書く
  // 戻り値: このフレームで車体の点が動ける距離の上限 D と小区間の数
  function stepFrame(st, steeringCmd, speedCmd, m = null, poses = null, remainingTime = dt) {
    const steerRate = m?.steerRate ?? c.steerRate, brake = m?.brakeDeceleration ?? c.brakeDeceleration, sg = m?.steerGain ?? 1, fmm = m?.frictionMultiplier ?? fm;
    const vg = m?.speedGain ?? 1, fcOn = !!m?.frictionCircle;
    const steering = finite(steeringCmd) ? clamp(steeringCmd * sg, -maxSteer, maxSteer) : 0;
    const speed = finite(speedCmd) ? clamp(speedCmd * vg, -maxSpeed, maxSpeed) : 0;
    let elapsed = 0, D = 0, n = 0;
    while (elapsed < remainingTime - 1e-12) {
      const h = Math.min(substep, .012 / maxRate, remainingTime - elapsed);
      const mu = surfaceMu(geometry, st.x, st.z, fmm);
      let nextSpeed;
      if (realDrive) {
        const rate = st.speed >= 0 && speed > st.speed ? Math.min(c.driveAccelerationCap, mu * 9.80665, (speed - st.speed) / c.driveTau) : brake;
        nextSpeed = approach(st.speed, speed, rate * h);
      } else nextSpeed = approach(st.speed, speed, c.acceleration * h);
      const nextSteering = approach(st.steering, steering, steerRate * h);
      let rawRate = nextSpeed / wb * Math.tan(nextSteering);
      let limit = mu * 9.80665 / Math.max(Math.abs(nextSpeed), .01);
      let yawRate = clamp(rawRate, -limit, limit);
      if (fcOn) {
        // 摩擦円(fast_core_modes.mjs の置き換えと同じ式): 縦の加減速を √((μg)² − a_lat²) まで
        const aLat = Math.abs(nextSpeed * yawRate), avail = Math.sqrt(Math.max(0, (mu * 9.80665) ** 2 - aLat ** 2)), aLong = (nextSpeed - st.speed) / h;
        if (Math.abs(aLong) > avail) {
          nextSpeed = st.speed + Math.sign(aLong) * avail * h;
          rawRate = nextSpeed / wb * Math.tan(nextSteering); limit = mu * 9.80665 / Math.max(Math.abs(nextSpeed), .01); yawRate = clamp(rawRate, -limit, limit);
        }
      }
      const yawMid = st.yaw + yawRate * h / 2;
      st.x = st.x + nextSpeed * Math.cos(yawMid) * h; st.z = st.z - nextSpeed * Math.sin(yawMid) * h;
      st.yaw = wrap(st.yaw + yawRate * h); st.speed = nextSpeed; st.steering = nextSteering;
      D += (Math.abs(nextSpeed) + Math.abs(yawRate) * vehicleRadius) * h;
      if (poses) { poses[3 * n] = st.x; poses[3 * n + 1] = st.z; poses[3 * n + 2] = st.yaw; }
      n++;
      elapsed += h;
    }
    return {D, n, elapsed};
  }
  return {stepFrame, config: c, wheelbase: wb, maxSteer, maxSpeed, L, W, dt, maxSubsteps: Math.ceil(dt / Math.min(substep, .012 / maxRate)) + 2};
}

// 近くの障害物だけの形(すき間の計算を速くする。外枠は全部残す)
function localGeometry(g, x, z, radius) {
  const near = (cx, cz, reach) => Math.hypot(cx - x, cz - z) - reach <= radius;
  return {segments: g.segments.filter(s => near(s.cx, s.cz, s.reach)), circles: g.circles.filter(s => near(s.x, s.z, s.r)),
    rects: g.rects.filter(s => near(s.x, s.z, s.reach)), bounds: g.bounds, surfaces: g.surfaces, boundSegments: g.boundSegments};
}

export function createShield({geometry, config, params = {}}) {
  const P = {...SHIELD_DEFAULTS, ...params, pessimistic: {...SHIELD_DEFAULTS.pessimistic, ...(params.pessimistic ?? {})},
    identify: {...SHIELD_DEFAULTS.identify, ...(params.identify ?? {})},
    governor: {...SHIELD_DEFAULTS.governor, ...(params.governor ?? {}), brake_probe: {...SHIELD_DEFAULTS.governor.brake_probe, ...(params.governor?.brake_probe ?? {})}},
    worst: {...SHIELD_DEFAULTS.worst, ...(params.worst ?? {})}, unstick: {...SHIELD_DEFAULTS.unstick, ...(params.unstick ?? {})}};
  let stallRun = 0, unstickUntil = -1;
  const model = createCarModel(geometry, config);
  const L = model.L, W = model.W, wb = model.wheelbase, maxSteer = model.maxSteer, dt = model.dt;
  const poses = new Float64Array(3 * model.maxSubsteps);
  const proj = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0}, pt = {x: 0, z: 0, tx: 0, tz: 0};
  const st = {x: 0, z: 0, yaw: 0, speed: 0, steering: 0};
  let geom = geometry;
  // ---- 走りながらの見積もり(v3)----
  // 返した指令の列 C[f](フレーム f で返した指令)と snapshot の δ_f・v_f。遅延 d の仮説では、フレーム f−1 → f で sim が使った指令は ゲイン × C[f−1−d](始めの d フレームは中立 0)
  const ID = P.identify, TOL = ID.tol;
  const nominalM = {steerRate: model.config.steerRate, brakeDeceleration: model.config.brakeDeceleration, steerGain: 1};
  let idLast = -1, idPrev = null, histS = [], histV = [], rMax = 0, broken = false, hyps = null;
  function idReset() {
    idLast = -1; idPrev = null; histS = []; histV = []; rMax = 0; broken = false;
    hyps = ID.delays.map(d => ({d, ok: true, shortMin: Infinity, shortMax: -Infinity, gLo: Infinity, gHi: -Infinity, bShort: Infinity, why: null}));
  }
  idReset();
  const cmdAt = (arr, k) => (k >= 0 && k < arr.length ? arr[k] : 0);
  function idObserve(state) {
    const f = state.frame;
    if (!ID.enabled || !Number.isInteger(f)) return;
    if (f === 0 || f <= idLast) { if (f === 0) idReset(); }
    if (idPrev && f === idLast + 1) {
      const d0 = idPrev.steering, d1 = state.steering, dd = d1 - d0, ad = Math.abs(dd);
      rMax = Math.max(rMax, ad / dt);
      const v0 = idPrev.speed, v1 = state.speed, dv = v1 - v0;
      for (const h of hyps) {
        if (!h.ok) continue;
        const k = f - 1 - h.d, c = cmdAt(histS, k), cv = cmdAt(histV, k);
        const t1 = clamp(ID.gain[0] * c, -maxSteer, maxSteer), t2 = clamp(ID.gain[1] * c, -maxSteer, maxSteer);
        const lo = Math.min(t1, t2), hi = Math.max(t1, t2);
        const inside = d1 >= lo - TOL && d1 <= hi + TOL, sat = ad / dt >= rMax - 1e-6;
        if (!inside) {
          // 届かなかった: 目標の側へ、舵の速さいっぱい(= 見た中の最大)で動いているはず。そうでなければこの仮説は合わない
          const toward = d1 > hi ? (d0 > d1 + TOL) : (d0 < d1 - TOL);
          if (!toward || !sat) { h.ok = false; h.why = `frame ${f}: 舵の動きが遅延 ${h.d} の指令と合わない`; continue; }
          h.shortMin = Math.min(h.shortMin, ad / dt); h.shortMax = Math.max(h.shortMax, ad / dt);
        } else if (!sat && Math.abs(c) > .02 && Math.abs(d1) < maxSteer - 1e-6) {
          // 確かに届いた(舵の速さいっぱいより小さく動いた): δ = ゲイン × c
          const g = d1 / c; h.gLo = Math.min(h.gLo, g); h.gHi = Math.max(h.gHi, g);
        }
        // 届かなかったフレームは全部同じ |Δδ|/dt(= 舵の速さ)で、それは見た中の最大(舵の速さの下限)以上
        if (h.shortMin < rMax - 1e-6 || h.shortMax - h.shortMin > 1e-6) { h.ok = false; h.why = `frame ${f}: 届かなかったフレームの舵の速さがそろわない`; continue; }
        if (h.gHi - h.gLo > 1e-5) { h.ok = false; h.why = `frame ${f}: 舵のゲインがそろわない`; continue; }
        // ブレーキ: 目標の速さ(ゲイン 0.9〜1.1 のどれでも)より上で止まったフレームは、ブレーキいっぱい。摩擦円では曲がりながらのブレーキが小さくなるので、
        // そういうフレームの中の最大を使う(どのフレームの減速も本当のブレーキ以下なので、最大は本当の値以下 = 甘くならない)
        if (dv < -TOL && v1 > Math.max(ID.gain[0] * cv, ID.gain[1] * cv) + TOL && v0 >= 0) h.bShort = Number.isFinite(h.bShort) ? Math.max(h.bShort, -dv / dt) : -dv / dt;
      }
      if (!hyps.some(h => h.ok)) broken = true;
    }
    idPrev = {steering: state.steering, speed: state.speed}; idLast = f;
  }
  function idRecord(state, cmd) {
    if (!ID.enabled || !Number.isInteger(state.frame)) return;
    histS[state.frame] = cmd.steering; histV[state.frame] = cmd.speed;
  }
  // 今の見積もり: 残った仮説のうち遅延が一番大きいもの。舵の速さは届かなかったフレームがあればその値、無ければ max(見た最大, 標準)
  function identified() {
    if (!ID.enabled) return {m: nominalM, delay: 0, known: false, d: null};
    const ok = hyps.filter(h => h.ok);
    const h = ok.length ? ok[ok.length - 1] : null;
    if (!h) return {m: {steerRate: Math.max(rMax, 1e-3) || nominalM.steerRate, brakeDeceleration: nominalM.brakeDeceleration, steerGain: 1}, delay: Math.max(...ID.delays), known: false, broken: true, d: null};
    const r = Number.isFinite(h.shortMin) ? h.shortMin : Math.max(rMax, nominalM.steerRate);
    const g = Number.isFinite(h.gLo) ? (h.gLo + h.gHi) / 2 : 1;
    const b = Number.isFinite(h.bShort) ? h.bShort : nominalM.brakeDeceleration;
    // v4.1: ブレーキを見た仮説のうちの最小(どのフレームの減速も本当のブレーキ以下なので、最小は甘くならない)
    const bs = ok.filter(x => Number.isFinite(x.bShort)).map(x => x.bShort);
    return {m: {steerRate: r, brakeDeceleration: b, steerGain: g}, delay: h.d, known: Number.isFinite(h.shortMin), d: ok.map(x => x.d), brakeSeen: Number.isFinite(h.bShort),
      brakeSeenAny: bs.length > 0, bAny: bs.length ? Math.min(...bs) : null};
  }
  // 遅延 d のときの「まだ使われていない指令」(古い順)。列がそろっていなければ前の指令をくり返す
  const pendS = new Float64Array(4), pendV = new Float64Array(4);
  function pending(state, prevCmd, delay) {
    const f = state.frame, aligned = Number.isInteger(f) && histS.length === f && ID.enabled;
    for (let i = 0; i < delay; i++) {
      const k = f - delay + i;
      if (aligned) { pendS[i] = cmdAt(histS, k); pendV[i] = cmdAt(histV, k); }
      else { pendS[i] = prevCmd.steering; pendV[i] = prevCmd.speed; }
    }
  }
  // 予備の操縦(状態 st・進み s の近く)。戻り値の s は射影の結果(次のフレームの探索の起点)
  const BK = P.backups;
  const BK0 = BK.find(b => !b.hold);
  function backup(state, sGuess, out, b = BK0) {
    projectWindow(state.x, state.z, sGuess - .5, sGuess + 1.5, proj);
    const sT = proj.s + b.la + b.k * Math.abs(state.speed), e = tableAt('emid', sT);
    loopPointAt(sT, pt);
    const tx = pt.x + e * pt.tz, tz = pt.z - e * pt.tx, dx = tx - state.x, dz = tz - state.z;
    const alpha = wrap(Math.atan2(-dz, dx) - state.yaw), ld = Math.max(Math.hypot(dx, dz), .3);
    out.steering = clamp(Math.atan2(2 * wb * Math.sin(alpha), ld), -maxSteer, maxSteer); out.speed = 0; out.s = proj.s;
    return out;
  }
  const cmdTmp = {steering: 0, speed: 0, s: 0};
  const qS = new Float64Array(4), qV = new Float64Array(4);
  // 1 つの候補の先読み。m = 模型、delay = 遅延のフレーム数(0〜2。まだ使われていない指令は pendS/pendV)、thr = しきい値、early = しきい値を割ったら打ち切る、b = 予備の操縦
  function rollout(state0, s0, first, m, delay, thr, early, b, H = P.H) {
    st.x = state0.x; st.z = state0.z; st.yaw = state0.yaw; st.speed = state0.speed; st.steering = state0.steering;
    let s = s0, c0 = clearance(geom, st.x, st.z, st.yaw, L, W, .5), minC = c0;
    for (let i = 0; i < delay; i++) { qS[i] = pendS[i]; qV[i] = pendV[i]; }
    for (let j = 0; j <= H; j++) {
      let iS, iV;
      if (j === 0) { iS = first.steering; iV = first.speed; }
      else if (b.hold) { iS = first.steering; iV = 0; }
      else { backup(st, s, cmdTmp, b); s = cmdTmp.s; iS = cmdTmp.steering; iV = cmdTmp.speed; }
      let aS = iS, aV = iV;
      if (delay) { aS = qS[0]; aV = qV[0]; for (let i = 1; i < delay; i++) { qS[i - 1] = qS[i]; qV[i - 1] = qV[i]; } qS[delay - 1] = iS; qV[delay - 1] = iV; }
      const r = model.stepFrame(st, aS, aV, m, poses);
      const c1 = clearance(geom, st.x, st.z, st.yaw, L, W, .5);
      if (Math.max(c0, c1) - r.D < thr) {
        for (let k = 0; k < r.n; k++) { const ck = clearance(geom, poses[3 * k], poses[3 * k + 1], poses[3 * k + 2], L, W, .5); if (ck < minC) minC = ck; }
      } else if (c1 < minC) minC = c1;
      c0 = c1;
      if (early && minC < thr) return {min: minC, pass: false};
    }
    return {min: minC, pass: minC >= thr};
  }
  // このフレームの 2 つの模型(見積もった模型 = v2 の「標準の模型」の代わり、悲観の模型)
  let MOD = null;
  function models(state, prevCmd) {
    const idn = identified();
    const pm = {steerRate: Math.min(P.pessimistic.steerRate, idn.m.steerRate), brakeDeceleration: Math.min(P.pessimistic.brakeDeceleration, idn.m.brakeDeceleration),
      steerGain: idn.m.steerGain, frictionMultiplier: P.pessimistic.frictionMultiplier ?? model.config.frictionMultiplier ?? 1,
      frictionCircle: !!P.pess_fc, speedGain: P.pess_speedGain ?? 1};
    const W = P.worst;
    const qm = W.enabled ? {steerRate: idn.m.steerRate, steerGain: idn.m.steerGain,
      brakeDeceleration: idn.brakeSeen ? idn.m.brakeDeceleration : Math.min(W.brakeUnknown, idn.m.brakeDeceleration),
      frictionMultiplier: (model.config.frictionMultiplier ?? 1) * W.frictionMultiplier, frictionCircle: !!W.frictionCircle, speedGain: W.speedGain} : null;
    MOD = {id: idn, nomM: idn.m, nomD: Math.min(2, idn.delay), pesM: pm, pesD: Math.min(2, Math.max(P.pessimistic.delay, idn.delay)), qM: qm, state, prevCmd};
    return MOD;
  }
  function check1(state, s, prevCmd, cand, early, bk, T = null) {
    const M = MOD && MOD.state === state && MOD.prevCmd === prevCmd ? MOD : models(state, prevCmd);
    const cN = T?.n ?? P.c_nominal, cQ = T?.q ?? P.worst.c, cP = T?.p ?? P.c_pessimistic;
    pending(state, prevCmd, M.nomD);
    const a = rollout(state, s, cand, M.nomM, M.nomD, cN, early, bk);
    if (early && !a.pass) return {pass: false, passNominal: false, nominal: a.min, pessimistic: null, min: a.min - cN};
    let q = null;
    if (M.qM) {
      pending(state, prevCmd, M.nomD);
      q = rollout(state, s, cand, M.qM, M.nomD, cQ, early, bk, P.worst.H);
      if (early && !q.pass) return {pass: false, passNominal: false, nominal: a.min, worst: q.min, pessimistic: null, min: Math.min(a.min - cN, q.min - cQ)};
    }
    pending(state, prevCmd, M.pesD);
    const b = rollout(state, s, cand, M.pesM, M.pesD, cP, early, bk);
    const qOk = q ? q.pass : true, qMin = q ? q.min - cQ : Infinity;
    return {pass: a.pass && qOk && b.pass, passNominal: a.pass && qOk, nominal: a.min, worst: q ? q.min : null, pessimistic: b.min,
      min: Math.min(a.min - cN, qMin, b.min - cP)};
  }
  // 予備の操縦の組のどれか 1 つで安全が続けば合格(どれも合格しなければ、しきい値との差が一番大きいもの)
  function check(state, s, prevCmd, cand, early = true, T = null) {
    let best = null, nom = false;
    for (let i = 0; i < BK.length; i++) {
      const r = check1(state, s, prevCmd, cand, early, BK[i], T);
      r.backup = i; nom = nom || r.passNominal;
      if (r.pass) return r;
      if (!best || r.min > best.min) best = r;
    }
    best.passNominal = nom;
    return best;
  }
  // state = snapshot.state、s = 追跡器の s、prevCmd = 前のフレームに返した指令、proposal = 方策の指令
  function filter(state, s, prevCmd, proposal) {
    idObserve(state);
    const r = filter0(state, s, prevCmd, proposal);
    idRecord(state, r.cmd);
    r.ident = MOD ? {sr: MOD.nomM.steerRate, d: MOD.nomD, g: MOD.nomM.steerGain, b: MOD.nomM.brakeDeceleration, known: MOD.id.known, hyp: MOD.id.d, broken: !!MOD.id.broken} : null;
    return r;
  }
  // 速さの見張りの上限(見積もりが確かなときだけ。標準の物理では Infinity)
  function speedCap(state) {
    const GV = P.governor;
    if (!GV.enabled || !MOD || !(state.frame >= GV.min_frame)) return Infinity;
    const known = MOD.id.known;
    if (!known && !GV.brake_without_known) return Infinity;
    const T = MOD.nomD * dt + GV.theta_rad / MOD.nomM.steerRate, Tn = GV.theta_rad / nominalM.steerRate;
    // ブレーキが標準より弱いと分かったら(止まれる距離 v²/2b を標準と同じにする)√(b/b_標準) も掛ける
    // v4.1: 舵の速さが分からなくても、ブレーキを見た仮説があればその最小で(brake_without_known)
    let b = known ? MOD.nomM.brakeDeceleration : nominalM.brakeDeceleration;
    if (GV.brake_without_known && MOD.id.brakeSeenAny) b = Math.min(b, MOD.id.bAny);
    const kb = GV.brake && b < nominalM.brakeDeceleration * (1 - 1e-9) ? Math.sqrt(b / nominalM.brakeDeceleration) : 1;
    // v4: ブレーキが弱いと分かったら、見積もれない μ・摩擦円のぶんの係数 weak_extra を掛ける(既定 1 = v3 と同じ)
    const kx = kb < 1 ? (GV.weak_extra ?? 1) : 1;
    const k = (known && T > Tn * (1 + 1e-9) ? Tn / T : 1) * kb * kx;
    return k < 1 ? GV.v_max * k : Infinity;
  }
  let lastProbe = -1e9;
  function filter0(state, s, prevCmd, proposal0) {
    const r = filter0a(state, s, prevCmd, proposal0);
    // v4.1: ブレーキの見積もりの探り(まだどの仮説もブレーキを見ていない・速さ ≥ v_min・提案をそのまま通すフレームだけ)
    const GV = P.governor, PR = GV.brake_probe;
    if (state.frame === 0) lastProbe = -1e9;
    if (GV.enabled && PR && PR.enabled && !r.replaced && MOD && !MOD.id.brakeSeenAny && state.frame >= GV.min_frame && state.speed >= PR.v_min
      && state.frame - lastProbe >= PR.every) {
      lastProbe = state.frame;
      return {...r, cmd: {steering: r.cmd.steering, speed: Math.min(r.cmd.speed, PR.factor * state.speed)}, replaced: true,
        fallback: (r.fallback ? r.fallback + '+' : '') + 'brake_probe'};
    }
    return r;
  }
  function filter0a(state, s, prevCmd, proposal0) {
    const v = Math.abs(state.speed);
    geom = localGeometry(geometry, state.x, state.z, (v + 1.5) * (P.H + 2) * model.dt + 1.0);
    models(state, prevCmd);
    const cap = speedCap(state);
    if (proposal0.speed > cap) {
      const proposal = {steering: proposal0.steering, speed: cap};
      const r = filter1(state, s, prevCmd, proposal);
      if (!r.replaced) return {...r, cmd: {steering: proposal.steering, speed: proposal.speed}, replaced: true, fallback: (r.fallback ? r.fallback + '+' : '') + 'governor', cap};
      return {...r, cap};
    }
    return filter1(state, s, prevCmd, proposal0);
  }
  function filter1(state, s, prevCmd, proposal) {
    if (Math.abs(state.speed) >= .05) stallRun = 0;
    if (state.frame === 0) { stallRun = 0; unstickUntil = -1; }
    const U = P.unstick;
    // unstick の続き: 始めてから hold フレームの間、今のすき間が c_nominal + exit_margin に戻るまでは、ゆるめたしきい値で検査する
    if (U.enabled && state.frame <= unstickUntil) {
      const cNow = clearance(geometry, state.x, state.z, state.yaw, L, W, .5);
      if (cNow < P.c_nominal + U.exit_margin) {
        const T = {n: U.c, q: U.c, p: Math.min(U.c, P.c_pessimistic)};
        const r0 = check(state, s, prevCmd, proposal, true, T);
        if (r0.pass) return {cmd: {steering: proposal.steering, speed: proposal.speed}, replaced: true, checked: 1, predicted: r0, fallback: 'unstick'};
        const bk = backup(state, s, {steering: 0, speed: 0, s: 0});
        const steers = [bk.steering, ...P.steer_set].sort((a, b) => Math.abs(a - bk.steering) - Math.abs(b - bk.steering));
        let checked = 1;
        for (const v of [...U.speeds, 0]) for (const d of steers) {
          checked++;
          const r = check(state, s, prevCmd, {steering: d, speed: v}, true, T);
          if (r.pass) return {cmd: {steering: d, speed: v}, replaced: true, checked, predicted: r, fallback: 'unstick'};
        }
      } else unstickUntil = -1;
    }
    const first = check(state, s, prevCmd, proposal);
    if (first.pass) return {cmd: {steering: proposal.steering, speed: proposal.speed}, replaced: false, checked: 1, predicted: first};
    if (P.keep_nominal && first.passNominal) return {cmd: {steering: proposal.steering, speed: proposal.speed}, replaced: false, checked: 1, predicted: first, fallback: 'proposal_nominal'};
    const bk = backup(state, s, {steering: 0, speed: 0, s: 0});
    const cands = [{steering: bk.steering, speed: 0, backup: true}];
    for (const d of P.steer_set) for (const sp of [0, proposal.speed]) cands.push({steering: d, speed: sp});
    cands.sort((p, q) => (Math.abs(p.steering - proposal.steering) - Math.abs(q.steering - proposal.steering)) || (Math.abs(p.speed - proposal.speed) - Math.abs(q.speed - proposal.speed)));
    let checked = 1;
    const passed = [];
    let firstNom = null;
    for (const c of cands) {
      checked++;
      const r = check(state, s, prevCmd, c);
      if (!r.pass && r.passNominal && !firstNom) firstNom = {c, r};
      if (r.pass) {
        // 止まっていて、合格が「速さ 0」だけのとき(すくみ): 予備の操縦の舵に一番近いもの(タイヤを逃げる向きに切っておく。build.md)
        if (Math.abs(state.speed) < .05 && c.speed === 0) { passed.push({c, r}); continue; }
        return {cmd: {steering: c.steering, speed: c.speed}, replaced: true, checked, predicted: r, fallback: 'pass'};
      }
    }
    if (passed.length) {
      passed.sort((p, q) => Math.abs(p.c.steering - bk.steering) - Math.abs(q.c.steering - bk.steering));
      stallRun++;
      const U = P.unstick;
      if (U.enabled && stallRun > U.after) {
        // すくみをほどく: ゆるめたしきい値で「はう」候補(予備の操縦の舵に近い順)
        const T = {n: U.c, q: U.c, p: Math.min(U.c, P.c_pessimistic)};
        const steers = [bk.steering, ...P.steer_set].sort((a, b) => Math.abs(a - bk.steering) - Math.abs(b - bk.steering));
        for (const v of U.speeds) for (const d of steers) {
          checked++;
          const r = check(state, s, prevCmd, {steering: d, speed: v}, true, T);
          if (r.pass) { unstickUntil = state.frame + U.hold; return {cmd: {steering: d, speed: v}, replaced: true, checked, predicted: r, fallback: 'unstick'}; }
        }
      }
      return {cmd: {steering: passed[0].c.steering, speed: 0}, replaced: true, checked, predicted: passed[0].r, fallback: 'stopped_turn_to_backup'};
    }
    if (P.keep_nominal && firstNom) return {cmd: {steering: firstNom.c.steering, speed: firstNom.c.speed}, replaced: true, checked, predicted: firstNom.r, fallback: 'nominal_pass'};
    // どれも合格しない: 予想の最小すき間(しきい値との差)が一番大きい候補。方策の提案も同じ物差しで並べる(同じなら提案を残す。build.md)
    let best = {c: proposal, r: check(state, s, prevCmd, proposal, false)};
    for (const c of cands) { const r = check(state, s, prevCmd, c, false); checked++; if (r.min > best.r.min + 1e-9) best = {c, r}; }
    if (best.c === proposal) return {cmd: {steering: proposal.steering, speed: proposal.speed}, replaced: false, checked, predicted: best.r, fallback: 'proposal_best'};
    return {cmd: {steering: best.c.steering, speed: best.c.speed}, replaced: true, checked, predicted: best.r, fallback: 'max_clearance'};
  }
  return {filter, check, model, params: P, identified, get identState() { return {rMax, broken, hyps: hyps.map(h => ({...h}))}; }};
}
