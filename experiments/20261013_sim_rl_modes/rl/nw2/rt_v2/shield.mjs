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
// 計算を減らす工夫(判定は変わらない): フレームの両端のすき間 c0・c1 と、そのフレームで車体の点が動ける距離の上限 D から、
//   max(c0, c1) − D ≥ しきい値 ならそのフレームの小区間はどれもしきい値以上(すき間は剛体の動きに対して 1-リプシッツ)。そうでなければ小区間ごとに測る。
import {clearance, surfaceMu, projectWindow, loopPointAt, G} from './features.mjs';
import {tableAt} from './line_profiles.mjs';

export const SHIELD_VERSION = 'jf-shield/v2';
export const SHIELD_DEFAULTS = Object.freeze({version: SHIELD_VERSION, H: 24, c_nominal: .07, c_pessimistic: .05,
  pessimistic: Object.freeze({steerRate: .85, delay: 1, brakeDeceleration: .8}),
  // 予備の操縦の組(la + k·|v| m 先を追う pure pursuit + ブレーキ、hold = 提案の舵を保ってブレーキ)。どれか 1 つで合格すればよい
  backups: Object.freeze([Object.freeze({la: .6, k: .2}), Object.freeze({la: .6, k: 0}), Object.freeze({hold: true})]),
  keep_nominal: true,   // 提案が標準の模型で合格なら替えない
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
  // st = {x, z, yaw, speed, steering}(書き換える)。m = {steerRate, brakeDeceleration}(上書き)。poses があれば小区間の姿勢を x,z,yaw の順に書く
  // 戻り値: このフレームで車体の点が動ける距離の上限 D と小区間の数
  function stepFrame(st, steeringCmd, speedCmd, m = null, poses = null, remainingTime = dt) {
    const steerRate = m?.steerRate ?? c.steerRate, brake = m?.brakeDeceleration ?? c.brakeDeceleration;
    const steering = finite(steeringCmd) ? clamp(steeringCmd, -maxSteer, maxSteer) : 0;
    const speed = finite(speedCmd) ? clamp(speedCmd, -maxSpeed, maxSpeed) : 0;
    let elapsed = 0, D = 0, n = 0;
    while (elapsed < remainingTime - 1e-12) {
      const h = Math.min(substep, .012 / maxRate, remainingTime - elapsed);
      const mu = surfaceMu(geometry, st.x, st.z, fm);
      let nextSpeed;
      if (realDrive) {
        const rate = st.speed >= 0 && speed > st.speed ? Math.min(c.driveAccelerationCap, mu * 9.80665, (speed - st.speed) / c.driveTau) : brake;
        nextSpeed = approach(st.speed, speed, rate * h);
      } else nextSpeed = approach(st.speed, speed, c.acceleration * h);
      const nextSteering = approach(st.steering, steering, steerRate * h);
      const rawRate = nextSpeed / wb * Math.tan(nextSteering);
      const limit = mu * 9.80665 / Math.max(Math.abs(nextSpeed), .01);
      const yawRate = clamp(rawRate, -limit, limit);
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
  const P = {...SHIELD_DEFAULTS, ...params, pessimistic: {...SHIELD_DEFAULTS.pessimistic, ...(params.pessimistic ?? {})}};
  const model = createCarModel(geometry, config);
  const L = model.L, W = model.W, wb = model.wheelbase, maxSteer = model.maxSteer;
  const poses = new Float64Array(3 * model.maxSubsteps);
  const proj = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0}, pt = {x: 0, z: 0, tx: 0, tz: 0};
  const st = {x: 0, z: 0, yaw: 0, speed: 0, steering: 0};
  let geom = geometry;
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
  // 1 つの候補の先読み。m = 模型(null = 標準)、delay = 遅延のフレーム数(0/1)、thr = しきい値、early = しきい値を割ったら打ち切る、b = 予備の操縦
  function rollout(state0, s0, prevCmd, first, m, delay, thr, early, b) {
    st.x = state0.x; st.z = state0.z; st.yaw = state0.yaw; st.speed = state0.speed; st.steering = state0.steering;
    let s = s0, c0 = clearance(geom, st.x, st.z, st.yaw, L, W, .5), minC = c0;
    let pendS = prevCmd.steering, pendV = prevCmd.speed;
    for (let j = 0; j <= P.H; j++) {
      let iS, iV;
      if (j === 0) { iS = first.steering; iV = first.speed; }
      else if (b.hold) { iS = first.steering; iV = 0; }
      else { backup(st, s, cmdTmp, b); s = cmdTmp.s; iS = cmdTmp.steering; iV = cmdTmp.speed; }
      let aS = iS, aV = iV;
      if (delay) { aS = pendS; aV = pendV; pendS = iS; pendV = iV; }
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
  const pessM = {steerRate: P.pessimistic.steerRate, brakeDeceleration: P.pessimistic.brakeDeceleration};
  function check1(state, s, prevCmd, cand, early, bk) {
    const a = rollout(state, s, prevCmd, cand, null, 0, P.c_nominal, early, bk);
    if (early && !a.pass) return {pass: false, passNominal: false, nominal: a.min, pessimistic: null, min: a.min - P.c_nominal};
    const b = rollout(state, s, prevCmd, cand, pessM, P.pessimistic.delay, P.c_pessimistic, early, bk);
    return {pass: a.pass && b.pass, passNominal: a.pass, nominal: a.min, pessimistic: b.min, min: Math.min(a.min - P.c_nominal, b.min - P.c_pessimistic)};
  }
  // 予備の操縦の組のどれか 1 つで安全が続けば合格(どれも合格しなければ、しきい値との差が一番大きいもの)
  function check(state, s, prevCmd, cand, early = true) {
    let best = null, nom = false;
    for (let i = 0; i < BK.length; i++) {
      const r = check1(state, s, prevCmd, cand, early, BK[i]);
      r.backup = i; nom = nom || r.passNominal;
      if (r.pass) return r;
      if (!best || r.min > best.min) best = r;
    }
    best.passNominal = nom;
    return best;
  }
  // state = snapshot.state、s = 追跡器の s、prevCmd = 前のフレームに返した指令、proposal = 方策の指令
  function filter(state, s, prevCmd, proposal) {
    const v = Math.abs(state.speed);
    geom = localGeometry(geometry, state.x, state.z, (v + 1.5) * (P.H + 2) * model.dt + 1.0);
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
      return {cmd: {steering: passed[0].c.steering, speed: 0}, replaced: true, checked, predicted: passed[0].r, fallback: 'stopped_turn_to_backup'};
    }
    if (P.keep_nominal && firstNom) return {cmd: {steering: firstNom.c.steering, speed: firstNom.c.speed}, replaced: true, checked, predicted: firstNom.r, fallback: 'nominal_pass'};
    // どれも合格しない: 予想の最小すき間(しきい値との差)が一番大きい候補。方策の提案も同じ物差しで並べる(同じなら提案を残す。build.md)
    let best = {c: proposal, r: check(state, s, prevCmd, proposal, false)};
    for (const c of cands) { const r = check(state, s, prevCmd, c, false); checked++; if (r.min > best.r.min + 1e-9) best = {c, r}; }
    if (best.c === proposal) return {cmd: {steering: proposal.steering, speed: proposal.speed}, replaced: false, checked, predicted: best.r, fallback: 'proposal_best'};
    return {cmd: {steering: best.c.steering, speed: best.c.speed}, replaced: true, checked, predicted: best.r, fallback: 'max_clearance'};
  }
  return {filter, check, model, params: P};
}
