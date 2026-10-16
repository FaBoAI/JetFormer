// 上限の試験(S4)の区間判定と上限の層のフィルター(2026-10-16、シミュレーター較正)。純粋な ESM で import なし。
//
// Node(cap_test.mjs)は普通に import し、ブラウザ(calib_cap.html の較正版のコア)は build_cap_core.mjs が
// この文字列をコアに埋め込む。どちらも同じ関数で動く(protocol cap_test.governor_v1_sim「Node とブラウザで同じ実装」)。
//
//   - 区間(位置、主): 車の位置を参照ループ R に射影した進み s(features.mjs と同じ追跡: リセットで目標ゲートの前後、
//     毎フレーム前の s の −0.5〜+1.5 m。距離 0.3 m 超では前へ 3 m まで探す = deviations D21)→ protocol の zones_oracle の区間
//   - 区間(ジャイロ、副): jetformer/safety_caps.py の phase_step / combine_zone の写し(シミュレーターのヨーから)
//   - 上限の層: calib_runtime.mjs の governorStep(= safety_caps.cap_step、試験 T6)を外から受け取る(import しないため)
//
// createCapFilter(...).filter は calibControl({filter}) の口の形 ({frame, steering, speed, state, config}) → {speed}。
// state は step の始めの状態(届いた指令はアプリの遅れのあと)。舵は変えない。速さの指令を大きくはしない。

export const CAP_RUNTIME_VERSION = 'jf-cap-runtime/v1';
export const ZONES_ORACLE = Object.freeze({R1_app: [5.02, 6.206], R1_turn: [6.206, 7.456], GM: [8.237, 12.601], DE: [12.626, 16.149]});
// 区間の判定(衝突の割り当て・通過)に使う R 上の区間(section_map.json。R1 の衝突は R1 の始まりの 0.3 m 手前から)
export const SECTIONS_R = Object.freeze({HS_a: [24.235, 26.1197], HS_b: [0, 3.837], LH: [3.861, 6.181], R1: [6.206, 8.216], GM: [8.237, 12.601],
  DE: [12.626, 16.149], SW: [16.174, 24.212]});
const PROJ_BEHIND = .5, PROJ_AHEAD = 1.5, FAR_TRIGGER = .3, FAR_AHEAD = 3.0;
const DEG = 180 / Math.PI;

// ------------------------------------------------------------------ 参照ループ R(reference_loop.json の x・z・s・route・gate_s)
export function createLoop(data) {
  const n = data.n, L = data.length_m;
  const X = Float64Array.from(data.x), Z = Float64Array.from(data.z), S = Float64Array.from(data.s);
  const segLen = new Float64Array(n), tx = new Float64Array(n), tz = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n, dx = X[j] - X[i], dz = Z[j] - Z[i], len = Math.hypot(dx, dz);
    segLen[i] = (i + 1 < n ? S[j] : L) - S[i]; tx[i] = dx / len; tz[i] = dz / len;
  }
  const gateS = [...data.gate_s], route = data.route.map(p => [p[0], p[1]]);
  const segmentAt = local => { let lo = 0, hi = n - 1; while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (S[mid] <= local) lo = mid; else hi = mid - 1; } return lo; };
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  // features.mjs の projectWindow と同じ計算(周をまたいだ s の範囲 [a, b] で最も近い点。同じ距離なら s の小さいほう)
  function projectWindow(x, z, a, b, out = {s: 0, d: 0}) {
    let lap = Math.floor(a / L), i = segmentAt(a - lap * L), best = Infinity;
    for (let guard = 0; guard <= n + 1; guard++) {
      const start = lap * L + S[i], len = segLen[i];
      if (start > b) break;
      const j = (i + 1) % n, dx = X[j] - X[i], dz = Z[j] - Z[i];
      const uLo = Math.max(0, (a - start) / len), uHi = Math.min(1, (b - start) / len);
      if (uHi >= uLo) {
        const u = clamp(((x - X[i]) * dx + (z - Z[i]) * dz) / (dx * dx + dz * dz), uLo, uHi);
        const px = X[i] + dx * u, pz = Z[i] + dz * u, d2 = (x - px) ** 2 + (z - pz) ** 2;
        if (d2 < best) { best = d2; out.s = start + u * len; }
      }
      i++; if (i === n) { i = 0; lap++; }
    }
    out.d = Math.sqrt(best);
    return out;
  }
  // 回した経路(アプリの reset で開始ゲート k から並べ直したもの)が、元の route の何番から始まるか(features.mjs と同じ)
  function routeOffset(r) {
    const m = route.length;
    if (!Array.isArray(r) || r.length !== m) throw new Error('cap_runtime: 経路のゲート数が標準(33)と違います');
    for (let k = 0; k < m; k++) {
      let same = true;
      for (let i = 0; i < m && same; i++) { const a = r[i], b = route[(k + i) % m]; same = a[0] === b[0] && a[1] === b[1]; }
      if (same) return k;
    }
    throw new Error('cap_runtime: 経路が標準の route を回したものではありません');
  }
  return {n, L, gateS, route, projectWindow, routeOffset};
}

// 進み s の追跡器(features.mjs の createFeatureTracker の resetProgress / advanceProgress と同じ)
export function createProgressTracker(loop, worldRoute) {
  const k = loop.routeOffset(worldRoute), m = worldRoute.length, L = loop.L;
  const targetS = Array.from({length: m}, (_, i) => loop.gateS[(k + i) % m]);
  const startS = loop.gateS[(k + m - 1) % m];
  const proj = {s: 0, d: 0};
  let s = 0;
  function reset(x, z, routeIndex = 0) {
    const i = Math.min(Math.max(0, routeIndex | 0), m - 1);
    const prev = i ? targetS[i - 1] : startS;
    let target = targetS[i]; while (target < prev) target += L;
    loop.projectWindow(x, z, prev - .5, target + .5, proj);
    s = proj.s;
    return s;
  }
  // 毎フレーム: 前の s の −0.5〜+1.5 m(features.mjs と同じ)。ただし射影の距離が 0.3 m を超えたら、前へ 3 m まで広げて
  // 0.05 m 以上近い点があればそちらにする(R の GM の大回りを通らない線で、追跡が止まって迷子になるのを防ぐ。deviations D21)
  const alt = {s: 0, d: 0};
  function advance(x, z) {
    loop.projectWindow(x, z, s - PROJ_BEHIND, s + PROJ_AHEAD, proj);
    if (proj.d > FAR_TRIGGER) {
      loop.projectWindow(x, z, s - PROJ_BEHIND, s + FAR_AHEAD, alt);
      if (alt.d < proj.d - .05) { proj.s = alt.s; proj.d = alt.d; jumps++; }
    }
    s = proj.s;
    return s;
  }
  let jumps = 0;
  return {reset, advance, get s() { return s; }, get d() { return proj.d; }, get jumps() { return jumps; }, routeOffset: k};
}

export function zoneOfS(s, L, zones = ZONES_ORACLE) {
  const u = ((s % L) + L) % L;
  for (const k of Object.keys(zones)) { const [a, b] = zones[k]; if (u >= a && u < b) return k; }
  return 'none';
}

// ------------------------------------------------------------------ ジャイロの位相(jetformer/safety_caps.py の写し、v1.1 の既定)
export const PHASE_PARAMS = Object.freeze({still_speed: .025, moving_speed: .1, max_dt: .2, gz_clip: 990, turn_on: -60, r1_peak: -170, r1_dpsi: -90,
  r1_dpsi_short: -40, r1_prev_left: 60, r1_prev_left_short: 90, r1_min_speed: .75, r1_min_gap_s: 5, app_psi: 300, gm_exit_psi: -20, de_s: 1.5,
  lost_s: 16, psi_max: 420, psi_min: -300, hist_s: 2, stop_speed: .3, stop_reset_s: 1, app_fallback_psi: 240});
export function phaseInitialState() {
  return {t: null, heading: 0, bias: 0, bias_n: 0, g_hist: [], h_hist: [], turn: null, anchor_t: null, anchor_psi: 0, stage: 'unknown', stage_t: 0, stop_t0: null};
}
function headingAt(hHist, tq, dflt) { let out = dflt; for (const [tt, hh] of hHist) { if (tt <= tq) out = hh; else break; } return out; }
export function phaseStep(state, t, gz, v, p = PHASE_PARAMS) {
  let bias = state.bias, biasN = state.bias_n;
  if (Math.abs(v) < p.still_speed && Math.abs(gz) < 5) { biasN = Math.min(biasN + 1, 500); bias += (gz - bias) / biasN; }
  const g = Math.max(-p.gz_clip, Math.min(p.gz_clip, gz)) - bias;
  let heading = state.heading;
  if (state.t !== null) { const dt = t - state.t; if (dt > 0 && dt <= p.max_dt && Math.abs(v) > p.moving_speed) heading += g * dt; }
  const gh = [...state.g_hist, g];
  const gs = gh.reduce((a, b) => a + b, 0) / gh.length;
  const gHist = gh.slice(-4);
  let hHist = state.h_hist;
  if (!hHist.length || t - hHist[hHist.length - 1][0] >= .05) hHist = [...hHist.filter(x => x[0] >= t - (p.hist_s + 1)), [t, heading]];
  let {anchor_t: anchorT, anchor_psi: anchorPsi, stage, stage_t: stageT} = state;
  const stopT0 = Math.abs(v) < p.stop_speed ? (state.stop_t0 !== null ? state.stop_t0 : t) : null;
  if (stopT0 !== null && t - stopT0 >= p.stop_reset_s && anchorT !== null) { anchorT = null; stage = 'unknown'; stageT = t; }
  let turn = state.turn, confirmed = false;
  if (gs < p.turn_on) {
    if (turn === null) {
      const prevLeft = heading - headingAt(hHist, t - p.hist_s, hHist[0][1]);
      turn = [t, heading, gs, prevLeft, Math.abs(v)];
    } else turn = [turn[0], turn[1], Math.min(turn[2], gs), turn[3], turn[4]];
    const [t0, psi0, gmin, prevLeft, v0] = turn;
    const dpsi = heading - psi0;
    const gapOk = anchorT === null || t0 - anchorT > p.r1_min_gap_s;
    if (gapOk && gmin <= p.r1_peak && v0 >= p.r1_min_speed && prevLeft > p.r1_prev_left
        && (dpsi <= p.r1_dpsi || (dpsi <= p.r1_dpsi_short && prevLeft > p.r1_prev_left_short))) {
      anchorT = t0; anchorPsi = psi0; stage = 'R1_turn'; stageT = t0; confirmed = true;
    }
  } else turn = null;
  const turningRight = turn !== null;
  let rel = anchorT === null ? null : heading - anchorPsi;
  if (anchorT !== null && !confirmed) {
    if (stage === 'R1_turn' && gs >= 0) { stage = 'GM'; stageT = t; }
    if (stage === 'GM' && rel >= p.gm_exit_psi) { stage = 'DE'; stageT = t; }
    if (stage === 'DE' && t - stageT >= p.de_s) { stage = 'none'; stageT = t; }
    if (stage === 'none' && rel >= p.app_psi) { stage = 'R1_app'; stageT = t; }
    if (stage === 'R1_app' && rel < p.app_fallback_psi) { anchorT = null; stage = 'unknown'; stageT = t; rel = null; }
  }
  const lost = anchorT === null || (t - anchorT > p.lost_s) || rel === null || rel > p.psi_max || rel < p.psi_min;
  const zone = lost ? 'unknown' : (stage === 'R1_app' && turningRight ? 'R1_turn' : stage);
  const next = {t, heading, bias, bias_n: biasN, g_hist: gHist, h_hist: hHist, turn, anchor_t: anchorT, anchor_psi: anchorPsi, stage, stage_t: stageT, stop_t0: stopT0};
  return {state: next, out: {zone, rel_psi: rel, turning_right: turningRight, r1_confirmed: confirmed, anchor_t: anchorT, gz_smooth: gs}};
}
// safety_caps.combine_zone(予測ステアの合図 (b) はシミュレーターの運転手には無いので false)
export function combineZone(zoneGyro, turningRight, predSignal = false, v = null, predMinSpeed = 1.0) {
  if (zoneGyro === 'R1_turn' || zoneGyro === 'GM' || zoneGyro === 'DE') return zoneGyro;
  if (predSignal && v !== null && v < predMinSpeed) predSignal = false;
  const signal = zoneGyro === 'R1_app' || predSignal;
  if (signal && turningRight) return 'R1_turn';
  if (signal) return 'R1_app';
  return zoneGyro;
}

// ------------------------------------------------------------------ 上限の層のフィルター
// loop: createLoop() の結果。route: アプリの activeWorld.route(回した経路)。governorStep・governorInitialState・params: calib_runtime.mjs のもの。
// zoneMode: 'oracle'(位置)/ 'gyro'(ジャイロの位相)。record: フレームごとの記録を残す。
export function createCapFilter({loop, route, governorStep, governorInitialState, params, zoneMode = 'oracle', zones = ZONES_ORACLE,
  phaseParams = PHASE_PARAMS, record = true, name = null}) {
  if (zoneMode !== 'oracle' && zoneMode !== 'gyro') throw new Error('cap_runtime: zoneMode は oracle か gyro です');
  const tracker = createProgressTracker(loop, route);
  let gov, phase, prevYaw, lastFrame, log;
  function reset() { gov = governorInitialState(); phase = phaseInitialState(); prevYaw = null; lastFrame = -1; log = []; }
  reset();
  function filter({frame, steering, speed, state, config}) {
    // フレーム 0(リセットのあと)で追跡を始める。同じフレームの 2 回目は前の答え
    if (frame === 0 || lastFrame < 0) { tracker.reset(state.x, state.z, 0); gov = governorInitialState(); phase = phaseInitialState(); prevYaw = null; log = []; }
    else tracker.advance(state.x, state.z);
    lastFrame = frame;
    const s = tracker.s, t = state.sim_time, v = state.speed;
    let zone, gzone = null;
    if (zoneMode === 'oracle') zone = zoneOfS(s, loop.L, zones);
    else {
      const dyaw = prevYaw === null ? 0 : Math.atan2(Math.sin(state.yaw - prevYaw), Math.cos(state.yaw - prevYaw));
      const gz = prevYaw === null ? 0 : dyaw / config.dt * DEG;
      const r = phaseStep(phase, t, gz, v, phaseParams);
      phase = r.state; gzone = r.out.zone;
      zone = combineZone(r.out.zone, r.out.turning_right, false, v);
    }
    prevYaw = state.yaw;
    const steerNorm = Math.max(-1, Math.min(1, steering / (config.maxSteer || .52)));
    const g = governorStep(gov, t, zone, v, speed, params, steerNorm);
    gov = g.state;
    if (record) log.push({f: frame, t, s, zone, gz: gzone, v, u: speed, u2: g.out.u, mode: g.out.mode, cap: Number.isFinite(g.out.cap_eff) ? g.out.cap_eff : null,
      bs: g.out.brake_start ? 1 : 0, st: steerNorm});
    return {speed: g.out.u};
  }
  return {filter, reset, get log() { return log; }, get s() { return tracker.s; }, zoneMode, name: name ?? `cap-${zoneMode}`};
}
