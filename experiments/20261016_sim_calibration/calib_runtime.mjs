// 較正の物理 jf-calib-v1 の実行部(2026-10-16、シミュレーター較正 S2)。純粋な ESM で import なし。
//
// calib_core.mjs の patchSimCore() が、この文字列を sim-core.js の写しに埋め込み、step の小区間 1 つ分の
// 「速さ・舵・ヨーレート」の計算をここに任せる(ユーザーの sim-core.js は変えない)。
//
// 物理は 2 種類。
//   'cal'  : jf-calib-v1(profiles/P_cal.json の semantics のとおり。参照の実装は calib_model.py の replay(physics='cal'))
//            舵: 純粋な遅れ d_s → 左右で別の区分線形の対応表 → 舵の速さ r_s → アンダーステア K_us → μ g/|v| の頭打ち
//            前後: 遅れ d_v → 速さの対応 T(u)(不感帯つき)× 電池 k_batt → 駆動(τ_up・a_cap)/ 惰性 c0 / こすれ k_s|v r| /
//                  ESC のブレーキ s_b·max(0, off − db_b) / 止まったあとの後退の手順
//            路面の μ はプロファイルの路面 ID ごとの値(× config.frictionMultiplier)
//   'orig' : 元の sim-core.js の式そのまま + 指令の純粋な遅れ(P_cfg の忠実度の再生と、P_default の再生の道具に使う)
//
// 再生と試験のための操作(calibControl): 準備の間の速さの固定(forceSpeed)、窓の始まりの速さ(setSpeed は sim 側)、
// 遅れのための過去の指令(prefill)、電池の係数(kBatt)、指令のフィルター(filter。上限の層を差し込む口)。
// 何も設定しなければ、プロファイルのとおりに動く。

export const CALIB_RUNTIME_VERSION = 'jf-calib-runtime/v1';
const G = 9.80665;
const STEER_FULL = .52;            // u_s の ±0.52 rad = PWM の端(ブリッジ・RL の先生の単位)
const BRAKE_US_PER_MPS = 627.5;    // ブレーキの要求 off [µs] = −u_v × 627.5(ブリッジの −(thr − 1498)/502 × 0.8 の逆)
const STEER_XS = [-1, -.9, -.75, -.5, -.25, 0, .25, .5, .75, .9, 1];
const SPEED_KNOTS = [.5, 1, 2, 3, 4, 5, 6, 7.24];
const ESC_REVERSE_DEFAULT = {neutral_s: .12, target_mps: -.8, accel: 1.5, stopped_mps: .05};

const clampV = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const approachV = (value, target, maximumDelta) => value + clampV(target - value, -maximumDelta, maximumDelta);

// numpy.interp と同じ計算(xp は増加。端の外は端の値。x が節に一致すれば節の値)
export function interp(x, xp, fp) {
  const n = xp.length;
  if (!(x > xp[0])) return x === x ? fp[0] : NaN;
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
  if (xp[lo] === x) return fp[lo];
  const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

// プロファイルの種類: jf-calib-profile/v1 → 'cal'、jf-calib-replay-orig/v1 → 'orig'
export function profileKind(profile) {
  if (!profile || typeof profile !== 'object') throw new Error('calib_runtime: プロファイルがありません');
  if (profile.schema === 'jf-calib-profile/v1' && profile.model === 'jf-calib-v1') return 'cal';
  if (profile.schema === 'jf-calib-replay-orig/v1') return 'orig';
  throw new Error('calib_runtime: 知らないプロファイルです: ' + profile.schema);
}

// プロファイル → 計算に使う値(検査つき)
export function compileProfile(profile, {brakeScenario = 'nominal'} = {}) {
  const kind = profileKind(profile);
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  if (kind === 'orig') {
    const d = profile.actuator_delay_s ?? 0;
    if (!finite(d) || d < 0 || d > .5) throw new Error('calib_runtime: actuator_delay_s は 0〜0.5 秒です');
    return {kind, d_s: d, d_v: d};
  }
  const st = profile.steering, lo = profile.longitudinal;
  const L = st.delta_L_rad, R = st.delta_R_rad;
  if (L.length !== 6 || R.length !== 6) throw new Error('calib_runtime: 舵の節は 6 つです');
  // 節 0 は左右で共通(delta_L_rad[0])。numpy の concatenate([-R[::-1], L[1:]]) の 5 番目を L[0] にしたもの
  const steerYs = [-R[5], -R[4], -R[3], -R[2], -R[1], L[0], L[1], L[2], L[3], L[4], L[5]];
  for (let i = 1; i < steerYs.length; i++) if (!(steerYs[i] >= steerYs[i - 1])) throw new Error('calib_runtime: 舵の対応が単調ではありません');
  const T = lo.T_values;
  if (T.length !== SPEED_KNOTS.length) throw new Error('calib_runtime: 速さの節は 8 つです');
  if (!(lo.u_db >= 0 && lo.u_db < SPEED_KNOTS[0])) throw new Error('calib_runtime: u_db は 0〜0.5 です');
  const speedXs = [0, lo.u_db, ...SPEED_KNOTS], speedYs = [0, 0, ...T];
  for (let i = 1; i < speedYs.length; i++) if (!(speedYs[i] >= speedYs[i - 1])) throw new Error('calib_runtime: 速さの対応が単調ではありません');
  const brake = brakeScenario === 'nominal' ? lo.brake : lo.brake_scenarios?.[brakeScenario];
  if (!brake) throw new Error('calib_runtime: ブレーキのシナリオがありません: ' + brakeScenario);
  const esc = {...ESC_REVERSE_DEFAULT, ...(lo.esc_reverse ?? {})};
  const c = {
    kind, steerYs, speedXs, speedYs,
    d_s: st.d_s, d_v: lo.d_v, r_s: st.r_s === null || st.r_s === undefined || st.r_s === Infinity ? null : st.r_s,
    K_us: st.K_us, wheelbaseProfile: st.wheelbase,
    tau_up: lo.tau_up, a_cap: lo.a_cap, tau_down: lo.tau_down, c0: lo.c0, k_s: lo.k_s,
    s_b: brake.s_b, db_b: brake.db_b, k_batt: lo.k_batt ?? 1, esc, brakeScenario,
    mu: {...profile.surfaces_mu}, muDefault: profile.default_surface_mu,
  };
  for (const k of ['d_s', 'd_v', 'K_us', 'tau_up', 'a_cap', 'tau_down', 'c0', 'k_s', 's_b', 'db_b', 'k_batt', 'muDefault'])
    if (!finite(c[k]) || c[k] < 0) throw new Error('calib_runtime: 値が正しくありません: ' + k);
  if (!(c.tau_up > 0)) throw new Error('calib_runtime: tau_up > 0 が要ります');
  if (c.r_s !== null && !(c.r_s > 0)) throw new Error('calib_runtime: r_s > 0 が要ります');
  for (const [k, v] of Object.entries(c.mu)) if (!finite(v) || v <= 0) throw new Error('calib_runtime: μ が正しくありません: ' + k);
  return c;
}

// 1 つのシミュレーションに 1 つ。config は sim-core.js の正規化した config(参照で持ち、値はその時点のものを使う)
export function createCalibRuntime(profile, config, meta = {}) {
  const P = compileProfile(profile, {brakeScenario: meta.brakeScenario ?? 'nominal'});
  const out = {nextSpeed: 0, nextSteering: 0, rawRate: 0, limit: 0, yawRate: 0};
  const maxDelay = Math.max(P.d_s, P.d_v);
  let hist, first, prefill, cur, curFrame, rLast, escSeen, escNeutral, escArmed, forceSpeed, kBatt, filter, filterName;
  function reset() {
    hist = new Map(); first = null; prefill = null; cur = {steering: 0, speed: 0}; curFrame = -1;
    rLast = 0; escSeen = false; escNeutral = 0; escArmed = false;
    forceSpeed = null; kBatt = P.kind === 'cal' ? P.k_batt : 1; filter = null; filterName = null;
  }
  reset();
  // フレームの始まり: 届いた指令を履歴に入れる(フィルターがあれば先に掛ける)
  function beginFrame(frame, steering, speed, state) {
    let c = {steering, speed};
    if (filter) {
      const f = filter({frame, steering, speed, state, config});
      if (f) c = {steering: f.steering ?? steering, speed: f.speed ?? speed};
    }
    cur = c; curFrame = frame; hist.set(frame, c);
    if (!first) first = c;
    // 遅れに要る分より古い指令は捨てる
    const keep = Math.ceil(maxDelay / config.dt) + 3;
    const old = frame - keep;
    if (hist.has(old)) hist.delete(old);
    return c;
  }
  // 時刻 τ − d を含むフレームの指令(calib_model.delayed_index と同じ floor((τ − d)/dt + 1e-9))
  function commandAt(tau, d) {
    if (d === 0) return cur;
    const j = Math.floor((tau - d) / config.dt + 1e-9);
    if (j >= curFrame) return cur;
    const h = hist.get(j);
    if (h) return h;
    if (j < 0 && prefill) {
      const p = prefill[prefill.length + j];
      if (p) return p;
    }
    return first ?? cur;
  }
  function steerTarget(u) {
    return interp(clampV(u / STEER_FULL, -1, 1), STEER_XS, P.steerYs);
  }
  function speedTarget(u) {
    return kBatt * interp(Math.max(u, 0), P.speedXs, P.speedYs);
  }
  function muOf(surface) {
    const base = Object.hasOwn(P.mu, surface.id) ? P.mu[surface.id] : P.muDefault;
    return base * config.frictionMultiplier;
  }
  function substepCal(state, surface, tau, h) {
    const mug = muOf(surface) * G;
    const cs = commandAt(tau, P.d_s), cv = commandAt(tau, P.d_v);
    const v = state.speed;
    let vNext;
    if (forceSpeed !== null) vNext = forceSpeed;
    else {
      const u = cv.speed;
      const scrub = P.k_s * Math.abs(v * rLast);
      const fwdCmd = u >= 0;
      let hold = false;
      if (fwdCmd) {
        const vt = speedTarget(u);
        if (v < 0) vNext = approachV(v, 0, 1.5 * h);
        else if (vt > v) {
          const a = Math.min(Math.min(P.a_cap, mug), (vt - v) / P.tau_up) - scrub;
          let vd = v + a * h;
          if (a > 0) vd = Math.min(vd, vt);
          vNext = Math.max(vd, 0);
        } else {
          const lift = P.tau_down > 0 ? Math.min(P.c0, (v - vt) / P.tau_down) : (v > vt ? P.c0 : 0);
          vNext = Math.max(v - (lift + scrub) * h, 0);
        }
      } else {
        const ab = P.s_b * Math.max(0, -Math.min(u, 0) * BRAKE_US_PER_MPS - P.db_b);
        if (v > P.esc.stopped_mps) vNext = Math.max(v - (P.c0 + scrub + ab) * h, 0);
        else if (escArmed) vNext = approachV(v, P.esc.target_mps, P.esc.accel * h);
        else { hold = true; vNext = Math.max(v - (P.c0 + ab) * h, Math.min(v, 0)); }
      }
      // ESC の手順: 止まってブレーキ側を受けた → 前進側(中立)を 0.12 秒以上 → 後退できる。走り出したら戻す
      if (hold) escSeen = true;
      if (hold || !fwdCmd) escNeutral = 0;
      else if (escSeen && fwdCmd && v <= P.esc.stopped_mps) escNeutral = escNeutral + h;
      escArmed = escArmed || (escSeen && escNeutral >= P.esc.neutral_s - 1e-12);
      if (vNext > P.esc.stopped_mps) { escSeen = false; escArmed = false; escNeutral = 0; }
    }
    const tgt = steerTarget(cs.steering);
    const dNext = P.r_s === null ? tgt : approachV(state.steering, tgt, P.r_s * h);
    const raw = vNext * Math.tan(dNext) / config.wheelbase / (1 + P.K_us * vNext * vNext);
    const limit = mug / Math.max(Math.abs(vNext), .01);
    const yawRate = clampV(raw, -limit, limit);
    rLast = yawRate;
    out.nextSpeed = vNext; out.nextSteering = dNext; out.rawRate = raw; out.limit = limit; out.yawRate = yawRate;
    return out;
  }
  // 元の式(sim-core.js 365〜379 行と同じ順の計算)+ 指令の遅れ
  function substepOrig(state, surface, tau, h) {
    const steering = commandAt(tau, P.d_s).steering, speed = commandAt(tau, P.d_v).speed;
    let nextSpeed;
    if (forceSpeed !== null) nextSpeed = forceSpeed;
    else if (config.longitudinalModel === 'real-drive-response-estimate-v1') {
      const rate = state.speed >= 0 && speed > state.speed
        ? Math.min(config.driveAccelerationCap, surface.mu * 9.80665, (speed - state.speed) / config.driveTau)
        : config.brakeDeceleration;
      nextSpeed = approachV(state.speed, speed, rate * h);
    } else nextSpeed = approachV(state.speed, speed, config.acceleration * h);
    const nextSteering = approachV(state.steering, steering, config.steerRate * h);
    const rawRate = nextSpeed / config.wheelbase * Math.tan(nextSteering);
    const limit = surface.mu * 9.80665 / Math.max(Math.abs(nextSpeed), .01);
    const yawRate = clampV(rawRate, -limit, limit);
    rLast = yawRate;
    out.nextSpeed = nextSpeed; out.nextSteering = nextSteering; out.rawRate = rawRate; out.limit = limit; out.yawRate = yawRate;
    return out;
  }
  const substep = P.kind === 'cal' ? substepCal : substepOrig;
  // 再生・試験の操作。返り値は今の内部の状態
  function control(o = {}) {
    if (Object.hasOwn(o, 'forceSpeed')) forceSpeed = o.forceSpeed === null ? null : +o.forceSpeed;
    if (Object.hasOwn(o, 'kBatt')) {
      if (!(o.kBatt > 0 && o.kBatt < 5)) throw new Error('calib_runtime: kBatt は 0〜5 です');
      kBatt = +o.kBatt;
    }
    if (Object.hasOwn(o, 'prefill')) prefill = o.prefill ? o.prefill.map(c => ({steering: +c.steering, speed: +c.speed})) : null;
    if (Object.hasOwn(o, 'filter')) { filter = o.filter || null; filterName = filter ? (o.filterName ?? 'filter') : null; }
    return {kind: P.kind, forceSpeed, kBatt, rLast, escSeen, escNeutral, escArmed, curFrame, filter: filterName, current: {...cur}};
  }
  function info() {
    return {runtime: CALIB_RUNTIME_VERSION, kind: P.kind, name: meta.name ?? profile.name ?? null, model: profile.model ?? null,
      profile_sha256: meta.profileSha256 ?? null, brake_scenario: P.kind === 'cal' ? P.brakeScenario : null,
      k_batt: P.kind === 'cal' ? P.k_batt : null, actuator_delay_s: {steer: P.d_s, speed: P.d_v}, filter: filterName};
  }
  return {reset, beginFrame, substep, control, info, steerTarget, speedTarget, commandAt, compiled: P};
}

// ------------------------------------------------------------------ 上限の層(安全層 v1)の 1 フレーム
// jetformer/safety_caps.py の cap_step を、そのまま写したもの(試験 T6 で同じ入力に同じ出力を確かめる)。
// 区間 zone は 'none' / 'unknown' / 'R1_app' / 'R1_turn' / 'GM' / 'DE'。v は今の速さ、u は指令(速さの単位、負 = ブレーキの要求)、
// steer は舵の指令(正規化 −1〜1、左 +。null なら見ない)。state は governorInitialState() から始め、返り値の state を次に渡す。
const GOV_ZONES = ['none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE'];
const CAL_T_KNOTS_U = [0.0, 0.4286189631087167, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.24];
const CAL_T_VALUES = [0.0, 0.0, 0.18008916970987934, 1.0154669645131382, 2.191791568056699, 3.336828397976429,
  4.149389098689263, 5.0070424923922605, 5.007080520819097, 5.012401396968912];
export const GOVERNOR_V1_SIM = Object.freeze({cap_R1_app: 2.5, cap_R1_turn: 2.5, cap_GM: 2.2, cap_DE: 3.0, cap_unknown: 3.0,
  brake_zones: ['R1_app'], brake_margin: .3, brake_min_speed: 1.0, brake_us: 200.0, brake_hold_s: .2, rebrake_lockout_s: .5,
  release_accel: 1.5, cap_free: 8.0, cap_command: 'identity', brake_steer_release: .3});
export function governorInitialState() { return {t: null, cap_eff: Infinity, braking: false, brake_t0: -Infinity, brake_t1: -Infinity}; }
function speedToCmdCalibrated(vCap) {
  if (!Number.isFinite(vCap)) return Infinity;
  const T = CAL_T_VALUES, U = CAL_T_KNOTS_U;
  if (vCap <= T[2]) return U[1] + (U[2] - U[1]) * Math.max(vCap, 0) / T[2];
  for (let j = 2; j < T.length - 1; j++) if (T[j] <= vCap && vCap <= T[j + 1] && T[j + 1] > T[j]) return U[j] + (U[j + 1] - U[j]) * (vCap - T[j]) / (T[j + 1] - T[j]);
  return U[U.length - 1];
}
function capOf(p, zone) {
  return {R1_app: p.cap_R1_app, R1_turn: p.cap_R1_turn, GM: p.cap_GM, DE: p.cap_DE, unknown: p.cap_unknown}[zone] ?? Infinity;
}
export function governorStep(state, t, zone, v, u, p = GOVERNOR_V1_SIM, steer = null) {
  if (!GOV_ZONES.includes(zone)) throw new Error('governor: 知らない区間です: ' + zone);
  const dt = state.t === null ? 0 : Math.max(0, t - state.t);
  const target = capOf(p, zone);
  let capEff;
  if (target <= state.cap_eff) capEff = target;
  else {
    capEff = Math.min(target, state.cap_eff + p.release_accel * dt);
    if (capEff >= p.cap_free) capEff = target;
  }
  let {braking, brake_t0: b0, brake_t1: b1} = state;
  let inBrakeZone = p.brake_zones.includes(zone);
  if (p.brake_steer_release !== null && p.brake_steer_release !== undefined && steer !== null && steer !== undefined && steer <= -p.brake_steer_release)
    inBrakeZone = false;
  if (braking) {
    const held = t - b0;
    if (!inBrakeZone || v < p.brake_min_speed || (held >= p.brake_hold_s - 1e-9 && v <= capEff)) { braking = false; b1 = t; }
  }
  let start = false;
  if (!braking && inBrakeZone && Number.isFinite(capEff) && v > capEff + p.brake_margin && v >= p.brake_min_speed && t - b1 >= p.rebrake_lockout_s - 1e-9) {
    braking = true; b0 = t; start = true;
  }
  let outU, mode;
  if (braking) { outU = Math.min(u, -p.brake_us / 502.0 * .8); mode = 'brake'; }
  else if (v > capEff) { outU = Math.min(u, 0); mode = outU < u ? 'neutral' : 'pass'; }
  else {
    const c = !Number.isFinite(capEff) ? Infinity : (p.cap_command === 'calibrated' ? speedToCmdCalibrated(capEff) : capEff);
    outU = Math.min(u, c); mode = outU < u ? 'cap' : 'pass';
  }
  return {state: {t, cap_eff: capEff, braking, brake_t0: b0, brake_t1: b1}, out: {u: outU, mode, cap_eff: capEff, zone, brake_start: start}};
}
