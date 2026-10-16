// 較正版のコアを作る(2026-10-16、シミュレーター較正 S2)。Node 用。
//
//   patchSimCore(元の sim-core.js の文字列, プロファイル | null, {fast, name, profileSha256, brakeScenario}) → 較正版の文字列
//   loadCalibratedCore(プロファイル | null, {fast, ...}) → 較正版のモジュール(data: URL から import)
//
// ユーザーの sim-core.js(sha256 aaeb2379…549e)は読むだけで変えない。読み込むときに文字列で差分を当てる。
//   - 元の sha256 を確かめる。違えば止まる(ユーザーが sim-core.js を更新したら気づける)
//   - 差分は 1 か所ずつ置き換える。置き換える場所が 1 か所でなければ止まる(10/13 の fast-sim-core.mjs と同じ書き方)
//   - calib_runtime.mjs の中身を文字列のまま先頭に埋め込む(data: URL の中からは相対の import ができないため)
//   - プロファイルが null なら、較正の部分は通らず、元とビット単位で同じ動き(試験 T1)
//   - fast を付けると、10/13 の高速化(近くの障害物だけを判定・step(action, true) で写しを省く・peek())も重ねる
// 物理を替えても snapshot().config は元と同じ形。report().model.calibration にプロファイルの名前と sha256 が入る。
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {compileProfile} from './calib_runtime.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const ORIGINAL_SIM_CORE = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js';
export const ORIGINAL_SIM_CORE_SHA256 = 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e';
export const RUNTIME_PATH = path.join(here, 'calib_runtime.mjs');
export const REPORT = '/home/jetson/jetformer_reports/sim_calibration_20261016';
export const PROFILES = {
  P_cal: path.join(REPORT, 'profiles/P_cal.json'),
  P_cfg: path.join(REPORT, 'profiles/P_cfg.json'),
};
export const PROFILE_SHA256 = {
  P_cal: '4b0e9348e367e64d93c4dc491de26b5dd5c7e2375a60f6f618f0d24ad85ddc14',
  P_cfg: 'b15a436b923c6e269e1cdc3b2e30368dcbaba126c387aac51b856c47d227d5c1',
};
export const sha256 = text => crypto.createHash('sha256').update(text).digest('hex');
export const sha256File = p => sha256(fs.readFileSync(p));

export function replaceOnce(text, from, to, tag = 'calib-core') {
  const first = text.indexOf(from);
  if (first < 0 || text.indexOf(from, first + from.length) >= 0)
    throw new Error(`${tag}: 置換箇所が1つではありません: ` + from.slice(0, 70).replace(/\n/g, '⏎'));
  return text.slice(0, first) + to + text.slice(first + from.length);
}

// 10/13 の高速化(experiments/20261013_sim_rl_bestlap/sim/fast-sim-core.mjs と同じ 5 か所の置換)
export function applyFastPatches(source) {
  let p = source;
  const r = (a, b) => { p = replaceOnce(p, a, b, 'fast-sim-core'); };
  r('function step(action = {}) {', 'function step(action = {}, lite = false) {');
  r('    while (elapsed < remainingTime-1e-12 && !state.terminal) {',
    `    const nearWorld = nearObstacles(state.x, state.z);
    while (elapsed < remainingTime-1e-12 && !state.terminal) {`);
  r('      const hit = collisionAt(trial, world, config);', '      const hit = collisionAt(trial, nearWorld, config);');
  r("    trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });\n    return snapshot();",
    "    trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });\n    return lite ? null : snapshot();");
  r('  return { reset, step, observe, snapshot, report, demoAction, setPaused };',
    `  // 近傍の障害物だけを返す。1フレームの移動量は |speed|<=maxSpeed なので maxSpeed*dt 以下。
  const reachBase = vehicleRadius + config.maxSpeed * config.dt + .05;
  const segmentReach = world.segments.map(s => (reachBase + s.radius) ** 2);
  const diskReach = world.disks.map(d => (reachBase + (d.shape === 'rectangle' ? Math.hypot(d.w, d.d) / 2 : d.radius)) ** 2);
  function nearObstacles(x, z) {
    const p = [x, z], segments = [], disks = [];
    for (let i = 0; i < world.segments.length; i++) if (pointSegmentDistanceSquared(p, world.segments[i].a, world.segments[i].b) <= segmentReach[i]) segments.push(world.segments[i]);
    for (let i = 0; i < world.disks.length; i++) { const d = world.disks[i]; if ((d.x - x) ** 2 + (d.z - z) ** 2 <= diskReach[i]) disks.push(d); }
    return { segments, disks, bounds: world.bounds };
  }
  function peek() { return { state, metrics, config }; }
  return { reset, step, observe, snapshot, report, demoAction, setPaused, peek, world, targets };`);
  return p;
}

// calib_runtime.mjs の文字列 → 埋め込める形(export を外して 1 つの関数で包む)
export function runtimeBlock(runtimeText) {
  const body = runtimeText.replace(/^export (const|function) /gm, '$1 ');
  if (/^\s*import\s/m.test(body) || /\bexport\b/.test(body.replace(/\/\/.*$/gm, '')))
    throw new Error('calib-core: calib_runtime.mjs に import / export が残っています');
  return `const __calib = (() => {\n${body}\nreturn { createCalibRuntime, compileProfile, CALIB_RUNTIME_VERSION, governorStep, governorInitialState, GOVERNOR_V1_SIM };\n})();\n`;
}

// 較正の差分。profile が null なら、較正の分岐は通らない(元と同じ動き)
export function patchSimCore(source, profile, options = {}) {
  const {fast = false, name = null, profileSha256 = null, brakeScenario = 'nominal', runtimeText = fs.readFileSync(RUNTIME_PATH, 'utf8'),
    expectedSha256 = ORIGINAL_SIM_CORE_SHA256} = options;
  const got = sha256(source);
  if (got !== expectedSha256) throw new Error(`calib-core: sim-core.js の sha256 が違います(${got})。差分を当てられません`);
  let p = fast ? applyFastPatches(source) : source;
  const r = (a, b) => { p = replaceOnce(p, a, b); };
  // 1) フレームの始まり: 指令を較正の実行部へ(遅れの履歴・フィルター)
  r('    actionLog.push({ frame: state.frame, steering, speed });',
    '    actionLog.push({ frame: state.frame, steering, speed });\n    if (__CAL) __CAL.beginFrame(state.frame, steering, speed, state);');
  // 2) 小区間の物理: 較正の実行部に任せる(プロファイルが無ければ元の式のまま)
  r(`      let nextSpeed;
      if (config.longitudinalModel === 'real-drive-response-estimate-v1') {
        const rate = state.speed >= 0 && speed > state.speed
          ? Math.min(config.driveAccelerationCap,surface.mu*9.80665,(speed-state.speed)/config.driveTau)
          : config.brakeDeceleration;
        nextSpeed = approach(state.speed,speed,rate*h);
      } else nextSpeed = approach(state.speed, speed, config.acceleration*h);
      const nextSteering = approach(state.steering, steering, config.steerRate*h);
      const rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      const limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      const yawRate = clamp(rawRate, -limit, limit);`,
  `      let nextSpeed, nextSteering, rawRate, limit, yawRate;
      if (__CAL) {
        const calOut = __CAL.substep(state, surface, state.sim_time+elapsed, h);
        nextSpeed = calOut.nextSpeed; nextSteering = calOut.nextSteering; rawRate = calOut.rawRate; limit = calOut.limit; yawRate = calOut.yawRate;
      } else {
      if (config.longitudinalModel === 'real-drive-response-estimate-v1') {
        const rate = state.speed >= 0 && speed > state.speed
          ? Math.min(config.driveAccelerationCap,surface.mu*9.80665,(speed-state.speed)/config.driveTau)
          : config.brakeDeceleration;
        nextSpeed = approach(state.speed,speed,rate*h);
      } else nextSpeed = approach(state.speed, speed, config.acceleration*h);
      nextSteering = approach(state.steering, steering, config.steerRate*h);
      rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      yawRate = clamp(rawRate, -limit, limit);
      }`);
  // 3) 実行部を作る(config は正規化したもの)
  r('  const world = normalizedWorld(worldInput), config = normalizedConfig(configInput);\n  const worldHash',
    '  const world = normalizedWorld(worldInput), config = normalizedConfig(configInput);\n  const __CAL = __CAL_PROFILE ? __calib.createCalibRuntime(__CAL_PROFILE, config, __CAL_META) : null;\n  const worldHash');
  // 4) reset で実行部も戻す
  r('    actionLog = []; trace = []; priorContact = null;', '    actionLog = []; trace = []; priorContact = null;\n    if (__CAL) __CAL.reset();');
  // 5) report().model.calibration
  r('provisional_vehicle_dimensions: true } });', 'provisional_vehicle_dimensions: true, ...(__CAL ? { calibration: __CAL.info() } : {}) } });');
  // 6) 再生・試験の操作 calibControl(プロファイルの無いコアでは使えない)
  r('  function setPaused(paused) {', `  function calibControl(options = {}) {
    if (!__CAL) throw new Error('calibControl: プロファイルの無いコアです');
    if (Object.hasOwn(options, 'setSpeed')) state.speed = +options.setSpeed;
    return __CAL.control(options);
  }
  function setPaused(paused) {`);
  r('demoAction, setPaused', 'demoAction, setPaused, calibControl');
  const meta = profile ? {name: name ?? profile.name ?? null, profileSha256: profileSha256 ?? sha256(JSON.stringify(profile)), brakeScenario} : null;
  if (profile) compileProfile(profile, {brakeScenario});   // 埋め込む前に Node 側で検査して止める
  const header = `// 較正版のコア(jf-calib)。元: dist/sim-core.js sha256 ${expectedSha256}。` +
    `生成: experiments/20261016_sim_calibration/calib_core.mjs patchSimCore(fast=${fast})。このファイルを手で直さない。\n` +
    runtimeBlock(runtimeText) +
    `const __CAL_PROFILE = ${profile ? JSON.stringify(profile) : 'null'};\n` +
    `const __CAL_META = ${meta ? JSON.stringify(meta) : 'null'};\n`;
  return header + p;
}

export function loadProfile(key) {
  const p = PROFILES[key];
  const text = fs.readFileSync(p);
  const got = sha256(text);
  if (got !== PROFILE_SHA256[key]) throw new Error(`calib-core: ${key} の sha256 が固めたものと違います: ${got}`);
  return {profile: JSON.parse(text), sha256: got, path: p};
}

// 忠実度の再生の道具: 元の式 + 純粋な遅れ(P_default は 0、P_cfg は 1/30 秒)
export const origReplayProfile = (delayS = 0, name = null) => ({schema: 'jf-calib-replay-orig/v1', name: name ?? `orig-delay-${delayS}`, actuator_delay_s: delayS});

const moduleCache = new Map();
export async function loadCalibratedCore(profile, options = {}) {
  const source = fs.readFileSync(options.sourcePath ?? ORIGINAL_SIM_CORE, 'utf8');
  const text = patchSimCore(source, profile, options);
  const key = sha256(text);
  if (!moduleCache.has(key)) moduleCache.set(key, await import(`data:text/javascript;base64,${Buffer.from(text).toString('base64')}`));
  return moduleCache.get(key);
}
