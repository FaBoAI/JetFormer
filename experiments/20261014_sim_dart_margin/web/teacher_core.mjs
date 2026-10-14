// DART の先生の「中身」(2026-10-14、sim_dart_20261014 の準備の段)。ブラウザの差し込みモジュールと Node の付け替えが同じこのファイルを使う(純粋な ESM)。
//
// 先生は 2 つ(protocol.json teachers):
//   fast   = T_fast: ベストラップの RL 方策(rl-teacher-bestlap.js と同じ計算。features.mjs の 79 次元・policy_forward.mjs・weights_fast.mjs)
//   margin = T_margin: 経路いろいろ(line r2)の方策を、線を左へ 30% 寄せて(c0 = +0.3・c_v = 1.0・線の版 v2)走らせる
//            (modes の mode_runtime.mjs の createModeTeacher(mode 'line') と同じ計算。cond97 の観測の包み・線 v2)
// どちらも、写した rt/ のファイル(元と sha256 が同じ)を組み合わせて同じ計算をする。同じかどうかは tests/teacher_core.test.mjs で
// 元のモジュール(rl-teacher-bestlap.js・createModeTeacher)とビット単位で比べる。
//
// 使い方(1 エピソードに 1 つ):
//   const core = createTeacherCore(kind, world, config)
//   フレーム 0: core.begin(state, routeIndex)      フレーム t > 0: core.advance(state, routeIndex)
//   その後 core.decide(state, prevA0, prevA1) → {steering, speed, a0, a1, s, dmin, ...}(状態は変えない)
//   prevA は「前のフレームで実際に適用した操作」を方策の単位([−1, 1]²)にしたもの(フレーム 0 は 0, 0)。
// 分かれ(fork): core.fork() は、今までに begin / advance に渡した位置の列を新しい core にもう一度流して、同じ内部状態の写しを作る
//   (進みの追跡 s・仮想ゲート・線の起点。観測と方策は計算しないので O(t) で軽い)。写しと元は互いに影響しない。
import {createFeatureTracker, actionToCommand, commandToAction, OBS_DIM, prepareGeometry, gateStatus} from './rt/features.mjs';
import {createPolicy} from './rt/policy_forward.mjs';
import {createObsWrapper, OBS_WRAPPERS} from './rt/obs_modes.mjs';
import {createLineProfile} from './rt/line_profiles.mjs';
import {policyFromWeights, sceneParams, BASE_SPEED_CAP} from './rt/mode_runtime.mjs';
import {obstaclesFromWorld} from './rt/layouts.mjs';

export const TEACHER_CORE_VERSION = 'jf-sim-dart-teacher-core/v1';
export const SPEED_CAP = 4.5;   // 両方の先生の速さの上限(protocol.json teachers.*.speed_cap_mps)

// protocol.json teachers.T_margin.params
export const MARGIN_PARAMS = Object.freeze({line: Object.freeze({family: 'const', c0: 0.3}), c_v: 1.0});

// kind ごとの重みの検査と、方策の関数(重いので 1 回だけ作り、写しと共有する)
function makeShared(kind, world, config, weights) {
  const geometry = prepareGeometry(world);
  if (kind === 'fast') {
    if (weights.obs_dim !== OBS_DIM) throw new Error('teacher_core: 最速の先生の重みの観測の次元が 79 ではありません');
    return {kind, geometry, act: createPolicy(weights), maxSteer: config.maxSteer ?? .52, speedCap: SPEED_CAP};
  }
  if (kind === 'margin') {
    // createModeTeacher(world, config, variation, {mode: 'line', weights, params: MARGIN_PARAMS}) と同じ準備
    const seed = (config?.seed ?? 1) >>> 0;
    const p = sceneParams('line', seed, MARGIN_PARAMS);
    if (p.c_v !== 1 || p.line.family !== 'const' || p.line.c0 !== .3) throw new Error('teacher_core: 余裕の線の条件が protocol と違います');
    const wrapperName = 'cond97', dim = OBS_WRAPPERS[wrapperName].dim;
    const act = policyFromWeights(weights, dim);
    const cV = p.c_v ?? 1, speedCap = BASE_SPEED_CAP * cV, maxSteer = config?.maxSteer ?? .52;
    const obstacles = obstaclesFromWorld(world);
    if (obstacles.length) throw new Error('teacher_core: 障害物のあるコースでは使えません');
    const lineVersion = weights?.meta?.line_profile ?? 'v1';
    if (lineVersion !== 'v2') throw new Error('teacher_core: 余裕の線の方策は線の版 v2 のはずです: ' + lineVersion);
    return {kind, geometry, act, maxSteer, speedCap, p, cV, obstacles, lineVersion, dim};
  }
  throw new Error('teacher_core: 知らない先生です: ' + kind);
}

export function createTeacherCore(kind, world, config, weights, {shared = null, line = null} = {}) {
  const S = shared ?? makeShared(kind, world, config, weights);
  const hist = [];   // begin / advance に渡した [x, z, routeIndex](fork の再生用)
  let tracker = null, wrapper = null, lineObj = line;
  const dim = kind === 'fast' ? OBS_DIM : S.dim;
  const obs = new Float64Array(dim), a = [0, 0], aux = {};
  if (kind === 'fast') {
    tracker = createFeatureTracker(world, config, {geometry: S.geometry});
  } else {
    // 線は最初の観測で起点(anchor)が決まる。fork の写しは元の線の物(起点が決まったもの)を共有する(起点が決まれば中身は変わらない)
    const lineFactory = s0 => { if (!lineObj) lineObj = createLineProfile(S.p.line, s0, {version: S.lineVersion}); return lineObj; };
    wrapper = createObsWrapper({wrapper: 'cond97', world, config, geometry: S.geometry, lineFactory, cV: S.cV, obstacles: S.obstacles});
  }
  const route = world.route;
  function begin(state, routeIndex) {
    if (hist.length) throw new Error('teacher_core: begin は 1 回だけです');
    hist.push([state.x, state.z, routeIndex]);
    if (kind === 'fast') tracker.resetProgress(state.x, state.z, routeIndex);
    else wrapper.reset(state, routeIndex);
  }
  function advance(state, routeIndex) {
    if (!hist.length) throw new Error('teacher_core: begin の前に advance は呼べません');
    hist.push([state.x, state.z, routeIndex]);
    if (kind === 'fast') tracker.advanceProgress(state.x, state.z);
    else wrapper.advance(state, routeIndex);
  }
  // 観測 → 方策 → 操作。routeIndex は fast だけが使う(sim の route_index。rl-teacher-bestlap.js と同じ)。margin は仮想ゲート(wrapper の中)
  function decide(state, prevA0, prevA1) {
    let routeIndex = hist.at(-1)[2];
    if (kind === 'fast') tracker.observe(state, routeIndex, prevA0, prevA1, obs, aux);
    else wrapper.observe(state, prevA0, prevA1, obs, aux);
    S.act(obs, a);
    const c = actionToCommand(a[0], a[1], S.speedCap, S.maxSteer);
    // 乱しの見張りに使う「今の目標ゲート」に対する位置は、sim の route_index で測る(設計の段の measure_teachers.mjs の gateCtx と同じ)
    const gs = gateStatus(route, routeIndex, state.x, state.z);
    return {steering: c.steering, speed: c.speed, a0: c.a0, a1: c.a1, s: aux.s, dmin: aux.d_min, ey: aux.lateral_error,
      gate_along: gs.along, gate_lateral: gs.lateral, e_star: aux.e_star};
  }
  function fork() {
    const c = createTeacherCore(kind, world, config, weights, {shared: S, line: lineObj});
    for (let i = 0; i < hist.length; i++) { const [x, z, ri] = hist[i]; if (i === 0) c.begin({x, z}, ri); else c.advance({x, z}, ri); }
    return c;
  }
  return {kind, begin, advance, decide, fork, get frames() { return hist.length; }, get shared() { return S; },
    get speedCap() { return S.speedCap; }, get maxSteer() { return S.maxSteer; }};
}

// 実際に適用した操作 → 次のフレームの観測の「前回の行動」(protocol.json dart.applied_and_previous_action)
// 乱していない成分は先生自身の値(a0, a1)をそのまま使う(乱しが無ければ A0 の記録とまったく同じ計算になる)。
// 乱した成分だけ、適用した操作を方策の単位に戻す。steerNoisy / speedNoisy は乱しの予定から決まる(値の比べではない)。
export function previousAction(clean, applied, steerNoisy, speedNoisy, speedCap, maxSteer) {
  if (!steerNoisy && !speedNoisy) return [clean.a0, clean.a1];
  const b = commandToAction(applied.steering, applied.speed, speedCap, maxSteer);
  return [steerNoisy ? b.a0 : clean.a0, speedNoisy ? b.a1 : clean.a1];
}
