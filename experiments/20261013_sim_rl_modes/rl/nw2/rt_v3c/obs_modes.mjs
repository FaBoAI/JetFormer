// モード別の観測の包み(2026-10-13、sim_rl_modes)。features.mjs(79 次元、ベストラップと同じファイル)は変えず、後ろに入力を足す。
// Node(学習の環境・試験)とブラウザの教師で同じファイル(純粋な ESM)。modes.json common.obs_wrappers のとおり:
//   base79 : 79 次元、目標ゲートは sim の route_index(ベストラップと同じ)
//   base79v: 79 次元、目標ゲートは「仮想」: sim の route_index。ただしゲートを外した(R 上の s が目標ゲートの s + 0.30 m を越えても
//            route_index が進まない)あとは、s から決めた次のゲート。外していなければ base79 と同じ値。
//            (実装の注意: 外したあと、sim の route_index が進んで仮想の目標と同じになったら sim の値に戻る。build.md)
//   cond97 : base79v + 先読み 16 点の e*(s+Δs)/0.5 + (e_y − e*(s))/0.5 + c_v
//   obst95 : base79v + 足した障害物(kind 'obstacle')のうち近い順に 4 個(3 m 以内・左右 100° 以内)の 車の座標系の前/3・左/3・半径/0.2・有無
import {createFeatureTracker, OBS_DIM, LOOKAHEAD_DS, LOOP_LENGTH, PROJECTION_WINDOW} from './features.mjs';

export const OBS_WRAPPERS = Object.freeze({
  base79: Object.freeze({dim: 79, virtual: false}), base79v: Object.freeze({dim: 79, virtual: true}),
  cond97: Object.freeze({dim: 97, virtual: true}), obst95: Object.freeze({dim: 95, virtual: true}),
});
export const VIRTUAL_MISS_MARGIN_M = .30;
export const OBST_K = 4, OBST_RANGE_M = 3, OBST_HALF_ANGLE = 100 * Math.PI / 180;

// 足した障害物(layouts.mjs の形 {shape: 'post'|'box', x, z, r, length?, yaw?})→ 車の座標系の入力 16 個を out[offset..] に書く
export function obstacleFeatures(obstacles, x, z, yaw, out, offset) {
  const c = Math.cos(yaw), sn = Math.sin(yaw), list = [];
  for (const o of obstacles ?? []) {
    let px = o.x, pz = o.z;
    if (o.shape === 'box') {
      // カプセルの芯の線分で車にいちばん近い点
      const h = o.length / 2, ux = Math.cos(o.yaw), uz = -Math.sin(o.yaw);
      const t = Math.max(-h, Math.min(h, (x - o.x) * ux + (z - o.z) * uz));
      px = o.x + t * ux; pz = o.z + t * uz;
    }
    const dx = px - x, dz = pz - z, fwd = dx * c - dz * sn, left = -dx * sn - dz * c, d = Math.hypot(dx, dz);
    if (d > OBST_RANGE_M || Math.abs(Math.atan2(left, fwd)) > OBST_HALF_ANGLE) continue;
    list.push({d, fwd, left, r: o.r});
  }
  list.sort((a, b) => a.d - b.d);
  for (let k = 0; k < OBST_K; k++) {
    const q = list[k], o = offset + 4 * k;
    if (q) { out[o] = q.fwd / 3; out[o + 1] = q.left / 3; out[o + 2] = q.r / .2; out[o + 3] = 1; }
    else { out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; }
  }
  return out;
}

// world: 回した経路の world(障害物を足したものでよい)。config: 観測に使う物理(標準の値)。geometry: 形(省略すると world から)。
// cond: {line: createLineProfile の戻り値の作り手 (s0) => profile, cV}。obstacles: 足した障害物の一覧(obst95)
export function createObsWrapper({wrapper = 'base79', world, config, geometry = null, lineFactory = null, cV = 1, obstacles = null}) {
  const W = OBS_WRAPPERS[wrapper];
  if (!W) throw new Error('obs_modes: 知らない観測の包みです: ' + wrapper);
  const tracker = createFeatureTracker(world, config, {geometry});
  const ts = tracker.targetS, n = ts.length, L = LOOP_LENGTH;
  const step = (i) => ((ts[(i + 1) % n] - ts[i]) % L + L) % L;
  let simIdx = 0, simS = 0, vIdx = 0, vS = 0, missed = false, lastRi = 0, misses = 0, line = null, s0 = 0;
  const g = {s: 0, obsIdx: 0, missed: false, missEvent: false, gateS: 0, simGateS: 0};
  function reset(state, routeIndex) {
    const ri = routeIndex ?? 0;
    tracker.resetProgress(state.x, state.z, ri);
    s0 = tracker.s;
    simIdx = ri % n;
    const g0 = ts[simIdx];
    simS = g0 + L * Math.ceil((s0 - PROJECTION_WINDOW.behind - g0) / L);
    vIdx = simIdx; vS = simS; missed = false; lastRi = ri; misses = 0;
    line = lineFactory ? lineFactory(s0) : null;
    Object.assign(g, {s: s0, obsIdx: W.virtual ? vIdx : ri, missed: false, missEvent: false, gateS: simS, simGateS: simS});
    return g;
  }
  // 毎フレーム(sim を 1 フレーム進めたあと)。戻り値 g: s・観測に使う目標ゲートの番号・外した印・報酬の頭打ちに使うゲートの s
  function advance(state, routeIndex) {
    const ri = routeIndex ?? 0;
    const s = tracker.advanceProgress(state.x, state.z);
    const gi = ri % n;
    for (let guard = 0; simIdx !== gi && guard < n; guard++) { simS += step(simIdx); simIdx = (simIdx + 1) % n; }
    g.missEvent = false;
    if (W.virtual) {
      if (!missed) {
        vIdx = simIdx; vS = simS;
        if (s > simS + VIRTUAL_MISS_MARGIN_M) { missed = true; g.missEvent = true; misses++; }
      } else if (ri !== lastRi && simIdx === vIdx) { missed = false; simS = vS; }
      if (missed) for (let guard = 0; s > vS + VIRTUAL_MISS_MARGIN_M && guard < n; guard++) { vS += step(vIdx); vIdx = (vIdx + 1) % n; }
    }
    lastRi = ri;
    g.s = s; g.missed = missed; g.obsIdx = W.virtual ? vIdx : ri; g.gateS = W.virtual ? vS : simS; g.simGateS = simS;
    return g;
  }
  // 観測(dim 次元)を out に。aux には features の生の値と、cond の e*・obst の情報を足す
  function observe(state, prevA0, prevA1, out, aux = {}) {
    tracker.observe(state, g.obsIdx, prevA0, prevA1, out, aux);
    if (wrapper === 'cond97') {
      const s = tracker.s;
      for (let j = 0; j < 16; j++) out[OBS_DIM + j] = line.eStar(s + LOOKAHEAD_DS[j]) / .5;
      const es = line.eStar(s);
      out[OBS_DIM + 16] = (aux.lateral_error - es) / .5;
      out[OBS_DIM + 17] = cV;
      aux.e_star = es;
    } else if (wrapper === 'obst95') {
      obstacleFeatures(obstacles, state.x, state.z, state.yaw, out, OBS_DIM);
    }
    return out;
  }
  return {wrapper, dim: W.dim, virtual: W.virtual, tracker, reset, advance, observe, gate: g,
    get line() { return line; }, get s0() { return s0; }, get misses() { return misses; }, get missed() { return missed; }};
}
