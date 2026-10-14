// sim-core の「動的な初期状態」(reset({initialState}))の確認: 単周モードでは速さ・舵角つきの途中の状態から始められ、
// 目標ゲートを全部通ると recovery_segment_complete で終わる。連続周回モードでは使えない(例外)。
//   node probe/recovery_api.mjs
import {envCore, writeJson, loadBestPolicy, runPolicyEpisode} from './common.mjs';

const res = {schema: 'sim-rl-modes-probe-recovery-api/v1'};
const core = await envCore.loadCore('fast');
const world = envCore.loadBaseWorld();
// ベストラップの方策の走りの途中(5 秒目)の状態を取り、そこから単周の回復区間として始める
const {act} = loadBestPolicy();
const r = await runPolicyEpisode({act});
const q = r.trace[149];   // フレーム 150 のあと
const st0 = {x: q.x, z: q.z, yaw: r.env.sim.report().trace[149].yaw, speed: q.v, steering: r.env.sim.report().trace[149].steering, routeIndex: q.ri};
res.source_state = st0;
for (const mode of ['single', 'continuous']) {
  try {
    const sim = core.createSimulation({...world, spawn: {x: world.route[0][0], z: world.route[0][1], yaw: 0}}, envCore.simConfig({episodeMode: mode, timeLimit: 20}));
    const snap = sim.reset({initialState: st0});
    let n = 0, s = sim.peek();
    while (!s.state.terminal && n < 900) { sim.step({steering: 0, speed: 0}, true); s = sim.peek(); n++; }
    res[mode] = {ok: true, start_speed: snap.state.speed, start_route_index: snap.metrics.route_index, end_reason: s.state.reason, frames: n,
      note: '操作 0(まっすぐ・速さ 0 の指令 = 1.2 m/s² で減速)で流しただけ'};
  } catch (e) { res[mode] = {ok: false, error: String(e.message)}; }
}
const p = writeJson('recovery_api.json', res);
console.log(JSON.stringify(res, null, 1)); console.log('書いた:', p);
