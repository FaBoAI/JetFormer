// 確認 r4: 自前のすき間(own_ck4.mjs clearOwn)と、シールドが使うすき間(features.mjs clearance)と、sim-core の衝突の決まり(reset の invalid_spawn)を
// ランダムな姿勢で比べる。シールドのすき間が本当より大きい(甘い)所があれば、シールドがだまされる
import {clearance, prepareGeometry} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4/runtime/features.mjs';
import {BASE_WORLD, obstacles, clearOwn, coreO, mulberry32, NOMINAL_CFG} from './own_ck4.mjs';
const O = obstacles(BASE_WORLD), g = prepareGeometry(BASE_WORLD);
const R = mulberry32(0x6e0c4e01);
let n = 0, nNear = 0, maxOver = -Infinity, maxUnder = -Infinity, worst = null, simChecked = 0, simDisagree = 0, simDisagreeEx = [];
const N = +(process.argv[2] ?? 300000), NS = +(process.argv[3] ?? 30000);
for (let i = 0; i < N; i++) {
  const x = O.B.minX + (O.B.maxX - O.B.minX) * R(), z = O.B.minZ + (O.B.maxZ - O.B.minZ) * R(), yaw = (R() * 2 - 1) * Math.PI;
  const a = clearOwn(O, x, z, yaw, .5), b = clearance(g, x, z, yaw, .42, .19, .5);
  n++;
  if (a < .5 - 1e-9) {
    nNear++;
    const aa = Math.max(0, a);   // features は 0 で切る
    const over = b - aa, under = aa - b;
    if (over > maxOver) { maxOver = over; worst = {x, z, yaw, own: a, feat: b}; }
    if (under > maxUnder) maxUnder = under;
  }
  if (i < NS) {
    const w = {...BASE_WORLD, spawn: {x, z, yaw}};
    const s = coreO.createSimulation(w, NOMINAL_CFG).reset({seed: 1});
    const hit = s.state.terminal && s.state.reason === 'invalid_spawn';
    simChecked++;
    if (hit !== (a <= 1e-12)) { simDisagree++; if (simDisagreeEx.length < 5) simDisagreeEx.push({x, z, yaw, own: a, hit}); }
  }
}
console.log(JSON.stringify({n, nNear, features_minus_own_max: maxOver, own_minus_features_max: maxUnder, worst, simChecked, simDisagree, simDisagreeEx}));
