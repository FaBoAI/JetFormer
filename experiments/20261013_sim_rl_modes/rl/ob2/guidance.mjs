// 障害物よけ r2(後半)の学習だけで使う「通り道の目標」(2026-10-14 学習の段 obstacle r2)。走行中の仕組み(mode_runtime.mjs)・観測には入らない。
// 障害物ごとに、障害物の s ±0.2 m(障害物が横の位置をふさいでいる所)で全部の障害物(コース + 足したもの)とのすき間 ≥ 8 cm で動ける横の範囲
// (R からの横ずれ、ゲートの窓 ±0.65 m の中)を 1 cm の格子で調べ、9 つの s で共通に動ける区間のうち一番広いものを選ぶ。
// 目標の横ずれ e_t = その区間の障害物に近い端から min(0.12 m, 幅/2) だけ内側(広い通り道ではすき間 ≈ 8 + 12 cm、狭い通り道では真ん中)。
// 学習の環境(env_modes.mjs run obstacle_r2b)は、次の障害物の手前 2.5 m から 0.3 m 過ぎるまで −w·min(1, ((e_y − e_t)/0.15)²)·ramp を足す。
import {clearance, loopPointAt, LOOP_LENGTH} from '../runtime/features.mjs';
import {obstacleOnlyGeometry, obstacleS} from '../runtime/layouts.mjs';
import {ROOM_TABLE} from '../runtime/room_table.mjs';
import {obstacleEy} from '../ob/ob_common.mjs';

const N = ROOM_TABLE.n, DS = ROOM_TABLE.ds_m, OFF_MAX = 1.0, STEP = .01, NOFF = Math.round(2 * OFF_MAX / STEP) + 1, K = 4;
export const GUIDANCE_VERSION = 'jf-ob-guidance/v1';

// grid: layouts.mjs buildBaseClearanceGrid(コースのすき間の格子)。戻り値 [{s(mod L), e_obstacle, e_target, lo, hi, width}]
export function passageTargets(grid, obstacles, {margin = .12, clear = .08, window = .65} = {}) {
  if (!obstacles?.length) return [];
  const added = obstacleOnlyGeometry(obstacles), pt = {x: 0, z: 0, tx: 0, tz: 0}, out = [];
  for (const o of obstacles) {
    const so = ((obstacleS(o) % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH, eyo = obstacleEy(o);
    const free = new Uint8Array(NOFF).fill(1);
    for (let d = -K; d <= K; d++) {
      const i = ((Math.round(so / DS) + d) % N + N) % N;
      loopPointAt(i * DS, pt);
      const yaw = Math.atan2(-pt.tz, pt.tx);
      for (let k = 0; k < NOFF; k++) {
        if (!free[k]) continue;
        const off = -OFF_MAX + k * STEP;
        if (Math.abs(off) > window || grid[i * NOFF + k] < clear) { free[k] = 0; continue; }
        if (clearance(added, pt.x + off * pt.tz, pt.z - off * pt.tx, yaw, .42, .19, .5) < clear) free[k] = 0;
      }
    }
    let best = null, run = 0, lo = 0;
    for (let k = 0; k <= NOFF; k++) {
      if (k < NOFF && free[k]) { if (!run) lo = -OFF_MAX + k * STEP; run++; continue; }
      if (run) { const hi = lo + (run - 1) * STEP; if (!best || hi - lo > best.hi - best.lo) best = {lo, hi}; }
      run = 0;
    }
    if (!best) { out.push({s: so, e_obstacle: eyo, e_target: null, lo: null, hi: null, width: 0}); continue; }
    const w = best.hi - best.lo, m = Math.min(margin, w / 2);
    const et = best.lo > eyo ? best.lo + m : best.hi < eyo ? best.hi - m : (best.lo + best.hi) / 2;
    out.push({s: so, e_obstacle: eyo, e_target: et, lo: best.lo, hi: best.hi, width: w});
  }
  return out;
}
// 車の R 上の s(周をまたいだ値)から、目標を使う障害物(手前 ahead から behind 過ぎるまでで一番近いもの)と ramp(1 = full より近い、ahead で 0)
export function activeTarget(targets, s, {ahead = 2.5, full = 1.0, behind = .3} = {}) {
  let best = null, bd = Infinity;
  for (const t of targets) {
    if (t.e_target == null) continue;
    let d = ((t.s - s) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH;
    if (d > LOOP_LENGTH / 2) d -= LOOP_LENGTH;
    if (d < -behind || d > ahead) continue;
    if (d < bd) { bd = d; best = t; }
  }
  if (!best) return null;
  return {t: best, ds: bd, ramp: bd <= full ? 1 : (ahead - bd) / (ahead - full)};
}
