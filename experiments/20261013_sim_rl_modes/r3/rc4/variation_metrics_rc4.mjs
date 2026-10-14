// 離脱復活 r4(2026-10-14): r3/rc/variation_metrics_rc3.mjs の写し(読み込む先だけ r3/rc4/rt)
// 離脱復活 r3(2026-10-14): rl/variation_metrics.mjs の写し(読み込む先だけ r3/rc/rt に)
// ばらつきの物差し VM1〜VM7(protocol.json variation_metrics の定義のとおり)と、乱しなしの線(nominal_line)。2026-10-13、sim_rl_modes。
// 入力は「シーン」の配列: {frames: [{f, s, ey, epsi, v, st(舵の指令 rad), dmin, l?}], s0, min_clearance?, laps?: [秒], events?, mask?}
//   s = 参照ループ R 上の進み(周をまたいで増え続ける値)、ey = R からの横ずれ(左が正)、epsi = 向きのずれ、v = 実際の速さ。
//   Node の走り(rl/episode.mjs の log + trace)と、記録した frames.jsonl の姿勢(sceneFromPoses で R に射影)の両方から作れる。
// probe/zoo.mjs の profile・spread・spreadT と同じ式(VM1 は教師 REC60 で 7.58 µs になることを試験で確かめる)。
import {LOOP_LENGTH, createFeatureTracker} from './rt/features.mjs';
import {ROOM_TABLE} from './rt/room_table.mjs';
import {createRng} from './rt/rng.mjs';

const L = LOOP_LENGTH, DS = .05, NB = Math.round(L / DS);
export const steerUs = a0 => 1442 + a0 * (a0 >= 0 ? 558 : 442);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
export const quantile = (arr, p) => { const b = [...arr].sort((x, y) => x - y); if (!b.length) return null; const i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// シーン → 絶対の s(mod L)の 5 cm のマスの値(開始から 3 m 以降の最初の通過)。値はそのマスの点を含むフレーム [s_k, s_{k+1}) のもの
export function binProfile(scene, key, {skip = 3, ds = DS} = {}) {
  const nb = Math.round(L / ds), out = new Float64Array(nb).fill(NaN), F = scene.frames, s0 = scene.s0 ?? F[0].s;
  for (let k = 0; k + 1 < F.length; k++) {
    const a = F[k].s, b = F[k + 1].s;
    if (a - s0 < skip || !(b > a)) continue;
    const val = typeof key === 'function' ? key(F[k]) : F[k][key];
    for (let i = Math.ceil(a / ds); i * ds < b; i++) {
      const j = ((i % nb) + nb) % nb;
      if (Number.isNaN(out[j])) out[j] = val;
    }
  }
  return out;
}
// 群の物差し: 全員がそろうマスで、各人と群の平均の |差| の平均
export function spread(profiles) {
  if (profiles.length < 2) return {mean_abs_dev: null, bins_used: 0};
  let sum = 0, n = 0, bins = 0;
  const nb = profiles[0].length;
  for (let b = 0; b < nb; b++) {
    let ok = true, m = 0;
    for (const p of profiles) { if (Number.isNaN(p[b])) { ok = false; break; } m += p[b]; }
    if (!ok) continue;
    m /= profiles.length;
    for (const p of profiles) { sum += Math.abs(p[b] - m); n++; }
    bins++;
  }
  return {mean_abs_dev: n ? sum / n : null, bins_used: bins};
}
export function VM1(scenes, frames = 600) {
  let sum = 0, n = 0;
  for (let t = 0; t < frames; t++) {
    const v = [];
    for (const sc of scenes) { const q = sc.frames[t]; if (!q) { v.length = 0; break; } v.push(steerUs(q.st / .52)); }
    if (v.length !== scenes.length) continue;
    const m = mean(v);
    for (const x of v) { sum += Math.abs(x - m); n++; }
  }
  return n ? sum / n : null;
}
export const VM2 = scenes => spread(scenes.map(sc => binProfile(sc, q => steerUs(q.st / .52))));
export const VM3 = scenes => spread(scenes.map(sc => binProfile(sc, 'ey')));
export function VM4(scenes) {
  const r = spread(scenes.map(sc => binProfile(sc, 'v')));
  const lap1 = scenes.map(sc => sc.laps?.[0]).filter(Number.isFinite), lap2 = scenes.map(sc => sc.laps?.[1]).filter(Number.isFinite);
  const st = a => a.length ? {n: a.length, min: Math.min(...a), median: quantile(a, .5), max: Math.max(...a), range: Math.max(...a) - Math.min(...a)} : {n: 0};
  return {...r, lap1: st(lap1), lap2: st(lap2)};
}
export function VM5(scenes) {
  const mins = scenes.map(sc => sc.min_clearance ?? Math.min(...sc.frames.map(q => q.dmin)));
  const all = scenes.flatMap(sc => sc.frames.map(q => q.dmin));
  return {scene_min: {min: Math.min(...mins), median: quantile(mins, .5)}, frame_p05: quantile(all, .05), frame_median: quantile(all, .5)};
}
export function VM6(scenes) {
  const ev = scenes.map(sc => sc.events?.length ?? 0), mf = scenes.map(sc => sc.mask ? sc.mask.reduce((a, b) => a + b, 0) / sc.mask.length : 0);
  const rt = scenes.flatMap(sc => (sc.recovery ?? []).filter(r => r.recovered_s != null).map(r => r.recovered_s));
  return {events_per_scene_mean: mean(ev), masked_fraction_mean: mean(mf), recovery_s: rt.length ? {n: rt.length, median: quantile(rt, .5), p90: quantile(rt, .9)} : {n: 0}};
}
// 状態の網羅: 開始から 3 m 以降のフレームを (s 0.25 m × e_y 0.05 m × 向きのずれ 0.1 rad × 速さ 0.5 m/s) のマスに入れた、訪れたマスの数。
// 比べる群は同じフレーム数 N(既定 9,000)を種 12345 で一様に間引いて数える(足りなければ全部)
export function VM7(scenes, N = 9000, seed = 12345) {
  const pts = [];
  for (const sc of scenes) { const s0 = sc.s0 ?? sc.frames[0].s; for (const q of sc.frames) if (q.s - s0 >= 3) pts.push(q); }
  let use = pts;
  if (pts.length > N) {
    const rng = createRng(seed), idx = Array.from(pts.keys());
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    use = idx.slice(0, N).map(i => pts[i]);
  }
  const cells = new Set();
  for (const q of use) {
    const sm = ((q.s % L) + L) % L;
    cells.add(`${Math.floor(sm / .25)}|${Math.floor(q.ey / .05)}|${Math.floor(q.epsi / .1)}|${Math.floor(q.v / .5)}`);
  }
  return {cells: cells.size, frames_used: use.length, frames_available: pts.length};
}
export function allMetrics(scenes, {vm7N = 9000} = {}) {
  return {scenes: scenes.length, VM1_us: VM1(scenes), VM2_us: VM2(scenes), VM3_m: VM3(scenes), VM4: VM4(scenes), VM5: VM5(scenes), VM6: VM6(scenes), VM7: VM7(scenes, vm7N)};
}
// 乱しなしの線(protocol definitions.nominal_line): 各回の開始から 3 m 以降の最初の通過の e_y・向きのずれを、絶対の s のマスで平均
// マスは room_table の格子(5 cm)に合わせる
export function nominalLine(scenes) {
  const n = ROOM_TABLE.n, ds = ROOM_TABLE.ds_m;
  const ey = scenes.map(sc => binProfile(sc, 'ey', {ds})), ps = scenes.map(sc => binProfile(sc, 'epsi', {ds}));
  const eyMean = [], psMean = [], cnt = [];
  for (let b = 0; b < n; b++) {
    const a = ey.map(p => p[b]).filter(x => !Number.isNaN(x)), c = ps.map(p => p[b]).filter(x => !Number.isNaN(x));
    cnt.push(a.length); eyMean.push(a.length ? mean(a) : null);
    psMean.push(c.length ? Math.atan2(mean(c.map(Math.sin)), mean(c.map(Math.cos))) : null);
  }
  // 抜けたマスは周期で線形に埋める(無い所は 0)
  const fill = arr => { const out = arr.slice(); for (let b = 0; b < n; b++) if (out[b] == null) { let l = b, r = b, k = 0; while (out[(l + n) % n] == null && k++ < n) l--; k = 0; while (out[r % n] == null && k++ < n) r++;
    const vl = out[((l % n) + n) % n], vr = out[r % n]; out[b] = vl == null || vr == null ? 0 : vl + (vr - vl) * (b - l) / (r - l); } return out; };
  return {schema: 'jf-sim-rl-nominal-line/v1', n, ds_m: ds, runs: scenes.length, ey_mean: fill(eyMean), psi_mean: fill(psMean), count: cnt};
}
export function nominalAt(line, s) {
  const u = (((s % L) + L) % L) / line.ds_m, i = Math.floor(u) % line.n, f = u - Math.floor(u), j = (i + 1) % line.n;
  return {ey: line.ey_mean[i] + (line.ey_mean[j] - line.ey_mean[i]) * f, psi: line.psi_mean[i] + wrap(line.psi_mean[j] - line.psi_mean[i]) * f};
}
// rl/episode.mjs の結果 → シーン(物差しの入力)
export function sceneFromEpisode(r) {
  const frames = r.log.map((q, i) => ({f: q.f, s: q.s, ey: q.ey, epsi: q.epsi, v: q.v, st: q.st, dmin: q.dmin, l: q.l}));
  return {frames, s0: frames[0]?.s, min_clearance: r.min_clearance, laps: (r.laps ?? []).map(l => l.elapsed), events: r.events};
}
// 記録の姿勢の列(フレームごとの操作前の姿勢 x, z, yaw, speed と route_index、舵の指令)→ シーン(R に射影)
export function sceneFromPoses(world, config, rows) {
  const tr = createFeatureTracker(world, config), out = new Float64Array(79), aux = {};
  const frames = [];
  rows.forEach((q, i) => {
    if (i === 0) tr.resetProgress(q.x, q.z, q.route_index ?? 0); else tr.advanceProgress(q.x, q.z);
    tr.observe({x: q.x, z: q.z, yaw: q.yaw, speed: q.speed, steering: q.steering ?? 0}, q.route_index ?? 0, 0, 0, out, aux);
    frames.push({f: i, s: aux.s, ey: aux.lateral_error, epsi: aux.heading_error, v: q.speed, st: q.st, dmin: aux.d_min, l: q.l ?? 1});
  });
  return {frames, s0: frames[0]?.s};
}
