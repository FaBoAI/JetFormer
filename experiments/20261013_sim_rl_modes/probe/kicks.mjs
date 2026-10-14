// 「離脱復活」の作り方を確かめる: いまのベストラップの方策(乱れなしで学習)が、乱れのあとに立て直せるか。
//  A. 行動の乱れ(ブラウザの差し込み口でもできる方法): ある時刻から N フレーム、舵を端(±0.52 rad)に固定する。速さは方策のまま。
//  B. 行動の雑音(DART 風): 毎フレーム舵に雑音を足す(白色・OU)。
//  C. 状態の乱れ(Node の高速版だけでできる方法): peek() の state を書き換えて、向き・横の位置を瞬間的にずらす。
// どれも連続周回 20 秒・標準の開始。eval モード(sim の終了だけ。ゲート逃しでは止まらない = 逃したら周が無効になる)。
//   node probe/kicks.mjs
import {features, envCore, loadBestPolicy, runPolicyEpisode, writeJson, quantiles} from './common.mjs';

const {act, sha256} = loadBestPolicy();
const res = {schema: 'sim-rl-modes-probe-kicks/v1', policy_sha256: sha256, A_steer_lock: [], A_summary: {}, B_noise: [], C_state: [], C_summary: {}};
const T_KICKS = Array.from({length: 37}, (_, i) => 1 + .5 * i);   // 1.0〜19.0 秒

function outcome(r, kickFrame) {
  const s = r.summary, tr = r.trace;
  const after = tr.filter(q => q.f > kickFrame);
  const maxEy = after.length ? Math.max(...after.slice(0, 90).map(q => Math.abs(q.ey))) : null;
  // 立て直し: 乱れの終わりのあと |e_y| < 0.10 m が 15 フレーム続いた最初の時刻(乱れの始まりから)
  let rec = null, run = 0;
  for (const q of after) { run = Math.abs(q.ey) < .10 ? run + 1 : 0; if (run >= 15) { rec = (q.f - 14 - kickFrame) / 30; break; } }
  const gm = r.env.stats.gate_miss_frame;
  return {reason: s.sim_reason, collisions: s.collisions, collider: s.collision_events?.[0]?.collider, laps: s.completed_laps, valid_laps: s.valid_laps,
    gate_miss: gm !== null, survived: s.sim_reason === 'collection_duration_complete' && s.collisions === 0 && gm === null,
    max_abs_ey_3s: maxEy, recover_s: rec, min_clearance: s.min_clearance};
}

// ---- A. 舵の固定 ----
for (const N of [3, 6, 9, 12]) for (const sign of [1, -1]) {
  let ok = 0, n = 0, rec = [], gateMiss = 0, coll = 0;
  for (const t of T_KICKS) {
    const f0 = Math.round(t * 30);
    const hook = (env, frame, a) => (frame >= f0 && frame < f0 + N ? {a0: sign, a1: a[1]} : null);
    const r = await runPolicyEpisode({act, hook});
    const o = outcome(r, f0 + N);
    res.A_steer_lock.push({N, sign, t, ...o});
    n++; if (o.survived) { ok++; if (o.recover_s !== null) rec.push(o.recover_s); }
    if (o.gate_miss) gateMiss++; if (o.collisions) coll++;
  }
  res.A_summary[`N${N}_${sign > 0 ? 'left' : 'right'}`] = {n, survived: ok, survival_rate: ok / n, collisions: coll, gate_miss: gateMiss,
    recover_s_q: quantiles(rec, [0, .5, 1])};
}
// ---- B. 舵の雑音(白色 σ と、OU 過程 θ=0.15/フレーム) ----
function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = u => () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
for (const kind of ['white', 'ou']) for (const sigma of [.1, .2, .3, .5]) {
  let ok = 0, laps = [], coll = 0, gm = 0;
  for (let seed = 1; seed <= 12; seed++) {
    const g = gauss(rng(seed * 7919 + sigma * 1000)); let ou = 0;
    const hook = (env, frame, a) => {
      let e;
      if (kind === 'white') e = sigma * g();
      else { ou += -.15 * ou + sigma * Math.sqrt(2 * .15) * g(); e = ou; }   // 定常の標準偏差 ≈ sigma
      return {a0: Math.max(-1, Math.min(1, a[0] + e)), a1: a[1]};
    };
    const r = await runPolicyEpisode({act, hook});
    const s = r.summary, g0 = r.env.stats.gate_miss_frame;
    const survived = s.sim_reason === 'collection_duration_complete' && s.collisions === 0 && g0 === null;
    if (survived) { ok++; laps.push(...s.laps.map(l => l.elapsed)); }
    if (s.collisions) coll++; if (g0 !== null) gm++;
  }
  res.B_noise.push({kind, sigma_a0: sigma, sigma_rad: +(sigma * .52).toFixed(3), runs: 12, survived: ok, collisions: coll, gate_miss: gm, lap_q: quantiles(laps, [0, .5, 1])});
}
// ---- C. 状態の乱れ(高速版の peek().state を書き換える。ブラウザでは使えない) ----
for (const kind of ['yaw', 'lateral']) for (const mag of kind === 'yaw' ? [.15, .3, .5] : [.05, .10, .20]) for (const sign of [1, -1]) {
  let ok = 0, n = 0, rec = [], invalid = 0;
  for (const t of T_KICKS) {
    const f0 = Math.round(t * 30);
    let applied = false;
    const hook = (env, frame) => {
      if (frame === f0 && !applied) {
        applied = true;
        const st = env.sim.peek().state;   // 高速版: 参照(次の小区間で { ...state } として引き継がれる)
        if (kind === 'yaw') st.yaw = Math.atan2(Math.sin(st.yaw + sign * mag), Math.cos(st.yaw + sign * mag));
        else { st.x += sign * mag * -Math.sin(st.yaw); st.z += sign * mag * -Math.cos(st.yaw); }   // 左 = (−sin, −cos)
      }
      return null;
    };
    const r = await runPolicyEpisode({act, hook});
    const o = outcome(r, f0);
    res.C_state.push({kind, mag, sign, t, ...o});
    n++; if (o.survived) { ok++; if (o.recover_s !== null) rec.push(o.recover_s); }
  }
  res.C_summary[`${kind}_${mag}_${sign > 0 ? 'left' : 'right'}`] = {n, survived: ok, survival_rate: ok / n, recover_s_q: quantiles(rec, [0, .5, 1])};
}
// 状態を書き換えたあとの sim が、そのまま進む(ゲート・周の数え方が続く)ことの確認: 何もしない書き換え(同じ値の代入)で結果が同じ
{
  const a = await runPolicyEpisode({act});
  const b = await runPolicyEpisode({act, hook: (env, frame) => { if (frame === 150) { const st = env.sim.peek().state; st.yaw = st.yaw + 0; } return null; }});
  res.C_noop_identical = JSON.stringify(a.summary.laps) === JSON.stringify(b.summary.laps);
}
const p = writeJson('kicks.json', res);
console.log(JSON.stringify({A: res.A_summary, B: res.B_noise, C: res.C_summary, noop: res.C_noop_identical}, null, 1));
console.log('書いた:', p);
