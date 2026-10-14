// 離脱復活 r4 の確認: ブラウザ(Firefox)の閉ループの走りを、自前の射影(ck_run.mjs project)で数え直し、Node の自前の行(rows_*.jsonl)と比べる(2026-10-15)
//   有効(valid_episode_r3: 600 フレーム・衝突 0・止まりなし・進み ≥ 1 周)・ゲートの外し(gate_miss_real)・出来事の列・終わり方
//   node ck_cl_compare_r4.mjs <browser.json> <runs のフォルダ> <組> <出力.json>
import fs from 'node:fs';
import {project, LOOP} from './ck_run.mjs';
const [bPath, runsDir, set, outPath] = process.argv.slice(2);
const B = JSON.parse(fs.readFileSync(bPath, 'utf8'));
const node = new Map();
for (const f of fs.readdirSync(runsDir).filter(f => /^rows_.*\.jsonl$/.test(f)))
  for (const l of fs.readFileSync(`${runsDir}/${f}`, 'utf8').split('\n')) { if (!l.trim()) continue; try { const r = JSON.parse(l); if (r.set === set) node.set(r.id, r); } catch {} }
const GM = 1.519 + .5;
function stall(v, events) {
  const T = v.length, ex = new Uint8Array(T);
  for (const e of events) if (e.type === 'speed_pulse' && e.kind === 'brake') for (let f = e.ps; f < Math.min(T, (e.pe ?? T) + 30); f++) ex[f] = 1;
  let run = 0, mx = 0;
  for (let f = 30; f < T; f++) { if (ex[f]) continue; if (Math.abs(v[f]) < .2) { run++; mx = Math.max(mx, run); } else run = 0; }
  return mx >= 30;
}
function gateMiss(s, g, lap) {
  let lastS = s[0], lastK = `${g[0]}|${lap[0]}`, n = 0;
  for (let f = 0; f < s.length; f++) { const k = `${g[f]}|${lap[f]}`; if (k !== lastK) { lastK = k; lastS = s[f]; } else if (s[f] - lastS > GM) { n++; lastS = s[f]; } }
  return n;
}
const rows = [];
const batches = Object.values(B.closed_loop_batch ?? {});
for (const b of batches) for (const r of (b.results ?? [])) {
  if (r.invalid_spawn) continue;
  let prev = null; const s = [], ey = [], epsi = [];
  for (let i = 0; i < r.x.length; i++) { const p = project(r.x[i], r.z[i], r.yaw[i], prev); s.push(p.s); ey.push(p.ey); epsi.push(p.epsi); prev = p.s; }
  const fin = project(r.final.x, r.final.z, r.final.yaw, prev);
  const evs = r.events.map(e => ({ps: e[0], pe: e[1], type: e[2], kind: e[3], cls: e[4]}));
  const validEp = r.reason === 'collection_duration_complete' && r.frames === 600 && r.collisions === 0;
  const st = stall(r.v, evs), prog = fin.s - s[0];
  const valid = validEp && !st && prog >= LOOP - 1e-9;
  const gm = gateMiss(s, r.g, r.lap);
  const n = node.get(r.id);
  let nv = null, ngm = null, nevs = null;
  if (n) {
    const nevt = n.events.map(e => ({ps: e.ps, pe: e.pe, type: e.type, kind: e.kind, cls: e.cls}));
    nv = n.valid_episode && !stall(n.pf.v, nevt) && (n.final.s - n.pf.s[0]) >= LOOP - 1e-9;
    let lastS = n.pf.s[0], lastK = `${n.pf.g[0]}|${n.pf.lap[0]}`; ngm = 0;
    for (let f = 0; f < n.pf.s.length; f++) { const k = `${n.pf.g[f]}|${n.pf.lap[f]}`; if (k !== lastK) { lastK = k; lastS = n.pf.s[f]; } else if (n.pf.s[f] - lastS > GM) { ngm++; lastS = n.pf.s[f]; } }
    nevs = JSON.stringify(nevt.map(e => [e.ps, e.pe, e.type, e.kind ?? null, e.cls]));
  }
  rows.push({id: r.id, browser: {reason: r.reason, frames: r.frames, collisions: r.collisions, valid, stall: st, progress: +prog.toFixed(3), gate_miss: gm, events: evs.length, shield: r.shield,
    l0: r.l.filter(x => x === 0).length},
    node: n ? {reason: n.reason, frames: n.frames, valid: nv, gate_miss: ngm, events: n.events.length} : null,
    same_events: n ? JSON.stringify(r.events) === nevs : null,
    // 自前の C2 の数え直し用(ck_browser_c2_r4.py): フレームごとの s・e_y・ψ のずれ・速さと出来事
    pf: {s: s.map(x => +x.toFixed(4)), ey: ey.map(x => +x.toFixed(4)), epsi: epsi.map(x => +x.toFixed(4)), v: r.v}, events_full: r.events, final_s: fin.s});
}
const have = rows.filter(r => r.node);
const sum = {n: rows.length, compared: have.length,
  browser_valid: rows.filter(r => r.browser.valid).length, node_valid: have.filter(r => r.node.valid).length,
  same_valid: have.filter(r => r.browser.valid === r.node.valid).length,
  browser_gate_miss_valid: rows.filter(r => r.browser.valid && r.browser.gate_miss > 0).length,
  node_gate_miss_valid: have.filter(r => r.node.valid && r.node.gate_miss > 0).length,
  same_gate_miss: have.filter(r => (r.browser.gate_miss > 0) === (r.node.gate_miss > 0)).length,
  node_ok_browser_miss: have.filter(r => r.node.valid && !r.node.gate_miss && r.browser.gate_miss > 0).map(r => r.id),
  node_ok_browser_invalid: have.filter(r => r.node.valid && !r.node.gate_miss && !r.browser.valid).map(r => r.id),
  same_event_list: have.filter(r => r.same_events).length,
  browser_collisions: rows.filter(r => r.browser.collisions).length, browser_stall: rows.filter(r => r.browser.stall).length};
fs.writeFileSync(outPath, JSON.stringify({summary: sum, rows}));
console.log(JSON.stringify(sum, null, 1));
