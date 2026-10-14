// 開始の分布(protocol_line_r2 start_distribution、開始のすき間 ≥ 0.12 m)の中で、R に対して |Δψ| ≥ 1 rad の開始の割合(走らせない)
import {drawStarts} from './jobs_ln2.mjs';
const r = drawStarts(3000, 0x5a5e0001);
const hi = r.starts.filter(s => Math.abs(s.dpsi) >= 1.0), g = {};
for (const s of hi) g[s.gate] = (g[s.gate] ?? 0) + 1;
console.log(JSON.stringify({accepted: r.starts.length, drawn: r.drawn, abs_dpsi_ge_1rad: hi.length, share: hi.length / r.starts.length, by_gate: g}));
