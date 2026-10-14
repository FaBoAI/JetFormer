#!/usr/bin/env node
// 周回の独立な確認(コマンド)。2026-10-13、RL 環境の段
//
//   node rl/verify_lap.mjs <sim のレポート.json> [--min-laps 2] [--min-clearance 0.03] [--allow-failure] [--out 結果.json]
//
// レポートは sim.report() + world + context.start_variation の形(rl/eval_sets.mjs の record、アプリの「JSON で保存」、
// results/shortcut-rc-learned-lap.json など)。中身は rl/lap_audit.mjs:
//   元の sim-core.js(sha256 確認済みのコピー)で再生 → 保存した軌跡・集計と一致、
//   自前の積分と数え直し(ゲート・黒シートの車幅・衝突・周回時間)→ sim と一致、
//   各周が 33 ゲート・黒シートの有効な横断ちょうど 1 回・衝突 0。
// ユーザーの qa/verify-learned-lap.mjs は「外部 ONNX・単周・標準の開始」の記録だけを数えるので、RL の周はこちらで確かめる。
// 通れば終了コード 0、通らなければ 1(--allow-failure なら 0)。
import fs from 'node:fs';
import {verifyReport, loadReport} from './lap_audit.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback = null) => { const i = args.indexOf(name); if (i < 0) return fallback; const v = args[i + 1]; args.splice(i, 2); return v; };
const flag = name => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; };
const out = opt('--out'), minLaps = opt('--min-laps'), minClearance = Number(opt('--min-clearance', '0')), allowFailure = flag('--allow-failure');
if (!args[0]) { console.error('使い方: node rl/verify_lap.mjs <report.json> [--min-laps N] [--min-clearance m] [--out result.json]'); process.exit(2); }
const report = loadReport(args[0]);
const result = await verifyReport(report, {minLaps: minLaps === null ? null : Number(minLaps), minClearance});
result.source = args[0];
if (out) fs.writeFileSync(out, JSON.stringify(result, null, 1) + '\n');
const {collision_events_own, ...shown} = result;
console.log(JSON.stringify(shown, null, 1));
console.error(result.pass ? `合格: ${result.laps_own.map(l => l.elapsed.toFixed(6) + ' 秒').join('、') || '(周なし)'}` : `不合格: ${result.failures.slice(0, 5).join(' / ')}`);
process.exit(result.pass || allowFailure ? 0 : 1);
