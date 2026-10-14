#!/usr/bin/env node
// no_wall r3 の組を走らせる(2026-10-14、r3 の学習の段)。nw3_sets.mjs の回の一覧を run_nw3r.mjs(元の sim-core.js・runtime_v4 = シールド v4)で走らせ、
// 1 行 1 回の JSONL に書く。止めても同じコマンドで続きから(書き終えた回は飛ばす)。
//   node r3/nw/run_sets_nw3.mjs --which test|validation --policy <policy.json> --out <dir> [--jobs 3] [--sets NW-N300-R3,...] [--limit N]
// 子プロセスに回を 1 つずつ配る(回の長さがばらつくので、順番の割り当てではなく空いた子に次を渡す)。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here = fileURLToPath(import.meta.url);
const args = process.argv.slice(2);
const opt = (k, d = null) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const key = j => `${j.set}|${j.id}`;

if (opt('--child') != null) {
  // 子: 親から {i, job} を受け取り、走らせて結果を返す
  const {runOne} = await import('./run_nw3r.mjs');
  process.on('message', async m => {
    if (m.quit) process.exit(0);
    const t = Date.now();
    let r;
    try { r = await runOne(m.job.spec); } catch (e) { r = {error: String(e && e.stack || e)}; }
    process.send({i: m.i, row: {i: m.i, set: m.job.set, id: m.job.id, judged: m.job.judged, physics: m.job.physics, corner: m.job.corner ?? null, of: m.job.of ?? null,
      name: m.job.name ?? null, wall_s: (Date.now() - t) / 1000, ...r}});
  });
  process.send({ready: true});
} else {
  const which = opt('--which'), out = opt('--out'), policy = opt('--policy'), jobs = +(opt('--jobs') ?? 3);
  if (!which || !out || !policy) throw new Error('--which・--out・--policy が要ります');
  const {buildJobs, PROTOCOL_PATH} = await import('./nw3_sets.mjs');
  const ALL = buildJobs(which).map((j, i) => ({...j, i}));
  let J = ALL;
  const only = opt('--sets');
  if (only) J = J.filter(j => only.split(',').includes(j.set));
  if (opt('--limit')) J = J.slice(0, +opt('--limit'));
  const policyAbs = path.resolve(policy), psha = sha(policyAbs);
  for (const j of J) if (j.spec.mode === 'no_wall') j.spec.policyPath = policyAbs;
  // --rt: 走らせる仕組みのフォルダー(既定 r3/nw/runtime_v4)。--rt-tag で行の鍵を分ける(同じ組を別の仕組みで走らせて比べるとき)
  if (opt('--rt')) for (const j of J) j.spec.rtDir = path.resolve(opt('--rt'));
  fs.mkdirSync(out, {recursive: true});
  const partPath = path.join(out, `rows_${which}.part.jsonl`), finalPath = path.join(out, `rows_${which}.jsonl`);
  const done = new Map();
  for (const p of [finalPath, partPath]) if (fs.existsSync(p)) for (const line of fs.readFileSync(p, 'utf8').split('\n')) {
    if (!line) continue;
    try { const r = JSON.parse(line); if (!r.error && r.policy_sha_full === psha) done.set(`${r.set}|${r.id}`, r); } catch { /* 途中で切れた行は捨てる */ }
  }
  const todo = J.filter(j => !done.has(key(j)));
  const here2 = path.dirname(here);
  const meta = {schema: 'jf-nw3-rows/v1', which, policy: policyAbs, policy_sha256: psha, protocol_sha256: sha(PROTOCOL_PATH), jobs_total: J.length,
    resumed_done: J.length - todo.length, started_at: new Date().toISOString(),
    code_sha256: Object.fromEntries(['run_nw3r.mjs', 'nw3_sets.mjs', 'run_sets_nw3.mjs', 'own_geom.mjs', 'fast_core_nw3.mjs', 'runtime_v4/shield.mjs', 'runtime_v4/mode_runtime.mjs']
      .map(f => [f, sha(path.join(here2, f))]))};
  const rtDir = opt('--rt') ? path.resolve(opt('--rt')) : path.join(here2, 'runtime_v4');
  meta.rt_dir = rtDir; meta.rt_shield_sha256 = sha(path.join(rtDir, 'shield.mjs')); meta.rt_mode_runtime_sha256 = sha(path.join(rtDir, 'mode_runtime.mjs'));
  console.error(`[${new Date().toTimeString().slice(0, 8)}] ${which}: 回 ${J.length}(済み ${J.length - todo.length}・残り ${todo.length})方策 ${psha.slice(0, 12)}`);
  const fd = fs.openSync(partPath, 'a');
  let next = 0, fin = 0;
  const t0 = Date.now();
  await new Promise((resolve, reject) => {
    if (!todo.length) return resolve();
    let alive = 0;
    for (let k = 0; k < Math.min(jobs, todo.length); k++) {
      const ch = fork(here, ['--child', String(k)], {stdio: ['ignore', 'inherit', 'inherit', 'ipc']});
      alive++;
      const give = () => { if (next < todo.length) { const j = todo[next++]; ch.send({i: j.i, job: j}); } else ch.send({quit: true}); };
      ch.on('message', m => {
        if (m.ready) return give();
        m.row.policy_sha_full = psha;   // この行を作った走らせ方の no_wall の方策(FAST の行は最速の方策で走らせたもの)
        fs.writeSync(fd, JSON.stringify(m.row) + '\n');
        fin++;
        if (fin % 50 === 0) console.error(`[${new Date().toTimeString().slice(0, 8)}] ${fin}/${todo.length}(${((Date.now() - t0) / 1000).toFixed(0)} 秒)`);
        give();
      });
      ch.on('exit', c => { alive--; if (c !== 0) reject(new Error('子プロセスが失敗しました ' + c)); else if (alive === 0) resolve(); });
    }
  });
  fs.closeSync(fd);
  // まとめ: 一覧の順に並べ、同じ回は 1 つに
  const rows = new Map(done);
  for (const line of fs.readFileSync(partPath, 'utf8').split('\n')) {
    if (!line) continue;
    try { const r = JSON.parse(line); if (r.policy_sha_full === psha) rows.set(`${r.set}|${r.id}`, r); } catch { /* 捨てる */ }
  }
  // 一覧の全部の順で(--sets で一部だけ走らせたときも、前に走らせた回を消さない)
  const ordered = ALL.map(j => rows.get(key(j))).filter(Boolean);
  const errors = ordered.filter(r => r.error).length;
  const missing = J.filter(j => !rows.has(key(j))).length;
  fs.writeFileSync(finalPath, ordered.map(r => JSON.stringify(r)).join('\n') + '\n');
  fs.rmSync(partPath, {force: true});
  meta.finished_at = new Date().toISOString(); meta.rows = ordered.length; meta.errors = errors; meta.missing_of_requested = missing; meta.wall_s = (Date.now() - t0) / 1000;
  meta.jobs_all = ALL.length;
  fs.writeFileSync(path.join(out, `rows_${which}.meta.json`), JSON.stringify(meta, null, 1) + '\n');
  console.error(`[${new Date().toTimeString().slice(0, 8)}] 終わり: 頼んだ ${J.length} 回のうち足りない ${missing}・全部の行 ${ordered.length}/${ALL.length}(エラー ${errors})、${meta.wall_s.toFixed(0)} 秒`);
  if (missing || errors) process.exitCode = 2;
}
