#!/usr/bin/env node
// F-ID の確かめ(2026-10-14、line r4 の段。r4b で rt4b に): fid_*.json の snapshot を 1 フレームずつ新しい教師(rt4b/mode_runtime.mjs、modeRuntime = F-ID の mode_runtime)に与え、
// 操作と info.jf が書いたものと文字列として同じかを数える(ブラウザでの確かめは昇格の段)。
//   node r3/line4b/fid_check_ln4b.mjs <fid.json> <policy.json> [...fid.json]
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createModeTeacher} from './rt4b/mode_runtime.mjs';
const [, , ...args] = process.argv;
const policyPath = args.find(a => a.endsWith('policy.json'));
const weights = JSON.parse(fs.readFileSync(policyPath, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const out = [];
for (const f of args.filter(a => a !== policyPath)) {
  const F = JSON.parse(fs.readFileSync(f, 'utf8'));
  const t = createModeTeacher(F.world, F.frames[0].snapshot.config, F.variation, {mode: 'line', weights, params: F.params, policySha256: sha, modeRuntime: F.mode_runtime});
  let same = 0, diff = 0, first = null;
  for (const [k, fr] of F.frames.entries()) {
    const a = t.action(fr.snapshot);
    if (JSON.stringify(a) === JSON.stringify(fr.action)) same++; else { diff++; if (first == null) first = k; }
  }
  out.push({file: f.split('/').pop(), frames: F.frames.length, same, diff, first_diff: first, metadata_same: JSON.stringify(t.metadata) === JSON.stringify(F.metadata),
    shield_frames: F.frames.filter(q => q.action.info.jf.sh).length});
}
console.log(JSON.stringify(out));
