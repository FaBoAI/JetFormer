// ブラウザが読む入口(dist/dart/entry-D1〜D3.js)を Node で動かし、記録の前の予想(prediction_Dn.json)と同じ操作列になることの試験。
// k の表(plan_Dn.mjs)が効いていること・info に jf が無いこと・info.dart の形も確かめる。2026-10-14、sim_dart_20261014。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {runCondition, baseWorld, SIM_ROOT} from '../lib/app_emulation.mjs';

const R = '/home/jetson/jetformer_reports/sim_dart_20261014/record_prep';
const pred = d => JSON.parse(fs.readFileSync(`${R}/prediction_${d}.json`, 'utf8'));
const plan = d => JSON.parse(fs.readFileSync(`${R}/plan_${d}.json`, 'utf8'));

async function runEntry(d, i) {
  const mod = await import(pathToFileURL(`${SIM_ROOT}/dist/dart/entry-${d}.js`).href);
  globalThis.__dartOverride = {};
  const hash = crypto.createHash('sha256'), buf = new Float64Array(2), infos = [];
  const {report} = await runCondition(plan(d)[i], mod, {world: await baseWorld(), onFrame: (b, a) => {
    buf[0] = a.steering; buf[1] = a.speed; hash.update(Buffer.from(buf.buffer)); infos.push(a.info);
  }});
  return {sha: hash.digest('hex'), infos, report, metadata: mod.DART_TEACHER_METADATA};
}

for (const [d, idxs] of [['D1', [0]], ['D2', [0]], ['D3', [0, 2]]]) {
  test(`${d}: 入口の操作列が予想と同じ・info の形`, async () => {
    const P = pred(d);
    for (const i of idxs) {
      const r = await runEntry(d, i);
      const s = P.scenes.find(x => x.index === i);
      const want = P.summary.noise ? s.chosen.actions_sha256 : s.clean.actions_sha256;
      assert.equal(r.sha, want, `${d} ${i} の操作列`);
      assert.equal(r.infos.length, r.report.trace.length);
      for (const inf of r.infos) { assert.ok(inf.dart); assert.equal(inf.jf, undefined); assert.equal(inf.dart.v, 1); assert.equal(inf.dart.clean.length, 2); }
      const p = r.infos[0].dart.p;
      assert.equal(p.dataset, d); assert.equal(p.seed, plan(d)[i].seed);
      if (P.summary.noise) {
        assert.equal(p.k, s.chosen.k); assert.equal(p.scale, s.chosen.scale);
        assert.ok(r.infos.some(x => x.dart.noise), '乱しが無い');
      } else assert.ok(r.infos.every(x => x.dart.noise === null));
      assert.equal(r.metadata.id, JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_dart_20261014/protocol.json', 'utf8')).recording.teacher_metadata.ids[d]);
      assert.equal(r.metadata.not_learned_policy, false);
    }
  });
}
