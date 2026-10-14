#!/usr/bin/env node
// 障害物よけ r2 で env_modes.mjs を変える前に、ほかの run(base・no_wall・recover・cond・obstacle・speed・line)の学習の環境が
// 今のファイルと同じに動くことを確かめる(2026-10-14 学習の段 obstacle r2。rl/speed_r2/compare_before.mjs と同じ作り方)。
// 新しい版(引数のパス)を rl/ に一時的に置いて import し、同じ種・同じ行動の列で開始・観測・報酬・終わり・集計を比べる。終わったら一時の写しを消す。
//   node rl/ob2/compare_before.mjs <新しい env_modes.mjs>
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import * as core from '../fast_core_modes.mjs';
import {createRng} from '../runtime/rng.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), rl = path.resolve(here, '..');
const tmp = path.join(rl, `.env_modes_new_${process.pid}.mjs`);
fs.copyFileSync(process.argv[2], tmp);
try {
  const B = await import(pathToFileURL(path.join(rl, 'env_modes.mjs')).href), A = await import(pathToFileURL(tmp).href);
  let checked = 0;
  for (const run of (process.argv[3] ?? 'base,no_wall,recover,cond,obstacle,speed,line').split(',')) {
    const ea = new A.ModeEnv({core, run}), eb = new B.ModeEnv({core, run});
    if (JSON.stringify(ea.cfg) !== JSON.stringify(eb.cfg)) throw new Error(`${run}: 設定が違います`);
    const ra = createRng(1234), rb = createRng(1234), act = createRng(99);
    for (let ep = 0; ep < 4; ep++) {
      ea.resetRandom(ra); eb.resetRandom(rb);
      if (JSON.stringify(ea.start) !== JSON.stringify(eb.start)) throw new Error(`${run}: 開始が違います`);
      if (JSON.stringify(ea.obstacles) !== JSON.stringify(eb.obstacles)) throw new Error(`${run}: 配置が違います`);
      for (let k = 0; k < 200 && !ea.done; k++) {
        const a0 = act.uniform() * .6 - .3, a1 = act.uniform() * 1.2 - .2;
        const x = ea.step(a0, a1), y = eb.step(a0, a1);
        if (x.reward !== y.reward || x.reason !== y.reason || x.terminated !== y.terminated) throw new Error(`${run}: ${ep} ${k} の報酬・終わりが違います`);
        for (let j = 0; j < ea.obs.length; j++) if (ea.obs[j] !== eb.obs[j]) throw new Error(`${run}: 観測が違います`);
        checked++;
      }
      if (JSON.stringify(ea.summary()) !== JSON.stringify(eb.summary())) throw new Error(`${run}: 集計が違います`);
    }
    console.log(`${run}: 今のファイルと同じ(4 回)`);
  }
  console.log(`全部同じ: ${checked} ステップ`);
} finally { fs.rmSync(tmp, {force: true}); }
