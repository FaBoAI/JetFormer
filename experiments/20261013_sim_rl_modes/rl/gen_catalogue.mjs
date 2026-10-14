#!/usr/bin/env node
// modes.json(jetformer_reports/sim_rl_modes_20261013/modes.json、設計の段で固定)から rl/runtime/catalogue.mjs を作る(2026-10-13)。
// ブラウザでは JSON の import 属性が使えないことがあるので、JS のモジュールにして配る(reference_loop.mjs と同じやり方)。
//   node rl/gen_catalogue.mjs
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/modes.json';
const EXPECTED = '635bf1ba79baa2eb7e71107d569819d5108d2096c25c36d72bdda96b98e5615a';   // protocol.json files.modes_json
const text = fs.readFileSync(SRC, 'utf8');
const sha = crypto.createHash('sha256').update(text).digest('hex');
if (sha !== EXPECTED) throw new Error(`modes.json の sha256 が protocol と違います: ${sha}`);
const obj = JSON.parse(text);
const out = `// 自動生成(rl/gen_catalogue.mjs)。手で変えない。元: ${SRC}\n// sha256(modes.json) = ${sha}\n` +
  `export const CATALOGUE_SHA256 = '${sha}';\nexport const CATALOGUE = Object.freeze(${JSON.stringify(obj)});\n`;
fs.writeFileSync(path.join(here, 'runtime/catalogue.mjs'), out);
console.log('書いた: rl/runtime/catalogue.mjs', sha);
