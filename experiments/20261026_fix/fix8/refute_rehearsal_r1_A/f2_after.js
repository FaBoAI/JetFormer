// 指摘 rehearsal-after-memo の確かめ(打ち消しの票 A)。本物の app.js の renderAfter を抜き出し、作り物の status で流す。
// 使い方: node f2_after.js <本物の GET /api/drive/status の JSON(起動のあと走っていない今の形)>
'use strict';
const fs = require('fs');
const { extract, el } = require('./extract.js');

const real = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const els = {};
const $ = (s) => (els[s] = els[s] || el(s === '#dr-after' ? { hidden: true } : {}));
const store = {};
const localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
const code = extract(['AFTER_DONE_KEY', 'afterDone', 'descField', 'renderAfter', 'fillAfterVoltage']);
const api = new Function('$', 'localStorage', 'document', code + '\nreturn { renderAfter, AFTER_DONE_KEY };')($, localStorage, { activeElement: null });

const shown = () => !$('#dr-after').classList.contains('hidden');
const out = [];
function step(name, st) {
  api.renderAfter(st);
  out.push(`${name}: パネル ${shown() ? '出る' : '出ない'}${shown() ? '(記録 ' + $('#dr-after-log').textContent + ')' : ''}`);
}
const bench = (blk, log) => ({ running: false, dry_run: false, record_log: log, ai_no: blk + 1, stats: { ai_s: 200 },
  description: `bench=real_bench_20261023 day=2026-11-03 block=${blk} cand=B model=clean_driving_20261011-nocmd-s3-20261011-024626 battery=#A V_before=8.31 V_after=__ check=合格 memo= official=1`,
  vbat: { supported: false } });

// ① 3 章 5: 基準の設定で説明欄が空・収集したまま走った朝の確かめ(説明欄に bench= が無い)
step('① 朝の発進の前進の確かめ(説明欄が空)', { running: false, dry_run: false, record_log: '20261103-081500', ai_no: 1, stats: { ai_s: 5 }, description: '', vbat: { supported: false } });
step('①b 朝の確かめ(説明欄が「block=」の無い文)', { running: false, dry_run: false, record_log: '20261103-081500', ai_no: 1, stats: { ai_s: 5 }, description: '発進の前進の確かめ', vbat: { supported: false } });
// ② ブロックの本
step('② ブロック 1 の本が止まった', bench(1, '20261103-083000'));
// ③ ② のメモを書いた(afterDone)あと
localStorage.setItem(api.AFTER_DONE_KEY, '20261103-083000');
step('③ ② のメモを書いたあと', bench(1, '20261103-083000'));
// ④ 次の本が止まった → 次の本だけ
step('④ 次の本(ブロック 2)が止まった', bench(2, '20261103-083400'));
// ⑤ 走行中
step('⑤ 走行中', Object.assign(bench(3, '20261103-083800'), { running: true }));
// ⑥ 起こし直したあと(本物の今の status は起動のあと走っていない形 = record_log・description が無い)
out.push(`本物の GET の status に record_log: ${'record_log' in real}・description: ${'description' in real}(boot_at ${real.boot_at})`);
step('⑥ 起こし直したあと(本物の GET の status をそのまま)', real);
console.log(out.join('\n'));
