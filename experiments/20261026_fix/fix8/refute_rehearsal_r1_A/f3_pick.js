// 本物の app.js の renderBenchPick に、写しの bench の日を渡して選ぶ欄の行の数を数える(打ち消しの票 A)
'use strict';
const fs = require('fs');
const { extract, el } = require('./extract.js');
const b = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const days = b.days.map((x) => ({ date: x.date, blocks: x.blocks.map((k) => Object.assign({ drivable: true, label: '' }, k)) }));
const els = {};
const $ = (s) => (els[s] = els[s] || el());
const state = { bench: { applied: null }, drive: { st: { running: false } } };
const esc = (s) => String(s);
const code = extract(['renderBenchPick']);
const renderBenchPick = new Function('$', 'state', 'esc', code + '\nreturn renderBenchPick;')($, state, esc);
renderBenchPick({ exists: true, today: '2026-11-03', days });
const opts = (els['#dr-bench-pick'].innerHTML.match(/<option value="\d+:\d+"[^>]*>[^<]*/g) || []);
console.log('選ぶ欄のベンチマークの行: ' + opts.length);
for (const o of opts.slice(-5)) console.log('  ' + o.replace(/<option[^>]*>/, ''));
