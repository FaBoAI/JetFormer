// 指摘 rehearsal-auto-vsrc の確かめ(打ち消しの票 A)。本物の app.js の関数を抜き出し、本物のサーバーの GET の status で流す。
// 使い方: node f1_auto_vsrc.js <GET /api/drive/status の JSON のファイル>
'use strict';
const fs = require('fs');
const { extract, el } = require('./extract.js');

const st = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const tpl = JSON.parse(fs.readFileSync('/home/jetson/jetformer/data/bench_candidates.json', 'utf8')).description_template;
const els = { '#dr-desc': el(), '#dr-speed-ceiling': el(), '#dr-speed-auto': el(), '#dr-record': el({ checked: true }) };
const $ = (s) => els[s] || el();
const drChecks = () => [{ checked: true }, { checked: true }];
const code = extract(['SPEED_FALLBACK', 'speedSpec', 'ceilSpecText', 'autoLapText', 'vbatSourceText', 'autoConfirmText',
                      'speedStartLines', 'prepStartLines']);
const api = new Function('$', 'drChecks', code + '\nreturn { speedSpec, vbatSourceText, autoConfirmText, speedStartLines, prepStartLines };')($, drChecks);

function line(txt, key) { return (txt.split('\n').find((l) => l.includes(key)) || '(無い)'); }
const out = [];
out.push('本物の GET の vbat: ' + JSON.stringify(st.vbat && { supported: st.vbat.supported, v: st.vbat.v }));
// ブロック 8(2-3)を押した直後 = 説明欄は雛形(V_before=__)
const desc0 = tpl.replace('{date}', '2026-11-03').replace('{block}', '8').replace('{id}', 'B').replace('{run}', 'clean_driving_20261011-nocmd-s3-20261011-024626');
els['#dr-desc'].value = desc0 + ' official=1';
out.push('説明欄(押した直後): ' + els['#dr-desc'].value);
// 4 章 4: 天井 ON → 自動合わせ ON の確認の文(説明欄に V_before を書く前)
els['#dr-speed-ceiling'].checked = true;
let t = api.autoConfirmText(api.speedSpec(st), st);
out.push('[4 章 4 の自動合わせのスイッチの確認の文] ' + line(t, '電圧の源'));
els['#dr-speed-auto'].checked = true;
// 4 章 5 で V_before を書かずに ▶(2-4 で書き忘れ)
let s = api.speedStartLines(st), p = api.prepStartLines();
out.push('[▶ の確認の文・V_before 書き忘れ] ' + line(s, '自動合わせ'));
out.push('[▶ の確認の文・ほかの注意(prepStartLines)] ' + JSON.stringify(p));
els['#dr-desc'].value = els['#dr-desc'].value.replace('battery=#__', 'battery=#A');
out.push('[battery=#A は書いて V_before は書き忘れ] 自動合わせの行: ' + line(api.speedStartLines(st), '自動合わせ') + ' / ほかの注意: ' + JSON.stringify(api.prepStartLines()));
// 4 章 5 で V_before=8.31 を書いて ▶
els['#dr-desc'].value = els['#dr-desc'].value.replace('V_before=__', 'V_before=8.31');
s = api.speedStartLines(st); p = api.prepStartLines();
out.push('[▶ の確認の文・V_before=8.31] ' + line(s, '自動合わせ'));
// 基板の電圧がある日(ファームあり)
const st2 = JSON.parse(JSON.stringify(st)); st2.vbat = Object.assign({}, st2.vbat, { supported: true, v: 8.3 });
els['#dr-desc'].value = desc0;
out.push('[ファームあり・説明欄に V_before なし] ' + line(api.speedStartLines(st2), '自動合わせ'));

// 手順書 4 章 6 の読み上げの行に「電圧の源」があるか
const rb = fs.readFileSync('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md', 'utf8').split('\n');
const i6 = rb.findIndex((l) => l.startsWith('   - 確認の文で **モデル・速度ゲイン'));
out.push(`手順書 ${i6 + 1} 行(4 章 6 の読み上げ)に「電圧の源」: ${rb[i6].includes('電圧の源')}・「自動合わせ: ON」: ${rb[i6].includes('速さのゲインの自動合わせ: ON')}`);
const i4 = rb.findIndex((l) => l.startsWith('4. **押したあとに入れるもの**'));
const i5 = rb.findIndex((l) => l.startsWith('5. **説明欄**'));
out.push(`手順書の順: 4 = ${i4 + 1} 行(スイッチ)・5 = ${i5 + 1} 行(説明欄)。4 章 1〜10 の中で「電圧の源」の出る行: ` +
  rb.slice(i4 - 4, rb.findIndex((l) => l.startsWith('## 5.'))).map((l, k) => (l.includes('電圧の源') ? i4 - 3 + k : null)).filter((x) => x).join(','));
console.log(out.join('\n'));
