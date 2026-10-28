// 打ち消しの票 A(反証 8-2 rehearsal-r2-vbefore-screen-vs-server)。本物の app.js をまるごと node の vm で読み、
// 本物の vbatSourceText・speedStartLines・autoConfirmText を呼ぶ(DOM・fetch・タイマーは自前の最小の偽物。指摘の sim_r2.js は使わない)。
// 使い方: node vsrc_app.js <app.js> <説明欄の JSON 配列>  → 1 行 1 JSON
'use strict';
const fs = require('fs');
const vm = require('vm');
const [, , appPath, descsJson] = process.argv;
const src = fs.readFileSync(appPath, 'utf8');

function mkEl(sel) {
  const cls = new Set();
  const el = {
    _sel: sel, src: '', href: '', value: '', checked: false, disabled: false, textContent: '', innerHTML: '', id: (sel || '').replace(/^#/, ''),
    dataset: {}, style: {}, children: [], options: [], files: [],
    classList: { add: (...a) => a.forEach((x) => cls.add(x)), remove: (...a) => a.forEach((x) => cls.delete(x)),
                 toggle: (x, f) => { const on = f === undefined ? !cls.has(x) : !!f; if (on) cls.add(x); else cls.delete(x); return on; },
                 contains: (x) => cls.has(x) },
    addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
    querySelector: () => mkEl('?'), querySelectorAll: () => [], appendChild: (c) => c, removeChild: (c) => c,
    insertAdjacentHTML() {}, setAttribute() {}, getAttribute: () => null, removeAttribute() {}, closest: () => null,
    focus() {}, blur() {}, click() {}, scrollIntoView() {}, getContext: () => new Proxy({}, { get: () => () => {} }),
    getBoundingClientRect: () => ({ width: 100, height: 100, left: 0, top: 0 }), play: () => Promise.resolve(), pause() {},
  };
  return el;
}
const els = new Map();
const document = {
  querySelector: (s) => { if (!els.has(s)) els.set(s, mkEl(s)); return els.get(s); },
  querySelectorAll: () => [], getElementById: (id) => document.querySelector('#' + id), createElement: (t) => mkEl(t),
  addEventListener() {}, body: mkEl('body'), visibilityState: 'hidden', activeElement: null, currentScript: null,
};
const ctx = {
  document, window: { addEventListener() {}, devicePixelRatio: 1, open() {} }, console,
  fetch: () => new Promise(() => {}), setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  requestAnimationFrame: () => 0, confirm: () => false, alert() {}, location: { href: '', search: '' }, navigator: {},
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} }, CSS: { escape: (s) => s }, Event: function () {},
  URLSearchParams, AbortController, performance: { now: () => 0 }, Date, Math, JSON, Promise,
};
ctx.window.document = document;
vm.createContext(ctx);
vm.runInContext(src, ctx, { filename: appPath });

const descs = JSON.parse(descsJson);
for (const d of descs) {
  ctx.__d = d;
  const out = vm.runInContext(`(() => {
    $('#dr-desc').value = __d;
    $('#dr-speed-ceiling').checked = true; $('#dr-speed-auto').checked = true;
    const st = { vbat: { supported: false }, params: { speed_ceiling: 1, speed_auto: 1, auto_target_mps: 4.27, auto_vref_v: 8.35 } };
    const sp = speedSpec(st);
    const line = speedStartLines(st).split('\\n').filter((x) => x.includes('自動合わせ'))[0] || '';
    return { desc: __d, vbatSourceText: vbatSourceText(st), start_line: line,
             confirm_src: (autoConfirmText(sp, st).split('\\n').filter((x) => x.startsWith('電圧の源'))[0] || '') };
  })()`, ctx);
  console.log(JSON.stringify(out));
}
