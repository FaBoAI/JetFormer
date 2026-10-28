// 本物の app.js から、名前を指定した関数・const の宣言だけを文として抜き出す(打ち消しの票 A。読むだけ)
'use strict';
const fs = require('fs');
const APP = '/home/jetson/jetformer/jetformer/web/static/app.js';

function matchEnd(src, start) {          // start は '{' か '(' の位置。文字列・テンプレート・コメントを飛ばして対の閉じを探す
  const open = src[start], close = open === '{' ? '}' : ')';
  let depth = 0, i = start;
  const stack = [];
  while (i < src.length) {
    const c = src[i], n = src[i + 1];
    if (c === '/' && n === '/') { i = src.indexOf('\n', i); if (i < 0) return -1; continue; }
    if (c === '/' && n === '*') { i = src.indexOf('*/', i) + 2; continue; }
    if (c === "'" || c === '"') { i++; while (src[i] !== c) { if (src[i] === '\\') i++; i++; } i++; continue; }
    if (c === '`') {                      // テンプレートの ${...} の中も数える
      i++;
      while (src[i] !== '`') {
        if (src[i] === '\\') { i += 2; continue; }
        if (src[i] === '$' && src[i + 1] === '{') { const e = matchEnd(src, i + 1); i = e + 1; continue; }
        i++;
      }
      i++; continue;
    }
    if (c === '/' && /[=(,:;!&|?{}\[\n]\s*$/.test(src.slice(Math.max(0, i - 20), i))) {   // 正規表現のリテラル
      i++;
      let cls = false;
      while (i < src.length) {
        if (src[i] === '\\') { i += 2; continue; }
        if (src[i] === '[') cls = true; else if (src[i] === ']') cls = false;
        else if (src[i] === '/' && !cls) break;
        i++;
      }
      i++; continue;
    }
    if (c === open) depth++;
    else if (c === close) { depth--; if (depth === 0) return i; }
    i++;
  }
  return -1;
}

function extract(names) {
  const src = fs.readFileSync(APP, 'utf8');
  const out = [];
  for (const name of names) {
    let m = new RegExp(`^(async )?function ${name}\\(`, 'm').exec(src);
    if (m) {
      const brace = src.indexOf('{', src.indexOf(')', m.index));
      // 引数の ( ) の中に { があることがあるので、( の対を先に探す
      const paren = src.indexOf('(', m.index);
      const pe = matchEnd(src, paren);
      const b = src.indexOf('{', pe);
      const e = matchEnd(src, b);
      out.push(src.slice(m.index, e + 1));
      continue;
    }
    m = new RegExp(`^const ${name} = `, 'm').exec(src);
    if (m) {
      // 行の終わりの ; まで(複数行なら対の閉じのあと)
      let i = m.index + m[0].length;
      let end = src.indexOf(';\n', i);
      const firstNl = src.indexOf('\n', i);
      const j = src.slice(i, firstNl);
      if (/[({\[]\s*$/.test(j) || (j.includes('=>') && /\{\s*$/.test(j))) {
        const ob = src.slice(i, firstNl).search(/[({\[]\s*$/) + i;
        end = matchEnd(src, ob === -1 + i ? src.indexOf('{', i) : ob);
        end = src.indexOf(';', end);
      }
      out.push(src.slice(m.index, end + 1));
      continue;
    }
    throw new Error('見つからない: ' + name);
  }
  return out.join('\n');
}

// 偽の DOM の要素
function el(init) {
  const cls = new Set(init && init.hidden ? ['hidden'] : []);
  return Object.assign({
    value: '', checked: false, disabled: false, textContent: '', innerHTML: '', dataset: {},
    classList: { toggle(c, on) { if (on === undefined) on = !cls.has(c); on ? cls.add(c) : cls.delete(c); return on; },
                 add(c) { cls.add(c); }, remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); } },
    addEventListener() {},
  }, init || {});
}

module.exports = { extract, el, APP };
