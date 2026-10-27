// 指摘 state-5 の確かめ: 古い画面の app.js の readDriveParams と走行開始の client を、偽の DOM で動かす(装置を開かない)
const fs = require('fs');
function grab(src, startPat, endPat) {
  const i = src.indexOf(startPat); if (i < 0) return '';
  const j = src.indexOf(endPat, i); return src.slice(i, j + endPat.length);
}
for (const f of process.argv.slice(2)) {
  const src = fs.readFileSync(f, 'utf8');
  const parts = [grab(src, 'const DR_PARAMS = [', '\n];'), grab(src, 'const DR_FLAGS = ', ';\n'),
                 grab(src, 'function readDriveParams() {', '\n}')];
  const els = {};
  const $ = (sel) => (els[sel] = els[sel] || { value: '0', checked: true });   // スイッチは全部 ON の画面でも
  const readDriveParams = new Function('$', parts.join('\n') + '\nreturn readDriveParams;')($);
  const p = readDriveParams();
  const m = src.match(/const body = \{ client: (\d+),/);
  console.log(JSON.stringify({ file: f.split('/').pop(), client: m ? +m[1] : null,
    has_launch_boost: 'launch_boost' in p, has_speed_ceiling: 'speed_ceiling' in p, keys: Object.keys(p) }));
}
