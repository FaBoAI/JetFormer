# -*- coding: utf-8 -*-
"""直す 8-2 回目・係 ui(rehearsal-r2-vbefore-screen-vs-server): 同じ説明欄で、画面(app.js の vbatSourceText。基板の電圧なし)と
サーバー(本物の speed_auto.v_before_of と drive._SpeedAuto。説明欄はサーバーが走行開始で頭に ai_no= を付けた形)の電圧の源を並べる。
本物の jetformer-web・基板には触れない(_SpeedAuto は計算だけ。app.js は node で vbatSourceText だけを切り出して動かす)。

  /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B \\
      experiments/20261026_fix/fix8/fix_ui_r2/vsrc/compare_vsrc.py <app.js> <出力の名前>
出力 out/<名前>.txt(行ごとに 説明欄 | 画面 | サーバー | 合う)
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = '/home/jetson/jetformer'
sys.path.insert(0, ROOT)
from jetformer import drive, speed_auto          # noqa: E402
from jetformer.web import bench as bench_web     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {speed_auto.SRC_NONE: 'なし', speed_auto.SRC_BOARD: '基板', speed_auto.SRC_DESC: '説明欄'}
# 入力の表: 指摘・票 A・票 B の形と、決めの境目(範囲の端・末尾の V・2 つ目の V_before・Vb_before・空白の字・全角・下線・指数・inf/nan)
TABLE = [
    'battery=#A V_before=8.31 check=合格 memo=', 'battery=#A V_before=831 check=合格', 'battery=#A V_before=8,31 check=合格',
    'battery=#A V_before=0.83 check=合格', 'battery=#A V_before=8.31V', 'battery=#A V_before=8.31v', 'battery=#A V_before=8.31VV',
    'battery=#A V_before=__ V_after=__', 'battery=#A V_before=８.３１', 'battery=#A V_before=８．３１', 'battery=#A V_before=9.5',
    'battery=#A V_before=8.31, check=合格', 'battery=#A V_before=4.15', 'V_before=6.0', 'V_before=9.0', 'V_before=5.999',
    'V_before=9.001', 'V_before=6', 'V_before=8.', 'V_before=.831e1', 'V_before=8.31e0', 'V_before=8_31', 'V_before=8.3_1',
    'V_before=8__3', 'V_before=+8.31', 'V_before=-8.31', 'V_before=nan V_before=8.31', 'V_before=x V_before=8.31',
    'V_before=8.31 V_before=9.9', 'V_before=inf', 'V_before=Infinity', 'Vb_before=8.27', 'V_before=__ Vb_before=8.27',
    'V_before=9.5 Vb_before=8.27', 'V_before=8.31 Vb_before=8.27', 'v_before=8.31', 'V_before =8.31', 'V_before= 8.31',
    'V_before=8.31　check=合格', 'V_before=8.31 check=合格', 'V_before=8.31\x1ccheck', 'V_before=8.31﻿',
    'memo=V_before=8.31', 'xV_before=8.31', 'V_before=8.31memo', 'V_before=٨.٣١', 'V_before=8.31 V', '', 'V_before=',
    'battery=#A V_before=8.31\ncheck=合格', 'V_before=0x8', 'V_before=8.31e', 'V_before=1e0_0',
]


def screen(app_js, descs):
    js = open(app_js, encoding='utf-8').read()
    m = re.search(r'function vbatSourceText\(st\) \{.*?\n\}', js, re.S)
    src = m.group(0) + '''
const DESCS = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf-8'));
let CUR = '';
const $ = (q) => ({ value: CUR });
const out = [];
for (const d of DESCS) { CUR = d; out.push(vbatSourceText({ vbat: { supported: false } })); }
console.log(JSON.stringify(out));
'''
    with tempfile.TemporaryDirectory() as td:
        p, q = os.path.join(td, 'v.js'), os.path.join(td, 'd.json')
        open(p, 'w', encoding='utf-8').write(src)
        open(q, 'w', encoding='utf-8').write(json.dumps(descs, ensure_ascii=False))
        r = subprocess.run(['node', p, q], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def server(desc):
    params = dict(drive.DEFAULT_PARAMS, speed_ceiling=1, speed_auto=1)
    sa = drive._SpeedAuto(params, bench_web.with_ai_no(desc, 7))      # サーバーが走行開始で足す形
    sa.gain(1_000_000_000, 1.00, 0.0, [])                            # 止まっている・基板の電圧のサンプルなし
    return speed_auto.v_before_of(desc), SRC.get(sa.src, sa.src), sa.v_used


def main():
    app_js, name = sys.argv[1], sys.argv[2]
    scr = screen(app_js, TABLE)
    lines = ['# 画面 = %s の vbatSourceText(基板の電圧なし)・サーバー = speed_auto.v_before_of と drive._SpeedAuto(ai_no= を頭に)' % app_js,
             '# 列: 説明欄 | 画面 | サーバーの v_before_of・電圧の源・使う電圧 | 合う']
    bad = 0
    for d, s in zip(TABLE, scr):
        v, src, used = server(d)
        s_short = '説明欄' if s.startswith('説明欄') else ('なし' if s.startswith('なし') else s)
        ok = s_short == src
        bad += not ok
        lines.append('%-48s | %s | %s %s %s | %s' % (json.dumps(d, ensure_ascii=False), s_short, v, src, used, '合う' if ok else '**違う**'))
    lines.append('違う: %d / %d' % (bad, len(TABLE)))
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    open(os.path.join(HERE, 'out', name + '.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
