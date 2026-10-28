#!/usr/bin/env python3
"""反証 8-1 回目 ui の指摘 toast-half-width-on-phone を打ち消す側で確かめる(票 A)。
本物の index.html を写した頁(style.css は本物を file:// で。app.js は読まない = 通信なし)を headless の Firefox(geckodriver・
この台本だけのポート 4477)で開き、走行タブを出し、#toast に本物の server._boot_refusal と同じ形の文(server.py の書式を写した)を
入れて、画面の幅 420・390・980 で箱の大きさと「停止中」のバッジ・走行タブの文・▶ との重なりを測る。jetformer-web には触れない。"""
import base64
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = Path('/home/jetson/jetformer/jetformer/web/static')
WD = 'http://127.0.0.1:4477'

# server._boot_refusal({'boot_at': 1000.0}, 2000.0) の書式(server.py 1041〜1043 行)。時刻は TZ=Asia/Tokyo の 1000・2000 秒
M = ('管理画面のサーバーが 09:33:20 に起動し直しました(この画面が読んだのは 09:16:40 に起動したサーバーの設定)。起動のたびに既定に戻るもの'
     '(試験の選択肢・天井の左・発進の前進の強さなど)が確認の文と違う値で走るので、始めません。表示が新しくなるのを待ち、止まっているときに'
     '朝の設定を入れ直して(手順書 7 章)、スイッチと確認の文を読んでから走行を開始してください')
REFUSAL = '走行は始まっていません(停止中)。' + M          # app.js の startRefusalText(止まっている・起き直しの断り)
SHORT = '停止しました'


def req(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(WD + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())['value']


def page():
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    html = html.replace('href="/static/style.css"', 'href="file://%s"' % (STATIC / 'style.css'))
    html = re.sub(r'<script src="/static/app.js"></script>', '', html)
    p = HERE / 'out' / 'a3_page.html'
    p.write_text(html, encoding='utf-8')
    return p


MEASURE = r'''
const [txt] = arguments;
document.querySelectorAll('.tab-page').forEach((p) => p.classList.toggle('active', p.id === 'tab-drive'));
const t = document.querySelector('#toast');
t.textContent = txt; t.className = 'toast error';
const r = (e) => { const b = e.getBoundingClientRect(); return {x: Math.round(b.left), y: Math.round(b.top), w: Math.round(b.width), h: Math.round(b.height)}; };
const ov = (a, b) => !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
const T = r(t);
const out = {vw: innerWidth, vh: innerHeight, chars: txt.length, toast: T, font: getComputedStyle(t).fontSize, maxw: getComputedStyle(t).maxWidth};
for (const sel of ['#dr-phase', '#dr-message', '#btn-dr-start']) {
  const e = document.querySelector(sel);
  e.scrollIntoView({block: 'end'});
  const T2 = r(t), E = r(e);
  out[sel] = {el: E, toast_at_scroll: T2, overlap: ov(T2, E)};
}
return out;
'''

AT_START = r'''
const r = (e) => { const b = e.getBoundingClientRect(); return {x: Math.round(b.left), y: Math.round(b.top), w: Math.round(b.width), h: Math.round(b.height)}; };
const ov = (a, b) => !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
const T = r(document.querySelector('#toast')), out = {toast: T};
for (const sel of ['#btn-dr-start', '#dr-phase', '#dr-message']) { const E = r(document.querySelector(sel)); out[sel] = {el: E, overlap: ov(T, E)}; }
return out;
'''


if __name__ == '__main__':
    pg = page()
    gd = subprocess.Popen(['/snap/bin/geckodriver', '--port', '4477'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = []
    try:
        for _ in range(60):
            try:
                req('GET', '/status')
                break
            except Exception:
                time.sleep(0.5)
        sid = req('POST', '/session', {'capabilities': {'alwaysMatch': {'moz:firefoxOptions': {'args': ['-headless']}}}}, timeout=120)['sessionId']
        try:
            # headless の Firefox の窓は 500 px より狭くならない(1 回目の出力: 420 を頼んでも見える幅 500)。狭い幅は iframe の中で測る
            req('POST', '/session/%s/window/rect' % sid, {'width': 1100, 'height': 900})
            for want in (420, 390, 980):
                wrap = HERE / 'out' / ('a3_wrap_%d.html' % want)
                wrap.write_text('<!doctype html><meta charset="utf-8"><body style="margin:0">'
                                '<iframe src="a3_page.html" style="width:%dpx;height:760px;border:0;display:block"></iframe>' % want,
                                encoding='utf-8')
                req('POST', '/session/%s/url' % sid, {'url': 'file://%s' % wrap})
                req('POST', '/session/%s/frame' % sid, {'id': 0})
                for name, txt in (('断りの出し直し', REFUSAL), ('短い文', SHORT)):
                    m = req('POST', '/session/%s/execute/sync' % sid, {'script': MEASURE, 'args': [txt]})
                    res.append(dict(m, name=name))
                    if name != '短い文':
                        req('POST', '/session/%s/execute/sync' % sid, {'script': "document.querySelector('#btn-dr-start').scrollIntoView({block: 'center'})", 'args': []})
                        m2 = req('POST', '/session/%s/execute/sync' % sid, {'script': AT_START, 'args': []})
                        res[-1]['at_start'] = m2
                        req('POST', '/session/%s/frame' % sid, {'id': None})
                        v = req('GET', '/session/%s/screenshot' % sid)
                        (HERE / 'out' / ('a3_toast_%d.png' % want)).write_bytes(base64.b64decode(v))
                        req('POST', '/session/%s/frame' % sid, {'id': 0})
        finally:
            req('DELETE', '/session/%s' % sid)
    finally:
        gd.terminate()
        gd.wait(timeout=30)
    lines = []
    for m in res:
        lines.append('幅 %(vw)s・高さ %(vh)s / %(name)s %(chars)s 字(font %(font)s・max-width %(maxw)s): 箱 %(toast)s' % m)
        for sel in ('#dr-phase', '#dr-message', '#btn-dr-start'):
            lines.append('    %s %s → 重なり %s(そのときの箱 %s)' % (sel, m[sel]['el'], m[sel]['overlap'], m[sel]['toast_at_scroll']))
        if 'at_start' in m:
            a = m['at_start']
            lines.append('    ▶ を画面の真ん中にしたとき: 箱 %s / ▶ %s 重なり %s / 「停止中」 %s 重なり %s / 走行タブの文 %s 重なり %s' % (
                a['toast'], a['#btn-dr-start']['el'], a['#btn-dr-start']['overlap'], a['#dr-phase']['el'], a['#dr-phase']['overlap'],
                a['#dr-message']['el'], a['#dr-message']['overlap']))
    txt = '\n'.join(lines)
    (HERE / 'out' / 'a3_toast_width.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)
