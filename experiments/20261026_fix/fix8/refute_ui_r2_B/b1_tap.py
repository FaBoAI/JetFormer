"""打ち消しの票 B(反証 8-2 回目・係 ui の float-stop-half-width-under-toast)。見方 B: 11/3 の使い方で本当に困るか。

headless の Firefox(geckodriver)で、作り物のサーバー fake_server_b.py(127.0.0.1:18823。POST /api/drive/stop だけ覚えて 200)に
本物の index.html・app.js・style.css。スマホの幅・高さの iframe(左上 0,0・枠なし = iframe の中の座標と窓の座標が同じ)。

1 つの画面の大きさごとに:
  1. 止まっている走行タブで ▶ を画面の 上 / 真ん中 / 下寄り に置く(人が ▶ を押す場所)。
  2. 作り物を「走行中」にして(▶ が通った形)、本物の toast() に 11/3 の毎回の走行開始の「試し」の文(error・6 秒)を渡す。
  3. 測る: トースト・浮いている ■・カードの ■ の箱、カードの ■ が画面の中に全部見えるか、トーストかぶさり。
  4. トーストと浮いている ■ が重なる所の真ん中を、WebDriver の本物のポインター(mouse と touch)で押す → 作り物のサーバーに
     POST /api/drive/stop(reason 停止ボタン)が届いたか。elementFromPoint の答えも。比べ: トーストの ■ の外の所を押す。
  5. トーストが 6 秒で消えるか(5.5 秒・6.5 秒で見る)= 手順書 4 章 8 の ■(3 周 約 25 秒 + R1 を抜ける約 3 秒のあと)の時刻には無い。
出力 out/b1_tap.txt。本物の jetformer-web・基板には触れない(GET もしない。jf-heavy の始める前の確かめを除く)。
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
WD = 'http://127.0.0.1:4523'
SRV = 'http://127.0.0.1:18823'
TRIAL = 'ブロックの計画と違うので「試し」(cand=EXP)として記録します: throttle_gain 0.9(基準 1.0)、speed_ceiling 1(基準 0)、ceil_left_mps 4.2(基準 4.1)'


def req(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(WD + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())['value']


def srv(path):
    with urllib.request.urlopen(SRV + path, timeout=10) as f:
        return json.loads(f.read().decode())


def fake(**kw):
    return srv('/fake/set?j=' + urllib.parse.quote(json.dumps(kw)))


class B:
    def __init__(self):
        self.srv = subprocess.Popen(['python3', '-I', '-B', str(HERE / 'fake_server_b.py'), '18823'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.gd = subprocess.Popen(['/snap/bin/geckodriver', '--port', '4523'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                req('GET', '/status'); srv('/fake/posts')
                break
            except Exception:
                time.sleep(0.5)
        v = req('POST', '/session', {'capabilities': {'alwaysMatch': {'moz:firefoxOptions': {'args': ['-headless']}}}}, timeout=120)
        self.sid = v['sessionId']
        req('POST', '/session/%s/window/rect' % self.sid, {'width': 1000, 'height': 1000})

    def go(self, url):
        req('POST', '/session/%s/url' % self.sid, {'url': url})

    def js(self, script, *args):
        return req('POST', '/session/%s/execute/sync' % self.sid, {'script': script, 'args': list(args)})

    def frame(self, i):
        req('POST', '/session/%s/frame' % self.sid, {'id': i})

    def tap(self, x, y, kind):
        a = {'actions': [{'type': 'pointer', 'id': 'p_' + kind, 'parameters': {'pointerType': kind},
                          'actions': [{'type': 'pointerMove', 'duration': 0, 'x': int(x), 'y': int(y), 'origin': 'viewport'},
                                      {'type': 'pointerDown', 'button': 0}, {'type': 'pause', 'duration': 80}, {'type': 'pointerUp', 'button': 0}]}]}
        try:
            req('POST', '/session/%s/actions' % self.sid, a)
            req('DELETE', '/session/%s/actions' % self.sid)
            return 'ok'
        except Exception as e:  # touch を受けない版なら書く
            return 'err ' + str(e)[:120]

    def shot(self, path):
        v = req('GET', '/session/%s/screenshot' % self.sid)
        open(path, 'wb').write(base64.b64decode(v))

    def close(self):
        try:
            req('DELETE', '/session/%s' % self.sid)
        except Exception:
            pass
        for p in (self.gd, self.srv):
            p.terminate()
            try:
                p.wait(10)
            except Exception:
                p.kill()


MEAS = """const q = (s) => { const e = document.querySelector(s); if (!e || e.classList.contains('hidden') || getComputedStyle(e).display === 'none') return null;
  const r = e.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; };
const t = document.querySelector('#toast');
return {vh: innerHeight, vw: document.documentElement.clientWidth, toast: q('#toast'), fstop: q('#btn-float-stop'), stop: q('#btn-dr-stop'),
        start: q('#btn-dr-start'), stopDisabled: document.querySelector('#btn-dr-stop').disabled, driving: document.body.classList.contains('driving'),
        toastPE: getComputedStyle(t).pointerEvents, toastZ: getComputedStyle(t).zIndex, fstopZ: getComputedStyle(document.querySelector('#btn-float-stop')).zIndex}"""
HIT = """const x = arguments[0], y = arguments[1]; const e = document.elementFromPoint(x, y);
const all = document.elementsFromPoint(x, y).slice(0, 3).map((n) => n.id || n.tagName.toLowerCase());
return {hit: e ? (e.id || e.tagName.toLowerCase()) : null, stack: all}"""


def inter(a, b):
    if not a or not b:
        return None
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, btm = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return None if r <= l or btm <= t else [l, t, r - l, btm - t]


def stop_posts():
    return [json.loads(p['body']).get('reason') for p in srv('/fake/posts') if p['path'] == '/api/drive/stop']




LOGGER = """if (!window.__evlog) { window.__evlog = [];
  for (const ty of ['touchstart', 'pointerdown', 'click']) document.addEventListener(ty, (e) => {
    const t = e.target; window.__evlog.push(ty + ':' + (t.id || (t.closest && t.closest('[id]') ? t.closest('[id]').id : t.tagName.toLowerCase()))); }, true); }
window.__evlog.length = 0; return true"""


def one_tap(b, x, y, kind):
    """トーストを出し直して 1.2 秒待ち、(x, y) を押す。トーストが出ていたか・届いたイベント・POST の reason を返す"""
    b.js("toast(arguments[0], true)", TRIAL)
    time.sleep(1.2)
    vis = b.js("return !document.querySelector('#toast').classList.contains('hidden')")
    b.js(LOGGER)
    fake(clear=True)
    b.frame(None)
    r = b.tap(x, y, kind)
    b.frame(0)
    time.sleep(0.8)
    ev = b.js("return window.__evlog.slice()")
    return vis, r, ev, stop_posts()


def main():
    sizes = [(390, 660), (390, 844), (412, 800), (430, 900)]
    if os.environ.get('JF_B1_QUICK'):
        sizes = sizes[:1]
    lines = ['# b1_tap: 走行中・「試し」のトースト(error・6 秒)が出ている間に、浮いている ■ を本物のポインター(touch・mouse)で押す。'
             'スクロールバーは幅を取らない(スマホと同じ。review_ui_r2 の p4 と同じ)',
             '# 押す所: 重なり = トーストと浮いている ■ が重なる所の真ん中 / ■ の下 = 浮いている ■ のトーストに隠れていない所の真ん中 / 外 = トーストの左の端から 10 px(■ の外)。各 2 回']
    b = B()
    tot = {}
    cases = 0
    card = {'full': 0, 'under_toast': 0, 'under_fstop': 0}
    gone = 0
    try:
        for (w, h) in sizes:
            for pos in ('上', '真ん中', '下寄り'):
                fake(running=False, clear=True)
                b.frame(None)
                b.go(SRV + '/phone?w=%d&h=%d' % (w, h))
                time.sleep(2.5)
                b.frame(0)
                b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
                time.sleep(2.0)
                frac = {'上': 0.2, '真ん中': 0.5, '下寄り': 0.72}[pos]
                b.js("const e = document.querySelector('#btn-dr-start'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight * arguments[0])", frac)
                time.sleep(0.3)
                fake(running=True)
                time.sleep(1.6)                               # 1 秒の輪で走行中を読む(本物の ▶ は pollDrive を直ぐ呼ぶ)
                b.js("toast(arguments[0], true)", TRIAL)       # 本物の ▶ の成功のあとと同じ呼び方(app.js 1542)
                t_show = time.time()
                time.sleep(0.3)
                m = b.js(MEAS)
                # 6 秒で消えるか(この 1 回だけ。出してから 5.5 秒・6.5 秒)
                time.sleep(max(0, t_show + 5.5 - time.time()))
                v55 = b.js("return !document.querySelector('#toast').classList.contains('hidden')")
                time.sleep(max(0, t_show + 6.5 - time.time()))
                v65 = b.js("return !document.querySelector('#toast').classList.contains('hidden')")
                m65 = b.js(MEAS)
                g = v55 and not v65 and inter(m65['toast'], m65['fstop']) is None
                gone += g
                ov = inter(m['toast'], m['fstop'])
                f = m['fstop']
                st = m['stop']
                cases += 1
                card['full'] += bool(st and st[1] >= 0 and st[1] + st[3] <= m['vh'])
                card['under_toast'] += bool(inter(m['toast'], st))
                card['under_fstop'] += bool(inter(m['fstop'], st))
                lines.append('%dx%d ▶%s | 画面 %dx%d | トースト %s | 浮いている ■ %s | 重なり %s(■ の高さ %d のうち縦 %s) | カードの ■ %s(画面の中に全部 %s・トーストと重なる %s・浮いている ■ と重なる %s・disabled %s) | トースト 5.5 秒 %s・6.5 秒 %s(6.5 秒の重なり %s) | pointer-events %s・z %s/%s' % (
                    w, h, pos, m['vw'], m['vh'], m['toast'], f, ov, f[3] if f else 0, ov[3] if ov else 0, st,
                    bool(st and st[1] >= 0 and st[1] + st[3] <= m['vh']), bool(inter(m['toast'], st)), bool(inter(m['fstop'], st)), m['stopDisabled'],
                    '出ている' if v55 else '無い', '出ている' if v65 else '無い', inter(m65['toast'], m65['fstop']), m['toastPE'], m['toastZ'], m['fstopZ']))
                if not ov:
                    continue
                spots = {'重なり': (ov[0] + ov[2] / 2, ov[1] + ov[3] / 2),
                         '■ の下': (f[0] + f[2] / 2, (ov[1] + ov[3] + f[1] + f[3]) / 2),
                         '外': (m['toast'][0] + 10, m['toast'][1] + m['toast'][3] / 2)}
                for kind in ('touch', 'mouse'):
                    for spot, (x, y) in spots.items():
                        hit = b.js(HIT, x, y)
                        for k in range(2):
                            vis, r, ev, reasons = one_tap(b, x, y, kind)
                            key = (kind, spot)
                            s = tot.setdefault(key, {'n': 0, 'stop': 0, 'vis': 0, 'err': 0})
                            s['n'] += 1
                            s['stop'] += reasons == ['停止ボタン']
                            s['vis'] += bool(vis)
                            s['err'] += r != 'ok'
                            lines.append('    %s %s (%d,%d) elementFromPoint %s %s | トースト %s | %s | イベント %s | POST %s' % (
                                kind, spot, x, y, hit['hit'], hit['stack'], '出ている' if vis else '無い', r, ev, reasons or 'なし'))
                if w == 390 and h == 660 and pos == '真ん中':
                    b.js("toast(arguments[0], true)", TRIAL)
                    time.sleep(0.3)
                    b.frame(None)
                    b.shot(str(OUT / 'b1_390x660_mid.png'))
                    b.frame(0)
    finally:
        try:
            fake(running=False)
        except Exception:
            pass
        b.close()
    lines.append('')
    lines.append('# まとめ: %d 通り(画面 %d × ▶ の位置 3)' % (cases, len(sizes)))
    for (kind, spot), s in tot.items():
        lines.append('  %s %s: %d 回押して止まった(POST /api/drive/stop reason 停止ボタン) %d | 押したときトーストが出ていた %d | 押せなかった(WebDriver の誤り) %d' % (
            kind, spot, s['n'], s['stop'], s['vis'], s['err']))
    lines.append('  カードの ■: 画面の中に全部 %d / %d・トーストと重なる %d・浮いている ■ と重なる %d' % (card['full'], cases, card['under_toast'], card['under_fstop']))
    lines.append('  トーストが 5.5 秒で出ていて 6.5 秒で消え、浮いている ■ との重なり 0: %d / %d' % (gone, cases))
    (OUT / 'b1_tap.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
