"""反証 8-2 回目・係 ui: 浮いている ■(.float-stop)も .toast と同じ left: 50% の形で画面の半分の幅・3 行になり、走行中のトースト
(body.driving .toast の bottom 96 px)が上の行を隠す(p4)。直しの案を iframe の中だけで足して(本物の style.css は変えない)測る。
headless の Firefox・作り物のサーバー(fake_server.py)。出力 out/p6_float_stop_fix.txt・out/p6_*.png"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wd import Browser, SRV_PORT  # noqa: E402
from p4_toast_phone import MEAS, over, fake, RUN_TEXTS  # noqa: E402

BASE = 'http://127.0.0.1:%s' % SRV_PORT
FIXES = {
    '今のまま': '',
    '案 1: .float-stop も文の幅(90% まで)': '.float-stop { width: max-content; max-width: 90%; }',
    '案 2: 案 1 + 走行中のトーストを浮いている ■ の上に(bottom 104px)': '.float-stop { width: max-content; max-width: 90%; } body.driving .toast { bottom: 104px; }',
}


def main():
    lines = []
    b = Browser(1100, 1000)
    try:
        for w, h in ((390, 660), (412, 800), (430, 900)):
            for fname, css in FIXES.items():
                fake(running=False)
                b.top()
                b.go(BASE + '/phone?w=%d&h=%d' % (w, h))
                time.sleep(2.5)
                b.frame(0)
                b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
                if css:
                    b.js("const s = document.createElement('style'); s.textContent = arguments[0]; document.head.appendChild(s)", css)
                time.sleep(1.5)
                fake(running=True)
                time.sleep(1.6)
                b.js("const e = document.querySelector('#btn-dr-stop'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)")
                for name, text in RUN_TEXTS.items():
                    b.js("toast(arguments[0], true)", text)
                    time.sleep(0.25)
                    m = b.js(MEAS)
                    t, fs = m['toast'], m['fstop']
                    dv = max(0, min(t[1] + t[3], fs[1] + fs[3]) - max(t[1], fs[1])) if t and fs else 0
                    out_top = t[1] < 0 if t else False
                    lines.append('%d×%d %s %s: トースト %s・浮いている ■ %s・重なり %d px²(縦 %d px)%s' % (
                        w, h, fname, name, t, fs, over(t, fs), dv, '・トーストが画面の上にはみ出す' if out_top else ''))
                    if w == 390 and name.startswith('計画'):
                        b.top(); b.shot(str(HERE / 'out' / ('p6_390x660_%d.png' % list(FIXES).index(fname)))); b.frame(0)
                fake(running=False)
    finally:
        try:
            fake(running=False)
        except Exception:
            pass
        b.close()
    (HERE / 'out' / 'p6_float_stop_fix.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
