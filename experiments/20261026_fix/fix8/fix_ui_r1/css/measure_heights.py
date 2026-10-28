"""fix8 ui r1: スマホの見える高さ(664・760・860 px)と幅(360・390・420)で、▶ を画面の真ん中にしたときの ▶ の断りの出し直し(本物の
_boot_refusal の文)のトーストと「停止中」のバッジ・走行タブの文の重なりを、直す前と今の style.css で測る(残りの確かめ)。
作り物のサーバー(fake_server.py。POST は 405)・headless の Firefox。出力 out/heights.txt"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure_toast as M  # noqa: E402
from wd import Browser  # noqa: E402

lines = []
for which, static in (('直す前', HERE / 'static_before'), ('今', Path('/home/jetson/jetformer/jetformer/web/static'))):
    b = Browser(1100, 1000, static=static)
    try:
        for h in (664, 760, 860):
            for w in (360, 390, 420):
                b.top()
                b.go('http://127.0.0.1:18793/phone?w=%d&h=%d' % (w, h))
                time.sleep(2.5)
                b.frame(0)
                b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
                time.sleep(2.5)
                b.js("const e = document.querySelector('#btn-dr-start'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)")
                m = b.js(M.MEAS, M.TEXTS['断りの出し直し'], True)
                t = m['toast']
                over = [k for k, key in (('バッジ「停止中」', 'phase'), ('走行タブの文', 'msg'), ('▶', 'start')) if M.overlaps(t, m[key])]
                lines.append('%s | 高さ %d・幅 %d: 箱 幅 %d 高さ %d(上 %d)・バッジ 上 %d・重なる %s' % (
                    which, h, w, t[2], t[3], t[1], m['phase'][1], '・'.join(over) or 'なし'))
                print(lines[-1], flush=True)
    finally:
        b.close()
(HERE / 'out' / 'heights.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
