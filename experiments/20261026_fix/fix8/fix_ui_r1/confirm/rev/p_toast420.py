"""スマホの幅(420 px)と 980 px で、走行タブの大事なトースト(起き直しの知らせ・▶ の断りの出し直し・一覧を読めていない)の幅・高さを
headless の Firefox で測る(この係。作り物のサーバー fake_server.py。本物の style.css をそのまま。直しの案は iframe の中だけで
style を足して比べる = 本物のファイルは変えない)。出力 out/toast_width.txt・toast_420.png・toast_420_fix.png"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json, time
from wd import Browser
from run import BR
OUT = pathlib.Path(__file__).resolve().parent / 'out'
B0 = 1793116404.711
TEXTS = {
    '▶ の断りの出し直し(本物の _boot_refusal の文 + 頭)': '走行は始まっていません(停止中)。' + BR({'boot_at': B0}, B0 + 3600),
    '起き直しの知らせ(入れ直し済み)': '管理画面のサーバーが 01:53 に起動し直しました。起動のあとに入れ直した設定があります(今は リカバリーの後ろの見張り ON・中立止めのあともリカバリーを残す ON・天井の左旋回中 4.2 m/s・発進の前進の強さ 200 µs)。朝の設定はスイッチと走行開始の確認の文で読む',
    '一覧を読めていない(DR_MODELS_MISS)': 'モデルの一覧を読めていません(一覧の問い合わせが通らないか遅れています。TensorRT エンジンが無いという意味ではありません)。少し待っても「モデル」の欄が空なら、ほかのタブ → 走行タブで開き直すか、画面を読み直してください',
}
FIX = '.toast { left: 5%; right: 5%; transform: none; max-width: none; }'
MEAS = """const q = (s) => { const r = document.querySelector(s).getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; };
return {vh: innerHeight, toast: q('#toast'), start: q('#btn-dr-start'), phase: q('#dr-phase'), msg: q('#dr-message'), maxw: getComputedStyle(document.querySelector('#toast')).maxWidth}"""
lines = []
b = Browser(1100, 1000)
try:
    for w, fix in ((420, False), (980, False), (420, True)):
        b.top()
        b.go('http://127.0.0.1:18791/phone?w=%d' % w)
        time.sleep(3)
        b.frame(0)
        b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
        if fix:
            b.js("const s = document.createElement('style'); s.textContent = arguments[0]; document.head.appendChild(s)", FIX)
        time.sleep(3)
        b.js("const e = document.querySelector('#btn-dr-start'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)")
        for name, text in TEXTS.items():
            b.js("toast(arguments[0], true)", text)
            time.sleep(0.3)
            m = b.js(MEAS)
            t = m['toast']
            over = [k for k, key in (('バッジ「停止中」', 'phase'), ('走行タブの文', 'msg'), ('▶', 'start'))
                    if t[1] < m[key][1] + m[key][3] and m[key][1] < t[1] + t[3] and t[0] < m[key][0] + m[key][2] and m[key][0] < t[0] + t[2]]
            lines.append('幅 %d%s・%s(%d 字): トースト 幅 %d・高さ %d(見える高さ %d の %d%%)・重なる: %s' % (
                w, '(直しの案)' if fix else '', name, len(text), t[2], t[3], m['vh'], round(100 * t[3] / m['vh']), '・'.join(over) or 'なし'))
            if name.startswith('▶'):
                b.top(); b.shot(str(OUT / ('toast_%d%s.png' % (w, '_fix' if fix else '')))); b.frame(0)
finally:
    b.close()
lines.insert(0, 'style.css の .toast: position fixed・left 50%・transform translateX(-50%)・max-width 90%(left 50% のため使える幅は画面の半分)。直しの案: ' + FIX)
(OUT / 'toast_width.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
