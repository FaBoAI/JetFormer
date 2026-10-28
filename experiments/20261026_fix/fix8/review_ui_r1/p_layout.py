"""スマホの画面の幅 420 px(iframe の幅。media query は 980 px 以下の 1 列)で、11/3 の朝の流れの走行タブを headless の Firefox で開き、
▶・状態のバッジ・走行タブの文(起き直しの知らせ・つながっていない)・トーストの位置を測る。作り物のサーバー(fake_server.py。POST はしない。
見張り・残す・起き直し・落ちるは作り物の status を変えて真似る)。本物の jetformer-web・車には触れない。出力 out/layout_420.txt と png"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json, time, urllib.parse, urllib.request
from wd import Browser

W = int(sys.argv[1]) if len(sys.argv) > 1 else 420
OUT = pathlib.Path(__file__).resolve().parent / 'out'
SET = lambda q: urllib.request.urlopen('http://127.0.0.1:18771/fake/set?j=' + urllib.parse.quote(json.dumps(q)), timeout=10).read()
IDS = ['#btn-dr-baseline', '#dr-bench-pick', '#dr-model', '#dr-rec-back-guard', '#dr-rec-hold-keep', '#btn-dr-start', '#dr-client-ver',
       '#dr-phase', '#dr-message', '#toast']
MEASURE = """
const r = {}; for (const s of arguments[0]) { const e = document.querySelector(s); if (!e) { r[s] = null; continue; }
  const b = e.getBoundingClientRect(); r[s] = {top: Math.round(b.top + scrollY), bottom: Math.round(b.bottom + scrollY), vtop: Math.round(b.top), vbottom: Math.round(b.bottom), left: Math.round(b.left), right: Math.round(b.right), h: Math.round(b.height), hidden: e.classList.contains('hidden') || getComputedStyle(e).display === 'none', disabled: !!e.disabled, text: (e.textContent || '').slice(0, 400)}; }
return {vw: innerWidth, vh: innerHeight, cw: document.documentElement.clientWidth, sw: document.documentElement.scrollWidth, sh: document.documentElement.scrollHeight, scrollY, r};"""
lines = []
def log(*a):
    s = ' '.join(str(x) for x in a); lines.append(s); print(s)

def measure(b, tag):
    m = b.js(MEASURE, IDS)
    log('## %s: 幅 %s(scrollWidth %s・横にはみ出し %s px)・高さ %s' % (tag, m['cw'], m['sw'], m['sw'] - m['cw'], m['sh']))
    for s in IDS:
        x = m['r'][s]
        if x:
            log('  %-20s top %5s bottom %5s 左右 %s〜%s%s%s 文: %s' % (s, x['top'], x['bottom'], x['left'], x['right'], ' 隠れ' if x['hidden'] else '',
                ' 押せない' if x['disabled'] else '', x['text'].replace('\n', ' ')[:200]))
    return m

b = Browser(900, 1000)
try:
    SET({'params': {'rec_back_guard': 0, 'rec_hold_keep': 0}, 'boot_at': 1793116404.711, 'down': False})
    b.go('http://127.0.0.1:18771/phone?w=%d' % W)
    time.sleep(3)
    b.frame(0)
    b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
    time.sleep(3)
    measure(b, '開いた直後')
    b.js("document.querySelector('#btn-dr-baseline').click()")
    time.sleep(1.5)
    SET({'params': {'rec_back_guard': 1, 'rec_hold_keep': 1}})     # 見張り → 残す(スイッチの POST の代わりに作り物の値)
    time.sleep(2)
    v = b.js("return [...document.querySelector('#dr-bench-pick').options].map((o) => o.value).find((x) => /:0$/.test(x))")
    b.js("const s = document.querySelector('#dr-bench-pick'); s.value = arguments[0]; s.dispatchEvent(new Event('change'))", v)
    time.sleep(2)
    m = measure(b, '基準の設定 → 見張り・残す → ブロック 1')
    # ▶ を画面の真ん中に(人が ▶ を押す前に見る画面)
    y = m['r']['#btn-dr-start']['top'] - m['vh'] // 2
    b.js('scrollTo(0, arguments[0])', y)
    time.sleep(0.5)
    b.top(); b.shot(str(OUT / ('layout_%d_start.png' % W))); b.frame(0)
    # 起き直し(作り物の boot_at を変え、見張り・残すを既定に)
    SET({'params': {'rec_back_guard': 0, 'rec_hold_keep': 0}, 'boot_at': 1793116404.711 + 3600})
    time.sleep(2)
    b.js('scrollTo(0, arguments[0])', y)
    m = measure(b, '起き直しを知った直後(▶ を画面の真ん中に)')
    vis = lambda x: x['vbottom'] > 0 and x['vtop'] < m['vh']
    for s in ('#btn-dr-start', '#dr-phase', '#dr-message', '#toast', '#dr-rec-back-guard'):
        log('  見える? %-20s %s(画面の中の top %s)' % (s, vis(m['r'][s]), m['r'][s]['vtop']))
    b.top(); b.shot(str(OUT / ('layout_%d_reboot.png' % W))); b.frame(0)
    time.sleep(6.5)
    m = measure(b, '6.5 秒後(トーストが消えたあと)')
    for s in ('#btn-dr-start', '#dr-phase', '#dr-message', '#toast'):
        log('  見える? %-20s %s(隠れ %s)' % (s, vis(m['r'][s]), m['r'][s]['hidden']))
    b.top(); b.shot(str(OUT / ('layout_%d_reboot_6s.png' % W))); b.frame(0)
    # Wi-Fi が切れた(作り物のサーバーが 503)
    SET({'down': True})
    time.sleep(3)
    b.js('scrollTo(0, arguments[0])', y)
    m = measure(b, 'つながらない(3 秒後。▶ を画面の真ん中に)')
    for s in ('#btn-dr-start', '#dr-phase', '#dr-message', '#toast'):
        log('  見える? %-20s %s(隠れ %s・押せない %s)' % (s, vis(m['r'][s]), m['r'][s]['hidden'], m['r'][s]['disabled']))
    log('  ▶ の title: %s' % b.js("return document.querySelector('#btn-dr-start').title"))
    b.top(); b.shot(str(OUT / ('layout_%d_down.png' % W))); b.frame(0)
    SET({'down': False})
finally:
    b.close()
(OUT / ('layout_%d.txt' % W)).write_text('\n'.join(lines) + '\n', encoding='utf-8')
