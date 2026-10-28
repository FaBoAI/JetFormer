"""反証 8-2 回目・係 ui: 8-1 の .toast の直し(width: max-content・max-width: min(90%, 960px))を、スマホの幅・高さ(390〜430 ×
660〜900)の headless の Firefox で測る。止まっているとき(▶ の断り・起き直しの知らせ・一覧を読めていない・読み込んでいます)と、
走っているとき(body.driving: トーストは下 96 px・浮いている ■ が出る。11/3 は毎回の走行開始で「ブロックの計画と違うので「試し」…」)。
重なりを見る相手: ▶・カードの ■・浮いている ■(#btn-float-stop)・状態のバッジ・CH3 の欄・走行タブの文。画面の外にはみ出すか。
比べ: 8-1 の前の style.css(tests/fixtures/fix8_ui_r1_before/style.css。作り物のサーバーが ?css=before で差し替える = 本物は変えない)。
作り物のサーバー fake_server.py(127.0.0.1:18797・POST は 405)に本物の index.html・app.js・style.css。読むだけ。
出力 out/p4_toast_phone.txt(表)・out/p4_*.png(いくつか)"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wd import Browser, SRV_PORT  # noqa: E402
from run import BR  # noqa: E402

OUT = HERE / 'out'
BASE = 'http://127.0.0.1:%s' % SRV_PORT
B0 = 1793116404.711
IDLE_TEXTS = {
    '断りの出し直し(207字級)': '走行は始まっていません(停止中)。' + BR({'boot_at': B0}, B0 + 3600),
    '起き直しの知らせ(入れ直し済み)': '管理画面のサーバーが 01:53 に起動し直しました。起動のあとに入れ直した設定があります(今は リカバリーの後ろの見張り ON・中立止めのあともリカバリーを残す ON・天井の左旋回中 4.2 m/s・発進の前進の強さ 200 µs)。朝の設定はスイッチと走行開始の確認の文で読む',
    '読み込んでいます(戻った直後)': 'サーバーの設定を読み込んでいます。少し待ってから押してください',
}
RUN_TEXTS = {
    '計画と違うので試し(毎回の開始)': 'ブロックの計画と違うので「試し」(cand=EXP)として記録します: throttle_gain 0.9(基準 1.0)、speed_ceiling 1(基準 0)、ceil_left_mps 4.2(基準 4.1)',
    '走行中は ON にできません': '走行中はリカバリーの後ろの見張りを ON にできません。止めてから ON にしてください(1 走行 = 1 設定)',
}
MEAS = """const q = (s) => { const e = document.querySelector(s); if (!e || e.classList.contains('hidden') || getComputedStyle(e).display === 'none') return null;
  const r = e.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; };
return {vw: document.documentElement.clientWidth, vh: innerHeight, toast: q('#toast'), start: q('#btn-dr-start'), stop: q('#btn-dr-stop'),
        fstop: q('#btn-float-stop'), phase: q('#dr-phase'), who: q('#dr-who'), msg: q('#dr-message'), driving: document.body.classList.contains('driving')}"""
NAMES = (('▶', 'start'), ('カードの ■', 'stop'), ('浮いている ■', 'fstop'), ('状態のバッジ', 'phase'), ('CH3 の欄', 'who'), ('走行タブの文', 'msg'))


def over(a, b):
    if not a or not b or b[2] == 0 or b[3] == 0:
        return 0
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def fake(**kw):
    import urllib.parse
    import urllib.request
    urllib.request.urlopen(BASE + '/fake/set?j=' + urllib.parse.quote(json.dumps(kw)), timeout=10).read()


def main():
    import os
    widths = [390, 400, 412, 420, 430] if not os.environ.get('JF_P4_QUICK') else [390]
    heights = [660, 700, 760, 800, 844, 900] if not os.environ.get('JF_P4_QUICK') else [660]
    rows = []
    b = Browser(1100, 1000)
    try:
        for css in ('now', 'before'):
            for w in widths:
                for h in heights:
                    fake(running=False)
                    b.top()
                    b.go(BASE + '/phone?w=%d&h=%d%s' % (w, h, '&css=before' if css == 'before' else ''))
                    time.sleep(2.5)
                    b.frame(0)
                    b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
                    time.sleep(2.0)
                    for pos in ('▶ を真ん中',):
                        b.js("const e = document.querySelector('#btn-dr-start'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)")
                        for name, text in IDLE_TEXTS.items():
                            b.js("toast(arguments[0], true)", text)
                            time.sleep(0.25)
                            m = b.js(MEAS)
                            rows.append((css, w, h, '止まっている', pos, name, m))
                    fake(running=True)
                    time.sleep(1.6)
                    for pos, script in (('▶・■ を真ん中', "const e = document.querySelector('#btn-dr-stop'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)"),
                                        ('状態のバッジを真ん中', "const e = document.querySelector('#dr-phase'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)"),
                                        ('いちばん上', "scrollTo(0, 0)")):
                        b.js(script)
                        for name, text in RUN_TEXTS.items():
                            b.js("toast(arguments[0], true)", text)
                            time.sleep(0.25)
                            m = b.js(MEAS)
                            rows.append((css, w, h, '走行中', pos, name, m))
                            if css == 'now' and w == 390 and h == 660 and name.startswith('計画') and pos.startswith('▶'):
                                b.top(); b.shot(str(OUT / 'p4_run_390x660_now.png')); b.frame(0)
                            if css == 'before' and w == 390 and h == 660 and name.startswith('計画') and pos.startswith('▶'):
                                b.top(); b.shot(str(OUT / 'p4_run_390x660_before.png')); b.frame(0)
                    fake(running=False)
    finally:
        try:
            fake(running=False)
        except Exception:
            pass
        b.close()
    lines = ['# .toast をスマホの幅・高さで(headless の Firefox・iframe = 画面。スクロールバーは幅を取らない)。now = 今の style.css(dbc8b79a)、before = 8-1 の前の写し(5842642f)',
             '# 列: css 幅 高さ 状態 位置 文 | トースト[左,上,幅,高さ] | 画面の外 | 重なる相手(重なりの面積 px²) | 浮いている ■ の箱']
    summary = {}
    for css, w, h, state, pos, name, m in rows:
        t = m['toast']
        out_of = []
        if t and (t[0] < 0 or t[0] + t[2] > m['vw']):
            out_of.append('横')
        if t and t[1] < 0:
            out_of.append('上')
        ov = ['%s %d' % (n, over(t, m[k])) for n, k in NAMES if over(t, m[k]) > 0]
        lines.append('%-6s %d %d %s %s %s | %s | %s | %s | %s' % (css, w, h, state, pos, name, t, '・'.join(out_of) or '-', '・'.join(ov) or 'なし', m['fstop']))
        key = (css, state, name)
        s = summary.setdefault(key, {'n': 0, 'fstop': 0, 'stop': 0, 'start': 0, 'phase': 0, 'out': 0, 'hmax': 0, 'wmin': 9999})
        s['n'] += 1
        s['fstop'] += over(t, m['fstop']) > 0
        s['stop'] += over(t, m['stop']) > 0
        s['start'] += over(t, m['start']) > 0
        s['phase'] += over(t, m['phase']) > 0
        s['out'] += bool(out_of)
        s['hmax'] = max(s['hmax'], t[3] if t else 0)
        s['wmin'] = min(s['wmin'], t[2] if t else 0)
    lines.append('')
    lines.append('# まとめ(css・状態・文ごと: 測った数・浮いている ■ に重なる・カードの ■ に重なる・▶ に重なる・状態のバッジに重なる・画面の外・高さの最大・幅の最小)')
    for k, s in summary.items():
        lines.append('%-6s %s %s: %d 回 | 浮いている ■ %d | カードの ■ %d | ▶ %d | バッジ %d | 画面の外 %d | 高さ最大 %d | 幅最小 %d' % (
            k[0], k[1], k[2], s['n'], s['fstop'], s['stop'], s['start'], s['phase'], s['out'], s['hmax'], s['wmin']))
    (OUT / 'p4_toast_phone.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[-(len(summary) + 1):]))


if __name__ == '__main__':
    main()
