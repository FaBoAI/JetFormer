"""fix8 ui r1(反証 8-1 回目 toast-half-width-on-phone): 走行タブのトーストの幅・高さと、状態のバッジ「停止中」・走行タブの文・▶ との
重なりを、headless の Firefox(iframe の幅 = 画面の幅)で測る。作り物のサーバー(fake_server.py。反証の係の写しに「返す static の場所」
を足しただけ。POST は 405 = 何も変えない。127.0.0.1:18793)。本物の jetformer-web・車には触れない。

    python measure_toast.py before   # 直す前の style.css(static_before/ = tests/fixtures/fix8_ui_r1_before の写し)に、直しの案を iframe の中だけで足して比べる
    python measure_toast.py after    # 今の style.css(jetformer/web/static。直したあと)

出力 out/toast_<before|after>.txt(と ▶ の断りの出し直しの png)。文は app.js の本物の toast() に渡す。▶ の断りの出し直しは本物の
server._boot_refusal(ast で取り出す。server.py を import しない = サーバーを作らない)。
"""
import ast
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wd import Browser  # noqa: E402

os.environ['TZ'] = 'Asia/Tokyo'
time.tzset()
SRV = Path('/home/jetson/jetformer/jetformer/web/server.py')
_fn = next(n for n in ast.parse(SRV.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == '_boot_refusal')
_ns = {'time': time}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), 'server.py', 'exec'), _ns)
B0 = 1793116404.711
TEXTS = {
    '断りの出し直し': '走行は始まっていません(停止中)。' + _ns['_boot_refusal']({'boot_at': B0}, B0 + 3600),
    '起き直しの知らせ': '管理画面のサーバーが 01:53 に起動し直しました。起動のあとに入れ直した設定があります(今は リカバリーの後ろの見張り ON・'
                 '中立止めのあともリカバリーを残す ON・天井の左旋回中 4.2 m/s・発進の前進の強さ 200 µs)。朝の設定はスイッチと走行開始の確認の文で読む',
    '一覧を読めていない': 'モデルの一覧を読めていません(一覧の問い合わせが通らないか遅れています。TensorRT エンジンが無いという意味ではありません)。'
                  '少し待っても「モデル」の欄が空なら、ほかのタブ → 走行タブで開き直すか、画面を読み直してください',
    'モデル名の長い文': 'モデル clean_driving_20261011-nocmd-s3-20261011-024626 が走行の一覧にありません(TensorRT エンジンが必要)',
    '短い文': '停止しました',
}
CANDS = {
    'もと': '',
    '案A left/right 5%': '.toast { left: 5%; right: 5%; transform: none; max-width: none; }',
    '案B max-content・90%': '.toast { width: max-content; max-width: 90%; }',
    '案C max-content・min(90%, 960px)': '.toast { width: max-content; max-width: 90%; max-width: min(90%, 960px); }',
}
WIDTHS = (360, 390, 420, 768, 980, 1280, 1920)
MEAS = """const [txt, err] = arguments;
toast(txt, err);
const q = (s) => { const r = document.querySelector(s).getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; };
const t = document.querySelector('#toast');
return {vw: document.documentElement.clientWidth, vh: innerHeight, toast: q('#toast'), start: q('#btn-dr-start'), phase: q('#dr-phase'), msg: q('#dr-message'),
        over: t.scrollWidth > t.clientWidth, cs: [getComputedStyle(t).width, getComputedStyle(t).maxWidth, getComputedStyle(t).left]}"""


def overlaps(t, e):
    return t[1] < e[1] + e[3] and e[1] < t[1] + t[3] and t[0] < e[0] + e[2] and e[0] < t[0] + t[2]


def main():
    which = sys.argv[1]
    static = HERE / 'static_before' if which == 'before' else Path('/home/jetson/jetformer/jetformer/web/static')
    cands = CANDS if which == 'before' else {'今の style.css': ''}
    lines = ['static: %s' % static]
    b = Browser(2000, 1000, static=static)
    try:
        for cname, css in cands.items():
            for w in WIDTHS:
                b.top()
                b.go('http://127.0.0.1:18793/phone?w=%d' % w)
                time.sleep(2.5)
                b.frame(0)
                b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
                if css:
                    b.js("const s = document.createElement('style'); s.textContent = arguments[0]; document.head.appendChild(s)", css)
                time.sleep(2.5)
                b.js("const e = document.querySelector('#btn-dr-start'); const r = e.getBoundingClientRect(); scrollTo(0, r.top + scrollY - innerHeight / 2)")
                for name, text in TEXTS.items():
                    m = b.js(MEAS, text, name != '短い文')
                    t = m['toast']
                    over = [k for k, key in (('バッジ「停止中」', 'phase'), ('走行タブの文', 'msg'), ('▶', 'start')) if overlaps(t, m[key])]
                    inside = t[0] >= 0 and t[0] + t[2] <= m['vw']
                    lines.append('%s | 幅 %4d | %s(%d 字): 箱 x %d 幅 %d 高さ %d(見える高さの %d%%)・画面の中 %s・文のはみ出し %s・重なる %s' % (
                        cname, w, name, len(text), t[0], t[2], t[3], round(100 * t[3] / m['vh']), 'はい' if inside else 'いいえ',
                        'あり' if m['over'] else 'なし', '・'.join(over) or 'なし'))
                    if name == '断りの出し直し' and w in (420, 1920):
                        b.top()
                        b.shot(str(HERE / 'out' / ('toast_%s_%s_%d.png' % (which, list(cands).index(cname), w))))
                        b.frame(0)
    finally:
        b.close()
    txt = '\n'.join(lines) + '\n'
    (HERE / 'out' / ('toast_%s.txt' % which)).write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
