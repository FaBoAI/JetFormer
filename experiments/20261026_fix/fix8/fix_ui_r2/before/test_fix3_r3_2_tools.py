"""直す 3 回目・2 組目(2026-10-27 fix3。反証 3 回目の指摘 r3-r3-rec-2・rec-3・rec-4)の試験。夜の道具だけ
(⑤ experiments/20261026_fix/diag_launch/collect.py・③ tools/standing_3lap.py・④ tools/quick_eval_by_params.py・tools/car_check.py)。
走行の動き(PCA9685 に書く値)・既定値・画面・bench_candidates.json は変えていない(test_untouched_files が sha256 で見る)。

  r3-r3-rec-2: 手順書 7 章の「空転の比 > 1.3 → 残りの ON を launch_us 200」を回 3 の途中で読む道具が無かった(1 本ごとの手順は
    car_check だけで、car_check に空転の比が無い)→ car_check に、発進の前進 ON の本だけ「発進の空転の比」の行(③ と同じ値・判定の
    番号には入れない)。
  r3-r3-rec-3: 手順書 7 章の「その本は数えない(memo に「試し」)」を ⑤ ③ ④ が読まず、試しの本が回 3 の比べに入って答えが変わった
    → memo に人が「試し」と書いた本を外す(走行開始でサーバーが足す「計画と違うので試し(元 cand=B)…」は外さない。11/3 はゲインが
    1.00 でない本の全部に付くので、外すと比べが消える)。
  r3-r3-rec-4: いちばんひどい空転(IMU の積分 ≤ 0.2 m/s)は比が None になり、⑤ の空転の条件と ③ の印から黙って抜けた
    → 車輪が 0.5 m/s より増えていれば > 1.3 とみなす。ほかの理由で比が出ない ON は「確かめられない = 満たさない」。

どの試験も「直す前の写し(tests/fixtures/fix3_r3_2_before/。sha256 を照らす)で指摘が起きる」と「その場面のほかは直す前と出力
(JSON・標準出力・文・終わりの番号)が同じ」を見る。作り物の記録だけ(tmp)。本物の data/raw・labels・装置・GPU は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix3_r3_2_tools.py
"""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

JF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
from test_fix3_r1_2 import Day, ON, PRM, load_mod  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402
import car_check as cc  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402
from jetformer.web import bench as wb  # noqa: E402

FIX = JF / 'tests' / 'fixtures' / 'fix3_r3_2_before'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
RUNBOOK = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
# 直す前(10/27 05:02)の sha256。この組は変えていない(走行の動き・既定値・画面・計画)
UNTOUCHED = {
    'jetformer/drive.py': 'b8c3ddba39681d81f14249981c770a9349e6dba3bcf5b4a8d88ee88afbc85db6',
    'jetformer/assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6',
    'jetformer/web/server.py': '6330f0853bbcc505479485c24f583989221ed7276c92160988f63dd484160149',
    'jetformer/web/bench.py': 'c2fdcff9a20128c9bf677fedb63f7a764065bb9eb1daa96820ac72b03fca810b',
    'jetformer/web/static/app.js': '2294bfd96ea812cc392566db729ef82d7fc60dac2c1bb4a9db86298c65babee0',
    'jetformer/web/static/index.html': 'ef81bb6017d63522bf4b870f0d156ce38673f5af3fd4249a31bcfbe019d191ec',
    'data/bench_candidates.json': '353ec3063be418892c5d5b07ba05a14b238ca5dbe47604c2e7c0cfdc19ec5961',
    'data/drive_settings.json': '590d81657f11099a7ba4f1b82d47ffacc0c3766daac52a48ba9f964fc317c419',
}
# この組のあと、10/27 fix4(10/26 の直しの小さな残り)が 2 本を変えた: drive.py は説明の文(モジュールの docstring)だけ
# (コードは AST で同じ)・web/bench.py は SWITCH_KEYS に試験の選択肢 3 つ。この組のときの版(上の sha256)は
# tests/fixtures/fix4_small_before/ の写し。中身の違いは tests/test_fix4_small.py が見る。fix4 のあとの版の sha256 はここ
FIX4_BEFORE = JF / 'tests' / 'fixtures' / 'fix4_small_before'
LATER_FIX4 = {
    'jetformer/drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
    'jetformer/web/bench.py': 'a633bec341a3220f3e0d6941fd5455cf7c1fa7e690ce66e5bf4787d3f55366c8',
}
# そのあと 10/27 fix5 safety(反証 4 回目の s1〜s3。rec_hold_keep = 1・rec_back_guard = 1 の日だけ効く)が assist.py・drive.py の
# コードを変えた。直す前の写しは tests/fixtures/fix5_safety_before/(drive.py = 上の LATER_FIX4 の版・assist.py = UNTOUCHED の版)。
# 中身の違い(スイッチが OFF ならビット単位で同じ)は tests/test_rec_fix5_safety.py が見る。fix5 のあとの版の sha256 はここ
FIX5_BEFORE = JF / 'tests' / 'fixtures' / 'fix5_safety_before'
LATER_FIX5 = {
    'jetformer/drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c',
    'jetformer/assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276',
}
# 同じ 10/27 fix5 の画面の係(反証 4 回目 t1・t2・t3)が web/server.py・web/static/app.js を変えた(index.html は同じ)。直す前の写しは
# tests/fixtures/fix5_ui_before/(上の UNTOUCHED の版)。中身の違いは tests/test_ui_fix5.py が見る。fix5 のあとの版の sha256 はここ
FIX5_UI_BEFORE = JF / 'tests' / 'fixtures' / 'fix5_ui_before'
LATER_FIX5_UI = {
    'jetformer/web/server.py': 'ad7a7d05f7a9167c2992d78d77e39f05f49153cc3b13d8651b166efb4a073b94',
    'jetformer/web/static/app.js': 'fcc18443ce4bcf07bd0d00cdb833437990e26a43f68cc3f70b0190821c05be63',
}
# そのあと 10/27 fix6 の画面の係(反証 6 回目 w1-ui-ui61-2。▶ の断りの文の出し直し。表示だけ)が web/static/app.js を変えた。直す前の写しは
# tests/fixtures/fix6_ui_before/(上の LATER_FIX5_UI の版)。中身の違いは tests/test_ui_fix6.py が見る。fix6 のあとの版の sha256 はここ
FIX6_UI_BEFORE = JF / 'tests' / 'fixtures' / 'fix6_ui_before'
LATER_FIX6_UI = {
    'jetformer/web/static/app.js': 'e41c2f3ea7960539a3b8cb500f3286b9027dbdc706f801679cdadf4ab049b120',
}
# そのあと同じ fix6 の画面の係の直す 6-2 回目(反証 w2-ui-w1-ui-r62-1。走行タブの表示の問い合わせの輪が止まらないように。表示と問い合わせ
# だけ)が web/static/app.js を変えた。直す前の写しは tests/fixtures/fix6_ui_r2_before/(上の LATER_FIX6_UI の版)。中身の違いは
# tests/test_ui_fix6_r2.py が見る。6-2 回目のあとの版の sha256 はここ
FIX6_UI_R2_BEFORE = JF / 'tests' / 'fixtures' / 'fix6_ui_r2_before'
LATER_FIX6_UI_R2 = {
    'jetformer/web/static/app.js': '908934c6297b9429355ddae5f94d3ea8d4618ba8936527248c10ba5a8d228852',
}
# そのあと 10/27 fix7 の画面の係の直す 7-1 回目(反証 w1-ui-r7-ui-1。起き直しの知らせの文を今の値から作る。表示だけ)が web/static/app.js を
# 変えた。直す前の写しは tests/fixtures/fix7_ui_before/(上の LATER_FIX6_UI_R2 の版)。中身の違いは tests/test_ui_fix7.py が見る。
# 7-1 回目のあとの版の sha256 はここ
FIX7_UI_BEFORE = JF / 'tests' / 'fixtures' / 'fix7_ui_before'
LATER_FIX7_UI = {
    'jetformer/web/static/app.js': '3cf098d93386f0f24f321093ec991e3114dfe10b249edb52cf4c5c908b03a3a9',
}
# そのあと 10/27 fix7 の画面の係の直す 7-2 回目(反証 w2-ui-w2-ui-r72-ui-1・-2。▶ の断りの文の出し直しと、走行タブを開いたときのモデルの
# 一覧の読み直し。表示と問い合わせだけ)が web/static/app.js を変えた。直す前の写しは tests/fixtures/fix7_ui_r2_before/(上の LATER_FIX7_UI
# の版)。中身の違いは tests/test_ui_fix7_r2.py が見る。7-2 回目のあとの版の sha256 はここ
FIX7_UI_R2_BEFORE = JF / 'tests' / 'fixtures' / 'fix7_ui_r2_before'
LATER_FIX7_UI_R2 = {
    'jetformer/web/static/app.js': '2b211259db71e3bb8549e99a77bd475745241fbdaecd50d7b63669ae00b9593e',
}
# 同じ 10/27 fix6 の安全の係(反証 6-1 回目 w1-safety-r61-safety-1。rec_hold_keep = 1・rec_back_guard = 1 の日だけ効く)が
# assist.py・drive.py を変えた。直す前の写しは tests/fixtures/fix6_safety_before/(上の LATER_FIX5 の版)。中身の違い(スイッチが
# OFF ならビット単位で同じ)は tests/test_rec_fix6_safety.py が見る。fix6 のあとの版の sha256 はここ
FIX6_BEFORE = JF / 'tests' / 'fixtures' / 'fix6_safety_before'
LATER_FIX6 = {
    'jetformer/drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
    'jetformer/assist.py': 'd1d62c66bef856002c14db8dd2fd9c35279a36878b2e4817a279e9fb8bf64165',
}
# そのあと 10/27 fix6 の安全の係の直す 6-2 回目(反証 6-2 回目 w2-safety-r62-safety-1・-2。両方 1 の日だけ効く)が assist.py を
# 変えた(drive.py は同じ)。直す前の写しは tests/fixtures/fix6_safety_r2_before/(上の LATER_FIX6 の版)。中身の違い(スイッチが
# OFF ならビット単位で同じ)は tests/test_rec_fix6_safety_r2.py が見る。6-2 回目のあとの版の sha256 はここ
FIX6_SAFETY_R2_BEFORE = JF / 'tests' / 'fixtures' / 'fix6_safety_r2_before'
LATER_FIX6_SAFETY_R2 = {
    'jetformer/assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4',
}
# そのあと 10/27 fix7 の安全の係の直す 7-1 回目(反証 w1-safety-rv7-safety-r1-1。rec_back_guard = 1 の日だけ効く)が assist.py・drive.py を
# 変えた。直す前の写しは tests/fixtures/fix7_safety_before/(assist.py = 上の LATER_FIX6_SAFETY_R2 の版・drive.py = LATER_FIX6 の版)。
# 中身の違い(見張りが OFF ならビット単位で同じ)は tests/test_rec_fix7_safety.py が見る。7-1 回目のあとの版の sha256 はここ
FIX7_SAFETY_BEFORE = JF / 'tests' / 'fixtures' / 'fix7_safety_before'
LATER_FIX7_SAFETY = {
    'jetformer/assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f',
    'jetformer/drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
}
# そのあと 10/27 fix7 の安全の係の直す 7-2 回目(反証 w2-safety-rv7-safety-r2-1。rec_back_guard = 1 の日だけ効く)が assist.py を変えた
# (drive.py は同じ)。直す前の写しは tests/fixtures/fix7_safety_r2_before/(上の LATER_FIX7_SAFETY の版)。中身の違い(見張りが OFF なら
# ビット単位で同じ)は tests/test_rec_fix7_safety_r2.py が見る。7-2 回目のあとの版の sha256 はここ
FIX7_SAFETY_R2_BEFORE = JF / 'tests' / 'fixtures' / 'fix7_safety_r2_before'
LATER_FIX7_SAFETY_R2 = {
    'jetformer/assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f',
}
# そのあと 10/28 fix8 の安全の係(safety-decided)が drive.py のモジュールの説明の文(docstring)の見張りの段に 1 文を足した
# (コメントだけ。docstring を除いた AST は同じ = 走行の動きは同じ)。直す前の写しは tests/fixtures/fix8_safety_doc_before/(= 上の
# LATER_FIX7_SAFETY の drive.py の版。SHA256SUMS)から同じようにたどる。fix8 のあとの版の sha256 はここ
FIX8_SAFETY_DOC_BEFORE = JF / 'tests' / 'fixtures' / 'fix8_safety_doc_before'
LATER_FIX8_SAFETY_DOC = {
    'jetformer/drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
}
# そのあと 10/28 fix8 の画面の係の直す 8-1 回目(反証 8-1 回目 stale-status-after-drive-tab-return。走行タブに戻った直後は表示の問い合わせを
# 読み直すまで ▶・基準の設定・ブロックを断る。表示と断る時だけ。送る物・版は同じ)が web/static/app.js を変えた。直す前の写しは
# tests/fixtures/fix8_ui_r1_before/(上の LATER_FIX7_UI_R2 の版。SHA256SUMS)。中身の違いは tests/test_ui_fix8_r1.py が見る。8-1 回目のあとの版の sha256 はここ
FIX8_UI_R1_BEFORE = JF / 'tests' / 'fixtures' / 'fix8_ui_r1_before'
LATER_FIX8_UI_R1 = {
    'jetformer/web/static/app.js': 'aa275bb5da0bbd740f38c217dd0ae9ee19240f061dafd8ca70f6916448837285',
}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_untouched_files():
    """走行の動き・既定値(DEFAULT_PARAMS・RESET_ON_BOOT は drive.py)・画面・計画のファイルは直す前と同じ。
    drive.py・bench.py は 10/27 fix4 の版でもよい(そのときは、この組のときの版の写しが上の sha256 と同じか見る)。
    drive.py・assist.py は 10/27 fix5 safety の版でもよい(そのときは fix5 の直す前の写しから同じようにたどる)。
    server.py・app.js は 10/27 fix5 画面の係の版でもよい(同じ)。app.js は 10/27 fix6 画面の係の版でもよい(fix5 の版へたどる)。
    drive.py・assist.py は 10/27 fix6 安全の係の版でもよい(fix6 の直す前の写し = fix5 の版へたどる)。assist.py は 10/27 fix6
    安全の係の直す 6-2 回目の版でもよい(6-2 回目の直す前の写し = 6-1 回目の版へたどる)。
    app.js は 10/27 fix6 画面の係の直す 6-2 回目の版でもよい(6-2 回目の直す前の写し = 6-1 回目の版へたどる)。
    app.js は 10/27 fix7 画面の係の直す 7-1 回目の版でもよい(7-1 回目の直す前の写し = 6-2 回目の版へたどる)。
    app.js は 10/27 fix7 画面の係の直す 7-2 回目の版でもよい(7-2 回目の直す前の写し = 7-1 回目の版へたどる)。
    drive.py・assist.py は 10/27 fix7 安全の係の直す 7-1 回目の版でもよい(7-1 回目の直す前の写し = fix6 の版へたどる)。
    assist.py は 10/27 fix7 安全の係の直す 7-2 回目の版でもよい(7-2 回目の直す前の写し = 7-1 回目の版へたどる)。
    drive.py は 10/28 fix8 安全の係の説明の文だけの版でもよい(その直す前の写し = 7-1 回目の版へたどる)。
    app.js は 10/28 fix8 画面の係の直す 8-1 回目の版でもよい(8-1 回目の直す前の写し = 7-2 回目の版へたどる)"""
    for rel, h in UNTOUCHED.items():
        cur = _sha(JF / rel)
        if rel in LATER_FIX8_SAFETY_DOC and cur == LATER_FIX8_SAFETY_DOC[rel]:
            cur = _sha(FIX8_SAFETY_DOC_BEFORE / Path(rel).name)  # fix8 安全の係の説明の文の直す前の写し(LATER_FIX7_SAFETY の版)
        if rel in LATER_FIX7_SAFETY_R2 and cur == LATER_FIX7_SAFETY_R2[rel]:
            cur = _sha(FIX7_SAFETY_R2_BEFORE / Path(rel).name)   # fix7 安全の係 7-2 回目の直す前の写し(LATER_FIX7_SAFETY の版)
        if rel in LATER_FIX7_SAFETY and cur == LATER_FIX7_SAFETY[rel]:
            cur = _sha(FIX7_SAFETY_BEFORE / Path(rel).name)      # fix7 安全の係 7-1 回目の直す前の写し(LATER_FIX6_SAFETY_R2・LATER_FIX6 の版)
        if rel in LATER_FIX6_SAFETY_R2 and cur == LATER_FIX6_SAFETY_R2[rel]:
            cur = _sha(FIX6_SAFETY_R2_BEFORE / Path(rel).name)   # fix6 安全の係 6-2 回目の直す前の写し(LATER_FIX6 の版)
        if rel in LATER_FIX6 and cur == LATER_FIX6[rel]:
            cur = _sha(FIX6_BEFORE / Path(rel).name)          # fix6 安全の係の直す前の写し(LATER_FIX5 の版)
        if rel in LATER_FIX8_UI_R1 and cur == LATER_FIX8_UI_R1[rel]:
            cur = _sha(FIX8_UI_R1_BEFORE / Path(rel).name)    # fix8 画面の係 8-1 回目の直す前の写し(LATER_FIX7_UI_R2 の版)
        if rel in LATER_FIX7_UI_R2 and cur == LATER_FIX7_UI_R2[rel]:
            cur = _sha(FIX7_UI_R2_BEFORE / Path(rel).name)    # fix7 画面の係 7-2 回目の直す前の写し(LATER_FIX7_UI の版)
        if rel in LATER_FIX7_UI and cur == LATER_FIX7_UI[rel]:
            cur = _sha(FIX7_UI_BEFORE / Path(rel).name)       # fix7 画面の係 7-1 回目の直す前の写し(LATER_FIX6_UI_R2 の版)
        if rel in LATER_FIX6_UI_R2 and cur == LATER_FIX6_UI_R2[rel]:
            cur = _sha(FIX6_UI_R2_BEFORE / Path(rel).name)    # fix6 画面の係 6-2 回目の直す前の写し(LATER_FIX6_UI の版)
        if rel in LATER_FIX6_UI and cur == LATER_FIX6_UI[rel]:
            cur = _sha(FIX6_UI_BEFORE / Path(rel).name)       # fix6 画面の係の直す前の写し(LATER_FIX5_UI の版)
        if rel in LATER_FIX5_UI and cur == LATER_FIX5_UI[rel]:
            cur = _sha(FIX5_UI_BEFORE / Path(rel).name)       # fix5 画面の係の直す前の写し(UNTOUCHED の版)
        if rel in LATER_FIX5 and cur == LATER_FIX5[rel]:
            cur = _sha(FIX5_BEFORE / Path(rel).name)          # fix5 の直す前の写し(fix4 の版かこの組の版)
        if rel in LATER_FIX4 and cur == LATER_FIX4[rel]:
            assert _sha(FIX4_BEFORE / Path(rel).name) == h, rel
            continue
        assert cur == h, rel


@pytest.fixture(scope='module')
def old():
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    for name, h in sums.items():
        assert _sha(FIX / name) == h, name
    return {'cc': load_mod('car_check_r32_before', FIX / 'car_check.py'),
            's3': load_mod('standing_3lap_r32_before', FIX / 'standing_3lap.py'),
            'qp': load_mod('qp_r32_before', FIX / 'quick_eval_by_params.py')}


def runc(D, name, path=COLLECT, args=(), patch=None):
    """⑤ collect.main を作り物の日で流す(patch(m) で台本の関数を差し替えられる)"""
    m = load_mod(name, path)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    if patch:
        patch(m)
    out = D.tmp / ('%s.json' % name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    txt = out.read_text(encoding='utf-8')
    return json.loads(txt), buf.getvalue(), txt


def no_path(out):
    return '\n'.join(x for x in out.splitlines() if not x.startswith('出力 '))


def both(D, name, args=(), patch=None):
    return runc(D, 'n_' + name, args=args, patch=patch), runc(D, 'o_' + name, path=FIX / 'collect.py', args=args, patch=patch)


def same(D, name, args=(), patch=None):
    (jn, on, tn), (jo, oo, to) = both(D, name, args, patch)
    return tn == to and no_path(on) == no_path(oo)


# ================================================================== 本物の流れの説明欄(r3-r3-rec-3)
BENCH_DATA = {'ok': True, 'baseline': wb.fallback_baseline(), 'candidates': [{'id': 'B', 'run': 'runB'}],
              'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'id': 'B'} for b in range(12, 20)]}]}


def on_kw(k30):
    """発進の前進が出た ON の発進(AI 側 k30/30 秒。起点まで 4.0 − k30/30 秒)"""
    t = k30 / 30.0
    return dict(t_ai=t, t_move=t + 0.1, launch=(t, 0.5, 1))


class RealDay(Day):
    """説明欄を本物の流れで作る: 雛形(web/bench._description)→ 走行開始の plan_check(ゲイン ≠ 1.00 で cand=EXP・memo に自動の
    「計画と違うので試し(元 cand=B)…」)→ ai_no。走り終わりのメモ(after = (止めた理由, memo))は after_description で scene.json に
    (POST /api/bench/after と同じ。log.json には入らない)"""

    def add_real(self, block, params, after=None, **kw):
        log = '20261103-13%02d00' % (self.n * 5)
        self.n += 1
        kw.setdefault('dur', 30.0)
        tpl = wb._description(wb.DESCRIPTION_TEMPLATE, '2026-11-03', block, 'B', 'runB')
        desc, diffs = wb.plan_check(tpl, 'runB', 5, params, data=BENCH_DATA)
        desc = wb.with_ai_no(desc, self.n)
        d, _, _ = write_standing_log(self.raw, log, run='runB', description=desc, ramp_s=0.5, params=params, **kw)
        if after is not None:
            new = wb.after_description(desc, 'C', '8.30', '7.95', after[0], after[1])
            for sc in sorted(p for p in d.iterdir() if p.is_dir()):
                (sc / 'scene.json').write_text(json.dumps({'description': new}, ensure_ascii=False), encoding='utf-8')
        return log, desc


G = dict(PRM, throttle_gain=0.92)
GON = dict(ON, throttle_gain=0.92)
HAND = ('その他', '手 9.10')
TRIAL = ('その他', '試し(ゲイン1.02で走った。走り直す)')


def round3_real(tmp, trial=None, trial_hit=False):
    """反証の probe_memo / verify_memo と同じ回 3(OFF 4 本・ON 4 本。起点まで ON 中央 2.913 秒・OFF 3.013 秒 = −0.100 で
    満たさない)。trial = 'gain': 3-6 の 1 本目をゲイン 1.02 で走ってしまい memo に「試し」(2.413 秒で R1)→ 3-6 を走り直した。
    'launch': 3-6 の 1 本目で発進の前進を入れ忘れて OFF で走り memo に「試し」→ 走り直した"""
    tmp.mkdir(parents=True, exist_ok=True)
    D = RealDay(tmp)
    D.add_real(12, G, HAND)
    D.add_real(13, GON, HAND, **on_kw(30))
    D.add_real(14, G)
    D.add_real(15, GON, HAND, **on_kw(30))
    D.add_real(16, G, HAND)
    tl = None
    if trial == 'gain':
        tl, _ = D.add_real(17, dict(GON, throttle_gain=1.02), TRIAL, hits=[(3.0, 6.0)] if trial_hit else (), **on_kw(48))
    elif trial == 'launch':
        tl, _ = D.add_real(17, G, ('その他', '試し(発進の前進を入れ忘れた)'), t_move=2.0)
    D.add_real(17, GON, HAND, **on_kw(36))
    D.add_real(18, G)
    D.add_real(19, GON, HAND, **on_kw(39))
    return D, tl


def test_rec3_auto_memo_is_on_every_round3_run(tmp_path):
    """前提(本物の流れ): 11/3 の本はゲインが 1.00 でないので、走行開始でサーバーが全部の本の memo に「計画と違うので試し」を足す。
    だから「memo に試しの字」だけで外すと回 3 の比べが消える → 自動の文は外さない"""
    D, _ = round3_real(tmp_path)
    for p in sorted(D.raw.iterdir()):
        desc = json.loads((p / 'log.json').read_text(encoding='utf-8'))['description']
        assert 'memo=計画と違うので試し(元 cand=B): ' in desc and 'cand=EXP' in desc
        assert s3.trial_memo(desc) is None


@pytest.mark.parametrize('args', [('--ignore-gain',), ()])
def test_rec3_trial_run_changes_answer_before_and_is_excluded_now(tmp_path, args):
    """反証の B・C: 試しの本(ゲイン 1.02 で 2.413 秒)が ON に入ると、直す前は「使ってよい」(答えが変わる)。今は外して、試しの本の
    無い日(A)と同じ比べ(満たさない)。「試しの 1 本」の行と JSON の trial_excluded"""
    A, _ = round3_real(tmp_path / 'a')
    B, tl = round3_real(tmp_path / 'b', trial='gain')
    (ja, oa, _), (jao, _, _) = both(A, 'a', args)
    (jb, ob, _), (jbo, obo, _) = both(B, 'b', args)
    assert jao['summary'][0]['compare']['all_ok'] is False and '→ 満たさない' in no_path(oa)
    assert jbo['summary'][0]['compare']['all_ok'] is True and '→ 使ってよい' in obo                       # 直す前(再現)
    assert jbo['summary'][0]['by_launch']['on']['n'] == 5
    g = jb['summary'][0]
    assert g['by_launch']['on']['n'] == 4 and g['compare'] == ja['summary'][0]['compare']
    assert g['compare']['all_ok'] is False and g['compare']['d_to_r1_s'] == pytest.approx(-0.1, abs=0.02)
    assert jbo['summary'][0]['compare']['d_to_r1_s'] == pytest.approx(-0.2, abs=0.02)
    row = [r for r in jb['launches'] if r['log'] == tl][0]
    tag = '%s@%.2f' % (tl, row['t_ai'])
    assert g['trial_excluded'] == ['%s(on・その他・試し(ゲイン1.02で走った。走り直す))' % tag]
    assert '試しの 1 本(説明欄の memo に「試し」。比べに入れない。ON の組にも OFF の組にも入れない。手順書 7 章): %s' % tag in ob
    assert '→ 満たさない' in ob and '→ 使ってよい' not in ob
    assert row['trial'] == 'その他・試し(ゲイン1.02で走った。走り直す)'
    assert [r for r in jb['launches'] if r['log'] != tl] == [r for r in jbo['launches'] if r['log'] != tl]
    # 試しの本の無い日は直す前とまるごと同じ(自動の「計画と違うので試し」と人の「手 9.10」があっても)
    assert same(A, 'a2', args)
    # --skip でも外せる(前と同じ): 試しを --skip すると、試しの行の無い A と同じまとめ
    jbs, _, _ = runc(B, 'n_bs', args=list(args) + ['--skip', tl])
    assert jbs['summary'][0]['compare'] == ja['summary'][0]['compare'] and 'trial_excluded' not in jbs['summary'][0]


def test_rec3_trial_launch_forgotten_off_is_not_in_off_group(tmp_path):
    """反証の E: 発進の前進を入れ忘れて OFF で走り memo に「試し」。直す前は OFF の組に入る(n = 5)。今は外す(n = 4)"""
    D, tl = round3_real(tmp_path, trial='launch')
    (jn, on, _), (jo, _, _) = both(D, 'e', ('--ignore-gain',))
    assert jo['summary'][0]['by_launch']['off']['n'] == 5
    g = jn['summary'][0]
    row = [r for r in jn['launches'] if r['log'] == tl][0]
    assert g['by_launch']['off']['n'] == 4
    assert g['trial_excluded'] == ['%s@%.2f(off・その他・試し(発進の前進を入れ忘れた))' % (tl, row['t_ai'])]


def test_rec3_trial_on_first_lap_hit_still_counts(tmp_path):
    """試しの ON の本の最初の周の衝突は「最初の周の衝突 0」に数える(止める条件。途中で変えた本と同じ)"""
    D, tl = round3_real(tmp_path, trial='gain', trial_hit=True)
    jn, on, _ = runc(D, 'n_h', args=('--ignore-gain',))
    g = jn['summary'][0]
    assert g['trial_on_collisions'] == 1 and g['compare']['checks']['first_lap_collisions_0'] is False
    assert '。うち ON の発進の最初の周の衝突 1(比べの「最初の周の衝突 0」には数える)' in on


def test_rec3_standing_3lap_and_by_params_mark_trial(tmp_path, old):
    """③: 試しの本は trial・発進を数えない(counted = False・表に「数えない(試し)」・設定の文に「試し」)。④: 別の組(--ignore gain でも)・
    印に「試し」。試しでない本(自動の文と「手 9.10」)は ③ の結果が直す前とまるごと同じ"""
    D, tl = round3_real(tmp_path, trial='gain')
    bench = tmp_path / 'bench.json'
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}]}), encoding='utf-8')
    logs, _ = s3.day_logs('20261103', D.raw)
    res = s3.run(logs, D.raw, tmp_path / 'nolab', str(bench))
    rt = [r for r in res if r['log'] == tl][0]
    assert rt['trial'] == 'その他・試し(ゲイン1.02で走った。走り直す)' and 'trial' in rt['settings']
    assert rt['settings_text'].endswith('・試し(その他・試し(ゲイン1.02で走った。走り直す))')
    assert rt['standing'] and all(s['counted'] is False and s['trial'] is True for s in rt['standing'])
    assert s3.best_standing(rt) is None
    md = s3.to_markdown(res)
    assert '・数えない(試し)' in md and '- 試し = 説明欄の memo に人が「試し」と書いた本' in md
    ores = old['s3'].run(logs, D.raw, tmp_path / 'nolab', str(bench))
    for a, b in zip(ores, res):
        if b['log'] != tl:
            assert json.dumps(a, default=s3._json_default, sort_keys=True) == json.dumps(b, default=s3._json_default, sort_keys=True)
    assert 'trial' not in [r for r in ores if r['log'] == tl][0]                     # 直す前(再現): 印なし
    md_new, runs_new, g_new = qp.to_markdown(res, '20261103', None, ('gain',))
    gt = [g for g in g_new if tl in g['logs']]
    assert len(gt) == 1 and gt[0]['logs'] == [tl] and '試し(数えない)' in gt[0]['text'] and gt[0]['standing_n'] == 0
    assert any(f.startswith('試し(その他・試し(ゲイン1.02') for f in [x for x in runs_new if x['log'] == tl][0]['flags'])
    # 試しの本を除いた日(③ の結果)なら ④ も直す前と同じ
    rest = [r for r in res if r['log'] != tl]
    for ig in ((), ('gain',), ('launch',)):
        assert qp.to_markdown(rest, '20261103', None, ig) == old['qp'].to_markdown(rest, '20261103', None, ig)


# ================================================================== r3-r3-rec-4(⑤・③)
def round3_imu(tmp, scales, us=None):
    """反証の probe_slip / probe_slip_none と同じ回 3: OFF・ON の交互 8 本。scales = {番号: IMU の前の加速 ÷ 車輪}"""
    tmp.mkdir(parents=True, exist_ok=True)
    D = Day(tmp)
    logs = []
    for i in range(8):
        kw = {}
        if i in scales:
            kw['imu_scale'] = scales[i]
        p = ON if i % 2 else PRM
        if us and i in us:
            p = dict(p, launch_us=us[i])
        lg, _ = D.add(12 + i, p, **kw)
        logs.append(lg)
    return D, logs


@pytest.mark.parametrize('sc', [0.05, 0.10])
def test_rec4_extreme_slip_fails_now(tmp_path, sc):
    """3-4(ON)の IMU だけ × 0.05・× 0.10(車輪 +1.8 m/s・IMU +0.09・+0.18 m/s)。直す前は比 None が黙って抜け「使ってよい」。
    今は「いちばんひどい空転」= 比 > 1.3 とみなし、空転の条件 False・満たさない。ほかの条件・発進ごとの行は同じ"""
    D, logs = round3_imu(tmp_path, {3: sc})
    (jn, on, _), (jo, oo, _) = both(D, 'x', ('--ignore-gain',))
    co, cn = jo['summary'][0]['compare'], jn['summary'][0]['compare']
    assert co['all_ok'] is True and co['checks']['slip'] is True and '→ 使ってよい' in oo           # 直す前(再現)
    assert cn['checks']['slip'] is False and cn['all_ok'] is False and '→ 満たさない' in on
    assert {k: v for k, v in cn['checks'].items() if k != 'slip'} == {k: v for k, v in co['checks'].items() if k != 'slip'}
    on_ = jn['summary'][0]['by_launch']['on']
    assert on_['slip_max'] == jo['summary'][0]['by_launch']['on']['slip_max']                       # 空転max の値は同じ
    assert on_['slip_extreme'] == ['%s@1.30(車輪 +1.80・IMU +%.2f m/s)' % (logs[3], round(1.8 * sc, 2))]
    assert '空転がいちばんひどい ON の発進 1 本(車輪が 0.5 m/s より増えたのに IMU の積分が 0.2 m/s 以下で比が出ない' in on
    assert jn['launches'] == jo['launches']


def test_rec4_all_on_extreme_was_already_not_satisfied(tmp_path):
    """反証の「安全な場合」: ON 4 本とも × 0.05 は前も満たさない(空転max —)。今は名前の行が出る"""
    D, logs = round3_imu(tmp_path, {1: 0.05, 3: 0.05, 5: 0.05, 7: 0.05})
    (jn, on, _), (jo, _, _) = both(D, 'all', ('--ignore-gain',))
    assert jo['summary'][0]['compare']['all_ok'] is False and jn['summary'][0]['compare']['all_ok'] is False
    assert len(jn['summary'][0]['by_launch']['on']['slip_extreme']) == 4 and '空転がいちばんひどい ON の発進 4 本' in on


@pytest.mark.parametrize('scales', [{3: 1.0}, {3: 0.6}, {}, {2: 0.05}, {0: 0.05, 2: 0.10}])
def test_rec4_same_as_before_when_ratio_is_out_or_off(tmp_path, scales):
    """比が出る ON(× 1.0・× 0.6 = 1.69 で前から満たさない)と、OFF の本だけ比が出ない日は、直す前とまるごと同じ(JSON・標準出力)"""
    D, _ = round3_imu(tmp_path, scales)
    for args in (('--ignore-gain',), ()):
        assert same(D, 's%d' % len(args), args)


def test_rec4_mixed_launch_us_uses_last_strength(tmp_path):
    """3-2(250 µs)がいちばんひどい空転 → 3-4・3-6・3-8 を 200 µs に下げた。比べは最後の強さ(200)の ON なので、空転の条件は 200 の
    本で見る(直す前と同じ答え)。250 の本の印は by_launch の on にだけ残り、行は出ない"""
    D, logs = round3_imu(tmp_path, {1: 0.05}, us={3: 200.0, 5: 200.0, 7: 200.0})
    (jn, on, _), (jo, oo, _) = both(D, 'mx', ('--ignore-gain',))
    assert jn['summary'][0]['compare'] == jo['summary'][0]['compare']
    assert jn['summary'][0]['launch_us_used'] == 200.0 and 'slip_extreme' not in jn['summary'][0]['on_used']
    assert jn['summary'][0]['by_launch']['on']['slip_extreme'] == ['%s@1.30(車輪 +1.80・IMU +0.09 m/s)' % logs[1]]
    assert 'いちばんひどい' not in on and no_path(on) == no_path(oo)


def _patch_slip(target, value):
    """⑤ の slip_window を、記録 target の発進だけ value にする"""
    def patch(m):
        orig = m.slip_window
        m.slip_window = lambda d, a, b: value if d['log'] == target else orig(d, a, b)      # noqa: E731
    return patch


@pytest.mark.parametrize('value,why', [
    (None, '窓が短い(LH・R1 が近い)か IMU が抜けた'),
    ({'t_a': 1.4, 't_b': 2.0, 'dv_wheel': 0.3, 'dv_imu': 0.1, 'ratio': None}, '車輪 +0.30・IMU +0.10 m/s。どちらもほとんど増えない')])
def test_rec4_unmeasured_on_slip_fails_now(tmp_path, value, why):
    """ON の発進の比が出ない(窓が短い・IMU が抜けた・どちらもほとんど増えない)。直す前は黙って抜け「使ってよい」。今は
    checks に slip_all_measured = False・満たさない・名前と理由の行"""
    D, logs = round3_imu(tmp_path, {})
    (jn, on, _), (jo, oo, _) = both(D, 'u', ('--ignore-gain',), patch=_patch_slip(logs[3], value))
    assert jo['summary'][0]['compare']['all_ok'] is True                                            # 直す前(再現)
    cn = jn['summary'][0]['compare']
    assert cn['checks']['slip_all_measured'] is False and cn['all_ok'] is False and cn['checks']['slip'] is True
    assert {k: v for k, v in cn['checks'].items() if k != 'slip_all_measured'} == jo['summary'][0]['compare']['checks']
    assert jn['summary'][0]['by_launch']['on']['slip_unmeasured'] == ['%s@1.30(%s)' % (logs[3], why)]
    assert '空転の比が出ない ON の発進 1 本: %s@1.30(%s) → 空転を確かめられない = 満たさない' % (logs[3], why) in on
    # OFF の発進の比が出なくても前と同じ
    D2, logs2 = round3_imu(tmp_path / 'off', {})
    assert same(D2, 'uo', ('--ignore-gain',), patch=_patch_slip(logs2[2], value))


def test_rec4_standing_3lap_marks_extreme_and_unmeasured(tmp_path, old, monkeypatch):
    """③: いちばんひどい空転の発進に「> 1.3 とみなす」の印(直す前は印なし)→ ④ の「確かめる」(最速・25 秒以下に入れない)。
    比が出ない発進(窓・IMU)にも「確かめる: 空転の比が出ない」。比が出る発進は前と同じ"""
    D, logs = round3_imu(tmp_path, {1: 0.6, 3: 0.05, 5: 1.0})
    names = {'runB': 'B'}
    a = s3.analyze_log(logs[3], D.raw, tmp_path / 'nolab', names)
    b = old['s3'].analyze_log(logs[3], D.raw, tmp_path / 'nolab', names)
    st_ = [s for s in a['standing'] if s.get('anchors')][0]
    assert st_['launch_slip']['ratio'] is None and 'いちばんひどい空転とみなす > 1.3' in st_['note']
    assert '空転' not in ([s for s in b['standing'] if s.get('anchors')][0].get('note') or '')       # 直す前(再現): 空転の印なし
    for i in (1, 5):
        x, y = s3.analyze_log(logs[i], D.raw, tmp_path / 'nolab', names), old['s3'].analyze_log(logs[i], D.raw, tmp_path / 'nolab', names)
        assert json.dumps(x, default=s3._json_default, sort_keys=True) == json.dumps(y, default=s3._json_default, sort_keys=True)
    assert qp.doubtful(st_)
    monkeypatch.setattr(s3, 'launch_slip', lambda *a, **k: None)       # 窓(0.6 秒)に IMU が 5 サンプルも無い = IMU が抜けた
    c = s3.analyze_log(logs[5], D.raw, tmp_path / 'nolab', names)
    assert [s for s in c['standing'] if s.get('anchors')][0]['note'].startswith('確かめる: 空転の比が出ない(動いてから 0.6 秒の間に IMU が抜けた)')
    # 車輪も IMU もほとんど増えない(空転していない)発進・IMU の無い呼び方には ③ の印を付けない(前と同じ。⑤ は満たさない)
    monkeypatch.setattr(s3, 'launch_slip', lambda *a, **k: {'t_a': 1.4, 't_b': 2.0, 'dv_wheel': 0.3, 'dv_imu': 0.1, 'ratio': None})
    c2 = s3.analyze_log(logs[5], D.raw, tmp_path / 'nolab', names)
    assert '空転' not in ([s for s in c2['standing'] if s.get('anchors')][0].get('note') or '')


# ================================================================== r3-r3-rec-2(car_check)
def cc_run(D, log, mod=cc):
    r = mod.check_log(log, D.tmp / 'nolab', D.raw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main([log, '--raw', str(D.raw), '--labels', str(D.tmp / 'nolab'), '--no-json'])
    return r, mod.to_text(r), rc, buf.getvalue()


def _noseconds(t):
    return '\n'.join(x for x in t.splitlines() if not (x.startswith('  (') and x.endswith('秒)')))


def _nores(r):
    return json.dumps({k: v for k, v in r.items() if k not in ('created_at', 'seconds', 'launch_slip')},
                      default=cc._json_default, sort_keys=True, ensure_ascii=False)


@pytest.mark.parametrize('sc,kind,advice', [
    (0.6, 'high', '→ 止まっているうちに、残りの ON を launch_us 200 に'),
    (0.05, 'extreme', '→ 止まっているうちに、残りの ON を launch_us 200 に'),
    (1.0, 'ok', None)])
def test_rec2_car_check_shows_launch_slip_for_on(tmp_path, old, sc, kind, advice):
    """回 3 の 3-2(最初の ON)の car_check に「発進の空転の比」の行。値は ③ standing_3lap と同じ。判定・終わりの番号は前と同じ。
    直す前の car_check には空転の比が無い(指摘の再現)"""
    D, logs = round3_imu(tmp_path, {1: sc})
    r, t, rc, out = cc_run(D, logs[1])
    ro, to, rco, _ = cc_run(D, logs[1], old['cc'])
    assert '空転' not in to and 'launch_slip' not in ro                                             # 直す前(再現)
    assert r['judge'] == ro['judge'] and rc == rco and _nores(r) == _nores(ro)
    ls = r['launch_slip']
    st_ = [s for s in s3.analyze_log(logs[1], D.raw, tmp_path / 'nolab3', {'runB': 'B'})['standing'] if s.get('t_move') is not None]
    assert [x['slip'] for x in ls['starts']] == [s.get('launch_slip') for s in st_]
    assert [x['kind'] for x in ls['starts']] == [kind] and ls['starts'][0]['started'] is True
    head = [x for x in t.splitlines() if '発進の空転の比' in x]
    assert len(head) == 1 and head[0].startswith('!' if advice else ' ')
    assert '発進の前進 ON・前進 250 µs' in head[0] and '発進 1(AI 側 1.30 秒・発進の前進 0.50 秒・速さに届いた)' in head[0]
    if kind == 'high':
        assert '%.2f > 1.3' % ls['starts'][0]['slip']['ratio'] in head[0]
    elif kind == 'extreme':
        assert 'いちばんひどい空転とみなす > 1.3' in head[0]
    else:
        assert '(≤ 1.3)' in head[0]
    lines = cc.launch_slip_lines(ls)
    assert (advice in t and len(lines) == 2) if advice else len(lines) == 1
    assert [x for x in _noseconds(t).splitlines() if x not in lines] == _noseconds(to).splitlines()
    assert head[0] in out


def test_rec2_car_check_200us_and_not_started(tmp_path):
    """もう 200 µs で > 1.3 →「⑤ は空転の条件を満たさない」。発進の前進が出ていない本(記録に欄が無い・出ず)は「数えない」で印なし"""
    D, logs = round3_imu(tmp_path, {1: 0.6}, us={1: 200.0})
    r, t, _, _ = cc_run(D, logs[1])
    assert '→ もう 200 µs 以下。⑤ collect.py は空転の条件を満たさない' in t and 'launch_us 200 に' not in t
    lg, _ = D.add(13, ON, imu_scale=0.6, launch=None)
    r2, t2, _, _ = cc_run(D, lg)
    x = r2['launch_slip']['starts'][0]
    assert x['started'] is False and x['kind'] == 'high' and r2['launch_slip']['warn'] is False
    assert '出ていないので 7 章の空転の行には数えない' in t2 and '  →' not in t2.split('発進の空転の比')[1]


@pytest.mark.parametrize('i', [0, 2])
def test_rec2_car_check_off_is_same_as_before(tmp_path, old, i):
    """発進の前進 OFF の本は、結果(JSON)・文・終わりの番号が直す前と同じ(かかった秒のほか)"""
    D, logs = round3_imu(tmp_path, {0: 0.6, 2: 0.05})
    r, t, rc, out = cc_run(D, logs[i])
    ro, to, rco, oo = cc_run(D, logs[i], old['cc'])
    assert 'launch_slip' not in r and _nores(r) == _nores(ro) and rc == rco
    assert _noseconds(t) == _noseconds(to) and _noseconds(out) == _noseconds(oo)


# ================================================================== 手順書
def test_runbook_reads_slip_each_run_and_trial():
    """手順書: 4 章 10 の car_check の表に「発進の空転の比」・7 章の空転の行が car_check で読む・250/200 は「最後の ON の強さ」・
    試しの書き方。「まだ決めていない」の文は消えた"""
    t = RUNBOOK.read_text(encoding='utf-8')
    ch4 = t.split('## 4. 1 本の手順')[1].split('## 5.')[0]
    ch6 = t.split('## 6. 本番の形の決め方')[1].split('## 7.')[0]
    ch7 = t.split('## 7. 止める条件')[1].split('## 8.')[0]
    ch81 = t.split('### 8.1')[1].split('### 8.2')[0]
    assert '| 発進の空転の比' in ch4 and '発進の空転の比' in ch7
    for c in (ch6, ch7, ch81):
        assert 'まだ決めていない' not in c and 'まだ決まっていない' not in c and '最後の ON の強さ' in c
    assert '「試し」' in ch7 and '計画と違うので試し' in ch7 and '試しの N 本' in ch81
    assert '空転がいちばんひどい ON の発進' in ch81 and '空転の比が出ない ON の発進' in ch81
