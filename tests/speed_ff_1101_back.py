"""2026-11-01 電圧の前送り(speed_ff。~/jetformer_reports/bench_20261030/speed_ff/README.md)の「見張りの段」。試験だけが使う
(走行・サーバー・道具は読まない。pytest が集める名前 test_*.py ではない)。

この直しは jetformer/drive.py・jetformer/web/server.py・jetformer/web/static/app.js・index.html・jetformer/recorder.py と、文書の版の
台本 experiments/20261026_fix/fix6/ui/docs_ban9.py を変え、画面の版を 10 → 11 にした。前の直しの試験には、ファイルの sha256・版・AST・既定値を決め打ちして「このあと誰かが直せば落ちて
知らせる」見張りがある。それぞれに、10/29 fix9・10/31 USB を外して走る と同じ形の 1 段を足す:

    今のファイルがこの直しの版(sha256 が LATER_SPEED_FF_1101)なら、直す前の写し tests/fixtures/speed_ff_1101_before/
    (SHA256SUMS と BEFORE_SHA と中身の sha256 の 3 つがそろうのを確かめてから。= 10/31 USB を外して走る のあとの版 = 11/1 の本物)
    で照らし、そこから前の段(LATER_TETHER_1031 → tests/fixtures/tether_1031_before/ …)をたどる。

1 つでも違えば(今のファイルがこの直しの版でない・写しの中身が違う)今まで通り今のファイルで照らす(= 落ちて知らせる)。弱めない。
直しの中身そのもの(変えた所だけか・OFF がビット単位で同じか など)は tests/test_speed_ff.py が見る。

本物 ~/jetformer を絶対パスで読む試験(文書の試験)は、そのパスの下の tests/fixtures/speed_ff_1101_before/ を使う(back の root)。
"""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path('tests') / 'fixtures' / 'speed_ff_1101_before'
# この直しの版(写しの最後のファイル)。直したら tests/test_speed_ff.py の test_later_is_this_fix が落ちて知らせる
LATER_SPEED_FF_1101 = {
    'jetformer/drive.py': 'dc719a7c7e6d53c848a5803bb93fea2a90abfda7db7131745e2c3d2af0dc166f',
    'jetformer/web/server.py': '1841a87da1e8be8a7171ebd80b548585f849b20ebd55928e091d52601f0fa091',
    'jetformer/web/static/app.js': '1d13b5c9fc85cdee8f97ac351af51ab79976af2d4dd50f584df59a6d90112b26',
    'jetformer/web/static/index.html': 'ddabf9c90c3e12755698ac6e9d94d2740699e16d7ae0bfdc7656e6e3b7ca1681',
    'jetformer/recorder.py': '01497034c9288e35c56b5e7dfc3378be87519a8d2a6bf29566d9aed6944d10f0',
    'experiments/20261026_fix/fix6/ui/docs_ban9.py': '725c38a8c561a2a7709db9bbf423bd17dee197eba9d664ddf52e50b2a1eb2a32',      # 文書の版の台本(版 11 → 写しの版を読む段)
    'tests/test_vbat_ui.py': '62552914263f8243189ba2c59f870ff0f9b8b313f3f7a61e52a83f130f16e143',                          # 版の 1 行(client: 10 → 11。tests/test_ui_fix8_r2.py が照らす)
}
# 2026-11-03 録画のロックの直しの版の recorder.py(tests/test_rec_imu_1103.py)。AI_CMD の欄は変えていないので、この段は同じ写しで照らす。
# 直したら tests/test_rec_imu_1103.py の test_later_is_this_fix が落ちて知らせる
ALSO_LATER = {'jetformer/recorder.py': 'b5d77cea4c363b6f9302977b4591402ae7502c42c30435d90ef42fe4c15a4783'}
# 直す前の写しの sha256(= 10/31 USB を外して走る のあとの版。drive.py・server.py・app.js・index.html は tests/test_ui_fix6.py・
# tests/test_fix3_r3_2_tools.py などの LATER_TETHER_1031 と同じ値。recorder.py は tether_fix/README.md 4.2 の b968d694…)
BEFORE_SHA = {
    'drive.py': '5431ea8ac653dabdaaa64f78075894f32ad771e3ed0c65b5c3384f203c448aa3',
    'server.py': '0c73072ae6df3b1e8c16b2e15859130003b595222237605ee2203597bf320605',
    'app.js': '6cbdc98db2ea56aa3a1355cca6b9b57f118609cda1b554a9606f1edacd8b569b',
    'index.html': 'cca3d8f8ddfcd2b0b1de78c8c5dcb95649d83b770e1f6701f3696f3e97678ba1',
    'recorder.py': 'b968d6943b0eebae2797b40b9fffdbed6bcf55b97da2a9824b287df798075199',
    'docs_ban9.py': 'e974f103fd9c9d2e312eb5fd454527a2b9b1177b41dbb0d99f9433e7cfe1e198',     # tether_1031_docsver のあと
    'test_vbat_ui.py': 'd44eabe77f78155c111fe0cd06e5007cee635a4c0797b0179a0d42a2f79da39b',  # = tests/test_ui_fix8_r2.py の LATER_TETHER_1031
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def before_dir(root=None):
    return Path(root if root is not None else ROOT) / FIXTURE


def check_before(root=None):
    """直す前の写し(root の tests/fixtures/speed_ff_1101_before/)の SHA256SUMS・BEFORE_SHA・中身の sha256 がそろうか。
    そろわなければ AssertionError(段を通さない)"""
    d = before_dir(root)
    lines = (d / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    sums = dict(reversed(x.split(None, 1)) for x in lines if x.strip())
    assert sums == BEFORE_SHA, '直す前の写し %s の SHA256SUMS が違います: %r' % (d, sums)
    for name, h in BEFORE_SHA.items():
        assert sha(d / name) == h, '直す前の写し %s の中身が違います' % (d / name)
    return d


def later_rel(path):
    """path がこの直しの版のファイルなら、その LATER_SPEED_FF_1101 の名前(root からの位置)。違えば None"""
    p = Path(path)
    s = p.as_posix()
    for rel, h in LATER_SPEED_FF_1101.items():
        if (s == rel or s.endswith('/' + rel)) and p.is_file() and sha(p) in (h, ALSO_LATER.get(rel)):
            return rel
    return None


def back(path, root=None):
    """今のファイル path がこの直しの版なら、直す前の写し(path の root の tests/fixtures/speed_ff_1101_before/<名前>。中身を確かめて
    から)。違えば path のまま(今まで通り)。root を渡さなければ path から決める(path = <root>/<LATER の名前>)"""
    p = Path(path)
    rel = later_rel(p)
    if rel is None:
        return p
    if root is None:
        s = p.as_posix()
        root = Path(s[:len(s) - len(rel)].rstrip('/') or '.')
    return check_before(root) / p.name


_MODS = {}


def back_module(mod, name='drive.py'):
    """今のモジュール mod(jetformer.drive など)のファイルがこの直しの版なら、直す前の写しを読んだモジュール(sha256 を確かめてから。
    同じ写しは 1 回だけ読む)。違えば mod のまま。既定値・決まりの表(DEFAULT_PARAMS・PARAM_LIMITS・RESET_ON_BOOT など)を前の版と
    比べる見張りが使う"""
    p = Path(mod.__file__)
    if later_rel(p) is None:
        return mod
    src = back(p)
    key = str(src)
    if key not in _MODS:
        spec = importlib.util.spec_from_loader('%s._speed_ff_1101_before' % mod.__name__, loader=None)
        m = importlib.util.module_from_spec(spec)
        m.__package__ = mod.__package__
        m.__file__ = key
        exec(compile(src.read_text(encoding='utf-8'), key, 'exec'), m.__dict__)
        _MODS[key] = m
    return _MODS[key]
