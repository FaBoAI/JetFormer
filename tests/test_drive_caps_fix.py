"""区間の速度上限(drive.py の zone_caps)の確認 V-drive で出た問題の直し(Fix-drive、2026-10-21)の試験。
実機・GPU・torch は使わない(tests/test_drive_caps.py の偽物と決まった時刻の回し方を使う)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big -- python -m pytest -q tests/test_drive_caps_fix.py

確かめること:
  - [major] v1.1 の 4 つめの上限(位相が不確か = unknown で 3.0 m/s)を、status の zone_caps_spec と画面の ON の確認で
    はっきり示す(ユーザーの了承は 3 区間だけ)。上限の値・走行の出力は変えない
  - [minor] 画面の数(stats の zone_applied・zone_brakes)は、PCA9685 に書けたフレームだけを数える(記録と同じ)
  - [minor] OFF の日に上限の計算が例外になったときの文は「上限を外しました」でなく「影の判定が止まった・出力は変わらない」
  - [minor] 走行中に OFF → ON はサーバーでも断る(ON → OFF はできる)。走っていなければ ON にできる
"""
import json
import os
import re
import shutil
import subprocess

import pytest

from jetformer import drive
from jetformer import safety_caps as sc

from test_drive_caps import PARAMS, FakePCA, FakeRecorder, run_det, scenario

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STATIC = os.path.join(ROOT, 'jetformer', 'web', 'static')
APP_JS = os.path.join(STATIC, 'app.js')
NODE = shutil.which('node')


# ---------------------------------------------------------------- [major] 4 つめの上限を示す
def test_spec_states_fourth_cap_and_values_unchanged():
    sp = drive.ZONE_CAPS_SPEC
    assert sp['version'] == 'v1.1'
    assert sp['caps_mps'] == {'R1': 2.5, 'GM': 2.2, 'DE': 3.0, 'unknown': 3.0}       # 値は変えていない(G1 の形のまま)
    assert sp['approved_zones'] == ['R1', 'GM', 'DE']                              # 了承は 3 区間だけ
    assert sp['unknown_rule'] == {'cap_mps': 3.0, 'lost_s': 16.0, 'stop_reset_s': 1.0}
    # 走行ループが使う設定と同じ値から作っている
    cfg = sc.ZONE_CAPS_V1_1
    assert (cfg.cap.cap_unknown, cfg.phase.lost_s, cfg.phase.stop_reset_s) == (3.0, 16.0, 1.0)
    assert drive._ZoneCaps().cfg is cfg
    note = sp['unknown_note']
    for words in ('4 つめ', '3.0 m/s', '1 周目', '16 秒', '1 秒以上止まった', '3 区間', '了承'):
        assert words in note, words
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    st = d.status()
    assert st['zone_caps_spec']['unknown_note'] == note and st['params']['zone_caps'] == 0
    json.dumps(st['zone_caps_spec'])                                             # 画面へそのまま送れる


def test_screen_shows_fourth_cap_in_panel_and_on_confirm():
    html = open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read()
    m = re.search(r'<div id="dr-zone-unknown"[^>]*>(.*?)</div>', html, re.S)
    assert m and '4 つめ' in m.group(1) and '3 区間' in m.group(1) and '1 周目' in m.group(1)
    assert 'G1 に合格し、ユーザーが了承したあと' in html
    js = open(APP_JS, encoding='utf-8').read()
    # ON の確認の文: 4 つめの説明を入れ、了承が 4 つめにも出ているかを聞く
    on = re.search(r"\$\('#dr-zone-caps'\)\.addEventListener\('change'.*?^\}\);\n", js, re.S | re.M).group(0)
    assert 'zoneUnknownNote(st)' in on and '4 つめ' in on and 'G1 に合格し、ユーザーが了承したあと' in on
    # 走行開始の確認の文(ON のとき)にも
    start = re.search(r"const zoneLine = .*", js).group(0)
    assert 'zoneUnknownCap' in start and '1 周目' in start
    if NODE:
        r = subprocess.run([NODE, '--check', APP_JS], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def _node(code):
    js = open(APP_JS, encoding='utf-8').read()
    parts = [re.search(r"^const ZONE_UNKNOWN_FALLBACK = '.*?';\n", js, re.M).group(0)]
    for name in ('zoneSpecText', 'zoneUnknownCap', 'zoneUnknownNote'):
        m = re.search(r'^function %s\(.*?^}\n' % name, js, re.S | re.M)
        assert m, name
        parts.append(m.group(0))
    r = subprocess.run([NODE, '-e', '\n'.join(parts) + '\n' + code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_fourth_cap_text_follows_server_spec():
    spec = json.dumps(drive.ZONE_CAPS_SPEC, ensure_ascii=False)
    no_unknown = dict(drive.ZONE_CAPS_SPEC, caps_mps=dict(drive.ZONE_CAPS_SPEC['caps_mps'], unknown=None), unknown_note='')
    got = _node('const sp = %s, sp2 = %s;\nconst cases = [{zone_caps_spec: sp}, {zone_caps_spec: sp2}, {}, null];\n'
                'process.stdout.write(JSON.stringify(cases.map((st) => [zoneSpecText(st), zoneUnknownCap(st), zoneUnknownNote(st)])));'
                % (spec, json.dumps(no_unknown, ensure_ascii=False)))
    # 今のサーバー: サーバーの説明をそのまま出す・ラベルにも 4 つめ
    assert got[0][1] == 3.0 and got[0][2] == drive.ZONE_CAPS_SPEC['unknown_note']
    assert 'R1 2.5' in got[0][0] and '4 つめ' in got[0][0] and '3.0' in got[0][0]
    # 4 つめの上限が無い設定なら出さない
    assert got[1][1] is None and got[1][2] == '' and '4 つめ' not in got[1][0]
    # 説明を返さないサーバー(再起動前)では v1.1 の控えの文(スイッチは押せないが、注意は出す)
    for g in got[2:]:
        assert g[1] == 3.0 and '4 つめ' in g[2] and '3 区間' in g[2]


# ---------------------------------------------------------------- [minor] 画面の数は書けたフレームだけ
class FlakyPCA(FakePCA):
    """ブレーキ(中立 + 200 µs)の最初の 2 回と、ほかの書き込みの 6 回に 1 回を失敗にする(続けての失敗にはならない:
    失敗のあと、中立で止める書き込みは通る)"""

    def __init__(self):
        super().__init__()
        self.n, self.brake_fail, self.failed = 0, 0, []

    def set_pair_us(self, sc_, s, tc, t):
        self.n += 1
        neutral = abs(t - 1496.0) < 0.5
        if not neutral and ((t >= 1690.0 and self.brake_fail < 2) or self.n % 6 == 0):
            self.brake_fail += t >= 1690.0
            self.failed.append((s, t))
            raise OSError('試験の書き込みの失敗')
        super().set_pair_us(sc_, s, tc, t)


def test_counters_count_only_written_frames(monkeypatch, tmp_path):
    frames, imu = scenario()
    pca = FlakyPCA()
    monkeypatch.setattr('test_drive_caps.FakePCA', lambda: pca)     # run_det が作る PCA を差し替える
    d, _, rec = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames, imu)
    rows = [r for _, r in rec.rows]
    n_applied = sum(bool(r['cap_applied']) for r in rows)
    n_brakes = sum(bool(r['cap_brake_start']) for r in rows)
    assert pca.brake_fail >= 2 and len(pca.failed) > 20                         # ブレーキの書き込みの失敗が起きた
    assert any(r['hold'] for r in rows)                                          # 書けなかったフレームは中立で止めた行
    assert n_applied > 20 and n_brakes >= 1
    assert d.st['stats']['zone_applied'] == n_applied                           # 記録の数と同じ(前は失敗も数えた)
    assert d.st['stats']['zone_brakes'] == n_brakes
    assert d.st['stats']['i2c_errors'] == len(pca.failed)


def test_counters_unchanged_without_write_failures(monkeypatch, tmp_path):
    frames, imu = scenario()
    d, _, rec = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames, imu)
    rows = [r for _, r in rec.rows]
    assert d.st['stats']['zone_applied'] == sum(bool(r['cap_applied']) for r in rows) > 20
    assert d.st['stats']['zone_brakes'] == sum(bool(r['cap_brake_start']) for r in rows) >= 2
    # OFF でも影のブレーキの始まりは数える(出力は変えない)
    d0, _, rec0 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    assert d0.st['stats']['zone_applied'] == 0
    assert d0.st['stats']['zone_brakes'] == sum(bool(r['cap_brake_start']) for _, r in rec0.rows) >= 2


# ---------------------------------------------------------------- [minor] OFF の日の失敗の文
def test_failure_wording_off_vs_on(monkeypatch, tmp_path):
    frames, imu = scenario(260)
    real = sc.zone_frame
    calls = [0]

    def boom(*a, **kw):
        calls[0] += 1
        if calls[0] == 40:
            raise RuntimeError('試験の例外')
        return real(*a, **kw)
    monkeypatch.setattr(sc, 'zone_frame', boom)
    d, _, rec = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    for text in (d.st['recording_note'], d.st['message']):
        assert '上限を外しました' not in text
        assert '影の判定' in text and '出力は変わりません' in text and 'RuntimeError' in text
    assert [(k, e['zone_caps']) for k, e in rec.events] == [('zone_caps_error', 0)]
    assert 'RuntimeError' in d.st['zone_caps_error']
    # ON の日は今まで通り「上限を外しました」
    calls[0] = 0
    d1, _, rec1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames, imu)
    assert '上限を外しました' in d1.st['recording_note'] and '上限を外しました' in d1.st['message']
    assert '影の判定' not in d1.st['message']
    assert [(k, e['zone_caps']) for k, e in rec1.events] == [('zone_caps_error', 1)]


def test_screen_failure_wording_off_vs_on():
    js = open(APP_JS, encoding='utf-8').read()
    fn = re.search(r'^function renderZoneSwitch\(.*?^}\n', js, re.S | re.M).group(0)
    assert 'srvOn' in fn and 'この走行では上限を外しました' in fn and '影の判定が計算の失敗で止まりました' in fn
    live = re.search(r'^function zoneLiveText\(.*?^}\n', js, re.S | re.M).group(0)
    assert 'z.on ?' in live and '外しました' in live and '影の判定' in live


# ---------------------------------------------------------------- [minor] 走行中の OFF → ON はサーバーでも断る
def test_set_params_refuses_zone_caps_on_while_running(monkeypatch, tmp_path):
    path = tmp_path / 'drive_settings.json'
    monkeypatch.setattr(drive, 'SETTINGS_PATH', path)
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    assert d.params['zone_caps'] == 0
    running = [True]
    monkeypatch.setattr(drive.Driver, 'running', property(lambda self: running[0]))
    before = dict(d.params)
    saved = path.read_text() if path.exists() else None
    with pytest.raises(ValueError, match='走行中は区間の速度上限を ON にできません'):
        d.set_params({'zone_caps': 1})
    with pytest.raises(ValueError):
        d.set_params({'zone_caps': 1, 'steer_gain': 1.2})                       # ほかの項目と一緒でも断る(何も変えない)
    assert d.params == before and (path.read_text() if path.exists() else None) == saved
    # ほかの項目・OFF のままの送り直しは今まで通り
    assert d.set_params({'steer_gain': 1.2})['steer_gain'] == 1.2
    assert d.set_params({'zone_caps': 0})['zone_caps'] == 0
    # 走っていなければ ON にできる → 走行中に ON のまま(1 → 1)の送り直し・ほかの項目は通る → OFF にはできる
    running[0] = False
    assert d.set_params({'zone_caps': 1})['zone_caps'] == 1
    running[0] = True
    assert d.set_params({'zone_caps': 1, 'steer_gain': 1.0})['zone_caps'] == 1
    assert d.set_params({'throttle_gain': 0.9})['zone_caps'] == 1
    assert d.set_params({'zone_caps': 0})['zone_caps'] == 0
    with pytest.raises(ValueError):
        d.set_params({'zone_caps': 0.8})                                        # 丸めて 1 になる値も断る
    assert d.params['zone_caps'] == 0


def test_server_maps_refusal_to_400():
    """server.py の /api/drive/params は ValueError を 400 にする(画面は理由を出し、スイッチを OFF に戻す)"""
    src = open(os.path.join(ROOT, 'jetformer', 'web', 'server.py'), encoding='utf-8').read()
    m = re.search(r"@app\.post\('/api/drive/params'\)\ndef api_drive_params.*?\n\n\n", src, re.S)
    assert m and 'driver.set_params(body)' in m.group(0) and 'except (ValueError, TypeError)' in m.group(0)
    assert 'HTTPException(400' in m.group(0)
    js = open(APP_JS, encoding='utf-8').read()
    on = re.search(r"\$\('#dr-zone-caps'\)\.addEventListener\('change'.*?^\}\);\n", js, re.S | re.M).group(0)
    assert 'catch (e) { toast(e.message, true); el.checked = !on; }' in on
