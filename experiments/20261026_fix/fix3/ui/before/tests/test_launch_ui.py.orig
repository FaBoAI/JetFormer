"""発進の前進の走行タブのスイッチ(index.html・app.js)と、基準の設定・ブロックで ON にしないこと(web/bench.py)。2026-10-26
画面の計算は node で確かめる(node が無ければ飛ばす)。サーバーは起動しない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_launch_ui.py
"""
import json
import re
import subprocess

import pytest

from jetformer import drive
from jetformer.web import bench

from test_vbat_ui import APP_JS, KEEP, NODE, _node, _src


def test_static_switch_default_off_and_warning():
    html = _src('index.html')
    for el in ('dr-launch-panel', 'dr-launch-boost', 'dr-launch-spec', 'dr-launch-note'):
        assert 'id="%s"' % el in html, el
    tag = re.search(r'<input type="checkbox" id="dr-launch-boost"[^>]*>', html).group(0)
    assert 'checked' not in tag and 'autocomplete="off"' in tag                  # 既定 OFF・再読み込みで前の ON を戻さない
    for words in ('試験の選択肢', '前が 1 m 以上空いている', 'ユーザーの了承', '車輪を浮かせた', '既定は OFF',
                  '起動するたびに OFF', '走行中は ON にできません', 'ToF ガードが OFF のときは始めません'):
        assert words in html, words
    assert html.index('id="dr-launch-boost"') < html.index('id="dr-start-kick"')   # 発進の補助のすぐ上
    js = _src('app.js')
    assert "p.launch_boost = $('#dr-launch-boost').checked ? 1 : 0" in js
    assert 'renderLaunchSwitch(st);' in js
    if NODE:
        r = subprocess.run([NODE, '--check', APP_JS], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_bench_values_never_turn_launch_on():
    base = bench.fallback_baseline()
    on = dict(base, params=dict(base['params'], launch_boost=1, launch_us=200.0))
    a, b, c, fb = _node(['BENCH_FALLBACK', 'benchFieldValues'],
                        'process.stdout.write(JSON.stringify([benchFieldValues(%s, ""), benchFieldValues(%s, ""), '
                        'benchFieldValues({params: {}}, null), BENCH_FALLBACK]));' % (json.dumps(on), json.dumps(base)))
    assert a['params']['launch_boost'] == 0 and a['params']['launch_us'] == 200.0      # 値はそのまま・スイッチは OFF
    assert any('発進の前進' in n for n in a['notes'])
    assert b['params'] == {k: v for k, v in base['params'].items() if k not in KEEP} and b['notes'] == []   # トリム・後ろの見張りは入れない(2026-10-26 fix2)
    assert 'launch_boost' not in c['params']                                           # 無い項目は足さない(古いサーバー)
    for k in drive.LAUNCH_KEYS:
        assert fb['params'][k] == drive.DEFAULT_PARAMS[k], k                           # 控えの値 = サーバーの既定


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_confirm_text_takes_values_from_launch_spec():
    params = dict(drive.DEFAULT_PARAMS, launch_us=200.0, launch_mps=2.2, launch_max_s=1.0)
    st = {'params': params, 'launch_spec': drive.launch_spec(params)}
    got = _node(['launchSupported', 'LAUNCH_FALLBACK', 'launchSpec', 'launchSpecText', 'launchConfirmText'],
                'const st = %s; const sp = launchSpec(st);\n'
                'process.stdout.write(JSON.stringify({sp, c: launchConfirmText(sp), ok: launchSupported(st), '
                'old: launchSupported({params: {steer_gain: 1}}), fb: launchSpec({params: {}})}));' % json.dumps(st))
    sp, c = got['sp'], got['c']
    assert (sp['us'], sp['mps'], sp['max_s']) == (200.0, 2.2, 1.0)
    assert '前進 200 µs 以上・2.2 m/s まで・最大 1.0 秒' in c and st['launch_spec']['note'] in c
    assert 'ユーザーの了承は出ていますか' in c and '走行中は ON にできません' in c and 'ToF ガードが OFF' in c
    assert got['ok'] is True and got['old'] is False
    assert (got['fb']['us'], got['fb']['mps'], got['fb']['max_s']) == (250, 2.5, 1.2)


def test_bench_block_cannot_turn_launch_on():
    warnings = []
    ov, _ = bench._block_overrides({'params': {'launch_boost': 1, 'launch_us': 200, 'throttle_gain': 0.9}}, warnings, 'B1')
    assert 'launch_boost' not in ov['params'] and ov['params']['launch_us'] == 200.0
    assert any('発進の前進はスイッチで入れてください' in w for w in warnings)
    assert 'launch_boost' in bench.SWITCH_KEYS and 'launch_boost' not in bench.PLAN_KEYS
    fb = bench.fallback_baseline()
    assert fb['params']['launch_boost'] == 0 and set(drive.LAUNCH_KEYS) <= set(fb['params'])
    w2 = []
    base = bench._baseline({'params': {'launch_boost': 1}}, w2)
    assert any('launch_boost が 1' in w for w in w2) and base['params']['launch_boost'] == 1   # 照らし合わせ用。画面は ON にしない


def test_launch_spec_values_come_from_code():
    sp = drive.launch_spec()
    from jetformer import assist as A
    assert sp['us'] == 250.0 and sp['mps'] == 2.5 and sp['max_s'] == 1.2
    assert sp['ttc_start_s'] == A.LaunchBoost.TTC_START_S and sp['end_yaw_dps'] == A.LaunchBoost.END_YAW_DPS
    assert sp['limits']['launch_us'] == [150.0, 300.0]
    assert '1.0 m 以上空いている' in sp['note'] and '走行中は ON にできません' in sp['note']
