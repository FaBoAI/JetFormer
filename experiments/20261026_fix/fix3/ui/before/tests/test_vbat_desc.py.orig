"""電池の電圧を説明欄へ(2026-10-25、speed_control_20261025/spec.md 4.2 節、6.2 節 V-4)の試験。
web/bench.py の決まりと、web/server.py の走行の開始・終わり(偽の電圧の列・偽の driver.start)。
サーバーは起動しない(import するだけ。センサー・PCA9685 には触れない)。torch を読まない

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_vbat_desc.py
"""

import json
import os
import sys
import threading
import time

import pytest
from fastapi import HTTPException

from jetformer import config
from jetformer.sensors import vehicle as vs
from jetformer.sensors.base import Sample
from jetformer.web import bench
from jetformer.web import server as srv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import section_scorecard  # noqa: E402

B = bench.B_RUN
DESC = ('bench=real_bench_20261023 day=2026-11-03 block=1 cand=B model=%s battery=#2 V_before=__ V_after=__ check=__ '
        'official=1 memo=R1 で 2 回' % B)
parse = section_scorecard.parse_description


# ---------------------------------------------------------------- bench.py の決まり

def test_before_fills_empty_and_adds_source():
    out = bench.vbat_before_description(DESC, 8.3129)
    d = parse(out)
    assert d['V_before'] == '8.31' and d['V_src'] == 'board' and 'Vb_before' not in d
    assert d['memo'] == 'R1 で 2 回' and d['cand'] == 'B' and d['V_after'] is None and d['battery'] == '2'
    assert out.endswith('memo=R1 で 2 回')                                  # memo= の後ろを壊さない
    # 欄が無い説明欄(ベンチマークでない)にも足す
    d = parse(bench.vbat_before_description('AI走行 B', 8.2))
    assert d['V_before'] == '8.20' and d['V_src'] == 'board'
    for empty in ('V_before=', 'V_before=__', 'V_before=#__'):
        assert parse(bench.vbat_before_description('x %s memo=' % empty, 8.0))['V_before'] == '8.00'


def test_before_keeps_human_value():
    out = bench.vbat_before_description(DESC.replace('V_before=__', 'V_before=8.20'), 8.31)
    d = parse(out)
    assert d['V_before'] == '8.20' and d['Vb_before'] == '8.31' and 'V_src' not in d
    assert bench.vbat_before_description(DESC, None) == DESC                 # 値が無ければ今まで通り
    assert bench.vbat_before_description(DESC, float('nan')) == DESC


def test_after_fills_empty_only():
    start = bench.vbat_before_description(DESC, 8.31)
    out = bench.vbat_after_description(start, 8.0549, 7.9213)
    d = parse(out)
    assert d['V_before'] == '8.31' and d['V_after'] == '8.05' and d['Vb_min'] == '7.92' and d['V_src'] == 'board'
    assert 'Vb_after' not in d and d['memo'] == 'R1 で 2 回'
    assert bench.vbat_after_description(out, 8.0512, 7.0) == out              # 同じ基板の値の 2 回目は何も変えない(空の欄だけ)
    human = bench.vbat_after_description(DESC.replace('V_after=__', 'V_after=7.62'), 8.05, None)
    d = parse(human)
    assert d['V_after'] == '7.62' and d['Vb_after'] == '8.05' and 'Vb_min' not in d
    assert bench.vbat_after_description(DESC, None, None) == DESC


def test_human_memo_over_board_value_keeps_board_in_vb():
    """走り終わりのメモで、基板の値で埋めた V_x を人の違う値に替えたら、基板の値は Vb_x に残る(V_x が基板の値 = V_src=board かつ Vb_x 無し)"""
    desc = bench.vbat_after_description(bench.vbat_before_description(DESC, 8.31), 8.05, 7.92)
    same = parse(bench.after_description(desc, v_before='8.31', v_after='8.05', reason='240秒走った'))
    assert same['V_before'] == '8.31' and 'Vb_before' not in same and 'Vb_after' not in same
    d = parse(bench.after_description(desc, v_before='8.25', v_after='8.02', reason='240秒走った'))
    assert d['V_before'] == '8.25' and d['Vb_before'] == '8.31' and d['V_after'] == '8.02' and d['Vb_after'] == '8.05'
    assert d['Vb_min'] == '7.92' and d['memo'].startswith('240秒走った')
    # 人が始めに書いた V_before(Vb_before あり)は、今まで通り上書きするだけ
    d = parse(bench.after_description(bench.vbat_before_description(DESC.replace('V_before=__', 'V_before=8.2'), 8.31),
                                      v_before='8.22'))
    assert d['V_before'] == '8.22' and d['Vb_before'] == '8.31'


def _log(tmp_path, monkeypatch, desc=DESC, n=2):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    log = '20261103-101500'
    (tmp_path / log).mkdir()
    (tmp_path / log / 'log.json').write_text(json.dumps({'description': desc}), encoding='utf-8')
    for i in range(1, n + 1):
        sd = tmp_path / log / ('%s-%04d' % (log, i))
        sd.mkdir()
        (sd / 'scene.json').write_text(json.dumps({'description': desc, 'mode_id': 5}), encoding='utf-8')
    return log


def _scene_desc(tmp_path, log):
    return [json.loads(p.read_text(encoding='utf-8'))['description'] for p in sorted((tmp_path / log).glob('*/scene.json'))]


def test_write_after_all_scenes_and_nightly_reads_it(tmp_path, monkeypatch):
    start = bench.vbat_before_description(DESC, 8.31)
    log = _log(tmp_path, monkeypatch, start)
    assert bench.write_vbat_after(log, 8.05, 7.92) == 2
    assert all(parse(d)['V_after'] == '8.05' and parse(d)['Vb_min'] == '7.92' for d in _scene_desc(tmp_path, log))
    assert bench.write_vbat_after(log, 8.05, 7.0) == 0                      # もう埋まっている(同じ値・Vb_min は空の欄だけ)
    desc, info = section_scorecard.effective_description(log, start, tmp_path)
    d = parse(desc)
    assert d['V_before'] == '8.31' and d['V_after'] == '8.05' and not info['conflicts']   # 夜の集計が読む
    with pytest.raises(ValueError):
        bench.write_vbat_after('../x', 8.0)


def test_after_memo_and_board_writer_in_either_order(tmp_path, monkeypatch):
    log = _log(tmp_path, monkeypatch, bench.vbat_before_description(DESC, 8.31))
    bench.api_bench_after({'log': log, 'battery': '2', 'V_after': '7.62', 'reason': '240秒走った'})   # 人が先
    bench.write_vbat_after(log, 8.05, 7.92)
    for d in map(parse, _scene_desc(tmp_path, log)):
        assert d['V_after'] == '7.62' and d['Vb_after'] == '8.05' and d['Vb_min'] == '7.92'
    log2 = '20261103-102000'
    (tmp_path / log2).mkdir()
    (tmp_path / log2 / 'log.json').write_text(json.dumps({'description': DESC}), encoding='utf-8')
    sd = tmp_path / log2 / (log2 + '-0001')
    sd.mkdir()
    (sd / 'scene.json').write_text(json.dumps({'description': DESC, 'mode_id': 5}), encoding='utf-8')
    bench.write_vbat_after(log2, 8.05, 7.92)                                 # 基板が先
    bench.api_bench_after({'log': log2, 'V_after': '8.05', 'reason': '240秒走った'})          # 画面が前もって入れた値のまま
    d = parse(_scene_desc(tmp_path, log2)[0])
    assert d['V_after'] == '8.05' and 'Vb_after' not in d and d['V_src'] == 'board'


# ---------------------------------------------------------------- server.py の走行の開始・終わり

@pytest.fixture
def tracker(monkeypatch):
    tr = vs.VbatTracker()
    clk = [100.0]
    monkeypatch.setattr(srv, 'vbat', tr)
    monkeypatch.setattr(srv, '_now_s', lambda: clk[0])
    monkeypatch.setattr(srv, 'VBAT_RUN', {})
    tr.clk = clk
    return tr


def feed(tr, t0, t1, v, erpm=0, hz=100):
    for i in range(int(round((t1 - t0) * hz))):
        t = t0 + i / hz
        tr.add(t, v(t) if callable(v) else v, erpm(t) if callable(erpm) else erpm)


@pytest.fixture
def fake_start(monkeypatch, tmp_path):
    calls = []

    def start(run, mode_id, record=False, description='', tags=None, dry_run=False, params=None):
        calls.append({'run': run, 'description': description, 'dry_run': dry_run})
        return run, 'tok'
    monkeypatch.setattr(srv.driver, 'start', start)
    monkeypatch.setattr(bench, 'AI_NO_PATH', tmp_path / 'ai_run_no.json')
    monkeypatch.setattr(bench, 'BENCH_PATH', tmp_path / 'no_bench.json')
    monkeypatch.setattr(srv.trainer.__class__, 'running', property(lambda self: False))
    return calls


def _start(desc, dry=False):
    return srv.api_drive_start({'client': srv.DRIVE_CLIENT_VERSION, 'run': B, 'mode_id': 5, 'description': desc,
                                'dry_run': dry, 'record': True, 'params': {}})


def test_drive_start_writes_v_before_when_still(tracker, fake_start):
    feed(tracker, 97.0, 100.0, 8.312)
    r = _start(DESC)
    d = parse(fake_start[-1]['description'])
    assert d['V_before'] == '8.31' and d['V_src'] == 'board' and d['ai_no'] == '1' and d['memo'] == 'R1 で 2 回'
    assert r['v_before'] == 8.31 and srv.VBAT_RUN['v_before'] == 8.31 and srv.VBAT_RUN['state'] == 'starting'
    # 人の数は残す
    _start(DESC.replace('V_before=__', 'V_before=8.25'))
    d = parse(fake_start[-1]['description'])
    assert d['V_before'] == '8.25' and d['Vb_before'] == '8.31' and srv.VBAT_RUN['human_before'] == '8.25'


def test_drive_start_does_not_write_when_moving_or_no_voltage(tracker, fake_start):
    feed(tracker, 97.0, 100.0, 8.3, erpm=lambda t: 2500 if t > 98.5 else 0)       # 動いている
    _start(DESC)
    assert fake_start[-1]['description'].count('V_before=__') == 1 and 'V_src' not in fake_start[-1]['description']
    assert srv.VBAT_RUN['v_before'] is None and '止まっていません' in srv.VBAT_RUN['v_before_note']
    tr2 = vs.VbatTracker()
    srv.vbat = tr2
    feed(tr2, 97.0, 100.0, float('nan'))                                           # 古いファーム
    _start(DESC, dry=True)
    assert fake_start[-1]['description'] == DESC and srv.VBAT_RUN['v_before'] is None
    srv.vbat = tracker


def _bad_sample(t, raw, erpm=0):
    return Sample(int(t * 1e9), {'vbat_v': float('nan'), 'vbat_ok': False, 'vbat_raw_v': raw, 'vbat_bad': True,
                                 'erpm': erpm})


@pytest.mark.parametrize('raw', [0.5, 3.0, 12.75])
def test_broken_voltage_is_shown_but_never_written(tracker, fake_start, raw):
    """6.0〜9.0 V の外(分圧の外れ・張り付き): 画面に「電圧の値がおかしい」(broken_v)を出し、説明欄には書かない(fix.md。
    前は V_before=0.50 を書き、走行のゲインも上げた)"""
    for i in range(300):
        tracker.on_sample('VEHICLE', _bad_sample(97.0 + i / 100, raw))
    v = srv.vbat_view()
    assert v['supported'] is True and v['v'] is None and v['broken_v'] == pytest.approx(raw) and v['range_v'] == [6.0, 9.0]
    json.dumps(v, allow_nan=False)
    _start(DESC)
    assert fake_start[-1]['description'].count('V_before=__') == 1 and 'V_src' not in fake_start[-1]['description']
    assert srv.VBAT_RUN['v_before'] is None
    # 古い解読・偽物で範囲の外が vbat_ok のまま来ても、追跡は数えない(説明欄にも書かない)
    tr2 = vs.VbatTracker()
    srv.vbat = tr2
    feed(tr2, 97.0, 100.0, raw)
    _start(DESC)
    assert fake_start[-1]['description'].count('V_before=__') == 1 and srv.VBAT_RUN['v_before'] is None
    srv.vbat = tracker
    assert srv._vbat_ok_v(raw) is None and srv._vbat_ok_v(8.3) == 8.3 and srv._vbat_ok_v(None) is None


def test_after_run_waits_for_still_then_writes(tracker, tmp_path, monkeypatch):
    start = bench.vbat_before_description(DESC, 8.31)
    log = _log(tmp_path, monkeypatch, start)
    # 走行 100〜160 秒: 動いていて電圧が 8.3 → 7.95 に下がる(加速で 0.2 秒だけ 7.4)。止まってから 3 秒で止まる
    feed(tracker, 100.0, 160.0, lambda t: 8.3 - 0.35 * (t - 100) / 60 - (0.5 if int(t * 10) % 50 == 0 else 0), erpm=-6000)
    tracker.clk[0] = 160.0
    clk = tracker.clk

    def sleep(dt):                                   # 時計を進めながら車両の値を足す(160〜163 秒は惰性で動いている)
        t0 = clk[0]
        clk[0] += dt
        feed(tracker, t0, clk[0], 8.06, erpm=lambda t: -800 if t < 163.0 else 0)
    srv.VBAT_RUN.update(id=7)
    v_after, v_min, n = srv.vbat_after_run(log, 100.0, 160.0, run_id=7, sleep=sleep)
    assert v_after == pytest.approx(8.06) and 163.95 <= clk[0] <= 164.65              # 止まって 1 秒たったところ
    assert v_min == pytest.approx(8.3 - 0.35 * 59.5 / 60, abs=0.01) and n == 2        # 1 秒ごとの中央値の最小(下がりは除く)
    for d in map(parse, _scene_desc(tmp_path, log)):
        assert d['V_after'] == '8.06' and d['Vb_min'] == '7.95' and d['V_before'] == '8.31'
    assert srv.VBAT_RUN['state'] == 'done' and srv.VBAT_RUN['v_after'] == 8.06 and srv.VBAT_RUN['scenes_written'] == 2


def test_after_run_gives_up_after_wait_and_does_not_touch_next_run(tracker, tmp_path, monkeypatch):
    log = _log(tmp_path, monkeypatch)
    feed(tracker, 100.0, 120.0, 8.0, erpm=-5000)
    tracker.clk[0] = 120.0
    clk = tracker.clk

    def sleep(dt):
        t0 = clk[0]
        clk[0] += dt
        feed(tracker, t0, clk[0], 7.9, erpm=-3000)                                   # ずっと動いている(15 秒で諦める)
    srv.VBAT_RUN.update(id=8)
    v_after, v_min, n = srv.vbat_after_run(log, 100.0, 120.0, run_id=8, sleep=sleep)
    assert v_after is None and v_min == pytest.approx(8.0) and n == 2 and clk[0] >= 135.0
    d = parse(_scene_desc(tmp_path, log)[0])
    assert d['V_after'] is None and d['Vb_min'] == '8.00'                            # 最小だけ書く
    assert '止まっていません' in srv.VBAT_RUN['note']
    srv.VBAT_RUN.clear()
    srv.VBAT_RUN.update(id=9, state='starting')                                      # 次の走行が始まっていた
    srv.vbat_after_run(None, 100.0, 120.0, wait_s=0, run_id=8)
    assert srv.VBAT_RUN == {'id': 9, 'state': 'starting'}


def test_drive_events_start_thread_with_record_log(tracker, monkeypatch):
    got = []
    done = threading.Event()

    def fake_after(log, t0, t1, run_id=None, **kw):
        got.append((log, t0, t1, run_id))
        done.set()
    monkeypatch.setattr(srv, 'vbat_after_run', fake_after)
    monkeypatch.setattr(srv.driver, 'st', {'phase': 'idle', 'record_log': '20261103-101500'})
    srv.VBAT_RUN.update(id=3, state='starting')
    tracker.clk[0] = 200.0
    srv._vbat_on_drive('drive_start', dry_run=False)
    assert srv.VBAT_RUN['state'] == 'running' and srv.VBAT_RUN['t_start'] == 200.0 and srv.VBAT_RUN['id'] == 3
    tracker.clk[0] = 260.0
    srv._vbat_on_drive('drive_stop', reason='停止ボタン', error=False)
    assert done.wait(5)
    assert got == [('20261103-101500', 200.0, 260.0, 3)] and srv.VBAT_RUN['state'] == 'waiting'
    srv._vbat_on_drive('drive_start', dry_run=True)                                  # 画面を通らずに始めた走行
    assert srv.VBAT_RUN.get('id') is None and srv.VBAT_RUN['state'] == 'running'


def test_live_and_status_are_json_safe(tracker, monkeypatch):
    sensor = srv.hub['VEHICLE']
    monkeypatch.setattr(sensor, 'latest', Sample(0, {'pwm': [1441, 1496, 1980], 'erpm': 0, 'valid': [True] * 3,
                                                     'select': False, 'moving': False, 'quality': 'snapshot',
                                                     'vbat_v': float('nan'), 'vbat_ok': False}))
    out = srv.api_live_values()
    assert out['vbat']['supported'] is False and out['vbat']['v'] is None
    json.dumps(out, allow_nan=False)
    feed(tracker, 98.0, 100.0, 7.65)
    v = srv.vbat_view()
    assert v == dict(v, supported=True, v=7.65, level='bad', warn_v=7.90, bad_v=7.70)
    st = srv.api_drive_status('')
    assert st['vbat']['v'] == 7.65 and 'run' in st['vbat']
    json.dumps(st['vbat'], allow_nan=False)


def test_calibration_endpoint(tracker, monkeypatch, tmp_path):
    path = tmp_path / 'vbat_calib.json'
    monkeypatch.setattr(vs, 'VBAT_CALIB_PATH', path)
    sensor = srv.hub['VEHICLE']
    monkeypatch.setattr(sensor, 'vbat_scale', 1.0)
    with pytest.raises(HTTPException) as e:
        srv.api_vbat_calib({'meter_v': 8.31})
    assert e.value.status_code == 409                                                 # 止まっている値が無い
    feed(tracker, 97.0, 100.0, 8.21)
    r = srv.api_vbat_calib({'meter_v': 8.31})
    assert r['scale'] == pytest.approx(8.31 / 8.21, abs=1e-4) and sensor.vbat_scale == r['scale']
    assert json.loads(path.read_text())['scale'] == r['scale']
    for bad in ({'meter_v': 'x'}, {'meter_v': 20}, {'meter_v': 9.9}):              # 9.9 / 8.21 = 1.21 は範囲の外
        with pytest.raises(HTTPException) as e:
            srv.api_vbat_calib(bad)
        assert e.value.status_code == 400
    assert srv.api_vbat_calib({'reset': True})['scale'] == 1.0 and sensor.vbat_scale == 1.0
    monkeypatch.setattr(srv.driver.__class__, 'running', property(lambda self: True))
    with pytest.raises(HTTPException) as e:
        srv.api_vbat_calib({'reset': True})
    assert e.value.status_code == 409


def test_server_wiring_is_light():
    src = open(os.path.join(os.path.dirname(srv.__file__), 'server.py'), encoding='utf-8').read()
    assert "hub['VEHICLE'].subscribe(vbat.on_sample)" in src and 'driver.listeners.append(_vbat_on_drive)' in src
    assert 'DRIVE_CLIENT_VERSION = 6' in src                                          # 画面の版は変えない
    assert time.monotonic() > 0
