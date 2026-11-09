"""起動して最初のモデルの読み込みの音声の案内・起動時の値の上書き・一覧を優先のモデルだけにする(2026-11-05 ユーザーの依頼)。
実際の再生はしない。時計は偽物で進める"""

import json
import threading
import time

import pytest

from jetformer import drive, sound


class FakePlayer(object):
    def __init__(self):
        self.said = []

    def notify(self, event, text='', chime='info', urgent=False):
        self.said.append((event, text, chime))
        return True


class Clock(object):
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def wait_for(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.002)
    return False


def texts(p):
    return [x[1] for x in p.said]


def make(alive=lambda: True, count=True):
    p, c = FakePlayer(), Clock()
    return p, c, sound.LoadCountdown(p, alive, poll_s=0.001, now=c, count=count)


def test_counts_every_10_seconds_then_ready():
    p, c, v = make()
    v.begin()
    assert wait_for(lambda: len(p.said) == 1)
    assert p.said[0] == ('loading', sound.LOAD_FIRST_TEXT, 'info')
    for n, left in enumerate((50, 40, 30, 20, 10), 2):
        c.t = 60 - left - 0.5
        time.sleep(0.01)
        assert len(p.said) == n - 1              # その秒になるまで言わない
        c.t = 60 - left
        assert wait_for(lambda: len(p.said) == n)
        assert p.said[-1] == ('loading', 'あと%d秒' % left, None)
    assert v.ready() is True
    assert p.said[-1] == ('loading', sound.LOAD_READY_TEXT, 'info')
    assert texts(p) == [sound.LOAD_FIRST_TEXT, 'あと50秒', 'あと40秒', 'あと30秒', 'あと20秒', 'あと10秒', sound.LOAD_READY_TEXT]


def test_early_ready_stops_the_count():
    p, c, v = make()
    v.begin()
    assert wait_for(lambda: len(p.said) == 1)
    c.t = 10
    assert wait_for(lambda: len(p.said) == 2)
    assert v.ready() is True
    c.t = 100
    time.sleep(0.02)
    assert texts(p) == [sound.LOAD_FIRST_TEXT, 'あと50秒', sound.LOAD_READY_TEXT]


def test_longer_than_60_seconds_waits_silently():
    p, c, v = make()
    v.begin()
    assert wait_for(lambda: len(p.said) == 1)
    for left in (50, 40, 30, 20, 10):
        c.t = 60 - left
        assert wait_for(lambda: p.said[-1][1] == 'あと%d秒' % left)
    c.t = 200
    time.sleep(0.02)
    assert p.said[-1][1] == 'あと10秒'
    assert v.ready() is True
    assert p.said[-1][1] == sound.LOAD_READY_TEXT


def test_run_failed_says_nothing_more():
    alive = [True]
    p, c, v = make(lambda: alive[0])
    v.begin()
    assert wait_for(lambda: len(p.said) == 1)
    alive[0] = False
    assert wait_for(lambda: v._done is None)
    c.t = 100
    time.sleep(0.02)
    assert texts(p) == [sound.LOAD_FIRST_TEXT]
    assert v.ready() is False                    # 次の走行の drive_start で「準備ができました」と言わない
    assert texts(p) == [sound.LOAD_FIRST_TEXT]


def test_ready_without_begin_does_nothing():
    p, c, v = make()
    assert v.ready() is False
    assert p.said == []


def test_second_ready_says_once():
    p, c, v = make()
    v.begin()
    assert wait_for(lambda: len(p.said) == 1)
    assert v.ready() is True
    assert v.ready() is False
    assert texts(p).count(sound.LOAD_READY_TEXT) == 1


def test_texts_are_prepared_and_event_is_listed():
    for t in [sound.LOAD_FIRST_TEXT, sound.LOAD_READY_TEXT] + ['あと%d秒' % s for s in (50, 40, 30, 20, 10)]:
        assert t in sound.COMMON_TEXTS
    assert sound.EVENTS['loading'][1] is True
    assert sound.DEFAULTS['events']['loading'] is True


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    monkeypatch.setattr(drive, 'RECOMMENDED_PATH', tmp_path / 'drive_recommended.json')
    monkeypatch.setattr(drive, 'BOOT_PARAMS_IN_TESTS', True)
    return tmp_path


def test_real_boot_file_is_not_read_under_pytest(tmp_path, monkeypatch):
    """server.py を読み込む試験の driver が本物の設定(前送り ON など)で動いて、ff_wait_v0 が本物のセンサーを開始しないように"""
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    (tmp_path / drive.BOOT_PARAMS_NAME).write_text(json.dumps({'speed_ff': 1, 'launch_boost': 1}), encoding='utf-8')
    assert drive.boot_params() == {}
    from jetformer.web import server
    assert server.driver.params['speed_ff'] == 0 and server.driver.params['launch_boost'] == 0


def test_boot_params_absent_is_empty(paths):
    assert drive.boot_params() == {}
    assert drive.DEFAULT_PARAMS['launch_boost'] == 0     # 既定の表は変えない


@pytest.mark.parametrize('doc,want', [({'launch_boost': 1}, {'launch_boost': 1}), ({'launch_boost': 0}, {'launch_boost': 0}),
                                      ({'launch_boost': 2}, {}), ({'launch_boost': True}, {}), ({'launch_boost': '1'}, {}),
                                      ({'recovery': 0, 'speed_ceiling': 1, 'lookahead': 5}, {}), ([1], {}),
                                      ({'speed_ff': 1, 'ff_vref_v': 8.1, 'steer_gain': 1.0, 'throttle_gain': 0.9},
                                       {'speed_ff': 1, 'ff_vref_v': 8.1, 'steer_gain': 1.0, 'throttle_gain': 0.9}),
                                      ({'throttle_gain': 2.5, 'steer_gain': -1, 'ff_vref_v': 9.5}, {}),
                                      ({'throttle_gain': '0.9', 'steer_gain': True, 'speed_ff': 2}, {}),
                                      ({'rec_back_guard': 1, 'rec_hold_keep': 1}, {'rec_back_guard': 1, 'rec_hold_keep': 1}),
                                      ({'rec_back_guard': 1}, {'rec_back_guard': 1}),
                                      ({'rec_hold_keep': 1}, {}), ({'rec_back_guard': 0, 'rec_hold_keep': 1}, {'rec_back_guard': 0})])
def test_boot_params_only_allowed_values(paths, doc, want):
    (paths / drive.BOOT_PARAMS_NAME).write_text(json.dumps(doc), encoding='utf-8')
    assert drive.boot_params() == want


def test_boot_params_broken_file(paths):
    (paths / drive.BOOT_PARAMS_NAME).write_text('{broken', encoding='utf-8')
    assert drive.boot_params() == {}


def test_driver_starts_with_launch_on_when_file_says_so(paths):
    (paths / drive.BOOT_PARAMS_NAME).write_text(json.dumps({'launch_boost': 1}), encoding='utf-8')
    (paths / 'drive_settings.json').write_text(json.dumps({'params': {'launch_boost': 0, 'rec_back_guard': 1}}), encoding='utf-8')
    d = drive.Driver({}, None)
    assert d.params['launch_boost'] == 1
    assert d.params['rec_back_guard'] == 0               # ほかの項目は今まで通り起動で既定へ
    assert d.engine_loaded is False


def test_driver_boot_gains_and_ff_override_saved_values(paths):
    (paths / drive.BOOT_PARAMS_NAME).write_text(json.dumps({'speed_ff': 1, 'ff_vref_v': 8.1, 'steer_gain': 1.0, 'throttle_gain': 0.9}), encoding='utf-8')
    (paths / 'drive_settings.json').write_text(json.dumps({'params': {'steer_gain': 1.05, 'throttle_gain': 0.95, 'speed_ff': 0}}), encoding='utf-8')
    d = drive.Driver({}, None)
    assert (d.params['steer_gain'], d.params['throttle_gain'], d.params['speed_ff'], d.params['ff_vref_v']) == (1.0, 0.9, 1, 8.1)
    p = d._ff_rules(d._opt_rules(d._launch_rules(d._speed_ctrl_rules(d._clean_params({}), False), False), False), False, {})
    assert (p['speed_ff'], p['steer_gain'], p['throttle_gain']) == (1, 1.0, 0.9)    # 走行開始と同じ決まりを通る
    with pytest.raises(Exception):                       # 前送り ON で走る前の電圧が無ければ、今まで通り断る
        d._ff_start(p, None, '電圧がありません')


def test_driver_boot_guard_and_keep_pass_the_start_rules(paths):
    (paths / drive.BOOT_PARAMS_NAME).write_text(json.dumps({'launch_boost': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1}), encoding='utf-8')
    d = drive.Driver({}, None)
    assert (d.params['rec_back_guard'], d.params['rec_hold_keep'], d.params['recovery']) == (1, 1, 1)
    p = d._opt_rules(d._launch_rules(d._clean_params({'launch_boost': 1, 'recovery': 1}), False), False)     # 走行開始と同じ決まり
    assert (p['launch_boost'], p['rec_back_guard'], p['rec_hold_keep']) == (1, 1, 1)


def test_driver_without_file_keeps_launch_off(paths):
    (paths / 'drive_settings.json').write_text(json.dumps({'params': {'launch_boost': 1}}), encoding='utf-8')
    assert drive.Driver({}, None).params['launch_boost'] == 0


def models(monkeypatch, names):
    monkeypatch.setattr(drive, 'RUNS_DIR', _FakeDir(names))
    monkeypatch.setattr(drive, 'model_info', lambda n: {'name': n, 'created_at': n, 'drivable': True})


class _Entry(object):
    def __init__(self, name):
        self.name = name


class _FakeDir(object):
    def __init__(self, names):
        self.names = names

    def exists(self):
        return True

    def iterdir(self):
        return [_Entry(n) for n in self.names]


def write_priority(paths, runs, **extra):
    doc = dict({'title': 't', 'models': [{'run': r, 'label': r} for r in runs]}, **extra)
    (paths / drive.PRIORITY_NAME).write_text(json.dumps(doc), encoding='utf-8')


def test_only_true_lists_priority_models_only(paths, monkeypatch):
    models(monkeypatch, ['a', 'b', 'c', 'd'])
    write_priority(paths, ['c', 'a'], only=True)
    assert [m['name'] for m in drive.list_models()] == ['c', 'a']


def test_without_only_lists_everything(paths, monkeypatch):
    models(monkeypatch, ['a', 'b', 'c', 'd'])
    write_priority(paths, ['c', 'a'])
    assert [m['name'] for m in drive.list_models()] == ['c', 'a', 'd', 'b']


def test_only_with_no_existing_priority_model_lists_everything(paths, monkeypatch):
    models(monkeypatch, ['a', 'b'])
    write_priority(paths, ['gone'], only=True)
    assert sorted(m['name'] for m in drive.list_models()) == ['a', 'b']


class _Tracker(object):
    def __init__(self, answers):
        self.answers = list(answers)
        self.n = 0

    def still_check(self, now, span_s=2.0, min_n=20, fresh_s=0.5, q=0.5):
        self.n += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class _Hub(object):
    def __init__(self, running):
        self.running = running
        self.started = 0

    def start(self):
        self.started += 1
        self.running = True


class _Drv(object):
    def __init__(self, hub):
        self.hub = hub
        self.running = False


@pytest.fixture
def srv(monkeypatch):
    from jetformer.web import server
    server.START_CANCEL.clear()                  # 前の試験の ■ の印を残さない(2026-11-06)
    server.START_PENDING[0] = 0
    yield server
    server.START_CANCEL.clear()
    server.START_PENDING[0] = 0


def _ff_setup(monkeypatch, srv, answers, running):
    hub = _Hub(running)
    monkeypatch.setattr(srv, 'vbat', _Tracker(answers))
    monkeypatch.setattr(srv, 'driver', _Drv(hub))
    return hub


def test_ff_wait_starts_sensors_and_waits_for_voltage(monkeypatch, srv):
    hub = _ff_setup(monkeypatch, srv, [(None, '車両の値が来ていません'), (None, '車両の値が 2.0 秒ぶんありません'), (8.3, '')], False)
    assert srv.ff_wait_v0(wait_s=2.0, poll_s=0.001) is True
    assert hub.started == 1 and srv.vbat.n == 3


def test_ff_wait_does_nothing_when_voltage_is_there(monkeypatch, srv):
    hub = _ff_setup(monkeypatch, srv, [(8.3, '')], True)
    assert srv.ff_wait_v0(wait_s=2.0, poll_s=0.001) is False
    assert hub.started == 0


def test_ff_wait_does_not_wait_when_the_car_is_moving(monkeypatch, srv):
    hub = _ff_setup(monkeypatch, srv, [(None, '車が止まっていません(|eRPM| の中央 900)')], True)
    assert srv.ff_wait_v0(wait_s=2.0, poll_s=0.001) is False
    assert hub.started == 0 and srv.vbat.n == 1


def test_ff_wait_gives_up_after_the_limit(monkeypatch, srv):
    hub = _ff_setup(monkeypatch, srv, [(None, '基板の電圧がありません')], False)
    t0 = time.monotonic()
    assert srv.ff_wait_v0(wait_s=0.05, poll_s=0.001) is True
    assert 0.04 < time.monotonic() - t0 < 2.0 and hub.started == 1


# ---------------------------------------------------------------- 2026-11-06 の直し
# 1. 試験の中では車のスピーカーを鳴らさない 2. すぐに失敗した走行では最初の一言も言わない
# 3. 車が止まっていないときの声 4. ▶ の返事を待つ間(電圧を待つ数秒)の ■ で走行を始めない
import inspect
import types

from fastapi import HTTPException


def test_server_player_does_not_play_under_pytest(monkeypatch, srv):
    def boom(*a, **k):
        raise AssertionError('aplay や amixer を呼んだ')
    monkeypatch.setattr(sound.subprocess, 'run', boom)
    srv.player._play('/nonexistent.wav')
    assert srv.player._play != srv.player._aplay


def test_run_failed_at_once_says_nothing():
    p, c, v = make(lambda: False)
    v.begin()
    assert wait_for(lambda: v._done is None)
    c.t = 100
    time.sleep(0.02)
    assert p.said == []
    assert v.ready() is False and p.said == []


def test_ready_at_once_says_only_ready():
    p, c = FakePlayer(), Clock()
    v = sound.LoadCountdown(p, lambda: True, poll_s=0.3, now=c, count=True)
    v.begin()
    assert v.ready() is True
    time.sleep(0.4)
    assert texts(p) == [sound.LOAD_READY_TEXT]


def test_ff_refusal_text_says_moving_when_the_car_moves():
    from jetformer.sensors import vehicle
    assert "'車が止まっていません(" in inspect.getsource(vehicle)      # still_check の理由の文
    assert sound.ff_refusal_text('車が止まっていません(|eRPM| の中央 900)') == sound.FF_MOVING_TEXT
    for note in ('車両の値が来ていません', '基板の電圧がありません', '基板の電圧の値がおかしい(3.10 V。5.0〜13.0 V の外)', '', None):
        assert sound.ff_refusal_text(note) == sound.FF_NO_V0_TEXT
    assert sound.FF_MOVING_TEXT in sound.COMMON_TEXTS


class _StartDrv(_Drv):
    def __init__(self):
        super(_StartDrv, self).__init__(_Hub(True))
        self.engine_loaded = True
        self.params = {}
        self.st = {}
        self.starts, self.stops = [], []

    def start(self, run, mode_id, **kw):
        self.starts.append(kw)
        self.running = True
        return run, 'tok'

    def stop(self, reason='停止ボタン'):
        if not self.running:
            raise RuntimeError('走行していません')
        self.stops.append(reason)


def _start_setup(monkeypatch, srv, wait, answers=((8.3, ''),)):
    drv = _StartDrv()
    monkeypatch.setattr(srv, 'driver', drv)
    monkeypatch.setattr(srv, 'vbat', _Tracker(list(answers)))
    monkeypatch.setattr(srv, 'player', FakePlayer())
    monkeypatch.setattr(srv, 'VBAT_RUN', {})
    monkeypatch.setattr(srv, '_client_version', lambda body: srv.DRIVE_CLIENT_VERSION)
    monkeypatch.setattr(srv, '_page_build', lambda body: 0)
    monkeypatch.setattr(srv, '_boot_refusal', lambda body, boot: None)
    monkeypatch.setattr(srv, 'trainer', types.SimpleNamespace(running=False))
    monkeypatch.setattr(srv, '_ff_requested', lambda body: True)
    monkeypatch.setattr(srv, 'ff_wait_v0', wait)
    return drv


def test_stop_while_waiting_for_voltage_cancels_the_start(monkeypatch, srv):
    got = {}

    def wait():
        got['stop'] = srv.api_drive_stop({})        # 電圧を待つ間に ■
        return True
    drv = _start_setup(monkeypatch, srv, wait)
    with pytest.raises(HTTPException) as ei:
        srv.api_drive_start({'dry_run': True, 'run': 'm'})
    assert ei.value.status_code == 409 and ei.value.detail == srv.START_CANCEL_TEXT
    assert got['stop'] == {'ok': True, 'cancelled': True}
    assert drv.starts == [] and srv.START_PENDING[0] == 0
    assert srv.player.said == []                    # 「電池の電圧を読めないので…」も言わない


def test_earlier_stop_does_not_block_the_next_start(monkeypatch, srv):
    drv = _start_setup(monkeypatch, srv, lambda: True)
    srv.START_CANCEL.set()                          # 前の走行の ■
    r = srv.api_drive_start({'dry_run': True, 'run': 'm'})
    assert r['ok'] is True and len(drv.starts) == 1 and drv.starts[0]['ff_v0'] == 8.3
    assert drv.stops == [] and srv.START_PENDING[0] == 0


def test_stop_without_a_waiting_start_is_409_as_before(monkeypatch, srv):
    monkeypatch.setattr(srv, 'driver', _StartDrv())
    with pytest.raises(HTTPException) as ei:
        srv.api_drive_stop({})
    assert ei.value.status_code == 409 and ei.value.detail == '走行していません'


def test_stop_during_a_run_still_stops(monkeypatch, srv):
    drv = _StartDrv()
    drv.running = True
    monkeypatch.setattr(srv, 'driver', drv)
    assert srv.api_drive_stop({'reason': 'キー操作'}) == {'ok': True}
    assert drv.stops == ['キー操作']


def test_stop_just_before_driver_start_stops_the_run(monkeypatch, srv):
    drv = _start_setup(monkeypatch, srv, lambda: True)
    real = drv.start

    def start(run, mode_id, **kw):
        srv.START_CANCEL.set()                      # ■ が driver.start の直前に届いた
        return real(run, mode_id, **kw)
    drv.start = start
    srv.api_drive_start({'dry_run': True, 'run': 'm'})
    assert len(drv.starts) == 1 and drv.stops == ['停止ボタン']


def test_ff_wait_stops_at_once_when_stop_is_pressed(monkeypatch, srv):
    hub = _ff_setup(monkeypatch, srv, [(None, '車両の値が来ていません')], False)
    srv.START_CANCEL.set()
    t0 = time.monotonic()
    assert srv.ff_wait_v0(wait_s=5.0, poll_s=0.001) is True
    assert time.monotonic() - t0 < 1.0 and hub.started == 1


def test_moving_car_says_the_moving_voice(monkeypatch, srv):
    drv = _start_setup(monkeypatch, srv, lambda: False, answers=((None, '車が止まっていません(|eRPM| の中央 900)'),))

    def start(run, mode_id, **kw):
        assert kw['ff_v0'] is None
        raise RuntimeError('電圧の前送りが ON なのに、走る前の電圧がありません')
    drv.start = start
    with pytest.raises(HTTPException) as ei:
        srv.api_drive_start({'dry_run': True, 'run': 'm'})
    assert ei.value.status_code == 409
    assert srv.player.said == [('warning', sound.FF_MOVING_TEXT, 'warn')]


# ---------------------------------------------------------------- 2026-11-06 ユーザー「初回の実行には30秒ほどかかります、だけにしてください」
def test_default_says_only_the_first_line():
    assert sound.LOAD_FIRST_TEXT == '初回の実行には30秒ほどかかります' and sound.LOAD_COUNT is False
    p, c, v = make(count=None)
    v.begin()
    assert wait_for(lambda: v._done is None)
    for t in (10, 20, 30, 40, 50, 60, 100):
        c.t = t
        time.sleep(0.005)
    assert v.ready() is False                       # 「準備ができました」も言わない
    assert p.said == [('loading', sound.LOAD_FIRST_TEXT, 'info')]


def test_default_ready_at_once_says_nothing():
    p, c = FakePlayer(), Clock()
    v = sound.LoadCountdown(p, lambda: True, poll_s=0.3, now=c)
    v.begin()
    assert v.ready() is True
    time.sleep(0.4)
    assert p.said == []


def test_default_failed_at_once_says_nothing():
    p, c, v = make(lambda: False, count=None)
    v.begin()
    assert wait_for(lambda: v._done is None)
    time.sleep(0.02)
    assert p.said == []


def test_server_uses_the_first_line_only(srv):
    assert srv.load_voice.count is False
