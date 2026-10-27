"""中立止めでリカバリーを残す(rec_hold_keep)・発進の前進が出さない区間の発進の補助(launch_no_kick)。2026-10-26 fix3 safety。

指摘(jetformer_reports/fix_20261026/fix2/critic.md・refute1_safety.md):
  safety1-1(major): drive の hold() の最後の helper.reset() が Recovery を全部戻す。中立止め 1 回(1 フレームでもよい)で
    リカバリー中止が解けてモデルの弱い前進が壁を押し、途中のリカバリーは取り消され、armed も消える(止まった車では二度と始まらない)。
  safety1-2(minor): 発進の前進が「前が近い・ToF が無い」で出さないと決めた区間でも、StartKick が前進 80 µs を最大 2 秒出す。
直し(どちらも既定 0・起動で 0・0 のとき手を入れる前とビット単位で同じ。写し tests/fixtures/fix3_before/ と比べる):
  rec_hold_keep = 1: AI 側のままの中立止めでは Recovery を戻さない(戻すのは CH3 を人側・人側の中立止め・走行の開始)。
  launch_no_kick = 1: launch_boost = 1 で「出さない」と決めた区間では StartKick も出さない(launch_boost = 0 では何もしない)。
実機・GPU・torch は使わない(drive の走行ループは tests/test_launch_drive.py の偽物で回す。画面は node の vm)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_rec_hold_keep.py
"""
import collections
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A
from jetformer import drive
from jetformer.web import bench

from test_drive_caps import assert_rows_equal_before, run_det, scenario
from test_launch_drive import C, N, PARAMS, build, launch_scene, run, tof_near

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix3_before'
BEFORE_SHA = {'drive.py': '216efc09981394f7213d846a065c55ca0b4e4c2f722a0a71980e15dbb44dad56',
              'assist.py': '6ee550c0b8e09881de3b4231254ec5a247bf209c0f2bdc454381ae2848fc9cfb',
              'app.js': '0b0057bd551d3192a7741cefd50676c8010ac784460500d1fd56ffd37cc183c7'}
NEW = ('rec_hold_keep', 'launch_no_kick')
DT = 1 / 30.0
REC_1010 = ROOT / 'data' / 'raw' / '20261010-181504' / '20261010-181504-0004'
REC_1023 = ROOT / 'data' / 'raw' / '20261023-205458' / '20261023-205458-0002'
NODE = shutil.which('node')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_before():
    """直す前の assist.py・drive.py(写し)をモジュールにする(写しの drive は写しの assist で動かす)"""
    for name, h in BEFORE_SHA.items():
        assert sha(FIX / name) == h, name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix3', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix3', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    assert not set(NEW) & set(mod.PARAM_LIMITS) and not hasattr(amod.LaunchBoost(1496.0), 'skipped')
    return amod, mod


def only_before(mod, p):
    return {k: v for k, v in p.items() if k in mod.DEFAULT_PARAMS}


def same_loop(a, b):
    """2 つの走行ループの結果 (driver, pca, rec) が、PCA9685 に書く値・記録の行・出来事で同じ"""
    (_, pca0, rec0), (_, pca1, rec1) = a, b
    assert pca0.pairs == pca1.pairs and pca0.single == pca1.single
    assert_rows_equal_before(rec0.rows, rec1.rows)
    assert rec0.events == rec1.events


# ---------------------------------------------------------------- 場面
def stuck_scene(dur, holds=(), human=()):
    """人側 1 秒 → AI 側。1.0〜2.0 秒は前へ 2 m/s(armed になる)、2.0 秒からは壁に当たって止まったまま(下がらない)。
    モデルはずっと弱い前進(中立 − 120)。holds = [(始まり, 長さ)]: その間は車両の値が来ない(0.15 秒を超えると中立止め。
    最後に届いた値は AI 側)。human = [(始まり, 長さ)]: その間は CH3 を人側(反証 1 回目 safety1 の _stuck_scene と同じ形)"""
    def sel(s):
        return s >= 1.0 and not any(a <= s < a + n for a, n in human)
    return build(dur, sel=sel, v=lambda s: 2.0 if 1.0 <= s < 2.0 else 0.0, pred=(C, N - 120.0),
                 veh_skip=lambda s: any(a <= s < a + n for a, n in holds))


def timeline(frames, rec):
    s_of = {f['cam_t']: f['s'] for f in frames}
    return [(round(s_of[t], 3), int(r['assist']), bool(r['hold']), float(r['out'][1]), bool(r['select'])) for t, r in rec.rows]


def near_scene(d_mm, dur=4.0):
    """人側 1 秒(止まっている)→ AI 側。車は止まったまま、前の真ん中 d_mm に立った物。モデルは中立(反証 1 回目 safety1 の 2)"""
    return build(dur, sel=lambda s: s >= 1.0, v=0.0, pred=(C, N), tof=tof_near(d_mm))


def recovery_scene(back_from=None, hit_at=None, holds=()):
    """走る(3 m/s)→ 2.0 秒で止まる(モデルは中立の帯)→ リカバリー。back_from 秒から後ろへ 1 m/s(開いたループ)。
    hit_at 秒に後ろの衝撃(IMU の横 6G)。holds の間は車両の値が来ない(tests/test_recovery_back_guard.py の場面 + 中立止め)"""
    def v(s):
        if s < 2.0:
            return 3.0
        if back_from is not None and s >= back_from and (hit_at is None or s < hit_at):
            return -1.0
        return 0.0
    return build(5.5, sel=True, v=v, pred=lambda s: (C, N - 150.0) if s < 1.8 else (C, N + 20.0),
                 ax=lambda s: 60.0 if (hit_at is not None and hit_at <= s < hit_at + 0.02) else 0.0,
                 veh_skip=lambda s: any(a <= s < a + n for a, n in holds))


def fuzz_scene(seed, dur=8.0):
    """反証 2 回目の safety2/fuzz_loop.py と同じでたらめな場面(人側・止まる・走る・後ろへ・衝撃・前の物・車両の値が来ない)"""
    import sys
    sys.path.insert(0, str(ROOT / 'experiments' / '20261026_fix' / 'fix2' / 'safety2'))
    import fuzz_loop
    return fuzz_loop.scene(random.Random(seed), dur)


# ================================================================ 1. 既定・起動・範囲
def test_before_copies():
    assert {k: v[:12] for k, v in BEFORE_SHA.items()} == {'drive.py': '216efc099813', 'assist.py': '6ee550c0b8e0',
                                                          'app.js': '0b0057bd551d'}
    load_before()


def test_params_default_off_and_off_on_boot(monkeypatch, tmp_path):
    _, before = load_before()
    for k in NEW:
        assert drive.DEFAULT_PARAMS[k] == 0 and drive.PARAM_LIMITS[k] == (0, 1) and k in drive.RESET_ON_BOOT
    assert {k: v for k, v in drive.DEFAULT_PARAMS.items() if k not in NEW} == before.DEFAULT_PARAMS   # ほかの既定値は同じ
    assert {k: v for k, v in drive.PARAM_LIMITS.items() if k not in NEW} == before.PARAM_LIMITS
    assert set(drive.RESET_ON_BOOT) - set(before.RESET_ON_BOOT) == set(NEW)
    assert drive.LAUNCH_KEYS == before.LAUNCH_KEYS                      # 発進の前進の 4 つは変えていない
    path = tmp_path / 'drive_settings.json'
    path.write_text(json.dumps({'params': {'rec_hold_keep': 1, 'launch_no_kick': 1, 'steer_trim_us': 15, 'start_kick': 1}}),
                    encoding='utf-8')
    monkeypatch.setattr(drive, 'SETTINGS_PATH', path)
    d = drive.Driver(hub=None, recorder=None)
    assert d.params['rec_hold_keep'] == 0 and d.params['launch_no_kick'] == 0      # 起動で 0
    assert d.params['steer_trim_us'] == 15 and d.params['start_kick'] == 1         # ほかは今まで通り残す
    for k in NEW:
        assert d._clean_params({k: 1})[k] == 1 and d._clean_params({k: 0.2})[k] == 0 and d._clean_params({k: 7})[k] == 1
        assert isinstance(d._clean_params({k: 0.7})[k], int)
        with pytest.raises(ValueError):
            d._clean_params({k: float('nan')})
    assert d._clean_params({})['rec_hold_keep'] == 0                                # 古い画面(項目なし)は今の値のまま
    assert bench.fallback_baseline()['params']['rec_hold_keep'] == 0


def test_running_cannot_turn_on_but_can_turn_off(monkeypatch, tmp_path):
    """走行中は 0 → 1 にできない(1 → 0 はできる。1 走行 = 1 設定。画面の試験の選択肢のスイッチと同じ)。後ろの見張りも同じ。
    止まっていれば入れられる。変えたら出来事 params に残る(fix2 state2-2)"""
    from test_drive_caps import FakeRecorder
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    monkeypatch.setattr(drive.Driver, 'running', property(lambda self: True))
    rec = FakeRecorder()
    d = drive.Driver(hub=None, recorder=rec)
    d.st = {'run': 'x'}
    for k in NEW + ('rec_back_guard',):
        with pytest.raises(ValueError, match='ON にできません'):
            d.set_params({k: 1})
        assert d.params[k] == 0
    assert d.set_params({'rec_hold_keep': 0, 'steer_gain': 1.1})['steer_gain'] == 1.1      # 0 のまま・ほかの項目は通る
    d.params = dict(d.params, rec_hold_keep=1, launch_no_kick=1, rec_back_guard=1)
    p = d.set_params({'rec_hold_keep': 0, 'launch_no_kick': 0, 'rec_back_guard': 0})         # 1 → 0 はできる
    assert (p['rec_hold_keep'], p['launch_no_kick'], p['rec_back_guard']) == (0, 0, 0)
    ev = [e[1] for e in rec.events if e[0] == 'params']
    assert ev and ev[-1]['changed'] == {'rec_back_guard': [1, 0], 'rec_hold_keep': [1, 0], 'launch_no_kick': [1, 0]}
    monkeypatch.setattr(drive.Driver, 'running', property(lambda self: False))
    p = d.set_params({'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 1})        # 止まっていればできる
    assert (p['rec_back_guard'], p['rec_hold_keep'], p['launch_no_kick']) == (1, 1, 1)


def test_hold_keep_needs_back_guard(monkeypatch, tmp_path):
    """rec_hold_keep は rec_back_guard が ON のときだけ(速さの自動合わせ・天井と同じ形)。見張りを OFF にすると一緒に OFF。
    走行ループも両方 1 のときだけ残す(片方だけなら直す前と同じ)"""
    from test_drive_caps import FakeRecorder
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    with pytest.raises(ValueError, match='後ろの見張りが ON のときだけ'):
        d.set_params({'rec_hold_keep': 1})
    assert d.params['rec_hold_keep'] == 0
    assert d.set_params({'rec_back_guard': 1})['rec_hold_keep'] == 0
    assert d.set_params({'rec_hold_keep': 1})['rec_hold_keep'] == 1
    p = d.set_params({'rec_back_guard': 0})                                                  # 見張りを OFF → 一緒に OFF
    assert (p['rec_back_guard'], p['rec_hold_keep']) == (0, 0)
    with pytest.raises(ValueError, match='後ろの見張りが ON のときだけ'):
        d._opt_rules(d._clean_params({'rec_hold_keep': 1}), False)                         # 走行開始(start)も同じ決まり
    assert '_opt_rules' in __import__('inspect').getsource(drive.Driver.start)
    assert d.set_params({'launch_no_kick': 1})['launch_no_kick'] == 1                        # launch_no_kick は単独で入る


# ================================================================ 2. OFF はビット単位で同じ(assist)
def _random_inputs(seed, n=3000):
    """でたらめな入力の列: 走る・止まる・後ろへ・ブレーキ・人側・衝撃・中立止め(1〜8 フレーム)。発進の前進の入力も混ぜる"""
    rng = random.Random(seed)
    t, out, mode, left = 0.0, [], 'drive', 0
    for _ in range(n):
        if left <= 0:
            mode = rng.choice(['drive', 'stop', 'back', 'brake', 'human', 'impact', 'hold', 'stop'])
            left = rng.randint(1, 8) if mode == 'hold' else rng.randint(5, 60)
        left -= 1
        t += rng.choice([DT, DT, DT, 0.05, 0.25])
        st = rng.uniform(950, 1950)
        if mode == 'drive':
            m, e = (st, rng.uniform(1250, 1480)), rng.uniform(-6000, -1200)
        elif mode in ('stop', 'hold'):
            m, e = (st, rng.uniform(1440, 1540)), rng.uniform(-200, 200)
        elif mode == 'back':
            m, e = (st, rng.uniform(1480, 1900)), rng.uniform(0, 4500)
        elif mode == 'brake':
            m, e = (st, rng.uniform(1550, 1900)), rng.uniform(-3000, 0)
        else:
            m, e = (st, rng.uniform(1300, 1900)), rng.uniform(-3000, 3000)
        ttc = rng.choice([30.0, 30.0, 5.0, 3.3, 2.0, 0.9, float('nan')])
        li = A.LaunchIn(us=rng.choice([150.0, 250.0, 300.0]), v=abs(e) * 0.5e-3, a=rng.uniform(-2, 6), gz=rng.uniform(-80, 80),
                        steer_rel=rng.uniform(-300, 300), ttc=ttc, ttc5=(ttc,) * rng.choice([5, 5, 4]),
                        tof_ok=rng.random() > 0.1, veh_ok=rng.random() > 0.05, impact_g=rng.choice([0.0, 0.0, 3.0]),
                        guard_changed=rng.random() < 0.05)
        out.append({'m': m, 'e': e, 'sel': mode != 'human', 'hold': mode == 'hold', 't': t,
                    'g': rng.uniform(2.6, 8.0) if (mode == 'impact' and rng.random() < 0.2) else rng.uniform(0, 1.5),
                    'rev_us': rng.choice([300.0, 300.0, 450.0, 150.0]), 'rev_s': rng.choice([0.4, 0.4, 0.2, 1.2]),
                    'launch': li if rng.random() < 0.7 else None, 'blocked': rng.random() < 0.1})
    return out


def _drive_assist(a, ins, keep=None, no_kick=None, launch=True, guard=False, kick_on=True, rec_on=True):
    """drive の走行ループと同じ呼び方で入力の列を通す。keep・no_kick が None なら直す前の呼び方(引数を渡さない)"""
    res = []
    for x in ins:
        if x['hold']:
            a.launch_hold()
            if keep is None:
                a.reset()
            else:
                a.reset(keep_recovery=bool(keep) and x['sel'])
            res.append('hold')
            continue
        kw = {} if no_kick is None else {'launch_no_kick': bool(no_kick)}
        r = a.step(x['m'], x['e'], x['sel'], x['t'], esc_on=True, rec_on=rec_on, impact_g=x['g'], rev_us=x['rev_us'],
                   rev_s=x['rev_s'], kick_on=kick_on, blocked=x['blocked'], launch=x['launch'] if launch else None,
                   rec_guard=guard, **kw)
        res.append((r, a.boost.event, a.launch_on, a.launch_end))
    return res


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('launch,guard,kick_on,rec_on', [(True, False, True, True), (False, False, True, True),
                                                         (True, True, True, True), (True, False, False, True),
                                                         (True, False, True, False)])
def test_assist_off_bit_identical_to_before(seed, launch, guard, kick_on, rec_on):
    amod, _ = load_before()
    ins = _random_inputs(seed)
    b = _drive_assist(amod.Assist((C, N)), ins, launch=launch, guard=guard, kick_on=kick_on, rec_on=rec_on)
    n = _drive_assist(A.Assist((C, N)), ins, keep=0, no_kick=0, launch=launch, guard=guard, kick_on=kick_on, rec_on=rec_on)
    assert b == n
    if not launch:          # 発進の前進が OFF なら launch_no_kick = 1 でも同じ(決めないので効かない)
        assert _drive_assist(A.Assist((C, N)), ins, keep=0, no_kick=1, launch=False, guard=guard, kick_on=kick_on,
                             rec_on=rec_on) == b
    kinds = collections.Counter(r[0][1] for r in n if r != 'hold')
    if rec_on:
        assert kinds[A.ASSIST_RECOVERY] > 30 and kinds[A.ASSIST_GIVEUP] >= 0
    assert sum(1 for r in n if r == 'hold') > 20


# ================================================================ 3. OFF はビット単位で同じ(走行ループ)
LOOP_SCENES = ['caps', 'launch', 'giveup_hold', 'mid_hold', 'near_on', 'nofloor_on', 'rec_back_hold', 'fuzz0', 'fuzz1', 'fuzz2']


def _loop_scene(name):
    """(frames, imu, params, run の追加の引数)"""
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, tof_guard=1)
    if name == 'launch':
        f, i = launch_scene(tof=lambda s: tof_near(400) if 1.6 <= s < 1.7 else None)
        return f, i, dict(p, launch_boost=1), {}
    if name == 'giveup_hold':
        f, i = stuck_scene(24.0, holds=[(13.0, 0.2), (17.5, 0.25)])
        return f, i, p, {}
    if name == 'mid_hold':
        f, i = stuck_scene(12.0, holds=[(3.45, 0.2), (5.0, 0.2)])
        return f, i, dict(p, rec_back_guard=1), {}
    if name == 'near_on':
        f, i = near_scene(500)
        return f, i, dict(p, launch_boost=1), {}
    if name == 'nofloor_on':
        f, i = near_scene(350)
        return f, i, dict(p, launch_boost=1), {'floor': False}
    if name == 'rec_back_hold':
        f, i = recovery_scene(back_from=2.95, hit_at=3.15, holds=[(3.0, 0.2)])
        return f, i, dict(p, rec_back_guard=1), {}
    if name.startswith('fuzz'):
        k = int(name[4:])
        rng = random.Random(1026 + k)
        f, i = fuzz_scene(2026 + k)
        return f, i, dict(p, launch_boost=1, speed_ceiling=1, speed_auto=rng.choice([0, 1]), tof_guard=rng.choice([1, 2, 3]),
                          zone_caps=rng.choice([0, 1]), rec_back_guard=rng.choice([0, 1])), {}
    raise KeyError(name)


@pytest.mark.parametrize('scene', ['caps'] + LOOP_SCENES[1:])
def test_loop_off_bit_identical(monkeypatch, tmp_path, scene):
    """rec_hold_keep = 0・launch_no_kick = 0(既定)の走行ループは、直す前の写しと PCA9685 に書く値・記録・出来事が同じ"""
    _, before = load_before()
    if scene == 'caps':
        frames, imu = scenario()
        r0 = run_det(before, monkeypatch, tmp_path, only_before(before, PARAMS), frames, imu)
        r1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, rec_hold_keep=0, launch_no_kick=0), frames, imu)
        assert len(r0[1].pairs) > 300
        same_loop(r0, r1)
        return
    frames, imu, p, kw = _loop_scene(scene)
    r0 = run(before, monkeypatch, tmp_path, only_before(before, p), frames, imu, **kw)
    r1 = run(drive, monkeypatch, tmp_path, dict(p, rec_hold_keep=0, launch_no_kick=0), frames, imu, **kw)
    assert len(r0[1].pairs) > 100
    same_loop(r0, r1)
    if scene in ('giveup_hold', 'mid_hold', 'rec_back_hold'):
        assert any(r['hold'] for _, r in r1[2].rows)                      # 中立止めを通った


@pytest.mark.parametrize('scene', ['caps', 'giveup_hold', 'mid_hold', 'fuzz0'])
def test_launch_no_kick_without_launch_boost_is_identical(monkeypatch, tmp_path, scene):
    """launch_boost = 0 なら launch_no_kick = 1 でも直す前と同じ(発進の前進が決めないので効かない)"""
    _, before = load_before()
    if scene == 'caps':
        frames, imu = scenario()
        r0 = run_det(before, monkeypatch, tmp_path, only_before(before, PARAMS), frames, imu)
        r1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, launch_no_kick=1), frames, imu)
    else:
        frames, imu, p, kw = _loop_scene(scene)
        p = dict(p, launch_boost=0)
        r0 = run(before, monkeypatch, tmp_path, only_before(before, p), frames, imu, **kw)
        r1 = run(drive, monkeypatch, tmp_path, dict(p, launch_no_kick=1), frames, imu, **kw)
    same_loop(r0, r1)


@pytest.mark.parametrize('scene', ['launch', 'near_on'])
def test_rec_hold_keep_without_ai_holds_is_identical(monkeypatch, tmp_path, scene):
    """AI 側の中立止めが無い走行は、rec_hold_keep = 1 でも直す前と同じ(効くのは中立止めのときだけ)"""
    _, before = load_before()
    frames, imu, p, kw = _loop_scene(scene)
    p = dict(p, rec_back_guard=1)
    r0 = run(before, monkeypatch, tmp_path, only_before(before, p), frames, imu, **kw)
    r1 = run(drive, monkeypatch, tmp_path, dict(p, rec_hold_keep=1), frames, imu, **kw)
    assert not any(r['hold'] and r['select'] for _, r in r1[2].rows)
    same_loop(r0, r1)


# ================================================================ 4. rec_hold_keep = 1 の効き(走行ループ)
P_STUCK = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, launch_boost=0, tof_guard=1)


def test_giveup_survives_hold(monkeypatch, tmp_path):
    """safety1-1 の 1a: リカバリー中止の 1.5 秒あとに中立止め 1 回(0.2 秒)。OFF は中止が解けてモデルの前進(中立 − 120)が
    人の交代まで出続ける(指摘の再現)。ON は中止のまま中立"""
    frames, imu = stuck_scene(24.0)
    tl = timeline(frames, run(drive, monkeypatch, tmp_path, P_STUCK, frames, imu)[2])
    g0 = next(s for s, k, h, o, sel in tl if k == A.ASSIST_GIVEUP)
    hold_at = round(g0 + 1.5, 3)
    frames, imu = stuck_scene(24.0, holds=[(hold_at, 0.2)])
    off = timeline(frames, run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=0), frames, imu)[2])
    on = timeline(frames, run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=1, rec_back_guard=1), frames, imu)[2])
    only = timeline(frames, run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=1), frames, imu)[2])
    assert only == off                                                                  # 見張りが OFF なら残さない(今まで通り)
    assert any(h and sel for s, k, h, o, sel in on if hold_at <= s < hold_at + 0.3)    # 中立止めを通った
    a_off = [x for x in off if x[0] >= hold_at + 0.4 and not x[2]]
    a_on = [x for x in on if x[0] >= hold_at + 0.4 and not x[2]]
    assert a_off and all(k == A.ASSIST_NONE and o == N - 120.0 for s, k, h, o, sel in a_off)     # 指摘: 壁を押し続ける
    assert a_on and all(k == A.ASSIST_GIVEUP and o == N for s, k, h, o, sel in a_on)             # 直し: 中止のまま中立
    assert [x for x in on if x[0] < hold_at] == [x for x in off if x[0] < hold_at]            # 中立止めの前は同じ


def test_recovery_in_progress_continues_after_hold(monkeypatch, tmp_path):
    """safety1-1 の 1b: リカバリーの後退の途中に中立止め 1 回。OFF は取り消され armed も消え、止まった車はリカバリーが
    二度と始まらない(モデルの前進が出続ける)。ON は続きから・そのあとも見つけ直し、3 回で中止(中立)"""
    frames, imu = stuck_scene(14.0)
    tl = timeline(frames, run(drive, monkeypatch, tmp_path, P_STUCK, frames, imu)[2])
    r0 = next(s for s, k, h, o, sel in tl if k == A.ASSIST_RECOVERY)
    hold_at = round(r0 + 0.45, 3)
    frames, imu = stuck_scene(14.0, holds=[(hold_at, 0.2)])
    off = timeline(frames, run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=0), frames, imu)[2])
    on = timeline(frames, run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=1, rec_back_guard=1), frames, imu)[2])
    a_off = [x for x in off if x[0] >= hold_at + 0.4 and not x[2]]
    a_on = [x for x in on if x[0] >= hold_at + 0.4 and not x[2]]
    assert not [x for x in a_off if x[1] in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP)]          # 指摘の再現
    assert all(o == N - 120.0 for s, k, h, o, sel in a_off)
    after_hold = [x for x in on if hold_at <= x[0] < hold_at + 0.6 and not x[2]]
    assert after_hold and after_hold[0][1] == A.ASSIST_RECOVERY                              # 続きから
    assert a_on[-1][1] == A.ASSIST_GIVEUP and a_on[-1][3] == N                               # 最後は中止(人を待つ)
    push_on = sum(1 for x in a_on if x[1] == A.ASSIST_NONE and x[3] < N)
    assert push_on < len(a_off) / 2                                                          # 壁を押す時間が短い(見つけ直すまでの間だけ)


def test_ch3_human_still_clears_giveup(monkeypatch, tmp_path):
    """ON でも CH3 を人側 → AI 側にすれば中止は解ける(今まで通りの「人が交代して直した」)"""
    frames, imu = stuck_scene(24.0)
    tl = timeline(frames, run(drive, monkeypatch, tmp_path, P_STUCK, frames, imu)[2])
    g0 = next(s for s, k, h, o, sel in tl if k == A.ASSIST_GIVEUP)
    hum = round(g0 + 1.0, 3)
    frames, imu = stuck_scene(24.0, holds=[(round(g0 + 0.5, 3), 0.2)], human=[(hum, 0.5)])
    for keep in (0, 1):
        t = timeline(frames, run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=keep, rec_back_guard=keep), frames,
                                 imu)[2])
        mid = [x for x in t if g0 + 0.8 <= x[0] < hum and not x[2]]
        back = [x for x in t if hum + 0.6 <= x[0] < hum + 1.2 and not x[2]]
        assert all(x[1] == A.ASSIST_GIVEUP for x in mid) == bool(keep)          # 中立止めのあと: ON は中止のまま
        assert back and all(x[1] != A.ASSIST_GIVEUP for x in back)              # 人側のあとは解けている(どちらも)


def test_drive_calls_keep_only_for_ai_side_holds(monkeypatch, tmp_path):
    """drive の hold は、rec_hold_keep = 1 で最新の車両の値が AI 側(dry run でない)のときだけ keep_recovery=True で戻す"""
    calls = []

    class Spy(A.Assist):
        def reset(self, keep_recovery=False):
            calls.append(keep_recovery)
            return super().reset(keep_recovery=keep_recovery)
    monkeypatch.setattr(drive.assist_mod, 'Assist', Spy)
    frames, imu = build(4.0, sel=lambda s: 1.0 <= s < 3.0, v=0.0, pred=(C, N),
                        veh_skip=lambda s: 0.4 <= s < 0.65 or 2.0 <= s < 2.25 or 3.3 <= s < 3.55)
    for keep, guard, dry in ((1, 1, False), (0, 1, False), (1, 1, True), (1, 0, False)):
        calls.clear()
        _, _, rec = run(drive, monkeypatch, tmp_path, dict(P_STUCK, rec_hold_keep=keep, rec_back_guard=guard), frames, imu,
                        dry_run=dry)
        holds = [(r['select']) for _, r in rec.rows if r['hold']]
        assert True in holds and False in holds                                # AI 側・人側の両方の中立止め
        kept = [c for c in calls if c]
        if keep and guard and not dry:
            assert len(kept) == holds.count(True)                              # AI 側の中立止めだけ残す
        else:
            assert kept == []


def test_rec_hold_keep_with_back_guard_no_reverse_into_rear_wall(monkeypatch, tmp_path):
    """後ろの見張り(rec_back_guard)と組み合わせ: 後退の途中の中立止めのあとに後ろへ当たっても、ON はすぐ前へ(後ろへ押さない)。
    見張りの手順(下がった時間・後ろの衝撃)と矛盾しない"""
    frames, imu = recovery_scene(back_from=2.95, hit_at=3.25, holds=[(3.05, 0.2)])
    res = {}
    for keep in (0, 1):
        for guard in (0, 1):
            _, _, rec = run(drive, monkeypatch, tmp_path, dict(PARAMS, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
            s_of = {f['cam_t']: f['s'] for f in frames}
            res[(keep, guard)] = {
                'rev_after_hit': sum(1 for t, r in rec.rows if not r['hold'] and r['out'][1] > N + 40.0 and s_of[t] >= 3.3),
                'events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard'],
                'holds': sum(1 for _, r in rec.rows if r['hold'])}
    assert all(v['holds'] > 0 for v in res.values())
    assert res[(1, 1)]['rev_after_hit'] == 0 and 'rear_impact' in res[(1, 1)]['events']
    assert res[(1, 1)]['rev_after_hit'] <= res[(0, 1)]['rev_after_hit']
    assert res[(1, 0)] == res[(0, 0)]                                          # 見張りが OFF なら rec_hold_keep は効かない


def test_assist_keep_without_guard_pushes_rear_wall():
    """見張りを条件にした理由(Assist の形で): 後退の途中の中立止めのあとに後ろへ当たると、見張りなしで残したリカバリーは
    やり直しで後ろを押し続ける(今までは中立止めの reset が手順を偶然切っていた)。見張りありならすぐ前へ"""
    def go(keep, guard):
        a = A.Assist((C, N))
        t, rev_after = 0.0, 0
        for i in range(30):
            a.step((C, N - 150.0), -5000, True, t); t += DT
        for i in range(75):                               # 2.5 秒(このあとの見つけ直しの前まで)
            s = i * DT
            if 0.0 <= s < 0.85:
                e = 0.0                                   # 止まった(0.8 秒で見つける → 'brake')
            elif s < 1.25:
                e = 2000.0                                # 下がっている(ブレーキの段から)
            else:
                e = 0.0                                   # 1.25 秒で後ろに当たって止まった
            if 1.15 <= s < 1.25:                          # 下がっている途中に中立止め 3 フレーム
                a.launch_hold(); a.reset(keep_recovery=keep); t += DT; continue
            g = 6.0 if 1.25 <= s < 1.33 else 0.0          # 当たった衝撃(後退を出し始めて 0.1 秒の内だけ見える)
            o, k = a.step((C, N + 20.0), e, True, t, impact_g=g, rec_guard=guard)
            if s >= 1.25 and o[1] > N + 40:
                rev_after += 1
            t += DT
        return rev_after
    assert go(True, False) > 15                       # 見張りなしで残す: 後ろを押す(0.5 秒より長い)
    assert go(True, True) == 0 and go(False, False) == 0 and go(False, True) == 0


# ================================================================ 5. rec_hold_keep の本物の記録(10/10)
def _replay_1010(keep, no_noise=False):
    """10/10 20261010-181504-0004(7.5G の衝突の 0.43 秒あとに中立止め 2 フレーム)を Assist に通す(反証 1 回目
    safety1/replay_1010_hold.py と同じ入力)。戻り値 最初のリカバリーの行"""
    z = np.load(REC_1010 / 'AI_CMD.npz'); v = np.load(REC_1010 / 'VEHICLE.npz'); imu = np.load(REC_1010 / 'IMU.npz')
    t = z['timestamps_us'] / 1e6; tv = v['timestamps_us'] / 1e6; ti = imu['timestamps_us'] / 1e6
    hg = np.hypot(imu['accel'][:, 0], imu['accel'][:, 1]) / 9.80665
    h = A.Assist((1443.0, 1498.0))
    for i in range(100, 200):
        if not z['select'][i]:
            return None
        if z['hold'][i]:
            h.launch_hold(); h.reset(keep_recovery=keep)
            continue
        e = float(v['erpm'][np.searchsorted(tv, t[i], side='right') - 1])
        if no_noise and i >= 143 and abs(e) >= 1000:
            e = 0.0
        m = (ti > t[i] - 0.15) & (ti <= t[i] + 0.01)
        g = float(hg[m].max()) if m.any() else 0.0
        o, k = h.step(tuple(float(x) for x in z['model_out'][i]), e, True, float(t[i]), esc_on=True, rec_on=True, impact_g=g)
        if k == A.ASSIST_RECOVERY:
            return i
    return None


@pytest.mark.skipif(not (REC_1010 / 'AI_CMD.npz').exists(), reason='10/10 の記録がありません')
def test_replay_1010_keep_starts_recovery_earlier():
    z = np.load(REC_1010 / 'AI_CMD.npz')
    assert bool(z['hold'][141]) and bool(z['hold'][142])
    assert _replay_1010(False) == 177 == int(np.where(z['assist'][100:200] == 2)[0][0] + 100)   # OFF = 記録と同じ
    on = _replay_1010(True)
    assert on is not None and on < 177                                          # ON は早く始まる
    assert _replay_1010(False, no_noise=True) is None                           # 雑音が無ければ OFF は人が取り返すまで始まらない
    on2 = _replay_1010(True, no_noise=True)
    assert on2 is not None and on2 < 190


# ================================================================ 6. launch_no_kick = 1 の効き
@pytest.mark.parametrize('d_mm,floor,why', [(350, True, 'near'), (500, True, 'near'), (900, True, 'near'), (350, False, 'tof')])
def test_launch_skip_suppresses_start_kick(monkeypatch, tmp_path, d_mm, floor, why):
    """safety1-2: 発進の前進が出さない(前が近い・ToF が無い)区間。OFF は StartKick が前進 80 µs を 2 秒(指摘の再現)。ON は出さない"""
    frames, imu = near_scene(d_mm)
    out = {}
    for nk in (0, 1):
        _, _, rec = run(drive, monkeypatch, tmp_path, dict(PARAMS, launch_boost=1, start_kick=1, tof_guard=1, launch_no_kick=nk),
                        frames, imu, floor=floor)
        rows = [r for _, r in rec.rows if r['select'] and not r['hold']]
        skips = [e[1] for e in rec.events if e[0] == 'launch_boost']
        out[nk] = {'fwd': sum(1 for r in rows if r['out'][1] < N), 'kick': sum(1 for r in rows if r['assist'] == A.ASSIST_START),
                   'skips': skips}
    assert [s['why'] for s in out[0]['skips']] == [why] and [s['why'] for s in out[1]['skips']] == [why]
    assert out[0]['kick'] > 30 and out[0]['fwd'] > 30 and 'no_kick' not in out[0]['skips'][0]
    assert out[1]['kick'] == 0 and out[1]['fwd'] == 0 and out[1]['skips'][0]['no_kick'] is True


def test_launch_started_is_same_with_or_without_no_kick(monkeypatch, tmp_path):
    """前が空いていて発進の前進が始まる区間は、launch_no_kick の 0 と 1 で同じ(出したあとはもとから StartKick を出さない)"""
    frames, imu = launch_scene()
    p = dict(PARAMS, launch_boost=1, start_kick=1)
    r0 = run(drive, monkeypatch, tmp_path, dict(p, launch_no_kick=0), frames, imu)
    r1 = run(drive, monkeypatch, tmp_path, dict(p, launch_no_kick=1), frames, imu)
    assert r1[0].st['stats']['launch_boosts'] == 2
    same_loop(r0, r1)


def test_next_section_decides_again(monkeypatch, tmp_path):
    """前が近くて出さなかった区間のあと、CH3 を人側 → AI 側にして前が空いていれば、次の区間は発進の前進が始まる"""
    frames, imu = build(6.0, sel=lambda s: 1.0 <= s < 3.0 or s >= 3.5, v=0.0, pred=(C, N),
                        tof=lambda s: tof_near(500) if s < 3.2 else None)
    _, _, rec = run(drive, monkeypatch, tmp_path, dict(PARAMS, launch_boost=1, start_kick=1, launch_no_kick=1), frames, imu)
    ev = [(e[1]['event'], e[1].get('why')) for e in rec.events if e[0] == 'launch_boost']
    assert ev[0] == ('skip', 'near') and ('start', None) in ev
    kinds = [r['assist'] for _, r in rec.rows if r['select'] and not r['hold']]
    assert A.ASSIST_LAUNCH in kinds and A.ASSIST_START not in kinds


@pytest.mark.parametrize('why,kw,erpm', [('near', dict(ttc5=(2.0,) * 5), 0.0), ('ttc_n', dict(ttc5=(30.0,) * 4), 0.0),
                                         ('tof', dict(tof_ok=False), 0.0), ('veh', dict(veh_ok=False), 0.0),
                                         ('moving', {}, 450.0), ('end', dict(impact_g=3.0), 0.0)])
def test_assist_each_skip_reason(why, kw, erpm):
    """Assist の形で: どの「出さない」理由でも、ON は StartKick を出さない。OFF は今まで通り出す"""
    base = dict(us=250.0, mps=2.5, max_s=1.2, v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=30.0, ttc5=(30.0,) * 5, tof_ok=True,
                veh_ok=True, impact_g=0.0, guard_changed=False)
    base.update(kw)
    li = A.LaunchIn(**base)
    for nk in (False, True):
        a = A.Assist((C, N))
        a.step((C, N), 0.0, False, -0.1)                      # 人側を見た(「AI 側にした」と分かる)
        kinds = []
        for i in range(30):
            out, k = a.step((C, N), erpm, True, i * DT, kick_on=True, launch=li, launch_no_kick=nk)
            kinds.append(k)
            if i == 0:
                assert a.boost.last_skip == why and a.boost.skipped
        assert (A.ASSIST_START in kinds) == (not nk), (why, nk)
        assert A.ASSIST_LAUNCH not in kinds


def test_assist_no_switch_and_hold_skip():
    """走行の始めから AI 側(no_switch)・AI 側にした最初が中立止め(hold)でも、ON は StartKick を出さない"""
    li = A.LaunchIn(ttc=30.0, ttc5=(30.0,) * 5, tof_ok=True, veh_ok=True)
    for case in ('no_switch', 'hold'):
        for nk in (False, True):
            a = A.Assist((C, N))
            if case == 'hold':
                a.step((C, N), 0.0, False, -0.1)
                a.launch_hold()
            k0 = a.step((C, N), 0.0, True, 0.0, kick_on=True, launch=li, launch_no_kick=nk)[1]
            assert a.boost.last_skip == case and (k0 == A.ASSIST_START) == (not nk)


# ================================================================ 7. 画面(app.js): 走行開始で送る値は変わらない
@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_fallback_has_new_keys_and_matches_server():
    js = (ROOT / 'jetformer' / 'web' / 'static' / 'app.js').read_text(encoding='utf-8')
    import re
    part = re.search(r'^const BENCH_FALLBACK = \{.*?^};\n', js, re.S | re.M).group(0)
    r = subprocess.run([NODE, '-e', part + '\nprocess.stdout.write(JSON.stringify(BENCH_FALLBACK));'], capture_output=True,
                       text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got['params'] == bench.fallback_baseline()['params'] and got['params']['rec_hold_keep'] == 0
    assert got['params']['launch_no_kick'] == 0


@pytest.fixture(scope='module')
def cands():
    c = bench.load_candidates()
    days = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    if not days:
        pytest.skip('data/bench_candidates.json に 2026-11-03 のブロックがありません')
    return dict(c, today='2026-11-03'), days[0]


@pytest.mark.skipif(NODE is None, reason='node がありません')
@pytest.mark.parametrize('srv', [{}, {'rec_hold_keep': 1}, {'launch_no_kick': 1}, {'rec_hold_keep': 1, 'launch_no_kick': 1},
                                 {'rec_hold_keep': 1, 'launch_no_kick': 1, 'rec_back_guard': 1, 'launch_boost': 1}])
def test_js_start_sends_same_values_and_shows_on_lines(tmp_path, cands, srv):
    """本物の app.js(今と直す前の写し)を node の vm で。基準の設定・ブロック・走行開始で送る値は直す前の写しと同じ
    (新しい 2 つは走行開始では送らない = サーバーの値のまま)。確認の文は、サーバーの値が ON の項目の行が足されるだけ
    (同じ画面で 2 つを 0 にしたサーバーと比べる。画面のほかの直し(係 state)とは切り分ける)"""
    from test_bench_ui_round2 import run_sim, starts
    _, di = cands
    steps = [['baseline'], ['start'], ['poll'], ['pick', di, 1], ['start'], ['poll'], ['pick', di, 2], ['start']]
    srv0 = dict(srv, rec_hold_keep=0, launch_no_kick=0)
    new = starts(run_sim(tmp_path, cands, steps, server=srv))
    new0 = starts(run_sim(tmp_path, cands, steps, server=srv0))
    old = starts(run_sim(tmp_path, cands, steps, server=srv, app=FIX / 'app.js'))
    assert len(new) == 3 and [x['sent'] for x in new] == [x['sent'] for x in old] == [x['sent'] for x in new0]
    for x in new:
        assert not set(NEW) & set(x['sent'])
    heads = {'rec_hold_keep': '中立止めのあともリカバリーを残す: ON', 'launch_no_kick': '発進の前進を出さない区間は発進の補助も出さない: ON'}
    for a, b in zip(new, new0):
        la = a['confirm'].split('\n')
        for k, head in heads.items():
            assert sum(1 for x in la if x.startswith(head)) == (1 if srv.get(k) else 0), (k, a['confirm'])
        rest = [x for x in la if not any(x.startswith(h) for h in heads.values())]
        assert rest == b['confirm'].split('\n')                               # ほかの行は 2 つが OFF のときと同じ
