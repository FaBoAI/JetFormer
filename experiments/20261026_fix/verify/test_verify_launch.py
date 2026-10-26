"""反証(verify)の試験: 発進の前進(launch_boost)。段 launch の試験とは別に、自分で書いた場面で確かめる。
実機・GPU・torch は使わない。tests/ の偽物(test_launch_drive の run・build)を使う。"""
import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A          # noqa: E402
from jetformer import drive                # noqa: E402
from jetformer import safety_caps as sc    # noqa: E402

import test_launch_drive as L              # noqa: E402
from test_drive_caps import assert_rows_equal_before   # noqa: E402

C, N = L.C, L.N
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)


def note(name, obj):
    (OUT / (name + '.json')).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))


# ---------------------------------------------------------------- 場面: CH3 を何度も・前の物が近づく・衝撃・古い値・ブレーキ
def busy_scene(seed):
    rng = random.Random(seed)
    toggles = sorted(rng.uniform(0.6, 7.5) for _ in range(8))

    def sel(s):
        n = sum(1 for t in toggles if t <= s)
        return n % 2 == 1

    def v(s):
        # AI 側で加速、人側で減速(実際の車とは関係なく、いろいろな速さが出るように)
        return max(0.0, 3.2 * math.sin(s * 1.3) ** 2 - 0.3) if s > 1.0 else 0.0

    def pred(s):
        k = int(s * 7) % 5
        return [(C, N + 120.0), (C, N - 150.0), (C + 220.0, N - 300.0), (C - 180.0, N - 60.0), (C, N)][k]

    def tof(s):
        if 2.0 <= s < 2.3:
            return L.tof_near(450)
        if 4.0 <= s < 4.4:
            return L.tof_near(900)
        if 6.0 <= s < 6.2:
            return L.tof_near(300)
        return None
    return L.build(8.0, sel=sel, v=v, pred=pred, tof=tof,
                   gz=lambda s: 90.0 if 3.0 <= s < 3.2 else 10.0 * math.sin(s),
                   ax=lambda s: 30.0 if 5.0 <= s < 5.01 else 0.0,
                   valid3=lambda s: not (5.5 <= s < 5.56),
                   veh_skip=lambda s: 6.5 <= s < 6.64)


COMBOS = [
    dict(tof_guard=2, start_kick=1, zone_caps=1, speed_ceiling=1, speed_auto=1),
    dict(tof_guard=3, start_kick=0, zone_caps=0, speed_ceiling=1, speed_auto=0),
    dict(tof_guard=1, start_kick=1, zone_caps=1, speed_ceiling=0, speed_auto=0),
    dict(tof_guard=0, start_kick=1, zone_caps=0, speed_ceiling=1, speed_auto=1, recovery=0, esc_reverse=0),
]


@pytest.mark.parametrize('ci', range(len(COMBOS)))
@pytest.mark.parametrize('seed', [1, 2])
@pytest.mark.parametrize('floor', [True, False])
def test_v1_off_bit_identical_full_stack(monkeypatch, tmp_path, ci, seed, floor):
    """OFF(値は既定でない)で、天井・自動合わせ・区間の上限・StartKick・ガード 0〜3 と一緒でも、PCA9685 に書く値・
    記録の今までの欄・出来事・stats が手を入れる前の写しと同じ"""
    before = L.load_before()
    frames, imu = busy_scene(seed)
    p = dict(L.PARAMS, **COMBOS[ci])
    pb = {k: v for k, v in p.items() if k in before.DEFAULT_PARAMS}
    d0, pca0, rec0 = L.run(before, monkeypatch, tmp_path, pb, frames, imu, floor=floor)
    d1, pca1, rec1 = L.run(drive, monkeypatch, tmp_path,
                           dict(p, launch_boost=0, launch_us=150.0, launch_mps=3.0, launch_max_s=2.0), frames, imu,
                           floor=floor)
    assert len(pca0.pairs) > 150
    assert pca0.pairs == pca1.pairs and pca0.single == pca1.single
    assert_rows_equal_before(rec0.rows, rec1.rows)
    assert rec0.events == rec1.events
    s0, s1 = dict(d0.st['stats']), dict(d1.st['stats'])
    for k in ('fps', 'latency_ms', 'latency_ms_max', 'infer_ms', 'zone_ms', 'zone_ms_max', 'ceil_ms', 'ceil_ms_max',
              'auto_ms', 'auto_ms_max'):
        s0.pop(k, None)
        s1.pop(k, None)
    assert s1.pop('launch_boosts') == 0
    assert s0 == s1
    assert all(r['launch_on'] is False and r['launch_end'] == 0 for _, r in rec1.rows)
    kinds = collections.Counter(r['assist'] for _, r in rec1.rows)
    note('v1_off_%d_%d_%d' % (ci, seed, int(floor)), {'frames': len(rec1.rows), 'assist_kinds': dict(kinds),
                                                     'holds': d1.st['stats']['holds']})


@pytest.mark.parametrize('seed', [1, 2])
def test_v1_on_dry_run_identical(monkeypatch, tmp_path, seed):
    """ON でも dry run なら出力・今の欄は同じ(発進の前進は出ない)"""
    before = L.load_before()
    frames, imu = busy_scene(seed)
    p = dict(L.PARAMS, **COMBOS[0])
    pb = {k: v for k, v in p.items() if k in before.DEFAULT_PARAMS}
    _, pca0, rec0 = L.run(before, monkeypatch, tmp_path, pb, frames, imu, dry_run=True)
    d1, pca1, rec1 = L.run(drive, monkeypatch, tmp_path, dict(p, launch_boost=1), frames, imu, dry_run=True)
    assert pca0.pairs == pca1.pairs and pca0.single == pca1.single
    assert_rows_equal_before(rec0.rows, rec1.rows)
    assert d1.st['stats']['launch_boosts'] == 0


def test_v2_assist_fuzz_off_identical():
    """Assist だけ: 写しの Assist と新しい Assist(launch=None)を、でたらめな入力 3 万フレームで比べる"""
    before = L.load_before()
    AB = before.assist_mod
    rng = np.random.default_rng(7)
    for trial in range(6):
        a0, a1 = AB.Assist(L.REF), A.Assist(L.REF)
        now = 100.0
        sel = True
        for i in range(5000):
            now += float(rng.choice([1 / 30, 1 / 30, 1 / 15, 0.2, 0.0]))
            if rng.random() < 0.03:
                sel = not sel
            erpm = float(rng.choice([0.0, -50.0, 150.0, -300.0, -1200.0, -6000.0, 900.0, rng.normal(0, 3000)]))
            st = float(rng.uniform(1000, 2000))
            th = float(rng.choice([N, N + 200, N - 250, rng.uniform(1000, 2000)]))
            kw = dict(esc_on=bool(rng.random() < 0.8), rec_on=bool(rng.random() < 0.8),
                      impact_g=float(rng.choice([0.0, 0.5, 3.0])), kick_on=bool(rng.random() < 0.5),
                      blocked=bool(rng.random() < 0.2))
            if rng.random() < 0.02:
                a0.reset()
                a1.reset()
            o0 = a0.step((st, th), erpm, sel, now, **kw)
            o1 = a1.step((st, th), erpm, sel, now, launch=None, **kw)
            assert o0 == o1, (trial, i, o0, o1)
            assert (a0.recoveries, a0.esc_count, a0.kick_count) == (a1.recoveries, a1.esc_count, a1.kick_count)
            assert a1.launch_on is False and a1.launch_end == 0


# ---------------------------------------------------------------- ON の安全
def approach_scene(d0_m, acc=6.0, appear=None):
    """人側 1.0 秒 → AI 側。0.1 秒止まって、そのあと acc で加速(出力とは無関係)。前の物は d0_m から近づく。
    appear = (秒, 距離 m): その秒から急に前に物"""
    def v(s):
        return 0.0 if s < 1.1 else min(acc * (s - 1.1), 4.0)

    def x(s):
        if s < 1.1:
            return 0.0
        t = s - 1.1
        tc = 4.0 / acc
        return 0.5 * acc * t * t if t < tc else 0.5 * acc * tc * tc + 4.0 * (t - tc)

    def tof(s):
        if appear is not None and s >= appear[0]:
            dd = appear[1] - (x(s) - x(appear[0]))
            return L.tof_near(max(int(dd * 1000), 60)) if dd > 0.06 else L.TOF_FREE
        if d0_m is None:
            return None
        dd = d0_m - x(s)
        return L.tof_near(max(int(dd * 1000), 60)) if dd > 0.06 else L.TOF_FREE
    frames, imu = L.build(3.0, sel=lambda s: s >= 1.0, v=v, tof=tof, pred=lambda s: (C, N - 120.0) if s >= 1.0 else (C, N))
    return frames, imu, x, v


@pytest.mark.parametrize('guard', [1, 2])
@pytest.mark.parametrize('d0', [0.95, 1.05, 1.2, 1.5])
def test_v3_near_object_at_start(monkeypatch, tmp_path, guard, d0):
    """前に物(d0 m)。1.0 m 未満なら始めない。始めたら、物に近づいて TTC < 0.8 × 2 フレーム(またはガード)で終わる。
    終わったときの残りの距離と速さを残す"""
    frames, imu, x, v = approach_scene(d0)
    d, pca, rec = L.run(drive, monkeypatch, tmp_path, dict(L.PARAMS, launch_boost=1, tof_guard=guard), frames, imu)
    ev = [e[1] for e in rec.events if e[0] == 'launch_boost']
    lf = [(t, r) for t, r in rec.rows if r['launch_on']]
    res = {'d0': d0, 'guard': guard, 'events': ev, 'launch_frames': len(lf)}
    if d0 < 1.0:
        assert d.st['stats']['launch_boosts'] == 0 and ev and ev[0]['why'] == 'near'
    else:
        assert d.st['stats']['launch_boosts'] == 1
        end = next((t, r) for t, r in rec.rows if r['launch_end'])
        s_end = (end[0] - L.T0) / 1e9
        res.update(end_code=end[1]['launch_end'], s_end=round(s_end, 3), v_end=round(v(s_end), 3),
                   left_m=round(d0 - x(s_end), 3), ttc_end=end[1]['ttc'])
        # 物にぶつかる前に終わる
        assert d0 - x(s_end) > 0.3, res
        for t, r in lf:
            assert r['out'][1] <= N - 250.0 + 1e-9
    note('v3_near_%s_g%d' % (d0, guard), res)


@pytest.mark.parametrize('guard', [1, 2])
def test_v3_sudden_object_after_start(monkeypatch, tmp_path, guard):
    """始めた直後(まだ 0.3 m/s 未満)に前 0.25 m に物が出た。何フレーム前進を出し続けるか"""
    frames, imu, x, v = approach_scene(None, acc=3.0, appear=(1.12, 0.25))
    d, pca, rec = L.run(drive, monkeypatch, tmp_path, dict(L.PARAMS, launch_boost=1, tof_guard=guard), frames, imu)
    lf = [(t, r) for t, r in rec.rows if r['launch_on']]
    after = [(round((t - L.T0) / 1e9, 3), r['ttc'], round(v((t - L.T0) / 1e9), 3)) for t, r in lf
             if (t - L.T0) / 1e9 >= 1.12]
    end = next(((t, r) for t, r in rec.rows if r['launch_end']), None)
    s_end = None if end is None else (end[0] - L.T0) / 1e9
    res = {'guard': guard, 'frames_after_object': len(after), 'rows': after,
           'end_code': None if end is None else end[1]['launch_end'],
           's_end': s_end, 'travel_after_object_m': None if s_end is None else round(x(s_end) - x(1.12), 3)}
    note('v3_sudden_g%d' % guard, res)
    assert end is not None and end[1]['launch_end'] == 6
    assert x(s_end) - x(1.12) < 0.25


def test_v3_ch3_flick_rearm_while_rolling(monkeypatch, tmp_path):
    """出している間に CH3 を 1 フレームだけ人側 → 次の AI 側で、まだ遅ければ(|eRPM| < 400 = 0.2 m/s)もう 1 回始まるか"""
    def v(s):
        return 0.0 if s < 1.1 else min(0.5 * (s - 1.1), 3.0)            # ゆっくり(0.5 m/s²)
    frames, imu = L.build(3.0, sel=lambda s: s >= 1.0 and not (1.3 <= s < 1.34), v=v,
                          pred=lambda s: (C, N + 100.0))
    d, pca, rec = L.run(drive, monkeypatch, tmp_path, dict(L.PARAMS, launch_boost=1), frames, imu)
    ev = [(e[1]['event'], e[1].get('code'), e[1].get('why')) for e in rec.events if e[0] == 'launch_boost']
    note('v3_ch3_flick', {'events': ev, 'count': d.st['stats']['launch_boosts']})
    assert ev[0][0] == 'start' and ev[1] == ('end', 7, None)
    for t, r in rec.rows:
        if not r['select']:
            assert r['assist'] == A.ASSIST_NONE and not r['launch_on']


@pytest.mark.parametrize('erpm_back', [150.0, 300.0])
def test_v3_rolling_back_at_start(monkeypatch, tmp_path, erpm_back):
    """AI 側にしたとき少し後ろへ動いている(eRPM +150・+300)。+300 は始めない。+150 は始めて、後ろが増えたら ⑨"""
    def v(s):
        if s < 1.0:
            return -erpm_back * sc.KV
        if s < 1.2:
            return -erpm_back * sc.KV
        return -0.2
    frames, imu = L.build(2.5, sel=lambda s: s >= 1.0, v=v, pred=lambda s: (C, N + 200.0))
    d, pca, rec = L.run(drive, monkeypatch, tmp_path, dict(L.PARAMS, launch_boost=1), frames, imu)
    ev = [e[1] for e in rec.events if e[0] == 'launch_boost']
    note('v3_back_%d' % erpm_back, {'events': ev})
    if erpm_back > 200:
        assert d.st['stats']['launch_boosts'] == 0 and ev[0]['why'] == 'end' and ev[0]['code'] == 9
    else:
        assert d.st['stats']['launch_boosts'] == 1
        assert [e for e in ev if e['event'] == 'end'][0]['code'] == 9


def test_v3_launchboost_fuzz_forward_only():
    """LaunchBoost だけ: でたらめな入力(nan・inf・負の us・範囲外)5 万回。出す値はいつも 中立 − 150 以下(前進)で、
    min(モデル, 中立 − us) と同じ。中立より後ろを出さない"""
    rng = np.random.default_rng(11)
    bad = [float('nan'), float('inf'), -float('inf')]
    n_on = 0
    for trial in range(50):
        b = A.LaunchBoost(N)
        now = 0.0
        b.human()
        for i in range(1000):
            now += float(rng.choice([1 / 30, 1 / 30, 1 / 30, 0.0, 0.5, -0.01]))
            if rng.random() < 0.05:
                b.human()
            if rng.random() < 0.003:
                b.hold()
            def r(lo, hi, good):
                if rng.random() < 0.02:
                    return float(rng.choice(bad))
                if rng.random() < 0.8:
                    return float(good)
                return float(rng.uniform(lo, hi))
            inp = A.LaunchIn(us=float(rng.uniform(-500, 900)), mps=float(rng.uniform(-5, 9)), max_s=float(rng.uniform(-1, 9)), v=r(-1, 4, 0.5),
                             a=r(-20, 20, 2.0), gz=r(-100, 100, 0.0), steer_rel=r(-500, 500, 0.0), ttc=r(0, 30, 30),
                             ttc5=tuple(r(0, 30, 30) for _ in range(int(rng.integers(3, 7)))),
                             tof_ok=bool(rng.random() < 0.97), veh_ok=bool(rng.random() < 0.97),
                             impact_g=r(0, 4, 0.0), guard_changed=bool(rng.random() < 0.01))
            thr = r(900, 2100, N + 100)
            erpm = r(-6000, 1000, 0.0)
            out, code = b.step(thr, erpm, now, None if rng.random() < 0.005 else inp)
            if out is not None:
                n_on += 1
                assert math.isfinite(out) and out < N
                want = N - b.cfg[0]
                assert 150.0 <= b.cfg[0] <= 300.0
                assert out == (thr if (math.isfinite(thr) and thr < want) else want)
    note('v3_fuzz', {'on_frames': n_on})
    assert n_on > 100


# ---------------------------------------------------------------- 天井・自動合わせとの組み合わせ(本物の ceil_step)
@pytest.mark.parametrize('zone', [0, 1])
def test_v4_with_real_ceiling_and_auto(monkeypatch, tmp_path, zone):
    frames, imu = L.launch_scene()
    p = dict(L.PARAMS, launch_boost=1, speed_ceiling=1, speed_auto=1, zone_caps=zone, throttle_gain=0.9)
    d, pca, rec = L.run(drive, monkeypatch, tmp_path, p, frames, imu)
    lf = [(t, r) for t, r in rec.rows if r['launch_on']]
    res = {'zone': zone, 'launch_frames': len(lf), 'count': d.st['stats']['launch_boosts'],
           'ceil_applied_in_launch': sum(1 for _, r in lf if r['ceil_applied']),
           'zone_applied_in_launch': sum(1 for _, r in lf if r.get('cap_applied')),
           'min_fwd_us': min(N - r['out'][1] for _, r in lf), 'max_fwd_us': max(N - r['out'][1] for _, r in lf),
           'gains': sorted(set(round(r['gains'][1], 4) for _, r in lf))}
    note('v4_ceiling_auto_zone%d' % zone, res)
    assert d.st['stats']['launch_boosts'] == 2
    for t, r in lf:
        assert r['out'][1] < N                          # 天井・上限は前進を弱めるだけ(後ろにしない)
    if zone == 0:
        assert res['ceil_applied_in_launch'] == 0 and res['min_fwd_us'] == 250.0


def test_v5_start_with_params_and_block_override():
    """bench のブロックの上書き・基準の設定で launch_boost が 1 にならない(bench.py の関数で)"""
    from jetformer.web import bench
    src = Path(bench.__file__).read_text(encoding='utf-8')
    assert 'launch_boost' in src
    assert 'launch_boost' in bench.SWITCH_KEYS


def test_v3_nan_config_defect_documented():
    """LaunchBoost._clip は NaN を通す(念のための切り詰めが NaN に効かない)。drive.py の入口は NaN を断るので、今の道からは届かない"""
    b = A.LaunchBoost(N)
    b.human()
    inp = A.LaunchIn(us=float('nan'), v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=30.0, ttc5=(30.0,) * 5, tof_ok=True, veh_ok=True)
    out, code = b.step(N + 100.0, 0.0, 0.0, inp)
    res = {'out_us_nan': out, 'cfg': list(b.cfg)}
    b2 = A.LaunchBoost(N)
    b2.human()
    inp2 = inp._replace(us=250.0, max_s=float('nan'))
    n = 0
    for i in range(300):
        o, c = b2.step(N + 100.0, 0.0, i / 30.0, inp2)
        if o is None:
            break
        n += 1
    res['frames_with_max_s_nan'] = n
    note('v3_nan_cfg', res)
    assert out is not None and math.isnan(out)
    assert n == 300            # ② で終わらない(止まったまま 10 秒出し続ける)


@pytest.mark.parametrize('kind', ['invalid', 'lowconf', 'bumper_rows'])
def test_v3_tof_blind_still_starts(monkeypatch, tmp_path, kind):
    """ToF の値は新しいが、前の物が見えない形(全部が無効・信頼度が低い・バンパーの行(28 行より下)だけに写る)。
    ガードの x_free は 9 m(空いている)になり、発進の前進は始まる(ToF に頼る限りのすき)"""
    if kind == 'invalid':
        t = np.full((42, 54), (4 << 13) | 0x1FFF, np.uint16)
    elif kind == 'lowconf':
        t = L.TOF_FREE.copy()
        t[4:28, 20:34] = (1 << 13) | 300
    else:
        t = L.TOF_FREE.copy()
        t[28:42, 20:34] = (4 << 13) | 150
    frames, imu = L.launch_scene(tof=lambda s: t)
    d, pca, rec = L.run(drive, monkeypatch, tmp_path, dict(L.PARAMS, launch_boost=1), frames, imu)
    first = next(r for _, r in rec.rows if r['select'] and not r['hold'])
    note('v3_tof_blind_%s' % kind, {'count': d.st['stats']['launch_boosts'], 'ttc_first_ai': first['ttc']})
    assert d.st['stats']['launch_boosts'] >= 1
