"""直す 7-1 回目・係 tools(2026-10-27。反証 7-1 回目の指摘 w1-tools-r7-1-tools-1)の試験。夜の道具だけ
(⑤ experiments/20261026_fix/diag_launch/collect.py・③ tools/standing_3lap.py・④ tools/quick_eval_by_params.py)。
走行の動き(PCA9685 に書く値)・既定値は変えていない(drive.py・assist.py は触っていない)。

  w1-tools-r7-1-tools-1  両方 ON の日(rec_back_guard = rec_hold_keep = 1。11/3 は全部の回)は、リカバリー中止が CH3 を人側にするまで
    解けない。中止で止まった車をその場で人側 → AI 側に戻して走り直すと(手順書 7 章の中立止めの行のとおり)、新しい AI の区間ができ、
    R1 の手前から出たその発進が ⑤ の ON/OFF の起点までの中央値に入って「満たさない」が「使ってよい」に変わった。③ もその走り直しを
    「ok・2 回目の発進」とし、前に衝突が無ければ counted(④ の止まってから 3 周にも入る)
    → 同じ本の中で一つ前の AI の区間がリカバリー中止で終わった発進に after_abort の印(standing_3lap.abort_resumes)。⑤ は途中で
    変えた本・試しと同じに比べから外し(ON の最初の周の衝突は数える)、名前の行を出す。③ は counted = False。④ は印と動く → A1

どの試験も「直す前の写し(tests/fixtures/fix7_tools_before/*.orig。sha256 を照らす)で指摘が起きる」か「その場面のほかは直す前と
出力(JSON・標準出力・表)が同じ」を見る。作り物の記録だけ(tmp)。本物の data/raw・labels・装置・GPU は使わない。
作り物のコースは反証の台本 experiments/20261026_fix/fix7/review_tools_r1/probes/gen_course.py と同じ作り方(距離で曲がりを決める
ので、コースの途中で止まり、その場から走り直す記録が作れる)。
環境変数 JF_FIX7_COLLECT・JF_FIX7_S3・JF_FIX7_QP で、試す版を差し替えられる(わざと壊した写しで試験が落ちるかを見るとき)。
2026-10-27 fix7 r2(直す 7-2 回目・係 tools): 印に kind を足し、ON の走り直しの空転・最初の周の欠けを比べの条件に数え、組の行に空転の比
を出すようにしたので、その欄・文に合わせた(見ることは同じ。中立止め・カメラの途切れの形の試験は tests/test_fix7_tools_r2.py)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix7_tools.py
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

JF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
from test_fix3_r1_2 import load_mod  # noqa: E402
import label_events as le  # noqa: E402

FIX = JF / 'tests' / 'fixtures' / 'fix7_tools_before'
OLD = FIX / 'collect.py.orig'
NEW = Path(os.environ.get('JF_FIX7_COLLECT') or (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'))
IG = ['--ignore-gain']
IGF = ['--ignore-gain', '--first-only']


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fixture_is_the_version_before():
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sorted(sums) == ['collect.py.orig', 'quick_eval_by_params.py.orig', 'standing_3lap.py.orig']
    for name, h in sums.items():
        assert _sha(FIX / name) == h, name
    assert 'after_abort' not in OLD.read_text(encoding='utf-8')
    assert 'def abort_resumes' not in (FIX / 'standing_3lap.py.orig').read_text(encoding='utf-8')
    assert 'after_abort' not in (FIX / 'quick_eval_by_params.py.orig').read_text(encoding='utf-8')


# ================================================================== ③④ の版(今・直す前)
def _pair(p3, pq, tag):
    """(standing_3lap, quick_eval_by_params) の組をファイルから読む。④ は渡した ③ を standing_3lap として読む"""
    s3m = load_mod('standing_3lap_%s' % tag, p3)
    saved = sys.modules.get('standing_3lap')
    sys.modules['standing_3lap'] = s3m
    try:
        qpm = load_mod('qp_%s' % tag, pq)
    finally:
        if saved is not None:
            sys.modules['standing_3lap'] = saved
        else:
            sys.modules.pop('standing_3lap', None)
    return s3m, qpm


@pytest.fixture(scope='module')
def new():
    """今の ③④(JF_FIX7_S3・JF_FIX7_QP があれば、その写し)"""
    if not os.environ.get('JF_FIX7_S3') and not os.environ.get('JF_FIX7_QP'):
        import standing_3lap as s3
        import quick_eval_by_params as qp
        return s3, qp
    return _pair(os.environ.get('JF_FIX7_S3') or (JF / 'tools' / 'standing_3lap.py'),
                 os.environ.get('JF_FIX7_QP') or (JF / 'tools' / 'quick_eval_by_params.py'), 'f7new')


@pytest.fixture(scope='module')
def old():
    """直す前の ③④ の写し"""
    return _pair(FIX / 'standing_3lap.py.orig', FIX / 'quick_eval_by_params.py.orig', 'f7old')


# ================================================================== 作り物のコース(gen_course と同じ)
REF = (1441.0, 1496.0)
T0_US = 1_800_000_000_000_000
G = 9.80665
KV = 0.5e-3
LAP_M = 18.0
START_TO_R1 = 5.0
SEGS = [(1.8, -100.0), (0.2, 0.0), (3.0, 60.0), (3.0, 0.0), (4.0, 45.0), (3.0, 0.0), (3.0, 60.0)]
EDGES = np.cumsum([0.0] + [L for L, _ in SEGS])
KAPS = np.array([k for _, k in SEGS])


def kappa(s):
    """距離 s [m](置いた場所 = 0。R1 の起点 = 5 m・23 m …)のヨー [deg/m]"""
    x = np.mod(np.asarray(s, float) - START_TO_R1, LAP_M)
    i = np.clip(np.searchsorted(EDGES, x, side='right') - 1, 0, len(KAPS) - 1)
    return KAPS[i]


def in_iv(t, ivs):
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for a, b in ivs or ():
        m |= (t >= a - 1e-9) & (t < b - 1e-9)
    return m


def write_log(raw, log, dur, vkeys, ai_iv, params, description, stall=(), giveup=(), launch=(), hits=(), events=(),
              hold=(), human_cmd=(), run='runB', scene_s=20.0):
    """vkeys [(秒, m/s)]・ai_iv CH3 が AI 側の間・stall カメラが止まった間・giveup リカバリー中止(assist 3・中立)・
    launch [(始め, 長さ)] 発進の前進・hits [(秒, 水平 G)]・hold 中立止め(AI_CMD の hold・assist 0)のフレームの間・
    human_cmd AI_CMD の select が先に人側になった間(車両の CH3 はまだ AI 側。assist 0)"""
    d = Path(raw) / log
    d.mkdir(parents=True)
    meta = {'description': description, 'tags': ['ai-drive'], 'mode_id': 5,
            'drive': {'run': run, 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1], 'dry_run': False, 'params': params}}
    (d / 'log.json').write_text(json.dumps(meta), encoding='utf-8')
    tk, vk = np.array([x[0] for x in vkeys], float), np.array([x[1] for x in vkeys], float)
    tg = np.arange(0.0, dur + 0.5, 0.001)
    vg = np.interp(tg, tk, vk)
    sg = np.concatenate([[0.0], np.cumsum(0.5 * (vg[1:] + vg[:-1]) * np.diff(tg))])
    ag = np.gradient(vg, tg)
    ft = np.arange(0, dur, 1 / 30.0)
    vt = np.arange(0, dur, 0.01)
    it = np.arange(0, dur, 0.01) + 0.003
    v_v = np.interp(vt, tg, vg)
    s_i, v_i = np.interp(it, tg, sg), np.interp(it, tg, vg)
    gz = kappa(s_i) * v_i
    acc = np.zeros((len(it), 3))
    acc[:, 2] = G
    acc[:, 1] = np.interp(it, tg, ag)
    for th, g in hits:
        k = int(np.searchsorted(it, th))
        acc[k:k + 3, 1] += [-0.5 * g * G, -g * G, -0.3 * g * G]
    e = -v_v / KV
    sel_v = in_iv(vt, ai_iv)
    keep = ~in_iv(ft, stall)
    sel_f = in_iv(ft, ai_iv) & ~in_iv(ft, human_cmd)
    s_f = np.interp(ft + 0.1, tg, sg)
    kap = np.radians(kappa(s_f))
    steer = REF[0] + np.clip(kap * 1e4 / 40.0, -480, 480)
    thr = np.full(len(ft), REF[1] - 200.0)
    assist = np.zeros(len(ft), np.int8)
    gu = in_iv(ft, giveup) & sel_f
    assist[gu] = 3
    steer[gu] = REF[0]
    thr[gu] = REF[1]
    hm = in_iv(ft, hold)
    assist[hm] = 0
    steer[hm] = REF[0]
    thr[hm] = REF[1]
    lon = np.zeros(len(ft), bool)
    lend = np.zeros(len(ft), np.int8)
    ev_lines = []
    for a0, ln in launch:
        m = (ft >= a0 - 1e-9) & (ft < a0 + ln - 1e-9)
        lon |= m
        assist[m] = 5
        thr[m] = REF[1] - 250.0
        k_end = int(np.searchsorted(ft, a0 + ln - 1e-9))
        if k_end < len(lend):
            lend[k_end] = 1
        ev_lines.append({'kind': 'launch_boost', 'event': 'start', 'us': 250.0, 'cam_t_us': int(T0_US + round(a0 * 1e6)),
                         't_us': int(T0_US + round(a0 * 1e6)) + 500})
    ev_lines += list(events)
    if ev_lines:
        (d / 'events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in ev_lines), encoding='utf-8')
    out = np.stack([steer, thr], axis=1)
    us = lambda x: (T0_US + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
    nsc = int(np.ceil(dur / scene_s))
    for si in range(nsc):
        a, b = si * scene_s, min((si + 1) * scene_s, dur)
        s = d / ('%s-%04d' % (log, si + 1))
        s.mkdir()
        fm = (ft >= a) & (ft < b) & keep
        vm, im = (vt >= a) & (vt < b), (it >= a) & (it < b)
        n = int(fm.sum())
        np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), raw=out[fm].astype(np.float32), out=out[fm].astype(np.float32),
                 gains=np.tile([1.0, float(params.get('throttle_gain', 0.9))], (n, 1)).astype(np.float32),
                 trim=np.zeros(n, np.float32), lookahead=np.full(n, 3, np.int16), hold=hm[fm], select=sel_f[fm],
                 dry_run=np.zeros(n, bool), assist=assist[fm], model_out=out[fm].astype(np.float32),
                 launch_on=lon[fm], launch_end=lend[fm])
        nv = int(vm.sum())
        np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=np.tile([1441, 1496, 2000], (nv, 1)).astype(np.int32),
                 valid=np.ones((nv, 3), bool), select=sel_v[vm], erpm=np.round(e[vm]).astype(np.int32),
                 board_t_us=(us(vt[vm]) - T0_US + 10_000_000))
        gyro = np.zeros((int(im.sum()), 3))
        gyro[:, 2] = gz[im]
        np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32), gyro=gyro.astype(np.float32))
        np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * n))
    return d


def launch_keys(t_ai, d_move, ramp, v=2.0, until=None):
    k = [(0.0, 0.0), (t_ai + d_move, 0.0), (t_ai + d_move + ramp, v)]
    if until is not None:
        k.append((until, v))
    return k


def time_at_distance(vkeys, s_target, dur):
    tk, vk = np.array([x[0] for x in vkeys], float), np.array([x[1] for x in vkeys], float)
    tg = np.arange(0.0, dur + 0.5, 0.001)
    vg = np.interp(tg, tk, vk)
    sg = np.concatenate([[0.0], np.cumsum(0.5 * (vg[1:] + vg[:-1]) * np.diff(tg))])
    return float(np.interp(s_target, sg, tg))


PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'


class Day(object):
    """作り物の 11/3 の回 3(反証の台本 probe_abort_resume.py と同じ): OFF 3-1・3-3・3-5・3-7(起点まで 3.013〜3.043 秒)・
    ON 3-2・3-4・3-6(2.813・2.863・2.943 秒)・3-8(ON。1 回目 2.973 秒。形は case)"""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.raw = self.tmp / 'raw'
        self.raw.mkdir(parents=True)
        self.bench = self.tmp / 'bench.json'
        self.bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                          'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                                     for b in range(12, 20)]}]}), encoding='utf-8')
        self.n = 0

    def add(self, block, prm, d_move, ramp, **kw):
        log = '20261103-13%02d00' % (self.n * 5)
        self.n += 1
        dur = kw.pop('dur', 30.0)
        vk = kw.pop('vk', None) or launch_keys(1.0, d_move, ramp, until=dur)
        ai = kw.pop('ai', None) or [(1.0, dur - 2.0)]
        launch = kw.pop('launch', [(1.0, 0.5)] if prm.get('launch_boost') else [])
        write_log(self.raw, log, dur, vk, ai, prm, DESC % (self.n, block), launch=launch, **kw)
        return log


def abort_book(D, block, prm, stop_before_r1=3.5, resume=True, hit=None, resolve=False, tail=None, bounce=None, hit_before=None,
               res_len=None, **kw):
    """3-8 の形(反証の B): 1 回目の発進(1.0 秒)→ 2 周目の R1(23 m)の (stop_before_r1 + 1) m 手前でカメラが 0.8 秒止まる → 中立の
    惰性で R1 の stop_before_r1 m 手前に止まる → リカバリー中止(中立)2 秒 → 人側 1.0 秒 → その場で AI 側(走り直し。ON なら発進の
    前進がまた出る)→ 3 周。resume = False なら人側にしたまま(A)。hit = (走り直しからの秒, G)。resolve = 中止のあと(OFF の日のように)
    中立止めで解けてモデルが 0.5 秒前へ出したあとに人側にした(中止で終わらない)。tail = 'hold'(区間の最後の 3 フレームが中立止め)/
    'human'(AI_CMD の select が車両の CH3 より 2 フレーム先に人側)。bounce = 走り直しを (秒) だけで人側にして、もう一度 AI 側"""
    base = launch_keys(1.0, 0.31, 0.3)
    t_st = time_at_distance(base + [(200.0, 2.0)], 23.0 - stop_before_r1 - 1.0, 200.0)
    t_stop = t_st + 1.0
    t_flip = t_stop + 2.0
    t_res = t_flip + 1.0
    on = bool(prm.get('launch_boost'))
    giveup = [(t_st + 0.8, t_flip)]
    hold, human_cmd = [], []
    if resolve:
        giveup = [(t_st + 0.8, t_flip - 0.6)]
    if tail == 'hold':
        hold = [(t_flip - 0.1, t_flip)]
    elif tail == 'human':
        human_cmd = [(t_flip - 0.07, t_flip)]
    if resume and res_len:                            # 走り直しを res_len 秒で人側にして、そのまま止めた(R1 に届かない)
        dur = t_res + res_len + 2.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 1.0), (t_res + res_len, 1.0),
                     (t_res + res_len + 1.0, 0.0), (dur, 0.0)]
        ai = [(1.0, t_flip), (t_res, t_res + res_len)]
        launch = [(1.0, 0.5), (t_res, 0.5)] if on else []
    elif resume:
        dur = t_res + 32.0
        if bounce:
            t2 = t_res + bounce + 1.0                 # 人側 1.0 秒のあと、もう一度 AI 側
            vk = base + [(t_st, 2.0), (t_stop, 0.0), (t2 + 0.15, 0.0), (t2 + 0.45, 2.0), (dur, 2.0)]
            ai = [(1.0, t_flip), (t_res, t_res + bounce), (t2, dur - 1.0)]
            launch = ([(1.0, 0.5), (t_res, 0.5), (t2, 0.5)] if on else [])
        else:
            vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
            ai = [(1.0, t_flip), (t_res, dur - 1.0)]
            launch = [(1.0, 0.5), (t_res, 0.5)] if on else []
        if resolve:                                   # 中止のあとモデルが前へ出した(0.4 m ほど)
            vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_flip - 0.6, 0.0), (t_flip - 0.3, 1.0), (t_flip, 0.0),
                         (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
    else:
        dur = t_flip + 3.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (dur, 0.0)]
        ai = [(1.0, t_flip)]
        launch = [(1.0, 0.5)] if on else []
    ev = [{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8, 'run': 'runB',
           'cam_epoch_us': int(T0_US + round((t_st + 0.8) * 1e6)), 't_us': int(T0_US + round((t_st + 0.8) * 1e6))}]
    hits = ([(t_res + hit[0], hit[1])] if hit else []) + ([tuple(hit_before)] if hit_before else [])
    log = D.add(block, prm, 0.31, 0.3, dur=dur, vk=vk, ai=ai, launch=launch, stall=[(t_st, t_st + 0.8)], giveup=giveup,
                events=ev, hits=hits, hold=hold, human_cmd=human_cmd, **kw)
    return log, {'t_st': t_st, 't_flip': t_flip, 't_res': t_res}


def round3(tmp, case='B', stop=3.5, b38=ON, **kw):
    """回 3。case: 'A' = 3-8 は走り直さない / 'B' = その場で走り直す / 'N' = 3-8 も中止なし(ふつうの 1 周)"""
    D = Day(tmp)
    D.add(12, PRM, 0.20, 0.6)
    D.add(13, ON, 0.15, 0.3)
    D.add(14, PRM, 0.21, 0.6)
    D.add(15, ON, 0.20, 0.3)
    D.add(16, PRM, 0.22, 0.6)
    D.add(17, ON, 0.28, 0.3)
    D.add(18, PRM, 0.23, 0.6)
    if case == 'N':
        return D, D.add(19, b38, 0.31, 0.3), None
    log, tt = abort_book(D, 19, b38, stop_before_r1=stop, resume=(case != 'A'), **kw)
    return D, log, tt


# ================================================================== ⑤ の流し方
def _n(name):
    return name.replace('-', '_').replace('.', '_')


def runc(D, name, path, args=()):
    """⑤ collect.main を作り物の日で流す。JF_FIX7_S3 があれば、その ③ を standing_3lap として読ませる"""
    m = load_mod(name, path)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('%s.json' % name)
    buf = io.StringIO()
    saved = sys.modules.get('standing_3lap')
    if os.environ.get('JF_FIX7_S3'):
        sys.modules['standing_3lap'] = load_mod('standing_3lap_f7env', os.environ['JF_FIX7_S3'])
    try:
        with contextlib.redirect_stdout(buf):
            m.main(['--day', '20261103', '--out', str(out)] + list(args))
    finally:
        if os.environ.get('JF_FIX7_S3'):
            if saved is not None:
                sys.modules['standing_3lap'] = saved
            else:
                sys.modules.pop('standing_3lap', None)
    txt = out.read_text(encoding='utf-8')
    return json.loads(txt), buf.getvalue(), txt


def both(D, name, args=()):
    """(今, 直す前) の (JSON, 標準出力, JSON の文)"""
    return runc(D, 'n7_' + _n(name), NEW, args), runc(D, 'o7_' + _n(name), OLD, args)


def no_path(out):
    return '\n'.join(x for x in out.splitlines() if not x.startswith('出力 '))


def g0(j):
    assert len(j['summary']) == 1
    return j['summary'][0]


def rows_of(j, log):
    return [r for r in j['launches'] if r['log'] == log]


def strip_abort(j, out):
    """今の ⑤ の出力から、fix7 で足した「中止のあとの走り直し」の印・欄・行を除いた形(比べから外したことのほかが同じかを見る)"""
    j = copy.deepcopy(j)
    for r in j['launches']:
        r.pop('after_abort', None)
    for g in j.get('summary') or []:
        g.pop('abort_resume_excluded', None)
        g.pop('abort_resume_on_collisions', None)
        g.pop('abort_resume_slip', None)              # fix7 r2(走り直しの ON の空転。あるときだけ)
        g.pop('abort_resume_slip_unmeasured', None)
    lines = []
    for x in out.splitlines():
        if x.startswith('  リカバリー中止のあとの走り直し ') or x.startswith('  → リカバリー中止のあとの走り直しの ON'):     # r2: 空転の行
            continue
        k = x.find(' 中止のあとの走り直し(中止 ')
        lines.append(x[:k] if k >= 0 else x)
    return j, '\n'.join(lines)


# ================================================================== ⑤ 指摘の再現と直し
@pytest.mark.parametrize('stop', [3.5, 4.5])
def test_c5_before_resume_turns_to_use_ok(tmp_path, stop):
    """直す前: その場で走り直した 3-8 の 2 回目の発進(R1 の stop m 手前から)が ON の中央値に入り、−0.165 秒「使ってよい」(反証の B)"""
    D, log, _ = round3(tmp_path, 'B', stop)
    jo, oo, _ = runc(D, 'o7_b%d' % int(stop * 10), OLD, IG)
    g = g0(jo)
    assert g['by_launch']['on']['n'] == 5 and g['by_launch']['on']['restarts'] == 1
    assert g['by_launch']['on']['to_r1_median_s'] == 2.863 and g['compare']['d_to_r1_s'] == -0.165
    assert g['compare']['all_ok'] is True and '→ 使ってよい' in oo
    r2 = rows_of(jo, log)[1]
    assert r2['start_no'] == 2 and abs(r2['dist_m'] - (stop + 0.03)) < 0.02 and 'after_abort' not in r2


@pytest.mark.parametrize('stop', [0.5, 1.5, 2.5, 3.5, 4.5])
def test_c5_resume_after_abort_is_excluded(tmp_path, stop):
    """今: 走り直しは比べ(起点まで・空転・LH・2 回目以降)から外れ、ON 4 本対 OFF 4 本の −0.125 秒「満たさない」(走り直さない日の A と
    同じ数)。行に after_abort、組に abort_resume_excluded、標準出力に名前の行。ほかの数・欄は直す前と同じ"""
    D, log, tt = round3(tmp_path, 'B', stop)
    (jn, on, _), (jo, oo, _) = both(D, 'b%d' % int(stop * 10), IG)
    g = g0(jn)
    assert g['by_launch']['on']['n'] == 4 and g['by_launch']['on']['restarts'] == 0
    assert g['by_launch']['on']['to_r1_median_s'] == 2.903 and g['compare']['d_to_r1_s'] == -0.125
    assert g['compare']['all_ok'] is False and g['compare']['checks']['to_r1'] is False
    assert '→ 満たさない' in on and '→ 使ってよい' not in on
    r1, r2 = rows_of(jn, log)
    assert 'after_abort' not in r1
    ab = r2['after_abort']
    assert set(ab) == {'kind', 'abort_s', 'ai_end_s', 'human_s'} and ab['kind'] == 'abort'     # fix7 r2: kind を足した
    assert abs(ab['abort_s'] - (tt['t_st'] + 0.8)) < 0.04 and abs(ab['ai_end_s'] - tt['t_flip']) < 0.04
    assert abs(ab['human_s'] - 1.0) < 0.07
    # fix7 r2: ON の走り直しの空転の比(比べの条件に数える)を後ろに出す
    assert len(g['abort_resume_excluded']) == 1 and g['abort_resume_excluded'][0].startswith(
        '%s@%.2f(on・中止 %s 秒・人側 %s 秒・起点まで %.3f 秒・空転の比' % (log, r2['t_ai'], ab['abort_s'], ab['human_s'], r2['to_r1_s']))
    assert 'not_reached' not in g
    assert 'abort_resume_on_collisions' not in g
    line = [x for x in on.splitlines() if x.startswith('  リカバリー中止のあとの走り直し 1 本')]
    assert len(line) == 1 and log in line[0] and 'ON の組にも OFF の組にも入れない' in line[0]
    row_line = [x for x in on.splitlines() if x.startswith(log) and '中止のあとの走り直し(中止 ' in x]
    assert len(row_line) == 1 and '比べに入れない' in row_line[0]
    # 直す前の行(after_abort のほか)は同じ
    assert [{k: v for k, v in r.items() if k != 'after_abort'} for r in jn['launches']] == jo['launches']


@pytest.mark.parametrize('stop', [3.5, 1.5])
def test_c5_same_as_no_resume_day(tmp_path, stop):
    """今の走り直しの日の比べ(compare・by_launch)は、走り直さない日(A。直す前も今も同じ)と同じ"""
    Da, _, _ = round3(tmp_path / 'a', 'A', stop)
    Db, _, _ = round3(tmp_path / 'b', 'B', stop)
    ja, _, _ = runc(Da, 'n7_a_%d' % int(stop * 10), NEW, IG)
    jb, _, _ = runc(Db, 'n7_bb_%d' % int(stop * 10), NEW, IG)
    assert g0(ja)['by_launch'] == g0(jb)['by_launch']
    cb = copy.deepcopy(g0(jb)['compare'])
    if stop == 1.5:      # fix7 r2: R1 の 1.5 m 手前からの走り直し(ON)は空転の比が出ない(窓が短い)→ slip_all_measured(fix7 の前と同じ)
        assert cb['checks'].pop('slip_all_measured') is False
    assert g0(ja)['compare'] == cb


@pytest.mark.parametrize('args', [[], IG, IGF])
@pytest.mark.parametrize('case', ['A', 'N'])
def test_c5_no_resume_is_bit_identical(tmp_path, case, args):
    """中止があっても走り直さない本(A)・中止の無い日(N)は、JSON・標準出力とも直す前とビット単位で同じ"""
    D, _, _ = round3(tmp_path, case)
    (jn, on, tn), (jo, oo, to) = both(D, case + str(len(args)), args)
    assert tn == to
    assert no_path(on) == no_path(oo)


def test_c5_first_only_summary_unchanged(tmp_path):
    """--first-only: 走り直しは 2 回目の発進なので、前から比べに入らない。まとめは直す前と同じ(行に印が付くだけ)"""
    D, log, _ = round3(tmp_path, 'B', 3.5)
    (jn, on, _), (jo, oo, _) = both(D, 'bf', IGF)
    assert jn['summary'] == jo['summary']
    js, os_ = strip_abort(jn, on)
    assert js == jo and no_path(os_) == no_path(oo)
    assert 'after_abort' in rows_of(jn, log)[1]


def test_c5_resume_strip_equals_before_except_comparison(tmp_path):
    """足した印・欄・行を除くと、変わるのは ON の組の数(n・中央値・restarts)と比べ(compare)だけ"""
    D, log, _ = round3(tmp_path, 'B', 3.5)
    (jn, on, _), (jo, oo, _) = both(D, 'bs', IG)
    js, os_ = strip_abort(jn, on)
    assert js['launches'] == jo['launches']
    gn, go = g0(js), g0(jo)
    assert gn['by_launch']['off'] == go['by_launch']['off']
    assert {k: v for k, v in gn.items() if k not in ('by_launch', 'compare')} == \
        {k: v for k, v in go.items() if k not in ('by_launch', 'compare')}
    changed = [x for x in zip(no_path(os_).splitlines(), no_path(oo).splitlines()) if x[0] != x[1]]
    assert len(no_path(os_).splitlines()) == len(no_path(oo).splitlines())
    assert all(a.startswith('  on ') or a.startswith('  ON − OFF') for a, _ in changed) and len(changed) == 2


def test_c5_on_resume_first_lap_collision_still_counts(tmp_path):
    """ON の走り直しの最初の周の衝突は「最初の周の衝突 0」に数える(衝突は止める条件)。直す前も「満たさない」"""
    D, log, _ = round3(tmp_path, 'B', 3.5, hit=(1.2, 6.0))
    (jn, on, _), (jo, oo, _) = both(D, 'bh', IG)
    g = g0(jn)
    assert g['abort_resume_on_collisions'] == 1
    assert g['compare']['checks']['first_lap_collisions_0'] is False and g['compare']['all_ok'] is False
    assert g['by_launch']['on']['collisions_first_lap'] == 0 and g['by_launch']['on']['n'] == 4
    assert 'うち ON の発進の最初の周の衝突 1' in on
    assert g0(jo)['compare']['checks']['first_lap_collisions_0'] is False


def test_c5_off_resume_is_excluded_from_off(tmp_path):
    """OFF の本の走り直しも OFF の組から外す(OFF の中央値が R1 の手前からの発進で速く出ない)。衝突は数えない(ON だけ)"""
    D, log, _ = round3(tmp_path, 'B', 3.5, b38=PRM, hit=(1.2, 6.0))
    (jn, on, _), (jo, oo, _) = both(D, 'boff', IG)
    gn, go = g0(jn), g0(jo)
    assert go['by_launch']['off']['n'] == 6 and go['by_launch']['off']['restarts'] == 1
    assert gn['by_launch']['off']['n'] == 5 and gn['by_launch']['off']['restarts'] == 0
    assert gn['abort_resume_excluded'][0].startswith('%s@' % log) and '(off・' in gn['abort_resume_excluded'][0]
    assert 'abort_resume_on_collisions' not in gn


def test_c5_resolved_abort_is_not_marked(tmp_path):
    """中止のあと(中立止めで解けて)モデルが前へ出してから人側にした区間は「中止で終わった」でない → 印を付けない(前と同じ)"""
    D, log, _ = round3(tmp_path, 'B', 3.5, resolve=True)
    (jn, on, tn), (jo, oo, to) = both(D, 'bres', IG)
    assert tn == to and no_path(on) == no_path(oo)
    assert all('after_abort' not in r for r in jn['launches'])


@pytest.mark.parametrize('tail', ['hold', 'human'])
def test_c5_abort_end_with_hold_or_human_frames(tmp_path, tail):
    """区間の最後のフレームが中立止め(assist 0)・AI_CMD が先に人側(assist 0)でも、中止で終わった区間と見る"""
    D, log, _ = round3(tmp_path, 'B', 3.5, tail=tail)
    jn, on, _ = runc(D, 'n7_tail_' + tail, NEW, IG)
    r2 = rows_of(jn, log)[1]
    assert 'after_abort' in r2
    assert g0(jn)['compare']['all_ok'] is False and g0(jn)['by_launch']['on']['n'] == 4


def test_c5_bounce_resume_is_carried(tmp_path):
    """走り直しを 0.3 秒(0.5 m も進まない)で人側にして、もう一度 AI 側: どちらの発進も印(2 つ目は carried 1)。0.3 秒の区間は
    R1 に届かないので前と同じ「R1 に届かなかった発進(on)」(満たさない)に入り、その行に「うちリカバリー中止のあとの走り直し 1」"""
    D, log, _ = round3(tmp_path, 'B', 3.5, bounce=0.3)
    (jn, on, _), (jo, oo, _) = both(D, 'bounce', IG)
    rs = rows_of(jn, log)
    assert [r['start_no'] for r in rs] == [1, 2, 3]
    assert 'after_abort' not in rs[0]
    assert 'carried' not in rs[1]['after_abort'] and rs[2]['after_abort']['carried'] == 1
    assert rs[1]['after_abort']['abort_s'] == rs[2]['after_abort']['abort_s']
    assert rs[1]['to_r1_s'] is None and rs[2]['to_r1_s'] is not None
    g, go = g0(jn), g0(jo)
    assert g['by_launch']['on']['n'] == 4 and g['compare']['all_ok'] is False and len(g['abort_resume_excluded']) == 1
    assert g['abort_resume_excluded'][0].startswith('%s@%.2f(' % (log, rs[2]['t_ai']))
    assert g['not_reached']['on']['after_abort'] == ['%s@%.2f' % (log, rs[1]['t_ai'])]
    assert {k: v for k, v in g['not_reached']['on'].items() if k != 'after_abort'} == go['not_reached']['on']
    assert g['compare']['checks']['on_all_reached_r1'] is False and go['compare']['checks']['on_all_reached_r1'] is False
    assert 'うちリカバリー中止のあとの走り直し 1(%s@%.2f)' % (log, rs[1]['t_ai']) in on


@pytest.mark.parametrize('b38', ['on', 'off'])
def test_c5_resume_not_reaching_r1_is_as_before(tmp_path, b38):
    """R1 に届かなかった走り直し(1.0 秒で人側にして止めた)は前と同じ「R1 に届かなかった発進」(ON なら満たさない)。足すのは行の後ろの
    「うちリカバリー中止のあとの走り直し 1」と not_reached の after_abort と発進の行の印だけ"""
    D, log, _ = round3(tmp_path, 'B', 3.5, b38=ON if b38 == 'on' else PRM, res_len=1.0)
    (jn, on, _), (jo, oo, _) = both(D, 'nr_' + b38, IG)
    rs = rows_of(jn, log)
    assert len(rs) == 2 and rs[1]['to_r1_s'] is None and 'after_abort' in rs[1]
    g, go = g0(jn), g0(jo)
    assert g['not_reached'][b38]['after_abort'] == ['%s@%.2f' % (log, rs[1]['t_ai'])]
    g2 = copy.deepcopy(g)
    del g2['not_reached'][b38]['after_abort']
    assert g2 == go and 'abort_resume_excluded' not in g
    assert any(x.startswith(log) and x.endswith('中止のあとの走り直し(中止 %s 秒・人側 %s 秒。R1 に届かなかった発進に数える)' % (
        rs[1]['after_abort']['abort_s'], rs[1]['after_abort']['human_s'])) for x in on.splitlines())
    if b38 == 'on':
        assert g['compare']['checks']['on_all_reached_r1'] is False and g['compare']['all_ok'] is False
    js, os_ = strip_abort(jn, on)
    for gg in js['summary']:
        gg['not_reached'][b38].pop('after_abort')
    assert js == jo
    lines_n = [x.replace('。うちリカバリー中止のあとの走り直し 1(%s@%.2f)' % (log, rs[1]['t_ai']), '') for x in no_path(os_).splitlines()]
    assert lines_n == no_path(oo).splitlines()


def test_c5_skip_still_works(tmp_path):
    """--skip 記録名 で 3-8 を外すと、走り直しも 1 回目も外れる(前と同じ)"""
    D, log, _ = round3(tmp_path, 'B', 3.5)
    (jn, on, _), (jo, oo, _) = both(D, 'bskip', IG + ['--skip', log])
    assert g0(jn) == g0(jo) and jn['summary_skipped'] == jo['summary_skipped']


def _write_labels(raw, lab, log):
    info, z = le.label_log(log, raw)
    lab.mkdir(exist_ok=True)
    (lab / (log + '.json')).write_text(json.dumps(info, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                                       encoding='utf-8')
    np.savez(lab / (log + '.npz'), **z)


def test_c5_default_days_are_bit_identical(tmp_path):
    """引数なし(10/23・10/24 を data/labels で読む)は、中止のあとに走り直した本があっても直す前とビット単位で同じ"""
    D = Day(tmp_path / 'd')
    D.add(12, PRM, 0.20, 0.6)
    log, _ = abort_book(D, 19, ON, 3.5)
    raw, lab, out = D.raw, tmp_path / 'lab', tmp_path / 'out'
    names = []
    for i, lg in enumerate(sorted(p.name for p in raw.iterdir())):
        nm = '20261023-1%d0000' % (3 + i)
        (raw / lg).rename(raw / nm)
        for sc in sorted((raw / nm).iterdir()):
            if sc.is_dir():
                sc.rename(raw / nm / sc.name.replace(lg, nm))
        _write_labels(raw, lab, nm)
        names.append(nm)
    res = []
    for name, path in (('n7_default', NEW), ('o7_default', OLD)):
        m = load_mod(name, path)
        m.RAW, m.LAB, m.OUT = raw, lab, out / name
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main([])
        res.append(((out / name / 'launches.json').read_bytes(), no_path(buf.getvalue())))
    assert res[0] == res[1]
    j = json.loads(res[0][0])
    assert len([r for r in j['launches'] if r['log'] == names[1]]) == 2       # 走り直しの発進の行はある(印は付けない)


# ================================================================== ③④
def run_s3(mods, D):
    s3m, qpm = mods
    logs, _ = s3m.day_logs('20261103', D.raw)
    R = s3m.run(logs, D.raw, D.tmp / 'nolab', None)
    md = s3m.to_markdown(R)
    qmd, runs, groups = qpm.to_markdown(R, '20261103', None, ('gain',))
    return R, md, qmd, runs, groups


def st_of(R, log):
    return [x for x in R if x['log'] == log][0]['standing']


def test_s3_before_counts_resume(tmp_path, old):
    """直す前: 走り直しを「ok・2 回目の発進」で counted = True とし、④ の発進 250 の組の止まってから 3 周に入れた(反証)"""
    D, log, _ = round3(tmp_path, 'B', 3.5)
    R, md, qmd, runs, groups = run_s3(old, D)
    s = st_of(R, log)[1]
    assert s['start_no'] == 2 and s['valid'] and s['counted'] and 'after_abort' not in s
    g = [x for x in groups if '発進 250' in x['text']][0]
    assert g['standing_n'] == 1 and abs(g['standing_best_s'] - 27.12) < 0.01


@pytest.mark.parametrize('stop', [3.5, 1.5])
def test_s3_resume_after_abort_not_counted(tmp_path, new, stop):
    """今: 走り直しは counted = False・after_abort。③ の表に「数えない(リカバリー中止のあとの走り直し…)」と注の 1 行。④ は止まって
    から 3 周 0 本・数えない 1・1 章の印「数えない発進 1(リカバリー中止のあとの走り直し)」・動く → A1 から外して 2 章の下に 1 行"""
    s3m, qpm = new
    D, log, tt = round3(tmp_path, 'B', stop)
    R, md, qmd, runs, groups = run_s3(new, D)
    s1, s2 = st_of(R, log)
    assert 'after_abort' not in s1
    assert s2['valid'] and s2['counted'] is False and abs(s2['after_abort']['human_s'] - 1.0) < 0.07
    assert s3m.best_standing(st_of_res(R, log)) is None
    assert '・数えない(リカバリー中止のあとの走り直し。中止 %.2f 秒・人側 1.0 秒)' % s2['after_abort']['abort_s'] in md
    assert s3m.ABORT_NOTE in md
    g = [x for x in groups if '発進 250' in x['text']][0]
    assert g['standing_n'] == 0 and g['standing_excluded_n'] == 1
    assert g['launch_to_r1_abort_excluded'] == ['%s@%.2f' % (log, s2['t_ai'])]
    run = [x for x in runs if x['log'] == log][0]
    assert '数えない発進 1(リカバリー中止のあとの走り直し)' in run['flags']
    assert not any('前に衝突' in f for f in run['flags'])
    assert any(x.startswith('- リカバリー中止のあとの走り直し(') and log in x for x in qmd.splitlines())


def st_of_res(R, log):
    return [x for x in R if x['log'] == log][0]


def test_s3_to_r1_median_without_resume(tmp_path, new):
    """④ の動く → A1 中央値・数は、走り直しを入れない: 今は走り直さない日(A)と同じ値・同じ数(4)"""
    Da, _, _ = round3(tmp_path / 'a', 'A', 3.5)
    Db, _, _ = round3(tmp_path / 'b', 'B', 3.5)
    ga = [x for x in run_s3(new, Da)[4] if '発進 250' in x['text']][0]
    gb = [x for x in run_s3(new, Db)[4] if '発進 250' in x['text']][0]
    assert ga['launch_to_r1_median_s'] == gb['launch_to_r1_median_s'] and ga['launch_n'] == gb['launch_n'] == 4


def test_s3_to_r1_median_changes_only_by_resume(tmp_path, new, old):
    """動く → A1 の数(launch_n)は、直す前は走り直しを入れて 5、今は 4"""
    D, log, _ = round3(tmp_path, 'B', 3.5)
    gn = [x for x in run_s3(new, D)[4] if '発進 250' in x['text']][0]
    go = [x for x in run_s3(old, D)[4] if '発進 250' in x['text']][0]
    assert go['launch_n'] == 5 and gn['launch_n'] == 4


@pytest.mark.parametrize('case', ['A', 'N'])
def test_s3_no_resume_is_identical(tmp_path, new, old, case):
    """走り直さない本・中止の無い日は、③ の結果・表と ④ の表・本・組が直す前と同じ"""
    D, _, _ = round3(tmp_path, case)
    Rn, mdn, qn, rn, gn = run_s3(new, D)
    Ro, mdo, qo, ro, go = run_s3(old, D)
    dump = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)      # noqa: E731
    assert dump(Rn) == dump(Ro) and mdn == mdo and qn == qo and dump(rn) == dump(ro) and dump(gn) == dump(go)


def test_s3_resume_strip_equals_before(tmp_path, new, old):
    """走り直しの本: after_abort と counted のほかの ③ の値は直す前と同じ"""
    D, log, _ = round3(tmp_path, 'B', 3.5)
    Rn = run_s3(new, D)[0]
    Ro = run_s3(old, D)[0]
    for rn, ro in zip(Rn, Ro):
        assert rn['log'] == ro['log']
        sn = [{k: v for k, v in s.items() if k not in ('after_abort', 'counted')} for s in rn.get('standing') or []]
        so = [{k: v for k, v in s.items() if k != 'counted'} for s in ro.get('standing') or []]
        assert sn == so
        assert {k: v for k, v in rn.items() if k != 'standing'} == {k: v for k, v in ro.items() if k != 'standing'}


def test_s3_coll_before_takes_the_flag(tmp_path, new):
    """1 回目の発進のあと(中止の前)に当たった本の走り直しは、④ の印は今まで通り「前に衝突」(中止のあとの方には数えない)。
    ③ の表には両方の「数えない」"""
    D = Day(tmp_path)
    log, tt = abort_book(D, 19, ON, 3.5, hit_before=(6.0, 5.0))
    R, md, qmd, runs, groups = run_s3(new, D)
    s2 = st_of(R, log)[1]
    assert s2['after_abort'] and s2['counted'] is False and s2['coll_before'] == 1 and s2['valid']
    run = [x for x in runs if x['log'] == log][0]
    assert '数えない発進 1(前に衝突)' in run['flags']
    assert not any('リカバリー中止のあとの走り直し' in f for f in run['flags'])
    row = [x for x in md.splitlines() if x.startswith('| ') and log in x and '2 回目の発進' in x][0]
    assert '数えない(この本の前の衝突 1)' in row and '数えない(リカバリー中止のあとの走り直し' in row


def test_abort_resumes_unit(new):
    """abort_resumes だけ: 中止で終わった区間の次の区間に印。中止の無い・解けた・最後のフレームの扱い・続き(carried)"""
    s3m = new[0]
    t = np.arange(0, 20, 0.1)
    ai = (t < 5) | ((t >= 6) & (t < 8)) | (t >= 9)
    a = np.zeros(len(t), np.int8)
    vt = t.copy()
    Df = t * 1.0                                     # 1 m/s で動き続ける(続かない)
    assert s3m.abort_resumes(t, ai, a) == {}
    assert s3m.abort_resumes(t, ai, None) == {}
    a[(t >= 4) & (t < 5)] = 3                        # 1 つ目の区間は中止で終わる
    r = s3m.abort_resumes(t, ai, a, vt=vt, Df=Df)
    i1 = int(np.nonzero(t >= 6)[0][0])
    assert list(r) == [i1] and r[i1] == {'kind': 'abort', 'abort_s': 4.0, 'ai_end_s': 4.9, 'human_s': 1.1}     # r2: kind
    # 区間の最後の 2 フレームが中立止め(assist 0)・AI_CMD が人側(assist 0)
    a2, h = a.copy(), np.zeros(len(t), bool)
    a2[[48, 49]] = 0
    h[[48, 49]] = True
    assert list(s3m.abort_resumes(t, ai, a2, hold=h, vt=vt, Df=Df)) == [i1]
    assert s3m.abort_resumes(t, ai, a2, vt=vt, Df=Df) == {}                  # 中立止めを除かなければ見落とす
    sel = np.ones(len(t), bool)
    sel[[48, 49]] = False
    assert list(s3m.abort_resumes(t, ai, a2, sel=sel, vt=vt, Df=Df)) == [i1]
    # 中止のあとモデルが走った(assist 0 が最後)
    a3 = a.copy()
    a3[(t >= 4.5) & (t < 5)] = 0
    assert s3m.abort_resumes(t, ai, a3, vt=vt, Df=Df) == {}
    # 走り直しの区間で 0.5 m も進まない → 次の区間も(carried 1)。vt・Df が無ければ続けない
    Df2 = np.where(t < 8.5, np.minimum(t, 5.0), 5.0 + (t - 8.5))
    i2 = int(np.nonzero(t >= 9)[0][0])
    r2 = s3m.abort_resumes(t, ai, a, vt=vt, Df=Df2)
    assert sorted(r2) == [i1, i2] and r2[i2]['carried'] == 1 and r2[i2]['abort_s'] == 4.0 and r2[i2]['human_s'] == 1.1
    assert 'carried' not in r2[i1]
    assert list(s3m.abort_resumes(t, ai, a, vt=None, Df=None)) == [i1]
    # 走り直しの区間(6.0〜7.9 秒)で 0.57 m 進んだら続かない・0.48 m なら続く(RESUME_MOVE_M より少ないときだけ)
    Df3 = np.where(t < 6, 5.0, np.where(t < 8, 5.0 + (t - 6) * 0.3, 5.6))
    assert list(s3m.abort_resumes(t, ai, a, vt=vt, Df=Df3)) == [i1]
    Df4 = np.where(t < 6, 5.0, np.where(t < 8, 5.0 + (t - 6) * 0.25, 5.5))
    assert sorted(s3m.abort_resumes(t, ai, a, vt=vt, Df=Df4)) == [i1, i2]
    assert s3m.RESUME_MOVE_M == 0.5
