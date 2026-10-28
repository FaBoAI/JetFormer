"""直す 7-2 回目・係 tools(2026-10-27。反証 7-2 回目の指摘 w2-tools-r7-2-tools-1・2)の試験。夜の道具だけ
(⑤ experiments/20261026_fix/diag_launch/collect.py・③ tools/standing_3lap.py・④ tools/quick_eval_by_params.py)。
走行の動き(PCA9685 に書く値)・既定値は変えていない(drive.py・assist.py・recorder.py は触っていない)。

  w2-tools-r7-2-tools-1  手順書 7 章「中立で停止中 → 車に近づく前に CH3 を人側」のとおり中立止めの途中で人側にすると、リカバリー中止
    (assist 3)は 1 フレームも書かれず(明けた最初のフレームでしか決まらない)、fix7 r1 の印(after_abort)が付かなかった。その場から
    戻した発進が ⑤ の ON の中央値に入り(−0.165 秒「使ってよい」)、③ は counted、④ の止まってから 3 周に 27.12 秒が入った。カメラが
    止まったまま人側にした形(フレームの途切れのまま区間が終わる)も同じ
    → standing_3lap.abort_resumes が、中立止め(0.5 秒以上)・カメラの途切れのまま止まって終わった区間(終わりで止まっていて、その区間で
    0.5 m 以上進んだ)のあとの発進にも after_abort(kind = 'hold' / 'stall'。中止は 'abort')。扱いは r1 と同じ
  w2-tools-r7-2-tools-2  fix7 r1 は走り直しの ON の発進を、起点までの中央値と一緒に、空転(比 > 1.3・いちばんひどい空転・比が出ない)と
    最初の周の記録の欠けの決まりからも外した(退行。car_check の「→ ⑤ は空転の条件を満たさない」と食い違った)
    → ⑤ が、R1 に届いた走り直しの ON の空転を checks の slip / slip_all_measured に、最初の周の欠けを on_all_readable に数える

どの試験も「直す前の写し(tests/fixtures/fix7_tools_before/r2/*.orig = fix7 r1 の版。sha256 を照らす)で指摘が起きる」か「その場面の
ほかは直す前と出力が同じ」を見る。fix7 の前の写し(tests/fixtures/fix7_tools_before/*.orig)とも比べる(空転・欠けは fix7 の前と同じに
戻る)。作り物の記録だけ(tmp)。本物の 10/10・10/24 の記録(data/raw・data/labels。読むだけ)を読む試験は、無ければ飛ばす。
作り物のコースは tests/test_fix7_tools.py と同じ(反証の台本 fix7/review_tools_r2/probes/gen_course2.py と同じ作り方)。
環境変数 JF_FIX7_COLLECT・JF_FIX7_S3・JF_FIX7_QP で、試す版を差し替えられる(わざと壊した写しで試験が落ちるかを見るとき)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix7_tools_r2.py
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
import test_fix7_tools as t7  # noqa: E402  作り物のコース・回 3 の日(同じ作り方)

FIX0 = JF / 'tests' / 'fixtures' / 'fix7_tools_before'          # fix7 の前(fce5d47)
FIX1 = FIX0 / 'r2'                                               # fix7 r1 の版(この直しの前)
NEW_C = Path(os.environ.get('JF_FIX7_COLLECT') or (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'))
NEW_S3 = Path(os.environ.get('JF_FIX7_S3') or (JF / 'tools' / 'standing_3lap.py'))
NEW_QP = Path(os.environ.get('JF_FIX7_QP') or (JF / 'tools' / 'quick_eval_by_params.py'))
VER = {'new': (NEW_C, NEW_S3, NEW_QP),
       'r1': (FIX1 / 'collect.py.orig', FIX1 / 'standing_3lap.py.orig', FIX1 / 'quick_eval_by_params.py.orig'),
       'f0': (FIX0 / 'collect.py.orig', FIX0 / 'standing_3lap.py.orig', FIX0 / 'quick_eval_by_params.py.orig')}
IG = ['--ignore-gain']
IGF = ['--ignore-gain', '--first-only']
PRM, ON = t7.PRM, t7.ON
RN = 'リカバリー中止のあとの走り直し'      # 名前は fix7 r1 のまま(手順書 7 章・8.1・tests/test_fix7_docs_r2.py が引く言葉)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fixture_is_the_r1_version():
    """直す前の写し(fix7 r1 の版)の sha256 と中身(r2 の名前・欄が無い)"""
    sums = dict(reversed(x.split()) for x in (FIX1 / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sorted(sums) == ['collect.py.orig', 'quick_eval_by_params.py.orig', 'standing_3lap.py.orig']
    for name, h in sums.items():
        assert _sha(FIX1 / name) == h, name
    s3o = (FIX1 / 'standing_3lap.py.orig').read_text(encoding='utf-8')
    assert 'def abort_resumes' in s3o and 'verpm=None' not in s3o and 'HOLD_NOTE' not in s3o and 'HOLD_KEEP_MAX_S' not in s3o
    co = (FIX1 / 'collect.py.orig').read_text(encoding='utf-8')
    assert 'after_abort' in co and 'abort_resume_slip' not in co and "vt, d['Df'], d['erpm'])" not in co
    qo = (FIX1 / 'quick_eval_by_params.py.orig').read_text(encoding='utf-8')
    assert 'launch_to_r1_abort_excluded' in qo and "('hold', 'stall')" not in qo


# ================================================================== 版の読み方
_MODS = {}


def mods(ver):
    """(standing_3lap, quick_eval_by_params) の組(④ は同じ版の ③ を standing_3lap として読む)"""
    if ver not in _MODS:
        _, p3, pq = VER[ver]
        s3m = load_mod('standing_3lap_r72_%s' % ver, p3)
        saved = sys.modules.get('standing_3lap')
        sys.modules['standing_3lap'] = s3m
        try:
            qpm = load_mod('qp_r72_%s' % ver, pq)
        finally:
            if saved is not None:
                sys.modules['standing_3lap'] = saved
            else:
                sys.modules.pop('standing_3lap', None)
        _MODS[ver] = (s3m, qpm)
    return _MODS[ver]


def runc(D, ver, args=(), tag=''):
    """⑤ collect.main を作り物の日で流す(同じ版の ③ を standing_3lap として読ませる)。(JSON, 標準出力, JSON の文)"""
    pc = VER[ver][0]
    name = 'c72_%s_%s_%d' % (ver, tag or 'x', len(args))
    m = load_mod(name, pc)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('%s_%s.json' % (name, abs(hash(tuple(args))) % 10000))
    buf = io.StringIO()
    saved = sys.modules.get('standing_3lap')
    sys.modules['standing_3lap'] = mods(ver)[0]
    try:
        with contextlib.redirect_stdout(buf):
            m.main(['--day', '20261103', '--out', str(out)] + list(args))
    finally:
        if saved is not None:
            sys.modules['standing_3lap'] = saved
        else:
            sys.modules.pop('standing_3lap', None)
    txt = out.read_text(encoding='utf-8')
    return json.loads(txt), buf.getvalue(), txt


def run_s3(D, ver):
    s3m, qpm = mods(ver)
    logs, _ = s3m.day_logs('20261103', D.raw)
    R = s3m.run(logs, D.raw, D.tmp / 'nolab', None)
    md = s3m.to_markdown(R)
    qmd, runs, groups = qpm.to_markdown(R, '20261103', None, ('gain',))
    return R, md, qmd, runs, groups


g0, rows_of, no_path = t7.g0, t7.rows_of, t7.no_path


def st_of(R, log):
    return [x for x in R if x['log'] == log][0]['standing']


def on250(groups):
    return [x for x in groups if '発進 250' in x['text']][0]


# ================================================================== 作り物の回 3(3-8 の形)
def scale_imu(raw, log, a, b, k):
    """その本の IMU の前の加速を a〜b 秒だけ k 倍(車輪は変えない = 空転の形。反証の台本の imu_scale と同じ)"""
    for p in sorted((Path(raw) / log).glob('%s-*/IMU.npz' % log)):
        z = dict(np.load(p))
        tt = (z['timestamps_us'].astype(np.int64) - t7.T0_US) / 1e6
        m = (tt >= a) & (tt < b)
        z['accel'][m, 1] *= k
        np.savez(p, **z)


def stop_book(D, block, prm, kind, stop=3.5, resume=True, d_move=0.31, **kw):
    """3-8 の形: 1 回目の発進(1.0 秒)→ 2 周目の R1(23 m)の (stop + 1) m 手前で止まり始め、R1 の stop m 手前で止まる(1 秒)→ 2 秒
    → CH3 を人側 1.0 秒 → その場で AI 側(走り直し。ON なら発進の前進がまた出る)→ 3 周。kind:
      'abort'  カメラ 0.8 秒 → リカバリー中止(assist 3)が人側まで(fix7 r1 の B と同じ)
      'hold'   中立止め(AI_CMD の hold = True・assist 0)が止まり始めから人側まで(assist 3 は無い。反証の H_hold)
      'stall'  カメラが止まり始めから人側の 0.5 秒あとまで止まる(フレームが無い。反証の S_stall)
      'none'   止まって人側にするだけ(モデルが止めた形。印なし)
    kw で write_log の stall・giveup・hold・human_cmd・vk・ai・launch を差し替えられる"""
    base = t7.launch_keys(1.0, d_move, 0.3)
    t_st = t7.time_at_distance(base + [(200.0, 2.0)], 23.0 - stop - 1.0, 200.0)
    t_stop = t_st + 1.0
    t_flip = t_stop + 2.0
    t_res = t_flip + 1.0
    on = bool(prm.get('launch_boost'))
    if resume:
        dur = t_res + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (t_res, dur - 1.0)]
        launch = [(1.0, 0.5), (t_res, 0.5)] if on else []
    else:
        dur = t_flip + 3.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (dur, 0.0)]
        ai = [(1.0, t_flip)]
        launch = [(1.0, 0.5)] if on else []
    w = {}
    if kind == 'abort':
        w = dict(stall=[(t_st, t_st + 0.8)], giveup=[(t_st + 0.8, t_flip)],
                 events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8, 'run': 'runB',
                          'cam_epoch_us': int(t7.T0_US + round((t_st + 0.8) * 1e6)),
                          't_us': int(t7.T0_US + round((t_st + 0.8) * 1e6))}])
    elif kind == 'hold':
        w = dict(hold=[(t_st, t_flip)])
    elif kind == 'stall':
        w = dict(stall=[(t_st, t_flip + 0.5)])
    tt = {'t_st': t_st, 't_stop': t_stop, 't_flip': t_flip, 't_res': t_res, 'dur': dur}
    over = {k: kw.pop(k) for k in list(kw) if k in ('vk', 'ai', 'launch', 'dur')}
    w.update(kw)
    w = {k: (v(tt) if callable(v) else v) for k, v in w.items()}
    over = {k: (v(tt) if callable(v) else v) for k, v in over.items()}
    log = D.add(block, prm, d_move, 0.3, dur=over.pop('dur', dur), vk=over.pop('vk', vk), ai=over.pop('ai', ai),
                launch=over.pop('launch', launch), **w)
    return log, tt


def day(tmp, kind='hold', stop=3.5, b38=ON, resume=True, d38=0.31, on_prm=ON, us=None, spin=None, imu_drop=None, **kw):
    """回 3(tests/test_fix7_tools.round3 と同じ 7 本 + 3-8)。us = {block: launch_us}(ON の本の強さを変える)。
    spin = IMU の前の加速の倍率(走り直しの発進から 1.2 秒)。imu_drop = 3-8 のシーン番号(その IMU.npz を消す)"""
    D = t7.Day(tmp)
    us = us or {}
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, on_prm, 0.15, 0.3), (14, PRM, 0.21, 0.6), (15, on_prm, 0.20, 0.3),
                               (16, PRM, 0.22, 0.6), (17, on_prm, 0.28, 0.3), (18, PRM, 0.23, 0.6)):
        D.add(blk, dict(prm, launch_us=us[blk]) if blk in us else prm, dm, ramp)
    p38 = dict(b38, launch_us=us[19]) if 19 in us else b38
    log, tt = stop_book(D, 19, p38, kind, stop=stop, resume=resume, d_move=d38, **kw)
    if spin is not None:
        scale_imu(D.raw, log, tt['t_res'], tt['t_res'] + 1.2, spin)
    if imu_drop is not None:
        (D.raw / log / ('%s-%04d' % (log, imu_drop)) / 'IMU.npz').unlink()
    return D, log, tt


# ================================================================== tools-1: 再現(直す前 = fix7 r1 の版)
@pytest.mark.parametrize('kind', ['hold', 'stall'])
@pytest.mark.parametrize('stop', [3.5, 4.5])
def test_r1_hold_stall_resume_turns_to_use_ok(tmp_path, kind, stop):
    """直す前: 中立止め・カメラの途切れのまま人側 → その場で AI 側の走り直しに印が付かず、⑤ ON n 5・−0.165 秒「使ってよい」、
    ③ counted、④ の発進 250 の組の止まってから 3 周に 27.12 秒(反証の H_hold・S_stall)"""
    D, log, _ = day(tmp_path, kind, stop)
    j, o, _ = runc(D, 'r1', IG, kind)
    g = g0(j)
    assert g['by_launch']['on']['n'] == 5 and g['compare']['d_to_r1_s'] == -0.165 and g['compare']['all_ok'] is True
    assert '→ 使ってよい' in o and all('after_abort' not in r for r in j['launches'])
    R, md, qmd, runs, groups = run_s3(D, 'r1')
    s2 = st_of(R, log)[1]
    assert s2['counted'] is True and 'after_abort' not in s2
    assert on250(groups)['standing_n'] == 1 and abs(on250(groups)['standing_best_s'] - 27.12) < 0.01


# ================================================================== tools-1: 今
@pytest.mark.parametrize('kind', ['hold', 'stall'])
@pytest.mark.parametrize('stop', [0.5, 1.5, 2.5, 3.5, 4.5])
def test_c5_hold_stall_resume_is_excluded(tmp_path, kind, stop):
    """今: 走り直しに after_abort(kind hold / stall)。⑤ は比べから外して ON 4 本対 OFF 4 本の −0.125 秒「満たさない」(中止で止まった
    B と同じ数)。組の行・発進の行に止まった形の文"""
    D, log, tt = day(tmp_path, kind, stop)
    j, o, _ = runc(D, 'new', IG, kind)
    g = g0(j)
    assert g['by_launch']['on']['n'] == 4 and g['by_launch']['on']['restarts'] == 0
    assert g['by_launch']['on']['to_r1_median_s'] == 2.903 and g['compare']['d_to_r1_s'] == -0.125
    assert g['compare']['all_ok'] is False and g['compare']['checks']['to_r1'] is False and '→ 使ってよい' not in o
    r1, r2 = rows_of(j, log)
    assert 'after_abort' not in r1
    ab = r2['after_abort']
    assert ab['kind'] == kind and abs(ab['human_s'] - (1.0 if kind == 'hold' else 1.0 + (tt['t_flip'] - tt['t_st']))) < 0.07
    assert abs(ab['abort_s'] - tt['t_st']) < 0.04
    if kind == 'hold':
        assert set(ab) == {'kind', 'abort_s', 'hold_s', 'ai_end_s', 'human_s'}
        assert abs(ab['hold_s'] - (tt['t_flip'] - tt['t_st'])) < 0.07 and abs(ab['ai_end_s'] - tt['t_flip']) < 0.04
        txt = '中立止め %s 秒から %s 秒' % (ab['abort_s'], ab['hold_s'])
    else:
        assert set(ab) == {'kind', 'abort_s', 'gap_s', 'ai_end_s', 'human_s'}
        assert abs(ab['gap_s'] - (tt['t_flip'] + 0.5 - tt['t_st'])) < 0.07 and abs(ab['ai_end_s'] - tt['t_st']) < 0.04
        txt = 'カメラの途切れ %s 秒から %s 秒' % (ab['abort_s'], ab['gap_s'])
    assert len(g['abort_resume_excluded']) == 1
    assert g['abort_resume_excluded'][0].startswith('%s@%.2f(on・%s・人側 %s 秒・起点まで %.3f 秒・空転の比' % (
        log, r2['t_ai'], txt, ab['human_s'], r2['to_r1_s']))
    line = [x for x in o.splitlines() if x.startswith('  %s 1 本' % RN)]
    assert len(line) == 1 and log in line[0] and 'ON の組にも OFF の組にも入れない' in line[0]
    assert any(x.startswith(log) and ' 中止のあとの走り直し(%s・人側 %s 秒。比べに入れない)' % (txt, ab['human_s']) in x
               for x in o.splitlines())


@pytest.mark.parametrize('kind', ['hold', 'stall'])
def test_c5_hold_stall_same_numbers_as_abort(tmp_path, kind):
    """中立止め・カメラの途切れのあとの走り直しの日の比べ(by_launch・compare)は、中止で止まった日(B)・走り直さない日と同じ"""
    Dk, _, _ = day(tmp_path / 'k', kind, 3.5)
    Da, _, _ = day(tmp_path / 'a', 'abort', 3.5)
    Dn, _, _ = day(tmp_path / 'n', kind, 3.5, resume=False)
    gk, ga, gn = (g0(runc(D_, 'new', IG, t_)[0]) for D_, t_ in ((Dk, 'k'), (Da, 'a'), (Dn, 'n')))
    for key in ('by_launch', 'compare'):
        assert gk[key] == ga[key] == gn[key], key


@pytest.mark.parametrize('kind', ['hold', 'stall'])
@pytest.mark.parametrize('stop', [3.5, 1.5])
def test_s3_hold_stall_resume_not_counted(tmp_path, kind, stop):
    """③: 走り直しは counted = False・after_abort(kind)。表に「数えない(中止・中立止めのあとの走り直し。中立止め X 秒から Y 秒 …)」と
    注の 1 行。④: 止まってから 3 周 0 本・数えない 1・1 章の印・動く → A1 から外して 2 章の下に 1 行"""
    D, log, tt = day(tmp_path, kind, stop)
    s3m, _ = mods('new')
    R, md, qmd, runs, groups = run_s3(D, 'new')
    s1, s2 = st_of(R, log)
    assert 'after_abort' not in s1 and s2['valid'] and s2['counted'] is False and s2['after_abort']['kind'] == kind
    ab = s2['after_abort']
    head = ('中立止め %.2f 秒から %.1f 秒' % (ab['abort_s'], ab['hold_s']) if kind == 'hold'
            else 'カメラの途切れ %.2f 秒から %.1f 秒' % (ab['abort_s'], ab['gap_s']))
    assert '・数えない(%s。%s・人側 %.1f 秒)' % (RN, head, ab['human_s']) in md
    assert s3m.ABORT_NOTE in md and s3m.HOLD_NOTE in md and s3m.best_standing([x for x in R if x['log'] == log][0]) is None
    g = on250(groups)
    assert g['standing_n'] == 0 and g['standing_excluded_n'] == 1 and g['launch_to_r1_abort_excluded'] == ['%s@%.2f' % (log, s2['t_ai'])]
    run = [x for x in runs if x['log'] == log][0]
    assert '数えない発進 1(%s)' % RN in run['flags'] and not any('前に衝突' in f for f in run['flags'])
    assert any(x.startswith('- %s(' % RN) and log in x and x.endswith('のあとの発進も同じ(③ の表の「中立止め X 秒から Y 秒」'
                                                                  '「カメラの途切れ X 秒から Y 秒」)。') for x in qmd.splitlines())


@pytest.mark.parametrize('kind', ['hold', 'stall'])
def test_s3_hold_stall_strip_equals_r1(tmp_path, kind):
    """③: after_abort と counted のほかの値は直す前(r1)と同じ"""
    D, log, _ = day(tmp_path, kind, 3.5)
    Rn, Ro = run_s3(D, 'new')[0], run_s3(D, 'r1')[0]
    for rn, ro in zip(Rn, Ro):
        assert rn['log'] == ro['log']
        assert [{k: v for k, v in s.items() if k not in ('after_abort', 'counted')} for s in rn.get('standing') or []] == \
            [{k: v for k, v in s.items() if k != 'counted'} for s in ro.get('standing') or []]
        assert {k: v for k, v in rn.items() if k != 'standing'} == {k: v for k, v in ro.items() if k != 'standing'}


def test_c5_real_shape_20261010_160306(tmp_path):
    """本物の 20261010-160306 の形(作り物): 前へ 1.56 m/s で走っているときに中立止め → 0.73 秒で 0 m/s → 中立止めのまま人側(人が
    0.74 m 動かして 1.93 秒)→ 止まった所から AI 側。印(kind hold・hold_s 0.73 前後)が付き、比べから外れる"""
    def vk(tt):
        th = tt['t_st'] + 0.5
        return t7.launch_keys(1.0, 0.31, 0.3) + [(th - 0.3, 2.0), (th, 1.56), (th + 0.73, 0.0), (th + 0.9, 0.0), (th + 1.3, 0.8),
                                                   (th + 1.8, 0.0), (th + 0.75 + 1.93 + 0.15, 0.0), (th + 0.75 + 1.93 + 0.45, 2.0),
                                                   (tt['dur'], 2.0)]

    def ai(tt):
        th = tt['t_st'] + 0.5
        return [(1.0, th + 0.75), (th + 0.75 + 1.93, tt['dur'] - 1.0)]

    def launch(tt):
        th = tt['t_st'] + 0.5
        return [(1.0, 0.5), (th + 0.75 + 1.93, 0.5)]
    D, log, tt = day(tmp_path, 'none', 3.5, hold=lambda tt: [(tt['t_st'] + 0.5, tt['t_st'] + 0.5 + 0.75)], vk=vk, ai=ai, launch=launch)
    j, o, _ = runc(D, 'new', IG, 'real')
    rs = rows_of(j, log)
    assert len(rs) == 2 and 'after_abort' not in rs[0]
    ab = rs[1]['after_abort']
    assert ab['kind'] == 'hold' and 0.70 <= ab['hold_s'] <= 0.80 and abs(ab['human_s'] - 1.93) < 0.07
    assert g0(j)['by_launch']['on']['n'] == 4 and len(g0(j)['abort_resume_excluded']) == 1
    jr, _, _ = runc(D, 'r1', IG, 'real')
    assert all('after_abort' not in r for r in jr['launches']) and g0(jr)['by_launch']['on']['n'] == 5


# ================================================================== tools-1: 印を付けない形(前と同じ)
NEG = {
    # 中立止めが 0.4 秒(HOLD_KEEP_MAX_S より短い。止まったのはモデル)
    'hold_short': dict(kind='none', hold=lambda tt: [(tt['t_flip'] - 0.4, tt['t_flip'])]),
    # 中立止め 0.6 秒のまま、まだ 1 m/s 以上で走っているうちに人側(終わりで止まっていない。人がそのあと止めた)
    'hold_moving': dict(kind='none', hold=lambda tt: [(tt['t_st'] + 0.2, tt['t_st'] + 0.8)],
                        vk=lambda tt: t7.launch_keys(1.0, 0.31, 0.3) + [(tt['t_st'], 2.0), (tt['t_st'] + 0.8, 1.5),
                                                                       (tt['t_stop'] + 0.5, 0.0), (tt['t_res'] + 0.15, 0.0),
                                                                       (tt['t_res'] + 0.45, 2.0), (tt['dur'], 2.0)],
                        ai=lambda tt: [(1.0, tt['t_st'] + 0.8), (tt['t_res'], tt['dur'] - 1.0)]),
    # カメラの途切れ 0.15 秒(le.GAP_S 以下)で人側
    'gap_short': dict(kind='none', stall=lambda tt: [(tt['t_flip'] - 0.15, tt['t_flip'])]),
    # カメラが止まったまま人側にし、途切れの間ずっと 0.2 m/s 以上で動いていた(人が運転した)
    'stall_moving': dict(kind='none', stall=lambda tt: [(tt['t_st'], tt['t_st'] + 1.2)],
                         vk=lambda tt: t7.launch_keys(1.0, 0.31, 0.3) + [(tt['t_st'], 2.0), (tt['t_st'] + 1.2, 0.8),
                                                                        (tt['t_stop'] + 1.0, 0.0), (tt['t_res'] + 0.15, 0.0),
                                                                        (tt['t_res'] + 0.45, 2.0), (tt['dur'], 2.0)],
                         ai=lambda tt: [(1.0, tt['t_st'] + 0.5), (tt['t_res'], tt['dur'] - 1.0)]),
}


@pytest.mark.parametrize('case', sorted(NEG))
def test_c5_not_marked_is_same_as_r1(tmp_path, case):
    """中立止めが短い・終わりで動いている・途切れが短い・途切れの間ずっと動いていた: 印を付けない。⑤ の JSON・標準出力、③④ の結果・表は
    直す前(r1)とバイト単位で同じ"""
    kw = dict(NEG[case])
    D, log, _ = day(tmp_path, kw.pop('kind'), 3.5, **kw)
    jn, on, tn = runc(D, 'new', IG, case)
    jo, oo, to = runc(D, 'r1', IG, case)
    assert all('after_abort' not in r for r in jn['launches']) and len(rows_of(jn, log)) == 2
    assert tn == to and no_path(on) == no_path(oo)
    Rn, mdn, qn, rn, gn = run_s3(D, 'new')
    Ro, mdo, qo, ro, go = run_s3(D, 'r1')
    dump = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)      # noqa: E731
    assert dump(Rn) == dump(Ro) and mdn == mdo and qn == qo and dump(rn) == dump(ro) and dump(gn) == dump(go)


def test_c5_never_moved_segment_is_not_marked(tmp_path):
    """置いた場所から一度も動かずに中立止めのまま人側にした区間(本物の 20261024-183123 の 40.10〜43.36 秒の形)のあとの発進は、印なし
    (その本の 1 回目の発進と同じ = 置いた場所から)"""
    D = t7.Day(tmp_path)
    D.add(12, PRM, 0.20, 0.6)
    dur = 40.0
    vk = [(0.0, 0.0), (8.0, 0.0), (8.3, 2.0), (dur, 2.0)]
    log = D.add(19, ON, 0.0, 0.3, dur=dur, vk=vk, ai=[(1.0, 4.3), (7.85, dur - 1.0)], launch=[(7.85, 0.5)], hold=[(1.5, 4.3)])
    jn, on, tn = runc(D, 'new', IG, 'nm')
    jo, oo, to = runc(D, 'r1', IG, 'nm')
    assert [r.get('after_abort') for r in jn['launches'] if r['log'] == log] == [None] * len(rows_of(jn, log))
    assert tn == to and no_path(on) == no_path(oo)


# ================================================================== tools-1: その場面が無ければ同じ
@pytest.mark.parametrize('args', [[], IG, IGF])
@pytest.mark.parametrize('case', ['N', 'A', 'Hn'])
def test_c5_no_resume_is_bit_identical_to_r1(tmp_path, case, args):
    """中止・中立止めの無い日(N)・中止のあと走り直さない日(A)・中立止めのまま人側にして走り直さない日(Hn): ⑤ の JSON・標準出力とも
    直す前(r1)・fix7 の前とバイト単位で同じ"""
    if case == 'N':
        D = t7.Day(tmp_path)
        for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, PRM, 0.21, 0.6), (19, ON, 0.31, 0.3)):
            D.add(blk, prm, dm, ramp)
    else:
        D, _, _ = day(tmp_path, 'abort' if case == 'A' else 'hold', 3.5, resume=False)
    outs = [runc(D, v, args, case) for v in ('new', 'r1', 'f0')]
    assert outs[0][2] == outs[1][2] == outs[2][2]
    assert no_path(outs[0][1]) == no_path(outs[1][1]) == no_path(outs[2][1])


@pytest.mark.parametrize('case', ['N', 'Hn'])
def test_s3_no_resume_is_identical_to_r1(tmp_path, case):
    """③④: 走り直しの無い日は、③ の結果・表と ④ の表・本・組が直す前(r1)と同じ"""
    if case == 'N':
        D = t7.Day(tmp_path)
        for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (19, ON, 0.31, 0.3)):
            D.add(blk, prm, dm, ramp)
    else:
        D, _, _ = day(tmp_path, 'hold', 3.5, resume=False)
    a, b = run_s3(D, 'new'), run_s3(D, 'r1')
    dump = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)      # noqa: E731
    assert dump(a[0]) == dump(b[0]) and a[1] == b[1] and a[2] == b[2] and dump(a[3]) == dump(b[3]) and dump(a[4]) == dump(b[4])


def _norm_c5(j, out):
    """r1 と今の ⑤ の出力の、kind・組の行(説明の文と空転の比)の違いを除く(中止の走り直しで、ほかが同じかを見る)"""
    j = copy.deepcopy(j)
    for r in j['launches']:
        if r.get('after_abort'):
            r['after_abort'].pop('kind', None)
    for g in j.get('summary') or []:
        g.pop('abort_resume_excluded', None)
    return j, '\n'.join(x for x in out.splitlines() if not x.startswith('  %s ' % RN))


@pytest.mark.parametrize('stop', [3.5, 4.5])
def test_c5_abort_resume_same_as_r1_but_names(tmp_path, stop):
    """中止のあとの走り直し(空転なし・欠けなし。r1 の B): 足した kind・組の行の説明の文と空転の比のほかは、⑤ の JSON・標準出力とも
    直す前(r1)と同じ(発進の行の「中止のあとの走り直し(中止 X 秒・人側 Y 秒。…)」も同じ)。abort_resume_excluded は後ろに
    「空転の比 X」が付くだけ"""
    D, log, _ = day(tmp_path, 'abort', stop)
    jn, on, _ = runc(D, 'new', IG, 'ab')
    jo, oo, _ = runc(D, 'r1', IG, 'ab')
    a, b = _norm_c5(jn, on), _norm_c5(jo, oo)
    assert a[0] == b[0] and no_path(a[1]) == no_path(b[1])
    r2 = rows_of(jn, log)[1]
    old = g0(jo)['abort_resume_excluded'][0]
    assert g0(jn)['abort_resume_excluded'] == [old.replace('秒)', '秒・空転の比 %.2f)' % r2['slip']['ratio'])]
    assert r2['after_abort']['kind'] == 'abort'


def test_c5_s3_qp_keep_the_words_the_runbook_quotes(tmp_path):
    """中止のあとの走り直しの文は fix7 r1 のまま(手順書 7 章・8.1 と tests/test_fix7_docs_r2.py の g1 が引く言葉): ⑤ の組の行の頭・発進の
    行の「中止のあとの走り直し(中止 X 秒・人側 Y 秒。」・③ の「数えない(リカバリー中止のあとの走り直し。中止 X 秒・人側 Y 秒)」と注・
    ④ の「数えない発進 1(リカバリー中止のあとの走り直し)」と 2 章の下の行・届かなかった発進の「うちリカバリー中止のあとの走り直し 1(」"""
    import re
    D, log, _ = day(tmp_path / 'f', 'abort', 3.5)
    j, o, _ = runc(D, 'new', IG, 'words')
    line = [x for x in o.splitlines() if x.startswith('  リカバリー中止のあとの走り直し ')]
    assert len(line) == 1 and line[0].startswith('  リカバリー中止のあとの走り直し 1 本(') and '手順書 7 章' in line[0]
    assert re.search(r' 中止のあとの走り直し\(中止 [\d.]+ 秒・人側 [\d.]+ 秒。', o)
    R, md, qmd, runs, groups = run_s3(D, 'new')
    assert re.search(r'数えない\(リカバリー中止のあとの走り直し。中止 [\d.]+ 秒・人側 [\d.]+ 秒\)', md)
    assert any(x.startswith('- リカバリー中止のあとの走り直し = ') for x in md.splitlines())
    assert '中立止め' not in md and mods('new')[0].HOLD_NOTE not in md          # 中止だけの日は中立止めの注を出さない
    assert '数えない発進 1(リカバリー中止のあとの走り直し)' in qmd
    ql = [x for x in qmd.splitlines() if x.startswith('- リカバリー中止のあとの走り直し(')]
    assert len(ql) == 1 and log in ql[0] and '中立で停止中' not in ql[0]       # 中止だけの日は中立止めの 1 文を足さない
    Ds, _, _ = day(tmp_path / 's', 'abort', 3.5,
                   vk=lambda tt: t7.launch_keys(1.0, 0.31, 0.3) + [(tt['t_st'], 2.0), (tt['t_stop'], 0.0), (tt['t_res'] + 0.15, 0.0),
                                                                  (tt['t_res'] + 0.45, 1.0), (tt['t_res'] + 1.0, 1.0),
                                                                  (tt['t_res'] + 2.0, 0.0), (tt['t_res'] + 3.0, 0.0)],
                   ai=lambda tt: [(1.0, tt['t_flip']), (tt['t_res'], tt['t_res'] + 1.0)], dur=lambda tt: tt['t_res'] + 3.0)
    js, os_, _ = runc(Ds, 'new', IG, 'words_s')
    assert any('。うちリカバリー中止のあとの走り直し 1(' in x for x in os_.splitlines() if 'R1 に届かなかった発進' in x)


# ================================================================== tools-2: 空転・最初の周の欠け
def kday(tmp, kind='abort', b38=None, **kw):
    """反証 probe_slip_resume の回 3: 発進の前進 ON は launch_us 200。3-8 の 1 回目 2.813 秒(ON の中央値 = OFF − 0.190 秒 → 使ってよい)。
    2 周目に R1 の 5 m 手前(置いた場所と同じ HS)で止まり、kind のあと人側 1.0 秒 → AI 側"""
    on200 = dict(ON, launch_us=200.0)
    return day(tmp, kind, 5.0, b38=b38 or on200, d38=0.15, on_prm=on200, **kw)


@pytest.mark.parametrize('kind', ['abort', 'hold', 'stall'])
def test_c5_resume_spin_counts_for_slip(tmp_path, kind):
    """走り直しの ON の発進が空転した(比 ≈ 1.85): 今は checks の slip = False で「満たさない」・組に abort_resume_slip・行。中止は
    直す前(r1)は「使ってよい」(退行)、fix7 の前は「満たさない」(比べに入れていた)。car_check の「→ ⑤ は空転の条件を満たさない」と同じ"""
    D, log, tt = kday(tmp_path, kind, spin=0.55)
    jn, on, _ = runc(D, 'new', IG, 'sp' + kind)
    g = g0(jn)
    r2 = rows_of(jn, log)[1]
    assert r2['after_abort']['kind'] == kind and r2['slip']['ratio'] > 1.3
    assert g['by_launch']['on']['n'] == 4 and g['by_launch']['on']['slip_max'] <= 1.3     # 比べの数(中央値・空転max)からは外したまま
    assert g['compare']['d_to_r1_s'] == -0.19 and g['compare']['checks']['to_r1'] is True
    assert g['compare']['checks']['slip'] is False and g['compare']['all_ok'] is False and '→ 満たさない' in on
    assert g['abort_resume_slip'] == ['%s@%.2f(空転の比 %.2f > 1.3)' % (log, r2['t_ai'], r2['slip']['ratio'])]
    assert '空転の比 %.2f > 1.3' % r2['slip']['ratio'] in g['abort_resume_excluded'][0]
    assert any(x.startswith('  → %sの ON の発進が空転した 1 本' % RN) and log in x for x in on.splitlines())
    if kind == 'abort':
        jo, oo, _ = runc(D, 'r1', IG, 'sp' + kind)
        assert g0(jo)['compare']['all_ok'] is True and '→ 使ってよい' in oo
        j0, o0, _ = runc(D, 'f0', IG, 'sp' + kind)
        assert g0(j0)['compare']['checks']['slip'] is False and g0(j0)['compare']['all_ok'] is False
        # car_check(担当。変えていない)は前から「!」と「→ ⑤ は空転の条件を満たさない」
        import car_check as cc
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cc.main([log, '--raw', str(D.raw), '--labels', str(D.tmp / 'nolab'), '--no-json'])
        assert '⑤ collect.py は空転の条件を満たさない' in buf.getvalue()


def test_c5_resume_no_spin_is_use_ok(tmp_path):
    """空転なし・欠けなしの走り直し(反証 K2): 今も「使ってよい」(r1 と同じ答え。走り直しは比べに入れない)。abort_resume_slip は無い"""
    D, log, _ = kday(tmp_path, 'abort')
    jn, on, _ = runc(D, 'new', IG, 'k2')
    g = g0(jn)
    assert g['compare']['all_ok'] is True and 'abort_resume_slip' not in g and 'abort_resume_slip_unmeasured' not in g
    assert '空転の比 1.02' in g['abort_resume_excluded'][0]
    jo, _, _ = runc(D, 'r1', IG, 'k2')
    assert g0(jo)['compare'] == g['compare'] and g0(jo)['by_launch'] == g['by_launch']


def test_c5_resume_extreme_spin(tmp_path):
    """走り直しの空転がいちばんひどい(比が出ないのに車輪が増えた): checks の slip = False、文は「いちばんひどい空転とみなす」"""
    D, log, _ = kday(tmp_path, 'abort', spin=0.05)
    jn, on, _ = runc(D, 'new', IG, 'ext')
    g, r2 = g0(jn), rows_of(jn, log)[1]
    assert r2['slip']['ratio'] is None and r2['slip']['dv_wheel'] > 0.5 and r2['slip']['dv_imu'] <= 0.2
    assert g['compare']['checks']['slip'] is False and g['compare']['all_ok'] is False
    assert len(g['abort_resume_slip']) == 1 and 'いちばんひどい空転とみなす' in g['abort_resume_slip'][0]
    jo, _, _ = runc(D, 'r1', IG, 'ext')
    assert g0(jo)['compare']['all_ok'] is True


def test_c5_resume_slip_unmeasured(tmp_path):
    """走り直しの空転の比が出ない(R1 の 1.5 m 手前から = 窓が短い): checks の slip_all_measured = False(fix7 の前と同じ)。
    abort_resume_slip_unmeasured と行"""
    D, log, _ = day(tmp_path, 'abort', 1.5)
    jn, on, _ = runc(D, 'new', IG, 'unk')
    g = g0(jn)
    assert g['compare']['checks']['slip_all_measured'] is False and 'abort_resume_slip' not in g
    assert g['abort_resume_slip_unmeasured'] == ['%s@%.2f(空転の比が出ない(窓が短い(LH・R1 が近い)か IMU が抜けた))' % (
        log, rows_of(jn, log)[1]['t_ai'])]
    assert any(x.startswith('  → %sの ON の発進で空転の比が出ない 1 本' % RN) for x in on.splitlines())
    jo, _, _ = runc(D, 'r1', IG, 'unk')
    j0, _, _ = runc(D, 'f0', IG, 'unk')
    assert 'slip_all_measured' not in g0(jo)['compare']['checks']
    assert g0(j0)['compare']['checks']['slip_all_measured'] is False


@pytest.mark.parametrize('kind', ['abort', 'stall'])
def test_c5_resume_first_lap_gap(tmp_path, kind):
    """走り直しの最初の周の IMU が抜けた(3-8 のシーン 0002 の IMU.npz が無い。反証 K4・K5): first_lap_gap_on に入り on_all_readable =
    False で「満たさない」。行に「(… 。中止・中立止めのあとの走り直し)」。中止は r1 で「使ってよい」(退行)、fix7 の前は「満たさない」"""
    D, log, _ = kday(tmp_path, kind, imu_drop=2)
    jn, on, _ = runc(D, 'new', IG, 'gap' + kind)
    g, r2 = g0(jn), rows_of(jn, log)[1]
    assert r2.get('first_lap_gap') and r2['after_abort']['kind'] == kind
    assert g['compare']['checks']['on_all_readable'] is False and g['compare']['all_ok'] is False
    assert g['first_lap_gap_on'] == ['%s@%.2f(%s。%s)' % (log, r2['t_ai'], r2['first_lap_gap'], RN)]
    assert '最初の周の記録が欠けた(%s)' % r2['first_lap_gap'] in g['abort_resume_excluded'][0]
    assert any(x.startswith('  最初の周の記録が欠けた発進(on)1 本') and log in x for x in on.splitlines())
    if kind == 'abort':
        jo, oo, _ = runc(D, 'r1', IG, 'gap')
        assert g0(jo)['compare']['all_ok'] is True and 'first_lap_gap_on' not in g0(jo)
        j0, _, _ = runc(D, 'f0', IG, 'gap')
        assert g0(j0)['compare']['checks']['on_all_readable'] is False


def test_c5_off_resume_spin_and_gap_not_counted(tmp_path):
    """OFF の本の走り直しの空転・欠けは数えない(空転・欠けの決まりは ON だけ。前と同じ)"""
    D, log, _ = kday(tmp_path, 'abort', b38=PRM, spin=0.55)
    jn, _, _ = runc(D, 'new', IG, 'off')
    g = g0(jn)
    assert 'abort_resume_slip' not in g and 'abort_resume_slip_unmeasured' not in g and 'first_lap_gap_on' not in g
    assert '空転' not in g['abort_resume_excluded'][0] and '(off・' in g['abort_resume_excluded'][0]
    jo, _, _ = runc(D, 'r1', IG, 'off')
    assert g0(jo)['compare'] == g['compare']


@pytest.mark.parametrize('resume_us, counted', [(200.0, True), (250.0, False)])
def test_c5_resume_spin_mixed_launch_us(tmp_path, resume_us, counted):
    """launch_us が混ざった組(手順書 7 章「残りの ON を 200 µs に」): 比べに使うのは最後の ON の強さ(200)。走り直しの空転も、その強さの
    走り直しだけを数える(250 のときの走り直しの空転は、比べの ON の 250 の空転と同じく数えない)"""
    if resume_us == 200.0:      # 3-2・3-4 は 250、3-6・3-8 は 200(3-8 の走り直しは 200)
        D, log, _ = kday(tmp_path, 'abort', us={13: 250.0, 15: 250.0, 17: 200.0, 19: 200.0}, spin=0.55)
    else:                       # 3-8 は 250(走り直しも 250)、3-2・3-4・3-6 と 3-8 のあとの 3-9 は 200
        D, log, _ = kday(tmp_path, 'abort', us={13: 200.0, 15: 200.0, 17: 200.0, 19: 250.0}, spin=0.55)
        D.bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                       'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                                  for b in range(12, 21)]}]}), encoding='utf-8')
        D.add(20, dict(ON, launch_us=200.0), 0.15, 0.3)
    jn, on, _ = runc(D, 'new', IG, 'mix%d' % int(resume_us))
    g = g0(jn)
    assert g.get('mixed_launch_us') == [200.0, 250.0] and g['launch_us_used'] == 200.0
    assert rows_of(jn, log)[1]['slip']['ratio'] > 1.3 and rows_of(jn, log)[1]['after_abort']['kind'] == 'abort'
    assert ('abort_resume_slip' in g) is counted
    assert g['compare']['checks']['slip'] is (not counted)


def test_c5_first_only_unchanged_with_spin(tmp_path):
    """--first-only: 走り直し(2 回目以降)はもとから入らない。空転した走り直しがあっても、まとめは直す前(r1)と同じ"""
    D, log, _ = kday(tmp_path, 'abort', spin=0.55)
    jn, _, _ = runc(D, 'new', IGF, 'fo')
    jo, _, _ = runc(D, 'r1', IGF, 'fo')
    assert jn['summary'] == jo['summary']


# ================================================================== abort_resumes だけ(hold・stall)
def _seg(t, *ivs):
    m = np.zeros(len(t), bool)
    for a, b in ivs:
        m |= (t >= a - 1e-9) & (t < b - 1e-9)
    return m


def test_abort_resumes_hold_stall_unit():
    """中立止め・カメラの途切れで止まって終わった区間の決まり(長さ・止まっている・進んだ・AI_CMD が人側・途切れ・続き・中止が先)"""
    s3m = mods('new')[0]
    t = np.round(np.arange(0, 20, 1 / 30.0), 4)
    ai = _seg(t, (0, 5), (6, 8), (9, 20))
    vt = np.round(np.arange(0, 20, 0.01), 3)
    mv2 = ((vt >= 6.2) & (vt < 7.5)) | (vt >= 9.3)        # 走り直しの区間(6〜8 秒・9 秒〜)は 1.3 m 以上進む(続きにならない)
    v = np.where((vt < 4.0) | mv2, 1.0, 0.0)              # 1 つ目の区間は 4 秒で止まる
    ve = -v / 0.5e-3
    _, Df = le.cumulative_distance(vt, ve)
    a = np.zeros(len(t), np.int8)
    i1 = int(np.nonzero(t >= 6)[0][0])
    i2 = int(np.nonzero(t >= 9)[0][0])
    assert s3m.HOLD_KEEP_MAX_S == 0.5
    # verpm が無ければ前と同じ(中止の無い本は {})
    h = _seg(t, (4.2, 5))
    assert s3m.abort_resumes(t, ai, a, hold=h, vt=vt, Df=Df) == {}
    r = s3m.abort_resumes(t, ai, a, hold=h, vt=vt, Df=Df, verpm=ve)
    assert list(r) == [i1] and r[i1]['kind'] == 'hold' and abs(r[i1]['abort_s'] - 4.2) < 0.02
    assert abs(r[i1]['hold_s'] - 0.8) < 0.04 and abs(r[i1]['human_s'] - 1.03) < 0.04 and 'carried' not in r[i1]
    # 0.5 秒の境(前に処理したフレームから): 0.50 秒は印・0.40 秒は印なし
    r50 = s3m.abort_resumes(t, ai, a, hold=_seg(t, (4.5, 5)), vt=vt, Df=Df, verpm=ve)
    assert list(r50) == [i1] and r50[i1]['hold_s'] == 0.5
    assert s3m.abort_resumes(t, ai, a, hold=_seg(t, (4.6, 5)), vt=vt, Df=Df, verpm=ve) == {}
    # 区間の最後のフレームが AI_CMD の人側(中立止めでない)でも、それを除いて見る
    sel = ~_seg(t, (4.9, 5))
    assert list(s3m.abort_resumes(t, ai, a, hold=_seg(t, (4.2, 4.9)), sel=sel, vt=vt, Df=Df, verpm=ve)) == [i1]
    assert s3m.abort_resumes(t, ai, a, hold=_seg(t, (4.2, 4.9)), vt=vt, Df=Df, verpm=ve) == {}
    # 終わりで動いている → 印なし
    ve_m = -np.where(vt < 5.5, 1.0, 0.0) / 0.5e-3
    _, Df_m = le.cumulative_distance(vt, ve_m)
    assert s3m.abort_resumes(t, ai, a, hold=h, vt=vt, Df=Df_m, verpm=ve_m) == {}
    # 区間で 0.5 m も進まない → 印なし(0.4 m)・0.6 m なら印
    for dist, want in ((0.4, False), (0.6, True)):
        ve2 = -np.where((vt < dist) | mv2, 1.0, 0.0) / 0.5e-3
        _, Df2 = le.cumulative_distance(vt, ve2)
        got = s3m.abort_resumes(t, ai, a, hold=h, vt=vt, Df=Df2, verpm=ve2)
        assert (list(got) == [i1]) if want else (got == {}), dist
    # カメラの途切れ: 区間の終わりのフレームが消え、次のフレーム(人側)まで GAP_S より長い → 印(stall)。0.17 秒は印なし
    for gap, want in ((0.25, True), (0.15, False)):
        keep = ~((t > 5 - gap) & (t < 5 - 1e-9))
        tt_, ai_, a_ = t[keep], ai[keep], a[keep]
        t_last = float(tt_[tt_ < 5 - 1e-9].max())
        got = s3m.abort_resumes(tt_, ai_, a_, vt=vt, Df=Df, verpm=ve)
        assert bool(got) is want, gap
        if want:
            x = list(got.values())[0]
            assert x['kind'] == 'stall' and x['abort_s'] == round(t_last, 3) and x['ai_end_s'] == round(t_last, 3)
            assert abs(x['gap_s'] - (5.0 - t_last)) < 0.006 and set(x) == {'kind', 'abort_s', 'gap_s', 'ai_end_s', 'human_s'}
    # 途切れの間ずっと動いている → 印なし。途切れの間に一度でも止まれば印
    keep = ~((t > 3.0) & (t < 5 - 1e-9))
    tt_, ai_, a_ = t[keep], ai[keep], a[keep]
    ve3 = -np.where(vt < 6.0, 1.0, 0.0) / 0.5e-3
    _, Df3 = le.cumulative_distance(vt, ve3)
    assert s3m.abort_resumes(tt_, ai_, a_, vt=vt, Df=Df3, verpm=ve3) == {}
    assert list(s3m.abort_resumes(tt_, ai_, a_, vt=vt, Df=Df, verpm=ve)) == [int(np.nonzero(tt_ >= 6)[0][0])]
    # 中立止め 0.33 秒(短い)のあとに途切れ 0.4 秒で人側 → 途切れの形(stall)。中立止め 0.83 秒のあとに途切れ → 中立止めの形(hold が先)
    keep = ~((t > 4.6) & (t < 5 - 1e-9))
    r4 = s3m.abort_resumes(t[keep], ai[keep], a[keep], hold=_seg(t, (4.3, 5))[keep], vt=vt, Df=Df, verpm=ve)
    assert [x['kind'] for x in r4.values()] == ['stall']
    keep = ~((t > 4.8) & (t < 5 - 1e-9))
    r5 = s3m.abort_resumes(t[keep], ai[keep], a[keep], hold=_seg(t, (4.0, 5))[keep], vt=vt, Df=Df, verpm=ve)
    assert [x['kind'] for x in r5.values()] == ['hold']
    # 中止(assist 3)が中立止めの前の最後なら中止の形(fix7 r1 と同じ。中立止め・途切れより先)
    a6 = a.copy()
    a6[(t >= 4.0) & (t < 4.4)] = 3
    r6 = s3m.abort_resumes(t, ai, a6, hold=_seg(t, (4.4, 5)), vt=vt, Df=Df, verpm=ve)
    assert list(r6) == [i1] and r6[i1]['kind'] == 'abort' and r6[i1]['abort_s'] == 4.0 and 'hold_s' not in r6[i1]
    # 続き: 中立止めのあとの走り直しで 0.3 m だけ進んでまた人側 → 次の区間も(carried 1・同じ止まり始め)
    v7 = np.where(vt < 4.0, 1.0, np.where((vt >= 6.5) & (vt < 7.0), 0.6, np.where(vt >= 9.3, 1.0, 0.0)))
    ve7 = -v7 / 0.5e-3
    _, Df7 = le.cumulative_distance(vt, ve7)
    r7 = s3m.abort_resumes(t, ai, a, hold=h, vt=vt, Df=Df7, verpm=ve7)
    assert sorted(r7) == [i1, i2] and r7[i2]['carried'] == 1 and r7[i2]['kind'] == 'hold'
    assert r7[i2]['abort_s'] == r7[i1]['abort_s'] and 'carried' not in r7[i1]
    # 中止の無い本で、verpm も無い呼び方・assist が無い呼び方
    assert s3m.abort_resumes(t, ai, None) == {}
    assert list(s3m.abort_resumes(t, ai, None, hold=h, vt=vt, Df=Df, verpm=ve)) == [i1]


# ================================================================== 本物の記録(読むだけ。無ければ飛ばす)
REAL = {'20261010-160306': 41.169, '20261024-183123': None, '20261010-181504': 37.704}
need_real = pytest.mark.skipif(not all((JF / 'data' / 'labels' / (lg + '.npz')).exists()
                                       and (JF / 'data' / 'raw' / lg / 'log.json').exists() for lg in REAL),
                               reason='本物の 10/10・10/24 の記録(ラベル)が無い')


@need_real
def test_real_marks_20261010_20261024():
    """本物(読むだけ): 20261010-160306 の 41.17 秒の発進に中立止めの印(中立止め 0.73 秒のまま人側 1.93 秒)。20261024-183123 の
    58.62 秒の発進は印なし(前の区間は置いた場所から一度も動いていない)。20261010-181504 の中止の印は r1 と同じ(kind のほか)"""
    s3n, s3o = mods('new')[0], mods('r1')[0]
    for lg, want in REAL.items():
        z = dict(np.load(JF / 'data' / 'labels' / (lg + '.npz')))
        lab = json.loads((JF / 'data' / 'labels' / (lg + '.json')).read_text(encoding='utf-8'))
        t = z['t_s'].astype(np.float64)
        ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
        V = le.load_log(lg, JF / 'data' / 'raw')['VEHICLE']
        vt = (V['timestamps_us'].astype(np.int64) - int(lab['t0_us'])) / 1e6
        ve = V['erpm'].astype(np.float64)
        _, Df = le.cumulative_distance(vt, ve)
        new = s3n.abort_resumes(t, ai, z['assist'], z['hold'], z['ai_cmd_select'], vt, Df, ve)
        old = s3o.abort_resumes(t, ai, z['assist'], z['hold'], z['ai_cmd_select'], vt, Df)
        marks = {round(float(t[i]), 3): v for i, v in new.items()}
        if lg == '20261010-160306':
            assert list(marks) == [want] and old == {}
            m = marks[want]
            assert m['kind'] == 'hold' and m['hold_s'] == 0.73 and m['human_s'] == 1.93 and m['abort_s'] == 38.504
        elif lg == '20261024-183123':
            assert marks == {} and old == {}
        else:
            assert list(marks) == [want] and {k: v for k, v in marks[want].items() if k != 'kind'} == list(old.values())[0]
            assert marks[want]['kind'] == 'abort'


@need_real
def test_real_s3_20261010_160306_counts_unchanged():
    """本物(読むだけ): ③ の 20261010-160306 の 41.17 秒の発進は、今は after_abort(中立止め)で数えない。前(r1)も「前に衝突」で数えない
    ので、数える発進・最速は変わらない(印の文だけ足す)"""
    s3n, s3o = mods('new')[0], mods('r1')[0]
    rn = s3n.analyze_log('20261010-160306', JF / 'data' / 'raw', JF / 'data' / 'labels')
    ro = s3o.analyze_log('20261010-160306', JF / 'data' / 'raw', JF / 'data' / 'labels')
    sn = [s for s in rn['standing'] if s['t_ai'] == 41.169]
    so = [s for s in ro['standing'] if s['t_ai'] == 41.169]
    assert len(sn) == len(so) == 1
    assert sn[0]['after_abort']['kind'] == 'hold' and sn[0]['counted'] is False and 'after_abort' not in so[0]
    assert so[0]['counted'] is False and so[0]['coll_before'] >= 1
    assert [s3n.counted(s) for s in rn['standing']] == [s3o.counted(s) for s in ro['standing']]
    assert s3n.best_standing(rn) == s3o.best_standing(ro)
