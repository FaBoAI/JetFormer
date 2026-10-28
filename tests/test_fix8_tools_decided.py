"""fix8 係 tools-decided(2026-10-28。main session の決め 10/28 00:5x = jetformer_reports/fix_20261026/user_decisions.md・
fix3/approvals.md 8 の「fix7 で残した読み方」①②)の試験。夜の道具だけ(⑤ experiments/20261026_fix/diag_launch/collect.py・
③ tools/standing_3lap.py・④ tools/quick_eval_by_params.py・② experiments/20261026_fix/plan/runs_1103.py・
experiments/20261026_fix/fix3/disk/disk_check.py)。走行の動き(PCA9685 に書く値)・既定値は変えていない(drive.py・assist.py・
recorder.py・server.py・app.js は触っていない)。

  ①  リカバリーの途中(assist 2。後退・gap・brake など)で人が CH3 を人側にした区間のあとの、同じ本の中の AI の区間の発進にも、fix7 の
      中止のあとの走り直しと同じ印(after_abort。kind 'recovery')を付け、⑤ の比べ・③ の止まってから 3 周・④ の「動く → A1」に入れない
      (ON の最初の周の衝突・空転・欠けは今までどおり数える)。名前は「リカバリー中止のあとの走り直し」のまま、形の文は
      「リカバリーの途中 X 秒から Y 秒」
  ②  試しの本・途中で変えた本の ON の発進も、空転の比と最初の周の記録の欠けを ⑤ の条件に数える(衝突と同じ。厳しい側)。比べ(起点まで
      などの中央値)には今まで通り入れない
  ③  ② runs_1103 の試験の選択肢の印の後ろの文・⑤ の混ざった注意の行に「わざと OFF にした日」(手順書 3 章)の読み方。options_line
      は本「—」も今まで通り区切りに並べ、本「—」から始まった区切りに回の本が続くときだけ「回の本は X(No.Y)から」を足す
  ④  disk_check.py の PYTEST_GB = 0.6(10/28 の実測。-o tmp_path_retention_policy=failed のとき。付けないと 7.1 = PYTEST_GB_KEEP)

どの試験も「直す前の写し(tests/fixtures/fix8_tools_decided_before/*.orig。sha256 を照らす)で起きる / 前と同じ」か「今の形」を見る。
作り物の記録だけ(tmp。作り方は tests/test_fix7_tools.py・test_fix7_tools_r2.py と同じ)。本物の 10/10〜10/24 の記録(data/raw・
data/labels。読むだけ)を読む試験は、無ければ飛ばす。
環境変数 JF_FIX8_COLLECT・JF_FIX8_S3・JF_FIX8_QP・JF_FIX8_RUNS・JF_FIX8_DISK で、試す版を差し替えられる(わざと壊した写しで試験が
落ちるかを見るとき。experiments/20261026_fix/fix8/decided_tools/mutate_fix8.py)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed \\
        --basetemp=<消せる場所> tests/test_fix8_tools_decided.py
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import re
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
import test_fix7_tools_r2 as t72  # noqa: E402  止まって終わる 3-8(stop_book)・空転(scale_imu)

FIX = JF / 'tests' / 'fixtures' / 'fix8_tools_decided_before'
NEW_C = Path(os.environ.get('JF_FIX8_COLLECT') or (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'))
NEW_S3 = Path(os.environ.get('JF_FIX8_S3') or (JF / 'tools' / 'standing_3lap.py'))
NEW_QP = Path(os.environ.get('JF_FIX8_QP') or (JF / 'tools' / 'quick_eval_by_params.py'))
NEW_RUNS = Path(os.environ.get('JF_FIX8_RUNS') or (JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'))
NEW_DISK = Path(os.environ.get('JF_FIX8_DISK') or (JF / 'experiments' / '20261026_fix' / 'fix3' / 'disk' / 'disk_check.py'))
VER = {'new': (NEW_C, NEW_S3, NEW_QP),
       'old': (FIX / 'collect.py.orig', FIX / 'standing_3lap.py.orig', FIX / 'quick_eval_by_params.py.orig')}
IG = ['--ignore-gain']
IGF = ['--ignore-gain', '--first-only']
PRM, ON = t7.PRM, t7.ON
ON200 = dict(ON, launch_us=200.0)
RN = 'リカバリー中止のあとの走り直し'      # 名前は fix7 r1 のまま(手順書 7 章・8.1・tests/test_fix7_docs_r2.py が引く言葉)
REC = 'リカバリーの途中'
REC_LINE = 'リカバリー(当たった・止まったあとのブレーキ・後退・前進など)の途中で人が CH3 を人側にした区間のあとも同じ'
REC_QP = ('リカバリー(当たった・止まったあとのブレーキ・後退・前進など)の途中で人が CH3 を人側にした区間のあとの発進も同じ'
          '(③ の表の「リカバリーの途中 X 秒から Y 秒」)。')
MARK_NEW = '(再起動で OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。手順書 3・7 章)'
MARK_OLD = '(再起動で OFF に戻った? 手順書 7 章)'
MIX_TAIL = '(3 章で止めて OFF にした本 X から先の OFF なら起こし直しではない = 数える。手順書 3・7 章)'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fixture_is_the_version_before():
    """直す前の写しの sha256(git の 9da6d0d の版)と中身(この直しの名前・欄が無い)"""
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sorted(sums) == ['car_check.py.orig', 'collect.py.orig', 'disk_check.py.orig', 'quick_eval_by_params.py.orig',
                            'runs_1103.py.orig', 'standing_3lap.py.orig']
    for name, h in sums.items():
        assert _sha(FIX / name) == h, name
    s3o = (FIX / 'standing_3lap.py.orig').read_text(encoding='utf-8')
    assert 'def abort_resumes' in s3o and 'ASSIST_RECOVERY' not in s3o and "'kind': 'recovery'" not in s3o
    assert "STOP_KIND_TEXT['recovery']" not in s3o and 'RECOVERY_NOTE' not in s3o
    co = (FIX / 'collect.py.orig').read_text(encoding='utf-8')
    assert 'abort_resume_slip' in co and '_slip_unmeasured' in co and 'midrun_slip' not in co and 'trial_slip' not in co
    assert 'gap_tag' not in co and MIX_TAIL not in co and 'RECOVERY_TEXT' not in co
    qo = (FIX / 'quick_eval_by_params.py.orig').read_text(encoding='utf-8')
    assert "('hold', 'stall')" in qo and "== 'recovery'" not in qo
    ro = (FIX / 'runs_1103.py.orig').read_text(encoding='utf-8')
    assert MARK_OLD in ro and 'OPT_MARK_NOTE' not in ro
    do = (FIX / 'disk_check.py.orig').read_text(encoding='utf-8')
    assert re.search(r'^PYTEST_GB = 3\.9 ', do, re.M) and 'PYTEST_GB_KEEP' not in do
    # car_check.py はこの直しで変えていない(写しと今が同じ)
    assert _sha(JF / 'tools' / 'car_check.py') == sums['car_check.py.orig']


# ================================================================== 版の読み方
_MODS = {}


def mods(ver):
    """(standing_3lap, quick_eval_by_params) の組(④ は同じ版の ③ を standing_3lap として読む)"""
    if ver not in _MODS:
        _, p3, pq = VER[ver]
        s3m = load_mod('standing_3lap_f8_%s' % ver, p3)
        saved = sys.modules.get('standing_3lap')
        sys.modules['standing_3lap'] = s3m
        try:
            qpm = load_mod('qp_f8_%s' % ver, pq)
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
    name = 'c8_%s_%s_%d' % (ver, tag or 'x', len(args))
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
st_of, on250 = t72.st_of, t72.on250


def same_outputs(D, args=(), tag=''):
    """⑤ の JSON・標準出力(出力の場所の行のほか)が直す前と今でバイト単位で同じか"""
    jn, on, tn = runc(D, 'new', args, 'n' + tag)
    jo, oo, to = runc(D, 'old', args, 'o' + tag)
    return tn == to and no_path(on) == no_path(oo)


# ================================================================== 作り物の記録
def set_assist(raw, log, a, b, code=2):
    """その本の AI_CMD の assist を a〜b 秒だけ code に(AI_CMD の select が AI 側で、中立止めでないフレームだけ。drive.py はリカバリーの
    フレームを assist 2 で書く)"""
    for p in sorted((Path(raw) / log).glob('%s-*/AI_CMD.npz' % log)):
        z = dict(np.load(p))
        tt = (z['timestamps_us'].astype(np.int64) - t7.T0_US) / 1e6
        m = (tt >= a - 1e-9) & (tt < b - 1e-9) & z['select'].astype(bool) & ~z['hold'].astype(bool)
        if m.any():
            asg = z['assist'].copy()
            asg[m] = code
            z['assist'] = asg
            np.savez(p, **z)


def rday(tmp, stop=3.5, b38=ON, rec=None, **kw):
    """回 3(tests/test_fix7_tools_r2.day の 7 本 + 3-8): 3-8 は 2 周目の R1 の stop m 手前で止まり、止まってからのリカバリー(assist 2。
    既定は止まり始めの 0.8 秒あと〜人側にするまで)の途中で CH3 を人側 1.0 秒 → その場で AI 側(走り直し。ON なら発進の前進がまた出る)。
    rec = (tt) → (始め, 終わり) でリカバリーの間を変える。kw は t72.day / stop_book(resume・hold・human_cmd・vk・ai・hits など)"""
    D, log, tt = t72.day(tmp, 'none', stop, b38=b38, **kw)
    a, b = rec(tt) if rec else (tt['t_st'] + 0.8, tt['t_flip'])
    set_assist(D.raw, log, a, b, 2)
    return D, log, tt


def edit_log(raw, log, fn):
    p = Path(raw) / log / 'log.json'
    j = json.loads(p.read_text(encoding='utf-8'))
    fn(j)
    p.write_text(json.dumps(j), encoding='utf-8')


def make_kind(raw, log, kind):
    """その本を「試し」(説明欄の memo に「試し」)か「途中で変えた」(log.json の params_end のステア 1.0 → 1.1)にする"""
    if kind == 'trial':
        edit_log(raw, log, lambda j: j.update(description=j['description'].replace('memo=x', 'memo=試し(ゲイン 1.02 で走った)')))
    elif kind == 'midrun':
        edit_log(raw, log, lambda j: j['drive'].update(params_end=dict(j['drive']['params'], steer_gain=1.1)))


def mtday(tmp, kind='trial', shape='spin', us=None, **kw):
    """回 3: OFF 4 本と ON 3 本(launch_us 200。tests/test_fix7_tools_r2.kday と同じ。ON の中央値 = OFF − 0.190 秒 → 使ってよい)+ 3-8 =
    kind(trial / midrun)の本の 1 回目の発進。shape: 'spin' 空転(IMU の前の加速を 0.55 倍 = 比 ≈ 1.8)/ 'nospin' / 'extreme'(IMU 0.05 倍 =
    いちばんひどい空転)/ 'unk'(ゆっくり出て IMU 0.3 倍 = 比が出ない)/ 'gap'(14 秒に AI 側・最初の周の途中のシーン 0002 の IMU.npz が
    無い)/ 'off_spin'(OFF の本の空転)/ 'noreach'(ON・3 秒で人側 = R1 に届かない)。us = {block: launch_us}。kw は 3-8 の write_log
    (hits など)"""
    us = us or {}
    D = t7.Day(tmp)
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON200, 0.15, 0.3), (14, PRM, 0.21, 0.6), (15, ON200, 0.20, 0.3),
                               (16, PRM, 0.22, 0.6), (17, ON200, 0.28, 0.3), (18, PRM, 0.23, 0.6)):
        D.add(blk, dict(prm, launch_us=us[blk]) if blk in us else prm, dm, ramp)
    p38 = PRM if shape == 'off_spin' else dict(ON200, launch_us=us.get(19, 200.0))
    if shape == 'gap':
        log = D.add(19, p38, 0.15, 0.3, dur=45.0, vk=t7.launch_keys(14.0, 0.15, 0.3, until=45.0), ai=[(14.0, 43.0)],
                    launch=[(14.0, 0.5)], **kw)
        (D.raw / log / ('%s-0002' % log) / 'IMU.npz').unlink()
    elif shape == 'unk':
        log = D.add(19, p38, 0.15, 3.0, **kw)
        t72.scale_imu(D.raw, log, 1.0, 2.5, 0.3)
    elif shape == 'noreach':
        log = D.add(19, p38, 0.15, 0.3, dur=8.0, vk=t7.launch_keys(1.0, 0.15, 0.3) + [(3.0, 2.0), (4.0, 0.0), (8.0, 0.0)],
                    ai=[(1.0, 3.0)], **kw)
        t72.scale_imu(D.raw, log, 1.0, 2.2, 0.55)
    else:
        log = D.add(19, p38, 0.15, 0.3, **kw)
        if shape in ('spin', 'off_spin'):
            t72.scale_imu(D.raw, log, 1.0, 2.2, 0.55)
        elif shape == 'extreme':
            t72.scale_imu(D.raw, log, 1.0, 2.2, 0.05)
    make_kind(D.raw, log, kind)
    return D, log


def rec_line(out):
    return [x for x in out.splitlines() if x.startswith('  %s ' % RN)]


# ================================================================== ① abort_resumes だけ(速い試験を先に)
def test_abort_resumes_recovery_unit():
    """abort_resumes: 区間の最後のフレーム(中立止め・AI_CMD が人側を除く)がリカバリーの途中(assist 2)なら次の区間に印(kind recovery・
    abort_s = 最後の続いたリカバリーの始め・rec_s = そこから区間の終わりまで)。中止・中立止めの印が先。verpm が無い呼び方でも見る"""
    s3m = mods('new')[0]
    import jetformer.assist as A
    assert s3m.ASSIST_RECOVERY == A.ASSIST_RECOVERY == 2
    t = np.arange(0, 20, 0.1)
    ai = (t < 5) | ((t >= 6) & (t < 8)) | (t >= 9)
    a = np.zeros(len(t), np.int8)
    vt = t.copy()
    Df = t * 1.0
    i1 = int(np.nonzero(t >= 6)[0][0])
    i2 = int(np.nonzero(t >= 9)[0][0])
    assert s3m.abort_resumes(t, ai, a) == {}
    a[(t >= 4) & (t < 5 - 1e-9)] = 2                 # 1 つ目の区間はリカバリーの途中で人側
    want = {'kind': 'recovery', 'abort_s': 4.0, 'rec_s': 0.9, 'ai_end_s': 4.9, 'human_s': 1.1}
    assert s3m.abort_resumes(t, ai, a, vt=vt, Df=Df) == {i1: want}
    assert s3m.abort_resumes(t, ai, a) == {i1: want}                         # verpm・vt・Df が無い呼び方でも
    assert mods('old')[0].abort_resumes(t, ai, a, vt=vt, Df=Df) == {}         # 直す前は印なし
    # 区間の中に前のリカバリー(解けた)があっても、abort_s は最後の続いたリカバリーの始め
    a2 = a.copy()
    a2[(t >= 2) & (t < 3)] = 2
    assert s3m.abort_resumes(t, ai, a2, vt=vt, Df=Df) == {i1: want}
    # リカバリーが解けてから人側(最後のフレームは assist 0)= 印なし
    a3 = a.copy()
    a3[(t >= 4.5) & (t < 5)] = 0
    assert s3m.abort_resumes(t, ai, a3, vt=vt, Df=Df) == {}
    # 最後の 2 フレームが中立止め・AI_CMD が人側(assist 0): 除いて見る。rec_s は区間の終わりまで
    a4, h = a.copy(), np.zeros(len(t), bool)
    a4[[48, 49]] = 0
    h[[48, 49]] = True
    assert s3m.abort_resumes(t, ai, a4, hold=h, vt=vt, Df=Df) == {i1: want}
    assert s3m.abort_resumes(t, ai, a4, vt=vt, Df=Df) == {}                  # 除かなければ見落とす
    sel = np.ones(len(t), bool)
    sel[[48, 49]] = False
    assert s3m.abort_resumes(t, ai, a4, sel=sel, vt=vt, Df=Df) == {i1: want}
    # 中止が先(最後のフレームが assist 3)
    a5 = a.copy()
    a5[49] = 3
    assert s3m.abort_resumes(t, ai, a5, vt=vt, Df=Df)[i1]['kind'] == 'abort'
    # 中立止め(0.5 秒以上・止まっている・0.5 m 以上進んだ)が先: kind hold。verpm が無ければリカバリー
    a6, h6 = a.copy(), (t >= 4.4) & (t < 5 - 1e-9)
    a6[h6] = 0
    ve = np.where(t < 4.4, -2000.0, 0.0)
    Df6 = np.minimum(t, 4.4)
    r6 = s3m.abort_resumes(t, ai, a6, hold=h6, vt=vt, Df=Df6, verpm=ve)
    assert r6[i1]['kind'] == 'hold' and r6[i1]['hold_s'] == 0.6
    r6b = s3m.abort_resumes(t, ai, a6, hold=h6, vt=vt, Df=Df6)
    assert r6b[i1] == {'kind': 'recovery', 'abort_s': 4.0, 'rec_s': 0.9, 'ai_end_s': 4.9, 'human_s': 1.1}
    # 走り直しの区間で 0.5 m も進まない → 次の区間も(carried 1)
    Df7 = np.where(t < 8.5, np.minimum(t, 5.0), 5.0 + (t - 8.5))
    r7 = s3m.abort_resumes(t, ai, a, vt=vt, Df=Df7)
    assert sorted(r7) == [i1, i2] and r7[i2] == dict(want, carried=1) and 'carried' not in r7[i1]
    # 走り直しの区間がまたリカバリーの途中で終わった → 次の区間は新しい印(続きではない)
    a8 = a.copy()
    a8[(t >= 7.5) & (t < 8 - 1e-9)] = 2
    r8 = s3m.abort_resumes(t, ai, a8, vt=vt, Df=Df7)
    assert r8[i2] == {'kind': 'recovery', 'abort_s': 7.5, 'rec_s': 0.4, 'ai_end_s': 7.9, 'human_s': 1.1}


def test_abort_text_recovery():
    """形の文: ③ は「リカバリーの途中 X 秒から Y 秒」(秒は 2 桁・1 桁)、⑤ は印の値のまま。ほかの形の文は前と同じ"""
    s3m, s3o = mods('new')[0], mods('old')[0]
    ab = {'kind': 'recovery', 'abort_s': 4.0, 'rec_s': 0.9}
    assert s3m.abort_text(ab) == 'リカバリーの途中 4.00 秒から 0.9 秒'
    assert s3m.STOP_KIND_TEXT == dict(s3o.STOP_KIND_TEXT, recovery=REC)
    for x in ({'kind': 'abort', 'abort_s': 3.0}, {'kind': 'hold', 'abort_s': 3.0, 'hold_s': 0.7}, {'abort_s': 3.0},
              {'kind': 'stall', 'abort_s': 3.0, 'gap_s': 0.3}):
        assert s3m.abort_text(x) == s3o.abort_text(x)
    cm, co = load_mod('c8_text_new', NEW_C), load_mod('c8_text_old', VER['old'][0])
    assert cm.abort_text(ab) == 'リカバリーの途中 4.0 秒から 0.9 秒' and cm.RECOVERY_TEXT == s3m.STOP_KIND_TEXT['recovery']
    for x in ({'kind': 'abort', 'abort_s': 3.0}, {'kind': 'hold', 'abort_s': 3.0, 'hold_s': 0.7}):
        assert cm.abort_text(x) == co.abort_text(x)


# ================================================================== ④ disk_check(速い)
def _du_20261027():
    t = (JF.parent / 'jetformer_reports' / 'fix_20261026' / 'fix6' / 'pytest_full_20261027.txt').read_text(encoding='utf-8')
    return re.findall(r'^(\d+(?:\.\d+)?)G\texperiments/20261026_fix/tmp_pytest_full_20261027$', t, re.M)[-1]


def test_disk_pytest_gb_matches_the_measurement():
    """PYTEST_GB = 10/28 の一式の実測(空き 10493M → いちばん少なくて 9913M = 580 MiB ≈ 0.6 GB。-o tmp_path_retention_policy=failed)、
    PYTEST_GB_KEEP = 10/27 の付けない形の実測 7.1 GB。説明の文に両方の数とオプション"""
    m = load_mod('disk_f8', NEW_DISK)
    rep = (JF.parent / 'jetformer_reports' / 'fix_20261026' / 'fix7' / 'pytest_full_20261028.txt').read_text(encoding='utf-8')
    assert '空きの一番少ないとき 9913M' in rep and 'tmp_path_retention_policy=failed' in rep
    assert m.PYTEST_GB == round((10493 - 9913) * 2 ** 20 / 1e9, 1) == 0.6
    assert m.PYTEST_GB_KEEP == float(_du_20261027()) == 7.1
    src = NEW_DISK.read_text(encoding='utf-8')
    head = src.split('PYTEST_GB = ')[0].rsplit('# 見積もりの数', 1)[1]
    for w in ('10493M', '9913M', '-o tmp_path_retention_policy=failed', '7.1 GB', 'fix7/pytest_full_20261028.txt',
              'fix6/pytest_full_20261027.txt', 'このオプションが要る'):
        assert w in head, w
    assert re.search(r'^PYTEST_GB = (\d+(?:\.\d+)?)', src, re.M).group(1) == '0.6'      # tests/test_fix7_docs_r1.py の f5 の読み方


def test_disk_print_line(tmp_path, monkeypatch, capsys):
    """「pytest 一式の最中」の行に、作業場所 0.6 GB(オプションあり)と付けないときの 7.1 GB の両方(作り物の空き 10 GB・空の場所)"""
    m = load_mod('disk_f8p', NEW_DISK)
    for k in ('RUNS', 'RAW', 'REP'):
        (tmp_path / k).mkdir()
        setattr(m, k, str(tmp_path / k))
    m.du = lambda p: 0
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: type('U', (), {'free': 10.0e9})())
    assert m.main() == 0
    out = capsys.readouterr().out
    left = 12 * m.RUN_GB + 12 * m.PRED_GB
    after1 = 10.0 - left
    want = ('pytest 一式の最中: %.2f GB(作業場所 0.6 GB。-o tmp_path_retention_policy=failed を付けたとき。付けないと作業場所 7.1 GB で '
            '%.2f GB)。終わって消せば %.2f GB に戻る' % (after1 - 0.6, after1 - 7.1, after1))
    assert want in out.splitlines()
    assert '朝 %.1f・夜' % (after1 - m.OTHER_GB[0] - m.E5_GB[0] - 0.6) in out       # 作業場所を残すと(0.6 GB)


# ================================================================== ③ 試験の選択肢の印の文(② runs_1103・⑤ collect)
def _rows(R, plan):
    return [{'log': '20261103-1%05d' % i, 'run_id': rid, 'ai_no': str(no), 'options': dict(o), 'diffs': []}
            for i, (rid, no, o) in enumerate(plan)]


def test_runs_mark_text_tells_deliberate_off():
    """② の「回 N の試験の選択肢が … と違う」の後ろの文: 「(再起動で OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。
    手順書 3・7 章)」(前は「(再起動で OFF に戻った? 手順書 7 章)」)。頭は手順書 3 章が引く「(再起動で OFF に戻った? …)」のまま"""
    on = {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
    off = {'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}
    plan = [('1-1', 2, on), ('1-2', 3, on), ('1-3', 4, off), ('2-1', 5, off)]
    Rn, Ro = load_mod('runs_f8_new', NEW_RUNS), load_mod('runs_f8_old', FIX / 'runs_1103.py.orig')
    rn, ro = _rows(Rn, plan), _rows(Ro, plan)
    Rn.round_options(rn)
    Ro.round_options(ro)
    head = '回 1 の試験の選択肢が 1-1(No.2)と違う: 後ろの見張り OFF(1-1 は ON)・リカバリーを残す OFF(1-1 は ON)'
    assert rn[2]['diffs'] == [head + MARK_NEW] and ro[2]['diffs'] == [head + MARK_OLD]
    assert [r['diffs'] for r in rn if r is not rn[2]] == [[], [], []]
    assert Rn.OPT_MARK_NOTE == MARK_NEW and MARK_NEW.startswith('(再起動で OFF に戻った? ')
    rb = (JF.parent / 'jetformer_reports' / 'fix_20261026' / 'runbook_20261103.md').read_text(encoding='utf-8')
    assert '「回 N の試験の選択肢が … と違う(再起動で OFF に戻った? …)」' in rb and 'わざと OFF にした日' in rb


def test_runs_options_line_unchanged(tmp_path, monkeypatch):
    """options_line の朝の形は変えていない: 本「—」(朝の発進の前進の確かめを収集したまま・memo に「試し」)も区切りに並べる(手順書 3 章 5・
    8.1 ②・tests/test_fix7_docs_r2.py::test_g3)。表と行は直す前とバイト単位で同じ(印の後ろの文のほか)"""
    base = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
            'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0,
            'launch_no_kick': 0}
    raw = tmp_path / 'raw'
    for name, desc, kw in (('20261103-081500', 'ai_no=1 memo=試し', {'launch_boost': 1}),
                           ('20261103-083000', 'day=2026-11-03 block=1 official=1 ai_no=2 battery=#A', {'rec_back_guard': 1, 'rec_hold_keep': 1}),
                           ('20261103-084000', 'day=2026-11-03 block=2 official=1 ai_no=3 battery=#A', {'rec_back_guard': 1, 'rec_hold_keep': 1}),
                           ('20261103-090000', 'day=2026-11-03 block=3 official=1 ai_no=4 battery=#A', {})):
        d = raw / name
        d.mkdir(parents=True)
        (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
            'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': dict(base, **kw), 'dry_run': False}}), encoding='utf-8')
    outs = {}
    for ver, path in (('new', NEW_RUNS), ('old', FIX / 'runs_1103.py.orig')):
        m = load_mod('runs_f8_l_%s' % ver, path)
        m.RAW = raw
        monkeypatch.setattr(sys, 'argv', ['runs_1103.py', '--day', '20261103'])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main()
        outs[ver] = buf.getvalue()
    line = [x for x in outs['new'].splitlines() if x.startswith('試験の選択肢(')]
    assert len(line) == 1 and line[0].endswith('20261103-081500(No.1)から なし → 1-1(No.2)から 後ろの見張り・リカバリーを残す'
                                               '(前の本と違う) → 1-3(No.4)から なし(前の本と違う)')
    assert outs['new'].replace(MARK_NEW, MARK_OLD) == outs['old']       # 違うのは印の後ろの文だけ
    assert MARK_NEW in outs['new']


def _options_line(rows_spec, tmp, monkeypatch):
    """作り物の log.json(rows_spec = [(記録の名前, 説明欄, 試験の選択肢)])を今と直す前の runs_1103 に通した表の下の試験の選択肢の行"""
    base = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
            'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0,
            'launch_no_kick': 0}
    raw = tmp / 'raw'
    for name, desc, kw in rows_spec:
        d = raw / name
        d.mkdir(parents=True)
        (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
            'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': dict(base, **kw), 'dry_run': False}}), encoding='utf-8')
    out = {}
    for ver, path in (('new', NEW_RUNS), ('old', FIX / 'runs_1103.py.orig')):
        m = load_mod('runs_f8_d_%s_%d' % (ver, len(rows_spec)), path)
        m.RAW = raw
        monkeypatch.setattr(sys, 'argv', ['runs_1103.py', '--day', '20261103'])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main()
        out[ver] = [x for x in buf.getvalue().splitlines() if x.startswith('試験の選択肢(')]
    return out


def test_runs_options_line_names_the_round_run_after_dash(tmp_path, monkeypatch):
    """本「—」から始まった区切りに回の本が続いたら、その区切りの後ろに「回の本は X(No.Y)から」(昼に本「—」のあと起こし直しのまま 3-1 を
    走った日。前は区切りが本「—」の名前だけで、手順書 8.1 ② の朝の「本「—」から始まる「なし」の区切り」と読み違えうる)。本「—」が最初で
    選択肢が同じまま回の本が続くときも。朝の手順どおり(本「—」のあと選択肢を入れて別の区切り)は前と同じ"""
    G = {'rec_back_guard': 1, 'rec_hold_keep': 1}
    o = _options_line([('20261103-083000', 'day=2026-11-03 block=1 official=1 ai_no=2', G),
                       ('20261103-120000', 'ai_no=9 memo=試し', {}),
                       ('20261103-121000', 'day=2026-11-03 block=12 official=1 ai_no=10', {})], tmp_path / 'd', monkeypatch)
    tail = '1-1(No.2)から 後ろの見張り・リカバリーを残す → 20261103-120000(No.9)から なし(前の本と違う'
    assert o['new'][0].endswith(tail + '。回の本は 3-1(No.10)から)') and o['old'][0].endswith(tail + ')')
    o2 = _options_line([('20261103-081500', 'ai_no=1 memo=試し', G),
                        ('20261103-083000', 'day=2026-11-03 block=1 official=1 ai_no=2', G),
                        ('20261103-084000', 'day=2026-11-03 block=2 official=1 ai_no=3', G)], tmp_path / 'f', monkeypatch)
    assert o2['new'][0].endswith(': 20261103-081500(No.1)から 後ろの見張り・リカバリーを残す(回の本は 1-1(No.2)から)')
    assert o2['old'][0].endswith(': 20261103-081500(No.1)から 後ろの見張り・リカバリーを残す')
    o3 = _options_line([('20261103-081500', 'ai_no=1 memo=試し', {}),
                        ('20261103-083000', 'day=2026-11-03 block=1 official=1 ai_no=2', G)], tmp_path / 'm', monkeypatch)
    assert o3['new'] == o3['old'] and o3['new'][0].endswith(
        '20261103-081500(No.1)から なし → 1-1(No.2)から 後ろの見張り・リカバリーを残す(前の本と違う)')
    o4 = _options_line([('20261103-081500', 'ai_no=1', G), ('20261103-082000', 'ai_no=2', {}), ('20261103-083000', 'ai_no=3', {})],
                       tmp_path / 'n', monkeypatch)     # 計画の本の無い走行だけ(全部本「—」)は前と同じ
    assert o4['new'] == o4['old']


def test_c5_mixed_options_line_tells_deliberate_off(tmp_path):
    """⑤ の「注意: 試験の選択肢の ON/OFF が混ざっている」の行の後ろに、わざと OFF にした日の読み方(JSON は直す前と同じ)"""
    D = t7.Day(tmp_path)
    D.add(12, PRM, 0.20, 0.6)
    D.add(13, ON, 0.15, 0.3)
    D.add(14, dict(PRM, rec_back_guard=0, rec_hold_keep=0), 0.21, 0.6)
    D.add(15, dict(ON, rec_back_guard=0, rec_hold_keep=0), 0.20, 0.3)
    jn, on, tn = runc(D, 'new', IG, 'mix')
    jo, oo, to = runc(D, 'old', IG, 'mix')
    assert tn == to
    ln = [x for x in on.splitlines() if x.startswith('  注意: 試験の選択肢の ON/OFF が混ざっている(')]
    lo = [x for x in oo.splitlines() if x.startswith('  注意: 試験の選択肢の ON/OFF が混ざっている(')]
    assert len(ln) == len(lo) == 1 and ln[0] == lo[0] + MIX_TAIL and lo[0].endswith('再起動で OFF に戻った本が無いか ② runs_1103 で見る')
    assert no_path(on).replace(MIX_TAIL, '') == no_path(oo)


# ================================================================== ① ⑤: 直す前の再現
@pytest.mark.parametrize('stop', [3.5, 4.5])
def test_old_recovery_resume_turns_to_use_ok(tmp_path, stop):
    """直す前: リカバリーの途中で人側 → その場で AI 側の走り直し(R1 の stop m 手前から)に印が付かず、⑤ ON n 5・−0.165 秒「使ってよい」、
    ③ counted、④ の発進 250 の組の止まってから 3 周に 27.12 秒"""
    D, log, _ = rday(tmp_path, stop)
    j, o, _ = runc(D, 'old', IG, 'rep')
    g = g0(j)
    assert g['by_launch']['on']['n'] == 5 and g['compare']['d_to_r1_s'] == -0.165 and g['compare']['all_ok'] is True
    assert '→ 使ってよい' in o and all('after_abort' not in r for r in j['launches'])
    R, md, qmd, runs, groups = run_s3(D, 'old')
    s2 = st_of(R, log)[1]
    assert s2['counted'] is True and 'after_abort' not in s2
    assert on250(groups)['standing_n'] == 1 and abs(on250(groups)['standing_best_s'] - 27.12) < 0.01


# ================================================================== ① ⑤: 今
@pytest.mark.parametrize('stop', [0.5, 1.5, 2.5, 3.5, 4.5])
def test_c5_recovery_resume_is_excluded(tmp_path, stop):
    """今: 走り直しに after_abort(kind recovery)。⑤ は比べから外して ON 4 本対 OFF 4 本の −0.125 秒「満たさない」(中止で止まった日と
    同じ数)。組の行(説明の文にリカバリーの 1 文)・発進の行に「リカバリーの途中 X 秒から Y 秒」"""
    D, log, tt = rday(tmp_path, stop)
    j, o, _ = runc(D, 'new', IG, 'rec')
    g = g0(j)
    assert g['by_launch']['on']['n'] == 4 and g['by_launch']['on']['restarts'] == 0
    assert g['by_launch']['on']['to_r1_median_s'] == 2.903 and g['compare']['d_to_r1_s'] == -0.125
    assert g['compare']['all_ok'] is False and g['compare']['checks']['to_r1'] is False and '→ 使ってよい' not in o
    r1, r2 = rows_of(j, log)
    assert 'after_abort' not in r1
    ab = r2['after_abort']
    assert set(ab) == {'kind', 'abort_s', 'rec_s', 'ai_end_s', 'human_s'} and ab['kind'] == 'recovery'
    assert abs(ab['abort_s'] - (tt['t_st'] + 0.8)) < 0.04 and abs(ab['ai_end_s'] - tt['t_flip']) < 0.04
    assert abs(ab['rec_s'] - (ab['ai_end_s'] - ab['abort_s'])) < 0.006 and abs(ab['human_s'] - 1.0) < 0.07
    txt = '%s %s 秒から %s 秒' % (REC, ab['abort_s'], ab['rec_s'])
    assert len(g['abort_resume_excluded']) == 1
    assert g['abort_resume_excluded'][0].startswith('%s@%.2f(on・%s・人側 %s 秒・起点まで %.3f 秒・空転の' % (
        log, r2['t_ai'], txt, ab['human_s'], r2['to_r1_s']))
    line = rec_line(o)
    assert len(line) == 1 and line[0].startswith('  %s 1 本(' % RN) and log in line[0]
    assert REC_LINE in line[0] and 'ON の組にも OFF の組にも入れない' in line[0] and '手順書 7 章' in line[0]
    assert any(x.startswith(log) and x.endswith(' 中止のあとの走り直し(%s・人側 %s 秒。比べに入れない)' % (txt, ab['human_s']))
               for x in o.splitlines())


def test_c5_recovery_same_numbers_as_abort(tmp_path):
    """リカバリーの途中で人側にしたあとの走り直しの日の比べ(by_launch・compare)は、中止で止まった日・走り直さない日と同じ"""
    Dr, _, _ = rday(tmp_path / 'r', 3.5)
    Da, _, _ = t72.day(tmp_path / 'a', 'abort', 3.5)
    Dn, _, _ = rday(tmp_path / 'n', 3.5, resume=False)
    gr, ga, gn = (g0(runc(D_, 'new', IG, t_)[0]) for D_, t_ in ((Dr, 'r'), (Da, 'a'), (Dn, 'n')))
    for key in ('by_launch', 'compare'):
        assert gr[key] == ga[key] == gn[key], key


def test_c5_recovery_launch_rows_same_but_mark(tmp_path):
    """発進ごとの行(launches)は after_abort のほか直す前と同じ。走り直さない日(印なし)は JSON・標準出力ともバイト単位で同じ"""
    D, log, _ = rday(tmp_path / 'b', 3.5)
    jn, _, _ = runc(D, 'new', IG, 'lr')
    jo, _, _ = runc(D, 'old', IG, 'lr')
    assert [{k: v for k, v in r.items() if k != 'after_abort'} for r in jn['launches']] == jo['launches']
    Dn, _, _ = rday(tmp_path / 'n', 3.5, resume=False)
    for args in ([], IG, IGF):
        assert same_outputs(Dn, args, 'nr%d' % len(args))


def test_c5_recovery_on_first_lap_collision_counts(tmp_path):
    """ON の走り直しの最初の周の衝突(6G)は「最初の周の衝突 0」に数える(abort_resume_on_collisions。比べの数は 4 本のまま)"""
    D, log, tt = rday(tmp_path, 3.5, hits=lambda tt: [(tt['t_res'] + 2.0, 6.0)])
    j, o, _ = runc(D, 'new', IG, 'hit')
    g = g0(j)
    assert g['by_launch']['on']['n'] == 4 and g.get('abort_resume_on_collisions') == 1
    assert g['compare']['checks']['first_lap_collisions_0'] is False and g['compare']['all_ok'] is False
    assert '・最初の周の衝突 ' in g['abort_resume_excluded'][0]


def test_c5_recovery_resume_not_reached(tmp_path):
    """R1 に届かなかった走り直し(1 秒で人側): 前と同じ「R1 に届かなかった発進(on)」で満たさない + 行の後ろに
    「。うちリカバリー中止のあとの走り直し 1(」。ほかは直す前と同じ"""
    D, log, tt = rday(tmp_path, 3.5,
                      vk=lambda tt: t7.launch_keys(1.0, 0.31, 0.3) + [(tt['t_st'], 2.0), (tt['t_stop'], 0.0), (tt['t_res'] + 0.15, 0.0),
                                                                     (tt['t_res'] + 0.45, 1.0), (tt['t_res'] + 1.0, 1.0),
                                                                     (tt['t_res'] + 2.0, 0.0), (tt['t_res'] + 3.0, 0.0)],
                      ai=lambda tt: [(1.0, tt['t_flip']), (tt['t_res'], tt['t_res'] + 1.0)], dur=lambda tt: tt['t_res'] + 3.0)
    jn, on, _ = runc(D, 'new', IG, 'nr')
    jo, oo, _ = runc(D, 'old', IG, 'nr')
    gn, go = g0(jn), g0(jo)
    nr = gn['not_reached']['on']
    assert nr['after_abort'] == ['%s@%.2f' % (log, rows_of(jn, log)[1]['t_ai'])] and gn['compare']['all_ok'] is False
    assert {k: v for k, v in nr.items() if k != 'after_abort'} == go['not_reached']['on']
    assert gn['compare'] == go['compare'] and gn['by_launch'] == go['by_launch']
    assert any('R1 に届かなかった発進(on)' in x and '。うち%s 1(' % RN in x for x in on.splitlines())


def test_c5_off_recovery_resume_is_excluded_from_off(tmp_path):
    """OFF の本の走り直しは OFF の組から外れる(6 → 5 本・2 回目以降 1 → 0)。衝突・空転は ON の条件に数えない"""
    D, log, _ = rday(tmp_path, 3.5, b38=PRM)
    jn, _, _ = runc(D, 'new', IG, 'off')
    jo, _, _ = runc(D, 'old', IG, 'off')
    gn, go = g0(jn), g0(jo)
    assert go['by_launch']['off']['n'] == 6 and go['by_launch']['off']['restarts'] == 1
    assert gn['by_launch']['off']['n'] == 5 and gn['by_launch']['off']['restarts'] == 0
    assert '(off・%s ' % REC in gn['abort_resume_excluded'][0] and 'abort_resume_slip' not in gn


def test_c5_first_only_unchanged(tmp_path):
    """--first-only: 走り直し(2 回目の発進)はもとから入らない。まとめは直す前と同じ"""
    D, log, _ = rday(tmp_path, 3.5)
    jn, _, _ = runc(D, 'new', IGF, 'fo')
    jo, _, _ = runc(D, 'old', IGF, 'fo')
    assert jn['summary'] == jo['summary']


@pytest.mark.parametrize('tail', ['hold', 'human'])
def test_c5_recovery_tail_hold_or_human(tmp_path, tail):
    """区間の最後のフレームが短い中立止め(0.1 秒)・AI_CMD が先に人側(2 フレーム)でも、除いて見るので印が付く(直す前は印なし)"""
    kw = ({'hold': lambda tt: [(tt['t_flip'] - 0.1, tt['t_flip'])]} if tail == 'hold'
          else {'human_cmd': lambda tt: [(tt['t_flip'] - 0.07, tt['t_flip'])]})
    D, log, tt = rday(tmp_path, 3.5, **kw)
    jn, _, _ = runc(D, 'new', IG, tail)
    jo, _, _ = runc(D, 'old', IG, tail)
    assert rows_of(jn, log)[1]['after_abort']['kind'] == 'recovery' and 'after_abort' not in rows_of(jo, log)[1]
    assert g0(jn)['compare']['d_to_r1_s'] == -0.125 and g0(jo)['compare']['d_to_r1_s'] == -0.165


def test_c5_recovery_resolved_is_not_marked(tmp_path):
    """リカバリーが解けて(assist 0 のフレーム 0.6 秒)から人側にした本は印なし: JSON・標準出力とも直す前とバイト単位で同じ"""
    D, log, _ = rday(tmp_path, 3.5, rec=lambda tt: (tt['t_st'] + 0.8, tt['t_flip'] - 0.6))
    jn, _, _ = runc(D, 'new', IG, 'res')
    assert 'after_abort' not in rows_of(jn, log)[1]
    for args in ([], IG):
        assert same_outputs(D, args, 'res%d' % len(args))


@pytest.mark.parametrize('first', ['hold', 'abort'])
def test_c5_fix7_marks_come_first(tmp_path, first):
    """リカバリーの途中のあとに中立止め(0.7 秒・止まって)/ 中止(0.5 秒)で終わった区間: fix7 の印(hold / abort)が先 = 直す前と出力が同じ"""
    if first == 'hold':
        D, log, _ = rday(tmp_path, 3.5, hold=lambda tt: [(tt['t_flip'] - 0.7, tt['t_flip'])])
    else:
        D, log, _ = rday(tmp_path, 3.5, rec=lambda tt: (tt['t_st'] + 0.8, tt['t_flip'] - 0.5),
                         giveup=lambda tt: [(tt['t_flip'] - 0.5, tt['t_flip'])])
    jn, _, _ = runc(D, 'new', IG, first)
    assert rows_of(jn, log)[1]['after_abort']['kind'] == first
    for args in ([], IG):
        assert same_outputs(D, args, first + str(len(args)))


@pytest.mark.parametrize('kind', ['N', 'abort', 'hold', 'stall', 'none'])
def test_c5_no_recovery_same_as_before(tmp_path, kind):
    """リカバリーの途中で終わる区間の無い日(ふつうの回 3・中止・中立止め・カメラの途切れ・止まって人側だけ)は、⑤ の JSON・標準出力が
    直す前とバイト単位で同じ(3 通りの引数)"""
    if kind == 'N':
        D, _, _ = t7.round3(tmp_path, 'N')
    else:
        D, _, _ = t72.day(tmp_path, kind, 3.5)
    for args in ([], IG, IGF):
        assert same_outputs(D, args, kind + str(len(args))), args


# ================================================================== ① ③④
@pytest.mark.parametrize('stop', [3.5, 1.5])
def test_s3_recovery_resume_not_counted(tmp_path, stop):
    """③: 走り直しは counted = False・after_abort(kind recovery)。表に「数えない(リカバリー中止のあとの走り直し。リカバリーの途中 X 秒から
    Y 秒・人側 Z 秒)」と注 RECOVERY_NOTE(中立止めの注は出さない)。④: 止まってから 3 周 0 本・数えない 1・1 章の印・動く → A1 から外して
    2 章の下の行(リカバリーの 1 文)"""
    D, log, tt = rday(tmp_path, stop)
    s3m, _ = mods('new')
    R, md, qmd, runs, groups = run_s3(D, 'new')
    s1, s2 = st_of(R, log)
    assert 'after_abort' not in s1 and s2['valid'] and s2['counted'] is False and s2['after_abort']['kind'] == 'recovery'
    ab = s2['after_abort']
    assert '・数えない(%s。%s %.2f 秒から %.1f 秒・人側 %.1f 秒)' % (RN, REC, ab['abort_s'], ab['rec_s'], ab['human_s']) in md
    assert s3m.ABORT_NOTE in md and s3m.RECOVERY_NOTE in md and s3m.HOLD_NOTE not in md
    assert s3m.RECOVERY_NOTE.startswith('- 「リカバリーの途中 X 秒から Y 秒」の%s = ' % RN)
    assert s3m.best_standing([x for x in R if x['log'] == log][0]) is None
    g = on250(groups)
    assert g['standing_n'] == 0 and g['standing_excluded_n'] == 1 and g['launch_to_r1_abort_excluded'] == ['%s@%.2f' % (log, s2['t_ai'])]
    run = [x for x in runs if x['log'] == log][0]
    assert '数えない発進 1(%s)' % RN in run['flags'] and not any('前に衝突' in f for f in run['flags'])
    ql = [x for x in qmd.splitlines() if x.startswith('- %s(' % RN)]
    assert len(ql) == 1 and log in ql[0] and ql[0].endswith(REC_QP) and '中立で停止中' not in ql[0]


def test_s3_recovery_strip_equals_old(tmp_path):
    """③: after_abort と counted のほかの値は直す前と同じ"""
    D, log, _ = rday(tmp_path, 3.5)
    Rn, Ro = run_s3(D, 'new')[0], run_s3(D, 'old')[0]
    for rn, ro in zip(Rn, Ro):
        assert rn['log'] == ro['log']
        assert [{k: v for k, v in s.items() if k not in ('after_abort', 'counted')} for s in rn.get('standing') or []] == \
            [{k: v for k, v in s.items() if k != 'counted'} for s in ro.get('standing') or []]
        assert {k: v for k, v in rn.items() if k != 'standing'} == {k: v for k, v in ro.items() if k != 'standing'}


@pytest.mark.parametrize('kind', ['N', 'abort', 'hold'])
def test_s3_qp_no_recovery_same_as_before(tmp_path, kind):
    """リカバリーの途中で終わる区間の無い日は、③ の結果・表と ④ の表・本・組が直す前と同じ"""
    if kind == 'N':
        D, _, _ = t7.round3(tmp_path, 'N')
    else:
        D, _, _ = t72.day(tmp_path, kind, 3.5)
    a, b = run_s3(D, 'new'), run_s3(D, 'old')
    assert a == b


def test_s3_coll_before_takes_the_flag(tmp_path):
    """前に衝突のある本のリカバリーの途中のあとの走り直しは、④ の 1 章で「前に衝突」の方に数える(③ の表には両方の「数えない」)"""
    D, log, tt = rday(tmp_path, 3.5, hits=lambda tt: [(tt['t_st'] + 0.5, 5.0)])
    R, md, qmd, runs, groups = run_s3(D, 'new')
    s2 = st_of(R, log)[1]
    assert s2['coll_before'] >= 1 and s2['after_abort']['kind'] == 'recovery'
    run = [x for x in runs if x['log'] == log][0]
    assert any(f.endswith('(前に衝突)') for f in run['flags']) and '数えない発進 1(%s)' % RN not in run['flags']
    assert '・数えない(この本の前の衝突 ' in md and '・数えない(%s。%s ' % (RN, REC) in md


# ================================================================== ②: 試し・途中で変えた本の ON の空転・最初の周の欠け
KIND_NAME = {'trial': '試しの本', 'midrun': '途中で変えた本'}
KIND_TAG = {'trial': '試し', 'midrun': '途中で変えた'}


@pytest.mark.parametrize('kind', ['trial', 'midrun'])
def test_c5_trial_midrun_on_spin_counts(tmp_path, kind):
    """試し・途中で変えた本の ON の発進が空転(比 ≈ 1.85): 今は checks の slip = False で「満たさない」・組に <kind>_slip・行。比べの数
    (ON 3 本・起点まで −0.165 秒)は直す前と同じ。直す前は「使ってよい」"""
    D, log = mtday(tmp_path, kind, 'spin')
    jn, on, _ = runc(D, 'new', IG, 'sp' + kind)
    jo, oo, _ = runc(D, 'old', IG, 'sp' + kind)
    gn, go = g0(jn), g0(jo)
    r = rows_of(jn, log)[0]
    assert r['slip']['ratio'] > 1.3 and r['launch_state'] == 'on' and r.get(kind)
    assert go['compare']['all_ok'] is True and '→ 使ってよい' in oo
    assert gn['by_launch'] == go['by_launch'] and gn['compare']['d_to_r1_s'] == go['compare']['d_to_r1_s'] == -0.165
    assert gn['compare']['checks']['slip'] is False and gn['compare']['all_ok'] is False and '→ 満たさない' in on
    assert gn['%s_slip' % kind] == ['%s@%.2f(空転の比 %.2f > 1.3)' % (log, r['t_ai'], r['slip']['ratio'])]
    assert any(x.startswith('  → %sの ON の発進が空転した 1 本' % KIND_NAME[kind]) and log in x and 'checks の slip' in x
               for x in on.splitlines())
    other = 'midrun' if kind == 'trial' else 'trial'
    assert '%s_slip' % other not in gn and 'abort_resume_slip' not in gn


@pytest.mark.parametrize('kind', ['trial', 'midrun'])
def test_c5_trial_midrun_extreme_spin(tmp_path, kind):
    """いちばんひどい空転(比が出ないのに車輪が増えた)も slip = False(文は「いちばんひどい空転とみなす」)"""
    D, log = mtday(tmp_path, kind, 'extreme')
    jn, _, _ = runc(D, 'new', IG, 'ex' + kind)
    gn = g0(jn)
    assert gn['compare']['checks']['slip'] is False and gn['compare']['all_ok'] is False
    assert len(gn['%s_slip' % kind]) == 1 and 'いちばんひどい空転とみなす' in gn['%s_slip' % kind][0]
    assert g0(runc(D, 'old', IG, 'ex' + kind)[0])['compare']['all_ok'] is True


@pytest.mark.parametrize('kind', ['trial', 'midrun'])
def test_c5_trial_midrun_slip_unmeasured(tmp_path, kind):
    """空転の比が出ない(車輪 +0.40・IMU +0.12 m/s): slip_all_measured = False・<kind>_slip_unmeasured・行"""
    D, log = mtday(tmp_path, kind, 'unk')
    jn, on, _ = runc(D, 'new', IG, 'un' + kind)
    gn = g0(jn)
    r = rows_of(jn, log)[0]
    assert r['slip']['ratio'] is None
    assert gn['compare']['checks']['slip_all_measured'] is False and gn['compare']['all_ok'] is False
    assert '%s_slip' % kind not in gn and gn['%s_slip_unmeasured' % kind] == [
        '%s@%.2f(空転の比が出ない(車輪 +%.2f・IMU +%.2f m/s。どちらもほとんど増えない))' % (log, r['t_ai'], r['slip']['dv_wheel'],
                                                                             r['slip']['dv_imu'])]
    assert any(x.startswith('  → %sの ON の発進で空転の比が出ない 1 本' % KIND_NAME[kind]) for x in on.splitlines())
    go = g0(runc(D, 'old', IG, 'un' + kind)[0])
    assert go['compare']['all_ok'] is True and 'slip_all_measured' not in go['compare']['checks']


@pytest.mark.parametrize('kind', ['trial', 'midrun'])
def test_c5_trial_midrun_first_lap_gap(tmp_path, kind):
    """試し・途中で変えた本の ON の発進の最初の周の IMU が抜けた(シーン 0002 の IMU.npz が無い): first_lap_gap_on に「。試し」
    「。途中で変えた」付きで入り on_all_readable = False で「満たさない」。直す前は「使ってよい」"""
    D, log = mtday(tmp_path, kind, 'gap')
    jn, on, _ = runc(D, 'new', IG, 'gap' + kind)
    jo, oo, _ = runc(D, 'old', IG, 'gap' + kind)
    gn, go = g0(jn), g0(jo)
    r = rows_of(jn, log)[0]
    assert r.get('first_lap_gap') and r['launch_state'] == 'on' and r.get('to_r1_s') is not None
    assert gn['first_lap_gap_on'] == ['%s@%.2f(%s。%s)' % (log, r['t_ai'], r['first_lap_gap'], KIND_TAG[kind])]
    assert gn['compare']['checks']['on_all_readable'] is False and gn['compare']['all_ok'] is False
    assert any(x.startswith('  最初の周の記録が欠けた発進(on)1 本') and log in x for x in on.splitlines())
    assert go['compare']['all_ok'] is True and 'first_lap_gap_on' not in go


@pytest.mark.parametrize('kind', ['trial', 'midrun'])
@pytest.mark.parametrize('shape', ['nospin', 'off_spin', 'noreach'])
def test_c5_trial_midrun_not_counted_shapes_same(tmp_path, kind, shape):
    """空転なし・OFF の本の空転・R1 に届かなかった ON(空転の比を出さない発進)は、直す前と JSON・標準出力ともバイト単位で同じ
    (条件に数えるのは ON の R1 に届いた発進の空転と、ON の最初の周の欠けだけ)"""
    D, log = mtday(tmp_path, kind, shape)
    for args in ([], IG):
        assert same_outputs(D, args, '%s%s%d' % (kind, shape, len(args))), args


@pytest.mark.parametrize('trial_us, counted', [(200.0, True), (250.0, False)])
def test_c5_trial_spin_mixed_launch_us(tmp_path, trial_us, counted):
    """launch_us が混ざった組(3-2・3-4 は 250、3-6 は 200 = 比べは最後の ON の強さ 200): 試しの ON の空転も、その強さの本だけを数える
    (走り直しの空転と同じ)"""
    D, log = mtday(tmp_path, 'trial', 'spin', us={13: 250.0, 15: 250.0, 17: 200.0, 19: trial_us})
    jn, _, _ = runc(D, 'new', IG, 'mx%d' % int(trial_us))
    g = g0(jn)
    assert g.get('mixed_launch_us') == [200.0, 250.0] and g['launch_us_used'] == 200.0
    assert ('trial_slip' in g) is counted and g['compare']['checks']['slip'] is (not counted)


def test_c5_trial_collisions_still_counted(tmp_path):
    """試しの ON の最初の周の衝突は前と同じに数える(trial_on_collisions。fix3 からの形)"""
    D, log = mtday(tmp_path, 'trial', 'nospin', hits=[(3.0, 6.0)])
    jn, _, _ = runc(D, 'new', IG, 'tc')
    jo, _, _ = runc(D, 'old', IG, 'tc')
    assert g0(jn).get('trial_on_collisions') == g0(jo).get('trial_on_collisions') == 1
    assert g0(jn)['compare'] == g0(jo)['compare'] and g0(jn)['compare']['checks']['first_lap_collisions_0'] is False


# ================================================================== 本物の記録(読むだけ。無ければ飛ばす)
REAL_REC = [('20261010-181504', 69.214, 65.883, 0.43, 2.9, None), ('20261010-181504', 85.668, 81.171, 0.1, 4.4, None),
            ('20261010-181504', 88.999, 81.171, 0.1, 0.07, 1), ('20261023-211828', 154.351, 146.191, 0.77, 7.39, None),
            ('20261024-135608', 27.346, 15.988, 0.1, 11.26, None), ('20261024-160847', 46.498, 39.97, 0.47, 6.06, None),
            ('20261024-160847', 65.85, 39.97, 0.47, 18.55, 1), ('20261024-184400', 32.043, 29.078, 0.57, 2.4, None)]
REAL_LOGS = sorted({x[0] for x in REAL_REC} | {'20261010-160306'})
need_real = pytest.mark.skipif(not all((JF / 'data' / 'labels' / (lg + '.npz')).exists()
                                       and (JF / 'data' / 'raw' / lg / 'log.json').exists() for lg in REAL_LOGS),
                               reason='本物の 10/10・10/23・10/24 の記録(ラベル)が無い')


def _real_marks(s3m, lg, with_verpm=True):
    z = dict(np.load(JF / 'data' / 'labels' / (lg + '.npz')))
    lab = json.loads((JF / 'data' / 'labels' / (lg + '.json')).read_text(encoding='utf-8'))
    t = z['t_s'].astype(np.float64)
    ai = z['ai'].astype(bool) & ~z['dry_run'].astype(bool)
    V = le.load_log(lg, JF / 'data' / 'raw')['VEHICLE']
    vt = (V['timestamps_us'].astype(np.int64) - int(lab['t0_us'])) / 1e6
    ve = V['erpm'].astype(np.float64)
    _, Df = le.cumulative_distance(vt, ve)
    r = s3m.abort_resumes(t, ai, z['assist'], z['hold'], z['ai_cmd_select'], vt, Df, ve)
    return {round(float(t[i]), 3): v for i, v in r.items()}


@need_real
def test_real_recovery_marks():
    """本物(読むだけ): リカバリーの途中で終わった 7 区間のあとの AI の区間 8 つ(続き 2)に kind recovery の印。fix7 の印(中止・中立止め)は
    前と同じ。直す前は recovery の印なし"""
    s3n, s3o = mods('new')[0], mods('old')[0]
    got = []
    for lg in REAL_LOGS:
        mn, mo = _real_marks(s3n, lg), _real_marks(s3o, lg)
        assert {k: v for k, v in mn.items() if v['kind'] != 'recovery'} == mo, lg
        assert all(v['kind'] != 'recovery' for v in mo.values())
        got += [(lg, k, v['abort_s'], v['rec_s'], v['human_s'], v.get('carried')) for k, v in sorted(mn.items())
                if v['kind'] == 'recovery']
    assert got == REAL_REC


@need_real
def test_real_s3_counts_unchanged():
    """本物(読むだけ): 印の付いた止まった状態からの発進 3 つ(20261023-211828@154.35・20261024-135608@27.35・20261024-160847@46.50)は
    前から数えない発進(前に衝突・途中で変えた / R1 に届かない)なので、③ の数える発進・最速は直す前と同じ"""
    s3n, s3o = mods('new')[0], mods('old')[0]
    for lg, t_ai in (('20261023-211828', 154.351), ('20261024-135608', 27.346), ('20261024-160847', 46.498)):
        rn = s3n.analyze_log(lg, JF / 'data' / 'raw', JF / 'data' / 'labels')
        ro = s3o.analyze_log(lg, JF / 'data' / 'raw', JF / 'data' / 'labels')
        sn = [s for s in rn['standing'] if s['t_ai'] == t_ai]
        so = [s for s in ro['standing'] if s['t_ai'] == t_ai]
        assert len(sn) == len(so) == 1 and sn[0]['after_abort']['kind'] == 'recovery' and 'after_abort' not in so[0]
        assert sn[0]['counted'] is False and s3o.counted(so[0]) is False
        assert [s3n.counted(s) for s in rn['standing']] == [s3o.counted(s) for s in ro['standing']]
        assert s3n.best_standing(rn) == s3o.best_standing(ro)
