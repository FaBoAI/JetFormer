"""打ち消しの票 A(反証 8-1 回目・係 tools の指摘 3 件)の共通の部品。読むだけの道具を作り物の記録で流す。

作り物の記録の書き方は tests/test_fix7_tools.write_log(プロジェクトの試験の作り物のコース。R1 の起点 = 5 m・23 m・41 m …、
1 周 18 m)を使う。指摘の台本(review_tools_r1/*.py)は読まない・使わない。
data/raw・labels は読まない(作り物の日だけ。tmp の場所に書いて最後に消す)。
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix7_tools as t7  # noqa: E402  作り物の記録の書き方(write_log・launch_keys・time_at_distance)

HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_refute_tools_A'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
S3 = JF / 'tools' / 'standing_3lap.py'
QP = JF / 'tools' / 'quick_eval_by_params.py'
T0_US = t7.T0_US

BASE = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
        'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
        'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}


def load_mod(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class MyDay(object):
    """作り物の 11/3。ブロック 12〜19 = 回 3 の 3-1〜3-8(bench の run_id)"""

    def __init__(self, name):
        self.tmp = TMP / name
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.raw = self.tmp / 'raw'
        self.raw.mkdir(parents=True)
        self.bench = self.tmp / 'bench.json'
        self.bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                          'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                                     for b in range(12, 20)]}]}), encoding='utf-8')
        self.n = 0
        self.logs = []

    def add(self, block, prm, d_move=None, ramp=None, memo='x', dur=30.0, vk=None, ai=None, launch=None, **kw):
        log = '20261103-14%02d00' % (self.n * 5)
        self.n += 1
        vk = vk or t7.launch_keys(1.0, d_move, ramp, until=dur)
        ai = ai or [(1.0, dur - 2.0)]
        if launch is None:
            launch = [(1.0, 0.5)] if prm.get('launch_boost') else []
        desc = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=%s' % (self.n, block, memo)
        t7.write_log(self.raw, log, dur, vk, ai, prm, desc, launch=launch, **kw)
        self.logs.append(log)
        return log


def scale_imu_fwd(raw, log, a, b, k):
    """IMU の前の加速(accel[:, 1])を a〜b 秒だけ k 倍(車輪の速さはそのまま = 空転の形)"""
    for p in sorted((Path(raw) / log).glob('%s-*/IMU.npz' % log)):
        z = dict(np.load(p))
        tt = (z['timestamps_us'].astype(np.int64) - T0_US) / 1e6
        m = (tt >= a) & (tt < b)
        acc = z['accel'].copy()
        acc[m, 1] *= k
        z['accel'] = acc
        np.savez(p, **z)


def set_assist(raw, log, a, b, code):
    """AI_CMD の assist を a〜b 秒(select が AI 側・中立止めでないフレーム)だけ code に"""
    for p in sorted((Path(raw) / log).glob('%s-*/AI_CMD.npz' % log)):
        z = dict(np.load(p))
        tt = (z['timestamps_us'].astype(np.int64) - T0_US) / 1e6
        m = (tt >= a - 1e-9) & (tt < b - 1e-9) & z['select'].astype(bool) & ~z['hold'].astype(bool)
        if m.any():
            asg = z['assist'].copy()
            asg[m] = code
            z['assist'] = asg
            np.savez(p, **z)


def run_collect(D, args=(), tag='x'):
    """⑤ collect.main(--day 20261103)を作り物の日で。(JSON, 標準出力)"""
    m = load_mod('refA_collect_%s' % tag, COLLECT)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('collect_%s.json' % tag)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue()


def run_s3(D):
    """③ standing_3lap.run と ④ quick_eval_by_params.to_markdown を作り物の日で"""
    s3m = load_mod('standing_3lap', S3)
    sys.modules['standing_3lap'] = s3m
    qpm = load_mod('refA_qp', QP)
    logs, _ = s3m.day_logs('20261103', D.raw)
    R = s3m.run(logs, D.raw, D.tmp / 'nolab', None)
    md = s3m.to_markdown(R)
    qmd, runs, groups = qpm.to_markdown(R, '20261103', None, ('gain',))
    return R, md, qmd, runs, groups


def summary_of(j):
    return j.get('summary') or []


def cmp_line(out):
    return [x for x in out.splitlines() if 'ON − OFF' in x]
