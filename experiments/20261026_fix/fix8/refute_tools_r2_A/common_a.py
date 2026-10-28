"""反証 8-2 回目・係 tools の指摘 2 件の打ち消しの票 A(2026-10-28)。読むだけ(コード・文書は変えない)。

自分で作った作り物の回 3(11/3 の形・block 12〜19 = 3-1〜3-8)を、⑤ collect.py の今(d2b210cc…)と直す前
(tests/fixtures/fix8_tools_r1_before/collect.py.orig = 26dfb087…)で流す。記録の書き方(npz の形)だけ tests/test_fix7_tools.write_log を
使う(作り物の速さ・止まり・本の数字は、この票で決めたもの。指摘の台本 review_tools_r2/ は使わない)。
作り物は ../tmp_refute_tools_r2_A/ に作り、終わりに消す。
"""
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
import test_fix7_tools as t7  # noqa: E402  記録の書き方(write_log)だけ

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
TMP = HERE.parent / 'tmp_refute_tools_r2_A'
NOW_C = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
OLD_C = JF / 'tests' / 'fixtures' / 'fix8_tools_r1_before' / 'collect.py.orig'
SHA = {NOW_C: 'd2b210cc1039e07cc56608d56d049f844149f4f28386905263026e63675035f2',
       OLD_C: '26dfb087bf92d5d39ec2b63220428f43d1a4bc3efc6fb69f2916cafe1de2a5bf'}
VER = {'now': NOW_C, 'old': OLD_C}
IG = ['--ignore-gain']

# 11/3 の回 3 の形(両方 ON の日 = 全部の回で見張り・残す ON)。発進の前進は ON の本だけ 1
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0,
       'start_kick': 1, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
ON = dict(PRM, launch_boost=1)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'
T_AI = 1.0                      # AI 側にした時刻
VMAX = 2.0
ACC_OFF, ACC_ON = 2.0 / 0.6, 2.0 / 0.3      # 発進の加速(OFF 0.6 秒・ON 0.3 秒で 2 m/s)
DEC = 0.9                       # drive が中立を書いた間の惰性 [m/s²](本物の人側・スティック中立の中央値 −0.87 に近い値)
FPS = 30.0


def check_sha():
    for p, h in SHA.items():
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == h, (p, got)


def load_mod(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_MODS = {}


def mod(ver):
    if ver not in _MODS:
        _MODS[ver] = load_mod('collect_a_%s' % ver, VER[ver])
    return _MODS[ver]


def sim(dur, t_move, acc, neutral=(), vmax=VMAX, dt=0.001):
    """自分の速さの作り: t_move に動き出し acc で vmax へ。neutral [(a, b)] の間は DEC で惰性。10 ms ごとの [(秒, m/s)]"""
    n = int(round(dur / dt)) + 1
    ts = np.arange(n) * dt
    v = np.zeros(n)
    neu = np.zeros(n, bool)
    for a, b in neutral:
        neu |= (ts >= a) & (ts < b)
    for i in range(1, n):
        if ts[i] < t_move:
            continue
        v[i] = max(0.0, v[i - 1] - DEC * dt) if neu[i] else min(vmax, v[i - 1] + acc * dt)
    k = np.arange(0, n, 10)
    return [(round(float(ts[j]), 3), float(v[j])) for j in k]


def time_to(vk, s_target):
    tk = np.array([x[0] for x in vk])
    vv = np.array([x[1] for x in vk])
    s = np.concatenate([[0.0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tk))])
    return float(np.interp(s_target, s, tk))


def gap_pattern(t_first, gap_frames, count):
    """t_first(30 fps のフレームの時刻にそろえる)のフレームのあと、gap_frames フレーム分の間(= gap_frames/30 秒。抜くのは
    gap_frames − 1 枚)を count 回続ける。(write_log の stall の間, drive の中立の間, 間の秒の並び)"""
    k0 = int(round(t_first * FPS))
    stall, neu, gaps = [], [], []
    k = k0
    for _ in range(count):
        a, b = k / FPS, (k + gap_frames) / FPS
        if gap_frames > 1:
            stall.append(((k + 1) / FPS - 0.001, (k + gap_frames - 1) / FPS + 0.001))
        if b - a > 0.1:
            neu.append((a + 0.1, b))
        gaps.append(round(b - a, 3))
        k += gap_frames
    return stall, neu, gaps


class Day(object):
    """作り物の 11/3 の回 3。add(block, prm, t_move, **kw)。log は 20261103-13MM00"""

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
        self.logs = {}

    def add(self, block, prm, dm, dur=30.0, neutral=(), stall=(), ai=None, vk=None, **kw):
        log = '20261103-13%02d00' % (self.n * 5)
        self.n += 1
        on = bool(prm.get('launch_boost'))
        if vk is None:
            vk = sim(dur, T_AI + dm, ACC_ON if on else ACC_OFF, neutral)
        ai = ai or [(T_AI, dur - 2.0)]
        launch = kw.pop('launch', [(T_AI, 0.5)] if on else [])
        t7.write_log(self.raw, log, dur, vk, ai, prm, DESC % (self.n, block), launch=launch, stall=stall, **kw)
        self.logs[block] = log
        return log


def run(D, ver, args=(), tag=''):
    """⑤ main を作り物の日で流す。(JSON, 標準出力)"""
    m = mod(ver)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('c_%s_%s_%d.json' % (ver, tag, len(args)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main(['--day', '20261103', '--out', str(out)] + list(args))
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue()


def cmp_groups(j):
    return [g for g in j['summary'] if g.get('compare')]


def verdict(g):
    c = g.get('compare') or {}
    if not c:
        return '比べなし'
    return '使ってよい' if c.get('all_ok') else '満たさない'


def brief(j, o, label):
    """組の答えの短い文: OFF・ON の本数・中央値・ON − OFF・答え・False の条件・外した本・注意の行"""
    gs = cmp_groups(j)
    if not gs:
        return '%s: 比べの組なし' % label
    g = gs[0]
    bl = g['by_launch']
    off, on = bl.get('off') or {}, bl.get('on') or {}
    offs = sorted(r['to_r1_s'] for r in j['launches'] if r.get('launch_state') == 'off' and r.get('to_r1_s') is not None
                  and not r.get('launch_stall'))
    chk = [k for k, v in g['compare']['checks'].items() if v is False]
    ex = []
    for key in ('launch_stall_excluded',):
        ex += g.get(key) or []
    nr = (g.get('not_reached') or {}).get('off')
    notes = [x.strip()[:46] for x in o.splitlines()
             if x.startswith('  注意: 発進から R1 の起点までにフレームの間が') or x.startswith('  発進から R1 の起点までにカメラが止まった発進')
             or x.startswith('  → OFF の発進が全部カメラの止まり')]
    return ('%s: OFF n %s 中央値 %s(比べに入った OFF の起点まで %s)・ON n %s 中央値 %s → ON − OFF %s %s・False %s・外した %d 本%s・行 %s' % (
        label, off.get('n'), off.get('to_r1_median_s'), offs, on.get('n'), on.get('to_r1_median_s'), g['compare']['d_to_r1_s'],
        verdict(g), chk or 'なし', len(ex), '・届かなかった OFF %d' % nr['n'] if nr else '', notes or 'なし'))


def onoff_line(o):
    return [x.strip() for x in o.splitlines() if 'ON − OFF' in x][:2]


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def cleanup():
    if TMP.exists():
        shutil.rmtree(TMP)
