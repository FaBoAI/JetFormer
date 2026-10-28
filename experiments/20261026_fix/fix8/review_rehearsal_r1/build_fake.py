"""反証 8-1 回目・係 rehearsal: 作り物の家(~)と 11/3 の 1 日の記録を作る。手順書 8.1・8.2・4 章 10 のコマンドを、HOME をここに替えて
そのまま流すため(本物の data/raw・labels・jetformer_reports には書かない)。

  tmp_rehearsal/<日の形>/home/jetformer/            ← 本物の jetformer/・tools/・夜の道具の台本の写し(/home/jetson/jetformer の文字を
                                                      この場所に置き換えた)と data/bench_candidates.json の写し・作り物の data/raw
  tmp_rehearsal/<日の形>/home/jetformer_reports/   ← 作り物の出力先(quick_eval の desk の写しだけ置く)

    python -I -B build_fake.py <日の形>      日の形: normal / trial_check / guard_off / reboot3 / resume
記録は gen_course_r8.py(fix7 tools r1 の作り物のコース)。説明欄は本物の web/bench.py の plan_check・with_ai_no で作る(走行開始の
サーバーと同じ文)。params は drive.DEFAULT_PARAMS に、その本の値を足す(log.json の drive.params と同じ形)。
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(HERE))
from jetformer import drive  # noqa: E402
from jetformer.web import bench as bw  # noqa: E402
import gen_course_r8 as gc  # noqa: E402

ROOT = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal'
RUNB = 'clean_driving_20261011-nocmd-s3-20261011-024626'
COPY_DIRS = ['jetformer', 'tools']
COPY_FILES = ['experiments/20261024_tune/quick_eval.py', 'experiments/20261026_fix/plan/runs_1103.py'] + \
    ['experiments/20261026_fix/diag_launch/%s' % n for n in ('collect.py', 'sim.py', 'fit_dyn.py', 'check_dyn.py', 'report_figs.py')]
DESK = Path('/home/jetson/jetformer_reports/prep_20261022/desk/desk_bench_merged.json')


def make_tree(home):
    fj = home / 'jetformer'
    for d in COPY_DIRS:
        shutil.copytree(JF / d, fj / d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'out', '*.engine', '*.onnx', '*.pth'))
    for f in COPY_FILES:
        (fj / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(JF / f, fj / f)
    n = 0
    for p in fj.rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.sh'):
            s = p.read_text(encoding='utf-8')
            if '/home/jetson/jetformer' in s:
                st = p.stat()
                p.write_text(s.replace('/home/jetson/jetformer', str(fj)), encoding='utf-8')
                n += 1
    (fj / 'data' / 'raw').mkdir(parents=True)
    (fj / 'data' / 'labels').mkdir(parents=True)
    shutil.copy2(JF / 'data' / 'bench_candidates.json', fj / 'data' / 'bench_candidates.json')
    for extra in ('modes.json',):
        if (JF / 'data' / extra).exists():
            shutil.copy2(JF / 'data' / extra, fj / 'data' / extra)
    dk = home / 'jetformer_reports' / DESK.relative_to('/home/jetson/jetformer_reports')
    dk.parent.mkdir(parents=True)
    shutil.copy2(DESK, dk)
    return fj, n


BENCH = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
BLOCKS = {b['block']: b for d in BENCH['days'] if d['date'] == '2026-11-03' for b in d['blocks']}
DATA = dict(bw.load_candidates(JF / 'data' / 'bench_candidates.json'))


def desc_for(block, ai_no, params, battery='A', vb='8.31', memo=None):
    """画面のブロックの雛形(app.js benchApply と同じ欄)→ 人が battery・V_before・check を書いた → 走行開始のサーバーの plan_check・with_ai_no"""
    if block is None:
        d = 'AI走行 %s' % RUNB
    else:
        b = BLOCKS[block]
        d = ('bench=real_bench_20261023 day=2026-11-03 block=%d cand=B model=%s battery=#%s V_before=%s V_after=__ check=合格 '
             'tune=%s throttle_gain=0.9 official=1 memo=' % (block, RUNB, battery, vb, b['tune']))
        d, _ = bw.plan_check(d, RUNB, 5, params, data=DATA)
    d = bw.with_ai_no(d, ai_no)
    if memo:
        d = bw.after_description(d, memo=memo)
    return d


def prm(**kw):
    p = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, lookahead=3, start_kick=1, throttle_gain=0.92)
    p.update(kw)
    return p


class Day(object):
    def __init__(self, raw):
        self.raw = raw
        self.t = 8 * 3600 + 30 * 60          # 08:30:00
        self.ai_no = 1                        # No.1 は朝の発進の前進の確かめ(収集を外した = 記録なし)
        self.books = []

    def name(self):
        h, m, s = self.t // 3600, (self.t // 60) % 60, self.t % 60
        return '20261103-%02d%02d%02d' % (h, m, s)

    def add(self, block, p, laps=3, standing=True, v=2.25, launch=None, memo=None, gap_min=4, ai_extra=None, hits=(), events=(),
            count_ai=True, **kw):
        """1 本: 置いた場所(HS)から t=1.0 秒に AI 側。standing なら 3 周目の線のあと R1 を抜けるまで(62 m)AI 側"""
        log = self.name()
        self.t += gap_min * 60
        if count_ai:
            self.ai_no += 1
        on = bool(p.get('launch_boost'))
        d_move, ramp = (0.12, 0.35) if on else (0.22, 0.6)
        dist_end = 18.0 * laps + (8.0 if standing else 6.0)
        base = gc.launch_keys(1.0, d_move, ramp, v=v)
        t_end = gc.time_at_distance(base + [(400.0, v)], dist_end, 400.0)
        dur = kw.pop('dur', None) or (t_end + 3.0)
        vk = base + [(t_end, v), (t_end + 1.5, 0.0), (dur, 0.0)]
        ai = kw.pop('ai', None) or [(1.0, t_end)]
        if ai_extra:
            ai = ai + ai_extra
        lch = launch if launch is not None else ([(1.0, 0.45)] if on else [])
        desc = desc_for(block, self.ai_no, p, memo=memo)
        gc.write_log(self.raw, log, dur, kw.pop('vk', None) or vk, ai, p, desc, launch=lch, hits=hits, events=events, run=RUNB, **kw)
        self.books.append({'log': log, 'block': block, 'run_id': BLOCKS[block]['run_id'] if block else None, 'ai_no': self.ai_no,
                           'launch_boost': p.get('launch_boost'), 'rec': (p.get('rec_back_guard'), p.get('rec_hold_keep')),
                           'memo': memo})
        return log


def build(kind):
    base = ROOT / kind
    if base.exists():
        shutil.rmtree(base)
    home = base / 'home'
    fj, n = make_tree(home)
    D = Day(fj / 'data' / 'raw')
    G = dict(rec_back_guard=1, rec_hold_keep=1)
    off_from = None
    if kind == 'trial_check':
        # 朝の発進の前進の確かめを収集したまま走った(memo に「試し」。説明欄に block が無い)。見張り・残すはまだ OFF
        D.ai_no = 0
        D.add(None, prm(throttle_gain=1.0, launch_boost=1), laps=0, standing=False, memo='試し', ai=[(1.0, 2.0)])
    gopt = G
    # 回 1
    for blk, kw in [(1, dict(laps=5, standing=False)), (2, {}), (3, {}), (4, dict(laps=5, standing=False)), (5, dict(laps=5, standing=False))]:
        if kind == 'guard_off' and blk == 3:
            gopt, off_from = {}, D.ai_no + 1          # 1-2 の最初のリカバリーがおかしい → 止めて OFF。紙に「本 1-3 から」
        ceil = 1 if blk == 4 else 0
        D.add(blk, prm(speed_ceiling=ceil, **gopt), **kw)
    # 回 2
    for blk, kw in [(6, dict(speed_ceiling=1, throttle_gain=0.95)), (7, {}), (8, dict(speed_ceiling=1, speed_auto=1, throttle_gain=0.95)),
                    (9, dict(speed_ceiling=1, speed_auto=1, throttle_gain=0.95)), (10, dict(speed_ceiling=1, ceil_left_mps=4.2, throttle_gain=0.95)),
                    (11, dict(speed_ceiling=1, throttle_gain=0.95))]:
        D.add(blk, prm(**dict(gopt, **kw)), laps=5, standing=False)
    # 回 3(回 2 のよい形 = 天井 4.1・0.95)。OFF 12・14・16・18、ON 13・15・17・19
    us = 250.0
    for blk in range(12, 20):
        on = blk % 2 == 1
        g = gopt
        if kind == 'reboot3' and blk >= 16:
            g = {}                                    # 3-4 のあとに起こし直して、見張り・残すを入れ直し忘れた
        if kind == 'guard_off3' and blk >= 14:
            gopt = g = {}                             # 3-2 の最初のリカバリーがおかしい → 止めて OFF。紙に「本 3-3 から」
            off_from = off_from or D.ai_no + 1
        if kind == 'us200' and blk >= 16:
            us = 200.0                                # 3-4 の car_check で空転の比 > 1.3 → 残りの ON を 200(紙に)
        if kind == 'resume' and blk == 14:
            # 3-3(OFF): 1 周目の GM の途中で人側(走ったまま)→ 止めて、その場で AI 側(紙に「走り直し: 3-3・時刻」)
            v = 2.25
            base_k = gc.launch_keys(1.0, 0.22, 0.6, v=v)
            t_a = gc.time_at_distance(base_k + [(400.0, v)], 12.0, 400.0)
            t_s, t_b = t_a + 1.2, t_a + 4.0
            t_end = t_b + 0.22 + 0.6 + (18.0 + 5.0 + 3.0) / v
            vk = base_k + [(t_a, v), (t_s, 0.0), (t_b + 0.22, 0.0), (t_b + 0.82, v), (t_end, v), (t_end + 1.5, 0.0), (t_end + 3.0, 0.0)]
            D.add(blk, prm(speed_ceiling=1, throttle_gain=0.95, launch_boost=0, **g), laps=1, standing=False, vk=vk,
                  ai=[(1.0, t_a), (t_b, t_end)], dur=t_end + 3.0)
            continue
        D.add(blk, prm(speed_ceiling=1, throttle_gain=0.95, launch_boost=int(on), launch_us=us, **g), laps=1, standing=False)
    # 回 4(本番の形 = 天井 + 発進の前進 ON)
    for blk in (20, 21, 22):
        D.add(blk, prm(speed_ceiling=1, throttle_gain=0.95, launch_boost=1, launch_us=us, **gopt), laps=3, standing=True)
    (base / 'manifest.json').write_text(json.dumps({'kind': kind, 'replaced_files': n, 'off_from': off_from, 'books': D.books},
                                                    ensure_ascii=False, indent=1), encoding='utf-8')
    print(kind, len(D.books), 'books', 'replaced', n, 'files', '->', home)


if __name__ == '__main__':
    for k in sys.argv[1:] or ['normal']:
        build(k)
