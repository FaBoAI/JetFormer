"""反証 8-2 回目・係 rehearsal: 作り物の家(~)と 11/3 の 1 日の記録を作る(8-1 の build_fake.py の写しに、8-1 のあとに入った手順の
日の形を足した)。手順書 8.1・8.2・4 章 10 のコマンドを、HOME をここに替えてそのまま流すため(本物の data/raw・labels・
jetformer_reports には書かない)。

  tmp_rehearsal_r2/<日の形>/home/jetformer/            ← 本物の jetformer/・tools/・夜の道具の台本の写し(/home/jetson/jetformer の文字を
                                                         この場所に置き換えた)と data/bench_candidates.json・modes.json の写し・作り物の data/raw
  tmp_rehearsal_r2/<日の形>/home/jetformer_reports/   ← 作り物の出力先(quick_eval の desk の写しだけ置く)

    python -I -B build_fake.py <日の形> ...
日の形:
  normal       両方 ON の日の 22 本(8-1 と同じ)
  scene_memo   normal + 朝の発進の前進の確かめを収集したまま走った本(本「—」)。あとで scene_path.py が「シーン」のタブの道で「試し」を書く
  rear1m       回 3 の 3-2(ON): 前の物の手前で止まり(10/23 の形)、後ろ約 1 m 以内なので、リカバリー(0.8 秒)の前に人側 → HS に置き直して
               同じ本の中で AI 側(走り直し。印は出ない形 = 紙に「走り直し」→ main session の --skip 記録名@T)
  rear1m_hit   rear1m の走り直しが最初の周の LH で 4.5G に当たる
  rear1m_late  rear1m で人側が少し遅れ、リカバリーの最初の 'brake'(assist 2)が 0.15 秒出たあとに人側(印が出る形)
  rear1m_late_hit  rear1m_late の走り直しが最初の周の LH で 4.5G に当たる
  abort_redo   回 4 の 4-1: 2 周目の SW に当たって止まり、リカバリー → 中止(中立で停止中)→ 人側 → ■ → HS に置いて同じ 20 をもう 1 本
  reboot3first 回 2 と回 3 の間に起き直し、3-1 は見張り・残す OFF のまま走り、3-2 の前に気づいて入れ直した
  rear1m4      回 4 の 4-2: 3 周目の終わり(53.5 m = HS の 0.5 m 手前)で前の物の手前に止まる → リカバリーの前に人側 → HS(54 m)に置いて
               同じ本の中で 3 周の走り直し(印の出ない形)
  form_trial   回 3 の 3-6(ON)を左 4.2 のまま走り(2-5 のあと 4.1 に戻し忘れた)、1 周目の R1(6.0 m)で 4.5G。あとで「シーン」のタブで「試し」
記録は gen_course_r8.py(8-1 の写しに recov・scene.json を足した)。説明欄は本物の web/bench.py の plan_check・with_ai_no で作る。
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

ROOT = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal_r2'
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


def dist_time(vk, s, dur=400.0):
    return gc.time_at_distance(vk + [(dur, vk[-1][1])], s, dur)


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
            count_ai=True, note=None, **kw):
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
                           'memo': memo, 'note': note})
        return log


def rear_stop_book(D, block, p, late=False, hit=False, s_stop=17.5, laps2=1):
    """10/23 の形: 1 周目の SW のあと(距離 17.5 m = HS の 0.5 m 手前)で前の物の手前に止まる。止まって 0.5 秒(リカバリーの 0.8 秒の前)に
    人側(late なら 0.8 秒でリカバリーの 'brake' が 0.15 秒出たあと)→ 人が 0.5 m 押して HS(18 m)に置き直す → 4 秒待って AI 側(走り直し)
    → 1 周 + R1 を抜けて人側。hit なら走り直しの最初の周の LH(21.5 m)で 4.5G"""
    v = 2.25
    on = bool(p.get('launch_boost'))
    d_move, ramp = (0.12, 0.35) if on else (0.22, 0.6)
    k = gc.launch_keys(1.0, d_move, ramp, v=v)
    t_dec = dist_time(k, s_stop - 0.5 * v * 0.6)          # 0.6 秒で 2.25 → 0(その間 0.675 m)
    t_s = t_dec + 0.6
    t_rec = t_s + 0.8
    t_h = t_rec + 0.15 if late else t_s + 0.5            # 人側
    t_p0, t_p1 = t_h + 1.5, t_h + 2.5                    # 人が 0.5 m 押す(0.5 m/s)
    t_b = t_p1 + 4.0                                     # 走り直し(AI 側)
    vk = k + [(t_dec, v), (t_s, 0.0), (t_p0, 0.0), (t_p0 + 0.05, 0.5), (t_p1 - 0.05, 0.5), (t_p1, 0.0),
              (t_b + d_move, 0.0), (t_b + d_move + ramp, v)]
    s_b = s_stop + 0.5
    # 走り直しは HS(s_b)から laps2 周 + R1 の起点を抜けるまで(s_b + 18 × laps2 + 5 + 3 m)
    t_end = dist_time(vk, s_b + 18.0 * laps2 + 8.0)
    t_lh = dist_time(vk, s_b + 3.5)
    dur = t_end + 3.0
    vk = vk + [(t_end, v), (t_end + 1.5, 0.0), (dur, 0.0)]
    lch = [(1.0, 0.45)] + ([(t_b, 0.45)] if on else [])
    recov = [(t_rec, t_rec + 0.15)] if late else []
    hits = [(t_lh, 4.5)] if hit else []
    log = D.add(block, p, laps=laps2, standing=laps2 >= 3, vk=vk, ai=[(1.0, t_h), (t_b, t_end)], dur=dur, launch=lch, recov=recov, hits=hits,
                note='rear1m: 止まり %.2f 秒・人側 %.2f 秒・走り直し %.2f 秒(HS 18 m から)%s' % (
                    t_s, t_h, t_b, '・LH 4.5G %.2f 秒' % t_lh if hit else ''))
    D.books[-1].update(t_s=round(t_s, 3), t_h=round(t_h, 3), t_b=round(t_b, 3), s_b=s_b, t_rec=round(t_rec, 3) if late else None,
                       t_lh=round(t_lh, 3) if hit else None)
    return log


def abort_book(D, block, p):
    """回 4 の 4-1: 2 周目の SW(33 m)で 5G に当たって止まり、リカバリー(assist 2)2 秒 → 中止(assist 3。中立で停止中)3 秒 → 人側 → ■"""
    v = 2.25
    on = bool(p.get('launch_boost'))
    d_move, ramp = (0.12, 0.35) if on else (0.22, 0.6)
    k = gc.launch_keys(1.0, d_move, ramp, v=v)
    t_hit = dist_time(k, 33.0)
    t_s = t_hit + 0.3
    vk = k + [(t_hit, v), (t_s, 0.0)]
    t_r0 = t_s + 0.5
    t_g0 = t_r0 + 2.0
    t_h = t_g0 + 3.0
    dur = t_h + 4.0
    vk = vk + [(dur, 0.0)]
    log = D.add(block, p, laps=3, standing=True, vk=vk, ai=[(1.0, t_h)], dur=dur, launch=[(1.0, 0.45)] if on else [],
                hits=[(t_hit, 5.0)], recov=[(t_r0, t_g0)], giveup=[(t_g0, t_h)],
                note='abort: SW 5G %.2f 秒・リカバリー %.2f〜%.2f・中止 %.2f〜%.2f・人側 → ■' % (t_hit, t_r0, t_g0, t_g0, t_h))
    return log


def build(kind):
    base = ROOT / kind
    if base.exists():
        shutil.rmtree(base)
    home = base / 'home'
    fj, n = make_tree(home)
    D = Day(fj / 'data' / 'raw')
    G = dict(rec_back_guard=1, rec_hold_keep=1)
    if kind == 'scene_memo':
        # 朝の発進の前進の確かめを収集したまま走った(説明欄に block が無い。見張り・残すはまだ OFF)
        D.ai_no = 0
        D.add(None, prm(throttle_gain=1.0, launch_boost=1), laps=0, standing=False, ai=[(1.0, 2.0)], note='朝の確かめ(本「—」)')
    gopt = G
    # 回 1
    for blk, kw in [(1, dict(laps=5, standing=False)), (2, {}), (3, {}), (4, dict(laps=5, standing=False)), (5, dict(laps=5, standing=False))]:
        ceil = 1 if blk == 4 else 0
        D.add(blk, prm(speed_ceiling=ceil, **gopt), **kw)
    # 回 2
    for blk, kw in [(6, dict(speed_ceiling=1, throttle_gain=0.95)), (7, {}), (8, dict(speed_ceiling=1, speed_auto=1, throttle_gain=0.95)),
                    (9, dict(speed_ceiling=1, speed_auto=1, throttle_gain=0.95)), (10, dict(speed_ceiling=1, ceil_left_mps=4.2, throttle_gain=0.95)),
                    (11, dict(speed_ceiling=1, throttle_gain=0.95))]:
        D.add(blk, prm(**dict(gopt, **kw)), laps=5, standing=False)
    # 回 3(回 2 のよい形 = 天井 4.1・0.95)。OFF 12・14・16・18、ON 13・15・17・19
    for blk in range(12, 20):
        on = blk % 2 == 1
        g = gopt
        if kind == 'reboot3first' and blk == 12:
            g = {}                                    # 回 2 と回 3 の間に起き直し、3-1 は入れ直し忘れ(3-2 の前に気づいて入れ直した)
        p = prm(speed_ceiling=1, throttle_gain=0.95, launch_boost=int(on), **g)
        if kind == 'form_trial' and blk == 17:
            p = prm(speed_ceiling=1, throttle_gain=0.95, launch_boost=1, ceil_left_mps=4.2, **g)
            k_ = gc.launch_keys(1.0, 0.12, 0.35, v=2.25)
            D.add(blk, p, laps=1, standing=False, hits=[(dist_time(k_, 6.0), 4.5)], note='form_trial: 左 4.2・1 周目の R1(6.0 m)4.5G')
            continue
        if kind.startswith('rear1m') and blk == 13:
            rear_stop_book(D, blk, p, late='late' in kind, hit=kind.endswith('_hit'))
            continue
        D.add(blk, p, laps=1, standing=False)
    # 回 4(本番の形 = 天井 + 発進の前進 ON)
    for blk in (20, 21, 22):
        p = prm(speed_ceiling=1, throttle_gain=0.95, launch_boost=1, **gopt)
        if kind == 'rear1m4' and blk == 21:
            rear_stop_book(D, blk, p, s_stop=53.5, laps2=3)
            continue
        if kind == 'abort_redo' and blk == 20:
            abort_book(D, blk, p)                     # 4-1 を中止で止めた → ■ → HS に置いて同じ 20 をもう 1 本
        D.add(blk, p, laps=3, standing=True)
    (base / 'manifest.json').write_text(json.dumps({'kind': kind, 'replaced_files': n, 'books': D.books},
                                                    ensure_ascii=False, indent=1), encoding='utf-8')
    print(kind, len(D.books), 'books', 'replaced', n, 'files', '->', home)


if __name__ == '__main__':
    for k in sys.argv[1:] or ['normal']:
        build(k)
