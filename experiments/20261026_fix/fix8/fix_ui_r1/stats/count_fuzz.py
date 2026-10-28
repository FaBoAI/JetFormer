"""tests/test_ui_fix8_r1.py のでたらめな流れ 2 本(② の日 120 本・戻ってすぐ ▶ の朝 16 本)の数を数える(試験と同じ種・同じ作り方)。
出力 out_count_fuzz.txt"""
import random
import sys
import tempfile
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tests'))
import test_ui_fix8_r1 as T  # noqa: E402
from jetformer.web import bench  # noqa: E402

HERE = Path(__file__).resolve().parent
lines = []
tmp = Path(tempfile.mkdtemp(dir=HERE))
# ② の日
rng = random.Random(81)
shapes = [T.G, T.G, T.G, T.OFF, {'rec_back_guard': 1}, dict(T.G, **T.NK)]
n_same = n_moved = n_runs = n_mark_new = n_mark_old = 0
for i in range(120):
    recs = []
    for r in rng.sample(sorted(T.ROUNDS), rng.randint(1, 3)):
        plan = rng.choice([T.G, T.G, T.OFF, dict(T.G, **T.NK)])
        for k in range(1, rng.randint(2, T.ROUNDS[r]) + 1):
            if rng.random() < 0.15:
                recs.append((None, rng.choice([T.OFF, T.G])))
            recs.append(('%d-%d' % (r, k), plan if rng.random() < 0.7 else rng.choice(shapes)))
    (tn, rn), (to, ro) = T.both_runs(tmp, recs, 'f%d' % i)
    n_runs += len(rn)
    n_mark_new += sum(1 for v in T.opt_marks(rn).values() if v)
    n_mark_old += sum(1 for v in T.opt_marks(ro).values() if v)
    if tn == to and rn == ro:
        n_same += 1
    else:
        n_moved += 1
lines.append('② の日 120 本(種 81): 本 %d・直す前とバイト単位で同じ日 %d・違う日 %d・試験の選択肢の印 今 %d 本 / 直す前 %d 本' % (
    n_runs, n_same, n_moved, n_mark_new, n_mark_old))
# 戻ってすぐ ▶ の朝
c0 = bench.load_candidates()
di = next(i for i, d in enumerate(c0.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks'))
c = dict(c0, today='2026-11-03')
rng = random.Random(81)
n_ok = n_bad_new = n_bad_old = n_start_old = n_refused = 0
for i in range(16):
    two = i % 2 == 1
    steps = T._fuzz_two(rng, c, di, two)
    pages = ['A', 'B'] if two else ['A']
    new = T.sim2(tmp, c, steps, pages, name='fn.json', server=T.server_params(throttle_gain=0.9))
    old = T.sim2(tmp, c, steps, pages, app=T.BEFORE / 'app.js', name='fo.json', server=T.server_params(throttle_gain=0.9))
    for s in new['steps']:
        if s['op'] == 'start' and s.get('status') == 200:
            n_ok += 1
            n_bad_new += not T.confirm_matches_server(s)
    n_refused += sum(t == T.LOADING for v in new['toasts'].values() for t in v)
    for s in old['steps']:
        if s['op'] == 'start' and s.get('status') == 200:
            n_start_old += 1
            n_bad_old += not T.confirm_matches_server(s)
lines.append('戻ってすぐ ▶ の朝 16 本(種 81): 今の版 走り出した ▶ %d・確認の文と走った値が違う ▶ %d・「読み込んでいます」のトースト %d / '
             '直す前 走り出した ▶ %d・違う ▶ %d' % (n_ok, n_bad_new, n_refused, n_start_old, n_bad_old))
import shutil  # noqa: E402
shutil.rmtree(tmp)
(HERE / 'out_count_fuzz.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
