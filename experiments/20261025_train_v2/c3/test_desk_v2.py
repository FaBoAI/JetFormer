#!/usr/bin/env python3
"""C3(desk_v2.py)の試験: 作り物の入力で式・集合のマスク・欄の名前を確かめる(numpy だけ。torch・GPU を読まない)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        /home/jetson/jetformer/experiments/20261025_train_v2/c3/test_desk_v2.py

照らす相手(写しではなく本物のコード・文):
  drive.apply_gains(drive.py から AST で関数だけ取り出す。drive は重いので読まない)・train_data.top_ramp(同じく AST)・
  tools/policy_zoo.perturb(import)・eval_v2.py の C3_FIELDS と門の関数・protocol_v2.json の fields_added・sign_rules・sets の文
"""
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_v2 as C  # noqa: E402

REPO = Path('/home/jetson/jetformer')
PROTO = json.load(open('/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2.json', encoding='utf-8'))
results = []


def check(name, ok, info=None):
    results.append((name, bool(ok)))
    print('%s  %s%s' % ('OK ' if ok else 'NG ', name, '' if info is None else '  ' + str(info)[:300]), flush=True)


def func_from(path, fname, extra_ns=None, consts=()):
    """ファイルから関数 fname(と定数 consts)だけを AST で取り出して実行した名前空間の関数"""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == fname)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in n.targets))]
    ns = {'np': np}
    ns.update(extra_ns or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    return ns[fname], ns


rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. 車に出る = drive.apply_gains
apply_gains, _ = func_from(REPO / 'jetformer' / 'drive.py', 'apply_gains')
ref = (1433.0, 1500.0)
raw = np.stack([rng.uniform(900, 2100, 400), rng.uniform(900, 2100, 400)], 1)
for g in (0.90, 1.00, 1.10):
    want = np.array([apply_gains(r, ref, {'steer_trim_us': 0.0, 'steer_gain': 1.0, 'throttle_gain': g}, 1000.0, 2000.0)
                     for r in raw])
    got = C.car_out(raw, ref, g)
    check('car_out = drive.apply_gains(ゲイン %.2f・ブレーキ側はそのまま・1000〜2000 に切る)' % g, np.allclose(got, want, atol=1e-9))
gs = rng.choice([0.9, 1.0, 1.1], 400)
ss = rng.choice([1.0, 1.2], 400)
tr = rng.choice([0.0, 5.0], 400)
want = np.array([apply_gains(r, ref, {'steer_trim_us': t, 'steer_gain': s, 'throttle_gain': g}, 1000.0, 2000.0)
                 for r, g, s, t in zip(raw, gs, ss, tr)])
check('car_out(フレームごとのゲイン・trim)= apply_gains', np.allclose(C.car_out(raw, ref, gs, ss, tr), want, atol=1e-9))

# ---------------------------------------------------------------- 2. ずれの試し
top_ramp_td, ns_td = func_from(REPO / 'jetformer' / 'train_data.py', 'top_ramp', consts=('TOP_FRACTION',))
check('TOP_FRACTION = train_data.TOP_FRACTION', ns_td['TOP_FRACTION'] == C.TOP_FRACTION)
for h in (224, 225, 6):
    check('top_ramp(%d) = train_data.top_ramp' % h, np.array_equal(C.top_ramp(h), top_ramp_td(h)))
w = C.top_ramp(224)[:, 0, 0]
check('top_ramp: 一番上 1・上 1/3 の境(行 75〜)で 0・単調に減る', w[0] == 1.0 and (w[75:] == 0).all() and (np.diff(w) <= 0).all()
      and 0 < w[74] < 0.02, w[70:76])
import policy_zoo as pz  # noqa: E402
img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
tof = rng.integers(0, 1 << 16, (42, 54), dtype=np.uint16)
for k in ('img_up3', 'img_down3', 'bright_0.8', 'bright_1.2', 'tof_down1'):
    a, b = C.perturb(k, img, tof), pz.perturb(k, img, tof)
    check('perturb(%s) = policy_zoo.perturb' % k, np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
flat = np.full((224, 224, 3), 100, np.uint8)
for f in (0.85, 1.15):
    o, t2 = C.perturb('top_%.2f' % f, flat, tof)
    col = o[:, 0, 0].astype(int)
    exp = np.clip(np.rint(100.0 * (1.0 + (f - 1.0) * w)), 0, 255).astype(int)
    check('perturb(top_%.2f): 一番上 ×%.2f・境から下はそのまま・ToF はそのまま' % (f, f),
          col[0] == int(round(100 * f)) and (col[75:] == 100).all() and np.array_equal(col, exp) and np.array_equal(t2, tof),
          col[[0, 37, 74, 75]])
o, _ = C.perturb('top_1.15', np.full((224, 224, 3), 250, np.uint8), tof)
check('perturb(top_1.15): 255 で切る', o.max() == 255 and o[0, 0, 0] == 255)
st_txt = PROTO['sets']['shift_tests']
check('ずれの試しは protocol の 7 つ(画像 3 行上・3 行下・明るさ ×0.8・×1.2・ToF 1 行下・上 1/3 ×0.85・×1.15)',
      len(C.SHIFTS) == 7 and '7 つ' in st_txt and all(x in st_txt for x in ('3 行上', '3 行下', '×0.8', '×1.2', 'ToF 1 行下',
                                                                           '上 1/3 ×0.85', '×1.15')), st_txt)

# ---------------------------------------------------------------- 3. 窓・入り・周
t = np.arange(0, 20, 1 / 30.0)
m = C.window_mask(t, [5.0, 10.0], -1.0, 0.0)
check('window_mask [tc − 1, tc)', m.sum() == 60 and m[np.searchsorted(t, 4.0)] and not m[np.searchsorted(t, 5.0)])
code = np.array([0, 4, 4, 1, 4, 4, 4, 0, 4])
check('run_starts(DE の入り)', list(C.run_starts(code, 4)) == [1, 4, 8])
lb = [(1.0, 9.0), (9.0, 18.5)]
lo = C.lap_of(np.array([0.5, 1.0, 8.99, 9.0, 18.49, 18.5, np.nan]), lb)
check('lap_of [a, b)', list(lo) == [-1, 0, 0, 1, 1, -1, -1])
L = lambda clean, model, s: {'clean': clean, 'model': model, 'lap_s': s}  # noqa: E731
cases = [(L(True, 'B', 8.5), False, False), (L(True, 'B', 9.49), False, False), (L(True, 'B', 9.5), True, True),
         (L(False, 'B', 8.5), True, False), (L(True, '遅く入る型', 8.9), True, True), (L(False, '最初のモデル', 12), True, False)]
check('lap_unused・lap_straight(きれいでない・B の 9.5 秒以上・ほかのモデル)',
      all(C.lap_unused(x) == u and C.lap_straight(x) == s for x, u, s in cases))

# ---------------------------------------------------------------- 4. スロットル・ステア・止まった場面・G-f
refB, refC = (1441.0, 1496.0), (1433.0, 1500.0)
N = 50
pb = np.zeros((N, 10, 2))
pc = np.zeros((N, 10, 2))
pb[..., 0], pb[..., 1] = 1441.0, 1400.0           # B: 前へ 96 µs
pc[..., 0], pc[..., 1] = 1451.0, 1420.0           # 候補: 前へ 80 µs(中立 1500)、ステア +10
allm = np.ones(N, bool)
v = C.thr_vs_b(pc, pb, refC, refB, allm, 0.90, 0.90)
check('thr_vs_b: 1500 + 0.9·(−80) − (1496 + 0.9·(−96)) = +18.4(+ = 緩める)', abs(v - 18.4) < 1e-9, v)
check('fwd_vs_b = −thr_vs_b(+ = 強く踏む)', abs(C.fwd_vs_b(pc, pb, refC, refB, allm, 0.90, 0.90) + 18.4) < 1e-9)
check('fwd_vs_b: ゲイン 1.00 対 0.90 = −(1420 − 1409.6)', abs(C.fwd_vs_b(pc, pb, refC, refB, allm, 1.00, 0.90) + 10.4) < 1e-9)
pc2 = pc.copy()
pc2[..., 1] = 1560.0                               # ブレーキ側: ゲインは掛からない
check('thr_vs_b: ブレーキ側はゲインを掛けない', abs(C.thr_vs_b(pc2, pb, refC, refB, allm, 0.90, 0.90) - (1560 - 1409.6)) < 1e-9)
check('steer_vs_b: 行 0 の予測ステアの差', abs(C.steer_vs_b(pc, pb, allm) - 10.0) < 1e-12)
check('空の集合は None', C.thr_vs_b(pc, pb, refC, refB, np.zeros(N, bool), 0.9, 0.9) is None and C.go_frac(pc, refC, np.zeros(N, bool), 0.9) is None)
pg = np.zeros((4, 10, 2))
pg[:, 0, 1] = [1500 - 40, 1500 - 30, 1500 - 33.4, 1500 - 34]   # 0.9 を掛けると −36・−27・−30.06・−30.6
check('go_frac: 行 0 の車に出るスロットル < 中立 − 30 の割合 = 0.75', abs(C.go_frac(pg, refC, np.ones(4, bool), 0.90) - 0.75) < 1e-12)
pf = np.zeros((3, 10, 2))
pf[:, 2, 1] = [1396.0, 1296.0, 1600.0]             # B の生の前進 100・200・0(後ろ向き)
lim = C.gf_limit(pf, refB, np.ones(3, bool))
check('gf_limit = 0.10 × B の生の前進の量の平均(後ろ向きは 0)', abs(lim - 0.10 * (100 + 200 + 0) / 3) < 1e-9 and
      abs(lim - 0.10 * C.raw_forward(pf, refB, np.ones(3, bool))) < 1e-9, lim)

# ---------------------------------------------------------------- 5. ずれの試し・速さの見込み
y = rng.normal(1500, 50, (30, 10, 2))
base = y + 3.0
sh = {k: y + 3.0 + (i + 1) for i, k in enumerate(C.SHIFTS)}       # どれも同じ向きにずらす → L1 = 3 + i + 1
dm, per = C.robust_dl1(base, sh, y)
check('robust_dl1: 7 つの L1 の増えの平均 = 4.0', abs(dm - 4.0) < 1e-9 and abs(per['img_up3'] - 1.0) < 1e-9, per)
d, l3 = C.lap_est(10.0)
check('lap_est: −0.02696 × 10 = −0.2696・25.75 + 3 × それ', abs(d + 0.2696) < 1e-12 and abs(l3 - (25.75 - 0.8088)) < 1e-9)
check('lap_est(None)', C.lap_est(None) == (None, None))

# ---------------------------------------------------------------- 6. 段 0 の 2(記録の model_out)
n = 200
pr = rng.normal(1450, 80, (n, 10, 2))
la = rng.choice([1, 3, 5], n)
gains = np.stack([np.ones(n), rng.choice([0.9, 1.0, 1.1], n)], 1)
trim = np.zeros(n)
mo = np.array([apply_gains(pr[i, la[i] - 1], refB, {'steer_trim_us': 0.0, 'steer_gain': gains[i, 0], 'throttle_gain': gains[i, 1]},
                           1000.0, 2000.0) for i in range(n)])
dd = C.stage0_2_diffs(pr, mo, refB, gains, trim, la)
check('stage0_2_diffs: 記録と同じ式なら差 0(先読み・フレームごとのゲイン)', np.abs(dd).max() < 1e-9)
s = C.stage0_2_summary(dd + np.array([3.0, -4.0]))
check('stage0_2_summary: 中央値 3・4 µs → 合格', s['pass'] is True and abs(s['median_abs_us_steer'] - 3) < 1e-9
      and abs(s['median_abs_us_throttle'] - 4) < 1e-9)
s = C.stage0_2_summary(dd + np.array([12.0, -12.0]))
check('stage0_2_summary: 12 µs → 外れ', s['pass'] is False)
s = C.stage0_2_summary(np.stack([np.full(n, 2.0), np.full(n, 14.0)], 1) + rng.normal(0, 0.1, (n, 2)))
check('stage0_2_summary: 合わせた中央値と片方ずつで分かれたら「決まりで決まらない」', s['pass'] is None and '決まりで決まらない' in s['why'],
      (s['median_abs_us_pooled'], s['median_abs_us_throttle']))

class _FakeRunner(object):
    def __init__(self):
        self.calls = 0

    def run_numpy(self, **feed):
        self.calls += 1
        return {}


fr = _FakeRunner()
ts_ = C.time_run_numpy(fr, {})
check('推論の速さ: engine_check と同じ 100 回の空回し + 600 回', fr.calls == 700 and len(ts_) == 600 and (ts_ >= 0).all())
md_, p99_ = C.trt_times(np.r_[np.full(99, 2.0), 5.0])
check('trt_times: 中央値・99% 点', md_ == 2.0 and 2.0 < p99_ <= 5.0)

# ---------------------------------------------------------------- 7. 集合のマスク(作り物の記録 1 本)
fps = 30.0
tt = np.arange(0, 40, 1 / fps)
nr = len(tt)
loc = np.zeros(nr, np.int8)
loc[(tt >= 6) & (tt < 8)] = C.LOC_DETAIL['DE']      # 周 0 の DE(入り 6.0)
loc[(tt >= 8) & (tt < 9)] = C.LOC_DETAIL['SW']
loc[(tt >= 9) & (tt < 10)] = C.LOC_DETAIL['HS']
loc[(tt >= 26) & (tt < 28)] = C.LOC_DETAIL['DE']    # 周 1 の DE(入り 26.0)
ai = np.ones(nr, bool)
ai[(tt >= 35)] = False                              # 35 秒から人
erpm = np.full(nr, -5000.0)
erpm[(tt >= 30) & (tt < 31)] = 100.0                # 止まった場面 1 秒(30 フレーム)
near = np.zeros(nr, bool)
near[(tt >= 4) & (tt <= 7)] = True                  # 5.0 秒の衝突の 1 秒前〜2 秒後
straight = (tt >= 12) & (tt < 13)                   # 周 0 の直線 1 秒
straight |= (tt >= 32) & (tt < 33)                  # 周 1 の直線 1 秒
dlog = {'ai': ai, 'near': near, 'erpm': erpm, 'straight': straight, 'loc_detail': loc,
        'model_out': np.full((nr, 2), 1450.0), 'hold': np.zeros(nr, bool), 'run': np.full(nr, C.B_RUN),
        'colls': [{'t_s': 5.0, 'ai': True, 'loc': 'R1', 'loc_detail': 'R1'},
                  {'t_s': 27.0, 'ai': True, 'loc': 'DE', 'loc_detail': 'DE'},
                  {'t_s': 18.0, 'ai': True, 'loc': 'other', 'loc_detail': 'SW'},
                  {'t_s': 36.0, 'ai': False, 'loc': 'R1', 'loc_detail': 'R1'}]}
laps_b = [(1.0, 20.0, {'clean': True, 'model': 'B', 'lap_s': 19.0}),      # B のきれいな周 19 秒(9.5 秒以上 → 使っていない)
          (20.0, 34.0, {'clean': False, 'model': 'B', 'lap_s': 8.9})]      # きれいでない周
de_ent = [(6.0, 0), (26.0, 1)]
k = np.arange(nr)
has = np.ones(nr, bool)
src = np.where(ai, 1, 0)
Mk = C.scene_masks(dlog, k, has, src, tt, laps_b, de_ent, [1.0, 20.0], True, True, np.zeros(nr, bool))
check('ai_pre_r1: [4, 5) の AI 30 フレーム', Mk['pre_r1'].sum() == 30)
check('ai_pre_de: [26, 27) 30 フレーム・人の衝突は数えない', Mk['pre_de'].sum() == 30 and not Mk['pre_r1'][tt >= 35].any())
check('ai_pre_sw: loc_detail SW の衝突 [17, 18)', Mk['pre_sw'].sum() == 30)
check('de_pass: 2 つの周の DE の入りから 0.5 秒(15 + 15)', Mk['de_pass'].sum() == 30
      and Mk['de_pass'][(tt >= 6) & (tt < 6.5)].all())
check('straight: きれいな周(B 19 秒)の直線だけ(きれいでない周の直線は入れない)', Mk['straight'].sum() == 30
      and not Mk['straight'][tt >= 20].any())
check('stand_pre: |eRPM| < 400・AI・衝突の前後でない', Mk['stand_pre'].sum() == 30)
check('gb: AI・衝突の前後でない(衝突の前後 [4, 7] を除く)', Mk['gb'].sum() == int((ai & ~near).sum()))
check('gb_de・sw・hs(loc_detail)', Mk['gb_de'].sum() == int(((loc == 4) & ai & ~near).sum()) and Mk['gb_sw'].sum() == 30
      and Mk['gb_hs'].sum() == 30)
check('gb_r1pre: 起点 1・20 の手前 1 秒(起点 1 の手前 [0, 1)・20 の手前 [19, 20))', Mk['gb_r1pre'].sum() == 60)
check('gb_deline: DE の入りから [0.5, 0.7)(26.5〜26.7 の 6 フレーム。6.5〜6.7 は衝突の前後 [4, 7] なので外す)',
      Mk['gb_deline'].sum() == 6 and Mk['gb_deline'][(tt >= 26.5) & (tt < 26.7)].all())
check('gbref: 手本の外・AI・衝突の前後でない', Mk['gbref'].sum() == Mk['gb'].sum())
keep = (tt >= 1) & (tt < 20)
Mk2 = C.scene_masks(dlog, k, has, src, tt, laps_b, de_ent, [1.0, 20.0], False, True, keep)
check('gbref: 手本のフレームを外す・gb は 3 本の記録だけ', Mk2['gbref'].sum() == int((ai & ~near & ~keep).sum()) and Mk2['gb'].sum() == 0)
check('s0_2: B の記録・model_out が数値・hold でない', Mk['s0_2'].sum() == int(ai.sum()))
d2 = dict(dlog, model_out=np.where((tt < 2)[:, None], np.nan, 1450.0), run=np.full(nr, 'bench_20261022-slow-sec-s1'))
check('s0_2: B 以外の記録は入れない', C.scene_masks(d2, k, has, src, tt, laps_b, de_ent, [], False, True, keep)['s0_2'].sum() == 0)
laps_used = [(1.0, 20.0, {'clean': True, 'model': 'B', 'lap_s': 9.2}), laps_b[1]]
Mk3 = C.scene_masks(dlog, k, has, src, tt, laps_used, de_ent, [], False, False, np.zeros(nr, bool))
check('B のきれいな 9.5 秒未満の周は de_pass・straight に入れない', Mk3['de_pass'].sum() == 15 and Mk3['straight'].sum() == 0)
has2 = has.copy()
has2[:150] = False
Mk4 = C.scene_masks(dlog, k, has2, src, tt, laps_b, de_ent, [1.0, 20.0], True, True, np.zeros(nr, bool))
check('label_index・vehicle_valid の無いフレームは入れない', Mk4['pre_r1'].sum() == 0)

# ---------------------------------------------------------------- 8. 欄(row_fields)
Mk['stand'] = Mk['stand_pre']                          # 本番は ToF の x_free > 1.0 m を cmd_sets で掛ける
NT = nr
pb_t = np.zeros((NT, 10, 2))
pb_t[..., 0], pb_t[..., 1] = 1441.0, 1400.0
pc_t = pb_t.copy()
pc_t[..., 1] = 1380.0                                  # 候補(中立 1500): 1500 + 0.9·(−120) = 1392、B 1409.6 → −17.6
pc_t[..., 0] = 1446.0
r = C.row_fields(pc_t, pb_t, refC, refB, Mk, 'bfast', [4, 5])
check('row_fields: de_pre_thr_vs_b = −17.6・straight_fwd = +17.6・G-b(fwd) = +17.6', abs(r['de_pre_thr_vs_b_us'] + 17.6) < 1e-9
      and abs(r['straight_fwd_vs_b_us'] - 17.6) < 1e-9 and abs(r['bfast_gb_fwd_vs_b_us'] - 17.6) < 1e-9)
check('row_fields: 1.00 対 0.90 = 1409.6 − 1380 = 29.6・DE の線 = +5', abs(r['bfast_gb_fwd_g100_vs_b090_us'] - 29.6) < 1e-9
      and abs(r['bfast_de_line_vs_b_us'] - 5.0) < 1e-9)
check('row_fields: lap_delta = −0.02696 × 17.6', abs(r['lap_delta_est_s'] + 0.02696 * 17.6) < 1e-9)
check('row_fields: standstill_go_frac = 1(前へ 108 µs)', r['standstill_go_frac'] == 1.0)
r2 = C.row_fields(pc_t, pb_t, refC, refB, Mk, 'v2', [4, 5])
check('row_fields: v2 には bfast_* を出さない・g9 も出さない', all(r2[k] is None for k in r2 if k.startswith('bfast_'))
      and r2['g9_minus_g5_pre_thr_us'] is None)
rows9 = np.nonzero(Mk['pre_r1'] | Mk['pre_de'])[0]
p9 = pc_t[rows9].copy()
p9[..., 1] = 1370.0                                    # ジャンル 9(ゲイン 1.00): 1370 − (1500 + 0.9·(−120)) = −22
r3 = C.row_fields(pc_t, pb_t, refC, refB, Mk, 'v2', [4, 5, 9], p9, rows9)
check('row_fields: g9_minus_g5 = 1370 − 1392 = −22(− = ジャンル 9 が強く踏む)', abs(r3['g9_minus_g5_pre_thr_us'] + 22.0) < 1e-9)

# ---------------------------------------------------------------- 9. 欄の名前(eval_v2・protocol)
import eval_v2 as E  # noqa: E402
miss = [k for k in E.C3_FIELDS if k not in C.FIELDS_C3]
check('FIELDS_C3 は eval_v2.C3_FIELDS をすべて含む(+ trt_p99_ms)', not miss and 'trt_p99_ms' in C.FIELDS_C3, miss)
names = set()
for key in PROTO['fields_added']:
    for part in key.split('・'):
        names.add(part)
expand = set()
for nm in names:
    if '<区間>' in nm:
        expand |= {nm.replace('<区間>', z) for z in ('R1_app', 'R1_turn', 'GM', 'DE', 'none', 'unknown')}
    elif nm != 'ref_values':
        expand.add(nm)
have = set(C.FIELDS_C3) | set(C.FIELDS_H1)
check('protocol fields_added の欄をすべて出す(H1 の欄は eval_v2.run_row で)', expand <= have, sorted(expand - have))
check('protocol fields_desk_bench_same(gates_pass を除く)を出す',
      set(PROTO['fields_desk_bench_same']['list']) - {'gates_pass'} <= have | {'trt_median_ms', 'trt_p99_ms', 'engine_ok'},
      sorted(set(PROTO['fields_desk_bench_same']['list']) - have))

# ---------------------------------------------------------------- 10. 数が protocol の文と同じ
gtxt = PROTO['sign_rules']['gains']
want = {'B': r'B ([0-9.]+)', 'g5': r'ジャンル 5 ([0-9.]+)', 'bfast': r'B-FAST ([0-9.]+)', 'g9': r'ジャンル 9 ([0-9.]+)'}
check('GAINS = protocol sign_rules.gains', all(abs(float(re.search(p, gtxt).group(1)) - C.GAINS[k]) < 1e-12 for k, p in want.items()), gtxt)
sets = PROTO['sets']
check('止まった場面: |eRPM| < 400・x_free > 1.0 m', '|eRPM| < 400' in sets['standstill'] and 'x_free > 1.0 m' in sets['standstill']
      and C.STAND_ERPM == 400 and C.TOF_FREE_M == 1.0)
check('止まった場面: 中立 − 30 µs', '中立 − 30 µs' in PROTO['fields_added']['standstill_go_frac'] and C.STAND_GO_US == 30)
check('DE の通過 0.5 秒・DE の線 0.5〜0.7 秒', '入って 0.5 秒' in sets['de_pass'] and '0.5〜0.7 秒' in PROTO['fields_added']['bfast_de_line_vs_b_us']
      and C.DE_PASS_S == (0.0, 0.5) and C.DE_LINE_S == (0.5, 0.7))
check('B の 9.5 秒以上', 'B の 9.5 秒以上' in sets['de_pass'] and 'B の 9.5 秒以上' in sets['straight'] and C.LAP_B_UNUSED_S == 9.5)
lt = PROTO['fields_added']['lap_delta_est_s・lap3_est_s']
check('速さの見込みの係数 −0.02696・25.75', '-0.02696' in lt and '25.75' in lt and C.LAP_COEF_S_PER_US == -0.02696 and C.LAP3_BASE_S == 25.75)
check('gb_main の 3 本', all(x in sets['gb_main'] for x in C.GB_LOGS))
check('門「推論」1.3 倍・止まった場面 300', '1.3' in PROTO['gates_v2']['inference'] and '300' in PROTO['gates_v2']['standstill']
      and C.INFER_FACTOR == 1.3 and C.STANDSTILL_MIN_FRAMES == 300)
check('standstill_gate_kind', C.standstill_gate_kind(300) == '門' and C.standstill_gate_kind(299) == '参考' and C.standstill_gate_kind(None) is None)

# ---------------------------------------------------------------- 11. 封
try:
    C.seal_guard(['20261024-183123'], PROTO['seal']['logs'])
    ok = False
except SystemExit as e:
    ok = '封' in str(e)
check('封の記録を読もうとしたら止まる', ok)
try:
    C.seal_guard(['20261024-162033', '20261023-210840'], PROTO['seal']['logs'])
    ok = True
except SystemExit:
    ok = False
check('学習の回の記録は通す', ok)
check('学習の回の 46 本に封の記録が無い', not set(C.ai_logs_of(PROTO)) & set(PROTO['seal']['logs']) and len(C.ai_logs_of(PROTO)) == 46)

# ---------------------------------------------------------------- 12. desk_v2.json の形が eval_v2 の門に入る
b_row = {'l1_best_us': 49.1, 'standstill_go_frac': 0.60, 'thr_bias_us': 11.0, 'steer_amp_ratio': 1.0}
cand = {'l1_best_us': 49.5, 'engine_ok': True, 'sanity_ok': True, 'trt_p99_ms': 2.6, 'steer_amp_ratio': 1.0, 'brake_lowspeed_frac': 0.0,
        'standstill_go_frac': 0.55, 'thr_bias_us': 12.0, 'straight_thr_vs_b_us': -3.0, 'bfast_gb_fwd_vs_b_us': 12.0,
        'bfast_gc_fwd_vs_b_us': 2.0, 'straight_fwd_vs_b_us': 3.0}
desk = {'b_trt_p99_ms_same_day': 2.5, 'standstill_frames': 400, 'bfast_gf_limit_us': 9.0, 'runs': {}}
g = E.gates_v2(cand, b_row, desk, 15.964)
check('eval_v2.gates_v2 が desk_v2 の欄で全部決まる', E.gates_pass(g) is True, g)
gb = E.gates_bfast(cand, b_row, desk)
check('eval_v2.gates_bfast が desk_v2 の欄で全部決まる(G-a〜G-f・安全)', E.gates_pass(gb) is True, gb)
gb = E.gates_bfast(dict(cand, bfast_gb_fwd_vs_b_us=9.9), b_row, desk)
check('G-b: +10 µs 未満なら不合格', gb['G-b'] is False)
g = E.gates_v2(cand, b_row, dict(desk, standstill_frames=299), 15.964)
check('止まった場面が 300 未満なら「参考」(合否に数えない)', g['standstill'] == '参考' and E.gates_pass(g) is True)

# ---------------------------------------------------------------- 13. report の道筋(作り物の予測・TensorRT で。本物の集合・H1・保存した予測を読む)
import tempfile  # noqa: E402
import types  # noqa: E402
if C.SETS_NPZ.exists():
    tmp = Path(tempfile.mkdtemp(prefix='c3_test_', dir=str(HERE / 'logs')))
    import atexit  # noqa: E402
    import shutil  # noqa: E402
    atexit.register(shutil.rmtree, str(tmp), True)          # 途中で落ちても作り物の予測を残さない
    C.PRED, C.TRT = tmp / 'pred', tmp / 'trt'
    S = C.load_sets()
    nT = len(S['frame'])
    base = 1450.0 + rng.normal(0, 30, (nT, 10, 2))
    names_t = {C.B_RUN: 0.0, 'v2_20261025-v2h-e60-s0': -5.0, 'bfast_20261025-e15-s0': -12.0, 'v2_20261025-v2h-e30-s0': -1.0}
    for nm, off in names_t.items():
        p_ = base.copy()
        p_[..., 1] += off                                    # スロットルを off µs ずらす(− = 強く踏む)
        C.save_npz(C.tab_file(nm), key=np.array(S['key']), pred=p_.astype(np.float32), genre=np.full(nT, 5, np.int8),
                   substituted=np.zeros(nT, bool))
    nm = 'v2_20261025-v2h-e60-s0'
    sv = C.saved_best(nm)
    with np.load(C.BUNDLE) as z:
        prim = z['primary'].astype(bool)
    rows = np.nonzero(prim)[0]
    shf = {('pred_' + k): (sv[rows] + (i + 1)).astype(np.float32) for i, k in enumerate(C.SHIFTS)}
    C.save_npz(C.shift_file(nm), rows=rows, pred_none_rows=np.arange(4), pred_none=sv[rows][:4].astype(np.float32),
               none_vs_saved_max_abs_us=np.array(0.0), **shf)
    job = '20991231-000000'
    eng = {}
    for nm2 in names_t:
        e = C.RUNS / nm2 / C.ENGINE
        st_ = e.stat()
        nfr = 50
        pr_ = np.full((nfr, 10, 2), 1450.0, np.float32)
        f_ = C.TRT / job / ('%s.npz' % nm2)
        C.save_npz(f_, pred=pr_, model_pred=pr_, ms=np.full(nfr, 2.5, np.float32), erpm=np.full(nfr, -100.0, np.float32),
                   part=np.zeros(nfr, np.int8))
        eng[nm2] = {'run': nm2, 'engine': {'size': st_.st_size, 'mtime_ns': st_.st_mtime_ns}, 'check_io_ok': True, 'frames': nfr,
                    'nonfinite_frames': 0, 'out_of_range_values': 0, 'run_numpy_median_ms': 2.4,
                    'run_numpy_p99_ms': 2.5 if nm2 == C.B_RUN else 2.6, 'file': str(f_)}
    C.write_json(C.TRT / ('%s.json' % job), {'schema': 'jf-c3-trt/v2', 'job': job, 'day': job[:8], 'engines': eng})
    C.write_json(C.TRT / '20991230-000000.json', {'schema': 'jf-c3-trt/v1', 'job': '20991230-000000', 'day': '20991230',
                                                  'engines': {C.B_RUN: dict(eng[C.B_RUN], run_numpy_p99_ms=9.9)}})
    outd = tmp / 'out'
    ens = 'v2_20261025-v2h-e60-s0+v2_20261025-v2h-e30-s0'
    rc = C.cmd_report(types.SimpleNamespace(out=str(outd), ens=[ens]))
    o = json.load(open(outd / 'desk_v2.json', encoding='utf-8'))
    rv, rb, rB = o['runs']['v2_20261025-v2h-e60-s0'], o['runs']['bfast_20261025-e15-s0'], o['runs'][C.B_RUN]
    check('report: 動く・4 本 + 2 本の平均の行', rc == 0 and len(o['runs']) == 5)
    re_ = o['runs']['ens:' + ens]
    check('report: 2 本の平均の行 = 予測の平均(−5 と −1 → −3 µs を 0.9 倍)・門は推論を除く',
          abs(re_['straight_thr_vs_b_us'] - C.thr_vs_b(base + np.array([0.0, -3.0]), base, (1433.0, 1500.0), (1441.0, 1496.0),
                                                      S['straight'].astype(bool), 0.9, 0.9)) < 1e-4
          and 'inference' not in re_['gates'] and re_['gates_pass'] is None and 'gates_pass_except_inference' in re_
          and re_['trt_p99_ms'] is None and re_['sanity_ok'] is True, re_.get('gates'))
    check('report: v2 の _thr_vs_b = 0.9 × (−5)(前進側)・straight_fwd = +4.5', abs(rv['de_pre_thr_vs_b_us'] - (-4.5) - 4.0 * 0.0)
          < 2.0 and abs(rv['straight_fwd_vs_b_us'] + rv['straight_thr_vs_b_us']) < 1e-6, (rv['de_pre_thr_vs_b_us'], rv['straight_fwd_vs_b_us']))
    yb = E.load_bundle(PROTO)['y'][rows]
    l0 = float(np.abs(sv[rows] - yb).mean())
    want_r = float(np.mean([float(np.abs(sv[rows] + (i + 1) - yb).mean()) - l0 for i in range(7)]))
    check('report: robust_dl1_us = 7 つの L1(H1)− そのまま の平均(別に数えた値と同じ)', abs(rv['robust_dl1_us'] - want_r) < 1e-4,
          (rv['robust_dl1_us'], want_r))
    check('report: B-FAST の G-b の欄がある・v2 には無い', rb['bfast_gb_fwd_vs_b_us'] is not None and rv['bfast_gb_fwd_vs_b_us'] is None)
    check('report: B の _vs_b は 0', abs(rB['de_pre_thr_vs_b_us']) < 1e-9 and abs(rB['straight_thr_vs_b_us']) < 1e-9)
    check('report: 同じ日の B の p99・推論の可否(速さを sanity の流れで測った v1 の仕事は使わない)',
          o['b_trt_p99_ms_same_day'] == 2.5 and rv['trt']['inference_ok_same_day'] is True)
    check('report: sanity_ok・低速のブレーキ(作り物は 0)', rv['sanity_ok'] is True and rv['brake_lowspeed_frac'] is not None)
    check('report: 門(eval_v2 の関数)が決まる', rv.get('gates_pass') in (True, False, None) and isinstance(rv.get('gates'), dict)
          and isinstance(rb.get('gates'), dict) and 'G-b' in rb['gates'], (rv.get('gates'), rb.get('gates')))
    check('report: H1 の欄(eval_v2.run_row)', rv.get('l1_best_us') is not None and rv.get('thr_bias_us') is not None)
    check('report: 段 0 の 2・4・8 の形', set(o['stage0']) == {'2', '4', '8'} and o['stage0']['4']['standstill_frames'] == int(S['stand'].sum())
          and o['stage0']['8']['b_trt_p99_ms_same_day'] == 2.5)
    check('report: eval_v2 が読む上の欄', all(k in o for k in ('b_trt_p99_ms_same_day', 'standstill_frames', 'bfast_gf_limit_us'))
          and o['bfast_gf_limit_us'] is not None)
    shutil.rmtree(str(tmp), ignore_errors=True)
else:
    check('report の道筋(sets_v2.npz が無いので飛ばした)', True)

ng = [n for n, ok in results if not ok]
print('\n%d 個中 %d 個 OK%s' % (len(results), len(results) - len(ng), '' if not ng else '・NG: %s' % ng))
sys.exit(1 if ng else 0)
