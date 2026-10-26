#!/usr/bin/env python3
"""机上の評価 v2(2026-10-25): 学習が終わったあとに、protocol_v2.json の指標・決まり・拒否権・印・門を結果に当てて表を出す。
車・装置に触れない。走行の API は呼ばない(GPU の予測の前に GET の状態だけ見る)。data/・既定値は変えない(読むだけ)。

    cd ~/jetformer
    # ① 予測(GPU。必ず jf-heavy --lock big。B の予測し直しは段 0 の確かめ 1 と同じ: 10/12 の保存と ≤ 0.01 µs)
    /home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        MALLOC_ARENA_MAX=2 /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261025_train_v2/eval_v2.py predict --b-crosscheck
    ... eval_v2.py predict --runs <学習の名前> ...        # v2・B-FAST は protocol_v2_lock2.json のあとだけ(無ければ止まる)
    # ② 判定(CPU・numpy だけ。torch を読まない)
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261025_train_v2/eval_v2.py judge
    ... eval_v2.py judge --dry-run-b                        # 空回し: 22 本の枠を全部モデル B の予測で埋めて、決まりの道筋を通す

予測の表: tools/rl_evaluate.pred_table(key 3d339512…、10,985 フレーム。10/21・10/22 の desk と同じ)。
集合: desk_inputs_20261022.npz(sha256 は protocol の reference_predictions_sha256.bundle)の primary = H1 4,670 フレーム・
interv = 介入 1,570 フレーム・r1_pre1s(R1 の起点の手前 1 秒)・zone(10/22 の zones2)・nonzone。ジャンルは H1 の aligned の mode。
5 本(A_s0・A_s1・v0-s2・v0-s3・B)は保存した予測(sha256 は protocol の値と照らす)。

この台本が計算する欄(protocol fields_desk_bench_same・fields_added のうち H1・介入の予測で出せるもの):
  l1_best_us・l1_steer_best_us・l1_thr_best_us・l1_last_us・l1_interv_us・l1_interv_last_us・l1_zone_<区間>_us・l1_day_us・
  l1_night_us・l1_genre4_us・l1_genre5_us(と _last)・steer_bias_us・thr_bias_us(と _last)・steer_amp_ratio・r1_app_thr_diff_us・
  nonzone_thr_diff_us・engine_ok・trt_median_ms・trt_p99_ms(engine_check.json の run_numpy)・印(長すぎ・上がり)
C3(机上のベンチ v2。まだ無い)が出す欄は /home/jetson/jetformer_reports/train_v2_20261025/desk/desk_v2.json から読む:
  sanity_ok・brake_lowspeed_frac・robust_dl1_us・de/r1/sw_pre_thr_vs_b_us・de_overshoot_vs_b_us・straight_thr_vs_b_us・
  straight_fwd_vs_b_us・standstill_go_frac・g9_minus_g5_pre_thr_us・bfast_gb_fwd_vs_b_us(_de・_sw・_hs)・bfast_gc_fwd_vs_b_us・
  bfast_gb_ref_fwd_vs_b_us・bfast_gb_fwd_g100_vs_b090_us・bfast_de_line_vs_b_us・lap_delta_est_s・lap3_est_s・同じ日の trt_p99_ms
  desk_v2.json の形: {"runs": {"<学習の名前>": {"<欄>": 値, ...}, ...}, "b_trt_p99_ms_same_day": 値, "standstill_frames": 数,
                      "bfast_gf_limit_us": 0.10 × B の直線の生の前進の量の平均}(B の行は B の学習の名前で)
  無い欄は「未(C3)」。その欄を使う門・決まりは「未」で、合格にしない。

出力: /home/jetson/jetformer_reports/train_v2_20261025/eval/(pred/・crosscheck_B.json・eval_v2.json・eval_v2.md・
      decisions_proposed.json)。空回しは eval/dryrun_B/。decisions.json(キューが読む答え)は書かない。
      main session が eval_v2.md と decisions_proposed.json を見て decisions.json に写す。
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
HERE = REPO / 'experiments' / '20261025_train_v2'
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL = REP / 'protocol_v2.json'
LOCK2 = REP / 'protocol_v2_lock2.json'
DECISIONS = REP / 'decisions.json'
DESK_C3 = REP / 'desk' / 'desk_v2.json'
EVAL = REP / 'eval'
PRED = EVAL / 'pred'
RUNS = REPO / 'data' / 'runs'
DATA = REPO / 'data'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ZOO = DATA / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = DATA / 'zoo' / 'real_accuracy_20261015' / 'pred'
H1_DS = DATA / 'datasets' / 'clean_driving_20261011'
ENGINE = 'jetformer_fp16.engine'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
# B と同じ作り方の 5 本(protocol_numbers.py と同じ。sha256 は protocol の reference_predictions_sha256)
FIVE = [
    ('A_s0', 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz', 0),
    ('A_s1', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30', ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz', 1),
    ('v0-s2', 'realacc_20261015-v0-s2', RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz', 2),
    ('v0-s3', 'realacc_20261015-v0-s3', RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz', 3),
    ('B', B_RUN, ZOO / 'B.npz', ZOO / 'B_last.npz', None),
]
ZONES = ('R1_app', 'R1_turn', 'GM', 'DE', 'none', 'unknown')
LOOKAHEAD_ROW = 2                  # 走る日の先読み 3(desk_lib の LOOKAHEAD_ROW)
TOO_LONG_US = 1.6                  # marks.too_long
RISE_US, RISE_EPOCHS = 1.0, 10     # marks.rise
AMP = (0.80, 1.25)
L1_VS_B_US = 10.0
BRAKE_FRAC = 0.01
STANDSTILL_MIN_FRAMES = 300
INFER_FACTOR = 1.3
STRAIGHT_US = 10.0
G9_MIN_US = -5.0
GA_PLUS_US = 1.0
GB_MIN_US, GC_MAX_US = 10.0, 5.0
C3_FIELDS = ('sanity_ok', 'brake_lowspeed_frac', 'robust_dl1_us', 'de_pre_thr_vs_b_us', 'r1_pre_thr_vs_b_us',
             'sw_pre_thr_vs_b_us', 'de_overshoot_vs_b_us', 'straight_thr_vs_b_us', 'straight_fwd_vs_b_us',
             'standstill_go_frac', 'g9_minus_g5_pre_thr_us', 'bfast_gb_fwd_vs_b_us', 'bfast_gb_fwd_vs_b_us_de',
             'bfast_gb_fwd_vs_b_us_sw', 'bfast_gb_fwd_vs_b_us_hs', 'bfast_gc_fwd_vs_b_us', 'bfast_gb_ref_fwd_vs_b_us',
             'bfast_gb_fwd_g100_vs_b090_us', 'bfast_de_line_vs_b_us', 'lap_delta_est_s', 'lap3_est_s')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x, nd=3):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    v = num(x)
    return None if v is None else round(v, nd)


def protocol():
    return load_json(PROTOCOL)


def pred_file(run, ck):
    return PRED / ('%s%s.npz' % (run, '' if ck == 'best' else '_last'))


# ---------------------------------------------------------------- 名前 → 腕・種

def parse_name(name):
    """学習の名前 → (家族, 腕, 種)。v2_20261025-<腕>-s<種>・bfast_20261025-<腕>-s<種>"""
    m = re.match(r'^(v2|bfast)_20261025-(.+)-s(\d+)$', name)
    if not m:
        return None, None, None
    fam, arm, seed = m.group(1), m.group(2), int(m.group(3))
    return fam, ('bfast-' + arm if fam == 'bfast' else arm), seed


def protocol_patterns(proto):
    out = []
    for r in proto['runs']:
        pat = '^' + re.escape(r['name']).replace(r'\{x\}', '(v2h|v2m|base)').replace(r'\{E\}', '(30|60)').replace(
            r'\{A\}', '(-aug2)?') + '$'
        out.append(re.compile(pat))
    return out


def slot_names(x='v2h', E=60, aug='AUG2'):
    """空回しの 22 本の名前(型を埋めたもの。順番は B-FAST → v2)"""
    proto = protocol()
    names = []
    isbf = lambda r: str(r['stage']) == 'B-FAST' or r['family'] in ('bfast', 'B-FAST')  # noqa: E731
    rows = [r for r in proto['runs'] if isbf(r)] + [r for r in proto['runs'] if not isbf(r)]
    for r in rows:
        n = r['name'].replace('{x}', x).replace('{E}', str(E)).replace('{A}', '-aug2' if aug == 'AUG2' else '')
        names.append(n)
    return names


# ---------------------------------------------------------------- 入力

def load_bundle(proto):
    want = proto['reference_predictions_sha256']['bundle']
    if sha256(BUNDLE) != want:
        raise SystemExit('desk_inputs_20261022.npz の sha256 が protocol と違います')
    with np.load(BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        S = {k: z[k] for k in ('part', 'scene', 'frame', 'erpm', 'primary', 'interv', 'r1_pre1s', 'zone', 'nonzone')}
        S['y'] = np.nan_to_num(np.asarray(z['y'], np.float64))
    if int(S['primary'].sum()) != proto['holdouts']['H1']['frames_desk'] or int(S['interv'].sum()) != proto['holdouts']['interv']['frames_desk']:
        raise SystemExit('H1・介入のフレーム数が protocol と違います')
    sealed = set(proto['seal']['logs'])
    if any(str(s)[:15] in sealed for s in np.unique(S['scene'])):
        raise SystemExit('予測の表に封の記録があります(11/1 より前は読まない)')
    S['genre'] = h1_genre(S['scene'], S['frame'], S['primary'])
    if int((S['primary'] & (S['genre'] == 4)).sum()) != proto['holdouts']['H1']['genre_frames']['4']:
        raise SystemExit('H1 のジャンル 4 のフレーム数が protocol と違います')
    day = set(proto['holdouts']['day_night']['day'])
    S['day'] = S['primary'] & np.isin(S['scene'], sorted(day))
    S['night'] = S['primary'] & ~S['day']
    return S


def h1_genre(scene, frame, prim):
    g = np.full(len(scene), -1, np.int16)
    for s in sorted(set(scene[prim].tolist())):
        with np.load(H1_DS / 'aligned' / ('%s.npz' % s), allow_pickle=False) as a:
            mode = a['mode']
        m = scene == s
        g[m] = mode[frame[m]]
    return g


def load_pred(path, ck=None):
    with np.load(path, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('%s: 予測の表の key が違います' % path)
        if ck and 'ckpt' in z.files and str(z['ckpt']) != ck:
            raise SystemExit('%s: ckpt が %s ではありません' % (path, ck))
        p = np.asarray(z['pred'], np.float64)
    if p.shape != (10985, 10, 2) or not np.isfinite(p).all():
        raise SystemExit('%s: 予測の形か値がおかしい' % path)
    return p


def five_preds(proto):
    ref = proto['reference_predictions_sha256']
    out = {}
    for rid, run, pb, pl, seed in FIVE:
        for ck, p in (('best', pb), ('last', pl)):
            if sha256(p) != ref[rid][ck]:
                raise SystemExit('%s の保存した予測(%s)の sha256 が protocol と違います' % (rid, ck))
        out[rid] = {'run': run, 'seed': seed, 'best': load_pred(pb), 'last': load_pred(pl)}
    return out


# ---------------------------------------------------------------- 欄の計算(numpy)

def metrics(pred, S):
    y, prim, iv = S['y'], S['primary'], S['interv']
    d = np.abs(pred - y)
    out = {'l1': float(d[prim].mean()), 'l1_steer': float(d[prim][:, :, 0].mean()), 'l1_thr': float(d[prim][:, :, 1].mean()),
           'l1_interv': float(d[iv].mean()),
           'steer_bias': float((pred[prim][:, 0, 0] - y[prim][:, 0, 0]).mean()),
           'thr_bias': float((pred[prim][:, 0, 1] - y[prim][:, 0, 1]).mean())}
    for g in (4, 5):
        out['l1_genre%d' % g] = float(d[prim & (S['genre'] == g)].mean())
    out['l1_day'] = float(d[S['day']].mean())
    out['l1_night'] = float(d[S['night']].mean())
    for z in ZONES:
        m = prim & (S['zone'] == z)
        out['l1_zone_%s' % z] = float(d[m].mean()) if m.any() else None
    hs = np.std(y[prim][:, :, 0])
    out['steer_amp_ratio'] = float(np.std(pred[prim][:, :, 0]) / hs) if hs > 0 else None
    r1 = prim & S['r1_pre1s']
    out['r1_app_thr_diff'] = float((pred[r1][:, LOOKAHEAD_ROW, 1] - y[r1][:, LOOKAHEAD_ROW, 1]).mean()) if r1.any() else None
    nz = prim & S['nonzone']
    out['nonzone_thr_diff'] = float((pred[nz][:, 0, 1] - y[nz][:, 0, 1]).mean()) if nz.any() else None
    return out


def engine_fields(run_dir):
    """engine_check.json が今のエンジンのものなら engine_ok と run_numpy の速さ(drive.engine_check_of と同じ規則)"""
    c = load_json(run_dir / 'engine_check.json')
    eng = run_dir / ENGINE
    cur = False
    if isinstance(c, dict) and eng.exists():
        st = eng.stat()
        e = c.get('engine') or {}
        try:
            cur = int(e.get('size', -1)) == st.st_size and int(e.get('mtime_ns', -1)) == st.st_mtime_ns
        except (TypeError, ValueError):
            cur = False
    if not cur:
        return {'engine_ok': False if eng.exists() else None, 'trt_median_ms': None, 'trt_p99_ms_engine_check': None,
                'engine_actions_max_us': None, 'engine_note': 'engine_check.json が無いか今のエンジンのものでない'}
    mx = num(c.get('actions_max_us'))
    ok = bool(c.get('accuracy_ok') is True and mx is not None and mx <= 2.5)
    return {'engine_ok': ok, 'trt_median_ms': num(c.get('run_numpy_median_ms')),
            'trt_p99_ms_engine_check': num(c.get('run_numpy_p99_ms')), 'engine_actions_max_us': mx,
            'engine_note': 'engine_check.json(%s)' % c.get('checked_at')}


def rise_mark(history):
    """marks.rise(参考): 検証の L1 の 5 エポックの移動平均が、それまでの最小より 1.0 µs 以上高い状態が 10 エポック続く"""
    v = [num(h.get('val_us')) for h in (history or [])]
    v = [x for x in v if x is not None]
    if len(v) < 5 + RISE_EPOCHS:
        return False
    ma = np.convolve(v, np.ones(5) / 5.0, mode='valid')
    run = 0
    for i in range(1, len(ma)):
        if ma[i] >= ma[:i].min() + RISE_US:
            run += 1
            if run >= RISE_EPOCHS:
                return True
        else:
            run = 0
    return False


def run_row(name, best, last, S, c3, run_dir=None, ref=None):
    mb = metrics(best, S)
    ml = metrics(last, S) if last is not None else None
    fam, arm, seed = parse_name(name)
    r = {'run': name, 'family': fam, 'arm': arm, 'seed': seed,
         'l1_best_us': mb['l1'], 'l1_steer_best_us': mb['l1_steer'], 'l1_thr_best_us': mb['l1_thr'],
         'l1_last_us': ml and ml['l1'], 'l1_interv_us': mb['l1_interv'], 'l1_interv_last_us': ml and ml['l1_interv'],
         'l1_day_us': mb['l1_day'], 'l1_night_us': mb['l1_night'],
         'l1_genre4_us': mb['l1_genre4'], 'l1_genre5_us': mb['l1_genre5'],
         'l1_genre4_last_us': ml and ml['l1_genre4'], 'l1_genre5_last_us': ml and ml['l1_genre5'],
         'steer_bias_us': mb['steer_bias'], 'thr_bias_us': mb['thr_bias'],
         'steer_bias_last_us': ml and ml['steer_bias'], 'thr_bias_last_us': ml and ml['thr_bias'],
         'steer_amp_ratio': mb['steer_amp_ratio'], 'r1_app_thr_diff_us': mb['r1_app_thr_diff'],
         'nonzone_thr_diff_us': mb['nonzone_thr_diff']}
    for z in ZONES:
        r['l1_zone_%s_us' % z] = mb['l1_zone_%s' % z]
    r['mark_too_long'] = None if ml is None else bool(ml['l1'] - mb['l1'] > TOO_LONG_US)
    if run_dir is not None:
        r.update(engine_fields(run_dir))
        res = load_json(run_dir / 'result.json') or {}
        cfg = load_json(run_dir / 'model_config.json') or {}
        r['mark_rise'] = rise_mark(res.get('history'))
        r['train_best_val_us'] = res.get('best_val_us')
        r['train_best_epoch'] = res.get('best_epoch')
        r['epochs_done'] = res.get('epochs_done')
        r['steer_center_us'] = cfg.get('steer_center_us')
        r['throttle_neutral_us'] = cfg.get('throttle_neutral_us')
        r['trained_modes'] = cfg.get('trained_modes')
    c3row = ((c3 or {}).get('runs') or {}).get(ref or name) or {}
    for k in C3_FIELDS:
        r[k] = c3row.get(k)
    same_day = c3row.get('trt_p99_ms')
    r['trt_p99_ms'] = num(same_day)
    r['trt_p99_source'] = 'C3(同じ日)' if same_day is not None else '未(C3)。engine_check の値は trt_p99_ms_engine_check(参考)'
    return r


# ---------------------------------------------------------------- 決まり(numpy なし。test_eval_v2.py で試す)

def mean(xs):
    return sum(xs) / len(xs)


def vs5(vals_best, vals_last, five_best, five_last, thr_vs5, key_best='l1_best', key_last='l1_last'):
    """judgement.vs5 → {'verdict': 良い・悪い・同等・未, ...}。vals_* = {種: 値}、five_* = 5 本の値の list"""
    sb = [v for v in vals_best.values() if v is not None]
    sl = [v for v in vals_last.values() if v is not None]
    n = len(sb)
    if n == 0 or ('n%d' % n) not in thr_vs5:
        return {'verdict': '未', 'n': n, 'why': '種が無いか、n = %d の閾値が無い' % n}
    fb, fl = mean(five_best), mean(five_last)
    tb = thr_vs5['n%d' % n][key_best]
    db = mean(sb) - fb
    good_b = db <= -tb and all(v < fb for v in sb)
    bad_b = db >= tb and all(v > fb for v in sb)
    out = {'n': n, 'mean_best': mean(sb), 'five_mean_best': fb, 'diff_best': db, 'thr_best': tb}
    if len(sl) == n:
        tl = thr_vs5['n%d' % n][key_last]
        dl = mean(sl) - fl
        good_l = dl <= -tl and all(v < fl for v in sl)
        bad_l = dl >= tl and all(v > fl for v in sl)
        out.update({'mean_last': mean(sl), 'five_mean_last': fl, 'diff_last': dl, 'thr_last': tl})
        if bad_b or bad_l:
            out['verdict'] = '悪い'
        elif good_b and good_l:
            out['verdict'] = '良い'
        else:
            out['verdict'] = '同等'
    else:
        out['verdict'] = '悪い' if bad_b else '未'
        out['why'] = 'last が全部の種にそろっていない'
    return out


def pairs(arm_best, base_best, arm_last, base_last, thr_pairs, key_best='l1_best', key_last='l1_last', last_only=False,
          not_worse_us=0.5):
    """judgement.pairs → {'verdict': 良い・悪い・同等・未, 'not_worse': bool or None, ...}。値は {種: 値}"""
    seeds = sorted(s for s in arm_best if s in base_best and arm_best[s] is not None and base_best[s] is not None)
    k = len(seeds)
    if k < 2 or ('k%d' % k) not in thr_pairs:
        return {'verdict': '未', 'not_worse': None, 'k': k, 'why': '同じ種の組が %d(2 組以上で決める)' % k}
    db = [arm_best[s] - base_best[s] for s in seeds]
    tb, tl = thr_pairs['k%d' % k][key_best], thr_pairs['k%d' % k][key_last]
    out = {'k': k, 'seeds': seeds, 'd_best': db, 'mean_d_best': mean(db), 'thr_best': tb}
    have_last = all(arm_last.get(s) is not None and base_last.get(s) is not None for s in seeds)
    if not have_last:
        out.update({'verdict': '未', 'not_worse': None, 'why': 'last がそろっていない'})
        return out
    dl = [arm_last[s] - base_last[s] for s in seeds]
    out.update({'d_last': dl, 'mean_d_last': mean(dl), 'thr_last': tl})
    out['not_worse'] = mean(db) <= not_worse_us and mean(dl) <= not_worse_us
    good = mean(db) <= -tb and mean(dl) <= -tl and all(d < 0 for d in db) and all(d < 0 for d in dl)
    bad_b = mean(db) >= tb and all(d > 0 for d in db)
    bad_l = mean(dl) >= tl and all(d > 0 for d in dl)
    bad = bad_l if last_only else (bad_b or bad_l)
    out['verdict'] = '悪い' if bad else ('良い' if good else '同等')
    return out


def diff_mean(arm, base):
    seeds = sorted(s for s in arm if s in base and arm[s] is not None and base[s] is not None)
    if not seeds:
        return None, 0
    return mean([arm[s] - base[s] for s in seeds]), len(seeds)


def vetoes(arm_rows, five_rows, T, v2h_rows=None):
    """vetoes(interv・bias)→ {'interv': bool or None, 'bias': bool or None, 'any': bool or None, ...}"""
    n = len(arm_rows)
    if n == 0 or ('n%d' % n) not in T['vs5']:
        return {'interv': None, 'bias': None, 'any': None, 'why': '種が無い'}
    iv = [r['l1_interv_us'] for r in arm_rows.values()]
    fiv = mean([r['l1_interv_us'] for r in five_rows])
    t_iv = T['vs5']['n%d' % n]['interv_best']
    v_iv = (mean(iv) - fiv > t_iv) and all(x > fiv for x in iv)
    out = {'interv': v_iv, 'interv_diff': mean(iv) - fiv, 'interv_thr': t_iv}
    if v2h_rows is not None:
        dm, k = diff_mean({s: r['steer_bias_us'] for s, r in arm_rows.items()},
                          {s: r['steer_bias_us'] for s, r in v2h_rows.items()})
        if dm is None or ('k%d' % k) not in T['pairs']:
            out.update({'bias': None, 'bias_why': '同じ種の v2-H が足りない'})
        else:
            t = T['pairs']['k%d' % k]['steer_bias_best']
            out.update({'bias': abs(dm) > t, 'bias_value': dm, 'bias_thr': t, 'bias_rule': '同じ種の v2-H との差の平均(k = %d)' % k})
    else:
        sb = mean([r['steer_bias_us'] for r in arm_rows.values()])
        fsb = mean([r['steer_bias_us'] for r in five_rows])
        t = T['vs5']['n%d' % n]['steer_bias_best']
        out.update({'bias': abs(sb - fsb) > t, 'bias_value': sb - fsb, 'bias_thr': t, 'bias_rule': '5 本の平均からの差(n = %d)' % n})
    out['any'] = None if out['bias'] is None else bool(out['interv'] or out['bias'])
    return out


def genre_skew(r60, r30, thr):
    """marks.genre_skew(60 と同じ種の 30、last)"""
    a = [r60.get('l1_genre4_last_us'), r30.get('l1_genre4_last_us'), r60.get('l1_genre5_last_us'), r30.get('l1_genre5_last_us')]
    if any(v is None for v in a):
        return None
    d4, d5 = a[0] - a[1], a[2] - a[3]
    t4, t5 = thr['genre4_last'], thr['genre5_last']
    return bool((d4 > t4 and d5 <= t5) or (d5 > t5 and d4 <= t4))


def gate(v):
    """True / False / None(未)"""
    return None if v is None else bool(v)


def gates_v2(r, b, desk, five_thr_bias_plus):
    g = {}
    g['engine_ok'] = gate(r.get('engine_ok'))
    g['sanity_ok'] = gate(r.get('sanity_ok'))
    bp99 = num((desk or {}).get('b_trt_p99_ms_same_day'))
    p99 = num(r.get('trt_p99_ms'))
    g['inference'] = None if (bp99 is None or p99 is None) else p99 <= INFER_FACTOR * bp99
    amp = num(r.get('steer_amp_ratio'))
    g['amp'] = None if amp is None else AMP[0] <= amp <= AMP[1]
    g['l1_vs_b'] = abs(r['l1_best_us'] - b['l1_best_us']) < L1_VS_B_US
    bf = num(r.get('brake_lowspeed_frac'))
    g['brake'] = None if bf is None else bf <= BRAKE_FRAC
    g['standstill'] = standstill_gate(r, b, desk)
    g['human_throttle'] = r['thr_bias_us'] <= five_thr_bias_plus
    st = num(r.get('straight_thr_vs_b_us'))
    g['straight'] = None if st is None else abs(st) <= STRAIGHT_US
    return g


def standstill_gate(r, b, desk):
    n = num((desk or {}).get('standstill_frames'))
    a, bb = num(r.get('standstill_go_frac')), num(b.get('standstill_go_frac'))
    if n is None or a is None or bb is None:
        return None
    if n < STANDSTILL_MIN_FRAMES:
        return '参考'
    return a >= bb - 0.10


def gates_bfast(r, b, desk):
    g = {}
    g['G-a'] = r['l1_best_us'] <= b['l1_best_us'] + GA_PLUS_US
    gb = num(r.get('bfast_gb_fwd_vs_b_us'))
    g['G-b'] = None if gb is None else gb >= GB_MIN_US
    gc = num(r.get('bfast_gc_fwd_vs_b_us'))
    g['G-c'] = None if gc is None else gc <= GC_MAX_US
    g['G-d'] = standstill_gate(r, b, desk)
    g['G-e'] = gate(r.get('engine_ok'))
    lim, sf = num((desk or {}).get('bfast_gf_limit_us')), num(r.get('straight_fwd_vs_b_us'))
    g['G-f'] = None if (lim is None or sf is None) else sf < lim
    g['sanity_ok'] = gate(r.get('sanity_ok'))
    bp99, p99 = num((desk or {}).get('b_trt_p99_ms_same_day')), num(r.get('trt_p99_ms'))
    g['inference'] = None if (bp99 is None or p99 is None) else p99 <= INFER_FACTOR * bp99
    amp = num(r.get('steer_amp_ratio'))
    g['amp'] = None if amp is None else AMP[0] <= amp <= AMP[1]
    bf = num(r.get('brake_lowspeed_frac'))
    g['brake'] = None if bf is None else bf <= BRAKE_FRAC
    return g


def gates_pass(g):
    """全部 True(「参考」は数えない)→ True、どれか False → False、ほかは None(未)"""
    vals = [v for v in g.values() if v != '参考']
    if any(v is False for v in vals):
        return False
    if all(v is True for v in vals):
        return True
    return None


# ---------------------------------------------------------------- 判定の全体

def judge(rows, five_rows_by_id, proto, desk, dec_in, src_label='decisions.json'):
    T = proto['thresholds_us']['values']
    five = list(five_rows_by_id.values())
    b = five_rows_by_id['B']
    five_ckpt = {'best': [r['l1_best_us'] for r in five], 'last': [r['l1_last_us'] for r in five]}
    arms = {}
    for r in rows:
        if r['arm']:
            arms.setdefault(r['arm'], {})[r['seed']] = r
    thr_bias_plus = T['five_mean']['thr_bias_best'] + T['pairs']['k2']['thr_bias_best']
    out = {'arms': {}, 'pairs': {}, 'decisions': {}, 'gates': {}, 'notes': []}

    def arm_vs5(arm):
        rs = arms.get(arm) or {}
        v = vs5({s: r['l1_best_us'] for s, r in rs.items()}, {s: r['l1_last_us'] for s, r in rs.items()},
                five_ckpt['best'], five_ckpt['last'], T['vs5'])
        v2h_arm = arm.replace('v2m', 'v2h', 1) if arm.startswith('v2m') else None
        vt = vetoes(rs, five, T, v2h_rows=arms.get(v2h_arm) if v2h_arm and v2h_arm in arms else None) if rs else \
            {'interv': None, 'bias': None, 'any': None}
        if v2h_arm and v2h_arm not in arms and rs:
            vt['bias_note'] = '同じ種の %s が無いので 5 本からの差で見た' % v2h_arm
        return {'seeds': sorted(rs), 'vs5': v, 'veto': vt,
                'too_long': sum(1 for r in rs.values() if r.get('mark_too_long')),
                'rise': sum(1 for r in rs.values() if r.get('mark_rise'))}

    for arm in sorted(arms):
        out['arms'][arm] = arm_vs5(arm)

    def vals(arm, key):
        return {s: r.get(key) for s, r in (arms.get(arm) or {}).items()}

    def pair(name, a, base, last_only=False, base_rows=None, keys=('l1_best_us', 'l1_last_us'), tk=('l1_best', 'l1_last')):
        brows = base_rows if base_rows is not None else (arms.get(base) or {})
        p = pairs(vals(a, keys[0]), {s: r.get(keys[0]) for s, r in brows.items()},
                  vals(a, keys[1]), {s: r.get(keys[1]) for s, r in brows.items()}, T['pairs'],
                  key_best=tk[0], key_last=tk[1], last_only=last_only)
        p.update({'arm': a, 'base': base})
        out['pairs'][name] = p
        return p

    # ---- D1(データ)
    d1 = {'rule': proto['decisions']['D1']['rule']}
    h = out['arms'].get('v2h-e30')
    if not h or h['vs5']['verdict'] == '未' or h['veto']['any'] is None:
        d1.update({'x': None, 'why': 'v2h-e30 がそろっていない(段 1)'})
    elif h['vs5']['verdict'] == '悪い' or h['veto']['any']:
        d1.update({'x': 'base', 'why': 'v2h-e30 が 5 本に対して「%s」・拒否権 %s' % (h['vs5']['verdict'], h['veto']['any'])})
    else:
        m = out['arms'].get('v2m-e30')
        ph = pair('D1: v2m-e30 対 v2h-e30(H1)', 'v2m-e30', 'v2h-e30')
        pi = pair('D1: v2m-e30 対 v2h-e30(介入)', 'v2m-e30', 'v2h-e30', keys=('l1_interv_us', 'l1_interv_last_us'),
                  tk=('interv_best', 'interv_last'))
        sb, k = diff_mean(vals('v2m-e30', 'steer_bias_us'), vals('v2h-e30', 'steer_bias_us'))
        iv, _ = diff_mean(vals('v2m-e30', 'l1_interv_us'), vals('v2h-e30', 'l1_interv_us'))
        tb, _ = diff_mean(vals('v2m-e30', 'thr_bias_us'), vals('v2h-e30', 'thr_bias_us'))
        if not m or ph['verdict'] == '未' or k < 2 or ('k%d' % k) not in T['pairs']:
            d1.update({'x': 'v2h', 'why': 'v2-H を軸(v2m-e30 の組が足りないので v2-M は見ない)'})
        else:
            tk = T['pairs']['k%d' % k]
            cond = {'H1 が悪くない': bool(ph['not_worse']),
                    'steer_bias の差の平均が ±%.3f の中' % tk['steer_bias_best']: abs(sb) <= tk['steer_bias_best'],
                    'l1_interv の差の平均 ≤ +%.3f' % tk['interv_best']: iv <= tk['interv_best'],
                    'thr_bias の差の平均 ≥ −%.3f' % tk['thr_bias_best']: tb >= -tk['thr_bias_best'],
                    'H1 か介入で「良い」': ph['verdict'] == '良い' or pi['verdict'] == '良い',
                    'v2-M に拒否権が無い': m['veto']['any'] is False}
            allok = all(cond.values())
            d1.update({'x': 'v2m' if allok else 'v2h', 'v2m_conditions': cond,
                       'values': {'steer_bias_diff': sb, 'interv_diff': iv, 'thr_bias_diff': tb, 'k': k},
                       'why': 'v2-M が全部を満たした' if allok else 'v2-H を軸(v2-M は %s が欠けた)' % '・'.join(
                           c for c, v in cond.items() if not v)})
    out['decisions']['D1'] = d1
    x = dec_in.get('x') or d1.get('x')
    out['notes'].append('D2 以降の x = %s(%s)' % (x, src_label if dec_in.get('x') else 'この評価の D1'))

    # ---- D2(エポック)
    d2 = {'rule': proto['decisions']['D2']['rule']}
    if x is None:
        d2.update({'E': None, 'why': 'D1 がまだ'})
    else:
        a60, a30 = '%s-e60' % x, '%s-e30' % x
        base_rows = None
        if x == 'base':
            base_rows = {r['seed']: r for r in five if r['seed'] is not None}
        p = pair('D2: %s 対 %s(last)' % (a60, a30), a60, a30, last_only=True, base_rows=base_rows)
        r60 = arms.get(a60) or {}
        r30 = base_rows if base_rows is not None else (arms.get(a30) or {})
        skews = {s: genre_skew(r60[s], r30[s], proto['marks']['genre_skew_threshold_us']) for s in r60 if s in r30}
        tl = sum(1 for r in r60.values() if r.get('mark_too_long'))
        vt = out['arms'].get(a60, {}).get('veto', {})
        if p['verdict'] == '未' or len(r60) < 2:
            d2.update({'E': None, 'why': '%s がそろっていない(段 2)' % a60})
        else:
            reasons = []
            if p['verdict'] == '悪い':
                reasons.append('60 が「悪い」')
            if tl >= 2:
                reasons.append('「長すぎ」が %d 本' % tl)
            if sum(1 for v in skews.values() if v) >= 2:
                reasons.append('「ジャンルの片寄り」が %d 組' % sum(1 for v in skews.values() if v))
            if vt.get('any'):
                reasons.append('60 の腕に拒否権')
            d2.update({'E': 30 if reasons else 60, 'why': '・'.join(reasons) if reasons else '30 に戻す理由が無い',
                       'too_long': tl, 'genre_skew': skews})
    out['decisions']['D2'] = d2
    E = dec_in.get('E') or d2.get('E')

    # ---- D3(拡張)
    d3 = {'rule': proto['decisions']['D3']['rule']}
    if x is None or E is None:
        d3.update({'aug': None, 'why': 'D1・D2 がまだ'})
    else:
        aa, an = '%s-aug2-e%d' % (x, int(E)), '%s-e%d' % (x, int(E))
        base_rows = {r['seed']: r for r in five if r['seed'] is not None} if (x == 'base' and int(E) == 30) else None
        p = pair('D3: %s 対 %s' % (aa, an), aa, an, base_rows=base_rows)
        ra = arms.get(aa) or {}
        rn = base_rows if base_rows is not None else (arms.get(an) or {})
        seeds = sorted(s for s in ra if s in rn)
        rob_a = [num(ra[s].get('robust_dl1_us')) for s in seeds]
        rob_n = [num(rn[s].get('robust_dl1_us')) for s in seeds]
        dd, _ = diff_mean({s: ra[s]['l1_day_us'] for s in seeds}, {s: rn[s]['l1_day_us'] for s in seeds})
        dn, _ = diff_mean({s: ra[s]['l1_night_us'] for s in seeds}, {s: rn[s]['l1_night_us'] for s in seeds})
        vt = out['arms'].get(aa, {}).get('veto', {})
        if p['verdict'] == '未' or len(seeds) < 2:
            d3.update({'aug': None, 'why': '%s がそろっていない(段 3)' % aa})
        elif any(v is None for v in rob_a + rob_n):
            d3.update({'aug': None, 'why': 'robust_dl1_us(C3 のずれの試し)がまだ'})
        else:
            cond = {'H1 が悪くない': bool(p['not_worse']),
                    'robust_dl1_us が全部の組で AUG2 < なし': all(a < n for a, n in zip(rob_a, rob_n)),
                    '平均(AUG2) ÷ 平均(なし) ≤ 0.8': mean(rob_n) > 0 and mean(rob_a) / mean(rob_n) <= 0.8,
                    'l1_day_us の差の平均 ≤ +1.0': dd <= 1.0, 'l1_night_us の差の平均 ≤ +1.0': dn <= 1.0,
                    'AUG2 の腕に拒否権が無い': vt.get('any') is False}
            d3.update({'aug': 'AUG2' if all(cond.values()) else 'なし', 'conditions': cond,
                       'why': '全部を満たした' if all(cond.values()) else '%s が欠けた' % '・'.join(c for c, v in cond.items() if not v)})
    out['decisions']['D3'] = d3
    aug = dec_in.get('aug') or d3.get('aug')

    # ---- D5a(v2 の候補の並び)と門
    final = None
    if x is not None and E is not None and aug is not None:
        final = '%s%s-e%d' % (x, '-aug2' if aug == 'AUG2' else '', int(E))
    d5 = {'rule': proto['decisions']['D5a']['rule'], 'final_arm': final, 'C1': None, 'C2': None}
    if final is None:
        d5['why'] = 'D1・D2・D3 がまだ'
    else:
        rs = arms.get(final) or {}
        vt = out['arms'].get(final, {}).get('veto', {})
        picks, undecided = [], []
        for s in (0, 1, 2, 3):
            if s not in rs:
                continue
            g = gates_v2(rs[s], b, desk, thr_bias_plus)
            gp = gates_pass(g)
            out['gates'][rs[s]['run']] = {'gates': g, 'pass': gp}
            if gp is True:
                picks.append(rs[s]['run'])
            elif gp is None:
                undecided.append(rs[s]['run'])
        if vt.get('any'):
            d5['why'] = '最終の腕に拒否権があるので C1・C2 は無し'
        else:
            d5['C1'] = picks[0] if picks else None
            d5['C2'] = picks[1] if len(picks) > 1 else None
            d5['why'] = '種 0 → 3 の順で門に全部合格した最初・次' + (
                '(門が「未」の種: %s)' % ' '.join(undecided) if undecided else '')
    out['decisions']['D5a'] = d5

    # ---- D4(2 本の平均。予備)は予測の平均が要るので judge_main で足す
    # ---- BF(B-FAST)
    bf = {'rule': proto['decisions']['BF']['rule'], 'BF1': None, 'BF2': None}
    rs = arms.get('bfast-e15') or {}
    vt = out['arms'].get('bfast-e15', {}).get('veto', {})
    picks, undecided = [], []
    for s in (0, 1, 2):
        if s not in rs:
            continue
        g = gates_bfast(rs[s], b, desk)
        gp = gates_pass(g)
        out['gates'][rs[s]['run']] = {'gates': g, 'pass': gp}
        if gp is True:
            picks.append(rs[s]['run'])
        elif gp is None:
            undecided.append(rs[s]['run'])
    if not rs:
        bf['why'] = 'B-FAST がまだ'
    elif vt.get('any'):
        bf['why'] = 'B-FAST の腕に拒否権があるので無し'
    else:
        bf['BF1'] = picks[0] if picks else None
        bf['BF2'] = picks[1] if len(picks) > 1 else None
        bf['why'] = '種 0 → 2 の順で門に全部合格した最初・次' + ('(門が「未」の種: %s)' % ' '.join(undecided) if undecided else '')
    out['decisions']['BF'] = bf
    out['proposed'] = {'D1': {'x': d1.get('x')}, 'D2': {'E': d2.get('E')}, 'D3': {'aug': d3.get('aug')}}
    return out


def d4_ensemble(rows_by_name, preds, S, T, x, E, aug):
    """D4: 軽量型 種 k + 多段型 種 k の予測の平均(k = 0・1)と、同じ種の軽量型単体(best だけ)"""
    if x is None or E is None or aug is None:
        return {'reserve': None, 'why': 'D1・D2・D3 がまだ'}
    light = 'v2_20261025-%s%s-e%d-s%%d' % (x, '-aug2' if aug == 'AUG2' else '', int(E))
    msf = 'v2_20261025-%s-msf%s-e%d-s%%d' % (x, '-aug2' if aug == 'AUG2' else '', int(E))
    d, have = [], []
    for k in (0, 1):
        a, m = light % k, msf % k
        if a in preds and m in preds:
            ens = (preds[a]['best'] + preds[m]['best']) / 2.0
            d.append(metrics(ens, S)['l1'] - rows_by_name[a]['l1_best_us'])
            have.append(k)
    if len(d) < 2:
        return {'reserve': None, 'why': '多段型と軽量型の組が足りない(段 4)', 'seeds': have}
    thr = T['pairs']['k2']['l1_best']
    ok = mean(d) <= -thr and all(v < 0 for v in d)
    return {'reserve': None if ok else False, 'mean_d_best': mean(d), 'd_best': d, 'thr': thr,
            'why': ('L1 は「良い」。門(推論を除く)は 2 本の平均の C3 の欄が要るので未' if ok else 'L1 が「良い」でない → 予備にしない'),
            'note': '予備でも主の候補にしない。11/3 の 1 枠に入れない(protocol decisions.D4)'}


# ---------------------------------------------------------------- 表

def fmt(v, nd=2):
    if v is None:
        return '未'
    if v == '参考':
        return '参考'
    if isinstance(v, (bool, np.bool_)):
        return '○' if v else '×'
    if isinstance(v, float):
        return ('%%.%df' % nd) % v
    return str(v)


def report_md(rows, five_rows, J, d4, meta):
    L = ['# 机上の評価 v2%s' % ('(空回し: 全部の枠をモデル B で埋めた)' if meta['dry_run'] else ''), '',
         '- 作った時刻: %s' % meta['made_at'],
         '- protocol_v2.json の sha256: `%s`' % meta['protocol_sha256'],
         '- 予測の表: H1 %d フレーム・介入 %d フレーム(desk_inputs_20261022.npz)' % (meta['h1'], meta['interv']),
         '- C3 の欄(desk_v2.json): %s' % (('あり(%s)' % meta['desk_path']) if meta['desk'] else '**まだ無い**(その欄を使う門・決まりは「未」)'),
         '- decisions.json の答え: %s' % (json.dumps(meta['dec_in'], ensure_ascii=False) if any(meta['dec_in'].values()) else 'まだ無い'),
         '- 机上の順位で実車の順位は決めない(plan.md 6.4)。', '']
    if meta['dry_run']:
        L += ['> 空回しです。22 本の枠の予測は全部モデル B の保存した予測(best・last)、エンジンの確認は B の engine_check.json です。',
              '> 結果は決まりの道筋が動くかの確かめで、判定ではありません。', '']
    L += ['## 1. 学習ごとの欄(H1・介入。単位 µs)', '',
          '| 名前 | L1 best | L1 last | ステア | スロットル | 介入 | 昼 | 夜 | ジャンル 4 | ジャンル 5 | ステアの偏り | スロットルの偏り | 振れ幅 | R1 手前 | 区間の外 | エンジン | 長すぎ |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in list(five_rows.values()) + rows:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['run'] if r.get('arm') else '5 本: %s' % r['rid'], fmt(r['l1_best_us']), fmt(r['l1_last_us']),
            fmt(r['l1_steer_best_us']), fmt(r['l1_thr_best_us']), fmt(r['l1_interv_us']), fmt(r['l1_day_us']),
            fmt(r['l1_night_us']), fmt(r['l1_genre4_us']), fmt(r['l1_genre5_us']), fmt(r['steer_bias_us']),
            fmt(r['thr_bias_us']), fmt(r['steer_amp_ratio'], 3), fmt(r['r1_app_thr_diff_us']), fmt(r['nonzone_thr_diff_us']),
            fmt(r.get('engine_ok')), fmt(r.get('mark_too_long'))))
    L += ['', '## 2. 腕 対 5 本(H1 の L1)と拒否権', '',
          '| 腕 | 種 | best の差 | 閾値 | last の差 | 閾値 | 判定 | 介入の差 | 拒否権 介入 | 偏り | 拒否権 偏り | 長すぎ |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, a in J['arms'].items():
        v, vt = a['vs5'], a['veto']
        L.append('| %s | %s | %s | %s | %s | %s | **%s** | %s | %s | %s | %s | %d 本 |' % (
            arm, '・'.join(str(s) for s in a['seeds']), fmt(v.get('diff_best')), fmt(v.get('thr_best'), 3),
            fmt(v.get('diff_last')), fmt(v.get('thr_last'), 3), v['verdict'], fmt(vt.get('interv_diff')),
            fmt(vt.get('interv')), fmt(vt.get('bias_value')), fmt(vt.get('bias')), a['too_long']))
    if J['pairs']:
        L += ['', '## 3. 同じ種の比べ', '', '| 比べ | 組 | best の差の平均 | 閾値 | last の差の平均 | 閾値 | 悪くない | 判定 |',
              '|---|---|---|---|---|---|---|---|']
        for name, p in J['pairs'].items():
            L.append('| %s | %s | %s | %s | %s | %s | %s | **%s** |' % (
                name, p.get('k'), fmt(p.get('mean_d_best')), fmt(p.get('thr_best'), 3), fmt(p.get('mean_d_last')),
                fmt(p.get('thr_last'), 3), fmt(p.get('not_worse')), p['verdict']))
    D = J['decisions']
    L += ['', '## 4. 決まりの答え(案。decisions.json に写すのは main session)', '', '| 決まり | 答え | 理由 |', '|---|---|---|',
          '| D1 データ | %s | %s |' % (fmt(D['D1'].get('x')), D['D1'].get('why')),
          '| D2 エポック | %s | %s |' % (fmt(D['D2'].get('E')), D['D2'].get('why')),
          '| D3 拡張 | %s | %s |' % (fmt(D['D3'].get('aug')), D['D3'].get('why')),
          '| D4 2 本の平均 | %s | %s |' % (fmt(d4.get('reserve')), d4.get('why')),
          '| D5a v2 の候補 | C1 %s・C2 %s | 最終の腕 %s。%s |' % (fmt(D['D5a'].get('C1')), fmt(D['D5a'].get('C2')),
                                                          fmt(D['D5a'].get('final_arm')), D['D5a'].get('why')),
          '| BF B-FAST の候補 | BF1 %s・BF2 %s | %s |' % (fmt(D['BF'].get('BF1')), fmt(D['BF'].get('BF2')), D['BF'].get('why'))]
    if J['gates']:
        L += ['', '## 5. 門(○ 合格・× 不合格・未 = 欄がまだ)', '']
        for name, g in J['gates'].items():
            L.append('- **%s**: %s → %s' % (name, '・'.join('%s %s' % (k, fmt(v)) for k, v in g['gates'].items()),
                                            {True: '全部合格', False: '不合格', None: '未'}[g['pass']]))
    L += ['', '## 6. 注', ''] + ['- %s' % n for n in J['notes']]
    L += ['- 閾値・決まりは protocol_v2.json(thresholds_us・judgement・vetoes・marks・decisions・gates_v2・gates_bfast)のまま。',
          '- 5 本の平均は保存した予測から同じフレームで計算した値(丸めない)。',
          '- 門「推論」は同じ日に測った B の p99 の 1.3 倍(C3)。engine_check の p99 は参考で、門に使わない。',
          '- C3 の欄が無いあいだは、門は「未」のまま(合格にしない)。']
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- コマンド

def cmd_judge(args):
    proto = protocol()
    S = load_bundle(proto)
    five = five_preds(proto)
    desk_path = Path(args.desk) if args.desk else DESK_C3
    desk = load_json(desk_path) if desk_path.exists() else None
    dec_raw = load_json(DECISIONS, {}) or {}
    dec_in = {'x': (dec_raw.get('D1') or {}).get('x'), 'E': (dec_raw.get('D2') or {}).get('E'),
              'aug': (dec_raw.get('D3') or {}).get('aug')}
    five_rows = {}
    for rid, f in five.items():
        r = run_row(f['run'], f['best'], f['last'], S, desk, run_dir=RUNS / f['run'] if (RUNS / f['run']).exists() else None)
        r.update({'rid': rid, 'seed': f['seed'], 'arm': None, 'family': 'ref'})
        five_rows[rid] = r
    rows, preds = [], {}
    if args.dry_run_b:
        out_dir = EVAL / 'dryrun_B'
        dec_in = {'x': None, 'E': None, 'aug': None}
        for kv in args.assume or []:          # 空回しだけ: 答えを仮に置いて、あとの決まり(D4・D5a・門)の道筋も通す
            k, v = kv.split('=', 1)
            if k not in dec_in:
                raise SystemExit('--assume は x・E・aug だけ: %s' % kv)
            dec_in[k] = int(v) if k == 'E' else v
        bb, bl = five['B']['best'], five['B']['last']
        src = 'data/zoo の保存'
        if pred_file(B_RUN, 'best').exists() and pred_file(B_RUN, 'last').exists():
            bb, bl = load_pred(pred_file(B_RUN, 'best'), 'best'), load_pred(pred_file(B_RUN, 'last'), 'last')
            src = 'eval/pred(この台本の predict で予測し直したもの)'
        for name in slot_names():
            preds[name] = {'best': bb, 'last': bl}
            rows.append(run_row(name, bb, bl, S, desk, run_dir=RUNS / B_RUN, ref=B_RUN))
        notes = ['空回し: 22 本の枠の予測 = B(%s)' % src]
        if args.assume:
            notes.append('空回しで仮に置いた答え: %s(本当の答えではない)' % ' '.join(args.assume))
    else:
        out_dir = EVAL
        pats = protocol_patterns(proto)
        names = sorted({p.name[:-4] for p in PRED.glob('*.npz') if not p.name.endswith('_last.npz')} if PRED.exists() else set())
        names = [n for n in names if any(pt.match(n) for pt in pats)]
        for name in names:
            best = load_pred(pred_file(name, 'best'), 'best')
            last = load_pred(pred_file(name, 'last'), 'last') if pred_file(name, 'last').exists() else None
            preds[name] = {'best': best, 'last': last}
            rows.append(run_row(name, best, last, S, desk, run_dir=RUNS / name))
        notes = ['予測のある学習: %d 本' % len(rows)]
    if args.desk and not args.dry_run_b:
        raise SystemExit('--desk は空回し(--dry-run-b)だけ')
    J = judge(rows, five_rows, proto, desk, dec_in, src_label='仮に置いた答え' if args.assume else 'decisions.json')
    J['notes'] = notes + J['notes']
    x = dec_in.get('x') or J['decisions']['D1'].get('x')
    E = dec_in.get('E') or J['decisions']['D2'].get('E')
    aug = dec_in.get('aug') or J['decisions']['D3'].get('aug')
    d4 = d4_ensemble({r['run']: r for r in rows}, preds, S, proto['thresholds_us']['values'], x, E, aug)
    J['decisions']['D4'] = d4
    meta = {'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'protocol_sha256': sha256(PROTOCOL), 'dry_run': bool(args.dry_run_b),
            'h1': int(S['primary'].sum()), 'interv': int(S['interv'].sum()), 'desk': desk is not None, 'dec_in': dec_in,
            'desk_path': str(desk_path),
            'script_sha256': sha256(Path(__file__)), 'lock2': LOCK2.exists()}
    clean = lambda r: {k: rnd(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in r.items()}  # noqa: E731
    write_json(out_dir / 'eval_v2.json', {'meta': meta, 'five': {k: clean(v) for k, v in five_rows.items()},
                                          'runs': [clean(r) for r in rows], 'judge': J})
    write_text(out_dir / 'eval_v2.md', report_md(rows, five_rows, J, d4, meta))
    write_json(out_dir / 'decisions_proposed.json', {'made_at': meta['made_at'], 'dry_run': meta['dry_run'],
                                                     'note': 'eval_v2.py の案。キューが読むのは decisions.json(main session が写す)',
                                                     **J['proposed'],
                                                     'why': {k: J['decisions'][k].get('why') for k in ('D1', 'D2', 'D3')}})
    log('書きました: %s(学習 %d 本)' % (out_dir / 'eval_v2.md', len(rows)))
    D = J['decisions']
    log('D1 %s / D2 %s / D3 %s / D4 %s / C1 %s・C2 %s / BF1 %s・BF2 %s' % (
        D['D1'].get('x'), D['D2'].get('E'), D['D3'].get('aug'), d4.get('reserve'), D['D5a'].get('C1'), D['D5a'].get('C2'),
        D['BF'].get('BF1'), D['BF'].get('BF2')))
    return 0


def cmd_predict(args):
    proto = protocol()
    refs = {r[1] for r in FIVE}
    names = list(args.runs or [])
    if args.b_crosscheck and B_RUN not in names:
        names.insert(0, B_RUN)
    if not names:
        raise SystemExit('--runs か --b-crosscheck を付けてください')
    new = [n for n in names if n not in refs]
    if new and not LOCK2.exists() and not args.allow_before_lock2:
        raise SystemExit('protocol_v2_lock2.json がまだ無いので、v2・B-FAST の予測はしません(protocol lock.second_lock): %s' % ' '.join(new))
    pats = protocol_patterns(proto)
    bad = [n for n in new if not any(p.match(n) for p in pats)]
    if bad:
        raise SystemExit('protocol の runs の名前ではありません: %s' % ' '.join(bad))
    desk_dir = REPO / 'experiments' / '20261021_bench_prep' / 'desk'
    for p in (str(REPO), str(REPO / 'tools'), str(desk_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import desk_gpu as G                  # 10/21 の desk の GPU の手順(predict_one・table)をそのまま使う
    import rl_policies as rp
    tab = G.table()
    sealed = set(proto['seal']['logs'])
    if any(str(s)[:15] in sealed for s in np.unique(tab['scene'])):
        raise SystemExit('予測の表に封の記録があります(11/1 より前は読まない)')
    G.check_ready()
    PRED.mkdir(parents=True, exist_ok=True)
    for name in names:
        run_dir = RUNS / name
        for ck in ('best', 'last'):
            path = pred_file(name, ck)
            if path.exists() and G.pred_ok(path, ck):
                log('%s(%s): 予測はあります' % (name, ck))
                continue
            if not (run_dir / ('%s.pth' % ck)).exists():
                log('%s: %s.pth がありません(飛ばす)' % (name, ck))
                continue
            t0 = time.time()
            res, genres, sub, meta = G.predict_one(run_dir, ck, tab)
            pred = res.pop('actions')
            extra = {'out_' + k: v for k, v in res.items()}
            rp._save_npz(path, pred=pred, genre=genres.astype(np.int8), substituted=sub, **extra, **meta)
            ok = G.pred_ok(path, ck)
            log('%s(%s): %d フレーム %.0f 秒・読み直し %s' % (name, ck, len(pred), time.time() - t0, 'OK' if ok else 'NG'))
            if not ok:
                raise SystemExit('%s の予測を読み直せません' % path)
    if args.b_crosscheck:
        out = {'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': B_RUN, 'limit_us': 0.01, 'what': '段 0 の確かめ 1'}
        okall = True
        for ck, saved in (('best', ZOO / 'B.npz'), ('last', ZOO / 'B_last.npz')):
            with np.load(pred_file(B_RUN, ck)) as a, np.load(saved) as b:
                mx = float(np.abs(a['pred'].astype(np.float64) - b['pred'].astype(np.float64)).max())
                same_genre = bool(np.array_equal(a['genre'], b['genre']))
                same_key = str(a['key']) == str(b['key']) == TABLE_KEY
            ok = same_key and same_genre and mx <= 0.01
            okall &= ok
            out[ck] = {'saved': str(saved), 'max_abs_us': round(mx, 6), 'same_genre': same_genre, 'same_key': same_key, 'pass': ok}
        out['pass'] = okall
        write_json(EVAL / 'crosscheck_B.json', out)
        log('B の予測し直し: %s' % json.dumps(out, ensure_ascii=False))
        if not okall:
            raise SystemExit('B の予測し直しが 10/12 の保存と合いません(≤ 0.01 µs)')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict', help='H1・介入の予測(GPU。jf-heavy --lock big で)')
    p.add_argument('--runs', nargs='*', help='学習の名前(data/runs の下)')
    p.add_argument('--b-crosscheck', action='store_true', help='B を予測し直して 10/12 の保存と比べる(段 0 の確かめ 1)')
    p.add_argument('--allow-before-lock2', action='store_true', help='lock2 の前でも v2・B-FAST を予測する(deviations.md に書くこと)')
    j = sub.add_parser('judge', help='決まりを当てて表を出す(CPU・numpy)')
    j.add_argument('--dry-run-b', action='store_true', help='空回し: 22 本の枠を全部 B の予測で埋める')
    j.add_argument('--assume', nargs='*', help='空回しだけ: 答えを仮に置く(例 x=v2h E=60 aug=AUG2)')
    j.add_argument('--desk', help='C3 の desk_v2.json の代わり(試しだけ。既定 %s)' % DESK_C3)
    args = ap.parse_args()
    return {'predict': cmd_predict, 'judge': cmd_judge}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
