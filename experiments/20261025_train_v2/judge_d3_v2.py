#!/usr/bin/env python3
"""D3(拡張)の判定(2026-10-26。D3 の係)。numpy だけ(torch・GPU は使わない)。車・装置に触れない。

    /home/jetson/bin/jf-heavy --lock none --mem 2000M --need-mb 1500 -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        /home/jetson/jetformer/experiments/20261025_train_v2/judge_d3_v2.py \
        --desk-aug  /home/jetson/jetformer_reports/train_v2_20261025/desk_v2_stage3.json \
        --desk-none <v2h-e60 の行を読む desk の JSON(別の係の desk/desk_v2.json か desk_v2_d3.json)> \
        --out /home/jetson/jetformer_reports/train_v2_20261025/stage3_judge.json

- 判定 A(eval_v2.py の judge() をそのまま呼ぶ): eval/pred の予測(best・last)と eval_v2.run_row で行を作り、
  C3 の欄は「AUG2 の 4 本 = --desk-aug、ほか = --desk-none」を合わせた desk で渡す。eval_v2.py は変えない
  (sha256 は lock2 の C4 の値と照らす)。eval/ の eval_v2.json・eval_v2.md・decisions_proposed.json は書かない。
- 判定 B(別のコード): 欄(H1 の L1 best・last・介入・steer_bias・昼・夜)は protocol_numbers.py の h1_metrics と
  この台本の昼夜のマスクで、robust_dl1_us は desk/pred の *_shift.npz と eval/pred の保存した best からこの台本で数え直し、
  決まり(judgement.pairs・vetoes・decisions.D3)は protocol_v2.json の文からこの台本で書き直した。
- A と B の欄(差 ≤ 0.001 µs)・条件・答えを照らす。違えば答えを出さない(aug = None)。
出力: --out の JSON だけ。decisions.json は書かない(係が写す)。data/・既定値は変えない。
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_v2 as E  # noqa: E402
import protocol_numbers as PN  # noqa: E402

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
EVAL_PRED = REP / 'eval' / 'pred'
DESK_PRED = REP / 'desk' / 'pred'
SHIFTS = ('img_up3', 'img_down3', 'bright_0.8', 'bright_1.2', 'tof_down1', 'top_0.85', 'top_1.15')
TOL_US = 0.001


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--desk-aug', required=True)
    ap.add_argument('--desk-none', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    for bad in (REP / 'desk' / 'desk_v2.json', REP / 'decisions.json', REP / 'eval' / 'eval_v2.json'):
        if out_path.resolve() == bad.resolve():
            raise SystemExit('その場所には書きません: %s' % bad)

    proto = load_json(REP / 'protocol_v2.json')
    lock2 = load_json(REP / 'protocol_v2_lock2.json')
    dec = load_json(REP / 'decisions.json')
    chk = {'protocol_v2_sha256': sha256(REP / 'protocol_v2.json'), 'protocol_v2_lock2_sha256': sha256(REP / 'protocol_v2_lock2.json'),
           'eval_v2_py_sha256': sha256(HERE / 'eval_v2.py'), 'protocol_numbers_py_sha256': sha256(HERE / 'protocol_numbers.py')}
    want_eval = lock2['files']['C4(キュー v2)']['eval_v2.py']['sha256']
    want_pn = lock2['files']['数え直しの台本']['protocol_numbers.py']['sha256']
    if chk['protocol_v2_sha256'] != dec['protocol_v2_sha256'] or chk['protocol_v2_lock2_sha256'] != dec['protocol_v2_lock2_sha256']:
        raise SystemExit('protocol・lock2 の sha256 が decisions.json と違います')
    if chk['eval_v2_py_sha256'] != want_eval or chk['protocol_numbers_py_sha256'] != want_pn:
        raise SystemExit('eval_v2.py・protocol_numbers.py の sha256 が lock2 と違います')
    T = proto['thresholds_us']['values']
    x, Ep = dec['D1']['x'], int(dec['D2']['E'])
    if dec['D3'].get('aug') is not None:
        raise SystemExit('decisions.json の D3.aug はもう埋まっています: %r' % dec['D3']['aug'])
    arm_a, arm_n = '%s-aug2-e%d' % (x, Ep), '%s-e%d' % (x, Ep)
    names_a = ['v2_20261025-%s-s%d' % (arm_a, s) for s in range(4)]
    names_n = ['v2_20261025-%s-s%d' % (arm_n, s) for s in range(4)]

    desk_a, desk_n = load_json(args.desk_aug), load_json(args.desk_none)
    merged = {k: v for k, v in desk_n.items() if k != 'runs'}
    merged['runs'] = dict(desk_n.get('runs') or {})
    src_rows = {n: str(args.desk_none) for n in merged['runs']}
    for n in names_a:
        if n in (desk_a.get('runs') or {}):
            merged['runs'][n] = desk_a['runs'][n]
            src_rows[n] = str(args.desk_aug)

    # ---------------- 判定 A: eval_v2.judge() をそのまま
    S = E.load_bundle(proto)
    five = E.five_preds(proto)
    five_rows = {}
    for rid, f in five.items():
        r = E.run_row(f['run'], f['best'], f['last'], S, merged, run_dir=E.RUNS / f['run'] if (E.RUNS / f['run']).exists() else None)
        r.update({'rid': rid, 'seed': f['seed'], 'arm': None, 'family': 'ref'})
        five_rows[rid] = r
    pats = E.protocol_patterns(proto)
    names = sorted({p.name[:-4] for p in E.PRED.glob('*.npz') if not p.name.endswith('_last.npz')})
    names = [n for n in names if any(pt.match(n) for pt in pats)]
    rows = []
    for n in names:
        best = E.load_pred(E.pred_file(n, 'best'), 'best')
        last = E.load_pred(E.pred_file(n, 'last'), 'last') if E.pred_file(n, 'last').exists() else None
        rows.append(E.run_row(n, best, last, S, merged, run_dir=E.RUNS / n))
    dec_in = {'x': x, 'E': Ep, 'aug': None}
    J = E.judge(rows, five_rows, proto, merged, dec_in)
    A = {'D3': J['decisions']['D3'], 'pair_h1': J['pairs'].get('D3: %s 対 %s' % (arm_a, arm_n)),
         'arm_aug2': J['arms'].get(arm_a), 'arm_none': J['arms'].get(arm_n),
         'D1_recomputed': J['decisions']['D1'].get('x'), 'D2_recomputed': J['decisions']['D2'].get('E')}
    rowmap = {r['run']: r for r in rows}

    # ---------------- 判定 B: 別のコード
    with np.load(PN.BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != PN.TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        y = np.nan_to_num(np.asarray(z['y'], np.float64))
        prim, interv, scene, frame = z['primary'].astype(bool), z['interv'].astype(bool), z['scene'].astype(str), z['frame']
    genre = PN.h1_genre(scene, frame, prim)
    dayset = set(proto['holdouts']['day_night']['day'])
    nightset = set(proto['holdouts']['day_night']['night'])
    day = prim & np.array([s in dayset for s in scene])
    night = prim & np.array([s in nightset for s in scene])      # protocol の夜 7 シーン(eval_v2 は H1 − 昼)
    if int((day | night).sum()) != int(prim.sum()) or bool((day & night).any()):
        raise SystemExit('昼・夜のシーンが H1 を分けていません')

    def mets(path):
        p = PN.load_pred(path)
        m = PN.h1_metrics(p, y, prim, interv, genre)
        d = np.abs(p - y)
        m['l1_day'] = float(d[day].mean())
        m['l1_night'] = float(d[night].mean())
        return p, m

    def robust(name, best_pred):
        f = DESK_PRED / ('%s_shift.npz' % name)
        if not f.exists():
            return None, None
        with np.load(f, allow_pickle=False) as zz:
            rws = zz['rows']
            if str(zz['key']) != PN.TABLE_KEY:
                raise SystemExit('%s: key が違います' % f)
            yy = y[rws]
            b = float(np.abs(best_pred[rws] - yy).mean())
            per = {k: float(np.abs(zz['pred_' + k].astype(np.float64) - yy).mean()) - b for k in SHIFTS}
        return avg(per.values()), per

    B = {'runs': {}}
    for n in names_a + names_n:
        pb, mb = mets(EVAL_PRED / ('%s.npz' % n))
        _, ml = mets(EVAL_PRED / ('%s_last.npz' % n))
        rb, per = robust(n, pb)
        B['runs'][n] = {'l1_best': mb['l1'], 'l1_last': ml['l1'], 'l1_interv': mb['l1_interv'], 'steer_bias': mb['steer_bias'],
                        'thr_bias': mb['thr_bias'], 'l1_day': mb['l1_day'], 'l1_night': mb['l1_night'],
                        'too_long': ml['l1'] - mb['l1'] > proto_too_long(proto), 'robust_dl1': rb, 'robust_per_shift': per}
    f5 = {}
    for rid, _run, pbp, plp in PN.FIVE:
        _, m = mets(pbp)
        f5[rid] = m
    five_iv = avg(m['l1_interv'] for m in f5.values())
    five_sb = avg(m['steer_bias'] for m in f5.values())
    ra = {s: B['runs'][names_a[s]] for s in range(4)}
    rn = {s: B['runs'][names_n[s]] for s in range(4)}
    k = 4
    tp = T['pairs']['k%d' % k]
    db = [ra[s]['l1_best'] - rn[s]['l1_best'] for s in range(k)]
    dl = [ra[s]['l1_last'] - rn[s]['l1_last'] for s in range(k)]
    nw = proto['thresholds_us']['not_worse_us']
    not_worse = avg(db) <= nw and avg(dl) <= nw
    good = avg(db) <= -tp['l1_best'] and avg(dl) <= -tp['l1_last'] and all(d < 0 for d in db) and all(d < 0 for d in dl)
    bad = (avg(db) >= tp['l1_best'] and all(d > 0 for d in db)) or (avg(dl) >= tp['l1_last'] and all(d > 0 for d in dl))
    tv = T['vs5']['n%d' % k]
    iv = [ra[s]['l1_interv'] for s in range(k)]
    v_iv = (avg(iv) - five_iv > tv['interv_best']) and all(v > five_iv for v in iv)
    sb = avg(ra[s]['steer_bias'] for s in range(k)) - five_sb
    v_sb = abs(sb) > tv['steer_bias_best']
    rob_a = [ra[s]['robust_dl1'] for s in range(k)]
    rob_n = [rn[s]['robust_dl1'] for s in range(k)]
    dday = avg(ra[s]['l1_day'] - rn[s]['l1_day'] for s in range(k))
    dnight = avg(ra[s]['l1_night'] - rn[s]['l1_night'] for s in range(k))
    if any(v is None for v in rob_a + rob_n):
        condB, augB = None, None
    else:
        condB = {'H1 が悪くない': not_worse,
                 'robust_dl1_us が全部の組で AUG2 < なし': all(a < b for a, b in zip(rob_a, rob_n)),
                 '平均(AUG2) ÷ 平均(なし) ≤ 0.8': avg(rob_n) > 0 and avg(rob_a) / avg(rob_n) <= 0.8,
                 'l1_day_us の差の平均 ≤ +1.0': dday <= 1.0, 'l1_night_us の差の平均 ≤ +1.0': dnight <= 1.0,
                 'AUG2 の腕に拒否権が無い': not (v_iv or v_sb)}
        augB = 'AUG2' if all(condB.values()) else 'なし'
    B.update({'pair_h1': {'d_best': db, 'mean_d_best': avg(db), 'd_last': dl, 'mean_d_last': avg(dl), 'thr_best': tp['l1_best'],
                          'thr_last': tp['l1_last'], 'not_worse': not_worse,
                          'verdict': '悪い' if bad else ('良い' if good else '同等')},
              'veto_aug2': {'interv': v_iv, 'interv_diff': avg(iv) - five_iv, 'interv_thr': tv['interv_best'],
                            'bias': v_sb, 'bias_value': sb, 'bias_thr': tv['steer_bias_best'], 'any': bool(v_iv or v_sb),
                            'five_mean_interv': five_iv, 'five_mean_steer_bias': five_sb},
              'robust': {'aug2': rob_a, 'none': rob_n,
                         'mean_aug2': None if None in rob_a else avg(rob_a), 'mean_none': None if None in rob_n else avg(rob_n),
                         'ratio': (avg(rob_a) / avg(rob_n)) if (None not in rob_a + rob_n and avg(rob_n) > 0) else None},
              'day_night': {'mean_d_day': dday, 'mean_d_night': dnight},
              'conditions': condB, 'aug': augB})

    # ---------------- A と B を照らす
    diffs = {}
    for n in names_a + names_n:
        r, b = rowmap.get(n), B['runs'][n]
        for ka, kb in (('l1_best_us', 'l1_best'), ('l1_last_us', 'l1_last'), ('l1_interv_us', 'l1_interv'),
                       ('steer_bias_us', 'steer_bias'), ('l1_day_us', 'l1_day'), ('l1_night_us', 'l1_night'),
                       ('robust_dl1_us', 'robust_dl1')):
            va, vb = (r or {}).get(ka), b.get(kb)
            diffs['%s.%s' % (n, ka)] = None if (va is None or vb is None) else abs(float(va) - float(vb))
    known = [v for v in diffs.values() if v is not None]
    max_diff = max(known) if known else None
    missing = [kk for kk, v in diffs.items() if v is None]
    condA = A['D3'].get('conditions')
    same = (A['D3'].get('aug') == augB and condA == condB and max_diff is not None and max_diff <= TOL_US and not missing)
    answer = A['D3'].get('aug') if same else None
    res = {'schema': 'jf-train-v2-d3-judge/v1', 'made_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'rule': proto['decisions']['D3']['rule'], 'x': x, 'E': Ep, 'arm_aug2': arm_a, 'arm_none': arm_n,
           'inputs': {'desk_aug': str(args.desk_aug), 'desk_none': str(args.desk_none), 'desk_rows_source': {n: src_rows.get(n) for n in names_a + names_n},
                      'desk_aug_made_at': desk_a.get('made_at'), 'desk_none_made_at': desk_n.get('made_at'),
                      'desk_aug_sha256': sha256(args.desk_aug), 'desk_none_sha256': sha256(args.desk_none),
                      'eval_pred_sha256': {n + ck: sha256(EVAL_PRED / ('%s%s.npz' % (n, ck))) for n in names_a + names_n for ck in ('', '_last')},
                      'shift_sha256': {n: sha256(DESK_PRED / ('%s_shift.npz' % n)) for n in names_a + names_n
                                       if (DESK_PRED / ('%s_shift.npz' % n)).exists()}},
           'checks': chk, 'A_eval_v2_judge': A, 'B_recount': B,
           'rowsA': {n: {kk: rowmap.get(n, {}).get(kk) for kk in (
               'l1_best_us', 'l1_last_us', 'l1_interv_us', 'l1_interv_last_us', 'steer_bias_us', 'thr_bias_us', 'l1_day_us',
               'l1_night_us', 'l1_genre4_last_us', 'l1_genre5_last_us', 'robust_dl1_us', 'mark_too_long', 'mark_rise',
               'engine_ok', 'train_best_val_us', 'train_best_epoch', 'epochs_done')} for n in names_a + names_n},
           'crosscheck': {'max_abs_diff_us': max_diff, 'tol_us': TOL_US, 'missing': missing,
                          'same_conditions': condA == condB, 'same_answer': A['D3'].get('aug') == augB, 'pass': same},
           'aug': answer,
           'marks_aug2_ref': {'too_long': (A['arm_aug2'] or {}).get('too_long'), 'rise': (A['arm_aug2'] or {}).get('rise'),
                              'note': 'protocol decisions.D3 の文に印は無い(参考)'}}
    write_json(out_path, res)
    print(json.dumps({'aug': answer, 'A': A['D3'].get('aug'), 'B': augB, 'conditions': condA, 'max_diff_us': max_diff,
                      'missing': missing}, ensure_ascii=False, default=str))
    return 0


def proto_too_long(proto):
    m = re.search(r'> ([0-9.]+) µs', proto['marks']['too_long'])
    return float(m.group(1)) if m else 1.6


if __name__ == '__main__':
    sys.exit(main())
