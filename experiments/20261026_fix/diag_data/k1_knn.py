#!/usr/bin/env python3
"""原因 A の診断 K1(numpy だけ・読むだけ): G1 の予測と埋め込みで、
(1) 学習のサンプルでの残差(出どころごと): B と v2-H は新しい人の正解をどれだけまねたか
(2) 机上の直線・止まった場面・10/10 の低速のフレームに近い学習のサンプル(埋め込みの近さ、k = 16)の出どころと正解
(3) 直し方の見込み: 学習のサンプルを外したときの近いサンプルの正解の平均(kNN の代わりの見込み)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I k1_knn.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
SETS = Path('/home/jetson/jetformer_reports/train_v2_20261025/desk/sets_v2.npz')
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
K = 16
NEUTRAL_B = 1496.0
BRAKE = 1598.0
ROW0, ROW2 = 0, 2
SESS = {0: '10/10', 1: '10/23', 2: '10/24-2'}
TK_SUBS = ('交代の直後3秒', '走り+交代の直後')


def load(name):
    with np.load(OUT / ('g1_%s.npz' % name)) as z:
        return {k: z[k] for k in z.files}


def main():
    D = np.load(OUT / 'd1_samples.npz')
    S = np.load(SETS)
    grp, sub, sess_tr = D['group'], D['sub'], D['sess']
    v_tr = D['v']
    new = grp != 'B41(10/10 人)'
    tk = new & np.isin(sub, TK_SUBS)
    run_ = new & (sub == '走りの区間')
    new2324 = new & np.isin(sess_tr, ['10/23', '10/24-2'])
    stop_tr = v_tr < 0.3
    brake_lab = D['thr0'] > BRAKE
    fwd_lab = NEUTRAL_B - D['thr2']
    variants = {
        'v2-H のまま': np.ones(len(grp), bool),
        '交代の直後を外す': ~tk,
        '新しい人の 10/23・10/24 を外す': ~new2324,
        '新しい人を全部外す(= B の 41 シーン)': ~new,
        '交代の直後を外す + 0.3 m/s 未満を外す': ~tk & ~stop_tr,
        '0.3 m/s 未満を外す(だけ)': ~stop_tr,
        '交代の直後を外す + 低速のブレーキの正解を外す': ~tk & ~(brake_lab & (v_tr < 1.0)),
    }
    names = sorted(p.stem[3:] for p in OUT.glob('g1_*.npz'))
    if B_RUN not in names:
        raise SystemExit('B の G1 がありません')
    G = {n: load(n) for n in names}
    part = G[B_RUN]['part']
    st_rows, sd_rows = G[B_RUN]['st_rows'], G[B_RUN]['sd_rows']
    st_sess = S['sess'][st_rows]
    tr = part == 0
    out = {'k': K, 'n_train': int(tr.sum()), 'variants_n': {k: int(m.sum()) for k, m in variants.items()},
           'base_share': {'新しい人': float(new.mean()), '交代の直後': float(tk.mean()), '走りの区間(新しい人)': float(run_.mean())},
           'runs': {}}
    pB = G[B_RUN]['pred']
    # 確かめ: G1 の B の直線の予測が C3 の保存した tab の予測と同じか
    with np.load('/home/jetson/jetformer_reports/train_v2_20261025/desk/pred/%s_tab.npz' % B_RUN) as z:
        tabB = z['pred'].astype(np.float64)
    out['check_B_straight_vs_c3_tab_max_abs_us'] = float(np.abs(pB[part == 1] - tabB[st_rows]).max())
    for n in names:
        g = G[n]
        p = g['pred'].astype(np.float64)
        E = g['emb'].astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        r = {'train_residual_thr_row2(予測 − 正解。+ = 正解より緩める)': {}, 'train_pred_vs_B_row2': {}, 'test': {}}
        ptr = p[tr]
        for gn, m in (('B41', ~new), ('新しい人 走りの区間', run_), ('新しい人 交代の直後', tk), ('新しい人 10/23・10/24', new2324),
                      ('新しい人 全部', new)):
            r['train_residual_thr_row2(予測 − 正解。+ = 正解より緩める)'][gn] = [
                float((ptr[m, ROW2, 1] - D['thr2'][m]).mean()), int(m.sum())]
            r['train_pred_vs_B_row2'][gn] = float((ptr[m, ROW2, 1] - pB[tr][m, ROW2, 1]).mean())
        Etr = E[tr]
        tests = {'直線 10/10': (part == 1, st_sess == 0), '直線 10/23': (part == 1, st_sess == 1),
                 '止まった場面': (part == 2, None), '10/10 の AI 走行の低速': (part == 3, None)}
        for tn, (pm, sm) in tests.items():
            idx = np.nonzero(pm)[0]
            if sm is not None:
                idx = idx[sm]
            if not len(idx):
                continue
            sim = E[idx] @ Etr.T                                   # [T, Ntr]
            res = {'n': int(len(idx)),
                   'model_vs_B_row2_us': float((p[idx, ROW2, 1] - pB[idx, ROW2, 1]).mean()),
                   'model_brake_frac_row0': float((p[idx, ROW0, 1] > BRAKE).mean()),
                   'B_brake_frac_row0': float((pB[idx, ROW0, 1] > BRAKE).mean()),
                   'variants': {}}
            for vn, vm in variants.items():
                cols = np.nonzero(vm)[0]
                s = sim[:, cols]
                nn = cols[np.argpartition(-s, K, axis=1)[:, :K]]          # [T, K] 学習のサンプルの番号
                res['variants'][vn] = {
                    'knn_fwd_row2_us': float(fwd_lab[nn].mean()),
                    'knn_brake_frac_row0': float(brake_lab[nn].mean()),
                    'share_new': float(new[nn].mean()), 'share_takeover': float(tk[nn].mean()),
                    'share_new_1023_1024': float(new2324[nn].mean())}
            full = res['variants']['v2-H のまま']['knn_fwd_row2_us']
            for vn in res['variants']:
                res['variants'][vn]['fwd_gain_vs_full_us'] = res['variants'][vn]['knn_fwd_row2_us'] - full
            r['test'][tn] = res
        out['runs'][n] = r
    (OUT / 'k1_knn.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print('check B straight vs C3 tab max abs', out['check_B_straight_vs_c3_tab_max_abs_us'])
    print('base share', out['base_share'])
    for n, r in out['runs'].items():
        print('==', n)
        print('  train residual', {k: [round(a, 1), m] for k, (a, m) in r['train_residual_thr_row2(予測 − 正解。+ = 正解より緩める)'].items()})
        print('  train pred vs B', {k: round(a, 1) for k, a in r['train_pred_vs_B_row2'].items()})
        for tn, res in r['test'].items():
            vv = res['variants']
            print('  %-22s n=%4d model−B=%6.1f brake m/B=%.3f/%.3f | ' % (tn, res['n'], res['model_vs_B_row2_us'],
                                                                     res['model_brake_frac_row0'], res['B_brake_frac_row0'])
                  + ' | '.join('%s: fwd %.0f (%+.0f) brk %.3f new %.2f tk %.2f' % (vn[:10], x['knn_fwd_row2_us'], x['fwd_gain_vs_full_us'],
                                                                              x['knn_brake_frac_row0'], x['share_new'], x['share_takeover'])
                               for vn, x in vv.items()))


if __name__ == '__main__':
    sys.exit(main())
