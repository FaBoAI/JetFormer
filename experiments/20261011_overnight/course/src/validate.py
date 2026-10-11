"""合わせ込みの検証: カメラの目印(スタートライン / LED ゲート)を周の偶奇で半分ずつ隠して合わせ、隠した方の誤差を見る。
カメラなし(1段目)の結果では、スタートライン 1→3 本目の間隔(真値 3.60 m)がどれだけ縮むかも見る。"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import course_geom as cg, track_fit as tf, fit_run as fr
from loader import L

def main(log, t0, t1, out):
    geo = cg.build(); C = cg.centerline(); cl = tf.Centerline(C); walls = cg.wall_segments(geo)
    d = L(log); psi, s, bias, _, _ = tf.raw_dr(d)
    m = (d.it >= t0) & (d.it <= t1); t = d.it[m]; psi = psi[m]; s = s[m] - s[m][0]; gz = d.gyr[m, 2]; gi = d.g[m]
    res = {'log': log, 't0': t0, 't1': t1}
    # 1段目(カメラなし)
    F, x, use, info = fr.fit_segment(d, t, psi, s, gz, gi, cl, walls, lines=False)
    X, Z, h = F.pose(x)
    te, lab = fr.detect_lines(d, t, X, Z, h)
    ei = np.clip(np.searchsorted(t, te), 0, len(t) - 1)
    grp = np.cumsum(np.r_[1, np.diff(te) > 1.2])
    sp = []
    for g in np.unique(grp):
        k = np.nonzero(grp == g)[0]
        if len(k) == 3 and list(lab[k]) == [0, 1, 2]:
            sp.append(float(X[ei[k[2]]] - X[ei[k[0]]]))
    res['stage1_line1_to_3_spacing_m'] = sp
    res['stage1_dist_scale'] = float(x[1]); res['stage1_gyro_scale'] = float(x[0])
    # 交差検証
    tl, tg = [], []
    for ho in (0, 1):
        F2, x2, u2, i2 = fr.fit_segment(d, t, psi, s, gz, gi, cl, walls, holdout=ho)
        tl += [r for r, u in zip(i2['line_residual_m'], i2['line_used']) if not u]
        tg += [r for r, u in zip(i2['gate_residual_m'], i2['gate_used']) if not u]
        res['holdout%d_in_line' % ho] = [r for r, u in zip(i2['line_residual_m'], i2['line_used']) if u]
        res['holdout%d_in_gate' % ho] = [r for r, u in zip(i2['gate_residual_m'], i2['gate_used']) if u]
    res['test_line_residual_m'] = tl; res['test_gate_residual_m'] = tg
    a = np.abs(tl); b = np.abs(tg)
    res['summary'] = {'test_line_abs_median_m': float(np.median(a)), 'test_line_abs_p90_m': float(np.percentile(a, 90)), 'n_line': len(a),
                      'test_gate_abs_median_m': float(np.median(b)) if len(b) else None, 'test_gate_abs_p90_m': float(np.percentile(b, 90)) if len(b) else None,
                      'n_gate': len(b), 'stage1_spacing_median_m': float(np.median(sp)) if sp else None}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res['summary'], ensure_ascii=False))

if __name__ == '__main__':
    log = sys.argv[1]
    main(log, float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
