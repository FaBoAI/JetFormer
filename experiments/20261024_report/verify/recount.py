"""verify/recount.py — 10/24 のレポートの数を、生の記録だけから別のやり方で数え直す。

使うもの: data/raw/<log>/<chunk>/{AI_CMD,VEHICLE,IMU}.npz と log.json だけ。
labels・scorecard.json・tt_data.json は使わない(比べるのは compare.py)。

- AI 側   = VEHICLE.select(CH3)が True かつ AI_CMD.dry_run が False。
- 向き     = IMU の gz(deg/s)から、止まっているときの中央値を引いて積分。
- R1 の起点 = 向きの山(左の LH の終わり)で、そのあと 1.6 秒以内に 100° 以上右へ回るところ。
- 偽の起点 = 前の起点からの向きの変化が 270° 未満(半周で右へ回った)。
- 周       = 起点 → 次の起点。5〜16 秒・全部 AI 側・VEHICLE の途切れ 0.2 秒以下・向きの変化 270〜450°。
- 衝突     = 水平の加速度(ax, ay から止まっているときの中央値を引く)が 2.5G 以上。0.6 秒以内はまとめ、最大を取る。
             AI 側のものだけ。交代(AI→人)の直前 3 秒に衝突があり、そのあと人が AI に戻して 2.5 秒以内の衝突は「立て直し中」。
- 場所     = 直前の本物の起点からの向き(psi_rel)と時間で分ける(下の loc())。カメラは見ない。
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawload

LOGS24 = ['20261024-134607', '20261024-134907', '20261024-135608', '20261024-135656', '20261024-160133',
          '20261024-160536', '20261024-160654', '20261024-160847', '20261024-162033', '20261024-162305',
          '20261024-183123', '20261024-183437', '20261024-183656', '20261024-183826', '20261024-184043',
          '20261024-184400', '20261024-184549', '20261024-185703', '20261024-185823', '20261024-185926']
LOGS23 = ['20261023-210840', '20261023-211020', '20261023-211118', '20261023-211209',
          '20261023-211308', '20261023-211357', '20261023-211448', '20261023-211547']
MODEL = {'clean_driving_20261011-nocmd-s3-20261011-024626': 'B', 'bench_20261022-slow-sec-s1': 'SLOW',
         'bench_20261021-ens-p2': 'ENS', 'bench_20261021-aug-s0': 'AUG',
         'clean_driving_20261011-act-s0-nocmd-stride3-e30': 'ACT', 'bench_20261021-wgt-s0': 'WGT'}


def nearest(t_src, x_src, t_dst):
    k = np.searchsorted(t_src, t_dst).clip(1, len(t_src) - 1)
    k = np.where(np.abs(t_src[k - 1] - t_dst) < np.abs(t_src[k] - t_dst), k - 1, k)
    return x_src[k]


def analyse(log):
    r = rawload.load(log)
    meta = r['meta']
    ai_no = rawload.desc_field(meta, 'ai_no')
    run = meta.get('drive', {}).get('run')
    vt, ve, vs = r['v_t'], r['v_erpm'], r['v_sel']
    dry_v = nearest(r['a_t'], r['a_dry'], vt)
    ai_v = vs & ~dry_v
    dtv = np.diff(vt, append=vt[-1]).clip(0, 0.05)
    ai_s = float((dtv * ai_v).sum())
    # ゲイン(AI 側のフレーム)
    sel_a = r['a_sel'] & ~r['a_dry']
    g = r['a_gain'][sel_a] if sel_a.any() else r['a_gain']
    gains = (float(np.median(g[:, 0])), float(np.median(g[:, 1])))
    gains_set = sorted(set(map(tuple, np.round(g, 3).tolist())))
    # 向き
    it = r['i_t']
    er_i = np.interp(it, vt, ve)
    st = np.abs(er_i) < 1
    bias = np.median(r['i_gz'][st]) if st.sum() > 50 else 0.0
    gz = r['i_gz'] - bias
    dti = np.diff(it, prepend=it[0]).clip(0, 0.05)
    psi = np.cumsum(gz * dti)
    ai_i = nearest(vt, ai_v.astype(int), it).astype(bool)
    # 起点の候補
    n = len(it)
    hz = 1.0 / np.median(np.diff(it))
    w16 = int(1.6 * hz); w08 = int(0.8 * hz)
    cands = []
    for k in range(w08, n - w16):
        if psi[k] < psi[k - w08:k + w08 + 1].max():
            continue
        drop = psi[k] - psi[k:k + w16].min()
        if drop >= 100 and abs(er_i[k]) > 1500:
            cands.append(k)
    anchors = []
    for k in cands:
        if anchors and it[k] - it[anchors[-1]] < 2.0:
            if psi[k] > psi[anchors[-1]]:
                anchors[-1] = k
            continue
        anchors.append(k)
    # 距離(車輪)
    spd_v = np.abs(ve) * 0.5e-3
    dist_v = np.concatenate([[0], np.cumsum(spd_v[:-1] * np.diff(vt).clip(0, 0.05))])
    A = []
    for j, k in enumerate(anchors):
        prev = anchors[j - 1] if j else None
        dpsi_prev = psi[k] - psi[prev] if prev is not None else None
        kv = np.searchsorted(vt, it[k])
        w = (vt >= it[k] - 0.4) & (vt < it[k])
        entry = float(np.max(-ve[w])) if w.any() else None
        A.append(dict(t=float(it[k]), ai=bool(ai_i[k]), dpsi_prev=dpsi_prev, entry_erpm=entry,
                      dist_prev=float(dist_v[kv] - dist_v[np.searchsorted(vt, it[prev])]) if prev is not None else None))
    # 偽の起点: 直前の「本物の」起点からの向きの変化が 270° 未満(かつ 16 秒以内)
    last = None
    for j, a in enumerate(A):
        k = anchors[j]
        if last is not None and psi[k] - psi[anchors[last]] < 270 and a['t'] - A[last]['t'] < 16:
            a['false'] = True
            a['dpsi_real'] = float(psi[k] - psi[anchors[last]])
        else:
            a['false'] = False
            last = j
    # 衝突
    acc = r['i_acc']
    bx = np.median(acc[st, 0]) if st.sum() > 50 else 0.0
    by = np.median(acc[st, 1]) if st.sum() > 50 else 0.0
    ah = np.hypot(acc[:, 0] - bx, acc[:, 1] - by) / rawload.G
    over = np.where(ah >= 2.5)[0]
    events = []
    for k in over:
        if events and it[k] - events[-1]['t_last'] <= 0.6:
            e = events[-1]
            e['t_last'] = it[k]
            if ah[k] > e['g']:
                e['g'] = float(ah[k]); e['t'] = float(it[k])
        else:
            events.append(dict(t=float(it[k]), t_first=float(it[k]), t_last=it[k], g=float(ah[k])))
    # 交代・戻し
    tr = np.diff(vs.astype(int))
    takeovers = vt[1:][tr == -1]
    handbacks = vt[1:][tr == 1]
    cols = []
    for e in events:
        ai_here = bool(nearest(vt, ai_v.astype(int), np.array([e['t_first']]))[0])
        if not ai_here:
            continue
        restart = False
        hb_before = handbacks[handbacks <= e['t_first']]
        if len(hb_before):
            hb = hb_before[-1]
            tk_before = takeovers[takeovers < hb]
            if len(tk_before) and e['t_first'] - hb <= 2.5:
                tk = tk_before[-1]
                if any((x['t_first'] <= tk) and (x['t_first'] >= tk - 3.0) for x in events):
                    restart = True
        cols.append(dict(t=round(e['t'], 3), t_first=round(e['t_first'], 3), g=round(e['g'], 2),
                         strong=e['g'] >= 4.0, restart=restart))
    # 場所(向きと時間)
    real = [a for a in A if not a['false']]
    for c in cols:
        prev = [a for a in real if a['t'] <= c['t_first'] + 0.05]
        if not prev or c['t_first'] - prev[-1]['t'] > 16:
            c['loc'] = 'start'; continue
        ta = prev[-1]['t']
        ka = np.searchsorted(it, ta); kc = np.searchsorted(it, c['t_first'])
        rel = psi[ka:kc + 1] - psi[ka]
        c['psi_rel'] = round(float(rel[-1]), 1)
        c['dt_anchor'] = round(c['t_first'] - ta, 2)
        c['loc'] = loc_of(rel, it[ka:kc + 1] - ta)
    # 周
    laps = []
    for j in range(1, len(A)):
        a0, a1 = A[j - 1], A[j]
        if a0['false']:
            continue
        lap = a1['t'] - a0['t']
        k0, k1 = np.searchsorted(vt, a0['t']), np.searchsorted(vt, a1['t'])
        seg_ai = bool(ai_v[k0:k1].all()) if k1 > k0 else False
        gap = float(np.diff(vt[k0:k1 + 1]).max()) if k1 > k0 else 9
        dpsi = a1['dpsi_prev']
        ok = 5 <= lap <= 16 and seg_ai and gap <= 0.2 and 270 <= dpsi <= 450
        wa = (r['a_t'] >= a0['t'] - 0.05) & (r['a_t'] <= a1['t'] + 0.05)
        gap_cmd = float(np.diff(r['a_t'][wa]).max()) if wa.sum() > 1 else 9
        cin = [c for c in cols if a0['t'] <= c['t_first'] < a1['t']]
        w = (vt >= a0['t']) & (vt < a1['t'])
        p90 = float(np.percentile(spd_v[w], 90)) if w.any() else None
        laps.append(dict(t0=round(a0['t'], 3), t1=round(a1['t'], 3), lap=round(lap, 3), ok=ok, false_end=a1['false'],
                         dpsi=round(dpsi, 1), dist=round(a1['dist_prev'], 1), clean=(len(cin) == 0), gap_cmd=round(gap_cmd, 3),
                         gmax=max([c['g'] for c in cin], default=0.0), p90=p90))
    # 発進の補助
    return dict(log=log, ai_no=ai_no, run=run, model=MODEL.get(run, run), gains=gains, gains_set=gains_set,
                params=meta.get('drive', {}).get('params', {}), desc_tg=rawload.desc_field(meta, 'throttle_gain'),
                ai_s=round(ai_s, 2), anchors=A, laps=laps, collisions=cols, gz_bias=float(bias))


def loc_of(rel, tt):
    """直前の起点からの向き rel(度)の流れで場所を分ける。
    R1: 右へ回っている間(最小の向きまで)。GM: 最小から −30° まで戻る間。DE: その後 +40° まで。
    SW: +40〜+150°。SWX(SW の出口〜HS の入口、出っ張りの辺り): +150° を越えて 1.2 秒以内。HS: そのあと LH に入るまで(+215° 未満)。LH: +215° 以上。"""
    kmin = int(np.argmin(rel[:min(len(rel), int(2.5 / max(np.median(np.diff(tt)) if len(tt) > 1 else 0.01, 1e-3)))] if len(rel) > 1 else 0))
    cur = rel[-1]
    if len(rel) - 1 <= kmin or tt[-1] < 0.3:
        return 'R1'
    after = rel[kmin:]
    up = np.where(after >= -30)[0]
    if len(up) == 0:
        return 'GM'
    k_gm_exit = kmin + up[0]
    a2 = rel[k_gm_exit:]
    up40 = np.where(a2 >= 40)[0]
    if len(up40) == 0:
        return 'DE'
    k_sw = k_gm_exit + up40[0]
    a3 = rel[k_sw:]
    up150 = np.where(a3 >= 150)[0]
    if len(up150) == 0:
        return 'SW'
    k150 = k_sw + up150[0]
    if cur >= 215:
        return 'LH'
    if tt[-1] - tt[k150] <= 1.2:
        return 'SWX'
    return 'HS'


if __name__ == '__main__':
    out = {}
    for lg in LOGS24 + LOGS23:
        try:
            out[lg] = analyse(lg)
        except Exception as ex:  # noqa
            out[lg] = dict(log=lg, error=repr(ex))
        d = out[lg]
        print(lg, d.get('ai_no'), d.get('model'), d.get('gains'), 'ai_s', d.get('ai_s'),
              'laps', sum(l['ok'] for l in d.get('laps', [])), 'cols', len(d.get('collisions', [])), d.get('error', ''))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recount.json'), 'w'),
              ensure_ascii=False, indent=1, default=float)
