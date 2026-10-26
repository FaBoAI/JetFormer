#!/usr/bin/env python3
"""解析 C の表と図(読むだけ)。collisions.py → col_data.json、steer_offset.py 20261023 20261024 → steer_offset.json、
cam_check.json(カメラで目で見た場所の分け方)を使う。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M python3 -I figures.py
出力: ~/jetformer_reports/bench_20261024_report/fig/col_f*.png・cam_*.png、この folder の col_tables.md・col_summary.json
図はどれも 2160×1350(14.4 × 9 インチ × 150 dpi)。日本語は Noto Sans CJK JP。"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIG = Path('/home/jetson/jetformer_reports/bench_20261024_report/fig')
LAB = Path('/home/jetson/jetformer/data/labels')
RAW = Path('/home/jetson/jetformer/data/raw')
FIGSIZE, DPI = (14.4, 9.0), 150
# 色(dataviz の既定の並び。回 = 1 青・2 橙・3 青緑。当たった = 赤・当たらなかった = 青)
C1, C2, C3, RED, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#898781'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
SESS = {1: ('1 回目 13:46〜(No.1〜4)', C1), 2: ('2 回目 16:01〜(No.5〜10・満充電)', C2),
        3: ('3 回目 18:31〜(No.11〜20・後輪の修理のあと)', C3)}


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.edgecolor': '#c3c2b7',
                         'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF,
                         'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.axisbelow': True, 'text.color': INK})
    return plt


def key(e):
    return '%s@%s' % (e['log'], e['t_s'])


LOC_ORDER = ['R1', 'GM', 'DE', 'SWO', 'SWX', 'HS', 'LH']
LOC_NAME = {'R1': 'R1\n(右ヘアピン)', 'GM': 'GM\n(緑マットの\nU ターン)', 'DE': 'DE\n(GM の出口)', 'SWO': 'SW の途中\n(奥の左カーブ)',
            'SWX': 'SW の出口の\n出っ張り', 'HS': 'HS\n(直線)', 'LH': 'LH\n(左ヘアピン)'}


def place(e, cam):
    """カメラで直した場所(R1・GM・DE・SWO・SWX・HS・LH)"""
    c = cam.get(key(e))
    if c == 'SWX':
        return 'SWX'
    if c in ('SWO', 'SWI'):
        return 'SWO'
    if c == 'HSE':
        return 'HS'
    if c == 'LHC':
        return 'LH'
    return {'SW': 'SWO'}.get(e['loc'], e['loc'])


def med_rng(xs, nd=1):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return '—'
    f = '%.' + str(nd) + 'f'
    return (f + '(' + f + '〜' + f + ')') % (np.median(xs), np.min(xs), np.max(xs))


def main():
    plt = setup()
    d = json.loads((HERE / 'col_data.json').read_text(encoding='utf-8'))
    cam = json.loads((HERE / 'cam_check.json').read_text(encoding='utf-8'))['events']
    so = json.loads((HERE / 'steer_offset.json').read_text(encoding='utf-8'))
    FIG.mkdir(parents=True, exist_ok=True)
    ev_all = d['events']
    ev = [e for e in ev_all if e['class'] == 'main']
    for e in ev:
        e['place'] = place(e, cam)
    S = {}
    # ---------------- 表 1: 回 × 場所(scorecard の loc_detail のまま / カメラで直したもの)
    lap_eq = {s: sum((v['lap_equiv'] or 0) for v in d['logs'].values() if v['session'] == s) for s in (1, 2, 3)}
    ai_s = {s: sum((v['ai_s'] or 0) for v in d['logs'].values() if v['session'] == s) for s in (1, 2, 3)}
    raw_tab = {s: {l: sum(1 for e in ev if e['session'] == s and e['loc'] == l) for l in ['R1', 'GM', 'DE', 'SW', 'HS', 'LH']} for s in (1, 2, 3)}
    fix_tab = {s: {l: sum(1 for e in ev if e['session'] == s and e['place'] == l) for l in LOC_ORDER} for s in (1, 2, 3)}
    strong_tab = {s: {l: sum(1 for e in ev if e['session'] == s and e['place'] == l and e['strong']) for l in LOC_ORDER} for s in (1, 2, 3)}
    S['lap_equiv'] = {s: round(v, 1) for s, v in lap_eq.items()}
    S['ai_s'] = {s: round(v, 1) for s, v in ai_s.items()}
    S['by_loc_detail'] = raw_tab
    S['by_place'] = fix_tab
    S['strong_by_place'] = strong_tab
    S['restart_events'] = [(e['ai_no'], e['loc'], e['t_s']) for e in ev_all if e['class'] != 'main']
    # ---------------- 表 2: 場所ごとの当たる前
    rows2 = []
    for l in LOC_ORDER:
        q = [e for e in ev if e['place'] == l]
        if not q:
            continue
        rows2.append({'place': l, 'n': len(q), 'strong': sum(e['strong'] for e in q),
                      'sessions': {s: sum(1 for e in q if e['session'] == s) for s in (1, 2, 3)},
                      'v_before': med_rng([e['speed_before_mps'] for e in q], 2),
                      'v_max_1p5': med_rng([e['v_max_1p5_mps'] for e in q], 2),
                      'steer_last': med_rng([e['steer_last_us'] for e in q], 0),
                      'steer_min': med_rng([e['steer_min_us'] for e in q], 0),
                      'steer_max': med_rng([e['steer_max_us'] for e in q], 0),
                      'thr_mean': med_rng([e['thr_mean_us'] for e in q], 0),
                      'thr_raw': med_rng([e['thr_raw_mean_us'] for e in q], 0),
                      'g': med_rng([e['g'] for e in q], 1),
                      'side': {k: sum(1 for e in q if e['side'] == k) for k in ('front', 'right', 'left')},
                      'model_vs_out_max_us': max([e['steer_model_minus_out_us'] or 0 for e in q])})
    S['pre_collision'] = rows2
    # ---------------- DE の通過
    de = d['de_passes']
    G = np.array(d['grid_de'])

    def arr(ps, k='shape'):
        return np.array([[np.nan if x is None else x for x in p[k]] for p in ps], float) if ps else np.zeros((0, len(G)))

    de_rate = {s: (sum(p['hit'] for p in de if p['session'] == s), sum(1 for p in de if p['session'] == s)) for s in (1, 2, 3)}
    S['de_pass_hits'] = de_rate
    S['de_hit_rel_s'] = med_rng([p['hit_rel_s'] for p in de if p['hit']], 2)
    grp = {
        'all_ok': [p for p in de if not p['hit']], 'all_hit': [p for p in de if p['hit']],
        'no9': [p for p in de if p['ai_no'] == '9'],
        's3_b090': [p for p in de if p['session'] == 3 and p['setting'] == 'B 速さ 0.90 ステア 1.00'],
    }
    S['de_groups'] = {}
    for g_, ps in grp.items():
        A = arr(ps)
        S['de_groups'][g_] = {'n': len(ps), 'hits': sum(p['hit'] for p in ps),
                              'steer_med_at': {('%.1f' % t): (None if not np.isfinite(A[:, i]).any() else round(float(np.nanmedian(A[:, i])), 0))
                                               for t in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7) for i in [int(np.argmin(np.abs(G - t)))]},
                              'overshoot_med': med_rng([p['overshoot_us'] for p in ps], 0),
                              'late_left_med': med_rng([p['late_left_us'] for p in ps], 0),
                              'entry_v': med_rng([p['entry_v_mps'] for p in ps], 2),
                              'vmax': med_rng([p['vmax_0_1p2_mps'] for p in ps], 2)}

    def y_at(p, t):
        tr = p.get('track')
        if not tr:
            return np.nan
        r = np.array(tr['rel_s'])
        return float(np.interp(t, r, tr['y'])) if r[-1] >= t - 0.02 else np.nan
    for g_, ps in grp.items():
        ys = [y_at(p, 0.5) for p in ps]
        S['de_groups'][g_]['track_y_0p5_m'] = med_rng(ys, 2)

    # ---------------- 図 1: 場所ごとの衝突(回ごと)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(LOC_ORDER))
    w = 0.26
    for j, s in enumerate((1, 2, 3)):
        vals = [fix_tab[s][l] for l in LOC_ORDER]
        lab, col = SESS[s]
        bars = ax.bar(x + (j - 1) * (w + 0.015), vals, width=w, color=col, label='%s・%.0f 周相当' % (lab, lap_eq[s]),
                      edgecolor=SURF, linewidth=2)
        for xi, v in zip(x + (j - 1) * (w + 0.015), vals):
            if v:
                ax.text(xi, v + 0.35, str(v), ha='center', va='bottom', fontsize=17, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LOC_NAME[l] for l in LOC_ORDER], fontsize=16)
    ax.set_ylabel('衝突の数(AI が操作中・2.5G 以上)')
    ax.set_ylim(0, 28)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', frameon=False)
    tot = {s: sum(fix_tab[s].values()) for s in (1, 2, 3)}
    ax.set_title('10/24 の衝突 %d 件: 3 回目は DE(%d 件)、2 回目は SW の出口の出っ張り(%d 件)に集まった'
                 % (sum(tot.values()), fix_tab[3]['DE'], fix_tab[2]['SWX']), loc='left', pad=16)
    fig.text(0.01, 0.015, '出典: since_1300/nightly/scorecard/scorecard.json の collision_events(class=main %d 件。立て直し中の 2 件は除く)。'
             '\n場所は loc_detail。SW・HS・LH と左に切っていた R1 はカメラで見て分け、2 回目の「R1」3 件・「HS」3 件は SW の出口の出っ張りに移した(cam_check.json)。'
             % len(ev), fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    f1 = FIG / 'col_f1_by_loc.png'
    fig.savefig(f1)
    plt.close(fig)

    # ---------------- 図 2: DE の舵の形
    fig, axs = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, sharey=True)

    def band(ax, ps, col, label, min_frac=0.5):
        A = arr(ps)
        n_ok = np.isfinite(A).sum(0)
        m = n_ok >= max(5, min_frac * len(ps))
        med = np.where(m, np.nanmedian(np.where(np.isfinite(A), A, np.nan), 0), np.nan)
        q1 = np.where(m, np.nanpercentile(A, 25, axis=0), np.nan)
        q3 = np.where(m, np.nanpercentile(A, 75, axis=0), np.nan)
        ax.fill_between(G, q1, q3, color=col, alpha=0.12, linewidth=0)
        ax.plot(G, med, color=col, lw=3, label=label, solid_capstyle='round')
        return med
    for ax in axs:
        ax.axhline(0, color='#c3c2b7', lw=1.2)
        ax.axvline(0, color=INK2, lw=1.2, ls='--')
        ax.axvspan(0.6, 1.0, color=RED, alpha=0.06, lw=0)
        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-260, 260)
        ax.set_xlabel('DE に入ってからの秒(負 = GM の出口の手前)')
        ax.text(0.02, 240, 'DE に入る', fontsize=14, color=INK2)
        ax.text(0.8, 240, '切り返し 0.6〜1.0 秒', fontsize=14, color=INK2, ha='center')
    a, b = axs
    ok, hit = grp['all_ok'], grp['all_hit']
    band(a, ok, C1, '当たらなかった %d 通過' % len(ok))
    band(a, hit, RED, '当たった %d 通過(当たるまで)' % len(hit))
    ht = [p['hit_rel_s'] for p in hit if p['hit_rel_s'] is not None and p['hit_rel_s'] <= 1.2]
    a.plot(ht, np.full(len(ht), -245), '|', color=RED, ms=18, mew=2)
    a.text(1.18, -228, '| = 当たった時刻', fontsize=13, color=INK2, ha='right')
    a.set_ylabel('ステア(中央からの µs。+ = 左・− = 右)')
    a.set_title('(a) 10/24 の全部: 当たった通過と当たらなかった\n通過の舵の形はほぼ同じ(左→右→左)', loc='left', fontsize=19)
    a.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1), frameon=False, fontsize=15)
    n9, s3 = grp['no9'], grp['s3_b090']
    band(b, n9, C2, 'No.9(2 回目 16:20)%d 通過・衝突 %d' % (len(n9), sum(p['hit'] for p in n9)))
    band(b, s3, C3, '3 回目 No.11・12・18 %d 通過・衝突 %d' % (len(s3), sum(p['hit'] for p in s3)))
    b.set_title('(b) 同じ B・速さ 0.90: No.9 は DE で左へ\n切り返さず、軽い右のまま抜けた', loc='left', fontsize=19)
    b.legend(loc='lower right', bbox_to_anchor=(1.0, 0.06), frameon=False, fontsize=15)
    fig.text(0.01, 0.015, '線 = 中央値、帯 = 25〜75%。当たった通過は当たった時刻で打ち切り、残りが半分未満の時刻は描かない。'
             '\nDE の通過 = tools/de_swing.py と同じ(ラベルの loc_code 3 が 1 秒以上・ずっと AI)。出典: data/labels/20261024-*.npz の out(車に出したステア)。',
             fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    f2 = FIG / 'col_f2_de_steer.png'
    fig.savefig(f2)
    plt.close(fig)

    # ---------------- 図 4: SW の出口の出っ張り — 入る速さと当たった割合
    swx_times = {}
    for e in ev:
        if e['place'] == 'SWX':
            swx_times.setdefault(e['log'], []).append(e['t_s'])
    rows4 = []
    matched = set()
    for log, info in sorted(d['logs'].items()):
        with np.load(LAB / (log + '.npz')) as z:
            t, det, ai, v = z['t_s'], z['loc_detail'], z['ai'], z['speed_mps']
        sw = det == 5
        e_ = np.flatnonzero(np.diff(np.r_[0, sw.astype(np.int8), 0]))
        for a0, b0 in zip(e_[::2], e_[1::2]):
            if b0 - a0 < 15 or not ai[a0:b0].all():
                continue
            ts, te = float(t[a0]), float(t[b0 - 1])
            hits = [h for h in swx_times.get(log, []) if ts - 0.2 <= h <= te + 1.6]
            for h in hits:
                matched.add((log, h))
            rows4.append({'log': log, 'session': info['session'], 'v_max': float(v[a0:b0].max()), 'hit': bool(hits)})
    bins = [(0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 9)]
    bl = ['3.5 m/s 未満', '3.5〜4.0', '4.0〜4.5', '4.5 m/s 以上']
    k_n = [(sum(r['hit'] for r in rows4 if lo <= r['v_max'] < hi), sum(1 for r in rows4 if lo <= r['v_max'] < hi)) for lo, hi in bins]
    S['swx_by_speed'] = {bl[i]: k_n[i] for i in range(len(bins))}
    S['swx_matched'] = '%d/%d' % (len(matched), sum(len(v) for v in swx_times.values()))
    S['sw_vmax_by_session'] = {s: med_rng([r['v_max'] for r in rows4 if r['session'] == s], 2) for s in (1, 2, 3)}
    S['sw_vmax_p90_by_session'] = {s: round(float(np.percentile([r['v_max'] for r in rows4 if r['session'] == s], 90)), 2) for s in (1, 2, 3)}
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    rate = [100 * k / n if n else 0 for k, n in k_n]
    xb = np.arange(len(bins))
    ax.bar(xb, rate, width=0.5, color=C1, edgecolor=SURF, linewidth=2)
    for xi, (k, n), r in zip(xb, k_n, rate):
        ax.text(xi, r + 2, '%d / %d 通過' % (k, n), ha='center', va='bottom', fontsize=18, color=INK)
    ax.set_xticks(xb)
    ax.set_xticklabels(['SW の中の最高 %s' % s if i == 0 else s for i, s in enumerate(bl)], fontsize=17)
    ax.set_ylabel('出っ張りに当たった割合(%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='x', visible=False)
    ax.set_title('SW の出口の出っ張りは速さで分かれる: SW で 4.5 m/s 以上なら 6 回中 4 回当たった', loc='left', pad=16)
    vs = S['sw_vmax_by_session']
    fig.text(0.01, 0.015, 'SW の通過 = ラベルの loc_detail が SW の区間(0.5 秒以上・ずっと AI)。当たった = その区間〜出口の 1.6 秒後に、カメラで出っ張りと確かめた衝突。'
             '\nSW の中の最高の速さの中央値: 1 回目 %s・2 回目 %s・3 回目 %s m/s(出っ張りの衝突 9 件のうち %s 件が通過に入った)。' % (vs[1].split('(')[0], vs[2].split('(')[0], vs[3].split('(')[0], S['swx_matched'].split('/')[0]),
             fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    f4 = FIG / 'col_f4_sw_speed.png'
    fig.savefig(f4)
    plt.close(fig)

    # ---------------- 図 5: まっすぐ進むステア(車の右への流れ)
    sess_of = lambda lg: 0 if lg.startswith('20261023') else (1 if int(lg[9:11]) < 15 else (2 if int(lg[9:11]) < 17 else 3))
    pts = []
    for r in so:
        ai_no = None
        for tok in r['description'].split():
            if tok.startswith('ai_no='):
                ai_no = tok.split('=')[1]
        try:
            ai_no = int(ai_no)
        except Exception:
            ai_no = None
        if r['log'].startswith('20261023') and not (ai_no and 8 <= ai_no <= 15):
            continue
        pts.append((sess_of(r['log']), ai_no, r['s0_us'], r['frames'], r['log']))
    S['steer_offset'] = {('10/23 公式' if s == 0 else '%d 回目' % s): {'median': round(float(np.median([p[2] for p in pts if p[0] == s])), 1),
                                                                   'runs': [(p[1], p[2]) for p in pts if p[0] == s]} for s in (0, 1, 2, 3)}
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    cols = {0: GRAY, 1: C1, 2: C2, 3: C3}
    names = {0: '10/23 公式 8 本(No.8〜15)', 1: SESS[1][0], 2: SESS[2][0], 3: SESS[3][0]}
    xpos = 0
    for s in (0, 1, 2, 3):
        q = [p for p in pts if p[0] == s]
        xs = np.arange(xpos, xpos + len(q))
        ax.scatter(xs, [p[2] for p in q], s=140, color=cols[s], edgecolor=SURF, linewidth=2, zorder=3, label=names[s])
        m = float(np.median([p[2] for p in q]))
        ax.plot([xs[0] - 0.4, xs[-1] + 0.4], [m, m], color=cols[s], lw=3, solid_capstyle='round')
        ax.text(xs[-1] + 0.55, m, '中央値 %+.0f' % m, va='center', ha='left', fontsize=15, color=INK)
        if s:
            for xi, p in zip(xs, q):
                ax.text(xi, p[2] - 2.2, 'No.%s' % p[1], ha='center', va='top', fontsize=12, color=INK2)
        xpos += len(q) + 3.4
    ax.axhline(0, color='#c3c2b7', lw=1.2)
    ax.set_xticks([])
    ax.set_xlim(-1, xpos - 0.6)
    ax.set_ylim(-30, 45)
    ax.set_ylabel('まっすぐ進むステア(µs、+ = 左に切って\nやっとまっすぐ = 車が右へ流れる)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', frameon=False, fontsize=15, ncol=2)
    ax.set_title('3 回目(後輪の修理のあと)は、まっすぐ進むのに左へ約 20 µs 多く切る必要があった', loc='left', pad=16)
    fig.text(0.01, 0.015, '推定: AI が走っている(1.5 m/s 超・衝突の前後 1 秒を除く)フレームで、曲率(ヨー ÷ 速さ)を 0.1 秒前のステアに直線で当てはめ、曲率 0 のステアを出した'
             '\n(steer_offset.py。遅れ 0.03〜0.2 秒でも同じ傾向)。ステアの中央の設定は全記録 1441 µs のまま。点の下の No. は 10/24 の AI 走行の番号。',
             fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    f5 = FIG / 'col_f5_steer_offset.png'
    fig.savefig(f5)
    plt.close(fig)

    # ---------------- 図 3: カメラの画像(人が写っていないコマを目で確かめて選んだ)
    from PIL import Image, ImageDraw, ImageFont
    swx_v = [e['speed_before_mps'] for e in ev if e['place'] == 'SWX']
    S['swx_speed_before'] = med_rng(swx_v, 1)
    S['swx_n'] = len(swx_v)
    picks = [
        ('cam_de_hit.png', '20261024-183437', 23.548, 'DE で当たる直前',
         '3 回目 No.12(B 0.90)。\nDE に入って 0.47 秒、\n当たる 0.27 秒前(11.3 G)。\n右の白い壁のすぐ横を\n走り、壁の角へ向かう'),
        ('cam_de_clean.png', '20261024-162033', 50.861, '当たらなかった通過の\n同じ場所',
         '2 回目 No.9(同じ B 0.90)。\nDE に入って 0.47 秒。\n車は道のまん中で、\n右の壁は遠い'),
        ('cam_sw_exit.png', '20261024-162033', 36.406, 'SW の出口\n(駐車場の出っ張り)',
         'No.9 の通過。右の白い箱と\n床の緑の P の印のところ。\n2 回目は %.1f〜%.1f m/s で\n曲がりきれず、ここに\n%d 回当たった' % (min(swx_v), max(swx_v), len(swx_v))),
        ('cam_no9_straight.png', '20261024-162033', 29.145, 'No.9 の 8 秒台の周の\n直線(HS)',
         'ベスト 8.25 秒の周、\n4.2 m/s。奥の黒い\nカーテンの先が LH'),
    ]
    font_t = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 34)
    font_c = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 28)
    font_h = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 42)
    font_s = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
    canvas = Image.new('RGB', (2160, 1350), SURF)
    dr = ImageDraw.Draw(canvas)
    dr.text((40, 24), '実車のカメラ(224×224 を拡大): DE の当たり方と、SW の出口の出っ張り', fill=INK, font=font_h)
    S['camera'] = []
    IMG = 560
    for i, (fn, log, ts, head, cap) in enumerate(picks):
        lj = json.loads((LAB / (log + '.json')).read_text(encoding='utf-8'))
        t0 = int(lj['t0_us'])
        best = None
        for sdir in sorted(p for p in (RAW / log).iterdir() if p.is_dir()):
            try:
                with np.load(sdir / 'CAM_FRONT.npz') as z:
                    tt = (z['timestamps_us'] - t0) / 1e6
                    j = int(np.argmin(np.abs(tt - ts)))
                    if best is None or abs(tt[j] - ts) < best[0]:
                        best = (abs(tt[j] - ts), str(sdir / str(z['files'][j])), float(tt[j]))
            except Exception:
                pass
        src = Image.open(best[1]).convert('RGB')
        src.resize((672, 672), Image.LANCZOS).save(FIG / fn)
        S['camera'].append({'file': str(FIG / fn), 'source': best[1], 't_s': round(best[2], 3), 'title': head.replace('\n', '')})
        col, row = i % 2, i // 2
        x0, y0 = 40 + col * 1060, 110 + row * 610
        canvas.paste(src.resize((IMG, IMG), Image.LANCZOS), (x0, y0))
        tx = x0 + IMG + 28
        dr.multiline_text((tx, y0 + 6), '(%d) %s' % (i + 1, head), fill=INK, font=font_t, spacing=8)
        nh = head.count('\n') + 1
        dr.multiline_text((tx, y0 + 18 + 44 * nh + 14), cap, fill=INK2, font=font_c, spacing=12)
        dr.text((tx, y0 + IMG - 28), '%s  t=%.2f 秒' % (log, best[2]), fill='#898781', font=font_s)
        # はみ出しの確かめ
        for txt, fnt in ((head, font_t), (cap, font_c)):
            for line in txt.split('\n'):
                wpx = dr.textlength(line, font=fnt)
                if tx + wpx > x0 + 1040:
                    print('はみ出し', fn, line, wpx)
    # キャプションの幅が足りないときは折り返しを確かめる(右の余白 = 1060 − 605 − 40)
    canvas.save(FIG / 'col_f3_cam.png')
    f3 = FIG / 'col_f3_cam.png'

    S['figures'] = [str(f1), str(f2), str(f3), str(f4), str(f5)]
    (HERE / 'col_summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in S.items() if k not in ('pre_collision',)}, ensure_ascii=False, indent=1)[:6000])
    for r in rows2:
        print(r)


if __name__ == '__main__':
    main()
