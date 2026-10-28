"""fix8 係 tools 直す 8-2 回目(読むだけ): 本物の 10/10・10/23・10/24 の、drive が中立を書いた時間の見積もりの合計 Σ の分布
(w2-tools-r8-2-short-gaps-add-up のしきいを決めるため)。

drive.py はフレームが 0.1 秒来ないと中立を書き(cam_evt.wait(0.1) → hold('カメラの映像が来ていません')。行は書かない)、古いフレーム・
古い ToF/IMU/車両の値のフレームは、その行で中立を書く(中立止めの行 = hold)。中立は次の処理したフレームまで続く。そこで、発進(AI 側に
した時刻 t0)から t_end までの、AI 側の処理したフレーム(中立止めの行でない)どうしの間 (a, b) ごとに
  A  max(0, b − (a + 0.1))                       … 反証の係・票 A・B と同じ(中立止めの行もフレームの無い間と同じに数える)
  H  max(0, b − min(a + 0.1, 間の最初の中立止めの行))  … 中立止めの行はその行から中立(drive のとおり)
を足す(端の t0・t_end も点に入れる = ⑤ launch_stall と同じ並び)。

  1  ⑤(直す前 = tests/fixtures/fix8_tools_r2_before/collect.py.orig)の --day D --ignore-gain の R1 に届いた発進の、発進 → R1 の起点
     (../out/real_old/launches_<D>_ignore-gain.json の t_ai・to_r1_s。z は ⑤ と同じ data/labels/<記録>.npz)
  2  AI 側の全部の 3.0 秒の窓(AI 側の処理したフレームから始めて滑らせる。票 B の D と同じ)
  3  AI の区間の始めから 3.5 秒の窓(止まって始めた区間 = 始めの |速さ| < 0.2 m/s だけ。⑤ の発進と同じ決め)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/fix_tools_r2/probes/real_sum.py
  → ../out/real_sum.txt・real_sum.json
"""
import json
from pathlib import Path

import numpy as np

LAB = Path('/home/jetson/jetformer/data/labels')
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'out'
DAYS = ('20261010', '20261023', '20261024')
AFTER = 0.1
STALL = 0.22


def segs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def gaps_of(tt, hh, t0, t_end):
    """(点の並び pts, 間ごとの中立の始め A・H)。pts = [t0] + 処理したフレーム(t0 < t < t_end)+ [t_end]"""
    good = tt[~hh]
    good = good[(good > t0) & (good < t_end)]
    pts = np.concatenate([[t0], good, [t_end]])
    hs = tt[hh]
    hs = hs[(hs >= t0) & (hs < t_end)]
    sa = pts[:-1] + AFTER
    sh = sa.copy()
    if len(hs):
        k = np.searchsorted(hs, pts[:-1], side='left')          # 間 (a, b) の最初の中立止めの行(a 以上)
        ok = k < len(hs)
        first = np.where(ok, hs[np.minimum(k, len(hs) - 1)], np.inf)
        inside = ok & (first < pts[1:])
        sh = np.where(inside, np.minimum(sa, first), sa)
    return pts, sa, sh


def sums(tt, hh, t0, t_end):
    pts, sa, sh = gaps_of(tt, hh, t0, t_end)
    g = np.diff(pts)
    na = np.clip(pts[1:] - sa, 0, None)
    nh = np.clip(pts[1:] - sh, 0, None)
    return {'max_gap': round(float(g.max()), 3), 'sum_a': round(float(na.sum()), 3), 'sum_h': round(float(nh.sum()), 3),
            'n_a': int((na > 0).sum()), 'n_h': int((nh > 0).sum()), 'holds': int(((tt >= t0) & (tt < t_end) & hh).sum())}


def load(log):
    z = np.load(LAB / (log + '.npz'))
    t = np.asarray(z['t_s'], np.float64)
    ai = np.asarray(z['ai'], bool)
    if 'dry_run' in z.files:
        ai &= ~np.asarray(z['dry_run'], bool)
    h = np.asarray(z['hold'], bool) if 'hold' in z.files else np.zeros(len(t), bool)
    v = np.asarray(z['speed_mps'], np.float64)
    return t, ai, h, v


def main():
    lines, js = [], {}
    # 1 ⑤ の R1 に届いた発進
    lines.append('# 1 ⑤(直す前)の --day D --ignore-gain の R1 に届いた発進の、発進 → R1 の起点(launch_stall と同じ並び)')
    allr = []
    for d in DAYS:
        j = json.loads((OUT / 'real_old' / ('launches_%s_ignore-gain.json' % d)).read_text(encoding='utf-8'))
        rr = []
        for r in j['launches']:
            if r.get('to_r1_s') is None:
                continue
            t, ai, h, v = load(r['log'])
            i0 = int(np.argmin(np.abs(t - r['t_ai'])))
            assert abs(t[i0] - r['t_ai']) < 1e-3 and ai[i0], (r['log'], r['t_ai'])
            i1 = i0
            while i1 + 1 < len(t) and ai[i1 + 1]:
                i1 += 1
            s = sums(t[i0:i1 + 1], h[i0:i1 + 1], float(t[i0]), float(t[i0]) + float(r['to_r1_s']))
            s.update(log=r['log'], t_ai=r['t_ai'], to_r1=r['to_r1_s'], state=r.get('launch_state'),
                     compared=not (r.get('midrun') or r.get('trial') or r.get('after_abort')))
            rr.append(s)
        allr += rr
        top = sorted(rr, key=lambda x: -x['sum_h'])[:3]
        lines.append(' %s: R1 に届いた発進 %d・いちばん長い間の最大 %.3f・Σ A の最大 %.3f・Σ H の最大 %.3f・中立止めの行のある発進 %d' % (
            d, len(rr), max(x['max_gap'] for x in rr), max(x['sum_a'] for x in rr), max(x['sum_h'] for x in rr),
            sum(1 for x in rr if x['holds'])))
        for x in top:
            lines.append('    %s@%.2f 起点まで %.3f・いちばん長い間 %.3f・Σ A %.3f(%d 回)・Σ H %.3f(%d 回)・中立止めの行 %d・比べに入る %s'
                         % (x['log'], x['t_ai'], x['to_r1'], x['max_gap'], x['sum_a'], x['n_a'], x['sum_h'], x['n_h'], x['holds'],
                            x['compared']))
    cmp_ = [x for x in allr if x['compared']]
    lines.append(' 3 日: R1 に届いた発進 %d(比べに入る %d)・Σ A の最大 %.3f(比べに入る %.3f)・Σ H の最大 %.3f(比べに入る %.3f)・'
                 'Σ H > 0.12 の発進 %d・> 0.22 %d' % (len(allr), len(cmp_), max(x['sum_a'] for x in allr), max(x['sum_a'] for x in cmp_),
                                                   max(x['sum_h'] for x in allr), max(x['sum_h'] for x in cmp_),
                                                   sum(1 for x in allr if x['sum_h'] > 0.12), sum(1 for x in allr if x['sum_h'] > 0.22)))
    js['launches'] = allr
    # 2・3 窓
    W, WL = 3.0, 3.5
    res = {}
    for d in DAYS + ('all',):
        res[d] = {'win': {'n': 0, 'maxa': 0.0, 'maxh': 0.0, 'top': []}, 'lw': {'n': 0, 'maxa': 0.0, 'maxh': 0.0, 'top': []},
                  'holdrows': 0, 'rows': 0}
    cnt = {k: {'a>0.12': 0, 'h>0.12': 0, 'a>0.2': 0, 'h>0.2': 0, 'a>0.22': 0, 'h>0.22': 0} for k in DAYS + ('all',)}
    cntl = {k: {'a>0.12': 0, 'h>0.12': 0, 'a>0.22': 0, 'h>0.22': 0} for k in DAYS + ('all',)}
    nlogs = 0
    for p in sorted(LAB.glob('*.npz')):
        log = p.stem
        d = log[:8]
        if d not in DAYS:
            continue
        t, ai, h, v = load(log)
        if not ai.any():
            continue
        nlogs += 1
        for i0, i1 in segs(ai):
            tt, hh = t[i0:i1 + 1], h[i0:i1 + 1]
            for k in (d, 'all'):
                res[k]['holdrows'] += int(hh.sum())
                res[k]['rows'] += len(tt)
            good = tt[~hh]
            if len(good) < 2:
                continue
            # 2: 処理したフレームから始まる 3 秒の窓(終わりが区間の最後の処理したフレームより前)。いちばん長い間 ≤ 0.22 秒の窓だけ数える
            for a in good:
                b = a + W
                if b > good[-1]:
                    break
                s = sums(tt, hh, float(a), float(b))
                if s['max_gap'] > STALL:
                    continue
                for k in (d, 'all'):
                    R = res[k]['win']
                    R['n'] += 1
                    R['maxa'] = max(R['maxa'], s['sum_a'])
                    R['maxh'] = max(R['maxh'], s['sum_h'])
                    for nm, val in (('a', s['sum_a']), ('h', s['sum_h'])):
                        for thr in (0.12, 0.2, 0.22):
                            if val > thr + 1e-9:
                                cnt[k]['%s>%s' % (nm, thr)] += 1
                    if s['sum_h'] > 0.12 and len(R['top']) < 400:
                        R['top'].append((s['sum_h'], s['sum_a'], log, round(float(a), 2), s['max_gap'], s['holds']))
            # 3: 止まって始めた区間の始めから 3.5 秒
            if abs(v[i0]) >= 0.2:
                continue
            t0 = float(tt[0])
            te = min(t0 + WL, float(tt[-1]))
            if te - t0 < 1.0:
                continue
            s = sums(tt, hh, t0, te)
            for k in (d, 'all'):
                R = res[k]['lw']
                R['n'] += 1
                R['maxa'] = max(R['maxa'], s['sum_a'])
                R['maxh'] = max(R['maxh'], s['sum_h'])
                for nm, val in (('a', s['sum_a']), ('h', s['sum_h'])):
                    for thr in (0.12, 0.22):
                        if val > thr + 1e-9:
                            cntl[k]['%s>%s' % (nm, thr)] += 1
                if s['sum_h'] > 0.05 or s['max_gap'] > 0.1:
                    R['top'].append((s['sum_h'], s['sum_a'], log, round(t0, 2), s['max_gap'], s['holds']))
    lines.append('')
    lines.append('# 2 AI 側の 3.0 秒の窓(処理したフレームから始める。いちばん長い間 ≤ 0.22 秒の窓 = 1 回の止まりでは外れない窓)')
    for k in DAYS + ('all',):
        R = res[k]['win']
        lines.append(' %-8s 窓 %d・Σ A の最大 %.3f・Σ H の最大 %.3f・数 %s・中立止めの行 %d / %d 行' % (
            k, R['n'], R['maxa'], R['maxh'], cnt[k], res[k]['holdrows'], res[k]['rows']))
    top = sorted(res['all']['win']['top'], reverse=True)
    seen = []
    for x in top:
        if all(not (x[2] == y[2] and abs(x[3] - y[3]) < 3.0) for y in seen):
            seen.append(x)
        if len(seen) >= 10:
            break
    lines.append('  Σ H の大きい窓(3 秒より離れたもの 10 個まで。Σ H・Σ A・記録・始め・いちばん長い間・中立止めの行): %s' % seen)
    lines.append('')
    lines.append('# 3 止まって始めた AI の区間の始めから 3.5 秒(R1 に届いたかは見ない。区間が 1 秒より短いものは除く)')
    for k in DAYS + ('all',):
        R = res[k]['lw']
        lines.append(' %-8s 窓 %d・Σ A の最大 %.3f・Σ H の最大 %.3f・数 %s' % (k, R['n'], R['maxa'], R['maxh'], cntl[k]))
    for x in sorted(res['all']['lw']['top'], reverse=True)[:12]:
        lines.append('    Σ H %.3f・Σ A %.3f・%s@%.2f・いちばん長い間 %.3f・中立止めの行 %d' % x)
    lines.append('')
    lines.append('記録 %d(AI 側のある記録)' % nlogs)
    js['windows'] = {k: {'win': {kk: vv for kk, vv in res[k]['win'].items() if kk != 'top'},
                         'launch_win': {kk: vv for kk, vv in res[k]['lw'].items() if kk != 'top'},
                         'counts': cnt[k], 'counts_launch_win': cntl[k]} for k in res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'real_sum.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'real_sum.json').write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
