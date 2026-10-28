"""打ち消しの票 B(反証 8-2 回目・係 tools の 2 件。読むだけ・直さない)の台本 1: 本物の 10/10・10/23・10/24 のカメラの止まりの分布。

data/labels/<記録>.npz(読むだけ)の AI 側(ai かつ dry_run でない)の、処理したフレーム(中立止めの行 hold を除く)どうしの間を、
⑤ collect.launch_stall と同じ数え方(中立止めの行はフレームに数えない)で数える。

  A  AI 側の全部のフレームの間の分布(0.05 秒ごと)・0.22 秒より長い間の一覧(そのときの速さ・前後 1.5 秒の衝突 2.5G 以上)
  B  3.0 秒の窓(発進 → R1 の起点は本物で約 2.9〜3.3 秒)を AI 側の各フレームから始めて滑らせ、
       stall  窓の中のいちばん長い間 > 0.22 秒(8-2 指摘 1 の外れる形)
       burst  いちばん長い間 ≤ 0.22 秒で Σ max(0, 間 − 0.1) > 0.12 秒(指摘 2 の形 = 外す 1 回の止まりと同じ中立の量)
       b10    同じく Σ > 0.10 秒
     の窓の割合(全部の窓 / 走っている(始めの速さ ≥ 0.3 m/s)・窓の前後 1 秒に 2.5G 以上の衝突が無い窓 = 発進の窓に近い)
  C  AI 側にした時刻(AI の区間の始め)から 3.5 秒の窓(本物の発進の窓。R1 に届いたかは見ない = 届かない発進も入る)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/refute_tools_r2_B/b1_real_gaps.py
  → out/b1_real_gaps.txt・out/b1_real_gaps.json
"""
import json
from pathlib import Path

import numpy as np

LAB = Path('/home/jetson/jetformer/data/labels')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
STALL, NOTE, W, WL = 0.22, 0.15, 3.0, 3.5


def segs(mask):
    """True の続く区間 [(i0, i1)](両端を含む)"""
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


def run(DAYS, title, sig):
    lines = ['## ' + title]
    allg = []            # (log, t, gap, v, coll)
    ai_time = 0.0
    nseg = 0
    win = {k: {'n': 0, 'stall': 0, 'burst': 0, 'b10': 0} for k in ('all', 'moving_nocoll')}
    win_ex = {'stall': [], 'burst': [], 'b10': []}
    lw = {'n': 0, 'stall': 0, 'burst': 0, 'b10': 0, 'note': 0, 'ex': []}
    nlogs = 0
    for p in sorted(LAB.glob('*.npz')):
        log = p.stem
        if log[:8] not in DAYS:
            continue
        z = np.load(p)
        t = np.asarray(z['t_s'], np.float64)
        ai = np.asarray(z['ai'], bool)
        if 'dry_run' in z.files:
            ai &= ~np.asarray(z['dry_run'], bool)
        if not ai.any():
            continue
        nlogs += 1
        h = np.asarray(z['hold'], bool) if 'hold' in z.files else np.zeros(len(t), bool)
        v = np.asarray(z['speed_mps'], np.float64)
        cg = np.asarray(z['collision_g'], np.float64) if 'collision_g' in z.files else np.zeros(len(t))
        coll_t = t[(cg >= 2.5)]
        for i0, i1 in segs(ai):
            nseg += 1
            tt = t[i0:i1 + 1]
            hh = h[i0:i1 + 1]
            vv = v[i0:i1 + 1]
            ai_time += float(tt[-1] - tt[0])
            tp = tt[~hh]
            vp = vv[~hh]
            if len(tp) < 2:
                continue
            g = np.diff(tp)
            for k in range(len(g)):
                near = bool(np.any(np.abs(coll_t - tp[k]) <= 1.5))
                allg.append((log, float(tp[k]), float(g[k]), float(vp[k]), near))
            # B 滑らせる窓(AI 側の処理したフレームから始める。窓の終わりが区間の中)
            neu = np.clip(g - 0.1, 0, None)
            for k in range(len(tp)):
                a = tp[k]
                b = a + W
                if b > tp[-1]:
                    break
                # 窓の中の間 g[k:j](tp[k]〜tp[j])+ 端: 窓の終わりをまたぐ間は窓の終わりまで(launch_stall と同じく t_end を点に入れる)
                j = int(np.searchsorted(tp, b + 1e-9, side='right')) - 1
                gg = list(g[k:j])
                if j < len(tp) - 1 and b - tp[j] > 0:
                    gg.append(b - tp[j])
                gg = np.asarray(gg) if gg else np.zeros(1)
                mx = float(gg.max())
                sn = float(np.clip(gg - 0.1, 0, None).sum())
                moving = vp[k] >= 0.3
                if mx <= STALL:
                    sig.append((sn, log, round(a, 2), round(mx, 3)))
                nocoll = not bool(np.any((coll_t >= a - 1.0) & (coll_t <= b + 1.0)))
                for key, ok in (('all', True), ('moving_nocoll', moving and nocoll)):
                    if not ok:
                        continue
                    win[key]['n'] += 1
                    if mx > STALL:
                        win[key]['stall'] += 1
                    elif sn > 0.12:
                        win[key]['burst'] += 1
                    if mx <= STALL and sn > 0.10:
                        win[key]['b10'] += 1
                if moving and nocoll:
                    if mx > STALL and len(win_ex['stall']) < 400:
                        win_ex['stall'].append((log, round(a, 2), round(mx, 3), round(sn, 3)))
                    elif mx <= STALL and sn > 0.12 and len(win_ex['burst']) < 400:
                        win_ex['burst'].append((log, round(a, 2), round(mx, 3), round(sn, 3)))
                    if mx <= STALL and sn > 0.10 and len(win_ex['b10']) < 400:
                        win_ex['b10'].append((log, round(a, 2), round(mx, 3), round(sn, 3)))
            # C AI 側にした時刻から 3.5 秒(端の t0 と窓の終わりを点に入れる = launch_stall と同じ)
            t0 = float(tt[0])
            te = min(t0 + WL, float(tt[-1]))
            if te - t0 < 1.0:
                continue
            q = tp[(tp > t0) & (tp < te)]
            pts = np.concatenate([[t0], q, [te]])
            gg = np.diff(pts)
            mx = float(gg.max())
            sn = float(np.clip(gg - 0.1, 0, None).sum())
            lw['n'] += 1
            lw['stall'] += int(mx > STALL)
            lw['note'] += int(NOTE < mx <= STALL)
            lw['burst'] += int(mx <= STALL and sn > 0.12)
            lw['b10'] += int(mx <= STALL and sn > 0.10)
            if mx > 0.10 or sn > 0.05:
                lw['ex'].append((log, round(t0, 2), round(te - t0, 2), round(mx, 3), round(sn, 3)))
    gaps = np.asarray([x[2] for x in allg])
    lines.append('# A AI 側の処理したフレーム(中立止めの行を除く)どうしの間(10/10・10/23・10/24、記録 %d・AI の区間 %d・AI 側 %.1f 秒)'
                 % (nlogs, nseg, ai_time))
    lines.append('  間の数 %d' % len(gaps))
    edges = [0, 0.05, 0.10, 0.134, 0.15, 0.167, 0.20, 0.201, 0.22, 0.30, 0.50, 1.0, 1e9]
    for a, b in zip(edges[:-1], edges[1:]):
        lines.append('  (%.3f, %.3f] %d' % (a, b, int(np.sum((gaps > a) & (gaps <= b)))))
    big = [x for x in allg if x[2] > STALL]
    lines.append('  0.22 秒より長い間 %d(走っている = 始めの速さ ≥ 0.3 m/s: %d・前後 1.5 秒に 2.5G 以上の衝突: %d・走っていて衝突なし: %d)' % (
        len(big), sum(x[3] >= 0.3 for x in big), sum(x[4] for x in big), sum(x[3] >= 0.3 and not x[4] for x in big)))
    for x in big:
        lines.append('    %s %.2f 秒 間 %.3f 速さ %.2f m/s 衝突の近く %s' % x)
    mid = [x for x in allg if NOTE < x[2] <= STALL]
    lines.append('  0.15〜0.22 秒の間 %d(走っていて衝突なし: %d)' % (len(mid), sum(x[3] >= 0.3 and not x[4] for x in mid)))
    for x in mid:
        lines.append('    %s %.2f 秒 間 %.3f 速さ %.2f m/s 衝突の近く %s' % x)
    lines.append('  1 分あたり: 0.22 秒より長い間 %.3f 回・0.15〜0.22 秒 %.3f 回(AI 側 %.1f 分)' % (
        len(big) / (ai_time / 60), len(mid) / (ai_time / 60), ai_time / 60))
    lines.append('')
    lines.append('# B 3.0 秒の窓(AI 側の各フレームから)')
    for key, w in win.items():
        lines.append('  %-14s 窓 %d・stall(> 0.22 秒)%d(%.4f)・burst(≤ 0.22 秒で中立 Σ > 0.12)%d(%.4f)・Σ > 0.10 %d(%.4f)' % (
            key, w['n'], w['stall'], w['stall'] / max(1, w['n']), w['burst'], w['burst'] / max(1, w['n']), w['b10'],
            w['b10'] / max(1, w['n'])))
    for k in ('stall', 'burst', 'b10'):
        ex = win_ex[k]
        # 窓の始めが 0.5 秒より離れたものを別の出来事に数える
        ev = []
        for x in ex:
            if not ev or ev[-1][0] != x[0] or x[1] - ev[-1][1] > 0.5:
                ev.append([x[0], x[1], x[2], x[3]])
            else:
                ev[-1][1] = x[1]
                ev[-1][2] = max(ev[-1][2], x[2])
                ev[-1][3] = max(ev[-1][3], x[3])
        lines.append('  走っていて衝突なしの窓の %s の出来事 %d: %s' % (k, len(ev), ev[:30]))
    lines.append('')
    lines.append('# C AI 側にした時刻から 3.5 秒の窓(本物の発進の窓。R1 に届かない発進も入る)')
    lines.append('  発進 %d・stall(> 0.22 秒)%d・0.15〜0.22 秒 %d・burst(≤ 0.22 秒で中立 Σ > 0.12)%d・Σ > 0.10 %d' % (
        lw['n'], lw['stall'], lw['note'], lw['burst'], lw['b10']))
    for x in lw['ex']:
        lines.append('    %s@%.2f 窓 %.2f 秒・いちばん長い間 %.3f・中立 Σ %.3f' % x)
    sv = np.asarray([x[0] for x in sig]) if sig else np.zeros(1)
    lines.append('# D いちばん長い間 ≤ 0.22 秒の 3.0 秒の窓 %d の中立の見積もり Σ max(0, 間 − 0.1): 最大 %.3f・Σ > 0.12 %d・≥ 0.15 %d・≥ 0.20 %d・≥ 0.30 %d'
                 % (len(sig), float(sv.max()), int(np.sum(sv > 0.12)), int(np.sum(sv >= 0.15 - 1e-9)), int(np.sum(sv >= 0.2 - 1e-9)),
                    int(np.sum(sv >= 0.3 - 1e-9))))
    s10 = [x for x in sig if x[0] > 0.10 + 1e-9]
    lines.append('  Σ > 0.10 秒の窓 %d のいちばん長い間の最小 %s 秒(0.15 秒より長い = ⑤ の「注意: … 0.15〜0.22 秒の発進」の行が出る間を含む窓 %d)'
                 % (len(s10), ('%.3f' % min(x[3] for x in s10)) if s10 else '—', sum(x[3] > 0.15 for x in s10)))
    top = sorted(sig, key=lambda x: -x[0])
    seen = []
    for x in top:
        if any(y[1] == x[1] and abs(y[2] - x[2]) < 3.0 for y in seen):
            continue
        seen.append(x)
        if len(seen) >= 8:
            break
    lines.append('  Σ の大きい窓(3 秒より離れたもの 8 つ): %s' % [(round(x[0], 3), x[1], x[2], x[3]) for x in seen])
    lines.append('')
    return lines, {'ai_time_s': ai_time, 'n_gaps': len(gaps), 'big': big, 'mid': mid, 'win': win,
                   'launch_win': {k: v for k, v in lw.items() if k != 'ex'}, 'sig_max': float(sv.max())}


def main():
    lines, js = [], {}
    for days, title in ((('20261010', '20261023', '20261024'), '10/10・10/23・10/24 の全部'), (('20261010',), '10/10'),
                        (('20261023',), '10/23'), (('20261024',), '10/24'), (('20261023', '20261024'), '10/23・10/24')):
        ln, j = run(days, title, [])
        lines += ln
        js[title] = j
    OUT.mkdir(parents=True, exist_ok=True)
    txt = '\n'.join(lines) + '\n'
    (OUT / 'b1_real_gaps.txt').write_text(txt, encoding='utf-8')
    (OUT / 'b1_real_gaps.json').write_text(json.dumps(js, ensure_ascii=False), encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
