"""打ち消しの票 B の台本 2: 11/3 の回 3(OFF・ON・OFF・ON・OFF・ON・OFF・ON の 8 本・--ignore-gain で 4 本対 4 本)の
「ON − OFF ≤ −0.15 秒 → 使ってよい」を、本物の止まりの割合と本物の起点までのばらつきで何度も引いて、
  old  8-1 の直す前(止まった発進も起点までが伸びたまま比べに入る)
  new  今(0.22 秒より長い止まりの発進は外す。ON に 1 本でもあれば満たさない(on_no_stall)・OFF が全部なら満たさない(off_no_stall))
  base 止まりが無い日(はかりのぶれだけ)
の「使ってよい」の割合を比べる。ほかの条件(衝突・空転・LH)は満たすとする(どれも ON の発進で決まり、OFF を外しても ON の値は変わらない)。

入力(読むだけ・自分で流した出力):
  out/b1_real_gaps.json   本物の止まりの割合(走っていて衝突の無い 3.0 秒の窓のうち 0.22 秒より長い間を含む割合)と止まりの長さ
  out/real_2026102{3,4}.json  本物の ⑤(--ignore-gain)の R1 に届いた発進の起点まで(同じ回・同じモデル・同じゲインの組で 4 本以上)

止まりで伸びる起点まで(OFF・ON とも): fix8/fix_tools_r1.md 3.2 の表(作り物の本物に近い速さ。1.6 秒からの止まり)の
  間 0.233 → +0.03・0.300 → +0.05・0.333 → +0.06・0.433 → +0.10・0.933 → +0.39 秒 を線でつなぎ、止まりの時刻の重み f ~ U(0.2, 1.0)
  (2 m/s に届いたあとは小さい = 8-2 指摘の 2.30 秒からは +0.01)を掛ける。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I -B experiments/20261026_fix/fix8/refute_tools_r2_B/b2_mc.py
  → out/b2_mc.txt
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
N = 200000
THR = -0.15
INF_G = np.array([0.22, 0.233, 0.300, 0.333, 0.433, 0.933, 3.0])
INF_D = np.array([0.0, 0.03, 0.05, 0.06, 0.10, 0.39, 1.6])


def residuals():
    res, groups = [], []
    for d in ('20261023', '20261024'):
        j = json.loads((OUT / ('real_%s.json' % d)).read_text(encoding='utf-8'))
        by = {}
        for r in j['launches']:
            if r.get('to_r1_s') is None or r.get('midrun') or r.get('trial') or r.get('after_abort'):
                continue
            by.setdefault((d, r.get('model'), r.get('gain')), []).append(float(r['to_r1_s']))
        for k, v in by.items():
            if len(v) >= 4:
                m = float(np.median(v))
                rr = [x - m for x in v]
                groups.append((k, [round(x, 3) for x in v], round(float(np.std(v, ddof=1)), 3)))
                res += rr
    return np.asarray(res), groups


def draw_noise(rng, kind, shape, res):
    if kind == 'normal0.15':
        return rng.normal(0.0, 0.15, shape)
    if kind == 'boot_trim':        # 本物の残差(|残差| < 1 秒)から引く
        r = res[np.abs(res) < 1.0]
        return rng.choice(r - np.median(r), size=shape)
    if kind == 'boot_all':         # 本物の残差(外れ値も入れる)
        return rng.choice(res - np.median(res), size=shape)
    raise ValueError(kind)


def sim(rng, delta, p, gaps, kind, res):
    off = 3.0 + draw_noise(rng, kind, (N, 4), res)
    on = 3.0 + delta + draw_noise(rng, kind, (N, 4), res)
    s_off = rng.random((N, 4)) < p
    s_on = rng.random((N, 4)) < p
    g_off = rng.choice(gaps, size=(N, 4))
    g_on = rng.choice(gaps, size=(N, 4))
    f_off = rng.uniform(0.2, 1.0, (N, 4))
    f_on = rng.uniform(0.2, 1.0, (N, 4))
    off_inf = off + np.where(s_off, np.interp(g_off, INF_G, INF_D) * f_off, 0.0)
    on_inf = on + np.where(s_on, np.interp(g_on, INF_G, INF_D) * f_on, 0.0)
    base = (np.median(on, 1) - np.median(off, 1)) <= THR + 1e-9
    old = (np.median(on_inf, 1) - np.median(off_inf, 1)) <= THR + 1e-9
    # new: 止まった発進を外す
    offm = np.where(s_off, np.nan, off)
    n_off = (~s_off).sum(1)
    with np.errstate(all='ignore'):
        med_off = np.nanmedian(offm, 1)
    any_on = s_on.any(1)
    new = (~any_on) & (n_off > 0) & ((np.median(on, 1) - med_off) <= THR + 1e-9)
    two = (s_off.sum(1) >= 2) & (~any_on)
    return {'base': base.mean(), 'old': old.mean(), 'new': new.mean(),
            'flip_up': (new & ~old).mean(), 'flip_down': (~new & old).mean(),
            'flip_up_two': (new & ~old & two).mean(), 'two': two.mean(),
            'new_off_le2_ok': (new & (n_off <= 2)).mean()}


def main():
    rng = np.random.default_rng(20261103)
    b1 = json.loads((OUT / 'b1_real_gaps.json').read_text(encoding='utf-8'))
    res, groups = residuals()
    lines = ['# 本物の起点までのばらつき(10/23・10/24 の ⑤ --ignore-gain。同じ日・同じモデル・同じゲインで R1 に届いた発進 4 本以上の組。'
             '途中で変えた・試し・走り直しを除く)']
    for k, v, sd in groups:
        lines.append('  %s n %d 起点まで %s・標準偏差 %.3f' % (k, len(v), v, sd))
    lines.append('  残差(組の中央値から)%d: %s' % (len(res), [round(float(x), 3) for x in sorted(res)]))
    rates = {}
    for key, title in (('10/23・10/24', '10/23・10/24'), ('10/10・10/23・10/24 の全部', '全部'), ('10/10', '10/10')):
        w = b1[key]['win']['moving_nocoll']
        p = w['stall'] / w['n']
        gaps = [x[2] for x in b1[key]['big'] if x[3] >= 0.3 and not x[4]]
        rates[title] = (p, gaps)
        lines.append('  止まりの割合 %s: 走っていて衝突の無い 3.0 秒の窓 %d のうち 0.22 秒より長い間を含む %d = p %.4f・止まりの長さ %s'
                     % (title, w['n'], w['stall'], p, sorted(round(g, 3) for g in gaps)))
    lines.append('')
    deltas = (-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05)
    for kind in ('normal0.15', 'boot_trim', 'boot_all'):
        lines.append('# 雑音 %s(N %d)。「使ってよい」の割合 base / old / new、new だけ「使ってよい」(old は満たさない)= flip↑・'
                     'うち OFF が 2 本以上外れて ON は外れない形 = flip↑2、old だけ「使ってよい」= flip↓' % (kind, N))
        for title, (p, gaps) in rates.items():
            lines.append(' p %.4f(%s)・OFF 2 本以上止まり かつ ON 止まりなし の日の割合 %s' % (
                p, title, '%.5f' % sim(np.random.default_rng(1), -0.10, p, gaps, kind, res)['two']))
            for dl in deltas:
                r = sim(rng, dl, p, gaps, kind, res)
                lines.append('   真の ON − OFF %+.2f: base %.4f old %.4f new %.4f | flip↑ %.5f(flip↑2 %.5f)flip↓ %.5f | new − old %+.5f'
                             % (dl, r['base'], r['old'], r['new'], r['flip_up'], r['flip_up_two'], r['flip_down'],
                                r['new'] - r['old']))
        lines.append('')
    OUT.mkdir(parents=True, exist_ok=True)
    txt = '\n'.join(lines) + '\n'
    (OUT / 'b2_mc.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
