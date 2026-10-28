"""反証 8-2 docs: 手順書 3・7 章・approvals 3 の「約 1 m より近い後ろの物には ON のほうが強く当たることがある / 約 1 m より遠い物には
ON のほうが遅く当たるか当たらない」の帯の境を、案の係の閉じたループの出力(fix8/proposal_r1/out/sweep_r1_*of2.json。読むだけ)で
距離ごとに数え直す(票 A・ana_r1.py と同じ分け方: 両方当たる組の速さの差が 0.1 m/s より大きいか・片方だけ当たるか)。

    python3 -I -B p2_band_by_d.py → out/p2_band_by_d.txt
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/proposal_r1/out')
TH = 0.1


def cls(a, b):
    if a is None and b is None:
        return 'neither'
    if b is None:
        return 'on_only'
    if a is None:
        return 'off_only'
    if a > b + TH:
        return 'on_faster'
    if b > a + TH:
        return 'off_faster'
    return 'same'


def main():
    rows, ds = [], None
    for f in sorted(SRC.glob('sweep_r1_*of2.json')):
        J = json.loads(f.read_text())
        ds = J['ds']
        rows += J['rows']
    lines = ['# 元: %s(%d 場面 × %d 距離 = %d 組)' % (', '.join(p.name for p in sorted(SRC.glob('sweep_r1_*of2.json'))),
                                                 len(rows), len(ds), len(rows) * len(ds)),
             '# 距離 d(止まった所から)ごとの 今の ON 対 今の OFF(288 場面 = 18 模型 × 16 場面)']
    tot = collections.Counter()
    for d in ds:
        c = collections.Counter(cls(r['now_on']['hit'][str(d)], r['now_off']['hit'][str(d)]) for r in rows)
        fit = collections.Counter(cls(r['now_on']['hit'][str(d)], r['now_off']['hit'][str(d)]) for r in rows if r['esc'] == 'fit')
        band = 'lt1' if d < 1.0 - 1e-9 else 'ge1'
        for k, v in c.items():
            tot[(band, k)] += v
        lines.append('d %.2f: ON が速い %3d・ON だけ %3d・OFF が速い %3d・OFF だけ %3d・同じ %3d・どちらも無し %3d | fit の 16 場面: ON が速い %2d・ON だけ %2d'
                     % (d, c['on_faster'], c['on_only'], c['off_faster'], c['off_only'], c['same'], c['neither'],
                        fit['on_faster'], fit['on_only']))
    lines.append('帯の合計: ' + json.dumps({'%s %s' % k: v for k, v in sorted(tot.items())}, ensure_ascii=False))
    worse = [d for d in ds if d >= 1.0 - 1e-9 and any(cls(r['now_on']['hit'][str(d)], r['now_off']['hit'][str(d)]) in ('on_faster', 'on_only') for r in rows)]
    lines.append('1.0 m 以上で ON が速い・ON だけ当たる組のある距離: %s' % worse)
    txt = '\n'.join(lines) + '\n'
    (HERE / 'out' / 'p2_band_by_d.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
