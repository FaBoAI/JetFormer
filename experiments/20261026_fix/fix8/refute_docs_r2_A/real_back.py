"""打ち消しの票 A: 本物の記録(data/raw の 10/10・10/23・10/24。読むだけ)の、AI 側のリカバリー(AI_CMD の assist 2)の始めごとに、
どれだけ下がったか(VEHICLE の eRPM を積分。v[m/s] = −eRPM × 0.5e-3 = 前へ正)と、始めから 0.25 秒('brake')の間に下がったか
(= 10/23 の形: ESC が後退を待っていて、最初の後退側ですぐ下がる)を出す。場面(scene)はつないで読む。
    python -I -B real_back.py → 標準出力・out/real_back.json
"""
import json
from pathlib import Path

import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
HERE = Path(__file__).resolve().parent
KV = 0.5e-3


def scenes_of(log):
    return sorted(p for p in (RAW / log).iterdir() if p.is_dir() and (p / 'AI_CMD.npz').exists()
                  and (p / 'VEHICLE.npz').exists())


def load(log):
    cat = {}
    for s in scenes_of(log):
        for k in ('AI_CMD', 'VEHICLE'):
            try:
                z = np.load(s / ('%s.npz' % k), allow_pickle=False)
                d = {n: z[n] for n in z.files if z[n].ndim > 0}
            except Exception:       # noqa: BLE001  0 バイト・壊れた npz は飛ばす
                d = None
            if d is None or 'timestamps_us' not in d:
                continue
            c = cat.setdefault(k, {})
            for n, v in d.items():
                c.setdefault(n, []).append(v)
    for k in cat:
        cat[k] = {n: np.concatenate(v) for n, v in cat[k].items() if len({x.shape[1:] for x in v}) == 1}
    return cat


def main():
    out = []
    for lg in sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name[:8] in ('20261010', '20261023', '20261024')):
        try:
            cat = load(lg)
        except Exception as e:     # noqa: BLE001
            print('skip', lg, e)
            continue
        if 'AI_CMD' not in cat or 'VEHICLE' not in cat or 'assist' not in cat['AI_CMD']:
            continue
        z, v = cat['AI_CMD'], cat['VEHICLE']
        o = np.argsort(z['timestamps_us'])
        ta = z['timestamps_us'][o].astype(np.int64)
        asg = z['assist'][o].astype(int)
        sel = z['select'][o].astype(bool)
        hold = z['hold'][o].astype(bool) if 'hold' in z else np.zeros(len(ta), bool)
        ov = np.argsort(v['timestamps_us'])
        tv = v['timestamps_us'][ov].astype(np.int64)
        er = v['erpm'][ov].astype(float)
        if len(tv) < 2:
            continue
        T0 = int(min(ta[0], tv[0]))
        tvs = (tv - T0) / 1e6
        vel = -er * KV                                   # 前へ正
        x = np.concatenate([[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(tvs))])
        rec = (asg == 2) & sel & ~hold
        starts = [i for i in range(len(ta)) if rec[i] and (i == 0 or not rec[i - 1])
                  and not (i > 0 and asg[i - 1] == 2 and sel[i - 1])]
        for i in starts:
            ts = (ta[i] - T0) / 1e6
            j = i
            while j + 1 < len(ta) and (asg[j + 1] == 2 or (hold[j + 1] and sel[j + 1])) and sel[j + 1]:
                j += 1
            te = (ta[j] - T0) / 1e6
            k_end = j + 1
            human_after = bool(k_end < len(ta) and not sel[k_end])
            m = (tvs >= ts) & (tvs <= te + 0.6)
            if m.sum() < 2:
                continue
            x0 = float(np.interp(ts, tvs, x))
            back = x0 - float(x[m].min())
            mb = (tvs >= ts) & (tvs <= ts + 0.25)
            back_in_brake = x0 - float(x[mb].min()) if mb.any() else 0.0
            vback = float(-vel[m].min())
            out.append({'log': lg, 't_rec': round(ts, 3), 't_end': round(te, 3), 'back_m': round(back, 3),
                        'back_in_first_0p25s_m': round(back_in_brake, 3), 'peak_back_mps': round(vback, 3),
                        'human_after': human_after})
    for r in out:
        print('%s rec %7.3f-%7.3f back %.3f m (first 0.25 s %.3f) peak %.2f m/s %s' % (
            r['log'], r['t_rec'], r['t_end'], r['back_m'], r['back_in_first_0p25s_m'], r['peak_back_mps'],
            '→ 人側' if r['human_after'] else ''))
    bk = sorted(r['back_m'] for r in out)
    print('リカバリーの始め', len(out), '下がった m: 最小 %.3f 中央 %.3f 最大 %.3f' % (bk[0], bk[len(bk) // 2], bk[-1]) if bk else '')
    print('1.0 m 以上', sum(b >= 1.0 for b in bk), '/', len(bk))
    print('10/23 の形(始め 0.25 秒で 0.05 m 以上下がった)', [(r['log'], r['t_rec'], r['back_m']) for r in out
                                                       if r['back_in_first_0p25s_m'] >= 0.05])
    (HERE / 'out' / 'real_back.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
