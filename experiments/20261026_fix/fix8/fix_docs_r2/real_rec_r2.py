"""fix8 係 docs 直す 8-2 回目: 本物の記録(data/raw の 10/10・10/23・10/24。読むだけ)の、AI 側のリカバリー(AI_CMD の assist 2)の
始めごとに、下がった距離・止まってから下がり始めるまでの秒・同じ本の中のリカバリーの数を出す(f8r2-docs-1023-first-recovery-off の
「ふつうの後退の幅」。refute_docs_r2_A/real_back.py と同じ読み方に、log.json の rev_s・rev_us と止まった時刻を足した)。
場面(scene)はつないで読む。v [m/s] = −eRPM × 0.5e-3(前へ正)。
  back_m    = 始めから(リカバリー・中立止めが続く間)の終わり + 0.6 秒までに、いちばん後ろへ行った距離(票 A と同じ)
  stop_t    = 始めの前で最後に前へ 0.15 m/s(Recovery.STOP_ERPM 300)より速かった時刻。noback_s = そこから、始めの位置より 0.05 m
              下がるまでの秒(始めから 5 秒の内に下がらなければ None。始めの前の跳ね返りは数えない)
  human_after = リカバリーの続きの次のフレームが人側(人が取った)
  giveup_frames = この本の AI_CMD の assist 3(中止)のフレームの数
    python -I -B real_rec_r2.py → 標準出力・out/real_rec_r2.json
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


def params_of(log):
    try:
        j = json.loads((RAW / log / 'log.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return (j.get('drive') or {}).get('params') or {}


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
        vel = -er * KV
        x = np.concatenate([[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(tvs))])
        rec = (asg == 2) & sel & ~hold
        starts = [i for i in range(len(ta)) if rec[i] and (i == 0 or not rec[i - 1])
                  and not (i > 0 and asg[i - 1] == 2 and sel[i - 1])]
        pr = params_of(lg)
        n_giveup = int(((asg == 3) & sel).sum())
        for i in starts:
            ts = (ta[i] - T0) / 1e6
            j = i
            while j + 1 < len(ta) and (asg[j + 1] == 2 or (hold[j + 1] and sel[j + 1])) and sel[j + 1]:
                j += 1
            te = (ta[j] - T0) / 1e6
            human_after = bool(j + 1 < len(ta) and not sel[j + 1])
            m = (tvs >= ts) & (tvs <= te + 0.6)
            if m.sum() < 2:
                continue
            x0 = float(np.interp(ts, tvs, x))
            back = x0 - float(x[m].min())
            fw = np.nonzero((tvs < ts) & (vel > 0.15))[0]
            stop_t = float(tvs[fw[-1]]) if len(fw) else None
            nb = None
            if stop_t is not None:
                k = np.nonzero((tvs >= ts) & (tvs <= ts + 5.0) & (x <= x0 - 0.05))[0]
                nb = round(float(tvs[k[0]]) - stop_t, 3) if len(k) else None
            out.append({'log': lg, 'rev_us': pr.get('rev_us'), 'rev_s': pr.get('rev_s'), 'rec_back_guard': pr.get('rec_back_guard'),
                        't_rec': round(ts, 3), 't_end': round(te, 3), 'back_m': round(back, 3),
                        'stop_t': None if stop_t is None else round(stop_t, 3), 'noback_s': nb,
                        'human_after': human_after, 'n_rec_in_log': len(starts), 'giveup_frames': n_giveup})
    for r in out:
        print('%s rev %s/%s rec %8.3f-%8.3f back %.3f m  stop→back %s s  %s  (この本のリカバリー %d・中止のフレーム %d)' % (
            r['log'], r['rev_us'], r['rev_s'], r['t_rec'], r['t_end'], r['back_m'], r['noback_s'],
            '→ 人側' if r['human_after'] else 'AI 側のまま', r['n_rec_in_log'], r['giveup_frames']))
    s300 = [r for r in out if r['rev_us'] == 300.0 and r['rev_s'] == 0.4]
    ai = [r for r in s300 if not r['human_after']]
    print('rev_s 0.4・rev_us 300 のリカバリーの始め', len(s300), '・AI 側のまま終わった', len(ai))
    for name, rr in (('全部', s300), ('AI 側のまま', ai)):
        bk = sorted(r['back_m'] for r in rr)
        nb = sorted(r['noback_s'] for r in rr if r['noback_s'] is not None)
        print(' %s: 下がった m 最小 %s 最大 %s / 止まってから下がるまで 秒 最小 %s 最大 %s' % (
            name, bk[0] if bk else None, bk[-1] if bk else None, nb[0] if nb else None, nb[-1] if nb else None))
    old = [r for r in out if r['rev_us'] != 300.0]
    print('rev_us が 300 でない(10/10 の前の既定)のリカバリー:', [(r['log'], r['t_rec'], r['back_m'], r['noback_s']) for r in old])
    print('中止(assist 3)のフレームのある本:', sorted({(r['log'], r['giveup_frames']) for r in out if r['giveup_frames']}))
    (HERE / 'out' / 'real_rec_r2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
