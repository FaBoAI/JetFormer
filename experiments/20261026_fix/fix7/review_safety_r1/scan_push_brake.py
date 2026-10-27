"""反証 7-1 回目・safety: 本物の記録(data/raw、読むだけ)で、リカバリーの 'brake' と同じ形
= 「止まったまま前進を出していた(壁を押していた)あとの後退側(このESCではブレーキ。下がれない)」の始めの 0.35 秒に、
eRPM が 300 を超えたサンプルがあるか(rec_back_guard の back_early が、ESC が下がれないのに 'gap' を飛ばす入り口)。
条件: 後退側の始めの前 0.5 秒の車両の値がどれも |eRPM| < 150(止まっている)で、その間に前進(< 中立 − 20)を 0.2 秒以上書いた。
ESC に届いた値は scan_brake_rollback と同じ(AI 側は AI_CMD の out、人側は pwm[1])。出力 out/scan_push_brake.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from scan_brake_rollback import ROOT, N, esc_input   # noqa: E402


def main():
    res = {'cases': 0, 'gt300': 0, 'gt300_ai': 0, 'cases_ai': 0, 'max_hist': {}, 'examples': []}
    hist = {}
    for vp in sorted(glob.glob(ROOT + '/2026*/*/VEHICLE.npz')):
        if os.path.getsize(vp) == 0:
            continue
        try:
            v = np.load(vp)
            ap = vp.replace('VEHICLE.npz', 'AI_CMD.npz')
            a = np.load(ap) if os.path.exists(ap) and os.path.getsize(ap) > 0 else None
            t, u, src = esc_input(v, a)
            e = v['erpm'].astype(int)
        except Exception:       # noqa: BLE001
            continue
        side = np.where(u > N + 40, 'r', np.where(u < N - 20, 'f', 'n'))
        n = len(t)
        for i in range(1, n):
            if side[i] != 'r' or side[i - 1] == 'r':
                continue
            # 前の中立でない値が前進
            k = i - 1
            while k >= 0 and side[k] == 'n':
                k -= 1
            if k < 0 or side[k] != 'f':
                continue
            w = (t >= t[i] - 500_000) & (t < t[i])
            if w.sum() < 10 or not (np.abs(e[w]) < 150).all():
                continue
            fw = w & (side == 'f')
            if fw.sum() == 0:
                continue
            idx = np.nonzero(fw)[0]
            # 前進を書いていた長さ(サンプルの間隔で)
            fwd_s = 0.0
            for q in idx:
                if q + 1 < n:
                    fwd_s += (t[q + 1] - t[q]) / 1e6
            if fwd_s < 0.2:
                continue
            b = (t >= t[i]) & (t < t[i] + 350_000)
            j = i
            while j < n and side[j] == 'r':
                j += 1
            b &= np.arange(n) < j                    # 後退側が続く間だけ
            if b.sum() == 0:
                continue
            mx = int(e[b].max())
            ai = bool(src[i])
            res['cases'] += 1
            res['cases_ai'] += int(ai)
            key = '<=0' if mx <= 0 else '%d-%d' % (mx // 100 * 100, mx // 100 * 100 + 99)
            hist[key] = hist.get(key, 0) + 1
            if mx > 300:
                res['gt300'] += 1
                res['gt300_ai'] += int(ai)
                tb = t[np.nonzero(b & (e > 300))[0][0]]
                res['examples'].append({'file': vp.replace(ROOT + '/', ''), 'i': int(i), 'ai': ai,
                                        'u_brake': float(u[i]), 'push_us': float(np.median(u[fw])),
                                        'fwd_s': round(fwd_s, 2), 'first_gt300_s': round((tb - t[i]) / 1e6, 3),
                                        'max_erpm': mx, 'n_gt300': int((e[b] > 300).sum()),
                                        'brake_len_s': round((t[j - 1] - t[i]) / 1e6, 3),
                                        'erpm': [int(x) for x in e[b]]})
    res['max_hist'] = dict(sorted(hist.items(), key=lambda kv: -1 if kv[0] == '<=0' else int(kv[0].split('-')[0])))
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_push_brake.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'examples'}, ensure_ascii=False))
    for x in res['examples']:
        print(x['file'], x['i'], 'ai' if x['ai'] else 'hu', 'push', x['push_us'], x['fwd_s'], 'first>300', x['first_gt300_s'],
              'max', x['max_erpm'], 'n', x['n_gt300'], 'brake', x['brake_len_s'], x['erpm'][:18])


if __name__ == '__main__':
    main()
