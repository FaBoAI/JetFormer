"""反証 7-1 回目・safety: 本物の記録(data/raw、読むだけ)で「前進のあとの後退側(= このESCではブレーキ。下がれない)」の間に
eRPM が BACKING_ERPM(300)を超えて後ろへ動いたサンプルがあるか。rec_back_guard = 1 は 'brake'・'wait' の間に eRPM > 300 を
1 回見るだけで「ESC は後退を待っていた・もう下がっている」として 'gap'(中立)を飛ばし 'reverse' へ進む(back_early)。
ESC に届いた値: AI 側(select)は AI_CMD の out(撮像時刻の行。中立止めの行も)、人側は VEHICLE の pwm[1]。
数える区間: 後退側(> 中立 + 40)が続く間で、その前の中立でない値が前進(< 中立 − 20)= ブレーキの区間。
区間の中で一度 |eRPM| < 300(止まった)になったあとに eRPM > 300 になったら「止まってから後ろへ転がった」。
装置・GPU・API には触れない。出力 out/scan_brake_rollback.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = '/home/jetson/jetformer/data/raw'
N = 1496.0


def esc_input(v, a):
    t = v['timestamps_us'].astype(np.int64)
    sel = v['select'].astype(bool)
    u = v['pwm'][:, 1].astype(float)
    ok = v['valid'][:, 1].astype(bool)
    u = np.where(ok, u, N)
    src = np.zeros(len(t), np.int8)
    if a is not None and len(a['timestamps_us']):
        at = a['timestamps_us'].astype(np.int64)
        ao = a['out'][:, 1].astype(float)
        j = np.searchsorted(at, t, side='right') - 1
        has = j >= 0
        ai_u = np.where(has, ao[np.clip(j, 0, None)], N)
        u = np.where(sel, ai_u, u)
        src = np.where(sel, 1, 0).astype(np.int8)
    return t, u, src


def main():
    res = {'files': 0, 'brake_runs': 0, 'runs_stop_then_back': 0, 'runs_back_any': 0, 'ai_runs': 0, 'ai_stop_then_back': 0,
           'examples': [], 'max_after_stop_hist': {}}
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
        res['files'] += 1
        side = np.where(u > N + 40, 'r', np.where(u < N - 20, 'f', 'n'))
        last_nn = None
        i = 0
        n = len(t)
        while i < n:
            if side[i] == 'r' and last_nn == 'f':
                j = i
                while j < n and side[j] == 'r':
                    j += 1
                if t[j - 1] - t[i] >= 100_000:          # 0.1 秒以上続いたブレーキ
                    res['brake_runs'] += 1
                    seg = e[i:j]
                    ai = bool(src[i])
                    res['ai_runs'] += int(ai)
                    stopped = np.nonzero(np.abs(seg) < 300)[0]
                    if (seg > 300).any():
                        res['runs_back_any'] += 1
                    if len(stopped):
                        k0 = stopped[0]
                        after = seg[k0:]
                        mx = int(after.max())
                        b = '%d-%d' % (mx // 100 * 100, mx // 100 * 100 + 99) if mx > 0 else '<=0'
                        hist[b] = hist.get(b, 0) + 1
                        if mx > 300:
                            res['runs_stop_then_back'] += 1
                            res['ai_stop_then_back'] += int(ai)
                            if len(res['examples']) < 40:
                                k1 = k0 + int(np.argmax(after > 300))
                                res['examples'].append({'file': vp.replace(ROOT + '/', ''), 'i': int(i), 'ai': ai,
                                                        'brake_s': round((t[j - 1] - t[i]) / 1e6, 3),
                                                        't_stop_to_back_s': round((t[i + k1] - t[i + k0]) / 1e6, 3),
                                                        'max_back_erpm': mx, 'u': float(u[i]),
                                                        'erpm': [int(x) for x in seg[:40]]})
                i = j
                last_nn = 'r'
                continue
            if side[i] != 'n':
                last_nn = side[i]
            i += 1
    res['max_after_stop_hist'] = dict(sorted(hist.items(), key=lambda kv: (kv[0] != '<=0', int(kv[0].split('-')[0]) if kv[0] != '<=0' else 0)))
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'scan_brake_rollback.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'examples'}, ensure_ascii=False))
    for x in res['examples'][:15]:
        print(x['file'], x['i'], 'ai' if x['ai'] else 'hu', x['brake_s'], x['t_stop_to_back_s'], x['max_back_erpm'], x['erpm'][:25])


if __name__ == '__main__':
    main()
