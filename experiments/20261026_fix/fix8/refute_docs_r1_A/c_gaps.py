"""打ち消しの票 A(f8r1-docs-approvals3-old-row の後半): 10/23・10/24 の AI 走行の記録(data/raw。読むだけ)で、
フレームの「間」をいくつかの数え方で数え、45,844(approvals 67 行。fix3 r3-1 の数)と 45,918(72 行。fix8 safety-decided)が
同じものの 2 つの数か、数え方の違いかを見る。0.5 秒より長い間の数も出す。"""
import json
from pathlib import Path

import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')


def load(root):
    out = []
    for s in sorted(d for d in root.iterdir() if d.is_dir()):
        p = s / 'AI_CMD.npz'
        if not p.exists():
            continue
        try:
            z = np.load(p)
            t = z['timestamps_us'].astype(np.int64)
            sel = z['select'].astype(bool)
            dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(t), bool)
            hold = z['hold'].astype(bool)
        except Exception as e:  # noqa: BLE001
            print('  skip', p, e)
            continue
        out.append((t, sel & ~dry, hold, sel))
    return out


res = {}
for day in ('20261023', '20261024'):
    cnt = {k: [0, 0] for k in ('A_proc_pairs_allsel', 'B_sel_rows_consec', 'C_proc_consec_any', 'D_proc_per_scene',
                               'E_all_rows_per_log', 'F_sel_rows_per_scene')}
    nlogs = 0
    for root in sorted(RAW.glob(day + '-*')):
        if not root.is_dir():
            continue
        parts = load(root)
        if not parts:
            continue
        t = np.concatenate([p[0] for p in parts])
        sel = np.concatenate([p[1] for p in parts])
        hold = np.concatenate([p[2] for p in parts])
        o = np.argsort(t, kind='stable')
        t, sel, hold = t[o], sel[o], hold[o]
        if not sel.any():
            continue
        nlogs += 1

        def add(key, gaps):
            gaps = np.asarray(gaps, float)
            cnt[key][0] += len(gaps)
            cnt[key][1] += int((gaps > 0.5).sum())
        proc = np.nonzero(sel & ~hold)[0]
        # A: scan_gaps10 と同じ(処理したフレームどうし・間の行が全部 AI 側)
        add('A_proc_pairs_allsel', [(t[b] - t[a]) / 1e6 for a, b in zip(proc, proc[1:]) if sel[a:b + 1].all()])
        # B: AI 側の行(中立止めも含む)の連続する 2 行(間に人側の行なし)
        sidx = np.nonzero(sel)[0]
        add('B_sel_rows_consec', [(t[b] - t[a]) / 1e6 for a, b in zip(sidx, sidx[1:]) if b == a + 1])
        # C: 処理したフレームどうし(間に人側があっても)
        add('C_proc_consec_any', np.diff(t[proc]) / 1e6)
        # D: シーンごとの処理したフレームどうし(シーンをまたがない。間が全部 AI 側)
        for (ts, sl, hd, _) in parts:
            pr = np.nonzero(sl & ~hd)[0]
            add('D_proc_per_scene', [(ts[b] - ts[a]) / 1e6 for a, b in zip(pr, pr[1:]) if sl[a:b + 1].all()])
            si = np.nonzero(sl)[0]
            add('F_sel_rows_per_scene', [(ts[b] - ts[a]) / 1e6 for a, b in zip(si, si[1:]) if b == a + 1])
        # E: その記録の全部の行(AI 側の行がある記録)
        add('E_all_rows_per_log', np.diff(t) / 1e6)
    res[day] = {'logs_with_ai': nlogs, **{k: {'n': v[0], 'gt0.5': v[1]} for k, v in cnt.items()}}

print(json.dumps(res, ensure_ascii=False, indent=1))
for k in res['20261023']:
    if k == 'logs_with_ai':
        continue
    print('%-24s 10/23+10/24 = %d (gt0.5 = %d)' % (k, res['20261023'][k]['n'] + res['20261024'][k]['n'],
                                                   res['20261023'][k]['gt0.5'] + res['20261024'][k]['gt0.5']))
