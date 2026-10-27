"""反証 1 回目・safety: 実車の記録で、AI 側の区間の中の「中立で止めた」フレームと、リカバリー中止(assist=3)のあとを数える(読むだけ)"""
import glob, json, os, sys, zipfile
import numpy as np

root = '/home/jetson/jetformer/data/raw'
days = sys.argv[1:] or ['202610']
out = {'logs': 0, 'scenes': 0, 'ai_frames': 0, 'ai_holds': 0, 'hold_reasons': {}, 'giveup_frames': 0,
       'giveup_then_hold_then_model': [], 'max_hold_gap_s': 0}
for d in sorted(glob.glob(root + '/2026*')):
    name = os.path.basename(d)
    if not any(name.startswith(x) for x in days):
        continue
    out['logs'] += 1
    for sc in sorted(glob.glob(d + '/*/AI_CMD.npz')):
        if os.path.getsize(sc) == 0:
            continue
        try:
            z = np.load(sc)
            keys = z.files
            hold = z['hold'].astype(bool); sel = z['select'].astype(bool); asst = z['assist'].astype(int)
            t = z['t_ns'] if 't_ns' in keys else z[keys[0]]
        except Exception as e:
            continue
        out['scenes'] += 1
        ai = sel & ~(z['dry_run'].astype(bool) if 'dry_run' in keys else False)
        out['ai_frames'] += int((ai & ~hold).sum())
        out['ai_holds'] += int((ai & hold).sum())
        out['giveup_frames'] += int((asst == 3).sum())
        # giveup のあと、select のまま hold を挟んで assist 0(モデルのまま)に戻った行
        n = len(hold)
        i = 0
        while i < n:
            if asst[i] == 3 and sel[i] and not hold[i]:
                j = i + 1
                while j < n and sel[j] and (asst[j] == 3 or hold[j]):
                    j += 1
                had_hold = bool(hold[i + 1:j].any())
                if j < n and sel[j] and not hold[j] and asst[j] != 3 and had_hold:
                    out['giveup_then_hold_then_model'].append({'scene': os.path.relpath(sc, root), 'row_giveup_last': int(j - 1),
                                                               'row_model': int(j),
                                                               'out_thr_after': [float(x) for x in z['out'][j:j + 5, 1]],
                                                               'assist_after': [int(x) for x in asst[j:j + 8]]})
                i = j
            else:
                i += 1
print(json.dumps(out, ensure_ascii=False, indent=1))
