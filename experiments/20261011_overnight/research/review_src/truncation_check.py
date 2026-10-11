"""Review check: share of AI frames whose discounted return (gamma=0.99 @30 fps, ~3.3 s) or 1 s advantage window
is cut by the end of the AI segment (takeover / log end / board reboot). Read-only."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, seg_mask  # noqa
LOGS = ['20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
        '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
        '20261010-204605']
tot = 0; w1 = 0; w33 = 0; per = {}
for log in LOGS:
    d = L(log)
    segs = seg_mask(d.sel)
    n_ai = 0; a1 = 0; a33 = 0
    for i0, i1 in segs:
        rem = d.at[i1] - d.at[i0:i1 + 1]
        n_ai += len(rem); a1 += int((rem < 1.0).sum()); a33 += int((rem < 3.3).sum())
    per[d.name] = {'segments': len(segs), 'ai_frames': n_ai, 'lt1s': round(a1 / max(n_ai, 1), 3),
                   'lt3.3s': round(a33 / max(n_ai, 1), 3)}
    tot += n_ai; w1 += a1; w33 += a33
    del d
res = {'ai_frames': tot, 'share_within_1s_of_segment_end': round(w1 / tot, 3),
       'share_within_3.3s_of_segment_end': round(w33 / tot, 3), 'per_log': per}
json.dump(res, open('truncation_check.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
