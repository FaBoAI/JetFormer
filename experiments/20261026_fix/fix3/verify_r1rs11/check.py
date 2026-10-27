"""指摘 r1-rs1-1 の打ち消しの確かめ(読むだけ・回すだけ。直さない)。場面 A(probe_refute1.scene_a)を
既定(rec_back_guard 0・rec_hold_keep 0)・見張りだけ・両方 で、中立止めあり/なしで比べる。
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/verify_r1rs11/check.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute1_safety'))
import probe_refute1 as P   # noqa: E402

P.TMP = HERE.parent / 'tmp_verify_r1rs11'


def rev_after(frames, rec, hit):
    rows = P.tl(frames, rec)
    after = [x for x in rows if x[0] >= hit and not x[2] and x[3] > P.N + 40.0]
    before_hit_rev = [x for x in rows if 2.0 <= x[0] < hit and not x[2] and x[3] > P.N + 40.0]
    return {'rev_s_after_hit': round(len(after) / 30.0, 2), 'us': sorted({x[3] for x in after}),
            'first_last': [after[0][0], after[-1][0]] if after else None,
            'rev_rows_before_hit': len(before_hit_rev),
            'guard': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}


out = {}
hit = 3.25
for drop_name, drop in (('no_hold', None), ('camdrop_0.25', (3.23, 3.48)), ('camdrop_0.134', (3.23, 3.364))):
    for keep, guard in ((0, 0), (0, 1), (1, 1)):
        frames, imu = P.scene_a(hit_at=hit, drop=drop)
        d, pca, rec = P.run(dict(P.PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
        out['%s keep=%d guard=%d' % (drop_name, keep, guard)] = rev_after(frames, rec, hit)
# 後ろの当たりが「後退(reverse)」段の途中(下がり始めて 0.3 秒後)で、中立止めなし・既定のとき
for hit2 in (3.6, 3.9):
    for keep, guard in ((0, 0), (1, 1)):
        for drop_name, drop in (('no_hold', None), ('camdrop_0.25', (hit2 - 0.02, hit2 + 0.23))):
            frames, imu = P.scene_a(hit_at=hit2, drop=drop)
            d, pca, rec = P.run(dict(P.PARAMS, recovery=1, rec_back_guard=guard, rec_hold_keep=keep), frames, imu)
            out['hit=%.2f %s keep=%d guard=%d' % (hit2, drop_name, keep, guard)] = rev_after(frames, rec, hit2)
print(json.dumps(out, ensure_ascii=False, indent=1))
(HERE / 'out.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
