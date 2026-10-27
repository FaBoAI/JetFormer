"""X8: 新しい 3 つ(rec_back_guard・rec_hold_keep・launch_no_kick)が 0 のとき、PCA9685 に書く値(時刻つき)と記録の行が
fix2 の前の写し(tests/fixtures/round2_before)とビット単位で同じか。閉じたループ・でたらめな場面"""
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'refute2_safety'))
import cl3                                    # noqa: E402
from cl3 import C, N, PARAMS                  # noqa: E402
import fuzz_cl as F                           # noqa: E402

BMOD, BAMOD = cl3.load_fixture('round2_before')
KEYS_SKIP = set()


def rows_equal(a, b):
    if len(a) != len(b):
        return False, 'len %d %d' % (len(a), len(b))
    for i, ((ta, ra), (tb, rb)) in enumerate(zip(a, b)):
        if ta != tb:
            return False, 'row %d t' % i
        for k in rb:
            if k == 'infer_ms':
                continue
            va, vb = ra.get(k), rb[k]
            if isinstance(vb, np.ndarray) or isinstance(va, np.ndarray):
                if not np.array_equal(np.asarray(va), np.asarray(vb), equal_nan=True):
                    return False, 'row %d %s' % (i, k)
            elif isinstance(vb, float) and isinstance(va, float) and np.isnan(va) and np.isnan(vb):
                continue
            elif isinstance(vb, tuple) or isinstance(va, tuple):
                if not np.array_equal(np.asarray(va, float), np.asarray(vb, float), equal_nan=True):
                    return False, 'row %d %s' % (i, k)
            elif va != vb:
                return False, 'row %d %s %r %r' % (i, k, va, vb)
    return True, ''


def main(n=40, seed=27):
    rng = random.Random(seed)
    bad = []
    for i in range(n):
        sc_ = F.scene_of(rng)
        p = F.params_of(rng)
        p.update(rec_back_guard=0, rec_hold_keep=0, launch_no_kick=0)
        if i % 3 == 0:
            p['rec_hold_keep'] = 1          # 見張り 0 のときは keep 1 でも残さない(走行ループは両方 1 のときだけ)
        sim1, rec1, pca1, _ = cl3.run(p, sc_)
        pb = {k: v for k, v in p.items() if k in BMOD.PARAM_LIMITS or k in BMOD.DEFAULT_PARAMS}
        sim2, rec2, pca2, _ = cl3.run(pb, sc_, mod=BMOD, amod=BAMOD)
        same_w = pca1.w == pca2.w
        same_r, why = rows_equal(rec1.rows, rec2.rows)
        ev1 = [(k, {kk: vv for kk, vv in d.items() if kk != 'cam_epoch_us'}) for k, d in rec1.events]
        ev2 = [(k, {kk: vv for kk, vv in d.items()}) for k, d in rec2.events]
        if ev1 != ev2:
            same_r, why = False, 'events %s / %s' % (ev1[:3], ev2[:3])
        if not (same_w and same_r):
            bad.append({'i': i, 'w': same_w, 'rows': why, 'params': {k: p[k] for k in ('start_kick', 'launch_boost', 'tof_guard', 'speed_ceiling', 'zone_caps', 'rec_hold_keep')}})
    print(json.dumps({'n': n, 'bad': bad}, ensure_ascii=False, indent=1))
    shutil.rmtree(cl3.TMP, ignore_errors=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
