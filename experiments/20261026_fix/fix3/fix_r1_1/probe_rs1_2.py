"""r1-rs1-2 の直しの確かめ(本物の drive._loop。偽物のセンサー。反証の台本 scratchpad/probe_loop.py と同じ場面)。
人側 1 秒 → AI 側(2 m/s)→ 2.0 秒に 3G で当たって止まる・モデルは中立 − 30。リカバリー 3 回のあと中止。
走行中に set_params で launch_no_kick 1 → 0・start_kick 0 → 1 を送ったときの StartKick(1416 µs)のフレームを数える。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/fix_r1_1/probe_rs1_2.py
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from jetformer import assist as A, drive          # noqa: E402
from test_launch_drive import C, N, PARAMS, build, run    # noqa: E402

TMP = HERE.parent / 'tmp_fix_r1_1_probe2'


def scene():
    return build(16.0, sel=lambda s: s >= 1.0, v=lambda s: 2.0 if s < 2.0 else 0.0,
                 ax=lambda s: 30.0 if 1.99 <= s < 2.03 else 0.0, pred=(C, N - 30.0))


def go(params, hook_s=None, change=None, tag='x'):
    frames, imu = scene()
    flag, res = {'on': False}, {}

    def hook(i, d):
        if hook_s is not None and 'done' not in res and frames[i]['s'] >= hook_s:
            flag['on'] = True
            try:
                res['p'] = d.set_params(change)
            except ValueError as e:
                res['err'] = str(e)
            flag['on'] = False
            res['done'] = frames[i]['s']
    tp = TMP / tag
    tp.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(drive.Driver, 'running', property(lambda self: flag['on']))
        d, pca, rec = run(drive, mp, tp, params, frames, imu, hook=hook)
    rows = [(round((t - frames[0]['cam_t']) / 1e9 + frames[0]['s'], 3), r) for t, r in rec.rows if not r['hold']]
    return rows, res


def kicks(rows, t_from):
    return sum(1 for t, r in rows if r['assist'] == A.ASSIST_START and t >= t_from)


def main():
    out = {}
    P = dict(PARAMS, launch_boost=1, launch_no_kick=1, start_kick=1, recovery=1, tof_guard=1)
    base, _ = go(P, tag='base')
    gu = [t for t, r in base if r['assist'] == A.ASSIST_GIVEUP][0]
    out['giveup_s'] = gu
    tg = gu + 1.0
    r1, res1 = go(P, tg, {'launch_no_kick': 0}, 'A')
    out['A nk 1->0 after giveup'] = {'accepted': 'p' in res1, 'err': res1.get('err'), 'start_frames': kicks(r1, tg),
                                     'same_as_no_change': [(t, r['out'][1], r['assist']) for t, r in r1]
                                     == [(t, r['out'][1], r['assist']) for t, r in base]}
    PB = dict(PARAMS, launch_boost=0, launch_no_kick=0, start_kick=0, recovery=1, tof_guard=1)
    b0, _ = go(PB, tag='B0')
    gub = [t for t, r in b0 if r['assist'] == A.ASSIST_GIVEUP][0]
    r2, res2 = go(PB, gub + 1.0, {'start_kick': 1}, 'B')
    out['B start_kick 0->1 after giveup'] = {'accepted': 'p' in res2, 'err': res2.get('err'), 'start_frames': kicks(r2, gub + 1.0)}
    r2b, res2b = go(dict(PB, start_kick=1), gub + 1.0, {'start_kick': 0}, 'B2')
    out['B2 start_kick 1->0 (allowed)'] = {'accepted': 'p' in res2b, 'err': res2b.get('err')}
    rv = [t for t, r in base if r['assist'] == A.ASSIST_RECOVERY and r['out'][1] > N + 100]
    tr = rv[2]
    r3, res3 = go(P, tr, {'launch_no_kick': 0}, 'C')
    out['C nk 1->0 mid reverse'] = {'at': tr, 'accepted': 'p' in res3, 'start_frames': kicks(r3, tr),
                                    'same_as_no_change': [(t, r['out'][1], r['assist']) for t, r in r3]
                                    == [(t, r['out'][1], r['assist']) for t, r in base]}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_rs1_2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in out.items():
        print(k, v)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
