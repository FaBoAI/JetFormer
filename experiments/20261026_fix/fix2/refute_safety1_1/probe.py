"""safety1-1 の反証(2026-10-26 fix2)。走行ループ drive._loop を偽物の装置で回す(tests/test_launch_drive.py の偽物)。
装置は開かない。直さない。結果は out.json。
  cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
     /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix2/refute_safety1_1/probe.py
"""
import collections, json, shutil, sys, tempfile
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from jetformer import assist as A, drive
from test_launch_drive import C, N, PARAMS, build, run

HERE = Path(__file__).resolve().parent
TMP = HERE / 'tmp'
P = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, launch_boost=0, tof_guard=1)


def scene(dur, hold_at=None, hold_len=0.2, sel=None):
    # 人側 1 秒 -> AI 側。1.0-2.0 秒 2 m/s(armed)。2.0 秒から壁で止まったまま。モデルはずっと 中立-120
    return build(dur, sel=sel or (lambda s: s >= 1.0), v=lambda s: 2.0 if 1.0 <= s < 2.0 else 0.0, pred=(C, N - 120.0),
                 veh_skip=(lambda s: hold_at <= s < hold_at + hold_len) if hold_at is not None else False)


def go(frames, imu, patch_reset=False):
    with pytest.MonkeyPatch.context() as mp:
        if patch_reset:      # 指摘の直し方(hold で Recovery を戻さない)を当てたときの比べ
            def reset(self):
                self.esc.reset(); self._esc_active = False
            mp.setattr(A.Assist, 'reset', reset)
        d = Path(tempfile.mkdtemp(dir=TMP))
        _, pca, rec = run(drive, mp, d, P, frames, imu)
        shutil.rmtree(d)
    s_of = {f['cam_t']: f['s'] for f in frames}
    return [(round(s_of[t], 3), int(r['assist']), bool(r['hold']), round(float(r['out'][1]), 1), bool(r['select']))
            for t, r in rec.rows], pca


def summ(tl, t_from):
    a = [x for x in tl if x[0] >= t_from and not x[2]]
    fwd = [x for x in a if x[3] < N]
    return {'n': len(a), 'kinds': collections.Counter((x[1], x[3]) for x in a).most_common(4),
            'fwd_frames': len(fwd), 'fwd_s': round(fwd[-1][0] - fwd[0][0], 2) if fwd else 0.0,
            'rec_or_giveup_frames': sum(1 for x in a if x[1] in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP))}


TMP.mkdir(exist_ok=True)
res = {}
tl0, _ = go(*scene(24.0))
g0 = next(x[0] for x in tl0 if x[1] == A.ASSIST_GIVEUP)
r0 = next(x[0] for x in tl0 if x[1] == A.ASSIST_RECOVERY)
res['base'] = {'first_recovery_s': r0, 'first_giveup_s': g0,
               'recovery_starts': [x[0] for i, x in enumerate(tl0) if x[1] == A.ASSIST_RECOVERY and (i == 0 or tl0[i-1][1] != A.ASSIST_RECOVERY)],
               'after_giveup': summ(tl0, g0 + 1.9)}
h = g0 + 1.5
tl1, _ = go(*scene(24.0, hold_at=h))
res['1a_hold_after_giveup'] = {'hold_at': round(h, 3), 'holds': sum(1 for x in tl1 if x[2]), 'after': summ(tl1, h + 0.4)}
tl1f, _ = go(*scene(24.0, hold_at=h), patch_reset=True)
res['1a_fixed_reset'] = {'after': summ(tl1f, h + 0.4)}
h2 = r0 + 0.45
tl2, _ = go(*scene(12.0, hold_at=h2))
res['1b_hold_mid_reverse'] = {'hold_at': round(h2, 3), 'before_hold': [x for x in tl2 if h2 - 0.1 <= x[0] < h2 + 0.25],
                              'after': summ(tl2, h2 + 0.4)}
tl2f, _ = go(*scene(12.0, hold_at=h2), patch_reset=True)
res['1b_fixed_reset'] = {'after': summ(tl2f, h2 + 0.4)}
# 人が CH3 を人側へ戻した(仕様どおりの解除)
tl3, _ = go(*scene(24.0, sel=lambda s: s >= 1.0 and not (g0 + 1.5 <= s < g0 + 2.0)))
res['human_takeover_after_giveup'] = {'after': summ(tl3, g0 + 2.4)}
shutil.rmtree(TMP, ignore_errors=True)
(HERE / 'out.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
