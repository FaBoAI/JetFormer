"""r1-rs1-3 の打ち消しの確かめ(読むだけ・回すだけ。装置は開かない・何も直さない)。
1) Assist の形: launch=None(launch_boost=0)で launch_no_kick=True/False の StartKick
2) 本物の drive._loop(probe_refute1.run の一時置き場を TMP に替える): 前 0.25/0.5/0.9 m・床の基準なし × launch_boost 0/1 × launch_no_kick 0/1
3) 車が動き出したら(eRPM ≤ −800)StartKick は 2 秒より早く終わるか
出力は標準出力だけ(ファイルに書かない)。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute1_safety'))
import probe_refute1 as P                  # noqa: E402
from probe_refute1 import A, C, N, PARAMS, build, run, tl, tof_near   # noqa: E402

P.TMP = HERE.parent / 'tmp_verify_r1rs13'
res = {}

# 1) Assist の形
for nk in (False, True):
    a = A.Assist((C, N))
    t, n_kick, outs = 0.0, 0, set()
    a.step((C, N), 0.0, False, t)              # 人側 1 フレーム
    for i in range(120):                       # AI 側 4 秒・止まったまま・モデル中立・blocked なし
        t = 1.0 + i / 30.0
        o, k = a.step((C, N), 0.0, True, t, kick_on=True, blocked=False, launch=None, launch_no_kick=nk)
        if k == A.ASSIST_START:
            n_kick += 1
            outs.add(o[1])
    res['assist launch=None launch_no_kick=%s' % nk] = {'kick_s': round(n_kick / 30.0, 2), 'us': sorted(outs),
                                                         'boost_skipped': a.boost.skipped}

# 2) 本物の走行ループ
for name, tof, floor in (('front_0.25m', tof_near(250), True), ('front_0.5m', tof_near(500), True),
                         ('front_0.9m', tof_near(900), True), ('no_floor_ref', None, False)):
    for lb in (0, 1):
        for nk in (0, 1):
            frames, imu = build(4.0, sel=lambda s: s >= 1.0, v=0.0, pred=(C, N), tof=tof)
            p = dict(PARAMS, start_kick=1, launch_boost=lb, launch_no_kick=nk)
            d, pca, rec = run(p, frames, imu, floor=floor)
            rows = tl(frames, rec)
            k = [x for x in rows if x[1] == A.ASSIST_START]
            ev = [e[1] for e in rec.events if e[0] == 'launch_boost']
            res['loop %s lb=%d nk=%d' % (name, lb, nk)] = {
                'kick_s': round(len(k) / 30.0, 2), 'kick_us': sorted({x[3] for x in k}),
                'first_last': [k[0][0], k[-1][0]] if k else None, 'launch_events': ev[:2]}

# 3) 車が動き出す場面(AI 側にして 0.3 秒後から 0.5 m/s。前 0.5 m)
frames, imu = build(4.0, sel=lambda s: s >= 1.0, v=lambda s: 0.5 if s >= 1.3 else 0.0, pred=(C, N), tof=tof_near(500))
d, pca, rec = run(dict(PARAMS, start_kick=1, launch_boost=0, launch_no_kick=1), frames, imu)
rows = tl(frames, rec)
k = [x for x in rows if x[1] == A.ASSIST_START]
res['loop front_0.5m moves_at_1.3s lb=0 nk=1'] = {'kick_s': round(len(k) / 30.0, 2),
                                                   'first_last': [k[0][0], k[-1][0]] if k else None}
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
