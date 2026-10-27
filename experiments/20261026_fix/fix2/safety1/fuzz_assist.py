"""assist.Assist をでたらめな列で回し、モーターに出す値の決まりを確かめる(読むだけ・装置なし)。
決まり: (1) 人側 → モデルのまま (2) 発進の前進 → 中立 − us 以下(前進だけ) (3) 止まっている(eRPM ≥ −150)のに
中立 + 40 µs より後ろを出すのはリカバリー・後退の受け付けだけ (4) 数でない値を出さない
(5) 発進の前進・発進の補助は 1 つの AI 区間で前進を始めるのは 1 回(終わったら同じ区間で再び出ない)"""
import json, math, random, sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A
N = 1496.0
bad = []
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for run in range(3000):
    h = A.Assist((1441.0, N))
    t = 100.0
    sel_prev = False
    launch_seen_end = False
    kick_seen_end = False
    kick_prev = False
    for k in range(200):
        t += random.choice([1 / 30, 1 / 30, 1 / 30, 0.0, 0.2, 0.5])
        sel = random.random() < 0.85 if sel_prev else random.random() < 0.3
        if random.random() < 0.05:
            h.launch_hold(); h.reset()
            continue
        if sel and not sel_prev:
            launch_seen_end = kick_seen_end = False
            kick_prev = False
        sel_prev = sel
        erpm = random.choice([0.0, random.uniform(-500, 500), random.uniform(-9000, 2000)])
        thr = random.choice([N, N + 100, N - 50, N - 300, random.uniform(1000, 2000)])
        model = (random.uniform(1000, 2000), thr)
        inp = None
        if random.random() < 0.8:
            inp = A.LaunchIn(us=random.choice([150, 250, 300]), mps=2.5, max_s=random.choice([0.5, 1.2, 2.0]),
                             v=-erpm * 0.5e-3, a=random.uniform(-2, 6), gz=random.uniform(-80, 80),
                             steer_rel=random.uniform(-300, 300), ttc=random.choice([0.5, 2.0, 5.0, 30.0]),
                             ttc5=tuple(random.choice([3.0, 4.0, 30.0]) for _ in range(5)), tof_ok=random.random() < 0.9,
                             veh_ok=random.random() < 0.95, impact_g=random.choice([0.0, 0.0, 3.0]), guard_changed=random.random() < 0.05)
        o, kind = h.step(model, erpm, sel, t, esc_on=random.random() < 0.5, rec_on=random.random() < 0.7,
                         impact_g=random.choice([0.0, 0.0, 0.0, 3.0]), kick_on=random.random() < 0.7,
                         blocked=random.random() < 0.2, launch=inp)
        ctx = dict(run=run, k=k, sel=sel, erpm=erpm, model=model, out=o, kind=kind)
        if not all(math.isfinite(x) for x in o):
            bad.append(('nan', ctx))
        if not sel and o != model:
            bad.append(('human_changed', ctx))
        if kind == A.ASSIST_LAUNCH and not (o[1] <= N - h.boost.cfg[0] + 1e-9):
            bad.append(('launch_weak', ctx))
        if kind in (A.ASSIST_LAUNCH, A.ASSIST_START) and o[1] >= N:
            bad.append(('launch_or_kick_not_forward', ctx))
        if sel and erpm >= A.STOPPED_ERPM and o[1] > N + 40 and kind not in (A.ASSIST_RECOVERY, A.ASSIST_ESC):
            if not (o[1] == model[1]):
                bad.append(('behind_neutral_changed', ctx))
        if sel and kind == A.ASSIST_LAUNCH and launch_seen_end:
            bad.append(('launch_twice', ctx))
        if sel and h.launch_end:
            launch_seen_end = True
print(json.dumps({'checked_runs': 3000, 'bad': len(bad), 'kinds': sorted(set(b[0] for b in bad)),
                  'examples': [dict(why=b[0], **{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in b[1].items()}) for b in bad[:6]]},
                 ensure_ascii=False, indent=1, default=str))
