"""r3-r3-safety-1 の確かめ(直さない)。Recovery を直接呼ぶ。drive の呼び方をまねる:
前へ走るフレーム(eRPM -5000)→ 当たり → (A) 当たりのあと 1 フレーム処理(衝撃・eRPM 0)してからカメラ停止 /
(B) 処理できずにカメラ停止。明けたフレームに hold_since・hold_g(6.5)を渡す。モデルは中立 -120 µs。"""
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A

C, N = 1500.0, 1508.0
MODEL = (C, N - 120.0)
FR = 1 / 30.


def run(case, L, keep=True, after_s=4.0):
    a = A.Assist((C, N))
    now = 100.0
    t_hit = None
    # 走る(armed にする)
    for i in range(60):
        a.step(MODEL, -5000.0, True, now, impact_g=0.2, rec_guard=True)
        now += FR
    t_hit = now - 0.025            # 最後に処理したフレームの 25 ms あとに当たり
    if case == 'A':                # 当たりのあとの 1 フレームを処理できた(衝撃 6.5G・止まった)
        now = t_hit + 0.008
        a.step(MODEL, 0.0, True, now, impact_g=6.5, rec_guard=True)
    last = now
    # カメラ停止: drive は 0.1 秒で中立止め(hold)。keep なら reset(keep_recovery=True)
    hold_t0 = last + 0.1
    a.reset(keep_recovery=keep)
    now = t_hit + L                # 明けた最初のフレーム
    out = []
    first = True
    while now < t_hit + L + after_s:
        kw = {}
        if keep and first:
            kw = dict(hold_g=6.5, hold_since=hold_t0, hold_back_s=0.0)
        o, k = a.step(MODEL, 0.0 if True else 0.0, True, now, impact_g=0.0, rec_guard=True, **kw)
        ev = a.rec.guard_event
        a.rec.guard_event = None
        out.append((round(now - (t_hit + L), 3), round(o[1], 1), A.ASSIST_NAMES[k], ev))
        first = False
        now += FR
    return out


def summary(out):
    gu = [t for t, thr, k, ev in out if k == 'リカバリー中止']
    rev = [t for t, thr, k, ev in out if k == 'リカバリー' and thr > N + 40]
    evs = [e for *_, e in out if e]
    return {'giveup_at': gu[0] if gu else None, 'first_reverse_side_at': rev[0] if rev else None,
            'max_thr': max(thr for _, thr, _, _ in out), 'events': evs}


for keep in (True, False):
    for case in ('A', 'B'):
        for L in (0.6, 1.0, 2.0):
            print('keep=%d case=%s L=%.1f' % (keep, case, L), summary(run(case, L, keep)))
