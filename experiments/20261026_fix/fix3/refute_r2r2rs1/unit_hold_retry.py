"""r2-r2-rs-1 の反証(直さない): Recovery.step だけを決まった入力で回し、'reverse' の始めに呼ばれない間(中立止め)を
はさんだとき、明けたフレームで backed_s = 0・el > CHECK_S なら 'gap' のやり直し(+50 µs)になるか、
中立止めの間に下がった時間が数えられるか(dt は 0.2 秒まで)を見る"""
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A   # noqa: E402

C, N = 1500.0, 1500.0
FPS = 1 / 30.0


def drive_to_reverse(r, t):
    """前へ走る → 止まる → 'reverse' に入るまで。戻り値 (t, 'reverse' に入った時刻)"""
    for _ in range(30):                     # 前へ 2000 eRPM(armed)
        r.step((C, N - 120), -2000, t, back_guard=True); t += FPS
    while r.phase != 'reverse':
        r.step((C, N - 120), 0, t, back_guard=True); t += FPS
    return t, r.t_phase


def case(hold_s, erpm_during, erpm_after, n_pre=2):
    r = A.Recovery((C, N))
    t, tr = drive_to_reverse(r, 0.0)
    outs = []
    for i in range(n_pre):                  # 'reverse' で処理したフレーム(まだ下がり始めを見ていない)
        o, k = r.step((C, N - 120), 0, t, back_guard=True); outs.append((round(t - tr, 3), r.phase, o)); t += FPS
    b0 = r.backed_s
    # 中立止め: step は呼ばれない(drive は Assist.reset(keep_recovery=True) で Recovery を戻さない)
    t += hold_s
    o, k = r.step((C, N - 120), erpm_after, t, back_guard=True)
    first = (round(t - tr, 3), r.phase, o, r.retries, round(r.backed_s, 3))
    # 続き: 止まったまま 0.3 秒(やり直しの 'gap')→ 'reverse' の出す値
    seq = []
    for _ in range(20):
        t += FPS
        o, k = r.step((C, N - 120), 0, t, back_guard=True)
        seq.append((r.phase, None if o is None else round(o[1] - N, 1)))
    return {'hold_s': hold_s, 'backed_before': b0, 'first_after': first,
            'rev_written_after': sorted(set(v for p, v in seq if p == 'reverse')), 'phases_after': [p for p, _ in seq][:12]}


def undercount(hold_s, erpm_after):
    """'reverse' で 0.1 秒下がっているのを見たあと、中立止め hold_s(この間ずっと下がっている想定)→ 明けたフレームの backed_s"""
    r = A.Recovery((C, N))
    t, tr = drive_to_reverse(r, 0.0)
    for e in (0, 0, 0, 800, 800, 800):
        r.step((C, N - 120), e, t, back_guard=True); t += FPS
    b0 = r.backed_s
    t += hold_s
    r.step((C, N - 120), erpm_after, t, back_guard=True)
    return {'hold_s': hold_s, 'erpm_after': erpm_after, 'backed_before': round(b0, 3), 'backed_after': round(r.backed_s, 3),
            'added': round(r.backed_s - b0, 3), 'phase': r.phase}


if __name__ == '__main__':
    print('CHECK_S', A.CHECK_S, 'RETRY_EXTRA_US', A.Recovery.RETRY_EXTRA_US)
    for h in (0.3, 0.45, 0.6, 1.0):
        print('retry', case(h, 800, 0))
    print('nohold ref', case(0.0, 0, 0, n_pre=0)['first_after'])
    for h, e in ((0.6, 0), (0.6, 800), (1.0, 800)):
        print('undercount', undercount(h, e))
