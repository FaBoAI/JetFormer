"""反証の打ち消し(指摘 w2-safety-w2-safety-r62-safety-2)を自分で確かめる(読むだけ。直さない・装置に触れない・GPU なし)。
1) 単体: assist.Recovery を直接まわし、'reverse' の途中の残した中立止めのあとに hold_rev_s(見積もり)だけ来たとき、
   CHECK_S のやり直しが起きるか(hold_rev_s = 0 と比べる)
2) 閉じたループ: 本物の drive._loop(tests/test_fix3_r2_1)+ 偽物の ESC(tf62.Car。ブレーキのあとの中立 neu 秒で下がれる)。
   場面 rev(前の壁 2.0 m・モデル 前進 120 µs)、既定 rev_s 0.4・rev_us 300、keep 1・見張り 1。
   'reverse' の始めから車両の値が L 秒来ない。今(now)と fix6 の直す前(fix6_before)、keep 0、止まりなしを比べる。
一時の場所は fix6/tmp_refute_w2s2(終わったら消す)。    python -I repro.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX6 = HERE.parent
sys.path.insert(0, str(FIX6 / 'refute62_safety'))
import c62 as K          # noqa: E402
import tf62 as T         # noqa: E402
import probe_false_ready as P   # noqa: E402
E = K.E
R = K.R
TMP = FIX6 / 'tmp_refute_w2s2'
R.TMP = TMP / 'rs'
R.CL.TMP = R.TMP
E.TMP = TMP / 'exp'


def unit():
    from jetformer import assist as A
    res = {}
    for est in (0.0, 0.11):
        r = A.Recovery((1500.0, 1508.0))
        r.armed = True
        r.phase = 'reverse'
        r.t_phase = r.t_check = 100.0
        r.t_last = 100.0
        out = []
        t = 100.0
        # 'reverse' 0.13 秒(止まったまま)→ 0.1 秒の中立止め → 明けて止まったまま
        for k in range(5):
            t = round(100.0 + 0.033 * k, 3)
            o = r.step((1500.0, 1388.0), 0.0, t, back_guard=True)
            out.append((round(t - 100, 3), r.phase, r.backed_s, r.retries, None if o[0] is None else o[0][1]))
        t_hold = t + 0.03
        t = t + 0.1
        o = r.step((1500.0, 1388.0), 0.0, t, back_guard=True, hold_since=t_hold, hold_back_s=0.0, hold_rev_s=est)
        out.append((round(t - 100, 3), r.phase, r.backed_s, r.retries, None if o[0] is None else o[0][1], 'hold'))
        t0 = t
        while t < t0 + 1.4:
            t = round(t + 0.033, 3)
            o = r.step((1500.0, 1388.0), 0.0, t, back_guard=True)
            out.append((round(t - 100, 3), r.phase, round(r.backed_s, 3), r.retries,
                        None if o[0] is None else o[0][1]))
            if r.phase != 'reverse':
                break
        res['hold_rev_s=%s' % est] = out
        first_change = next((x for x in out if x[1] != 'reverse'), None)
        print('unit hold_rev_s=%.2f: hold frame' % est, [x for x in out if len(x) == 6], 'first leave reverse', first_change)
    return res


def closed(neu, cells):
    T.ESC.update(neu=neu, rdelay=0.08)
    p = E.P1()
    base = T.scene('rev', 2.0, 120.0, 0.5)
    out = {'neu': neu}
    sim, rec, pca = T.run('now', p, base)
    out['nogap_now1'] = P.metrics(sim, rec, pca, p)
    print('neu', neu, 'no veh gap now1', {k: out['nogap_now1'][k] for k in ('back_m', 'esc_push_s', 'rev_dead_s', 'retry')},
          flush=True)
    cnt = {'n': 0, 'now1_fail': 0, 'bef1_fail': 0, 'now0_fail': 0}
    for d0, L in cells:
        b = (round(2.52 + d0, 3), round(2.52 + d0 + L, 3))
        sc = dict(base, veh_skip=[b])
        row = {}
        for nm, mod, pp in (('now1', 'now', p), ('bef1', 'fix6_before', p), ('now0', 'now', dict(p, rec_hold_keep=0))):
            s_, r_, pc_ = T.run(mod, pp, sc)
            m = P.metrics(s_, r_, pc_, pp)
            m['t_fake_rev_on'] = next((t for t, rv, c in s_.car.revlog if t > 2.0 and rv), None)
            row[nm] = m
        cnt['n'] += 1
        for nm in ('now1', 'bef1', 'now0'):
            if row[nm]['back_m'] < 0.1 and row[nm]['esc_push_s'] > 0:
                cnt[nm + '_fail'] += 1
        out['d0=%+.2f L=%.2f' % (d0, L)] = row
        print('neu', neu, 'skip', b, {nm: (m['back_m'], m['esc_push_s'], m['rev_dead_s'], m['retry'], m['t_escape'],
                                           m['t_fake_rev_on']) for nm, m in row.items()}, flush=True)
    out['count'] = cnt
    print('neu', neu, cnt, flush=True)
    return out


def main():
    res = {}
    try:
        res['unit'] = unit()
        main_cell = [(0.0, 0.2)]
        res['neu0.2_main'] = closed(0.2, main_cell)
        grid = [(d0, L) for d0 in (-0.1, -0.05, 0.0, 0.05, 0.1, 0.2) for L in (0.16, 0.2, 0.25, 0.3, 0.4)]
        res['neu0.2_grid'] = closed(0.2, grid)
        res['neu0.1_grid'] = closed(0.1, [(d0, L) for d0 in (-0.05, 0.0, 0.05) for L in (0.2, 0.3)])
    finally:
        K.cleanup()
        shutil.rmtree(TMP, ignore_errors=True)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'repro.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
