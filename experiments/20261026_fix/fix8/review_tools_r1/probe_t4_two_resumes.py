"""反証 8-1 回目・係 tools・台本 5(読むだけ): 1 本に 2 回の走り直し・走り直しの直後の中止 で、印(after_abort)と ⑤ ③ ④ の答えが
決めどおりか(OFF 4 本・ON 3 本は tests/test_fix7_tools.round3 と同じ。3-8 = 発進の前進 ON):

  2rec  1 回目の発進 → 2 周目に R1 の 3.5 m 手前で止まる → リカバリー(assist 2)の途中で人側 1 秒 → その場で AI 側(走り直し 1)→
        走り直し 1 は R1 を過ぎて最初の周の途中(+4 秒)で 6G に当たって止まる → リカバリーの途中で人側 1 秒 → その場で AI 側(走り直し 2)→ 3 周
  2rec_pre  同じだが、走り直し 1 は R1 の手前(+0.9 秒)で当たって止まる(R1 に届かない)
  abort_after  走り直し 1 が 0.4 m ほど出たところでカメラが 0.8 秒止まって止まり、明けて中止(assist 3)→ 人側 1 秒 → 走り直し 2 → 3 周
  bounce_rec  走り直し 1 を 0.3 秒(0.5 m 未満)で人側 → 1 秒 → 走り直し 2(続き carried)→ 3 周
⑤ は --ignore-gain。出力 out/probe_t4_two_resumes.txt

  jf-heavy --lock none --mem 1500M -- python -B probe_t4_two_resumes.py
"""
import common as C
from common import t7, t8, PRM, ON


def day(tmp, shape):
    D = t7.Day(tmp)
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, PRM, 0.21, 0.6), (15, ON, 0.20, 0.3),
                               (16, PRM, 0.22, 0.6), (17, ON, 0.28, 0.3), (18, PRM, 0.23, 0.6)):
        D.add(blk, prm, dm, ramp)
    base = t7.launch_keys(1.0, 0.31, 0.3)
    t_st = t7.time_at_distance(base + [(200.0, 2.0)], 23.0 - 3.5 - 1.0, 200.0)
    t_stop, t_flip = t_st + 1.0, t_st + 3.0
    r1 = t_flip + 1.0                       # 走り直し 1
    kw = {}
    if shape in ('2rec', '2rec_pre'):
        dt_hit = 4.0 if shape == '2rec' else 0.9
        th = r1 + dt_hit                    # 走り直し 1 の当たり
        f2 = th + 2.0                       # 2 つ目のリカバリーの途中で人側
        r2 = f2 + 1.0                       # 走り直し 2
        dur = r2 + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (r1 + 0.15, 0.0), (r1 + 0.45, 2.0), (th, 2.0), (th + 0.3, 0.0),
                     (r2 + 0.15, 0.0), (r2 + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (r1, f2), (r2, dur - 1.0)]
        kw['hits'] = [(th, 6.0)]
        recs = [(t_st + 0.8, t_flip), (th + 0.4, f2)]
        kw['launch'] = [(1.0, 0.5), (r1, 0.5), (r2, 0.5)]
    elif shape == 'abort_after':
        ts2 = r1 + 0.55                     # 走り直し 1 が 0.4 m ほど出たところでカメラが止まる
        f2 = ts2 + 0.8 + 2.0
        r2 = f2 + 1.0
        dur = r2 + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (r1 + 0.15, 0.0), (r1 + 0.45, 1.2), (ts2, 1.2), (ts2 + 0.7, 0.0),
                     (r2 + 0.15, 0.0), (r2 + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (r1, f2), (r2, dur - 1.0)]
        kw.update(stall=[(ts2, ts2 + 0.8)], giveup=[(ts2 + 0.8, f2)])
        recs = [(t_st + 0.8, t_flip)]
        kw['launch'] = [(1.0, 0.5), (r1, 0.5), (r2, 0.5)]
    else:                                   # bounce_rec
        f2 = r1 + 0.3
        r2 = f2 + 1.0
        dur = r2 + 32.0
        vk = base + [(t_st, 2.0), (t_stop, 0.0), (r1 + 0.1, 0.0), (r1 + 0.3, 0.8), (r1 + 0.6, 0.0),
                     (r2 + 0.15, 0.0), (r2 + 0.45, 2.0), (dur, 2.0)]
        ai = [(1.0, t_flip), (r1, f2), (r2, dur - 1.0)]
        recs = [(t_st + 0.8, t_flip)]
        kw['launch'] = [(1.0, 0.5), (r1, 0.3), (r2, 0.5)]
    log = D.add(19, ON, 0.31, 0.3, dur=dur, vk=vk, ai=ai, **kw)
    for a, b in recs:
        t8.set_assist(D.raw, log, a, b, 2)
    return D, log


def main():
    lines = []
    for shape in ('2rec', '2rec_pre', 'abort_after', 'bounce_rec'):
        D, log = day(C.fresh('t4_' + shape), shape)
        j, out, _ = C.runc(D, 'new', C.IG, 't4' + shape)
        g = j['summary'][0]
        lines.append('== %s(%s)' % (shape, log))
        for r in j['launches']:
            if r['log'] == log:
                lines.append('  ⑤ 発進 %.2f 秒(start_no %s): to_r1 %s・最初の周の衝突 %s・印 %s' % (
                    r['t_ai'], r.get('start_no'), r.get('to_r1_s'), r.get('collisions_first_lap'), r.get('after_abort')))
        lines.append('  ' + C.group_text(g))
        R, md, qmd, runs, groups = t8.run_s3(D, 'new')
        lines.append('  ③ 3-8 の発進: %s' % [(s['t_ai'], s['counted'], (s.get('after_abort') or {}).get('kind'), s.get('coll_before'))
                                            for s in t8.st_of(R, log)])
        g250 = t8.on250(groups)
        lines.append('  ④ 発進 250 の組: 止まってから 3 周 %s・数えない %s' % (
            g250.get('standing_list_s') or g250.get('standing_best_s'), g250.get('standing_excluded_n')))
    C.write('probe_t4_two_resumes.txt', lines)
    C.cleanup()


if __name__ == '__main__':
    main()
