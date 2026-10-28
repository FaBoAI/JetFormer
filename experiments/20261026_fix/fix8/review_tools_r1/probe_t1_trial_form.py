"""反証 8-1 回目・係 tools・台本 1(読むだけ): 「走った値が計画と違った → 試し」(手順書 7 章)の本の ON の最初の周の衝突・空転・
最初の周の記録の欠けが、回 3 の判定(⑤ の「ON − OFF」の行)に数えられるか。

手順書 7 章「走った値が計画と違った」の行: 「⑤ は「試しの N 本」に名前を出し、ON の組にも OFF の組にも入れない。ただし試しの ON の最初の周の
衝突は「最初の周の衝突 0」に数える。空転(比 > 1.3・比が出ない)・最初の周の記録の欠けも数える(どれかがあれば回 3 は「満たさない」。
8.1 ⑤。10/28 main session の決め・fix8 tools)」。

回 3 の作り物(tests/test_fix8_tools_decided.mtday と同じ: OFF 4 本・ON 3 本(launch_us 200)= ON の中央値 OFF − 0.190 秒 → 使ってよい)の
3-8 を、試し(memo に「試し」)の ON の発進(launch_us 200)にし、その 1 回目の発進で 空転(比 ≈ 1.85)/ 最初の周の衝突 6G / 最初の周の
IMU の欠け のどれかを起こす。3-8 の設定を:
  same     計画どおり(memo だけ「試し」)
  ceil     天井 ON(計画は OFF。「走った値が計画と違った」)
  left42   回 3 の全部が天井 ON・左 4.1 で、3-8 だけ左 4.2(2-5 のあと 4.1 に戻し忘れた)
⑤ は手順書どおり --ignore-gain。出力 out/probe_t1_trial_form.txt

  jf-heavy --lock none --mem 1500M -- python -B probe_t1_trial_form.py
"""
import common as C
from common import t7, t72, t8, PRM, ON200


def day(tmp, shape, form):
    base_prm = dict(PRM)
    if form == 'left42':
        base_prm.update(speed_ceiling=1, ceil_left_mps=4.1)
    on_prm = dict(ON200, **{k: base_prm[k] for k in ('speed_ceiling', 'ceil_left_mps')})
    D = t7.Day(tmp)
    for blk, prm, dm, ramp in ((12, base_prm, 0.20, 0.6), (13, on_prm, 0.15, 0.3), (14, base_prm, 0.21, 0.6),
                               (15, on_prm, 0.20, 0.3), (16, base_prm, 0.22, 0.6), (17, on_prm, 0.28, 0.3), (18, base_prm, 0.23, 0.6)):
        D.add(blk, prm, dm, ramp)
    p38 = dict(on_prm)
    if form == 'ceil':
        p38.update(speed_ceiling=1)
    elif form == 'left42':
        p38.update(ceil_left_mps=4.2)
    kw = {}
    if shape == 'hit':
        kw['hits'] = [(3.0, 6.0)]
    if shape == 'gap':
        log = D.add(19, p38, 0.15, 0.3, dur=45.0, vk=t7.launch_keys(14.0, 0.15, 0.3, until=45.0), ai=[(14.0, 43.0)],
                    launch=[(14.0, 0.5)])
        (D.raw / log / ('%s-0002' % log) / 'IMU.npz').unlink()
    else:
        log = D.add(19, p38, 0.15, 0.3, **kw)
        if shape == 'spin':
            t72.scale_imu(D.raw, log, 1.0, 2.2, 0.55)
    t8.make_kind(D.raw, log, 'trial')
    return D, log


def main():
    lines = []
    for shape in ('spin', 'hit', 'gap'):
        for form in ('same', 'ceil', 'left42'):
            D, log = day(C.fresh('t1_%s_%s' % (shape, form)), shape, form)
            j, out, _ = C.runc(D, 'new', C.IG, 't1%s%s' % (shape, form))
            lines.append('== 3-8 = 試しの ON・%s・形 %s(%s)' % (shape, form, log))
            for g in j['summary']:
                lines.append('  ' + C.group_text(g))
            lines += ['  | ' + x for x in out.splitlines() if x.startswith('ON − OFF') or ' → 試しの本' in x or '試しの' in x]
    C.write('probe_t1_trial_form.txt', lines)
    C.cleanup()


if __name__ == '__main__':
    main()
