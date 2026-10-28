"""fix8 係 safety 直す 8-1 回目: 案 qA・qC を、反証 8-1 回目の係の場面の組(fix8/review_safety_r1/exp11.py の lw・rd・rd2・sl・hm・
fz。読み込んで使うだけ。その台本・出力は変えない)で、今の両方 ON・両方 OFF と並べる。本物の drive._loop の閉じたループ(h11)。
案の版は qload.load_version の写し(h11._MODS に入れて h11.run の版の名前で回す)。一時の場所は fix8/tmp_safety_r1(h11.TMP を
差し替える。反証の係の一時の場所は使わない)。
場面ごとに: 案の ON が今の ON とビット単位で同じか(h11.same_run)・違えば、反証の係と同じ「悪い印」(exp11.worse)を
案の ON 対 今の OFF・案の ON 対 今の ON で出す。下がる距離(最初のリカバリー)・escape の壁押し・後ろの当たりの最大の速さも。
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/proposal_r1/exp_q12.py <kind> <i/n> [seed count]
    → out/exp_q12_<kind>_<i>of<n>.json
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent / 'review_safety_r1'
TAG = 'q12_' + '_'.join(sys.argv[1:3]).replace('/', 'of')
os.environ['RS8_TAG'] = TAG
os.environ.setdefault('R1_TAG', TAG)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REV))
import h11  # noqa: E402
import exp11  # noqa: E402
import qload  # noqa: E402

h11.TMP = HERE.parent / 'tmp_safety_r1' / 'h11' / TAG
PROPS = ('qA', 'qC')
for _p in PROPS:
    h11._MODS[_p] = qload.load_version(_p)


def seg0(m):
    s = (m['segs'] or [{}])[0]
    return {k: s.get(k) for k in ('t0', 't1', 'back', 'vmin', 'esc_push', 'at_wall')}


def one(sd):
    p = exp11.params_of(sd)
    t_from = sd.get('ai_from', 1.0)
    runs = {'now_on': h11.run('now', h11.ON(p), exp11.build(sd)), 'now_off': h11.run('now', h11.OFF(p), exp11.build(sd))}
    for q in PROPS:
        runs['%s_on' % q] = h11.run(q, h11.ON(p), exp11.build(sd))
    ms = {k: h11.metrics(v, t_from) for k, v in runs.items()}
    row = {'sd': sd}
    for k, m in ms.items():
        row[k] = {'rear_n': m['rear_n'], 'rear_max_v': m['rear_max_v'], 'front_push_s': m['front_push_s'],
                  'zero_push': m['zero_push'], 'front_max_v': m['front_max_v'], 'giveup_t': m['giveup_t'],
                  'guards': [g[0] for g in m['guards'][:10]], 'seg0': seg0(m), 'recoveries': m['recoveries']}
    for q in PROPS:
        same, why = h11.same_run(runs['%s_on' % q], runs['now_on'])
        row['%s_same_as_now_on' % q] = same
        if not same:
            row['%s_first_diff' % q] = str(why)[:200]
            row['%s_worse_than_off' % q] = exp11.worse(ms['%s_on' % q], ms['now_off'])
            row['%s_worse_than_now_on' % q] = exp11.worse(ms['%s_on' % q], ms['now_on'])
            row['now_on_worse_than_%s' % q] = exp11.worse(ms['now_on'], ms['%s_on' % q])
    row['now_on_worse_than_off'] = exp11.worse(ms['now_on'], ms['now_off'])
    return row


def main(kind, part='0/1', seed='8801', count='300'):
    qload.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    gen = {'lw': exp11.gen_lw, 'rd': exp11.gen_rd, 'sl': exp11.gen_sl, 'rd2': exp11.gen_rd2, 'hm': exp11.gen_hm}.get(kind)
    sds = list(gen()) if gen else list(exp11.gen_fz(int(seed), int(count)))
    res = []
    t_s = time.time()
    try:
        for k, sd in enumerate(sds):
            if k % pn != pi:
                continue
            res.append(one(sd))
            if len(res) % 25 == 0:
                print(kind, len(res), '%.0fs' % (time.time() - t_s), flush=True)
    finally:
        h11.cleanup()
    name = 'exp_q12_%s_%dof%d.json' % (kind, pi, pn)
    (HERE / 'out' / name).write_text(json.dumps({'kind': kind, 'n': len(res), 'rows': res, 'elapsed_s': round(time.time() - t_s, 1)},
                                                ensure_ascii=False, default=str))
    print('done', kind, part, len(res), '%.0fs' % (time.time() - t_s))


if __name__ == '__main__':
    main(*sys.argv[1:])
