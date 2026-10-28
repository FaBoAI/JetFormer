"""指摘 3(w1-tools-r8-1-recovery-just-ended)を打ち消すつもりで確かめる。

(1) standing_3lap.abort_resumes だけ(自分で作った配列): AI の区間 [1, 10) 秒・[11, 19) 秒。2 m/s で走り 5.5 秒に止まる(そのまま)。
    リカバリー(assist 2)を 6.0 秒 〜 区間の終わりの k フレーム前まで。k = 0・1・2・9・30・54(1.8 秒)。中立止め・AI_CMD の人側は無し。
    2 つ目の区間の始め(11 秒)に印が付くか。
(2) ⑤ collect.py --ignore-gain と ③ standing_3lap の作り物の回 3(自分で決めた値): OFF 4 本(d_move 0.20〜0.23・ramp 0.6)、ON 3 本
    (0.15・0.20・0.28・ramp 0.3)、3-8 = ON: 1.0 秒に発進 → 2 周目の R1 の 3.5 m 手前で止まる(1 秒で)→ 止まって 0.8 秒からリカバリー
    (assist 2)→ 区間の終わりの k フレーム前に終わる → 止まって 2 秒で CH3 を人側 1.0 秒 → その場で AI 側(発進の前進がまた出る)。
    ⑤ の ON の本数・ON − OFF・after_abort、③ の counted を k ごとに。
(3) 手順書の文(読むだけ): 「リカバリーが終わってから人側にしたときは出ない」「走り直したら紙に」があるか。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/refute_tools_r1_A')))
import rcommon as rc  # noqa: E402

KS = (0, 1, 2, 9, 30, 54)
P = dict(rc.BASE)
ON = dict(P, launch_boost=1)


def unit():
    s3 = rc.load_mod('refA_s3_unit', rc.S3)
    t = np.arange(0, 20, 1 / 30.0)
    ai = ((t >= 1.0) & (t < 10.0)) | ((t >= 11.0) & (t < 19.0))
    vt = np.arange(0, 20, 0.01)
    v = np.where((vt >= 1.3) & (vt < 5.5), 2.0, 0.0)
    v = np.where(vt >= 11.3, 2.0, v)
    Df = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(vt))])
    verpm = -v / 0.5e-3
    i_end = int(np.nonzero(ai & (t < 10.0))[0][-1])
    out = []
    for k in KS:
        a = np.zeros(len(t), np.int8)
        i_rec0 = int(np.searchsorted(t, 6.0))
        a[i_rec0:i_end + 1 - k] = 2
        r = s3.abort_resumes(t, ai, a, np.zeros(len(t), bool), np.ones(len(t), bool), vt, Df, verpm)
        i_res = int(np.searchsorted(t, 11.0 - 1e-9))
        out.append({'k': k, 'last_assist': int(a[i_end]), 'mark': r.get(i_res)})
    return out


def build(k, stop=3.5):
    D = rc.MyDay('t3_k%d_s%s' % (k, stop))
    for blk, prm, dm, ramp in ((12, P, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, P, 0.21, 0.6), (15, ON, 0.20, 0.3),
                               (16, P, 0.22, 0.6), (17, ON, 0.28, 0.3), (18, P, 0.23, 0.6)):
        D.add(blk, prm, dm, ramp)
    base = rc.t7.launch_keys(1.0, 0.31, 0.3)
    t_st = rc.t7.time_at_distance(base + [(200.0, 2.0)], 23.0 - stop - 1.0, 200.0)
    t_stop = t_st + 1.0
    t_flip = t_stop + 2.0
    t_res = t_flip + 1.0
    dur = t_res + 32.0
    vk = base + [(t_st, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
    log = D.add(19, ON, vk=vk, dur=dur, ai=[(1.0, t_flip), (t_res, dur - 1.0)], launch=[(1.0, 0.5), (t_res, 0.5)])
    # リカバリー: 止まって 0.8 秒から、区間の終わり(t_flip の前の最後のフレーム)の k フレーム前まで
    ft = np.arange(0, dur, 1 / 30.0)
    last = float(ft[ft < t_flip - 1e-9][-1])
    rc.set_assist(D.raw, log, t_stop + 0.8, last + 1 / 30.0 - k / 30.0 + 1e-6, 2)
    return D, log, {'t_st': t_st, 't_stop': t_stop, 't_flip': t_flip, 't_res': t_res}


def e2e():
    out = []
    for stop in (3.5, 4.5):
        for k in KS:
            D, log, tt = build(k, stop)
            j, txt = rc.run_collect(D, ['--ignore-gain'], tag='t3_k%d_%s' % (k, stop))
            jf, _ = rc.run_collect(D, ['--ignore-gain', '--first-only'], tag='t3f_k%d_%s' % (k, stop))
            g = [x for x in rc.summary_of(j) if x.get('compare')][0]
            gf = [x for x in rc.summary_of(jf) if x.get('compare')][0]
            rows = [x for x in j['launches'] if x['log'] == log]
            R, md, qmd, runs, groups = rc.run_s3(D)
            st = [x for x in R if x['log'] == log][0]['standing']
            out.append({'stop': stop, 'k': k, 'on_n': g['by_launch']['on']['n'], 'd_to_r1': g['compare']['d_to_r1_s'],
                        'all_ok': g['compare']['all_ok'],
                        'first_only_on_n': gf['by_launch']['on']['n'], 'first_only_all_ok': gf['compare']['all_ok'],
                        'first_only_d': gf['compare']['d_to_r1_s'],
                        'rows': [(x['t_ai'], x.get('to_r1_s'), (x.get('after_abort') or {}).get('kind')) for x in rows],
                        's3': [(round(s.get('t_ai', 0) or 0, 2), s.get('counted'), (s.get('after_abort') or {}).get('kind'))
                               for s in st]})
    return out


def doc_check():
    rb = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md').read_text(encoding='utf-8')
    return {'リカバリーが終わってから人側にしたときは出ない': 'リカバリーが終わってから人側にしたときは出ない' in rb,
            '走り直したら紙に「走り直し: 本・時刻」と書く': '走り直したら紙に「走り直し: 本・時刻」と書く' in rb,
            '印が無ければ、⑤ は main session が `--skip 記録名@AI側にした秒` で外す': '印が無ければ、⑤ は main session が `--skip 記録名@AI側にした秒` で外す' in rb,
            '当たって止まったあと、その場で AI 側に戻したくなった | **どの回でも**戻さない':
                '当たって止まったあと、その場で AI 側に戻したくなった | **どの回でも**戻さない' in rb}


def main():
    res = {'unit': unit(), 'e2e': e2e(), 'doc': doc_check()}
    (rc.HERE / 'out').mkdir(exist_ok=True)
    (rc.HERE / 'out' / 't3_recovery_edge.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                         encoding='utf-8')
    for u in res['unit']:
        print('unit k=%-2d 区間の最後の assist %d → 印 %s' % (u['k'], u['last_assist'], u['mark']))
    for e in res['e2e']:
        print('e2e stop %.1f k=%-2d ⑤ ON n %d d %s → %s | --first-only ON n %d d %s → %s | 行 %s | ③ %s' % (
            e['stop'], e['k'], e['on_n'], e['d_to_r1'], '使ってよい' if e['all_ok'] else '満たさない', e['first_only_on_n'],
            e['first_only_d'], '使ってよい' if e['first_only_all_ok'] else '満たさない', e['rows'], e['s3']))
    print('手順書:', res['doc'])


if __name__ == '__main__':
    main()
