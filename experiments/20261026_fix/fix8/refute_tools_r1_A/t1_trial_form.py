"""指摘 1(w1-tools-r8-1-trial-form)を打ち消すつもりで確かめる。

回 3 の作り物(自分で決めた値): 計画の形 = 天井 ON・左 4.2。OFF 4 本(3-1・3-3・3-5・3-7)、ON 3 本(3-2・3-4・3-6。250 µs)、
3-8 = ON の発進で、走り終わりのメモに「試し」(手順書 7 章「起こし直した」の行: 入れ直す前に走った本の memo に「試し」)。
3-8 の形: same(計画どおり 左 4.2)/ left41(起こし直しで左 4.1 に戻ったまま)/ ceil_off(天井 OFF)/ auto_on(自動合わせ ON)。
3-8 の最初の周: none / hit(2.5 秒に 6G = R1 の前)/ spin(IMU の前の加速 ×0.55 = 空転の比 ≈ 1.8)/ gap(最初の周の途中の
シーンの IMU.npz が無い)。
比べ(⑤ --ignore-gain)の「回 3 の計画の形」の組の ON − OFF の行と checks、ほかの組の行を出す。
対照: 3-8 を試しにしない left41(fix5 c3 の form_split が出るはず)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/refute_tools_r1_A')))
import rcommon as rc  # noqa: E402

P = dict(rc.BASE, speed_ceiling=1, ceil_left_mps=4.2)
ON = dict(P, launch_boost=1)
FORMS = {'same': {}, 'left41': {'ceil_left_mps': 4.1}, 'ceil_off': {'speed_ceiling': 0}, 'auto_on': {'speed_auto': 1}}
TRIAL_MEMO = '試し(起こし直しで左を入れ直す前に走った)'


def build(form, defect, trial=True):
    D = rc.MyDay('t1_%s_%s_%d' % (form, defect, int(trial)))
    for blk, prm, dm, ramp in ((12, P, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, P, 0.21, 0.6), (15, ON, 0.20, 0.3),
                               (16, P, 0.22, 0.6), (17, ON, 0.28, 0.3), (18, P, 0.23, 0.6)):
        D.add(blk, prm, dm, ramp)
    p38 = dict(ON, **FORMS[form])
    memo = TRIAL_MEMO if trial else 'x'
    if defect == 'gap':
        log = D.add(19, p38, memo=memo, dur=45.0, vk=rc.t7.launch_keys(14.0, 0.15, 0.3, until=45.0), ai=[(14.0, 43.0)],
                    launch=[(14.0, 0.5)])
        (D.raw / log / ('%s-0002' % log) / 'IMU.npz').unlink()
    else:
        kw = {'hits': [(2.5, 6.0)]} if defect == 'hit' else {}
        log = D.add(19, p38, 0.15, 0.3, memo=memo, **kw)
        if defect == 'spin':
            rc.scale_imu_fwd(D.raw, log, 1.0, 2.2, 0.55)
    return D, log


def plan_group(summ):
    """回 3(3-…)の計画の形(天井 1・左 4.2・自動 0)の組"""
    gs = [g for g in summ if g['speed_ceiling'] == 1 and g['ceil_left_mps'] == 4.2 and g['speed_auto'] == 0
          and g.get('by_launch')]
    return gs[0] if gs else None


def main():
    res = []
    lines = []
    cases = [(f, d, True) for f in FORMS for d in ('none', 'hit', 'spin', 'gap')] + [('left41', 'none', False),
                                                                                      ('left41', 'hit', False)]
    for form, defect, trial in cases:
        D, log = build(form, defect, trial)
        j, out = rc.run_collect(D, ['--ignore-gain'], tag='%s_%s_%d' % (form, defect, int(trial)))
        summ = rc.summary_of(j)
        g = plan_group(summ)
        r38 = [r for r in j['launches'] if r['log'] == log]
        c = (g or {}).get('compare') or {}
        others = [x for x in summ if x is not g]
        rec = {'form': form, 'defect': defect, 'trial': trial,
               'plan_n_on': ((g or {}).get('by_launch', {}).get('on') or {}).get('n'),
               'plan_n_off': ((g or {}).get('by_launch', {}).get('off') or {}).get('n'),
               'plan_d_to_r1': c.get('d_to_r1_s'), 'plan_all_ok': c.get('all_ok'), 'plan_checks': c.get('checks'),
               'plan_form_split': (g or {}).get('form_split'),
               'plan_trial_keys': sorted(k for k in (g or {}) if k.startswith('trial')),
               '38_row': {k: r38[0].get(k) for k in ('trial', 'to_r1_s', 'collisions_first_lap', 'first_lap_gap')} if r38 else None,
               '38_slip': (r38[0].get('slip') or {}).get('ratio') if r38 else None,
               'other_groups': [{'form': '天井%s(%s) 自動%s' % (x['speed_ceiling'], x['ceil_left_mps'], x['speed_auto']),
                                 'keys': sorted(k for k in x if k.startswith('trial') or k in ('compare', 'form_split')),
                                 'by_launch': sorted(x.get('by_launch') or {})} for x in others]}
        res.append(rec)
        lines.append('=== 3-8 の形 %s・最初の周 %s・試し %s' % (form, defect, trial))
        lines += ['  ' + x for x in out.splitlines() if x.startswith('11/03') or x.startswith('  ON − OFF') or '試し' in x
                  or '注意: 同じ回' in x or '→ 試しの本' in x or '最初の周の記録が欠けた発進' in x]
    (rc.HERE / 'out').mkdir(exist_ok=True)
    (rc.HERE / 'out' / 't1_trial_form.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    (rc.HERE / 'out' / 't1_trial_form.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for r in res:
        print('%-8s %-5s 試し=%-5s 計画の組 ON %s・OFF %s d_to_r1 %s → %s  checks=%s  form_split=%s  trial欄=%s  ほかの組=%s' % (
            r['form'], r['defect'], r['trial'], r['plan_n_on'], r['plan_n_off'], r['plan_d_to_r1'],
            {True: '使ってよい', False: '満たさない', None: '—'}[r['plan_all_ok']],
            {k: v for k, v in (r['plan_checks'] or {}).items() if v is not True}, bool(r['plan_form_split']), r['plan_trial_keys'],
            [(o['form'], o['keys'], o['by_launch']) for o in r['other_groups']]))


if __name__ == '__main__':
    main()
