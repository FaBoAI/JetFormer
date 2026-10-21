#!/usr/bin/env python3
"""V-drive: v_run_all.sh の結果を全部比べて、1 つの JSON にまとめる。

    python -I v_summarize.py <結果の置き場> <書く JSON>
"""
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v_compare as vc  # noqa: E402
import v_g1  # noqa: E402

D = Path(sys.argv[1])
LOGS = ['20261010-172910', '20261010-173445', '20261010-203924', '20261010-204208', '20261010-204605',
        '20261010-171639', '20261010-172505', '20261010-181504']
ZN = ('none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE')
out = {'off': [], 'on': [], 'toggle': [], 'fail': [], 'g1': [], 'zone_ms': [], 'unknown_share': []}
for log in LOGS:
    p = lambda n: D / n  # noqa: E731
    if not p('new_%s_perturb_z1_ifail.pkl' % log).exists():
        continue
    L = {n: vc.load(p(n)) for n in ['old_%s_%s.pkl' % (log, v) for v in ('plain', 'perturb', 'dry')]
         + ['new_%s_%s_z0.pkl' % (log, v) for v in ('plain', 'perturb', 'dry')]
         + ['new_%s_%s.pkl' % (log, v) for v in ('dry_z1', 'plain_z1', 'perturb_z1', 'toggle', 'perturb_z1_zfail', 'perturb_z1_ifail')]}
    g = lambda n: L[n]  # noqa: E731
    for v in ('plain', 'perturb', 'dry'):
        r = vc.cmp_off(g('old_%s_%s.pkl' % (log, v)), g('new_%s_%s_z0.pkl' % (log, v)))
        out['off'].append(r)
        out['zone_ms'].append({'log': log, 'variant': v, **r['zone_ms_per_wait'], 'loop_old': r['loop_ms_old'],
                               'loop_new': r['loop_ms_new']})
    # dry_run で zone_caps 1 でも変更前と同じ(層は動かない)
    r = vc.cmp_off(g('old_%s_dry.pkl' % log), g('new_%s_dry_z1.pkl' % log))
    r['variant'] = 'dry_zone_on'
    r['ok'] = r['pairs_equal'] and r['row_field_diffs'] == 0 and not r['new_cap_applied_any']
    out['off'].append(r)
    for v in ('plain', 'perturb'):
        out['on'].append(vc.cmp_on(g('new_%s_%s_z0.pkl' % (log, v)), g('new_%s_%s_z1.pkl' % (log, v))))
    out['toggle'].append(vc.cmp_toggle(g('old_%s_plain.pkl' % log), g('new_%s_toggle.pkl' % log)))
    for f in ('zfail', 'ifail'):
        r = vc.cmp_fail(g('old_%s_perturb.pkl' % log), g('new_%s_perturb_z1_%s.pkl' % (log, f)))
        r['kind'] = f
        out['fail'].append(r)
    for n in ('new_%s_plain_z1.pkl' % log, 'new_%s_plain_z0.pkl' % log):
        out['g1'].append(v_g1.check(p(n)))
    # ON で出力を変えたフレームの区間の内訳(位相が不確か = unknown の 3.0 m/s がどれだけ効くか)
    on = g('new_%s_plain_z1.pkl' % log)
    rows = [r for _, r in on['rows']]
    app = collections.Counter(ZN[int(r['cap_zone'])] for r in rows if r['cap_applied'])
    act = collections.Counter(ZN[int(r['cap_zone'])] for r in rows if r['cap_mode'] >= 0)
    out['unknown_share'].append({'log': log, 'applied_by_zone': dict(app), 'active_by_zone': dict(act),
                                 'applied': sum(app.values()), 'active': sum(act.values())})
    del L

summ = {k: {'n': len(v), 'ok': sum(bool(x.get('ok')) for x in v)} for k, v in out.items() if k in ('off', 'on', 'toggle', 'fail', 'g1')}
zm = [x for x in out['zone_ms']]
summ['zone_ms'] = {'mean_of_means': float(np.mean([x['mean'] for x in zm])) if zm else None,
                   'max_p99': max(x['p99'] for x in zm) if zm else None, 'max_max': max(x['max'] for x in zm) if zm else None}
ua = collections.Counter()
for x in out['unknown_share']:
    ua.update(x['applied_by_zone'])
summ['applied_by_zone_total'] = dict(ua)
out['summary'] = summ
with open(sys.argv[2], 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(summ, ensure_ascii=False, indent=1))
for k in ('off', 'on', 'toggle', 'fail', 'g1'):
    for x in out[k]:
        if not x.get('ok'):
            print('NG', k, json.dumps(x, ensure_ascii=False, default=str)[:900])
