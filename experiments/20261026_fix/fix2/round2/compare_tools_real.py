"""直す 2 回目: 夜の道具の「前と同じ」を本物の記録で確かめる(読むだけ)。直す前の写し(before_round2)と今の
section_scorecard.score_log・standing_3lap.analyze_log・car_check.check_log を、10/23・10/24 の AI の記録に通して比べる。
新しい欄(start_no・coll_before・counted・midrun・collisions の max_g など・car_check の collisions)は除いて比べ、
新しい欄の値も数える(midrun が空・counted = valid か など)。出力 out/compare_tools_real.json"""
import importlib.util
import json
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
BEF = JF / 'experiments/20261026_fix/fix2/before_round2/tools'
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
from jetformer import config  # noqa: E402
import label_events as le  # noqa: E402,F401
import section_scorecard as ss1  # noqa: E402
import standing_3lap as s31  # noqa: E402
import car_check as cc1  # noqa: E402


def load(name, path, deps):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in deps}
    sys.modules.update(deps)
    try:
        spec.loader.exec_module(m)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return m


ss0 = load('ss_before', BEF / 'section_scorecard.py', {})
s30 = load('s3_before', BEF / 'standing_3lap.py', {'section_scorecard': ss0})
cc0 = load('cc_before', BEF / 'car_check.py', {'section_scorecard': ss0})
NEW_STAND = ('start_no', 'coll_before', 'counted')


def strip(o, drop):
    if isinstance(o, dict):
        return {k: strip(v, drop) for k, v in o.items() if k not in drop}
    if isinstance(o, list):
        return [strip(x, drop) for x in o]
    return o


def jd(o):
    return json.loads(json.dumps(o, default=ss1._json_default))


logs, _ = s31.day_logs('20261023')
l2, _ = s31.day_logs('20261024')
logs += l2
out = {'logs': len(logs), 'score_log_same': 0, 'score_log_diff': [], 'standing_same': 0, 'standing_diff': [],
       'car_check_same': 0, 'car_check_diff': [], 'midrun_nonempty': [], 'not_counted_valid': [], 'errors': []}
lab = config.DATA_DIR / 'labels'
for lg in logs:
    try:
        a = jd(ss0.score_log(lg, lab, config.RAW_DIR))
        b = jd(ss1.score_log(lg, lab, config.RAW_DIR))
        if a == b:
            out['score_log_same'] += 1
        else:
            out['score_log_diff'].append(lg)
        ra = jd(s30.analyze_log(lg, config.RAW_DIR, lab, None))
        rb = jd(s31.analyze_log(lg, config.RAW_DIR, lab, None))
        if rb.get('midrun'):
            out['midrun_nonempty'].append([lg, rb['midrun']])
        out['not_counted_valid'] += [[lg, s['t_ai'], s['coll_before']] for s in rb.get('standing') or []
                                     if s.get('valid') and not s.get('counted')]
        rb2 = dict(rb)
        rb2.pop('midrun', None)
        if 'collisions' in rb2:
            rb2['collisions'] = {k: v for k, v in rb2['collisions'].items() if k in ('all', 'strong')}
        rb2 = strip(rb2, NEW_STAND)
        if 'settings' in rb2:
            rb2['settings'] = {k: v for k, v in rb2['settings'].items() if k != 'midrun'}
        if ra == rb2:
            out['standing_same'] += 1
        else:
            out['standing_diff'].append([lg, sorted(k for k in set(ra) | set(rb2) if ra.get(k) != rb2.get(k))])
        ca = jd(cc0.check_log(lg, lab, config.RAW_DIR))
        cb = jd(cc1.check_log(lg, lab, config.RAW_DIR))
        for x in (ca, cb):
            x.pop('created_at', None)
            x.pop('seconds', None)
        cb.pop('collisions', None)
        if ca == cb:
            out['car_check_same'] += 1
        else:
            out['car_check_diff'].append([lg, sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))])
    except Exception as e:      # noqa: BLE001
        out['errors'].append([lg, '%s: %s' % (type(e).__name__, e)])
p = Path(__file__).parent / 'out'
p.mkdir(exist_ok=True)
(p / 'compare_tools_real.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: (v if not isinstance(v, list) else v[:8]) for k, v in out.items()}, ensure_ascii=False, indent=1))
