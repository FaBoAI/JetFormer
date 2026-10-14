"""離脱復活 r4 の確認: ブラウザの閉ループの走り(ck_cl_compare_r4.mjs の出力)で、出来事の成功(C2)・戻るまで(C3)を自前に数え直す(2026-10-15)。
乱しなしの線は Node の NOM-r4(同じ方策の乱しなしの 8 回)。数え方は ck_analyze_r4.py の recoveries と同じ(protocol definitions.recovered・recovery_success)
  python3 -I ck_browser_c2_r4.py <compare.json> <runs のフォルダ> <out.json>
"""
import importlib.util
import json
import sys

cmp_path, runs, out = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('ana', __file__.replace('ck_browser_c2_r4.py', 'ck_analyze_r4.py'))
sys.argv = ['ck_analyze_r4.py', runs, '/dev/null']
ana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana)
C = json.load(open(cmp_path))
ev = []
for r in C['rows']:
    b = r['browser']
    row = {'pf': r['pf'], 'frames': b['frames'], 'collisions': b['collisions'],
           'events': [{'n': i + 1, 'type': e[2], 'kind': e[3], 'cls': e[4], 'sign': None, 'ps': e[0], 'pe': e[1], 's': r['pf']['s'][e[0]] if e[0] < len(r['pf']['s']) else 0}
                      for i, e in enumerate(r['events_full'])]}
    ev += ana.recoveries(row, ana.NL['r4'])
ok = [e for e in ev if e['ok']]
ts = [e['t'] for e in ok]
res = {'runs': len(C['rows']), 'events': len(ev), 'success': f'{len(ok)}/{len(ev)}', 'C2': round(len(ok) / len(ev), 4), 'C2_ci95': ana.wilson(len(ok), len(ev)),
       'C3_median_s': round(ana.q(ts, .5), 3), 'C3_p90_s': round(ana.q(ts, .9), 3), 'events_per_run': round(len(ev) / len(C['rows']), 3),
       'failed': [{k: e[k] for k in ('type', 'kind', 'cls', 'ps', 'pe', 't', 'why')} | {'s': round(e['s'] % ana.L, 2)} for e in ev if not e['ok']]}
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
