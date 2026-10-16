"""上限の試験(S4)とブラウザの S-DIV(S5)の結果を simtest.json にまとめる(2026-10-16)。

  python simtest_report.py

読むもの: cap_test/cap_test.json(cap_analyze.py)・cap_test/starts.json・cap_test/pp_path.json・browser/closed_loop/<ページ>/SDIV-s0/*.json・
10/15 の S-DIV の結果(sim_diverse_train_20261015/eval/closed_loop/SDIV-s0)・tests/ のログ。書くもの: simtest.json だけ。
"""
import glob
import hashlib
import json
import os

R = '/home/jetson/jetformer_reports/sim_calibration_20261016'
PREV = '/home/jetson/jetformer_reports/sim_diverse_train_20261015/eval/closed_loop/SDIV-s0'
DRIVERS = ['PP-1.0', 'PP-1.2', 'PP-1.4', 'RL-fastest', 'RL-speed-0.75']
WORLD = json.load(open('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/sim/world-v3.json'))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def obj_name(cid):
    k, i = cid.rsplit('-', 1)
    i = int(i)
    if k == 'segment':
        s = WORLD['segments'][i]
        return f"板 ({s['a'][0]:.2f}, {s['a'][1]:.2f})→({s['b'][0]:.2f}, {s['b'][1]:.2f})"
    if k == 'disk':
        d = WORLD['disks'][i]
        return f"{'支柱の脚' if d['kind'] == 'post-foot' else '支柱'} ({d['x']:.2f}, {d['z']:.2f})"
    return cid


def browser_run(p):
    r = json.load(open(p))
    m = r.get('metrics') or {}
    fs = r.get('final_state') or {}
    ce = (m.get('collision_events') or [None])[0]
    oc = r.get('outcome') or {}
    return {'name': r.get('name'), 'frames': len(r.get('actions') or []), 'gates': m.get('passed_gates'), 'collisions': m.get('collisions'),
            'finished': bool(m.get('finished')), 'reason': fs.get('reason') or oc.get('reason'), 'elapsed_s': round(m.get('elapsed') or 0, 3),
            'collision': {'collider': ce['collider'], 'object': obj_name(ce['collider']), 'x': round(ce['x'], 3), 'z': round(ce['z'], 3), 't': round(ce['time'], 3)} if ce else None,
            'calibration': ((r.get('model') or {}).get('calibration') or {}).get('name'),
            'calibration_filter': ((r.get('model') or {}).get('calibration') or {}).get('filter'),
            'max_speed': round(max((q.get('speed') or 0) for q in (r.get('trace') or [{}])), 3)}


def main():
    ct = json.load(open(os.path.join(R, 'cap_test', 'cap_test.json')))
    S, C = ct['summaries'], ct['comparisons']
    table = {}
    for phys in ('P_cal', 'P_default'):
        for d in DRIVERS:
            for g in ('OFF', 'ON-O25'):
                k = '__'.join([d, g, phys])
                s = S.get(k)
                if not s:
                    continue
                table[k] = {'complete': s['complete'], 'collisions': s['collisions'], 'collisions_per_lap': s['collisions_per_lap'],
                            'laps_driven': s['laps_driven'], 'collisions_by_section': {a: b for a, b in s['collisions_by_section'].items() if b},
                            'collision_objects': [{'section': o['section'], 'object': obj_name(o['collider']), 'n': o['n']} for o in s['collision_objects']],
                            'R1_passes': s['R1']['passes'], 'R1_collisions': s['R1']['collisions'], 'R1_entry_v_median': s['R1']['entry_v_median'],
                            'R1_entry_v_max': s['R1']['entry_v_max'], 'R1_entry_gt_2_5_frac': s['R1']['entry_gt_2_5_frac'],
                            'GM_passes': s['GM']['passes'], 'GM_collisions': s['GM']['collisions'], 'GM_start_v_median': s['GM']['start_v_median'],
                            'DE_max_v_median': s['DE']['max_v_median'], 'lap_time_median_s': s['lap_time']['median'],
                            'governor_active_frac': {z: v['active_frac'] for z, v in (s.get('governor') or {}).items()},
                            'governor_brake_frames': sum(v['brake_frames'] for v in (s.get('governor') or {}).values()) if s.get('governor') else None}
    # ブラウザ
    browser = {}
    for page in ('index', 'calib', 'calib_cap'):
        runs = [browser_run(p) for p in sorted(glob.glob(os.path.join(R, 'browser', 'closed_loop', page, 'SDIV-s0', 'SDIV-s0-*.json')))]
        if runs:
            single = [x for x in runs if x['name'].endswith('-single')]
            browser[page] = {'runs': runs, 'n': len(single), 'completed': sum(x['finished'] and x['collisions'] == 0 for x in single),
                             'collided': sum(x['collisions'] > 0 for x in single)}
    prev = [browser_run(p) for p in sorted(glob.glob(os.path.join(PREV, 'SDIV-s0-E*-single.json')))]
    browser['index_10_15_reference'] = {'runs': prev, 'n': len(prev), 'completed': sum(x['finished'] and x['collisions'] == 0 for x in prev),
                                        'collided': sum(x['collisions'] > 0 for x in prev)}
    e0_same = None
    pa, pb = os.path.join(R, 'browser', 'closed_loop', 'index', 'SDIV-s0', 'SDIV-s0-E0-single.json'), os.path.join(PREV, 'SDIV-s0-E0-single.json')
    if os.path.exists(pa):
        a, b = json.load(open(pa)), json.load(open(pb))
        e0_same = {'actions_identical': a['actions'] == b['actions'], 'trace_identical': a['trace'] == b['trace']}
    browser['index_E0_vs_10_15'] = e0_same
    screens = sorted(os.path.relpath(p, os.path.join(R, 'screens')) for p in glob.glob(os.path.join(R, 'screens', '**', '*.png'), recursive=True))
    out = {
        'schema': 'jf-sim-calib-simtest/v1',
        'request': '10/16「今週は走行できないので、ほかにできることは?」→ 選んだ「7 シミュレーターを実車に近づける」の、区間の速度上限 v1(R1 2.5・GM 2.2・DE 3.0 m/s)の'
                   'シミュレーターでの試験(S4・S5)。実車は走らせていない',
        'protocol_sha256': open(os.path.join(R, 'protocol.sha256')).read().split()[0],
        'inputs': {'P_cal_sha256': '4b0e9348e367e64d93c4dc491de26b5dd5c7e2375a60f6f618f0d24ad85ddc14',
                   'sim_core_sha256': 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
                   'starts': json.load(open(os.path.join(R, 'cap_test', 'starts.json')))['starts'],
                   'pp_path': {k: v for k, v in json.load(open(os.path.join(R, 'cap_test', 'pp_path.json'))).items() if k in ('use', 'sigma_m', 'length_m', 'checks')},
                   'governor': 'v1.1(GOVERNOR_V1_SIM: 上限 R1 2.5・GM 2.2・DE 3.0、ブレーキは R1_app で上限 + 0.3 かつ 1 m/s 以上、中立 + 200 µs、保つ 0.2 秒・再び始めない 0.5 秒、'
                               '舵が右 30% 以上ならブレーキしない、区間を出たら 1.5 m/s² で上限を戻す、上限 → 指令はそのまま)'},
        'conditions_run': sorted(S.keys()),
        'episodes': sum(s['starts'] for s in S.values()),
        'verdicts': ct['verdicts'],
        'primary_table': table,
        'comparisons': C,
        'r1_275_vs_25': ct['r1_275_vs_25'],
        'sim_real_consistency': ct['sim_real_consistency'],
        'real_reference': ct['real_reference'],
        'browser_sdiv': browser,
        'screens': screens,
        'tests': {'cap_tests_log': 'tests/s4_cap_tests.log', 't7': 'tests/t7_cap_browser.json', 't8_vectors': 'tests/t8_phase_vectors.json'},
        'files': {'cap_test_json': 'cap_test/cap_test.json', 'runs': 'cap_test/runs/*.json', 'figures': ['fig/cap_r1_entry.png', 'fig/cap_collisions_sections.png', 'fig/cap_map_collisions.png'],
                  'browser_page': 'sim/dist/calib_cap.html', 'browser_core': 'sim/dist/_calib/sim-core-jf-calib-v1-real-20261010-cap-v1.1-O25.js'},
    }
    with open(os.path.join(R, 'simtest.json.tmp'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(os.path.join(R, 'simtest.json.tmp'), os.path.join(R, 'simtest.json'))
    print(json.dumps({k: (v['n'], v['completed'], v['collided']) for k, v in browser.items() if isinstance(v, dict) and 'n' in v}, ensure_ascii=False))
    print(json.dumps(e0_same))


if __name__ == '__main__':
    main()
