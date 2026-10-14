"""確認の段(review、2026-10-14): 確認の結果を evaluation.json の review_20261014 に入れる(evaluation.md は eval/render_md.py が作る)

    python3 -I attach_review.py

読むもの(すべて review/ の下。この段の道具が作った): own_check_closed_loop.json・own_check_rl_teacher_records.json・
random_starts/{draw.json, own_check.json, user_verifier_lap_items.json}・recorded_scenes/{draw.json, own_check.json}・
derived_numbers.json・dataset_check/*.json・sim_manifest_check.json・pytest_full.log。
evaluation.json のほかの欄は変えない(変える前の写しは review/evaluation.before_review.json)。
"""
import json
import re
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
RV = REP / 'review'


def load(p):
    return json.loads((RV / p).read_text(encoding='utf-8'))


def own_summary(d):
    laps = [l for v in d.values() for l in v['laps_own']]
    return {'runs': len(d), 'agree_with_sim': sum(v['agree_with_sim'] for v in d.values()),
            'max_trace_diff': max(v['max_trace_diff'] for v in d.values()),
            'laps_counted': len(laps), 'laps_best_lap_valid': sum(l['best_lap_valid'] for l in laps)}


def main():
    cl, rt = load('own_check_closed_loop.json'), load('own_check_rl_teacher_records.json')
    rs, rsd, uv = load('random_starts/own_check.json'), load('random_starts/draw.json'), load('random_starts/user_verifier_lap_items.json')
    sc, scd = load('recorded_scenes/own_check.json'), load('recorded_scenes/draw.json')
    dn = load('derived_numbers.json')
    ri, lv, sh = load('dataset_check/reimport_check.json'), load('dataset_check/live_vs_reimport.json'), load('dataset_check/rebuild_label_shift.json')
    sm = load('sim_manifest_check.json')
    log = (RV / 'pytest_full.log').read_text(encoding='utf-8', errors='replace')
    m = re.search(r'(\d+) failed, (\d+) passed, (\d+) skipped, (\d+) deselected', log)
    failed = re.findall(r'^FAILED (\S+)', log, re.M)
    sr = dn['SR_deviation_from_RL_line']
    fs = dn['first_straight_nearest_obstacles']
    kinds = {}
    for v in dn['SR_collisions_12'].values():
        kinds[v['kind']] = kinds.get(v['kind'], 0) + 1
    block = {
        'note': '確認の段(review.md)で、Python の自前の積分・判定(review/independent_lap_check.py。JS のコードを使わない)などで数え直した値',
        'independent_recheck': {
            'closed_loop_59': own_summary(cl),
            'rl_teacher_records_18': own_summary(rt),
            'random_starts': {'draw': rsd, 'results': {k: {'reason': v['reason_own'], 'laps_s': [l['time_s'] for l in v['laps_own']],
                                                           'best_lap_valid': [l['best_lap_valid'] for l in v['laps_own']],
                                                           'min_clearance_m': v['min_clearance_own_m'], 'agree_with_sim': v['agree_with_sim']}
                                                       for k, v in rs.items()},
                              'user_verifier_lap_items': {k: v['lap_items_pass'] for k, v in uv.items()}},
            'recorded_scenes_2_random': {'draw': scd, 'results': {k: {'laps_s': [l['time_s'] for l in v['laps_own']],
                                                                      'best_lap_valid': [l['best_lap_valid'] for l in v['laps_own']],
                                                                      'agree_with_sim': v['agree_with_sim']} for k, v in sc.items()}},
        },
        'failure_shape_numbers': {
            'first_straight_nearest_obstacle_m': {'RL_E0': {'id': fs['RL_E0'][0]['id'], 'kind': fs['RL_E0'][0]['kind'], 'd': fs['RL_E0'][0]['d']},
                                                  'teacher_E0': {'id': fs['teacher_E0'][0]['id'], 'kind': fs['teacher_E0'][0]['kind'], 'd': fs['teacher_E0'][0]['d']},
                                                  'definition': fs['definition']},
            'rail_segment11_min_clearance_m': {'RL_E0': dn['rail_segment11_clearance_first_straight']['RL_E0']['min_clearance_m'],
                                               'teacher_E0': dn['rail_segment11_clearance_first_straight']['teacher_E0']['min_clearance_m']},
            'whole_lap_min_clearance_m': {'RL_E0': dn['rail_segment11_clearance_first_straight']['RL_E0_whole_lap_min_clearance_any_m'],
                                          'teacher_E0': dn['rail_segment11_clearance_first_straight']['teacher_E0_whole_lap_min_clearance_any_m']},
            'SR_dev_at_crash_vs_RL_E0_m': sr['at_crash_vs_E0_range_m'], 'SR_dev_at_crash_same_start_m': sr['at_crash_same_start_range_m'],
            'SR_dev_at_0p5s_same_start_m': sr['at_0p5s_same_start_range_m'],
            'gate_point4_lateral_m': {k: (v or {}).get('lateral_m') for k, v in dn['gate_point4'].items()},
            'SR_s0_E0_relabel_minus_used_steer_rad_frames15_26': dn['SR_s0_E0_relabel_steer_frames_15_26']['rl_minus_used_range_rad'],
            'SR_collision_kinds_12': kinds,
            'source': 'review/derived_numbers.json(review/derived_numbers.py)'},
        'dataset': {'reimport_with_current_code': {k: ri[k] for k in ('listed', 'card', 'label_mask_applied', 'scenes_tab', 'train_reader', 'archive_sha256')},
                    'live_equals_fresh_reimport_files': lv['same'], 'live_differs_only': lv['diff'],
                    'rebuild_from_scenes_tab_label_shift': sh},
        'user_simulator_manifest': {'files': sm['manifest_files'], 'ok': sm['checked_ok'], 'mismatch': len(sm['mismatch']), 'missing': len(sm['missing']),
                                    'extra': sm['extra_not_in_manifest'], 'dataset_zip_ok': sm['dataset_zip']['ok']},
        'pytest_full': {'failed': int(m.group(1)), 'passed': int(m.group(2)), 'skipped': int(m.group(3)), 'deselected': int(m.group(4)),
                        'failed_tests': failed, 'this_experiment_tests': '13 passed(experiments/20261013_sim_rl_bestlap)',
                        'attribution': '失敗 5 件はほかの作業(sim_rl_modes の作業中の試験 4 件・シミュレーター統合の tests/test_sim_runner.py 1 件)。'
                                       'この作業は jetformer/ のコードを変えていない'},
    }
    ev = json.loads((REP / 'evaluation.json').read_text(encoding='utf-8'))
    ev['review_20261014'] = block
    (REP / 'evaluation.json').write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(block, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main()
