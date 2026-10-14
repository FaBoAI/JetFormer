"""昇格の段(2026-10-15、r3/promote): 第 4 回で記録した見本(no_wall・recover・line)と、記録の段の見本(fastest・speed)のばらつきの物差しを
jetformer_reports/sim_rl_modes_20261013/r3/variation.json にまとめる(記録の段の record/make_variation_json.py と同じ形 + 第 4 回の群)。

  入力: r3/promote/record/variation_raw_5modes.json(fastest・no_wall r4・recover r4・speed・line r4 の 5 モード)、
        r3/promote/record/variation_raw_4modes.json(記録の段と同じ 4 モードの組み合わせ。no_wall・recover だけ r4 に替えたもの)、
        r3/promote/record/replay/<key>.json(replay_check_r4.mjs)・verify/<key>.json(record/verify_mode_dataset.py)・
        recover は check_recorded/<key>.json(check_recorded_scenes_rc4.mjs)、前の記録の段の variation.json(比べ)
    python3 -I r3/promote/make_variation_json_r3.py <プランの ID>
"""
import datetime
import json
import sys
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
R3 = R / 'r3'
OUT = R3 / 'promote' / 'record'
pid = sys.argv[1]
plan = json.loads(Path(f'/home/jetson/jetformer/data/sim_rl/plans/{pid}.json').read_text())
raw5 = json.loads((OUT / 'variation_raw_5modes.json').read_text())
raw4 = json.loads((OUT / 'variation_raw_4modes.json').read_text())
prev = json.loads((R / 'variation.json').read_text())
T = raw5['teacher']
t3, t7, t2 = T['all']['VM3_m']['mean_abs_dev'], T['first15']['VM7']['cells'], T['all']['VM2_us']['mean_abs_dev']


def compact(g, vm7=None):
    e = g['extra']
    out = {'scenes': g['scenes'], 'frames': g['frames'],
           'VM1_steer_time_aligned_us': g['VM1_us'], 'VM2_steer_position_aligned_us': g['VM2_us']['mean_abs_dev'],
           'VM3_lateral_spread_m': g['VM3_m']['mean_abs_dev'], 'VM3_bins_used': g['VM3_m']['bins_used'],
           'VM4_speed_spread_mps': g['VM4']['mean_abs_dev'], 'lap1_s': g['VM4']['lap1'], 'lap2_s': g['VM4']['lap2'],
           'VM5_clearance': {'scene_min_m': g['VM5']['scene_min'], 'frame_p05_m': g['VM5']['frame_p05'], 'frame_median_m': g['VM5']['frame_median']},
           'VM6_recovery': g['VM6'], 'VM7_state_cells': vm7 if vm7 is not None else g['VM7']['cells'], 'VM7_frames_used': g['VM7']['frames_used'],
           'start_gates_distinct': len(e['start_gates']), 'start_offset_m': {k: e['start_offset_m'].get(k) for k in ('min', 'max')},
           'start_yaw_deg': {k: e['start_yaw_deg'].get(k) for k in ('min', 'max')}, 'scenarios': e['scenarios'], 'speed_levels_c_v': e['speed_levels'],
           'speed_mps_after3m': {k: e['speed_mps_after3m'].get(k) for k in ('p05', 'median', 'p95', 'mean')},
           'abs_lateral_m_after3m': {k: e['ey_abs_m_after3m'].get(k) for k in ('median', 'p95')},
           'clearance_frame_m': {k: e['clearance_frame_m'].get(k) for k in ('p05', 'median')},
           'shield_frames': e['shield_frames'], 'collisions': e['collisions']}
    out['ratio_vs_teacher'] = {'VM2': out['VM2_steer_position_aligned_us'] / t2, 'VM3': out['VM3_lateral_spread_m'] / t3, 'VM7': out['VM7_state_cells'] / t7}
    return out


def pooled(raw):
    return {**compact(raw['pooled'], vm7=raw['pooled']['VM7_equal_frames']['cells']), 'scenes_per_mode': raw['pooled']['scenes_per_mode'],
            'modes': list(raw['inputs']), 'VM7_equal_frames': raw['pooled']['VM7_equal_frames'], 'leave_one_out': raw['pooled_leave_one_out']}


val = {}
for k in plan['modes']:
    imp = (plan.get('imports') or {}).get(k) or {}
    if imp.get('state') != 'done':
        val[k] = {'import': imp}
        continue
    rp = json.loads((OUT / 'replay' / f'{k}.json').read_text())
    vf = json.loads((OUT / 'verify' / f'{k}.json').read_text())
    sc = rp['scenes']
    v = {
        'dataset': imp['dataset'],
        'user_style_check': {'pass': vf['pass'], 'scenes_pass': vf['pass_scenes'], 'scenes': vf['scenes'], 'teacher_same_all_scenes': vf['teacher_same_all_scenes'],
                             'card': vf['card'], 'masked_frames': vf['masked_frames'], 'tags': vf['tags'], 'genre_frames': vf['genre_frames'],
                             'expanded_scenes': {kk: vf['expanded_scenes'][kk] for kk in ('log', 'created', 'all_exist', 'all_tags_ok')},
                             'min_unique_rgb': min(p['unique_decoded_camera_images'] for p in vf['per_scene']),
                             'min_unique_tof': min(p['unique_packed_tof_frames'] for p in vf['per_scene']),
                             'max_imu_float32_error': max(p['max_imu_float32_error'] for p in vf['per_scene'])},
        'replay': {'open_loop_max_pose_err_m': max(s['open_loop_max_pose_err'] for s in sc),
                   'teacher_max_abs_steering': max(s['teacher_max_abs_steering'] for s in sc),
                   'teacher_max_abs_speed': max(s['teacher_max_abs_speed'] for s in sc),
                   'teacher_info_mismatch_frames': sum(s['teacher_info_mismatch'] for s in sc),
                   'pass_identity_1e-6': rp['pass_identity'], 'identity_fail_scenes': [s['plan_scene'] for s in sc if not s['pass_identity']],
                   'rules': rp['rules'], 'pass_rules': rp['pass_rules'], 'rule_fail_scenes': [{'scene': s['plan_scene'], 'why': s['rule_fail']} for s in sc if s['rule_fail']],
                   'collisions': sum(s['collisions'] for s in sc), 'valid_laps': {str(n): sum(1 for s in sc if s['valid_laps'] == n) for n in sorted({s['valid_laps'] for s in sc})},
                   'min_clearance_m': min(s['min_clearance_m'] for s in sc), 'max_stall_s': max(s['stall_s'] for s in sc),
                   'max_stall_r3_frames': max(s['r4']['stall_r3_frames'] for s in sc), 'max_shield_fraction': max(s['r4']['shield_fraction'] for s in sc),
                   'reverse_frames': sum(s['r4']['back_frames'] + s['r4']['neg_speed_cmd_frames'] for s in sc),
                   'gate_misses': sum(s['r4']['gate_misses'] for s in sc), 'min_progress_m': min(s['r4']['progress_m'] for s in sc),
                   'learn0_frames': sum(s['learn0_frames'] for s in sc), 'shield_frames': sum(s['shield_frames'] for s in sc),
                   'prediction_first_diff_frame': [min((s['prediction']['first_action_diff_frame'] for s in sc if s['prediction']['first_action_diff_frame'] is not None), default=None),
                                                   max((s['prediction']['first_action_diff_frame'] for s in sc if s['prediction']['first_action_diff_frame'] is not None), default=None)],
                   'prediction_same_scenes': sum(1 for s in sc if s['prediction']['first_action_diff_frame'] is None),
                   'prediction_max_pose_diff_m': max(s['prediction']['max_pose_diff_m'] for s in sc)}}
    cr = OUT / 'check_recorded' / f'{k}.json'
    if cr.exists():
        c = json.loads(cr.read_text())
        v['check_recorded_scenes_rc4'] = {'scenes': c.get('scenes'), 'accept': c.get('accept'), 'rule': c.get('rule'),
                                          'rejected': [{'scene': r['scene'], 'gate_misses': r.get('gate_misses'), 'stall': r.get('stall'), 'progress_ok': r.get('progress_ok')}
                                                       for r in c.get('results', []) if not r.get('accept')]}
    val[k] = v

fails = [{'scene': sid, **{kk: v.get(kk) for kk in ('frames', 'collisions', 'reason', 'replaced_by', 'session_id')}} for sid, v in plan['progress'].items() if v.get('status') == 'failed']
prev_modes = {k: prev['modes'][k] for k in prev['modes']}
out = {
    'schema': 'jf-sim-rl-modes-variation-summary/v1', 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'request': 'シミュレーターの走行に、データセットのばらつきが出る強化学習モードを用意(2026-10-13 21:20)。第 4 回の昇格の段 = 合格した 3 モード(絶対壁にぶつからない・離脱復活・経路いろいろ)の見本の記録とばらつき',
    'plan': {'id': pid, 'minutes': plan['minutes'], 'scenes': len(plan['order']), 'val_ratio': plan['val_ratio'], 'seed': plan['seed'],
             'calibration': plan['calibration'], 'created': plan['created_at'], 'finished': plan['updated_at'], 'failed_scenes': fails,
             'recorded_via': 'http://192.168.55.1:8080 の JetFormer「シミュレーター」タブ → 強化学習モードのカード → プラン → 新しいタブで「開始」(DISPLAY :0 の snap Firefox)'},
    'datasets': {**{k: ((plan.get('imports') or {}).get(k) or {}).get('dataset') for k in plan['modes']},
                 'fastest': raw5['inputs'].get('fastest', {}).get('dataset'), 'speed': raw5['inputs'].get('speed', {}).get('dataset')},
    'inputs_5modes': raw5['inputs'], 'teacher_dataset': T['dataset'],
    'definitions': {**prev['definitions'],
                    'r3_note': '物差しのコードは記録の段と同じ(record/variation.mjs・rl/variation_metrics.mjs)。離脱復活の立て直しの乱しなしの線は、登録簿の r4 の方策で作り直したもの'
                               '(r3/promote/record/nominal_line_recover_r4.json)。全モードの合わせはモードごとに同じシーン数・VM7 はモードごとに同じフレーム数(合計 9,000)'},
    'teacher': {'all60': compact(T['all']), 'first15': compact(T['first15'])},
    'modes': {k: compact(g) for k, g in raw5['modes'].items()},
    'pooled_5modes': pooled(raw5),
    'pooled_4modes_r4': pooled(raw4),
    'previous_record_stage': {'file': str(R / 'variation.json'), 'pooled_4modes': {kk: prev['pooled'][kk] for kk in ('VM2_steer_position_aligned_us', 'VM3_lateral_spread_m', 'VM4_speed_spread_mps', 'VM7_state_cells', 'ratio_vs_teacher')},
                              'modes': {k: {kk: v[kk] for kk in ('VM2_steer_position_aligned_us', 'VM3_lateral_spread_m', 'VM4_speed_spread_mps', 'VM7_state_cells', 'ratio_vs_teacher')} for k, v in prev_modes.items()}},
    'recover_detail': raw5['modes'].get('recover', {}).get('recover'),
    'judged_5modes': raw5['judged'], 'judged_4modes_r4': raw4['judged'],
    'validation': val,
    'files': {'raw_5modes': str(OUT / 'variation_raw_5modes.json'), 'raw_4modes': str(OUT / 'variation_raw_4modes.json'), 'figdata': str(OUT / 'figdata_5modes.json'),
              'replay': str(OUT / 'replay'), 'verify': str(OUT / 'verify'), 'figures': [str(p) for p in sorted((R3 / 'fig').glob('promote_record_*.png'))]},
}
(R3 / 'variation.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: {kk: round(v['ratio_vs_teacher'][kk], 2) for kk in v['ratio_vs_teacher']} for k, v in {**out['modes'], 'pooled5': out['pooled_5modes'], 'pooled4': out['pooled_4modes_r4']}.items()}, ensure_ascii=False))
print(json.dumps({k: (v.get('replay') and {'id': v['replay']['pass_identity_1e-6'], 'rules': v['replay']['pass_rules'], 'clr': v['replay']['min_clearance_m'],
                                          'laps': v['replay']['valid_laps'], 'gm': v['replay']['gate_misses'], 'check': v['user_style_check']['pass']}) for k, v in val.items()}, ensure_ascii=False))
