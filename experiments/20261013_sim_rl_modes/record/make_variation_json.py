"""記録の段: variation_raw.json(record/variation.mjs)・確かめ(replay/・verify/)・プランから、報告用の variation.json を作る(2026-10-14)
    python -I record/make_variation_json.py <プランの ID>
"""
import datetime
import json
import sys
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
OUT = R / 'record'
pid = sys.argv[1]
plan = json.loads(Path(f'/home/jetson/jetformer/data/sim_rl/plans/{pid}.json').read_text())
raw = json.loads((OUT / 'variation_raw.json').read_text())
T = raw['teacher']
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


val = {}
for k in plan['modes']:
    rp = json.loads((OUT / 'replay' / f'{k}.json').read_text())
    vf = json.loads((OUT / 'verify' / f'{k}.json').read_text())
    sc = rp['scenes']
    val[k] = {
        'dataset': plan['imports'][k]['dataset'],
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
                   'rules': rp['rules'], 'pass_rules': rp['pass_rules'], 'rule_fail_scenes': [s['plan_scene'] for s in sc if s['rule_fail']],
                   'collisions': sum(s['collisions'] for s in sc), 'valid_laps': {str(n): sum(1 for s in sc if s['valid_laps'] == n) for n in sorted({s['valid_laps'] for s in sc})},
                   'min_clearance_m': min(s['min_clearance_m'] for s in sc), 'max_stall_s': max(s['stall_s'] for s in sc),
                   'learn0_frames': sum(s['learn0_frames'] for s in sc), 'shield_frames': sum(s['shield_frames'] for s in sc),
                   'prediction_first_diff_frame': [min(s['prediction']['first_action_diff_frame'] for s in sc), max(s['prediction']['first_action_diff_frame'] for s in sc)],
                   'prediction_max_pose_diff_m': max(s['prediction']['max_pose_diff_m'] for s in sc)},
    }
sens = OUT / 'replay' / 'recover_s030_sensitivity.json'
if 'recover' in val and sens.exists():
    d = json.loads(sens.read_text())
    val['recover']['replay']['identity_note'] = (
        f"{d['plan_scene']} の {len(d['frames'])} フレーム(f{d['frames'][0]['frame']}〜{d['frames'][-1]['frame']})で、記録の速さの指令が Node の教師と最大 "
        f"{max(f['d_rec_vs_A'] for f in d['frames']):.4f} m/s 違う(舵は同じ・info も同じ)。同じ開ループの再生の状態を x だけ {-d['eps']:.0e} m ずらして教師に与えると、"
        f"{sum(1 for f in d['frames'] if f['rec_matches_variant'] == 2)}/{len(d['frames'])} フレームが記録と 1e-9 以内で一致する。参照ループの頂点で向きのずれの観測が跳ぶ所で、"
        "ブラウザと Node の状態の 1e-14 m ほどの違いが別の側に落ちたもの(方策・ランタイムの違いではない)。ほかの 59 シーンは 1e-6 以内")
    val['recover']['replay']['sensitivity_file'] = str(sens)
fails = [{'scene': sid, **{kk: v.get(kk) for kk in ('frames', 'collisions', 'reason', 'replaced_by', 'session_id')}} for sid, v in plan['progress'].items() if v.get('status') == 'failed']
out = {
    'schema': 'jf-sim-rl-modes-variation-summary/v1', 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'request': 'シミュレーターの走行に、データセットのばらつきが出る強化学習モードを用意(2026-10-13 21:20)。この段 = 見本の記録とばらつきの物差し',
    'plan': {'id': pid, 'minutes': plan['minutes'], 'scenes': len(plan['order']), 'val_ratio': plan['val_ratio'], 'seed': plan['seed'],
             'calibration': plan['calibration'], 'started': '2026-10-14T11:25:37', 'finished': plan['updated_at'], 'failed_scenes': fails,
             'recorded_via': 'http://192.168.55.1:8080 の JetFormer「シミュレーター」タブ → 強化学習モードのカード → プラン → 新しいタブで「開始」(DISPLAY :0 の Firefox 157)'},
    'datasets': {k: plan['imports'][k]['dataset'] for k in plan['modes']},
    'teacher_dataset': T['dataset'],
    'definitions': {
        'source': 'protocol.json variation_metrics(VM1〜VM7)。記録した frames.jsonl の操作前の姿勢を方策の観測と同じ射影(features.mjs)で参照ループ R に載せて計算(Node の予想ではない)',
        'VM1': '時刻(フレーム番号)でそろえた舵 PWM の、群の平均からの |差| の平均 µs(今日の 7.6 µs と同じ式。開始が違う群では意味が薄い)',
        'VM2': '位置 s(5 cm のマス・各シーンの開始から 3 m 以降の最初の通過)でそろえた舵 PWM の |差| の平均 µs',
        'VM3': 'VM2 と同じ式の横ずれ e_y [m](線のばらつき)', 'VM4': 'VM2 と同じ式の速さ [m/s]',
        'VM5': 'すき間(車体の長方形と柵・柱の距離)。シーンの最小は lap_audit の正確な値(replay_check)、フレームの分布は features.mjs の clearance',
        'VM6': '乱しの出来事・学習から外す割合・立て直し(protocol definitions.recovered、乱しなしの線は登録簿の離脱復活の方策で作り直したもの)',
        'VM7': '開始から 3 m 以降のフレームを (s 0.25 m × e_y 5 cm × 向きのずれ 0.1 rad × 速さ 0.5 m/s) のマスに入れた訪れたマスの数(最大 9,000 フレーム、種 12345)。全モード合わせはモードごとに 2,250 フレーム',
        'teacher_reference': '前の教師のデータを同じコードで計算。VM1・VM2・VM3 は 60 シーン、VM7 は先頭 15 シーン(8,460 フレーム。モードの 15 シーンは 8,284〜8,375 フレーム)'},
    'teacher': {'all60': compact(T['all']), 'first15': compact(T['first15'])},
    'modes': {k: compact(g) for k, g in raw['modes'].items()},
    'pooled': {**compact(raw['pooled'], vm7=raw['pooled']['VM7_equal_frames']['cells']), 'scenes_per_mode': raw['pooled']['scenes_per_mode'],
               'VM7_equal_frames': raw['pooled']['VM7_equal_frames'], 'leave_one_out': raw['pooled_leave_one_out']},
    'recover_detail': raw['modes'].get('recover', {}).get('recover'),
    'judged': raw['judged'],
    'validation': val,
    'files': {'raw': str(OUT / 'variation_raw.json'), 'figdata': str(OUT / 'figdata.json'), 'replay': str(OUT / 'replay'), 'verify': str(OUT / 'verify'),
              'figures': [str(p) for p in sorted((R / 'fig').glob('record_*.png'))]},
}
(R / 'variation.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: {kk: round(v['ratio_vs_teacher'][kk], 2) for kk in v['ratio_vs_teacher']} for k, v in {**out['modes'], 'pooled': out['pooled']}.items()}, ensure_ascii=False))
print(json.dumps({k: {'check': v['user_style_check']['pass'], 'id': v['replay']['pass_identity_1e-6'], 'rules': v['replay']['pass_rules'], 'pose': v['replay']['open_loop_max_pose_err_m'],
                      'clr': v['replay']['min_clearance_m'], 'laps': v['replay']['valid_laps'], 'pred': v['replay']['prediction_first_diff_frame'], 'predpose': round(v['replay']['prediction_max_pose_diff_m'], 2),
                      'rgb': v['user_style_check']['min_unique_rgb'], 'tof': v['user_style_check']['min_unique_tof']} for k, v in val.items()}, ensure_ascii=False))
