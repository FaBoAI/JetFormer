import json,sys
S=json.load(open(sys.argv[1]))
keys=['n','collisions','min_clearance_cm','median_min_clearance_cm','below_5cm','below_8cm','valid_episode','valid_and_lap','shield_fraction','governor_fraction','runs_with_shield','shield_lt5s_frames','known_frame','broken_frames','rho_req_gt1','rho_comb_gt1','rho_cmd_gt1','max_scene_rho_comb_gt1','audit_pos_err_max','overlap_frames_without_collision','own_crossings_outside','laps_not_33_or_cross','lap_distance_range','labels_l0','bad_keys','mean_running_speed','lap1_median_s','fallbacks','errors']
for k,v in S['sets'].items():
    if len(sys.argv)>2 and k not in sys.argv[2].split(','): continue
    if 'split' in k:
        for kk,vv in v.items(): print(k,kk,{x:vv.get(x) for x in ['n','collisions','min_clearance_cm','below_5cm','below_8cm','valid_and_lap','shield_fraction']})
        continue
    if '_vs_' in k: print(k, v); continue
    print('==',k, json.dumps({x:v.get(x) for x in keys},ensure_ascii=False))
    print('  collisions:', json.dumps(v.get('collision_runs'),ensure_ascii=False)[:3000])
    print('  worst:', json.dumps(v.get('worst')[:6],ensure_ascii=False)[:2500])
    print('  not_valid:', json.dumps(v.get('not_valid'),ensure_ascii=False)[:1500])
