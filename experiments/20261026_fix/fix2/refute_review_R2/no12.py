import sys, json
sys.path.insert(0, '/home/jetson/jetformer'); sys.path.insert(0, '/home/jetson/jetformer/tools')
import standing_3lap as s3
import numpy as np
for lg in sys.argv[1:]:
    r = s3.analyze_log(lg, names=s3.candidate_names())
    for s in r.get('standing') or []:
        if s.get('method'):
            print(lg, 'No.', r.get('ai_no'), 'Lrun', r.get('lap_dist_median_m'), 'r1_anchors', r.get('r1_anchors'))
            print(json.dumps({k: s.get(k) for k in ('t_ai','t_ai_end','t_move','anchors','d_pre_m','line_ends','end_s','laps_line_s','laps_anchor_s','dist_anchor_m','method','valid','why','from_move_s','dist3_from_move_s','note')}, ensure_ascii=False))
