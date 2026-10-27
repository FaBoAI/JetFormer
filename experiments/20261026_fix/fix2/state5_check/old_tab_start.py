"""指摘 state-5 の確かめ: 古い画面(launch_boost・speed_ceiling を送らない)の走行開始で、サーバーに残った ON が使われるか。
装置を開かない: Driver の設定ファイルは tmp に、model_info は偽物、走行ループ _run は何もしない。"""
import json, re, sys, tempfile
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
from jetformer import drive as D          # noqa: E402
from jetformer.web import bench as B      # noqa: E402

tmp = Path(sys.argv[1])
D.SETTINGS_PATH = tmp / 'drive_settings.json'          # 本物の drive_settings.json は触らない
D.model_info = lambda name: {'trained_modes': [0], 'drivable': True, 'postprocess': None}

src = (ROOT / 'jetformer/web/server.py').read_text(encoding='utf-8')
ver = int(re.search(r'^DRIVE_CLIENT_VERSION = (\d+)', src, re.M).group(1))
old_body = {'client': 6, 'params': {'steer_gain': 1.0, 'throttle_gain': 0.5, 'steer_trim_us': 0, 'lookahead': 1,
                                    'osc_percent': 0, 'rev_us': 180, 'rev_s': 0.6, 'speed_cap': 0, 'speed_scale': 1,
                                    'ensemble_j': 0, 'ensemble_m': 0.5, 'esc_reverse': 1, 'recovery': 1,
                                    'tof_guard': 1, 'tcp_mode': 0, 'zone_caps': 0}}   # c247daa の readDriveParams の項目
gate_ok = not (int(old_body.get('client') or 0) < ver)
out = {'DRIVE_CLIENT_VERSION': ver, 'old_client': old_body['client'], 'old_tab_passes_gate': gate_ok}


def new_driver():
    d = D.Driver(hub=None, recorder=None)
    d._run = lambda *a, **k: None                     # 走行ループを動かさない
    return d


# 1) 新しい画面で発進の前進を ON(POST /api/drive/params {launch_boost:1} と同じ)→ 古い画面で走行開始
d = new_driver()
out['boot_launch_boost'] = d.params['launch_boost']
d.set_params({'launch_boost': 1})
d.start('X', 0, params=old_body['params'])
d.thread.join(1)
out['case1_after_old_start_launch_boost'] = d.params['launch_boost']

# 2) 新しい画面で launch_boost=1 の本を走り終えた → 古い画面で走行開始
d = new_driver()
d.start('X', 0, params=dict(old_body['params'], launch_boost=1)); d.thread.join(1)
d.start('X', 0, params=old_body['params']); d.thread.join(1)
out['case2_after_old_start_launch_boost'] = d.params['launch_boost']

# 3) 速さの天井も同じ(10/24 版の画面は speed_ceiling も送らない)
d = new_driver()
d.set_params({'speed_ceiling': 1})
d.start('X', 0, params=old_body['params']); d.thread.join(1)
out['case3_after_old_start_speed_ceiling'] = d.params['speed_ceiling']

# 4) 対照: 新しい画面は launch_boost=0 を送るので OFF になる
d = new_driver()
d.set_params({'launch_boost': 1})
d.start('X', 0, params=dict(old_body['params'], launch_boost=0, speed_ceiling=0, speed_auto=0)); d.thread.join(1)
out['case4_new_tab_start_launch_boost'] = d.params['launch_boost']

# 5) ベンチマークの照らし合わせ(plan_check)は、送られない項目を比べない → cand はそのまま(試しにならない)
data = {'ok': True, 'baseline': {'mode_id': 0, 'params': {'launch_boost': 0, 'speed_ceiling': 0, 'throttle_gain': 0.5}},
        'candidates': [{'id': 'A', 'run': 'X'}], 'days': [{'date': '1103', 'blocks': [{'block': '1', 'id': 'A', 'run': 'X'}]}]}
desc = 'bench=1 cand=A block=1 day=1103'
desc2, diffs = B.plan_check(desc, 'X', 0, old_body['params'], data=data)
out['case5_plan_check_diffs_old_tab'] = diffs
out['case5_desc_after'] = desc2
print(json.dumps(out, ensure_ascii=False, indent=1))
