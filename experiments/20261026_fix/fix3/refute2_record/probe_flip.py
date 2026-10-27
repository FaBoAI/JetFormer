"""反証 2 回目・係 record: 人が当たる直前に人側にした本の衝突が、car_check・standing_3lap に出るか(作り物・tmp だけ)"""
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import car_check as cc  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)
for tag, h in (('flip_before', 7.95), ('flip_after', 8.2)):
    raw = tmp / tag
    raw.mkdir()
    log = '20261103-140000'
    write_standing_log(raw, log, dur=12.0, hits=[(8.0, 9.0)], stop_after=8.05, human_after=h, run='runB',
                       description='ai_no=15 day=2026-11-03 block=15 memo=x', params={'throttle_gain': 0.9, 'launch_boost': 1})
    r = cc.check_log(log, raw / 'nolab', raw)
    a = s3.analyze_log(log, raw, raw / 'nolab', {'runB': 'B'})
    print(tag, 'car_check 衝突:', cc.ss.collision_text(r['collisions']), '| standing_3lap 記録の衝突:', a['collisions']['text'],
          '| 発進の行の衝突:', [(s.get('collisions') or {}).get('n') for s in a['standing']])
shutil.rmtree(tmp, ignore_errors=True)
