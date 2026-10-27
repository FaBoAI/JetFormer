"""反証 r2-rec2-4: 人側にした直後の 9G が、car_check の文のどこかに出るか(作り物・tmp だけ)"""
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import car_check as cc  # noqa: E402
import section_scorecard as ss  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)
for tag, h in (('flip_before_0.05', 7.95), ('flip_before_0.3', 7.7), ('flip_after', 8.2)):
    raw = tmp / tag
    raw.mkdir()
    log = '20261103-140000'
    write_standing_log(raw, log, dur=12.0, hits=[(8.0, 9.0)], stop_after=8.05, human_after=h, run='runB',
                       description='ai_no=15 day=2026-11-03 block=15 memo=x', params={'throttle_gain': 0.9, 'launch_boost': 1})
    info, lab, src = ss._labels(log, raw / 'nolab', raw)
    print('=====', tag, 'label collisions:', [(c['t_s'], c['g'], c['ai'], c['loc']) for c in info['events']['collisions']],
          'suspects:', len(info['events'].get('collision_suspects', [])),
          'takeovers:', [(x['t_s'], x.get('failure'), x.get('class')) for x in info['events'].get('takeovers', [])])
    r = cc.check_log(log, raw / 'nolab', raw, labels=(info, lab, src))
    print(cc.to_text(r))
    print('8G を含む行:', [l for l in cc.to_text(r).split('\n') if '9.0' in l or '8G' in l])
shutil.rmtree(tmp, ignore_errors=True)
