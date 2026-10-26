"""DE の揺れの表(tools/de_swing.py、2026-10-24)。合成のラベルで、通過の切り出し・行き過ぎ・切り返し・衝突の数え方と、
説明欄(--assign)・設定ごとのまとめを確かめる

    cd ~/jetformer && python -m pytest -q tests/test_de_swing.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import de_swing  # noqa: E402

C = 1441.0


def lab(n=200, de=(60, 111), ai_all=True, hit_at=None, early=-150.0, late=+90.0):
    t = np.arange(n) / 30.0
    loc = np.zeros(n, np.int8)
    loc[de[0]:de[1]] = de_swing.DE_CODE
    out = np.full((n, 2), C, np.float32)
    a = de[0]
    out[a + 5, 0] = C + early                 # 入って約 0.17 秒の右
    out[a + 24, 0] = C + late                 # 入って 0.8 秒の左
    coll = np.zeros(n, bool)
    if hit_at is not None:
        coll[a + hit_at] = True
    ai = np.ones(n, bool) if ai_all else np.r_[np.ones(n // 2, bool), np.zeros(n - n // 2, bool)]
    return {'loc_code': loc, 't_s': t, 'ai': ai, 'collision': coll, 'out': out}


def test_one_pass_values():
    ps = de_swing.passes(lab(), C)
    assert len(ps) == 1
    p = ps[0]
    assert p['overshoot_us'] == -150.0 and p['late_left_us'] == 90.0 and not p['hit'] and p['frames'] == 51


def test_hit_cuts_late_window_and_carry():
    p = de_swing.passes(lab(hit_at=20), C)[0]           # 0.67 秒で当たる → 0.8 秒の左は数えない
    assert p['hit'] and p['late_left_us'] == 0.0
    p = de_swing.passes(lab(de=(60, 100), hit_at=45), C)[0]   # 区間の終わり(100)から 5 フレーム後 → 0.3 秒以内なので衝突
    assert p['hit']


def test_short_or_human_pass_is_skipped():
    assert de_swing.passes(lab(de=(60, 80)), C) == []      # 1 秒未満
    assert de_swing.passes(lab(ai_all=False, de=(80, 140)), C) == []   # 途中で人に替わる


def test_build_groups_and_assign(tmp_path):
    for i, (lg, params, l) in enumerate([('20261024-200000', {'steer_gain': 1.0, 'lookahead': 3}, lab()),
                                         ('20261024-200100', {'steer_gain': 1.0, 'lookahead': 4}, lab(hit_at=26)),
                                         ('20261024-200200', {'steer_gain': 1.0, 'lookahead': 3}, lab(early=-50.0))]):
        (tmp_path / (lg + '.json')).write_text(json.dumps({
            'log': lg, 'ai_log': True, 'run': 'B', 'description': 'ai_no=%d bench=x cand=B block=1' % (i + 1),
            'ref_us': {'steer_center': C}, 'params_at_start': params}), encoding='utf-8')
        np.savez(tmp_path / (lg + '.npz'), **l)
    (tmp_path / '20261024-200300.json').write_text(json.dumps({'log': '20261024-200300', 'ai_log': False}))
    np.savez(tmp_path / '20261024-200300.npz', **lab())
    logs = ['20261024-200000', '20261024-200100', '20261024-200200', '20261024-200300']
    res = de_swing.build(logs, tmp_path, raw_dir=tmp_path, assign={'20261024-200100': 'ai_no=2 bench=x cand=EXP block=1'})
    assert [r['cand'] for r in res['rows']] == ['B', 'EXP', 'B']             # 人の記録は数えない・assign が勝つ
    g = {x['setting'].split('lookahead=')[1].split()[0]: x for x in res['groups']}
    assert g['3']['passes'] == 2 and g['3']['hits'] == 0 and g['3']['overshoot_med_us'] == -100.0
    assert g['3']['overshoot_lt_120'] == 1
    assert g['4']['passes'] == 1 and g['4']['hits'] == 1 and g['4']['hit_rate'] == 1.0
    md = de_swing.to_md(res, '20261024')
    assert '| 20261024-200100 | 2 | EXP | 1 | 1 |' in md
