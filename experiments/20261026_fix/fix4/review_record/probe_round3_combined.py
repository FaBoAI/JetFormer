"""反証 4 回目(記録と夜の集計): 直す 3 回目の 4 件(r3-r3-rec-1〜4)を 1 つの作り物の回 3 でまとめて確かめる。

本物の流れの説明欄(雛形 → plan_check → ai_no、走り終わりのメモは after_description で scene.json)= tests/test_fix3_r3_2_tools.RealDay。
  朝 09:00 の電源断の本(log.json・npz が 0 バイト)・3-1 OFF・3-2 ON 250(空転 IMU × 0.6 = 比 1.69)→ 残りの ON を 200 に・
  3-3 OFF・3-4 ON 200・3-5 OFF・3-6 ON 200 で memo に「試し」(最初の周に 5G)→ 走り直し 3-6 ON 200・3-7 OFF・3-8 ON 200(IMU × 0.05 =
  いちばんひどい空転)。
見ること: car_check の空転の行(3-2・3-8)、⑤ の注意・最後の強さ・試し・衝突・いちばんひどい空転・読めない記録、③ の試し。
本物の data/raw・labels は読まない。作り物は tmp_r3/ に作り、終わったら消す。
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
import test_fix3_r3_2_tools as T  # noqa: E402
import car_check as cc  # noqa: E402
import standing_3lap as s3  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = HERE / 'tmp_r3'
OUT = HERE / 'out'


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    OUT.mkdir(exist_ok=True)
    D = T.RealDay(TMP)
    G, GON = T.G, T.GON
    G200 = dict(GON, launch_us=200.0)
    names = {}
    names['3-1'], _ = D.add_real(12, G, T.HAND)
    names['3-2'], _ = D.add_real(13, GON, T.HAND, imu_scale=0.6, **T.on_kw(30))
    names['3-3'], _ = D.add_real(14, G, T.HAND)
    names['3-4'], _ = D.add_real(15, G200, T.HAND, **T.on_kw(30))
    names['3-5'], _ = D.add_real(16, G, T.HAND)
    names['3-6 試し'], _ = D.add_real(17, G200, ('その他', '試し(確認の文を読み忘れた)'), hits=[(3.0, 5.0)], **T.on_kw(36))
    names['3-6'], _ = D.add_real(17, G200, T.HAND, **T.on_kw(36))
    names['3-7'], _ = D.add_real(18, G, T.HAND)
    names['3-8'], _ = D.add_real(19, G200, T.HAND, imu_scale=0.05, **T.on_kw(39))
    out = {'names': names}
    # car_check の空転の行(3-2・3-4・3-8)
    for k in ('3-2', '3-4', '3-8'):
        r = cc.check_log(names[k], D.tmp / 'nolab', D.raw)
        out['car_check ' + k] = [x for x in cc.to_text(r).splitlines() if '空転' in x or x.startswith('  →')]
    # ⑤(電源断の本なし)
    j, o, _ = T.runc(D, 'n_a', args=['--ignore-gain'])
    g = j['summary'][0]
    out['collect_A'] = {'checks': g['compare']['checks'], 'all_ok': g['compare']['all_ok'], 'launch_us_used': g.get('launch_us_used'),
                        'trial_excluded': g.get('trial_excluded'), 'trial_on_collisions': g.get('trial_on_collisions'),
                        'slip_extreme': (g.get('on_used') or {}).get('slip_extreme'),
                        'lines': [x for x in o.splitlines() if x.startswith('  ')]}
    # ⑤(朝の電源断の本を足す: log.json・npz が 0 バイト)
    z = D.raw / '20261103-090000'
    (z / '20261103-090000-0001').mkdir(parents=True)
    (z / 'log.json').write_bytes(b'')
    for n in ('AI_CMD', 'VEHICLE', 'IMU', 'CAM_FRONT'):
        (z / '20261103-090000-0001' / (n + '.npz')).write_bytes(b'')
    j2, o2, _ = T.runc(D, 'n_b', args=['--ignore-gain'])
    g2 = j2['summary'][0]
    out['collect_B_zero_logjson'] = {'checks': g2['compare']['checks'], 'all_ok': g2['compare']['all_ok'],
                                      'unknown': g2.get('unreadable_unknown')}
    # ③ 試しの本
    res = s3.run([names['3-6 試し'], names['3-6']], D.raw, D.tmp / 'nolab', None)
    out['standing_trial'] = [(r['log'], r.get('trial'), [s.get('counted') for s in r['standing']]) for r in res]
    (OUT / 'probe_round3_combined.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
