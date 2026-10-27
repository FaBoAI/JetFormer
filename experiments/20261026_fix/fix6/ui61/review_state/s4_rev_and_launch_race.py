"""S4a: approvals.md 10 の (b)「リカバリーの下がる量を小さくする(画面のスライダー)」を選んだ日に、朝にスライダーで rev_s 0.30・
rev_us 250 を入れてから、手順書 4 章どおり 基準の設定 → ブロック → ▶。送る値と本物の Driver の値。
S4b: S2 と同じ入れ替わりを発進の前進のスイッチで(3-3 OFF のあとの 3-4 ON。クリックのとき表示の問い合わせが 1 本飛んでいた)。
"""
import json
import common as C

c = C.bench_today()
di = C.day_index(c)
out = {}
# S4a
steps = [['slide', '#dr-rev-s', 0.3], ['slide', '#dr-rev-us', 250], ['poll'], ['read', 'server'],
         ['baseline'], ['pick', di, C.blk(c, di, 1)], ['read', 'note'], ['start'], ['posts']]
res, _ = C.sim(c, steps, server=C.server_params(rev_s=0.4, rev_us=300.0))
d = C.driver(params=C.server_params(rev_s=0.4, rev_us=300.0))
C.replay(d, C.posts(res)[0])
st = C.starts(res)[0]
out['S4a_rev_slider'] = {'server_after_morning_slider': {k: C.reads(res, 'server')[0][k] for k in ('rev_s', 'rev_us')},
                         'block_note': C.reads(res, 'note')[0], 'sent': {k: st['sent'][k] for k in ('rev_s', 'rev_us')},
                         'confirm_has_rev_line': any(('下がる' in l or '後退' in l) for l in st['confirm'].split('\n')),
                         'real_driver': {k: d.params[k] for k in ('rev_s', 'rev_us')}}
# S4b
pre = C.server_params(throttle_gain=1.0, speed_ceiling=1, launch_boost=0)
b15 = C.blk(c, di, 15)
for name, s in (('get_in_flight_at_click', [['gstart'], ['defer', True], ['checkn', '#dr-launch-boost', True], ['gend'],
                                            ['read', 'launch'], ['resolve'], ['read', 'launch'], ['defer', False], ['start'], ['posts']]),
                ('normal', [['check', '#dr-launch-boost', True], ['read', 'launch'], ['read', 'launch'], ['start'], ['posts']]),
                ('after_next_poll', [['gstart'], ['defer', True], ['checkn', '#dr-launch-boost', True], ['gend'], ['resolve'],
                                     ['defer', False], ['poll'], ['read', 'launch'], ['start'], ['posts']])):
    res, _ = C.sim(c, [['pick', di, b15], ['check', '#dr-speed-ceiling', True]] + s, server=pre)
    d = C.driver(params=pre)
    C.replay(d, [p for ps in C.posts(res) for p in ps])
    st = C.starts(res)[0]
    out['S4b_' + name] = {'launch_checkbox': C.reads(res, 'launch'), 'sent_launch_boost': st['sent']['launch_boost'],
                          'confirm_launch_line': [l for l in st['confirm'].split('\n') if l.startswith('発進の前進')],
                          'real_driver_launch_boost': d.params['launch_boost']}
print(json.dumps(out, ensure_ascii=False, indent=1))
(C.HERE / 'out' / 's4_rev_and_launch_race.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
C.cleanup()
