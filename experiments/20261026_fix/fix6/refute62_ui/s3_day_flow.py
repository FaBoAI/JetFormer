"""S3: 11/3 の 1 日を手順書(3 章・4 章・5 章)と bench_candidates.json の after_press どおりに流す(本物の app.js + 本物の drive.Driver)。

朝: 基準の設定 → dry run → 外す → 発進の前進の確かめ(収集を外す)→ 見張り・残すを ON(スイッチ)。
回 1〜4: 基準の設定 → ベンチマーク N → ゲインを ±0.01 で表の値 → 押したあとに入れるもの(スイッチ・curl)→ ▶ → 止める。
走行開始ごとに、本物の Driver が走る値と、計画の値・確認の文の行を比べる。違いがあれば出す。
"""
import json
import common as C

c = C.bench_today()
di = C.day_index(c)
UP, DOWN = '#btn-dr-tg-up', '#btn-dr-tg-down'


def gain(target, cur=0.90):
    n = int(round((target - cur) / 0.01))
    return [['click', UP if n > 0 else DOWN]] * abs(n)


# (ブロック, ゲイン, 天井, 自動合わせ, 発進の前進, 押す前の curl, 押したあとの curl, 左, 強さ)
P = []
P += [(1, 0.92, 0, 0, 0, [], [], 4.1, 250.0), (2, 0.93, 0, 0, 0, [], [], 4.1, 250.0), (3, 0.95, 0, 0, 0, [], [], 4.1, 250.0),
      (4, 0.95, 1, 0, 0, [], [], 4.1, 250.0), (5, 0.95, 0, 0, 0, [], [], 4.1, 250.0)]
P += [(6, 0.97, 1, 0, 0, [], [], 4.1, 250.0), (7, 0.97, 0, 0, 0, [], [], 4.1, 250.0), (8, 1.00, 1, 1, 0, [], [], 4.1, 250.0),
      (9, 1.00, 1, 1, 0, [], [], 4.1, 250.0), (10, 1.00, 1, 0, 0, [], [{'ceil_left_mps': 4.2}], 4.2, 250.0),
      (11, 1.00, 1, 0, 0, [], [{'ceil_left_mps': 4.1}], 4.1, 250.0)]
# 回 3: 回 2 のよい形 = 天井(左 4.1)。3-2 の空転の比 > 1.3 → 3-3 の前に launch_us 200
P += [(12, 1.00, 1, 0, 0, [], [], 4.1, 250.0), (13, 1.00, 1, 0, 1, [], [], 4.1, 250.0),
      (14, 1.01, 1, 0, 0, [{'launch_us': 200}], [], 4.1, 200.0), (15, 1.01, 1, 0, 1, [], [], 4.1, 200.0),
      (16, 1.02, 1, 0, 0, [], [], 4.1, 200.0), (17, 1.02, 1, 0, 1, [], [], 4.1, 200.0),
      (18, 1.02, 1, 0, 0, [], [], 4.1, 200.0), (19, 1.02, 1, 0, 1, [], [], 4.1, 200.0)]
P += [(20, 1.00, 1, 0, 1, [], [], 4.1, 200.0), (21, 1.00, 1, 0, 1, [], [], 4.1, 200.0), (22, 1.02, 1, 0, 1, [], [], 4.1, 200.0)]

steps = [['poll'],
         # 朝の dry run(基準の設定 → dry → ▶ → 止める → 外す)
         ['baseline'], ['check', '#dr-dry', True], ['start'], ['running', False], ['check', '#dr-dry', False],
         # 発進の前進の確かめ(基準の設定 → 収集を外す → 発進の前進 ON → ▶ → 止める)
         ['baseline'], ['check', '#dr-record', False], ['check', '#dr-launch-boost', True], ['start'], ['running', False],
         # 試験の選択肢(見張りが先)
         ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['poll']]
plan = [None, None]
for (no, g, ceil, auto, launch, api_before, api_after, left, us) in P:
    s = [['api', b] for b in api_before] + [['baseline'], ['poll'], ['pick', di, C.blk(c, di, no)], ['poll']] + gain(g)
    s += [['api', b] for b in api_after]
    if ceil:
        s.append(['check', '#dr-speed-ceiling', True])
    if auto:
        s.append(['check', '#dr-speed-auto', True])
    if launch:
        s.append(['check', '#dr-launch-boost', True])
    if no == 1:
        s.append(['slide', '#dr-trim', 15])        # car_check の「直す」(1-1 のあとのつもり。ここでは 1-1 の前に入れる)
    s += [['poll'], ['start'], ['running', False]]
    steps += s
    plan.append({'block': no, 'throttle_gain': g, 'speed_ceiling': ceil, 'speed_auto': auto, 'launch_boost': launch,
                 'ceil_left_mps': left, 'launch_us': us, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0,
                 'zone_caps': 0, 'start_kick': 1, 'steer_trim_us': 15.0})
steps.append(['posts'])
res, toasts = C.sim(c, steps, server=C.server_params())
sts = C.starts(res)
plist = C.posts(res)[-1]
d = C.driver(params=C.server_params())
snaps, i = [], 0
for path, body, status in plist:
    if path == '/api/drive/params':
        try:
            d.set_params(body); got = 200
        except ValueError as e:
            got = 400
        if got != status:
            snaps.append(('MISMATCH', path, body, got, status))
    elif path == '/api/drive/start':
        d.start(body['run'], body['mode_id'], params=body['params'], description=body.get('description', ''), dry_run=body['dry_run'])
        d.thread.join(1)
        snaps.append(dict(d.params, _desc=body.get('description'), _dry=body['dry_run'], _record=body['record']))
starts_real = [s for s in snaps if isinstance(s, dict)]
problems = [s for s in snaps if not isinstance(s, dict)]
rows = []
for k, (want, real, st) in enumerate(zip(plan, starts_real, sts)):
    if want is None:
        rows.append({'n': k, 'what': 'dry' if real['_dry'] else 'launch-check', 'launch_boost': real['launch_boost'],
                     'record': real['_record'], 'desc': real['_desc']})
        continue
    diff = {kk: [vv, real[kk]] for kk, vv in want.items() if kk != 'block' and abs(float(real[kk]) - float(vv)) > 1e-9}
    lines = [l for l in st['confirm'].split('\n') if any(w in l for w in ('見張り', '残す', '補助も', '天井', '自動合わせ', '発進の前進:', '速度ゲイン', 'トリム'))]
    # 確認の文の行が本物の値と合うか
    cdiff = []
    if (+real['speed_ceiling'] == 1) != any('速さの天井: ON' in l for l in lines): cdiff.append('天井の行')
    if (+real['speed_auto'] == 1) != any('自動合わせ: ON' in l for l in lines): cdiff.append('自動合わせの行')
    if (+real['launch_boost'] == 1) != any(l.startswith('発進の前進: ON') for l in lines): cdiff.append('発進の前進の行')
    if (+real['rec_back_guard'] == 1) != any('後ろの見張り: ON' in l for l in lines): cdiff.append('見張りの行')
    if (+real['rec_hold_keep'] == 1) != any('残す: ON' in l for l in lines): cdiff.append('残すの行')
    if real['speed_ceiling'] and not any('左旋回中 %.1f' % real['ceil_left_mps'] in l for l in lines): cdiff.append('左の値')
    if real['launch_boost'] and not any('前進 %.0f µs' % real['launch_us'] in l for l in lines): cdiff.append('強さの値')
    if not any('速度ゲイン %.2f' % real['throttle_gain'] in l for l in lines): cdiff.append('ゲインの値')
    rows.append({'n': k, 'block': want['block'], 'diff_vs_plan': diff, 'confirm_vs_real': cdiff, 'dry': real['_dry'],
                 'record': real['_record'], 'desc_block': (real['_desc'] or '').split('block=')[1].split()[0] if 'block=' in (real['_desc'] or '') else None})
out = {'n_starts': len(sts), 'post_status_mismatch': problems, 'rows': rows,
       'toasts_errors': [t for t in toasts if any(w in t for w in ('できません', '失敗', '断', 'ません'))]}
print(json.dumps(out, ensure_ascii=False, indent=1))
(C.HERE / 'out' / 's3_day_flow.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
C.cleanup()
