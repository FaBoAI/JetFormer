"""cap4_replay.json から cap4.md の表を出す(標準ライブラリだけ)。  python -I cap4_tables.py <cap4_replay.json>"""
import json, sys
d = json.load(open(sys.argv[1]))
P = d['per_log']
ORDER = ['20261010-172910', '20261010-173445', '20261010-203924', '20261010-204208', '20261010-204605',
         '20261010-171639', '20261010-172505', '20261010-181504', '20261010-155820', '20261010-160306', '20261010-163247']
ROLE = {'g1': 'G1', 'other_ai': 'AI', 'early_ai': '最初の 3 本'}
print('### 表 A: 1 走行あたりの「位相が不確か」の時間(AI 側の時間、秒。最後の区間が unknown = 4 つめの上限が効く時間)\n')
print('| 走行 | 組 | AI の時間 | unknown(合図と合わせたあと) | 割合 | 走り始め〜最初の R1 | R1 から 16 秒超 | 1 秒以上の停止のあと | 引き返し | 1 秒以上のかたまり |')
print('|---|---|---|---|---|---|---|---|---|---|')
for lg in ORDER:
    e = P[lg]; u = e['unknown_time']; c = u['final_unknown_by_cause_s']
    print('| %s | %s | %.1f | %.1f | %.0f%% | %.1f | %.1f | %.1f | %.1f | %d |' % (
        e['name'], ROLE[e['role']], u['ai_s'], u['final_unknown_s'], 100 * u['final_unknown_s'] / u['ai_s'],
        c['start'], c['lost16'], c['stop'], c['fallback'], u['episodes_n_ge_1s']))
for grp, label in (('G1_logs_5', 'G1 の 5 本'), ('AI_logs_8', 'AI の 8 本'), ('AI_logs_11', '11 本')):
    u = d['summary'][grp]['unknown_time']; c = u['final_unknown_by_cause_s']; n = u['runs']
    print('| **%s の 1 本あたり** | | %.1f | %.1f | %.0f%% | %.1f | %.1f | %.1f | %.1f | |' % (
        label, u['ai_s'] / n, u['final_unknown_s'] / n, 100 * u['final_unknown_s'] / u['ai_s'],
        c['start'] / n, c['lost16'] / n, c['stop'] / n, c['fallback'] / n))
print()
print('### 表 B: 上限が掛かったフレーム(影の再生、AI 側のフレーム)\n')
print('| 走行 | AI のフレーム | (a) v1.1 掛かった | 割合 | うち unknown | (b) 4 つめなし 掛かった | 割合 | unknown で掛かった時間 (a) 秒 | 前進の差 (a) µs | 前進の差 (b) µs |')
print('|---|---|---|---|---|---|---|---|---|---|')
for lg in ORDER:
    e = P[lg]; a = e['frames']['3zone']; b = e['frames']['noU']
    print('| %s | %d | %d | %.0f%% | %d | %d | %.0f%% | %.1f | %.1f | %.1f |' % (
        e['name'], a['ai_frames'], a['capped_frames'], 100 * a['capped_share'], a['by_zone']['unknown']['capped_frames'],
        b['capped_frames'], 100 * b['capped_share'], a['by_zone']['unknown']['capped_s'], a['dthr_us_mean_all_ai'], b['dthr_us_mean_all_ai']))
print()
print('### 表 C: まとめ(区間の内訳・ブレーキ・時間の損)\n')
Z = ['none', 'unknown', 'R1_app', 'R1_turn', 'GM', 'DE']
for grp, label in (('G1_logs_5', 'G1 の 5 本'), ('AI_logs_8', 'AI の 8 本')):
    S = d['summary'][grp]
    print('**%s**(AI %.1f 秒・%d フレーム)\n' % (label, S['3zone']['ai_s'], S['3zone']['ai_frames']))
    print('| | (a) v1.1 | (b) 4 つめなし | 差 (a)−(b) |')
    print('|---|---|---|---|')
    a, b = S['3zone'], S['noU']
    print('| 上限が掛かったフレーム | %d(%.1f%%) | %d(%.1f%%) | %d |' % (a['capped_frames'], 100 * a['capped_share'], b['capped_frames'], 100 * b['capped_share'], a['capped_frames'] - b['capped_frames']))
    print('| 内訳 ' + '・'.join(Z) + ' | ' + '・'.join(str(a['capped_frames_by_zone'][z]) for z in Z) + ' | ' + '・'.join(str(b['capped_frames_by_zone'][z]) for z in Z) + ' | |')
    print('| 掛かったフレームのうち unknown | %.1f%% | %.1f%% | |' % (100 * a['capped_share_of_capped_by_zone']['unknown'], 100 * b['capped_share_of_capped_by_zone']['unknown']))
    print('| unknown で掛かった時間 | %.1f 秒 | %.1f 秒 | %.1f 秒 |' % (a['unknown_capped_s'], b['unknown_capped_s'], a['unknown_capped_s'] - b['unknown_capped_s']))
    uc = a['unknown_capped_by_cause_s']
    print('| 同・理由ごと(走り始め・16 秒超・停止のあと・引き返し) | %.1f・%.1f・%.1f・%.1f | | |' % (uc['start'], uc['lost16'], uc['stop'], uc['fallback']))
    print('| 前進の差(上限なしの出力 − 上限つき、AI の時間の平均) | %.1f µs | %.1f µs | %.1f µs |' % (a['dthr_us_mean_all_ai'], b['dthr_us_mean_all_ai'], a['dthr_us_mean_all_ai'] - b['dthr_us_mean_all_ai']))
    bzA, bzB = a['by_zone']['unknown'], b['by_zone']['unknown']
    print('| 同・unknown の時間だけ | %.1f µs | %.1f µs | |' % (bzA['dthr_us_sum_x_s'] / bzA['ai_s'], bzB['dthr_us_sum_x_s'] / bzB['ai_s']))
    print('| ブレーキの始まり(影の再生) | %d | %d | %d |' % (a['shadow_brake_starts'], b['shadow_brake_starts'], a['shadow_brake_starts'] - b['shadow_brake_starts']))
    print('| ブレーキの始まり(仮想の閉ループ・AI の区間全部) | %d | %d | |' % (a['virtual_brake_starts'], b['virtual_brake_starts']))
    print('| ブレーキの規則の違反(1 m/s 未満・旋回中・往復) | %s | %s | |' % ('・'.join(str(v) for v in a['virtual_rules'].values()), '・'.join(str(v) for v in b['virtual_rules'].values())))
    print('| 時間の損(仮想の閉ループ・AI の区間全部 %.0f 秒) | %.2f 秒 | %.2f 秒 | **%.2f 秒**(240 秒あたり %.2f 秒) |' % (
        a['virtual_stretch_s'], a['virtual_loss_s'], b['virtual_loss_s'], a['virtual_loss_s'] - b['virtual_loss_s'],
        240 * (a['virtual_loss_s'] - b['virtual_loss_s']) / a['virtual_stretch_s']))
    print('| 同・unknown の中の損 | %.2f 秒 | %.2f 秒 | |' % (a['virtual_loss_by_zone_s']['unknown'], b['virtual_loss_by_zone_s']['unknown']))
    print()
print('### 表 D: 位相が unknown のまま入った R1(AI の通過、11 本)\n')
print('| 走行 | 起点 秒 | 記録の進入 m/s | 衝突 | 合図(予測ステア) | 仮想の進入 (a) | 仮想の進入 (b) |')
print('|---|---|---|---|---|---|---|')
for lg in ORDER:
    for r in P[lg]['r1_passes_ai']:
        if r['approach_gyro_zone'] != 'unknown':
            continue
        col = '強い衝突' if r['r1_strong'] else ('衝突' if r['r1_collision'] else 'なし')
        f = lambda x: '%.2f' % x if x is not None else '—'
        print('| %s | %.1f | %.2f | %s | %s | %s | %s |' % (P[lg]['name'], r['ta'], r['entry_mps'], col, 'あり' if r['sig_zone'] else '**なし**',
                                                       f(r.get('virtual_entry_3zone')), f(r.get('virtual_entry_noU'))))
print()
print('### 表 E: 止める決まり(10/10 の 11 本。止めたら数え直したときの回数)\n')
K = [('A2_3laps', 'A2(3 周)'), ('A2_2laps(了承なしの読み方)', '2 周'), ('orig_a_consecutive_strong_same_loc', '元の文 (a) 4 G が 2 件続く'),
     ('orig_b_strong_same_loc_2laps', '元の文 (b) 4 G が 2 周続く')]
print('| 走行 | AI 秒 | 止まって取り返した衝突 | ' + ' | '.join(k[1] for k in K) + ' |')
print('|---|---|---|' + '---|' * len(K))
for lg in ORDER:
    s = P[lg]['stop_rules']
    cells = []
    for k, _ in K:
        r = s['rules'][k]
        cells.append('%d' % r['stops'] + ('(AI %.0f 秒で %s)' % (r['first']['ai_s_at_stop'], r['first']['loc']) if r['first'] else ''))
    print('| %s | %.0f | %d | %s |' % (P[lg]['name'], s['ai_s'], s['takeovers_after_collision'], ' | '.join(cells)))
print()
for lg in ('20261010-172910', '20261010-173445'):
    s = P[lg]['stop_rules']
    print('- %s: 止まって取り返した衝突の場所と周 = %s' % (P[lg]['name'], ', '.join('%s %s 周目(%.0f 秒)' % (l, k, t) for l, t, k in s['takeover_times'])))
    for k, _ in K:
        r = s['rules'][k]
        print('  - %s: %s' % (k, '; '.join('%s 秒(AI %s 秒)%s' % (x['t_s'], x['ai_s_at_stop'], x['loc']) for x in r['all']) or 'なし'))
