"""Cap4 の確かめ(標準ライブラリだけ): (1) (a) v1.1 の再生が 10/16 の G1(g1_replay.json v1.1 R1_2.5)と同じ数を出す
(2) (b) は cap_unknown だけが inf で、ほかの値は (a) と同じ (3) unknown の外では (a)(b) のフレームの数が同じ
(4) G1 の 6 項目の合否が (a)(b) で同じ (5) 止める決まりの 2 周(AI の通過だけ)が Fix-all の stop_rule_a2.json と同じ本数
  python -I check_cap4.py <cap4_replay.json>"""
import json, sys
d = json.load(open(sys.argv[1]))
g = json.load(open('/home/jetson/jetformer_reports/sim_calibration_20261016/g1_replay.json'))['v1.1']
fx = json.load(open('/home/jetson/jetformer/experiments/20261021_bench_prep/fix_all/stop_rule_a2.json'))
ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(('OK  ' if cond else 'NG  ') + name)


G = d['g1_gate']
gg = g['summary']['gate']
check('1 R1 の合図 10/11 が同じ', G['R1_signal_ge_90pct']['value'] == gg['R1_signal_ge_90pct']['value'])
check('2 強い衝突 8/9 が同じ', G['R1_signal_strong_ge_90pct']['value'] == gg['R1_signal_007_board_ge_90pct']['value'])
check('3 誤った合図(ログごと)が同じ', [round(x, 2) for x in G['false_signal_le_1_per_min']['value'].values()]
      == [round(x, 2) for x in gg['false_signal_le_1_per_min_all5']['value'].values()])
check('5 ラップの損 0.152 が同じ', G['lap_loss_le_0.3_3zone']['value'] == gg['lap_loss_le_0.3_R1_2.5']['value'])
for lg in d['g1_logs']:
    a = d['per_log'][lg]['frames']['3zone']
    ms = g['logs'][lg]['shadow']['R1_2.5']['mode_share']
    check('%s のモードの割合が g1_replay と同じ' % lg, all(abs(a['mode_frames'][m] / a['ai_frames'] - ms[m]) < 1e-9 for m in ms))
pa, pb = d['variant_params']['3zone'], d['variant_params']['noU']
check('(b) は cap_unknown だけが違う', [k for k in pa if pa[k] != pb[k]] == ['cap_unknown'] and pb['cap_unknown'] is None)
for lg, e in d['per_log'].items():
    a, b = e['frames']['3zone']['by_zone'], e['frames']['noU']['by_zone']
    check('%s: unknown の外の上限のフレームが (a)(b) で同じ' % lg,
          all(a[z]['capped_frames'] == b[z]['capped_frames'] for z in a if z not in ('unknown', 'none')))
for k in ('G1_pass', 'G1_protocol_pass'):
    check('%s が (a)(b) で同じ' % k, G[k + '_3zone'] == G[k + '_noU'])
check('ブレーキの規則 (a)(b) とも違反なし', G['brake_invariants_3zone']['pass'] and G['brake_invariants_noU']['pass'])
n2 = sum(1 for e in d['per_log'].values() if e['stop_rules']['rules']['A2_2laps_ai_passes_only(stop_rule_a2 と同じ数え方)']['stops'] > 0)
n3 = sum(1 for e in d['per_log'].values() if e['stop_rules']['rules']['A2_3laps_ai_passes_only(stop_rule_a2 と同じ数え方)']['stops'] > 0)
check('2 周(AI の通過だけ)が働く本数 %d = stop_rule_a2 の %d' % (n2, fx['fired_logs']['e_takeover_2laps']), n2 == fx['fired_logs']['e_takeover_2laps'])
check('3 周(AI の通過だけ)が働く本数 %d = stop_rule_a2 の %d' % (n3, fx['fired_logs']['d_takeover_3laps']), n3 == fx['fired_logs']['d_takeover_3laps'])
print('%d / %d OK' % (sum(ok), len(ok)))
sys.exit(0 if all(ok) else 1)
