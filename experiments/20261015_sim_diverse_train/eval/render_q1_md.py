#!/usr/bin/env python3
"""eval_sim.json → eval_sim.md(sim_diverse_train_20261015 の Q1 の評価の段、2026-10-15)。数字はすべて eval_sim.json と
review/recount_check.json から作る(手で書かない)。DAgger の節は eval/dagger_status.json があれば、それを写す。
"""
import json
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
ev = json.loads((R / 'eval_sim.json').read_text())
rc_path = R / 'review' / 'recount_check.json'
rc = json.loads(rc_path.read_text()) if rc_path.exists() else None
dg_path = R / 'eval' / 'dagger_status.json'
dg = json.loads(dg_path.read_text()) if dg_path.exists() else None
S, V = ev['summary'], ev['verdicts']
STARTS = list(ev['starts'])
GATE0 = ['E0', 'E1', 'E2', 'E3', 'E4', 'E6', 'E7', 'E8', 'E9']
REASON = {'collision': '衝突', 'time_limit': '時間切れ', 'route_complete': '完走', 'aborted': '中断'}
KIND = {'post-foot': 'ポストの足', 'wall': '壁', 'segment': '壁', 'disk': '柱', 'post': 'ポスト'}
NAMES = {'A0': 'A0(10/14・きれいな RL 20 分)', 'A1': 'A1(10/14・A0 + DART 30 分)', 'A2': 'A2(10/14・余裕の線 10 分)',
         'A2@1015': 'A2(今日走らせ直し)', 'S-DIV': '**S-DIV**(今日・5 モード + DART + 余裕の線 45 分)', 'S-MOD': 'S-MOD(今日・5 モードだけ 25 分)',
         'A4': 'A4(今日・S-DIV + DAgger)', 'A0s1': 'A0 種 1(10/13)', 'S-T': 'S-T 種 0(10/13・作者の先生のデータ)', 'S-T1': 'S-T 種 1(10/13)',
         'EXT': '作者の ONNX(10/13)', 'T_fast': 'T_fast(RL 最速の先生・Node・遅延 0)', 'T_margin': 'T_margin(余裕の線の先生・Node・遅延 0)'}
lines = []
w = lines.append


def f(x, n=2, unit=''):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'はい' if x else 'いいえ'
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f'{x}{unit}'
    return f'{x:.{n}f}{unit}'


main_arms = [a for a in ('A0', 'A1', 'A2', 'A2@1015', 'S-DIV', 'S-MOD', 'A4') if a in S]
today_arms = [a for a in ('S-DIV', 'S-MOD', 'A4', 'A2@1015') if a in S]
w('# シミュレーターの閉ループ評価(Q1)— いろいろな走りのデータで学習した JetFormer は走れるようになったか')
w('')
w(f"作成 {ev['created_at']}。数字はすべて `eval_sim.json`(この表のもと)から作った。protocol.json の sha256 `{ev['protocol_sha256'][:16]}…`"
  f"(記録 `{ev['protocol_sha256_recorded'][:16]}…`、{'同じ' if ev['protocol_sha256'] == ev['protocol_sha256_recorded'] else '**違う**'})。")
w('実車は走らせていない(実車・PCA9685・I2C に触れていない。実車での試験はこの作業の範囲の外)。')
w('')
w('## 0. 結論')
w('')
q = V.get('Q1_S-DIV_vs_yesterday')
if q:
    w(f"- **Q1(S-DIV 対 10/14 の A0・A1・A2): 「{q['verdict']}」**。S-DIV の完走 c = {q['c_S-DIV']} / {S['S-DIV']['n_ran']}"
      f"(10/14 の一番多い腕 c_ref = {q['c_ref']})、最初の直線を抜けた fs = {q['fs_S-DIV']} / {q['gate0_ran']}、"
      f"点 4 のゲートを数えた g4 = {q['g4_S-DIV']} / {q['gate0_ran']}、進んだ道のりの中央値 d = {f(q['d_S-DIV_m'], 1)} m"
      f"(A0 {f(S['A0']['d_median_m'], 1)}・A1 {f(S['A1']['d_median_m'], 1)}・A2 {f(S['A2']['d_median_m'], 1)} m)。")
vv = V.get('Q1_variety_vs_volume') or {}
if vv.get('c_S-MOD') is not None:
    w(f"- **ばらつき対量(S-MOD 対 A1): 「{vv['verdict']}」**({vv['why']})。S-MOD c = {vv['c_S-MOD']}・g4 = {vv['g4_S-MOD']}・d = {f(vv['d_S-MOD_m'], 1)} m、"
      f"A1 c = {vv['c_A1']}・g4 = {vv['g4_A1']}・d = {f(vv['d_A1_m'], 1)} m(1.5 × d(A1) = {f(vv['threshold_1p5_d_A1_m'], 1)} m)。")
else:
    w(f"- ばらつき対量(S-MOD 対 A1): {vv.get('verdict', '—')}")
ad = V.get('Q1_DART_margin_addon(S-DIV vs S-MOD)')
if ad:
    w(f"- **DART・余裕の線の上乗せ(S-DIV 対 S-MOD): 「{ad['verdict']}」**(完走の差 {ad['c_diff']:+d}・g4 の差 {ad['g4_diff']:+d}・fs の差 {ad['fs_diff']:+d})。")
if 'Q1b_DAgger' in V:
    x = V['Q1b_DAgger']
    w(f"- **Q1b(DAgger、A4 対 {x['base_arm']}): 「{x['verdict']}」**(完走の差 {x['c_diff']:+d}・g4 の差 {x['g4_diff']:+d}・fs の差 {x['fs_diff']:+d})。")
elif dg:
    w(f"- Q1b(DAgger): {dg.get('verdict_text', '—')}")
# 止まった場所のまとまり(衝突の位置を 0.3 m の升でまとめた、多い順。参考)
def clusters(arm):
    pts = {}
    for e in STARTS:
        r = ev['runs'].get(f'{arm}/{e}')
        c = (r or {}).get('collision')
        if r and r['collisions'] and c and c.get('x') is not None:
            k = (round(c['x'] / .3) * .3, round(c['z'] / .3) * .3)
            pts.setdefault(k, []).append(e)
    return sorted(pts.items(), key=lambda kv: -len(kv[1]))
for a in ('S-DIV', 'S-MOD', 'A4', 'A2', 'A1'):
    if a in S:
        cl = clusters(a)
        if cl:
            w(f"- 衝突した場所({a}、0.3 m の升でまとめた多い順): " + '、'.join(f"({x:+.1f}, {z:+.1f}) に {len(es)} 本({'・'.join(es)})" for (x, z), es in cl[:3]) +
              f"。衝突しなかった走り: 完走 {S[a]['n_completed']} 本・時間切れ {sum(1 for e in STARTS if (ev['runs'].get(f'{a}/{e}') or {}).get('reason') == 'time_limit')} 本。")
b = V['best_arm']
w(f"- 一番良い腕(完走 → g4 → d の順): **{b['arm'] or '決まらない(同点)'}**(並び: " +
  '、'.join(f"{x[0]} c{x[1]}・g4 {x[2]}・d {x[3]:.1f} m" for x in b['ranking']) + ')。')
same = ev.get('same_as_yesterday') or {}
if same:
    w(f"- 前日との同一性: {'**同じ**(陽性対照の作者の ONNX の E0 が 10/14 と同じ周回時間で完走・A1 の E0 が 10/14 と同じ軌跡)。A0・A1・A2 の 10/14 の数字をそのまま比べた' if same.get('same_as_yesterday') else '**同じとは言えない** → A2 を今日走らせ直した(deviations.md)'}。")
w('')
w('## 1. やったこと')
w('')
w(f"- 場所: {ev['where']}。")
st = ev['settings']
w(f"- 設定(10/14 と同じ): 制御の遅れ {st['control_delay_frames']} フレーム・先読み {st['lookahead']}・ゲイン 舵 {st['steer_gain']} / 前進 {st['throttle_gain']}・"
  f"ジャンル {st['mode']}・単周の上限 {st['single_lap_limit_s']} 秒・{st['physics_and_sensors']}。")
w('- 開始(10/14 と同じ 12 個。[ゲート, 左ずれ m, 向き 度]): ' + '、'.join(f"{e} {v}" for e, v in ev['starts'].items()) + '。')
w('- 順番(protocol eval_Q1.sessions_and_order): 1) 陽性対照(作者の ONNX の E0)→ 2) 前日との同一性(A1 の E0)→ 3) S-DIV の E0〜E11 + E0 の 2 回目 + フライングラップ'
  + (' → 4) S-MOD の E0〜E11 + フライング' if 'S-MOD' in S else '') + (' → 5) A4' if 'A4' in S else '') + '。')
w('- 一式(ZIP)と ONNX:')
w('')
w('| ラベル | ZIP | ZIP の sha256 | ONNX の sha256(一式の記録) |')
w('|---|---|---|---|')
for lab, x in ev['bundles'].items():
    w(f"| {lab} | `{Path(x['zip']).name}` | `{x['zip_sha256'][:16]}…` | `{(x['onnx_sha256_expected'] or '')[:16]}…` |")
w('')
w('## 2. 陽性対照と前日との同一性')
w('')
pc = ev.get('positive_control_today')
if pc:
    w(f"- 作者の ONNX の E0: {REASON.get(pc['reason'], pc['reason'])}・{f(pc['lap_s'], 3)} 秒・{pc['passed_gates']}/33 ゲート・"
      f"ユーザーの確認器 {('PASS' if pc['user_verifier'] else 'FAIL') if pc['user_verifier'] is not None else '—'}(10/14: 完走 9.022 秒・PASS)。")
if same:
    a1 = same.get('A1_s0_E0') or {}
    if a1:
        w(f"- A1-s0 の E0: 今日 {REASON.get(a1['today']['reason'], a1['today']['reason'])}・{a1['today']['frames']} フレーム・{a1['today']['passed_gates']} ゲート、"
          f"10/14 {REASON.get(a1['yesterday']['reason'], a1['yesterday']['reason'])}・{a1['yesterday']['frames']} フレーム。軌跡の差の最大 {a1['max_abs_xy_diff_m']} m"
          f"(決まりは ≤ 1e-6 m)、操作の差の最大 {a1['max_abs_action_diff']}。→ {'同じ' if a1['same'] else '**違う**'}。")
    ex = same.get('positive_control_EXT_E0') or {}
    if ex:
        w(f"- 作者の ONNX の E0 の軌跡の差(10/14 との比べ): 最大 {ex['max_abs_xy_diff_m']} m・フレーム数 {ex['today']['frames']} / {ex['yesterday']['frames']}。")
T = S.get('T_fast'), S.get('T_margin')
if T[0]:
    w(f"- 先生(Node・遅延 0)の単周: T_fast {T[0]['n_completed']}/{T[0]['n_ran']}(中央値 {f(T[0]['lap_median_s'], 3)} 秒)、"
      f"T_margin {T[1]['n_completed']}/{T[1]['n_ran']}(中央値 {f(T[1]['lap_median_s'], 3)} 秒)。10/14 の値と{'同じ' if T[0]['same_as_1014'] and T[1]['same_as_1014'] else '**違う**'}。")
w('')
w('## 3. 腕ごとの結果(単周 E0〜E11。判定の量)')
w('')
w('| 腕 | 走行 | 完走 c | 最初の直線を抜けた fs(ゲート 0 の 9 本) | 点 4 のゲート g4(9 本) | 進んだ道のりの中央値 d [m] | 止まるまでの時間の中央値 [秒](完走しなかった走りだけ) | 完走の周回時間の中央値 [秒] | T_fast との比 |')
w('|---|---|---|---|---|---|---|---|---|')
for a in main_arms + [x for x in ('S-T', 'S-T1', 'EXT', 'A0s1') if x in S]:
    s = S[a]
    w(f"| {NAMES.get(a, a)} | {s['n_ran']} | {s['n_completed']} | {s['fs']} / {s['gate0_ran']} | {s['g4']} / {s['gate0_ran']} | {f(s['d_median_m'], 1)} | "
      f"{f(s['stop_time_median_s'], 2)} | {f(s['lap_median_s'], 3)} | {f(s['lap_ratio_to_T_fast_median'], 3)} |")
for t in ('T_fast', 'T_margin'):
    if t in S:
        w(f"| {NAMES[t]} | {S[t]['n_ran']} | {S[t]['n_completed']} | — | — | — | — | {f(S[t]['lap_median_s'], 3)} | — |")
w('')
w('S-T・作者の ONNX・A0 種 1 は E0〜E5 の 6 本だけ(10/13 のベストラップの評価の記録)。A0 の E0〜E5 は 10/13 の S-R 種 0(A0 と同じ学習結果)の評価の記録で、10/14 の A0 もこれを使った。A0 の E6〜E11・A1・A2 は 10/14 の記録(どれも走らせ直していない)。')
w('d は 12 本の「止まるまでに参照の周に沿って進んだ道のり」の中央値(完走は 1 周 = '
  f"{ev['loop_length_m']:.1f} m)。参照の周を外れたまま走ると 1 周を越えることがある(S-T の時間切れの走りなど)。")
w('')
w('## 4. 判定(protocol.json verdicts_Q1。結果を見る前に固定)')
w('')
for k, x in V.items():
    if not isinstance(x, dict) or 'verdict' not in x:
        continue
    w(f"### {k}: 「{x['verdict']}」")
    w('')
    rule = x.get('rule')
    if isinstance(rule, dict):
        for rk, rv in rule.items():
            w(f"- 決まり「{rk}」: {rv}")
    elif rule:
        w(f"- 決まり: {rule}")
    nums = {kk: vv for kk, vv in x.items() if kk not in ('verdict', 'rule')}
    w(f"- 数: `{json.dumps(nums, ensure_ascii=False)}`")
    w('')
sp = V.get('speed') or {}
w('### 速さ(判定には使わない)')
w('')
for a, x in sp.items():
    if a == 'rule':
        continue
    w(f"- {a}: 完走の周回時間の中央値 {f(x['lap_median_s'], 3)} 秒、T_fast との比の中央値 {f(x['ratio_to_T_fast_median'], 3)}、"
      f"T_margin との比 {f(x['ratio_to_T_margin_median'], 3)}(作者の ONNX の E0 は 9.022 秒)。最速の先生に近い(≤ 1.05): {f(x['close_to_fastest_teacher'])}")
w(f"- 決まり: {sp.get('rule', '')}")
w('')
w('### 開始ごとの比べ(参考): 進んだ道のり')
w('')
for k, x in (V.get('paired_reference') or {}).items():
    w(f"- {k}: 長い {x['longer']}・短い {x['shorter']}・同じ {x['ties']}(符号検定の p = {f(x['sign_test_p_two_sided_reference'], 3)}、参考)")
w('')
w('### E0〜E5 での外の参照(判定には使わない)')
w('')
w('| 腕 | E0〜E5 の完走 |')
w('|---|---|')
for a, x in (V.get('external_reference_E0_E5') or {}).items():
    w(f"| {NAMES.get(a, a)} | {x['completed']} / {x['ran']} |")
w('')
w('## 5. 走行ごと(今日の腕と 10/14 の A1・A2)')
w('')
for a in today_arms + ['A2', 'A1']:
    if a not in S:
        continue
    w(f"### {NAMES.get(a, a)}")
    w('')
    w('| 開始 | 終わり方 | 時間 [秒] | ゲート | 道のり [m] | 最初の直線 | 点 4 | 最初に外したゲート(横 m) | ぶつかった物(x, z) | 最小すき間 [m] | T_fast の線から 1 秒 / 最後 / 最大 [cm] | 先生に任せて立て直せた最後(T_fast / T_margin)[止まる何秒前] |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for e in STARTS:
        r = ev['runs'].get(f'{a}/{e}')
        if not r:
            continue
        col = r.get('collision') or {}
        fg = r.get('first_gate_miss') or {}
        pf = r.get('path_dev_fast_m') or {}
        tk = r.get('takeover') or {}
        tf = (tk.get('fast') or {}).get('seconds_before_end')
        tm = (tk.get('margin') or {}).get('seconds_before_end')
        t = r['lap_s'] if r['counted_complete'] else r['sim_time']
        w(f"| {e} | {'**完走**' if r['counted_complete'] else REASON.get(r['reason'], r['reason'])} | {f(t, 2)} | {r['passed_gates']} | {f(r.get('progress_m'), 1)} | "
          f"{f(r.get('first_straight_passed')) if e in GATE0 else '—'} | {f(r.get('g4')) if e in GATE0 else '—'} | "
          f"{('点 %s(%+.2f)' % (fg.get('point'), fg.get('lateral_m'))) if fg else '—'} | "
          f"{(KIND.get(col.get('kind'), col.get('kind')) + '(%.2f, %.2f)' % (col.get('x') or 0, col.get('z') or 0)) if col else '—'} | {f(r.get('min_clearance_m'), 3)} | "
          f"{f((pf.get('at_1s') or 0) * 100 if pf.get('at_1s') is not None else None, 1)} / {f((pf.get('at_end') or 0) * 100 if pf.get('at_end') is not None else None, 1)} / "
          f"{f((pf.get('max') or 0) * 100 if pf.get('max') is not None else None, 1)} | {f(tf, 1)} / {f(tm, 1)} |")
    w('')
notes = []
for a in today_arms:
    for e in GATE0:
        r = ev['runs'].get(f'{a}/{e}')
        fg = (r or {}).get('first_gate_miss') or {}
        if r and r.get('g4') and fg.get('point') == 4:
            notes.append(f"{a} の {e}: 点 4 のゲートを最初は {abs(fg['lateral_m']) - 0.65:.2f} m 外し({fg['t']:.2f} 秒)、コースを 1 周回ってから数えた"
                         f"(protocol の g4 の定義「数えたゲート ≥ 4」では通ったに入る。止まったのは {r['sim_time']:.2f} 秒・{r['passed_gates']} ゲート)")
if notes:
    w('補足(g4 の数え方):')
    w('')
    for n in notes:
        w(f'- {n}')
    w('')
w('「最初に外したゲート」は、ゲートの線を |横| > 0.65 m で越え、そのとき sim がそのゲートを数えなかったもの。線の延長を少しだけ外れても sim が数えたもの'
  '(すれすれ。測る位置の違い)は外しに入れない。')
w('')
w('## 6. 決定性・フライングラップ(参考)')
w('')
det = ev.get('determinism_SDIV_E0')
if det:
    w(f"- S-DIV の E0 を 2 回: 1 回目 {REASON.get(det['run1']['reason'], det['run1']['reason'])}・{det['run1']['frames']} フレーム、2 回目 {REASON.get(det['run2']['reason'], det['run2']['reason'])}・"
      f"{det['run2']['frames']} フレーム。位置・速さの差の最大 {det['max_abs_diff_xz_speed']} → {'同じ(決定的)' if det['identical'] else '**違う**'}。")
for k, x in (ev.get('flying_reference') or {}).items():
    laps = x.get('laps') or []
    w(f"- {k}(E0 から 20 秒の連続周回): {REASON.get(x['reason'], x['reason'])}・{f(x['sim_time'], 2)} 秒・{x['passed_gates']} ゲート・衝突 {x['collisions']}・"
      f"有効な周 {sum(1 for l in laps if l.get('valid'))}(1 周目 {f(x.get('lap1_s'), 3)} 秒・フライング {f(x.get('flying_lap_s'), 3)} 秒)。")
w('')
w('## 7. 確かめ')
w('')
va = ev['verification_all_runs']
w(f"- 今日の走行 {va['runs_today']} 本のうち、記録した操作を元の sim-core.js で再生した自前の数え直しが一致: {va['consistency_pass_today']} 本(verify_eval.mjs)。")
w(f"- 先生の目の再生(teacher_eval.mjs)で終わり方・フレーム数が同じ: {va['replay_same_teacher_eval']} / {va['teacher_eval_rows']} 本。")
w(f"- 設定(モデルの sha256・遅延・先読み・ゲイン・ジャンル・開始・時間の上限)が計画と違う走行: {len(va['setting_mismatches'])} 本 {va['setting_mismatches'] or ''}。読めない記録: {len(va['unreadable'])} 本。")
uv = ev.get('user_verifier') or {}
if uv:
    w('- ユーザーの確認器(qa/verify-learned-lap.mjs <結果> --model <ONNX>、E0 の単周): ' + '、'.join(f"{k} {'PASS' if v.get('pass') else 'FAIL'}" for k, v in uv.items()) +
      '(完走していない走りは FAIL になるのが正しい。完走の判定に使うのは完走した E0 だけ)。')
if rc:
    w(f"- 別の道具での数え直し(review/recount_q1.py、生の JSON だけを読む): 腕ごとの c・fs・g4・本数が " +
      '、'.join(f"{k} {'同じ' if v['same_counts'] else '**違う**'}" for k, v in rc['arms'].items()) + '。判定: ' +
      '、'.join(f"{k} {'同じ' if v['same'] else '**違う**'}({v['eval']} / {v['recount']})" for k, v in rc['verdicts'].items()) + '。')
    w('  d は別の線(world.route の 33 点の折れ線)で測り直し、1 周の割合 × 26.1 m に直した値: ' +
      '、'.join(f"{k} {f(v['d_eval_m'], 1)} → {f(v['d_recount_polyline_scaled_m'], 1)} m" for k, v in rc['arms'].items()) + '(線が違うので少し違ってよい)。')
    ru = rc['replay_user_tool']
    w(f"- ユーザーの replay-report.mjs(有効な版 releases/20261013-144013-6f610dd5 の dist/sim-core.js)で全走行を再生: {ru['pass']} / {ru['runs']} PASS"
      f"{'(FAIL ' + str(ru['fail']) + ')' if ru['fail'] else ''}{'(再生していない ' + str(ru['missing']) + ')' if ru['missing'] else ''}。")
    w(f"- 突き合わせ全体: {'**すべて同じ**' if rc['all_same'] else '**違いあり**(review/recount_check.json)'}。")
spd = ev.get('speed_of_eval') or []
if spd:
    w('- 評価の速さ: ' + '、'.join(f"{x['label']} {x['phase']}(試み {x['attempt']}) {x['frames']} フレーム・{x['fps']} フレーム/秒" +
                                   (f"・止まった {len(x['stopped'])} 本" if x.get('stopped') else '') for x in spd) + '。')
w('')
if dg:
    w('## 8. DAgger(Q1b)')
    w('')
    for line in dg.get('md_lines', []):
        w(line)
    w('')
w('## 9. 図と画面')
w('')
FIGDOC = {'fig/eval_completion.png': '完走の数と、止まるまでに進んだ道のり(腕ごと)', 'fig/eval_distance_time.png': '開始ごとの止まるまでの時間・道のり',
          'fig/eval_trajectories.png': '軌跡と止まった場所(× 衝突・□ 時間切れ)', 'fig/eval_first_corner.png': '最初の曲がり(点 4 のゲート)の拡大',
          'fig/eval_deviation.png': 'T_fast の線からの距離の時間変化', 'fig/eval_screens_sdiv_e2.png': 'S-DIV の E2 の完走の画面と車載カメラ',
          'fig/dagger_states.png': 'DAgger の状態集めと切り替えのフレーム k'}
for p in list(ev['figures']) + [q for q in ('fig/eval_screens_sdiv_e2.png', 'fig/dagger_states.png') if (R / q).exists() and q not in ev['figures']]:
    w(f'- `{p}` — {FIGDOC.get(p, "")}')
scr = sorted((R / 'screens').glob('*.png'))
if scr:
    w(f"- 画面(screens/、{len(scr)} 枚): 窓・車載カメラ・ToF。例 " + '、'.join(f'`{p.name}`' for p in scr[:6]) + ' …')
w('')
w('## 10. 限界')
w('')
w('- 種は 1 つ(種 0)だけ。シミュレーターは決定的で、ばらつくのは開始だけ。12 本の差は「この 1 本の学習結果」の差で、学習の種によるばらつきは測っていない。')
w('- 評価はシミュレーターの中だけ。実車の試験はこの作業の範囲の外(実車・PCA9685・I2C に触れていない)。')
w('- d(進んだ道のり)は最速の先生の参照の周に沿った長さ。ゲートを外して周の内側を走ると、道のりが伸びても周を回ったとは言えない。')
(R / 'eval_sim.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(R / 'eval_sim.md', len(lines), '行')
