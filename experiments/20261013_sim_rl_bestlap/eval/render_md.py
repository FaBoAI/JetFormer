#!/usr/bin/env python3
"""evaluation.json から evaluation.md(閉ループ評価の結果の表)を作る(2026-10-14、段 E)

    python -I eval/render_md.py
数字はすべて evaluation.json から読む(手で書き写さない)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_common import REPORT, STARTS  # noqa: E402

LABELS = ['SR-s0', 'SR-s1', 'ST-s0', 'ST-s1', 'EXT']
NAMES = {'SR-s0': 'S-R 種 0', 'SR-s1': 'S-R 種 1', 'ST-s0': 'S-T 種 0', 'ST-s1': 'S-T 種 1', 'EXT': '外部の参照(作者の ONNX)'}
REASON = {'collision': '衝突', 'time_limit': '時間切れ', 'route_complete': '完走', 'collection_duration_complete': '20 秒'}


def f3(x):
    return '—' if x is None else f'{x:.3f}'


def cell(r):
    """停止から 1 周のマス: 完走なら秒、だめなら終わり方・ゲート数・時刻"""
    if r is None:
        return '(未実行)'
    if r['completed'] and r['single_lap_s'] is not None:
        return f"**{r['single_lap_s']:.3f}**"
    why = REASON.get(r['reason'], r['reason'])
    extra = ''
    if r.get('collision_event'):
        extra = f"・{r['collision_event']['raw'].get('kind', '')}"
    elif r.get('gate_misses_orig'):
        g = r['gate_misses_orig'][0]
        extra = f"・点 {g['route_point']} のゲートを外した({100 * (g['lateral_m'] - .65):.1f} cm 外)"
    return f"× {why} {r['passed_gates']}/33・{r['sim_time']:.2f} 秒{extra}"


def main():
    ev = json.loads((REPORT / 'evaluation.json').read_text())
    acc = ev['acceptance_H2']
    ref = ev['references']
    S = ev['single_runs']
    L = []
    w = L.append
    w('# 閉ループ評価の結果(sim_rl_bestlap_20261013、段 E)')
    w('')
    w(f"- 作成 {ev['created_at']}。手順は protocol.json(sha256 `{ev['protocol_sha256'][:8]}…{ev['protocol_sha256'][-4:]}`、記録した値 "
      f"`{ev['protocol_sha256_recorded'][:8]}…` と{'一致' if ev['protocol_sha256'] == ev['protocol_sha256_recorded'] else '不一致'})の closed_loop_eval・acceptance_H2 のとおり。基準は変えていない。")
    w(f"- 場所: {ev['where']}")
    w('- 設定: 制御遅延 1 フレーム・先読み 1・舵ゲイン 1.0・前進ゲイン 1.0(外部の参照は 1.01)・ジャンル 8(外部の参照は 4)・単周の上限 15 秒・'
      '物理とセンサーは記録の 60 条件(REC60)と同じ(RC 応答の物理・measured-height-approx)。')
    w('- 物語の本文は README.md。この文書は表と判定だけ。数字はすべて `evaluation.json` から作った(`eval/render_md.py`)。')
    w('')
    w('## 1. 判定(H2: JetFormer が RL のベストラップを学んだか)')
    w('')
    a, b, c = acc['a_completion'], acc['b_close_to_rl'], acc['c_faster_than_teacher_trained']
    w(f"**判定: {acc['verdict']}**")
    w('')
    w('| 基準(protocol.json acceptance_H2) | 値 | 合否 |')
    w('|---|---|:-:|')
    w(f"| (a) S-R の種 0・1 がどちらも E0 を完走しユーザーの確認器が PASS、かつ 12 回のうち 9 回以上完走 | E0 の両方: {'はい' if a['E0_both_seeds_complete_and_user_verifier_pass'] else 'いいえ'}・完走 {a['completed_of_12']}/12 | {'○' if a['pass'] else '×'} |")
    med = b['median_ratio']
    w(f"| (b) S-R の完走した回の T(S-R, e)/T_RL(e) の中央値 ≤ 1.05 | {'完走した回が無い' if med is None else f'{med:.3f}'} | {'○' if b['pass'] else '×'} |")
    if c['rule'] == 'main':
        cv = f"組 {c['n_pairs']}・差の中央値 {c['median_diff']:.3f} 秒・Δ {c['delta']:.3f} 秒"
    else:
        ext = c.get('ext_E0')
        cv = f"組 {c['n_pairs']}(3 未満)→ 予備の基準: S-R の E0 {', '.join(f3(x) for x in c.get('sr_E0', []))} 対 外部の参照の E0 {f3(ext)} − 0.10 秒"
    w(f"| (c) S-R と S-T の両方が完走した組で T(S-T) − T(S-R) の中央値 > Δ(組が 3 未満なら予備の基準) | {cv} | {'○' if c['pass'] else '×'} |")
    w('')
    w('## 2. 停止から 1 周(単周・E0〜E5)')
    w('')
    hdr = '| モデル | ' + ' | '.join(f'{e}(ゲート {g}・{o:+.2f} m・{y:+.0f}°)' for e, (g, o, y) in STARTS.items()) + ' |'
    w(hdr)
    w('|---|' + '---|' * len(STARTS))
    for lab in LABELS:
        w(f'| {NAMES[lab]} | ' + ' | '.join(cell(S.get(f'{lab}/{e}')) for e in STARTS) + ' |')
    w('| RL の方策(遅延 0・Node・元のコア) | ' + ' | '.join(f3(ref['T_RL'][e]) for e in STARTS) + ' |')
    w('| RC 教師(遅延 0・Node。E0〜E2 だけが教師の約束の範囲) | ' + ' | '.join(f3(ref['T_teacher'].get(e)) for e in STARTS) + ' |')
    w('')
    w(f"- 単位は秒。× は完走しなかった回(終わり方・通過したゲート / 33・終わった時刻)。「点 n」は元の経路の点の番号。")
    w(f"- 参考: RL の方策の遅延 1 の E0 は {ref['RL_delay1_STD']:.3f} 秒、RC 教師の遅延 1 は {REASON.get(ref['teacher_delay1_STD']['reason'])}"
      f"(ゲート {ref['teacher_delay1_STD']['passed_gates']} のあと)。外部の参照の作者の記録(作者の環境のブラウザ)は {ref['learned_onnx_user_record_s']:.3f} 秒。")
    w('')
    w('## 3. フライングラップ(連続周回 20 秒、E0・E1)')
    w('')
    w('| モデル | E0: 終わり方 / 1 周目 / 2 周目 | E1: 終わり方 / 1 周目 / 2 周目 |')
    w('|---|---|---|')
    for lab in LABELS:
        cells = []
        for e in ('E0', 'E1'):
            fl = ev['flying'].get(f'{lab}/{e}')
            if not fl:
                cells.append('(未実行)')
                continue
            why = REASON.get(fl['reason'], fl['reason'])
            laps = ' / '.join((f"{l['elapsed']:.3f}" + ('' if l['valid'] else '(無効)')) for l in fl['laps']) or 'なし'
            cells.append(f"{why}({fl['sim_time']:.2f} 秒・ゲート {fl['passed_gates']}) / {laps}")
        w(f'| {NAMES[lab]} | ' + ' | '.join(cells) + ' |')
    rf, tf = ref['RL_flying'], ref['teacher_flying']
    w(f"| RL の方策(遅延 0) | 20 秒 / {rf['E0']['lap1']:.3f} / {rf['E0']['lap2']:.3f} | 20 秒 / {rf['E1']['lap1']:.3f} / {rf['E1']['lap2']:.3f} |")
    w(f"| RC 教師(遅延 0) | 20 秒 / {tf['E0']['lap1']:.3f} / {tf['E0']['lap2']:.3f} | 20 秒 / {tf['E1']['lap1']:.3f} / {tf['E1']['lap2']:.3f} |")
    w('')
    w('- 「(無効)」は sim は周として数えたが、自前の数え直しで 33 ゲート・黒シートの有効な横断ちょうど 1 回・衝突 0 のどれかを満たさなかった周(ベストラップの定義に合わない)。')
    w('')
    w('## 4. 参考(合否に使わない)')
    w('')
    det = ev.get('determinism') or {}
    if det:
        w(f"- **決定性**: S-R 種 0 の E0 を 2 回: {REASON.get(det['run1']['reason'])}・{det['run1']['frames']} フレーム と "
          f"{REASON.get(det['run2']['reason'])}・{det['run2']['frames']} フレーム、軌跡の差の最大 {det['max_abs_diff_common_frames']}。"
          f"{'同じだったので、3 回の中央値は使っていない' if det['identical'] else '違った(deviations.md)'}。")
    g = ev.get('gain_sensitivity_SR_s0_E0') or {}
    if g:
        w('- **ゲインの感度**(S-R 種 0 の E0): ' + '、'.join(
            f"{k}: {REASON.get(v['reason'], v['reason'])}{'' if v['lap_s'] is None else ' ' + f3(v['lap_s'])}(ゲート {v['passed_gates']})" for k, v in sorted(g.items())))
    ex = ev.get('extra_runs_not_judged') or []
    if ex:
        w('- **事後に足した参考の条件**(E0 の単周。結果を見てから決めた。deviations.md):')
        w('')
        w('| 走行 | 遅延 | 先読み | 舵 / 前進ゲイン | 終わり方(時刻・場所) | 停止から 1 周 | ゲート | ユーザーの確認器 |')
        w('|---|---:|---:|---|---|---:|---:|---|')
        for r in ex:
            where = ''
            if r.get('collision_event'):
                ce = r['collision_event']
                where = f"・{ce['raw'].get('kind', '')}(x {ce['x']:.2f}, z {ce['z']:.2f})"
            elif r.get('gate_misses_orig'):
                where = f"・点 {r['gate_misses_orig'][0]['route_point']} のゲートを外した"
            uv = r.get('user_verifier')
            w(f"| {r['name']} | {r['control_delay_ticks']} | {r['lookahead']} | {r['steer_gain']} / {r['throttle_gain']} | {REASON.get(r['reason'], r['reason'])}"
              f"({r['sim_time']:.2f} 秒{where}) | {f3(r['single_lap_s'])} | {r['passed_gates']} | {'—' if uv is None else ('PASS' if uv else 'FAIL')} |")
        w('')
    ed = ev.get('diff_vs_external_reference_s') or {}
    if not any(d and any(x is not None for x in d.values()) for d in ed.values()):
        w('- **外部の参照との差**: 学習したモデルと外部の参照の両方が完走した開始は無かった(差は計算できない)。')
    else:
        w('- **外部の参照との差**(T(モデル) − T(外部の参照)、両方が完走した開始だけ): ' + '、'.join(
            f"{e}: " + ', '.join(f"{NAMES[k]} {v:+.3f}" for k, v in d.items() if v is not None) for e, d in ed.items() if d and any(x is not None for x in d.values())))
    off = ev.get('offline_l1_us') or {}
    if off:
        w('')
        w('- **オフライン L1**(学習の段の `train/offline_l1.json`。全フレーム 7,080・間引きなし・PyTorch FP32。単位 µs):')
        w('')
        w('| モデル | RL の val: 舵 1 フレーム先 / スロットル 1 フレーム先 / 10 ステップの平均 | 教師の val: 舵 / スロットル / 10 ステップの平均 |')
        w('|---|---|---|')
        for mk in ('S-R/s0', 'S-R/s1', 'S-T/s0', 'S-T/s1'):
            o = off.get(mk, {})
            def t(sn):
                v = o.get(sn)
                return '—' if not v else f"{v['steer_step1_us']:.1f} / {v['throttle_step1_us']:.1f} / {v['both_mean10_us']:.1f}"
            w(f"| {mk} | {t('RL_val')} | {t('teacher_val')} |")
        w('')
    to = ev.get('takeover_diagnostic_not_judged')
    if to:
        w('- **引き継ぎの診断**(`eval/takeover.json`。ブラウザの走行の操作を Node の元のコアで再生し、フレーム k(3 フレームおき)から先を RL の方策'
          '(遅延 0・正解の状態)に任せて単周を完走できるか。付け直し = 同じ状態で RL の方策が出す舵と、JetFormer が実際に使った舵の差):')
        w('')
        w('| 走行 | ブラウザの終わり方 | フレーム数 | RL が引き継いで完走できた最後の k | RL が引き継いでも完走できなかった最初の k | 付け直しの舵の差の平均(全体 / 最初の 20 フレーム)[rad] |')
        w('|---|---|---:|---:|---:|---|')
        for r in to['rows']:
            w(f"| {r['name']} | {REASON.get(r['browser_reason'], r['browser_reason'])} | {r['browser_frames']} | {r['last_recoverable_frame']} | "
              f"{'—' if r['first_unrecoverable_frame'] is None else r['first_unrecoverable_frame']} | {r['relabel_steer_abs_mean_rad']:.3f} / {r['relabel_steer_abs_mean_first20_rad']:.3f} |")
        w('')
    w('## 5. 確かめたこと')
    w('')
    v = ev['verification_all_runs']
    w(f"- 全 {v['runs']} 走行で、記録した操作を元の sim-core.js で再生すると軌跡・集計が一致し、自前の積分とゲート・黒シート・衝突の数え直しも一致した"
      f"(記録の正しさ {v.get('consistency_pass_strict', v['consistency_pass'])}/{v['runs']})。設定(モデルの sha256・遅延・先読み・ゲイン・開始)が計画と違う走行: {len(v['setting_mismatches'])}。")
    uv = ev.get('user_verifier') or {}
    if uv:
        w('- ユーザーの `qa/verify-learned-lap.mjs <結果> --model <ONNX>`(E0 の単周): ' + '、'.join(
            f"{k}: {'PASS' if x['pass'] else 'FAIL'}" + ('' if x['pass'] else f"({(x.get('failures') or ['?'])[0][:60]})") for k, x in sorted(uv.items())))
    w('')
    rv = ev.get('review_20261014')
    if rv:   # 確認の段(review.md、2026-10-14)が足した欄。数え直しと「失敗の形」の数字の出どころ
        ic, fs = rv['independent_recheck'], rv['failure_shape_numbers']
        w('### 5.1 確認の段(review.md)で数え直したこと')
        w('')
        c, t = ic['closed_loop_59'], ic['rl_teacher_records_18']
        w(f"- Python の自前の積分・判定(`review/independent_lap_check.py`。JS のコードを使わない)で、閉ループの {c['runs']} 走行すべてが sim の集計と一致"
          f"({c['agree_with_sim']}/{c['runs']}、軌跡の差の最大 {c['max_trace_diff']:.1e})。数えた周 {c['laps_counted']} のうち、ベストラップの定義に合う周 {c['laps_best_lap_valid']}"
          f"(合わない 3 周は 17 秒台の周で、黒シートの横断が 2 回)。RL・教師の記録 {t['runs']} 本も一致({t['agree_with_sim']}/{t['runs']})。")
        rs = ic['random_starts']
        d = rs['draw']
        w(f"- ランダムな 2 つの開始(種 {d['seed']}): P1 = ゲート {d['P1']['gate']}・{d['P1']['offset']:+.3f} m・{d['P1']['yawDeg']:+.2f}°、"
          f"P2 = ゲート {d['P2']['gate']}・{d['P2']['offset']:+.3f} m・{d['P2']['yawDeg']:+.2f}°。選んだ RL の方策を元の sim-core.js で走らせ、自前の判定で数え直した:")
        for k in ('P1-single', 'P1-continuous', 'P2-single', 'P2-continuous'):
            x = rs['results'][k]
            w(f"  - {k}: {' / '.join(f'{s:.3f}' for s in x['laps_s'])} 秒、ベストラップの定義に合う {sum(x['best_lap_valid'])}/{len(x['best_lap_valid'])}、"
              f"最小すき間 {100 * x['min_clearance_m']:.1f} cm、sim と一致 {'○' if x['agree_with_sim'] else '×'}")
        w('  - ユーザーの確認器の周の項目(完走・衝突・ゲート・黒シート・再生): ' + '、'.join(
            f"{k}: {'合格' if v else '不合格'}" for k, v in rs['user_verifier_lap_items'].items()))
        sc = ic['recorded_scenes_2_random']
        w(f"- データセットのランダムな 2 シーン(種 {sc['draw']['seed']}): " + '、'.join(
            f"{k}: {' / '.join(f'{s:.3f}' for s in v['laps_s'])} 秒(有効 {sum(v['best_lap_valid'])}/{len(v['best_lap_valid'])})" for k, v in sc['results'].items()))
        w('')
        w('### 5.2 失敗の形の数字(README 5.3 節。`review/derived_numbers.json`)')
        w('')
        n = fs['first_straight_nearest_obstacle_m']
        w(f"- 最初の直線({n['definition']})でいちばん近い障害物: RL の E0 = {n['RL_E0']['id']}({n['RL_E0']['kind']})まで {100 * n['RL_E0']['d']:.1f} cm、"
          f"教師の E0 = {n['teacher_E0']['id']}({n['teacher_E0']['kind']})まで {100 * n['teacher_E0']['d']:.1f} cm。"
          f"S-R がぶつかったレール segment-11 までは RL {100 * fs['rail_segment11_min_clearance_m']['RL_E0']:.1f} cm・教師 {100 * fs['rail_segment11_min_clearance_m']['teacher_E0']:.1f} cm。")
        w(f"- 周全体の最小すき間(標準の単周): RL {100 * fs['whole_lap_min_clearance_m']['RL_E0']:.1f} cm・教師 {100 * fs['whole_lap_min_clearance_m']['teacher_E0']:.1f} cm。")
        a, b, e = fs['SR_dev_at_crash_vs_RL_E0_m'], fs['SR_dev_at_crash_same_start_m'], fs['SR_dev_at_0p5s_same_start_m']
        w(f"- S-R(E0〜E4 の 10 本)の RL の線からのずれ: ぶつかった時点で {100 * a[0]:.0f}〜{100 * a[1]:.0f} cm(E0 の RL の線から)・"
          f"{100 * b[0]:.0f}〜{100 * b[1]:.0f} cm(同じ開始の RL の走りから)。0.5 秒の時点で {100 * e[0]:.1f}〜{100 * e[1]:.1f} cm(同じ開始の RL から)。")
        g = fs['gate_point4_lateral_m']
        w('- 点 4 のゲートを通った横の位置(半幅 0.65 m。副ステップで計算): ' + '、'.join(f"{k} {v:.3f} m" for k, v in g.items() if v is not None))
        r = fs['SR_s0_E0_relabel_minus_used_steer_rad_frames15_26']
        w(f"- S-R 種 0 の E0、フレーム 15〜26: 同じ状態で RL の方策が出す舵 − JetFormer が使った舵 = {r[0]:+.2f}〜{r[1]:+.2f} rad。"
          'S-R の 12 本のぶつかった相手: ' + '・'.join(f"{k} {v}" for k, v in fs['SR_collision_kinds_12'].items()))
        w('')
    w('## 6. 図')
    w('')
    for fpath in ev['figures']:
        w(f'![{Path(fpath).stem}]({fpath})')
        w('')
    w('## 7. ファイル')
    w('')
    w('- 1 走行ずつの全記録(sim.report() + world + context + 毎フレームの推論): `eval/closed_loop/<モデル>/<走行>.json`、軌跡: 同じ名前の `.trajectory.csv`')
    w('- 1 走行ずつの確認と要約: `eval/closed_loop/runs_verified.json`、表: `eval/runs.csv`、ユーザーの確認器の出力: `eval/closed_loop/user_verifier/`')
    w('- 判定: `evaluation.json`(`acceptance_H2`)・`eval/verdict.json`、一式の sha256: `eval/bundles/bundles.json`、ブラウザの実行の記録: `eval/closed_loop/session_*.json`')
    w('- 画面: `screens/eval_*.png`(窓・車載カメラ・ToF)')
    (REPORT / 'evaluation.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(REPORT / 'evaluation.md')


if __name__ == '__main__':
    main()
