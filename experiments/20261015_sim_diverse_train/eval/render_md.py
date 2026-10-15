#!/usr/bin/env python3
"""evaluation.json → evaluation.md(sim_dart_20261014 の評価の段、2026-10-14)。数字はすべて evaluation.json から作る。
    python -I eval/render_md.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_common import CORE, EXT_STARTS, REPORT, STARTS  # noqa: E402

ev = json.loads((REPORT / 'evaluation.json').read_text())
S = ev['summary']
V = ev['verdicts']
NAMES = {'A0-s0': 'A0(S-R 種 0。きれいな RL 20 分)', 'A1-s0': 'A1(A0 + DART 10 分)', 'A2-s0': 'A2(余裕の線 10 分)', 'A0-s1': 'A0 種 1(S-R 種 1)',
         'ST-s0': 'S-T 種 0(作者の先生のデータ)', 'ST-s1': 'S-T 種 1', 'EXT': '作者の ONNX'}
REASON = {'collision': '衝突', 'time_limit': '時間切れ', 'route_complete': '完走', 'aborted': '中断'}
KIND = {'rail': 'レール', 'post-foot': '柱の脚', 'curtain': '幕', 'wall': '壁'}


def f3(x):
    return '—' if x is None else f'{x:.3f}'


def m1(x):
    return '—' if x is None else f'{x:.1f} m'


def cm(x):
    return '—' if x is None else f'{x * 100:.1f}'


def cell(lab, e):
    r = ev['single_runs'].get(f'{lab}/{e}')
    if r is None:
        return '·'
    if r['counted_complete']:
        return f"**{r['single_lap_s']:.3f}**"
    s = f"× {REASON.get(r['reason'], r['reason'])} {r['passed_gates']}/33・{r['sim_time']:.2f} 秒"
    c = r.get('collision') or {}
    if c.get('kind'):
        s += f"・{KIND.get(c['kind'], c['kind'])}"
    gm = (r.get('teacher_gate_miss') or None)
    if gm:
        s += f"・点 {gm['point']} 外し"
    return s


def main():
    L = []
    w = L.append
    w('# 閉ループ評価の結果(sim_dart_20261014、評価の段)')
    w('')
    w(f"- 作成 {ev['created_at']}。手順は protocol.json(sha256 `{ev['protocol_sha256'][:8]}…`、記録した値 `{ev['protocol_sha256_recorded'][:8]}…` と"
      f"{'一致' if ev['protocol_sha256'] == ev['protocol_sha256_recorded'] else '不一致'})の eval・verdicts のとおり。基準は変えていない(違えたことは deviations.md 22〜)。")
    w(f"- 場所: {ev['where']}")
    st = ev['settings']
    w(f"- 設定: 制御遅延 {st['control_delay_frames']} フレーム・先読み {st['lookahead']}・舵ゲイン {st['steer_gain']}・前進ゲイン {st['throttle_gain']}・"
      f"ジャンル {st['mode']}・単周の上限 {st['single_lap_limit_s']} 秒・{st['physics_and_sensors']}・種 {st['seed']}・{st['scenario']}。")
    w('- 開始: ' + '、'.join(f'{e} = ゲート {g}・左 {o:+.2f} m・{y:+.0f}°' for e, (g, o, y) in STARTS.items()) + '。')
    w(f"- 比べる相手のうち A0-s0 の E0〜E5・A0 種 1・S-T・作者の ONNX の E0〜E5 は、ベストラップの評価の記録(同じアプリ・同じ計画・同じ物理)を読んだもの"
      f"(走らせ直していない)。A0-s0 の E6〜E11 と、作者の ONNX の E0(陽性対照)は今日走らせた。")
    w('- 物語の本文は README.md。この文書は表と判定。数字は `evaluation.json` から作った(`eval/render_md.py`)。')
    w('')
    # ---- 要点(数字は evaluation.json から)
    R = ev['single_runs']

    def runs(lab):
        return [R[f'{lab}/{e}'] for e in STARTS if f'{lab}/{e}' in R]

    def near(r, x, z, tol=.35):
        c = r.get('collision') or {}
        return c.get('x') is not None and abs(c['x'] - x) < tol and abs(c['z'] - z) < tol
    g4 = {lab: sum(1 for r in runs(lab) if (r.get('teacher_gate_miss') or {}).get('point') == 4) for lab in ('A1-s0', 'A2-s0')}
    tm = ev['gate_point4']['teacher_margin_m']
    tf = [v['T_fast'] for v in tm.values() if v['T_fast'] is not None]
    tg = [v['T_margin'] for v in tm.values() if v['T_margin'] is not None]
    def missed4(r):
        return (r.get('teacher_gate_miss') or {}).get('point') == 4
    a1_post = [r for r in runs('A1-s0') if near(r, -.2, -2.2) and missed4(r)]
    a2_post = [r for r in runs('A2-s0') if near(r, -2.12, -2.32) and missed4(r)]
    other_post = [r for lab in ('A0-s0', 'A1-s0', 'A2-s0') for r in runs(lab) if near(r, -.2, -2.2) and not missed4(r)]
    cov = ev.get('dart_coverage_reference') or {}
    tko = [r['takeover']['fast']['last_recoverable_frame'] / 30 for r in runs('A1-s0')
           if (r.get('takeover') or {}).get('fast', {}).get('last_recoverable_frame') is not None
           and (r.get('teacher_gate_miss') or {}).get('point') == 4]
    full = [r for k, r in R.items() if r.get('progress_full_lap_t') and k.split('/')[0] in ('A0-s0', 'A1-s0', 'A2-s0')]
    g4lat = [abs(r['teacher_gate_miss']['lateral_m']) for lab in ('A1-s0', 'A2-s0') for r in runs(lab) if (r.get('teacher_gate_miss') or {}).get('point') == 4]

    def trange(rs):
        t = [r['sim_time'] for r in rs]
        return f'{min(t):.2f}〜{max(t):.2f} 秒' if t else '—'

    def starts_of(rs):
        return '・'.join(r['name'].split('-')[2] for r in rs)
    seg3 = (ev.get('dart_coverage_reference') or {}).get('segments_0_to_4', {}).get('3', {})
    w('## 0. 要点')
    w('')
    w(f"- **完走**: A0 {S['A0-s0']['n_completed']} / {S['A0-s0']['n_ran']}、A1(DART){S['A1-s0']['n_completed']} / {S['A1-s0']['n_ran']}、"
      f"A2(余裕の線){S['A2-s0']['n_completed']} / {S['A2-s0']['n_ran']}。どの腕も 1 周も完走しなかった。protocol の判定は Q1 **{V['Q1_DART']['verdict']}**・"
      f"Q2 **{V['Q2_MARGIN']['verdict']}**(完走の差は 0 で、「ゲート 0 の開始で 8 本以上が最初の直線を抜けた」の条件で)。")
    w(f"- **A0 の失敗の形は直った**: A0 は最初の直線で 1 秒前後に壁へぶつかった(ゲート 0 の開始で抜けたのは {S['A0-s0']['first_straight_passed']} / 9)。"
      f"A1 は {S['A1-s0']['first_straight_passed']} / 9、A2 は {S['A2-s0']['first_straight_passed']} / 9 で抜けた。止まるまでの時間の中央値は "
      f"A0 {S['A0-s0']['stop_time_median_s']:.2f} 秒 → A1 {S['A1-s0']['stop_time_median_s']:.2f} 秒・A2 {S['A2-s0']['stop_time_median_s']:.2f} 秒、"
      f"進んだ道のりは {S['A0-s0']['progress_m_median_failed']:.1f} m → {S['A1-s0']['progress_m_median_failed']:.1f} m・{S['A2-s0']['progress_m_median_failed']:.1f} m"
      f"(1 周 {ev['single_runs']['A1-s0/E0'].get('progress_m', 0) / max(1e-9, ev['single_runs']['A1-s0/E0'].get('lap_fraction') or 1):.1f} m)。")
    def pdmed(lab, key, scale=1.0):
        v = [(r.get('path_dev_teacher_m') or {}).get(key) for r in runs(lab)]
        v = [x * scale for x in v if x is not None]
        return statistics.median(v) if v else None
    t10m = {lab: statistics.median([r['path_dev_teacher_m']['first_frame_over_10cm'] / 30 for r in runs(lab)
                                    if (r.get('path_dev_teacher_m') or {}).get('first_frame_over_10cm') is not None]) for lab in ('A0-s0', 'A1-s0', 'A2-s0')}
    w(f"- **先生の線から離れるのが遅くなった**: その腕の先生の同じ開始の線からの距離(中央値)は、1 秒の時点で A0 {pdmed('A0-s0', 'at_1s', 100):.1f} cm・"
      f"A1 {pdmed('A1-s0', 'at_1s', 100):.1f} cm・A2 {pdmed('A2-s0', 'at_1s', 100):.1f} cm。10 cm を超えた時刻の中央値は A0 {t10m['A0-s0']:.2f} 秒・"
      f"A1 {t10m['A1-s0']:.2f} 秒・A2 {t10m['A2-s0']:.2f} 秒(4 章)。")
    w(f"- **次に止まる場所は点 4 のゲート**: 最初の直線を抜けたあと、A1 の {g4['A1-s0']} 本・A2 の {g4['A2-s0']} 本が、点 4 のゲートの手前で曲がりの内側へ入り、"
      f"ゲートを外した(線の延長を {min(g4lat):.1f}〜{max(g4lat):.1f} m 内側で越えた)。先生も点 4 は内側の端の近くを通る(ゲートの端まで T_fast {min(tf) * 100:.1f}〜{max(tf) * 100:.1f} cm、"
      f"T_margin {min(tg) * 100:.1f}〜{max(tg) * 100:.1f} cm。5 章)。余裕を持たせた線は壁からの余裕を広げた線で、このゲートの余裕はほとんど増えていない。")
    w(f"- **ゲートを外したあとも車は走り続け、別の場所でぶつかった**(sim は外したゲートを目標に残すだけで、車の動きは変わらない): "
      f"A1 は {len(a1_post)} 本({starts_of(a1_post)})が点 21〜22 の柱の脚(x ≈ −0.2, z ≈ −2.2)に {trange(a1_post)} で、"
      f"A2 は {len(a2_post)} 本({starts_of(a2_post)})が点 23〜24 の柱の脚(x ≈ −2.1, z ≈ −2.3)に {trange(a2_post)} でぶつかった"
      f"(どちらも点 4 を外した走り。点 21〜22 の柱の脚には、ゲート 20 から始める E5 でも " + '・'.join(r['name'] for r in other_post) + ' がぶつかった)。'
      + ('参照の周に沿って 1 周ぶん進んだ走りもある(' + '、'.join(f"{r['name']} {r['progress_full_lap_t']:.2f} 秒" for r in full)
         + '。点 4 を外したので周にはならない。その後 2 周目でぶつかった)。' if full else ''))
    if tko:
        w(f"- **先生への引き継ぎ**(参考): 点 4 を外した A1 の走りで、T_fast に任せて完走できた最後の時刻は {min(tko):.2f}〜{max(tko):.2f} 秒(最初の直線の終わり、"
          f"点 4 の手前)。それより後は、先生でも点 4 のゲートを通れない。")
    if cov:
        ue = cov['run_ends_by_arm']
        w(f"- **DART の乱しが無かった場所**(参考、結果を見てから足した): D1 で乱しを入れたのは 33 区間のうち {33 - len(cov['unperturbed_segments'])} 区間。"
          f"乱しが 1 フレームも無い区間(点 " + '・'.join(str(i) for i in cov['unperturbed_segments']) + ' から次の点まで)で、'
          f"A1 の {ue['A1-s0']['end_in_unperturbed_segment']} / {ue['A1-s0']['runs']} 本が止まった(A0 は {ue['A0-s0']['end_in_unperturbed_segment']} / {ue['A0-s0']['runs']} 本)。"
          '乱しは、先 2 m に壁から 5 cm 離れて動ける幅が 60 cm 以上ある場所(黒シートの区間と裏の直線は除く)で、しかも次のゲートの手前 1.2 m では'
          '車がゲートの中心から 30 cm 以内のときだけ始める決まり(protocol dart.guards)。先生は点 4 のゲートの中心から 55〜60 cm の所を通るので、'
          f"点 3 → 4 の区間で始まった乱しは {seg3.get('starts', '—')} 回(前の区間から続いた乱しのフレームは {seg3.get('noisy', '—')})。")
    pc = ev.get('positive_control_today') or {}
    w(f"- **評価のしくみの確かめ**: 作者の ONNX の E0(陽性対照)は {f3(pc.get('lap_s'))} 秒で完走・ユーザーの確認器 PASS(ベストラップと同じ値)。"
      f"A1 の E0 を 2 回走らせて軌跡が同じ(決定性)。今日の {ev['verification_all_runs']['runs_today']} 走行すべてで、記録の操作の再生と数え直しが一致。")
    w('- **今日は答えない問い**: Q3(余裕の線 + DART)と Q4(DAgger)。A3・A4・種 1 は削った(8 章)。')
    w('')
    w('## 1. 判定(protocol.json verdicts。結果を見る前に固定)')
    w('')
    w('| 問い | 腕の完走 c | A0-s0 の完走 | 差 | ゲート 0 の開始で最初の直線を抜けた | 判定 | Fisher の p(参考) |')
    w('|---|---:|---:|---:|---:|---|---:|')
    for q, lab in (('Q1_DART', 'A1-s0'), ('Q2_MARGIN', 'A2-s0')):
        v = V[q]
        w(f"| {q}(A1・DART)" if q == 'Q1_DART' else f"| {q}(A2・余裕の線)")
        L[-1] += (f" | {v['c_arm']} / {v['n_runs_arm']} | {v['c_A0_s0']} / {v['n_runs_A0']} | {v['diff']:+d} | "
                  f"{v['first_straight_passed_gate0_starts']} / {v['gate0_starts_run']} | **{v['verdict']}** | {v['fisher_p_two_sided_reference']:.2f} |")
    w(f"| Q3_BOTH(A3) | — | — | — | — | {V['Q3_BOTH']['verdict']} | — |")
    w(f"| Q4_DAGGER(A4) | — | — | — | — | {V['Q4_DAGGER']['verdict']} | — |")
    w('')
    w(f"- 決まり(Q1・Q2 共通の形): 効いた = c(腕) ≥ 6 かつ c(腕) − c(A0-s0) ≥ 4。一部効いた = それでなく、差 ≥ 2、またはゲート 0 の開始で 8 本以上が最初の直線を抜けた。"
      f"それ以外 = 効かなかった。{V['Q1_DART']['rule_note']}")
    sc = V['Q2_MARGIN']['speed_cost']
    w(f"- Q2 の速さの代わり: A2 が完走した開始が {len(sc['per_start'])} 本なので、"
      + ('周回時間の比べはできない。' if not sc['per_start'] else f"A2 の中央値 {f3(sc['A2_lap_median_s'])} 秒、同じ開始の T_fast {f3(sc['T_fast_node_median_same_starts_s'])} 秒・"
                                                                   f"T_margin {f3(sc['T_margin_node_median_same_starts_s'])} 秒・A1 {f3(sc['A1_lap_median_same_starts_s'])} 秒。"))
    sv = V['speed_vs_teacher']
    parts = []
    for lab, v in sv.items():
        parts.append(f"{lab[:2]} = " + ('—(完走なし)' if v['median_ratio'] is None else f"{v['median_ratio']:.3f}"))
    w('- 先生に近い速さ(完走した走りの T(腕, e) / T_先生(e) の中央値 ≤ 1.05): ' + '、'.join(parts) + '。')
    b = V['best_arm']
    if b['arm'] is None:
        w('- 一番良い腕(完走の数、同じなら完走した周の時間の中央値): **決められない**(完走 ' + '・'.join(f'{k} {v}' for k, v in b['c'].items())
          + '。3 つとも完走 0 で、同点を分ける周回時間も無い)。決まりの外の参考: 止まるまでに進んだ道のりの中央値は 3 章。')
    else:
        w(f"- 一番良い腕(完走の数、同じなら完走した周の時間の中央値): **{b['arm']}**(完走 " + '・'.join(f'{k} {v}' for k, v in b['c'].items()) + ')。')
    w('')
    w('## 2. 停止から 1 周(単周・E0〜E11)')
    w('')
    hdr = '| モデル | ' + ' | '.join(STARTS) + ' | 完走 |'
    w(hdr)
    w('|---|' + '---|' * len(STARTS) + '---:|')
    for lab in ('A0-s0', 'A1-s0', 'A2-s0', 'A0-s1', 'ST-s0', 'ST-s1', 'EXT'):
        s = S[lab]
        w(f"| {NAMES[lab]} | " + ' | '.join(cell(lab, e) for e in STARTS) + f" | {s['n_completed']} / {s['n_ran']} |")
    for kind, lab in (('fast', 'T_fast(Node・遅延 0)'), ('margin', 'T_margin(Node・遅延 0)')):
        tn = ev['teachers_node_delay0']
        w(f'| {lab} | ' + ' | '.join(f3(tn[e][kind]['lap_s']) for e in STARTS) + f" | {sum(1 for e in STARTS if tn[e][kind]['lap_s'])} / 12 |")
    w('')
    w('- 単位は秒。太字 = 完走(route_complete・衝突 0・33 ゲート・黒シート 1 回、再生の数え直しが一致、設定が計画どおり。E0 はユーザーの確認器も PASS)。'
      '× = 完走しなかった回(終わり方・数えたゲート / 33・止まった時刻・ぶつかった物・最初に外したゲート)。· = 走らせていない(ベストラップは E0〜E5 だけ)。'
      '「点 n 外し」= 元の経路の点 n のゲートの線を、ゲートの幅(±0.65 m)の外で越えた、またはその手前で曲がって線を越えなかった(sim はそのゲートを目標に残すので、その後のゲートは数えない)。')
    w('')
    w('## 3. 腕ごとのまとめ')
    w('')
    w('| 腕 | 完走 | うち E0〜E5 | うち E6〜E11 | 最初の直線を抜けた(ゲート 0 の開始) | 止まった時刻の中央値 | 止まるまでに進んだ道のりの中央値 | 完走した周の中央値 |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|')
    for lab in ('A0-s0', 'A1-s0', 'A2-s0', 'A0-s1', 'ST-s0', 'ST-s1', 'EXT'):
        s = S[lab]
        w(f"| {NAMES[lab]} | {s['n_completed']} / {s['n_ran']} | {s['n_completed_E0_E5']} / {s['n_ran_E0_E5']} | "
          f"{s['n_completed_E6_E11']} / {s['n_ran_E6_E11']} | {s['first_straight_passed']} / {s['first_straight_starts']} | "
          f"{f3(s['stop_time_median_s'])} 秒 | {m1(s.get('progress_m_median_failed'))} | {f3(s['lap_median_s'])} |")
    w('')
    w('## 4. 止まった時刻と場所・先生の線からの距離・先生への引き継ぎ')
    w('')
    w('先生の線からの距離 = 走行のフレーム f の位置から、その腕の先生(A0・A1・S-T・作者の ONNX は T_fast、A2 は T_margin)が同じ開始から走った線のうち'
      '前後 1.5 秒の部分のいちばん近い点まで。引き継ぎ = フレーム k まで走行の操作、k から先を先生(遅延 0)に任せて単周を完走できる最後の k(参考)。')
    w('')
    w('| 走行 | 終わり方 | 時刻 | 場所(x, z) | 進んだ道のり | 最初に外したゲート(横の位置。先生は同じゲートで) | 距離 0.5 秒 | 距離 1 秒 | 距離 最後 | 距離 最大(ゲートを外すまで) | 10 cm を超えた時刻 | 引き継ぎ T_fast | 引き継ぎ 腕の先生 |')
    w('|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|')
    for lab in ('A0-s0', 'A1-s0', 'A2-s0'):
      for e in STARTS:
        r = ev['single_runs'].get(f'{lab}/{e}')
        if r is None or r['counted_complete']:
            continue
        c = r.get('collision') or {}
        pd = r.get('path_dev_teacher_m') or {}
        gm = r.get('teacher_gate_miss')
        gms = '—' if not gm else (f"点 {gm['point']}({gm['t']:.2f} 秒、横 {gm['lateral_m']:+.2f} m。先生 {gm['teacher_lateral_m']:+.2f} m)"
                                   if gm.get('teacher_lateral_m') is not None else f"点 {gm['point']}({gm['t']:.2f} 秒、横 {gm['lateral_m']:+.2f} m)")
        t10 = pd.get('first_frame_over_10cm')
        tk = r.get('takeover') or {}

        def tko(k):
            v = tk.get(k)
            if not v:
                return '—'
            if v['last_recoverable_frame'] is None:
                return 'どこからも不可'
            return f"k = {v['last_recoverable_frame']}({v['last_recoverable_frame'] / 30:.2f} 秒。止まる {v['seconds_before_end']:.2f} 秒前)"
        own = 'margin' if lab == 'A2-s0' else 'fast'
        place = f"{KIND.get(c.get('kind'), c.get('kind') or '')}({c['x']:.2f}, {c['z']:.2f})" if c.get('x') is not None else '—'
        prog = '—' if r.get('progress_m') is None else f"{r['progress_m']:.1f} m({r['lap_fraction'] * 100:.0f}%)"
        w(f"| {r['name']} | {REASON.get(r['reason'], r['reason'])} | {r['sim_time']:.2f} | {place} | {prog} | {gms} | {cm(pd.get('at_0p5s'))} | {cm(pd.get('at_1s'))} | "
          f"{cm(pd.get('at_end'))} | {cm(pd.get('max_before_first_gate_miss'))} | {'—' if t10 is None else f'{t10 / 30:.2f}'} | {tko('fast')} | "
          f"{tko(own) if own != 'fast' else '(同じ)'} |")
    w('')
    w('- 進んだ道のり = 参照の周 R に沿って進んだ長さ(最速の先生の進みの追跡の s。ゲートを数えたかとは関係ない)。括弧は 1 周の長さに対する割合。')
    w('- 距離の単位は cm(A0-s0 の E0〜E5 もこの段で同じ道具で測り直した。ベストラップの「衝突の時点で 10〜18 cm」と同じ範囲)。')
    w('')
    w('- A0-s0 の E0〜E5 は、ベストラップの記録の走行(名前 SR-s0-…)。距離はこの段の道具で測り直し、引き継ぎはベストラップの eval/takeover.json(k を 3 おきに全部試したもの)。')
    full = [(k, r) for k, r in ev['single_runs'].items() if r.get('progress_full_lap_t') and k.split('/')[0] in ('A0-s0', 'A1-s0', 'A2-s0')]
    if full:
        w('- 参照の周 R に沿った進みが 1 周に届いた走り(ゲートを外したので周としては数えない): ' + '、'.join(
            f"{r['name']}({r['progress_full_lap_t']:.2f} 秒で 1 周ぶん進み、{r['sim_time']:.2f} 秒に{REASON.get(r['reason'], r['reason'])})" for k, r in full) + '。')
    gp = ev.get('gate_point4') or {}
    if gp:
        w('## 5. 点 4 のゲート(最初の直線の終わりの曲がり。点 4〜6)')
        w('')
        w(gp['text'])
        w('')
        w('| 開始 | T_fast の横の位置(余裕) | T_margin の横の位置(余裕) | A1 | A2 | A0-s0 |')
        w('|---|---|---|---|---|---|')
        for e, row in gp['rows'].items():
            w(f"| {e} | {row['T_fast']} | {row['T_margin']} | {row['A1-s0']} | {row['A2-s0']} | {row['A0-s0']} |")
        w('')
    w('## 6. 確かめ')
    w('')
    pc = ev.get('positive_control_today')
    if pc:
        w(f"- **陽性対照**(作者の ONNX の E0、今日): {REASON.get(pc['reason'], pc['reason'])}・{f3(pc['lap_s'])} 秒・ゲート {pc['passed_gates']}・"
          f"ユーザーの確認器 {'PASS' if pc['user_verifier'] else 'FAIL'}(ベストラップの評価では {f3(pc['bestlap_E0_s'])} 秒)。")
    d = ev.get('determinism') or {}
    if d:
        w(f"- **決定性**(A1 の E0 を 2 回): {REASON.get(d['run1']['reason'])}・{d['run1']['frames']} フレーム と {REASON.get(d['run2']['reason'])}・"
          f"{d['run2']['frames']} フレーム、軌跡の差の最大 {d['max_abs_diff_common_frames']}。{'同じだったので 3 回の中央値は使っていない。' if d['identical'] else '違った(deviations.md)。'}")
    va = ev['verification_all_runs']
    w(f"- **再生の数え直し**: 今日の {va['runs_today']} 走行のうち {va['consistency_pass_today']} 走行で、記録した操作を元の sim-core.js で再生した軌跡・集計が一致(差 ≤ 1e-9)。"
      f"先生の目の見直し(teacher_eval)でも {va['replay_same_teacher_eval']} / {va['teacher_eval_rows']} 走行で同じ終わり方・フレーム数。"
      f"設定(モデルの sha256・遅延・先読み・ゲイン・ジャンル・開始)が計画と違う走行: {len(va['setting_mismatches'])}。")
    uv = ev['user_verifier']
    w('- **ユーザーの確認器**(qa/verify-learned-lap.mjs、E0 の単周): ' + '、'.join(
        f"{k} {'PASS' if v.get('pass') else 'FAIL'}" + ('(ベストラップ)' if v.get('source') else '') for k, v in sorted(uv.items())) +
      '。完走しなかった走行は確認器が FAIL を出す(周が無いため)。完走の判定に使うのは完走した E0 だけ。')
    w('')
    fl = ev.get('flying_reference') or {}
    if fl:
        w('## 7. 参考: 連続周回 20 秒(E0)')
        w('')
        w('| モデル | 終わり方 | 時刻 | ゲート | 1 周目 | 2 周目 |')
        w('|---|---|---:|---:|---:|---:|')
        for lab, v in fl.items():
            w(f"| {NAMES.get(lab, lab)} | {REASON.get(v['reason'], v['reason'])} | {v['sim_time']:.2f} | {v['passed_gates']} | {f3(v['lap1_s'])} | {f3(v['flying_lap_s'])} |")
        w('')
    w('## 8. 削ったもの')
    w('')
    w('- A3(余裕の線 + DART)は記録の段で削った(deviations.md 11)。A4(DAgger)と種 1 は電源断のあとの時間割で削った(deviations.md 17)。')
    w('- 評価の本数は削っていない(E0〜E11 を A0-s0・A1・A2 の 3 つの腕で全部走らせ、フライングラップも走らせた)。')
    w('')
    w('## 9. 図とファイル')
    w('')
    for f in ev['figures']:
        w(f'- `{f}`')
    for k, v in ev['files'].items():
        w(f'- {k}: `{v}`')
    (REPORT / 'evaluation.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(REPORT / 'evaluation.md')


if __name__ == '__main__':
    main()
