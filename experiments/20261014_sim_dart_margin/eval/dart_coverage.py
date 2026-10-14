#!/usr/bin/env python3
"""参考(結果を見てから足した。合否に使わない): DART のデータ(D1)で乱しを入れた場所と、A1 が先生の線から離れた場所を、コースの区間ごとに並べる
(sim_dart_20261014 の評価の段、2026-10-14)。データは読むだけ。

    python -I eval/dart_coverage.py

  - 区間 i = 元の経路の点 i → 点 i+1。各フレームの位置を、前のフレームの区間から先 3 つの区間の中でいちばん近い区間に当てる(開始はゲート 0)
  - D1 の 30 シーン(frames.jsonl の action_info.dart.noise): 乱したフレーム、乱しの始まり(イベント)の区間ごとの数
  - A1 の単周(ゲート 0 の開始): 先生(T_fast)の線から 5 cm・10 cm を最初に超えたフレームの区間(teacher_eval.json の series)
  - 乱しの見張り(protocol dart.guards.gate_guard: 次の目標ゲートの手前 1.2 m で |横| > 0.30 m なら始めない)と、T_fast が点 4 のゲートを
    横 約 −0.56 m で通ることの関係を確かめるため、点 3 → 点 4 の区間で乱しが始まった数を書く
結果 → eval/dart_coverage.json、図 fig/eval_dart_coverage.png
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_common import CL, EVAL, REPORT  # noqa: E402

D1 = sorted(glob.glob('/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-dart-fast-10min_*'))[0]


def seg_dist(p, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dz * dz
    u = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L)) if L > 0 else 0.0
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dz)


def segments_of(xy, route, start=0):
    n = len(route)
    cur, out = start, []
    for p in xy:
        best, bi = 1e9, cur
        for k in range(0, 4):
            i = (cur + k) % n
            d = seg_dist(p, route[i], route[(i + 1) % n])
            if d < best - 1e-9:
                best, bi = d, i
        cur = bi
        out.append(bi)
    return out


def main():
    a1e0 = json.loads((CL / 'A1-s0' / 'A1-s0-E0-single.json').read_text())
    route = a1e0['world']['route']   # ゲート 0 の開始 = 元の経路のまま
    assert a1e0['context']['start_variation']['gate'] == 0
    n = len(route)
    noisy = [0] * n
    frames = [0] * n
    starts = [0] * n
    scenes = 0
    for f in sorted(glob.glob(f'{D1}/simulation/scenes/*/frames.jsonl')):
        rows = [json.loads(line) for line in open(f)]
        xy = [(r['pose_pre_action']['x'], r['pose_pre_action']['z']) for r in rows]
        segs = segments_of(xy, route)
        prev_ev = -1
        for r, s in zip(rows, segs):
            dt = (r.get('action_info') or {}).get('dart') or {}
            frames[s] += 1
            if dt.get('noise'):
                noisy[s] += 1
                if dt.get('ev') != prev_ev:
                    starts[s] += 1
                prev_ev = dt.get('ev')
        scenes += 1
    te = json.loads((EVAL / 'teacher_eval.json').read_text())
    a1 = {}
    for r in te['rows']:
        if not r['name'].startswith('A1-s0-') or r['start_variation']['gate'] != 0 or 'rep2' in r['name']:
            continue
        xy = list(zip(r['series']['x'], r['series']['z']))
        segs = segments_of(xy, route)
        p = r['series']['path_teacher']
        f5 = next((i for i, v in enumerate(p) if v is not None and v > .05), None)
        f10 = next((i for i, v in enumerate(p) if v is not None and v > .10), None)
        a1[r['name']] = {'over_5cm_frame': f5, 'over_5cm_segment': None if f5 is None else segs[f5],
                         'over_10cm_frame': f10, 'over_10cm_segment': None if f10 is None else segs[f10]}
    # 腕ごとの止まった区間(最後の位置)と、最初に外したゲートの点
    ends = {}
    for r in te['rows']:
        lab = {'SR-s0': 'A0-s0'}.get(r['label'], r['label'])
        if lab not in ('A0-s0', 'A1-s0', 'A2-s0') or 'rep2' in r['name'] or r['browser_reason'] == 'route_complete':
            continue
        g = r['start_variation']['gate']
        xy = list(zip(r['series']['x'], r['series']['z']))
        segs = segments_of(xy, route, start=(g - 1) % n if g else 0)
        es = segs[-1]
        gm = r.get('first_gate_miss')
        ends[r['name']] = {'arm': lab, 'end_segment': es, 'end_segment_noisy_frac': noisy[es] / frames[es] if frames[es] else None,
                           'end_segment_event_starts': starts[es], 'first_gate_miss_point': gm['point'] if gm else None,
                           'reason': r['browser_reason']}
    by_arm = {}
    for v in ends.values():
        a = by_arm.setdefault(v['arm'], {'runs': 0, 'end_in_unperturbed_segment': 0})
        a['runs'] += 1
        a['end_in_unperturbed_segment'] += int(noisy[v['end_segment']] == 0)
    out = {'schema': 'sim-dart-coverage/v1', 'note': '参考。区間 i = 元の経路の点 i → 点 i+1', 'dataset': D1, 'scenes': scenes,
           'frames_per_segment': frames, 'noisy_frames_per_segment': noisy, 'event_starts_per_segment': starts,
           'noisy_frames_total': sum(noisy), 'event_starts_total': sum(starts), 'A1_first_departure': a1,
           'unperturbed_segments': [i for i in range(n) if noisy[i] == 0], 'run_ends': ends, 'run_ends_by_arm': by_arm,
           'segment_3_to_4': {'frames': frames[3], 'noisy_frames': noisy[3], 'event_starts': starts[3]},
           'segments_0_to_4': {i: {'frames': frames[i], 'noisy': noisy[i], 'starts': starts[i]} for i in range(5)}}
    (EVAL / 'dart_coverage.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    # 図
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_eval import jp_font, style, SURFACE, COLORS
    jp_font()
    fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
    style(ax, 'D1(DART 10 分)で乱しを入れた場所(区間 i = 点 i → 点 i+1。灰の帯 = 乱しが 1 フレームも無い区間)、'
              'A1 が T_fast の線から 10 cm 離れた区間(▼)、止まった区間(×)',
          '乱したフレームの割合 [%]', '区間(元の経路の点の番号)')
    frac = [100 * noisy[i] / frames[i] if frames[i] else 0 for i in range(n)]
    ax.bar(range(n), frac, .7, color='#2a78d6', alpha=.8, label='乱したフレーム / その区間のフレーム')
    cnt = {}
    for v in a1.values():
        if v['over_10cm_segment'] is not None:
            cnt[v['over_10cm_segment']] = cnt.get(v['over_10cm_segment'], 0) + 1
    top = max(frac) if frac else 1
    for s, c in cnt.items():
        ax.plot(s, top * 1.08, marker='v', color=COLORS['A1-s0'], ms=8 + 2 * c)
        ax.text(s, top * 1.17, str(c), ha='center', fontsize=8, color=COLORS['A1-s0'])
    for arm, dy, mk in (('A1-s0', .0, 'x'), ('A2-s0', .07, 'x'), ('A0-s0', .14, 'x')):
        cc = {}
        for v in ends.values():
            if v['arm'] == arm:
                cc[v['end_segment']] = cc.get(v['end_segment'], 0) + 1
        for sgi, k in cc.items():
            ax.plot(sgi, -top * (.08 + dy), marker=mk, color=COLORS[arm], ms=6 + 1.5 * k, mew=2, clip_on=False)
        ax.plot([], [], marker=mk, color=COLORS[arm], ls='none', label=f'{arm[:2]} が止まった区間(大きいほど本数が多い)')
    for i in range(n):
        if noisy[i] == 0:
            ax.axvspan(i - .45, i + .45, color='#e4e3df', alpha=.6, lw=0)
    ax.set_xticks(range(n))
    ax.set_ylim(-top * .3, top * 1.62)
    ax.set_yticks([t for t in range(0, int(top) + 1, 20)])
    ax.plot([], [], marker='v', color=COLORS['A1-s0'], ls='none', label='A1 が T_fast の線から最初に 10 cm 離れた区間(数字 = 本数)')
    ax.legend(fontsize=8, loc='upper right', frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT / 'fig' / 'eval_dart_coverage.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps({k: out[k] for k in ('scenes', 'noisy_frames_total', 'event_starts_total', 'unperturbed_segments', 'run_ends_by_arm', 'run_ends')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
