#!/usr/bin/env python3
"""10月26日のレポート(約 1 分)の動画: 学習 v3 の段 1 の「物差し」(probe)の図とナレーションを、probe の json から作る
(CPU だけ・読むだけ。GPU・torch は使わない)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- nice -n 10 \\
        /usr/bin/python3 -I /home/jetson/jetformer/experiments/20261026_fix/fix4/video_train_fig.py

入力(読むだけ)
  - ~/jetformer/experiments/20261026_fix/logs/probe/v3_20261026-<腕>-aug2-e60-s<種>_best.json(best.pth の物差し。入れ子の json:
    {<run の場所>: {st10_thr_vs_b_us, st23_thr_vs_b_us, stand_brake_side_frac, low10_brake_frac, ...}})。無い本は「まだ」
  - 段 0 の参考: logs/probe/abl-same-e15-s0_best.json(v2 と同じ正解・15 エポック)
  - 門の値: ~/jetformer_reports/fix_20261026/protocol_v3.json の thresholds_us.probe_*(|st10| ≤ 11.111・|st23| ≤ 14.123 µs・
    stand_brake_side ≤ 0.064・low10 ≤ 0.01。protocol_v3 の「学習中の門」)
  - 判定(E1〜E3)の 1 行: ~/jetformer_reports/summary_20261026/video/judge_e123.json。"placeholder": true のあいだは使わない。
    main session が E1〜E3 を decisions_v3.json に写したあと、text・say・(banner)を書いて "placeholder": false にすると、
    ナレーションの 3 行目と図の下の帯に入る
出力
  - ~/jetformer_reports/summary_20261026/video/fig/r2_train_v3.png(2160 × 1350。腕 × 種の表と、直線の点の図)
  - ~/jetformer_reports/summary_20261026/video/train_v3_narr.json(セリフ・まとめの項目・数。make_storyboard.py が読む)

決まり: セリフに時刻・先の予定・先の日付を入れない(過去の 10/10・10/23 は可)。「2本目」と言わない。say に英字・µ・% を残さない。
数はすべて json から(手で書かない)。物差しの門は best.pth で見る(protocol_v3 の gates_v3.probe)。
"""
import argparse
import datetime
import glob
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch  # noqa: E402

EXP = Path('/home/jetson/jetformer/experiments/20261026_fix')
PROBE = EXP / 'logs' / 'probe'
PROTOCOL = Path('/home/jetson/jetformer_reports/fix_20261026/protocol_v3.json')
VIDEO = Path('/home/jetson/jetformer_reports/summary_20261026/video')
OUT_FIG = VIDEO / 'fig'
OUT_JSON = VIDEO / 'train_v3_narr.json'
JUDGE = VIDEO / 'judge_e123.json'
D8SRC = Path('/home/jetson/jetformer_reports/day8_explainer_20261017/video/src/make_slides_day8.py')

ARMS = [('v3d', '364 のスロットルを\nB の予測に差し替え'), ('v3n', '364 と 37 を外す'), ('v3b', '比べ(B のデータだけ)')]
SEEDS = [0, 1, 2, 3]
RUN = 'v3_20261026-%s-aug2-e60-s%d'
REF = 'abl-same-e15-s0'                         # 段 0: v2 と同じ正解(原因 A を確かめた対照)
KEYS = ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac', 'low10_brake_frac')


def load_probe(name):
    p = PROBE / (name + '_best.json')
    if not p.exists():
        return None
    d = json.load(open(p, encoding='utf-8'))
    v = next(iter(d.values())) if d and all(isinstance(x, dict) for x in d.values()) else d   # 入れ子({run: {...}})
    if not all(k in v for k in KEYS):
        sys.exit('物差しの鍵が足りない: %s' % p)
    return {k: float(v[k]) for k in KEYS}


def gates():
    th = json.load(open(PROTOCOL, encoding='utf-8'))['thresholds_us']
    return {'st10': float(th['probe_st10']['value']), 'st23': float(th['probe_st23']['value']),
            'stand': float(th['probe_stand_brake_side']['value']), 'low10': float(th['probe_low10']['value'])}


def judge_line():
    """E1〜E3 の 1 行(無い・仮なら None)"""
    if not JUDGE.exists():
        return None
    j = json.load(open(JUDGE, encoding='utf-8'))
    if j.get('placeholder', True) or not (j.get('text') or '').strip():
        return None
    return {'text': j['text'].strip(), 'say': (j.get('say') or j['text']).strip(),
            'expression': j.get('expression', 'normal'), 'pose': j.get('pose', 'point'),
            'banner': (j.get('banner') or '').strip()}


def judge_template():
    """judge_e123.json が無ければ、仮の形を書いておく(main session が埋める)"""
    if JUDGE.exists():
        return
    tmpl = {'placeholder': True,
            '_how': 'E1〜E3 を decisions_v3.json に写したあと、text(吹き出し・60 字まで)・say(読み・英字と µ と % なし)・'
                    'banner(図の下の帯・48 字くらいまで。空なら text)を書き、placeholder を false にする。'
                    'そのあと video_train_fig.py → make_storyboard.py → check → narr_len.py --fit → 書き出し(fix4/video.md)。'
                    '時刻と、先の日付は書かない。',
            'text': '', 'say': '', 'banner': '', 'expression': 'normal', 'pose': 'point'}
    with open(JUDGE, 'w', encoding='utf-8') as f:
        json.dump(tmpl, f, ensure_ascii=False, indent=1)
    print('wrote', JUDGE, '(仮)')


def judge_run(v, g):
    ok = {'st10': abs(v['st10_thr_vs_b_us']) <= g['st10'], 'st23': abs(v['st23_thr_vs_b_us']) <= g['st23'],
          'stand': v['stand_brake_side_frac'] <= g['stand'], 'low10': v['low10_brake_frac'] <= g['low10']}
    ok['all'] = all(ok.values())
    return ok


def fnum(x):
    return ('%+.1f' % x).replace('-', '−')


def collect(g):
    runs = []
    for arm, _ in ARMS:
        for s in SEEDS:
            name = RUN % (arm, s)
            v = load_probe(name)
            runs.append({'name': name, 'arm': arm, 'seed': s, 'done': v is not None, 'probe': v,
                         'ok': judge_run(v, g) if v else None})
    return runs


def narration(runs, g, judge):
    done = [r for r in runs if r['done']]
    n, ntot = len(done), len(runs)
    if not done:
        sys.exit('済んだ本が無い(logs/probe/v3_*_best.json が無い)')
    main = [r for r in done if r['ok']['st23'] and r['ok']['stand'] and r['ok']['low10']]
    fail10 = [r for r in done if not r['ok']['st10']]
    allok = [r for r in done if r['ok']['all']]
    if len(main) == n:
        a_t = ('%d本とも' % n) if n == ntot else ('済んだ%d本とも' % n)
        a_b = ('%d 本とも' % n) if n == ntot else ('済んだ %d 本とも' % n)
    else:
        a_t = '%d本中%d本で' % (n, len(main))
        a_b = '%d 本中 %d 本で' % (n, len(main))
    l1 = {'text': '学習v3は%s、10/23の直線とブレーキの癖が消えました。' % a_t,
          'say': '学習ブイスリーは、%s、10月23日の直線と、ブレーキの癖が消えました。' % a_t,
          'expression': 'happy' if len(main) == n else 'normal', 'pose': 'point'}
    if fail10:
        pas_t = ('門を通ったのは%d本です。' % len(allok)) if allok else '門を通った本はありません。'
        pas_s = ('門を通ったのは%d本です。' % len(allok)) if allok else '門を通った本は、ありません。'
        l2 = {'text': 'ただ10/10の直線で%d本がBより緩め、物差しの%s' % (len(fail10), pas_t),
              'say': 'ただ、10月10日の直線で%d本がビーより緩め、物差しの%s' % (len(fail10), pas_s),
              'expression': 'worried', 'pose': 'point'}
    else:
        l2 = {'text': '10/10の直線も門の中で、物差しの門を通ったのは%d本です。' % len(allok),
              'say': '10月10日の直線も門の中で、物差しの門を通ったのは%d本です。' % len(allok),
              'expression': 'happy', 'pose': 'point'}
    lines = [l1, l2]
    if judge:   # 判定の行を入れるときは l2 を外す(約 1 分に収める。10/10 の直線は図と帯で見せる)
        lines = [l1, {k: judge[k] for k in ('text', 'say', 'expression', 'pose')}]
    lo = min(r['probe']['st10_thr_vs_b_us'] for r in fail10) if fail10 else None
    hi = max(r['probe']['st10_thr_vs_b_us'] for r in fail10) if fail10 else None
    if fail10:
        banner = '%s、10/23 の直線とブレーキの癖が消えた。10/10 の直線で %d 本が門の外(%s〜%s µs)' % (
            a_b, len(fail10), fnum(lo), fnum(hi))
    else:
        banner = '%s、10/23 の直線とブレーキの癖が消えた。物差しの門は %d 本が合格' % (a_b, len(allok))
    if judge:
        banner = judge['banner'] or judge['text']
    return {'n_done': n, 'n_total': ntot, 'n_main_ok': len(main), 'n_fail_st10': len(fail10), 'n_all_ok': len(allok),
            'fail_st10_range_us': [lo, hi], 'all_ok_runs': [r['name'] for r in allok],
            'lines': lines, 'banner': banner,
            'summary_item': '学習v3: 物差しの門 %d/%d本' % (len(allok), n),
            'judge_used': bool(judge)}


# ---------------------------------------------------------------- 図
def load_helpers():
    spec = importlib.util.spec_from_file_location('slides_day8', str(D8SRC))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    H.OUT = OUT_FIG
    return H


def draw(H, runs, g, ref, nar):
    INK, INK2, GRID, GREEN, RED, GRAY, BLUE, ORANGE = H.INK, H.INK2, H.GRID, H.GREEN, H.RED, H.GRAY, H.BLUE, H.ORANGE
    n, ntot = nar['n_done'], nar['n_total']
    fig = H.slide('結果: 学習 v3 の段 1(学習中の物差し)',
                  '済んだ %d / %d 本。best.pth の物差し(B との差)。門: 直線 10/10 ±%.1f・10/23 ±%.1f µs・'
                  '後ろ側 ≤ %.1f%%・低速のブレーキ ≤ %.0f%%'
                  % (n, ntot, g['st10'], g['st23'], g['stand'] * 100, g['low10'] * 100))
    # ---- 左: 腕 × 種の表
    x0, y0, w, h = 0.025, 0.14, 0.585, 0.69
    ax = fig.add_axes([x0, y0, w, h])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cols = [('腕', 0.215), ('種', 0.065), ('10/10 の直線\n[µs]', 0.15), ('10/23 の直線\n[µs]', 0.15),
            ('止まった場面\nの後ろ側', 0.15), ('低速の\nブレーキ', 0.13), ('門', 0.14)]
    xs = [0]
    for _, cw in cols:
        xs.append(xs[-1] + cw)
    hh = 0.095
    nrow = 1 + len(runs)                          # 段 0 の参考 + 12 本
    rh = (1 - hh) / nrow
    ax.add_patch(FancyBboxPatch((0, 1 - hh), 1, hh, boxstyle='square,pad=0', fc=INK, ec='none'))
    for (t, _), a, b in zip(cols, xs[:-1], xs[1:]):
        ax.text((a + b) / 2, 1 - hh / 2, t, ha='center', va='center', fontsize=12.5, fontweight='bold', color='white',
                linespacing=1.15)

    def row_y(i):
        return 1 - hh - (i + 1) * rh

    def cell(i, c, txt, col=INK, bold=False, size=13.5, ha='center'):
        xx = (xs[c] + xs[c + 1]) / 2 if ha == 'center' else xs[c] + 0.012
        ax.text(xx, row_y(i) + rh / 2, txt, ha=ha, va='center', fontsize=size, color=col,
                fontweight='bold' if bold else 'normal', linespacing=1.15)

    # 段 0 の参考の行(v2 と同じ正解)
    ax.add_patch(FancyBboxPatch((0, row_y(0)), 1, rh, boxstyle='square,pad=0', fc='#efeeea', ec=GRID, lw=1.0))
    cell(0, 0, '参考: v2 と同じ正解\n(段 0・15 エポック)', INK2, size=11.5, ha='left')
    cell(0, 1, '0', INK2, size=12.5)
    if ref:
        okr = judge_run(ref, g)
        for c, k, pct in ((2, 'st10_thr_vs_b_us', False), (3, 'st23_thr_vs_b_us', False),
                          (4, 'stand_brake_side_frac', True), (5, 'low10_brake_frac', True)):
            v = ref[k]
            ok = okr[{2: 'st10', 3: 'st23', 4: 'stand', 5: 'low10'}[c]]
            cell(0, c, ('%.1f%%' % (v * 100)) if pct else fnum(v), INK2 if ok else RED, bold=not ok, size=12.5)
        cell(0, 6, '×' if not okr['all'] else '○', RED if not okr['all'] else GREEN, bold=True, size=17)
    # 12 本
    for i, r in enumerate(runs, start=1):
        arm_i = [a for a, _ in ARMS].index(r['arm'])
        face = '#ffffff' if arm_i % 2 == 0 else '#f6f8fb'
        if r['done'] and r['ok']['all']:
            face = H.tint(GREEN, 0.10)
        ax.add_patch(FancyBboxPatch((xs[1], row_y(i)), 1 - xs[1], rh, boxstyle='square,pad=0', fc=face, ec=GRID, lw=1.0))
        cell(i, 1, str(r['seed']), INK, size=13)
        if not r['done']:
            ax.add_patch(FancyBboxPatch((xs[2], row_y(i)), 1 - xs[2], rh, boxstyle='square,pad=0', fc='#f3f3f1',
                                        ec=GRID, lw=1.0))
            cell(i, 2, 'まだ(学習の順番待ち・学習中)', GRAY, size=12.5, ha='left')
            continue
        v, ok = r['probe'], r['ok']
        for c, k, gk, pct in ((2, 'st10_thr_vs_b_us', 'st10', False), (3, 'st23_thr_vs_b_us', 'st23', False),
                              (4, 'stand_brake_side_frac', 'stand', True), (5, 'low10_brake_frac', 'low10', True)):
            txt = ('%.1f%%' % (v[k] * 100)) if pct else fnum(v[k])
            cell(i, c, txt, INK if ok[gk] else RED, bold=not ok[gk], size=13.5)
        cell(i, 6, '○' if ok['all'] else '×', GREEN if ok['all'] else RED, bold=True, size=18)
    # 腕の欄(4 行をまとめる)
    for arm_i, (arm, desc) in enumerate(ARMS):
        rows = [i for i, r in enumerate(runs, start=1) if r['arm'] == arm]
        top, bot = row_y(rows[0]) + rh, row_y(rows[-1])
        ax.add_patch(FancyBboxPatch((0, bot), xs[1], top - bot, boxstyle='square,pad=0', fc=H.tint(BLUE, 0.07),
                                    ec=GRID, lw=1.0))
        ax.text(0.012, (top + bot) / 2 + 0.028, arm, ha='left', va='center', fontsize=17, fontweight='bold', color=INK)
        ax.text(0.012, (top + bot) / 2 - 0.035, desc, ha='left', va='center', fontsize=11.2, color=INK2, linespacing=1.2)
        ax.plot([0, 1], [bot, bot], color=INK2, lw=1.6)
    for i, r in enumerate([None] + runs):              # 縦の線(まだの行は種の右だけ)
        for xx in (xs[1:-1] if (r is None or r['done']) else xs[1:3]):
            ax.plot([xx, xx], [row_y(i), row_y(i) + rh], color=GRID, lw=1.0)
    # ---- 右: 直線の点(腕ごと)
    xr, wr = 0.645, 0.33

    def dots(yb, hb, key, gate, title, ref_val=None):
        axd = fig.add_axes([xr, yb, wr, hb])
        names = [a for a, _ in ARMS]
        axd.axvspan(-gate, gate, color=H.tint(GREEN, 0.18), zorder=0)
        axd.axvline(0, color=INK2, lw=1.2, zorder=1)
        for xg in (-gate, gate):
            axd.axvline(xg, color=GREEN, lw=1.6, ls='--', zorder=1)
        vals = []
        for r in runs:
            if not r['done']:
                continue
            yy = len(names) - 1 - names.index(r['arm']) + (r['seed'] - 1.5) * 0.09
            v = r['probe'][key]
            vals.append(v)
            ok = abs(v) <= gate
            axd.scatter([v], [yy], s=120, color=GREEN if ok else RED, edgecolor='white', lw=1.2, zorder=3)
        axd.set_yticks(range(len(names)))
        axd.set_yticklabels(names[::-1], fontsize=14, fontweight='bold')
        lim = max([gate * 1.6, 5.0] + [abs(x) * 1.15 for x in vals])
        axd.set_xlim(-lim, lim)
        axd.set_ylim(-0.6, len(names) - 0.4)
        axd.set_title(title, fontsize=14, fontweight='bold', color=INK, loc='left', pad=8)
        axd.text(gate, -0.55, ' 門 ±%.1f' % gate, fontsize=11.5, color=GREEN, va='bottom', ha='left', fontweight='bold')
        H.style_ax(axd, 12)
        axd.grid(axis='y', visible=False)
        return axd

    dots(0.50, 0.29, 'st10_thr_vs_b_us', g['st10'], '10/10 の直線(B との差 µs・+ = 緩める)')
    dots(0.15, 0.25, 'st23_thr_vs_b_us', g['st23'],
         '10/23 の直線(B との差 µs)' + ('\n段 0 の v2 と同じ正解は %s(枠の外)' % fnum(ref['st23_thr_vs_b_us']) if ref else ''))
    H.banner(fig, nar['banner'], size=18 if len(nar['banner']) <= 46 else 16.5)
    H.save(fig, 'r2_train_v3')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-fig', action='store_true', help='図を作らず、ナレーションの json だけ')
    args = ap.parse_args()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    judge_template()
    g = gates()
    runs = collect(g)
    ref = load_probe(REF)
    judge = judge_line()
    nar = narration(runs, g, judge)
    out = {'_note': 'video_train_fig.py が probe の json から作った。手で直さない(直すなら台本か judge_e123.json)',
           'written_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
           'source': str(PROBE), 'gates': g, 'reference_stage0_same': ref,
           'runs': [{'name': r['name'], 'arm': r['arm'], 'seed': r['seed'], 'done': r['done'], 'probe': r['probe'],
                     'ok': r['ok']} for r in runs]}
    out.update(nar)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('wrote', OUT_JSON)
    print('済んだ %d/%d 本・10/23 とブレーキ ok %d・10/10 の直線で外 %d・物差しの門 合格 %d(%s)・判定の行 %s'
          % (nar['n_done'], nar['n_total'], nar['n_main_ok'], nar['n_fail_st10'], nar['n_all_ok'],
             '・'.join(nar['all_ok_runs']) or 'なし', 'あり' if nar['judge_used'] else 'なし(仮)'))
    for ln in nar['lines']:
        print('  セリフ:', ln['text'])
        print('  読み  :', ln['say'])
    print('  帯    :', nar['banner'])
    if not args.no_fig:
        H = load_helpers()
        H.jp_font()
        draw(H, runs, g, ref, nar)


if __name__ == '__main__':
    main()
