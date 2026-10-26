#!/usr/bin/env python3
"""desk_bench_20261022.json → 図(fig/slow_variants.png)と desk_bench_20261022.md(CPU・matplotlib だけ。torch を読まない)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        /home/jetson/jetformer/experiments/20261022_prep/desk/desk_report2.py
図の形・色は 10/21 の desk_report.py と同じ(setup を読む)。
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_lib2 as D2  # noqa: E402

D2.register()
import desk_report as R  # noqa: E402  (10/21 の台本。色と setup)

OUT = Path('/home/jetson/jetformer_reports/prep_20261022/desk')
FIG = OUT / 'fig'
NOTES = HERE / 'desk_notes2.md'


def f(x, fmt='%.1f'):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return '合格' if x else '不合格'
    return fmt % x


def figure(doc):
    plt = R.setup()
    rows = doc['candidates']
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    n = len(rows)
    ys = np.arange(n)[::-1].astype(float)
    h = 0.36
    series = (('r1_app_thr_diff_us', 'R1 の手前 1 秒(r1_app_thr_diff_us・行 index 2)', R.BLUE,
               lambda r: (r.get('ref') or {}).get('r1_app_thr_diff_ci90_us')),
              ('nonzone_thr_diff_us', '区間の外 none・unknown(nonzone_thr_diff_us・行 index 0)', R.ORANGE,
               lambda r: (r.get('ref_nonzone') or {}).get('ci90_scene_boot_us')))
    for j, (key, lab, col, ci) in enumerate(series):
        off = h / 2 + 0.02 if j == 0 else -(h / 2 + 0.02)
        for y, r in zip(ys, rows):
            v = r.get(key)
            if v is None:
                ax.text(0.5, y + off, '値なし', va='center', color=R.INK2, fontsize=12)
                continue
            ax.barh(y + off, v, height=h, color=col, edgecolor=R.SURFACE, linewidth=2,
                    label=lab if y == ys[0] else None)
            e = ci(r)
            xt = v
            if e and e[0] is not None:
                ax.plot([e[0], e[1]], [y + off, y + off], color=R.INK, lw=1.2)
                for xx in e:
                    ax.plot([xx, xx], [y + off - 0.08, y + off + 0.08], color=R.INK, lw=1.2)
                xt = max(v, e[1])
            ax.text(max(xt, 0), y + off, '  %+.1f' % v, va='center', ha='left', color=R.INK, fontsize=12)
    ax.axvline(0, color=R.GRAY, lw=1.2)
    ax.axvline(D2.A10_R1_MIN_US, color=R.GRAY, lw=1.5, ls='--')
    ax.text(D2.A10_R1_MIN_US, ys[0] + 0.66, 'A10 の下限 +20(R1 の手前)', ha='center', va='bottom', color=R.INK2, fontsize=12)
    ax.set_yticks(ys)
    pick = (doc.get('a10_c1') or {}).get('pick')

    def tick(r):
        tag = ''
        if r['id'] == pick and r['id'] != 'B':
            tag = '\n(A10 の答え C1)'
        elif r.get('gates_pass') is False:
            tag = '\n(門で外れる)'
        return '%s\n%s%s' % (r['id'], r['label'], tag)
    ax.set_yticklabels([tick(r) for r in rows])
    ax.set_ylim(-0.7, n + 0.05)
    ax.set_xlabel('予測のスロットル − 人のスロットル [µs](正 = 人より緩める)')
    ax.set_title('遅く入る型の比べ(保留の人の走行 4,670 フレーム・best の重み・線は 90% 区間)', loc='left', color=R.INK)
    ax.grid(axis='x', color=R.GRID, lw=0.8)
    ax.set_axisbelow(True)
    vals = [0.0, 45.0]
    for r in rows:
        for key, _, _, ci in series:
            if r.get(key) is not None:
                vals.append(r[key])
            vals.extend(v for v in (ci(r) or []) if v is not None)
    ax.set_xlim(min(vals) - 4, max(vals) + 9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.12), ncol=2, frameon=False, fontsize=12)
    fig.tight_layout()
    p = FIG / 'slow_variants.png'
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def md(doc, fig_path):
    rows = doc['candidates']
    by = {r['id']: r for r in rows}
    c1 = doc['a10_c1']
    L = []
    a = L.append
    a('# 机上のベンチ(10/23 の段 Desk・A10)')
    a('')
    a('- 作った時刻: %s。台本: `experiments/20261022_prep/desk/`(desk_eval2.py・desk_report2.py)。' % doc['created_at'])
    a('- 対象: B・SLOW(10/21 の予測を読むだけ)と、A10 の新しい 3 本(SLOW-SEC0・SLOW1・SLOW-SEC1)。')
    a('- 欄と集合は 10/21 の desk_bench.json と同じ。足した欄は nonzone_thr_diff_us(A10)だけ。')
    a('- protocol.json の sha256: %s(%s)。' % (doc['protocol_sha256'][:12] + '…', '記録と同じ' if doc['protocol_sha256_ok'] else '**記録と違う**'))
    a('')
    a('![遅く入る型の比べ](fig/%s)' % fig_path.name)
    a('')
    a('## 1. 表(保留 4,670 フレーム・best の重み。L1 は µs)')
    a('')
    a('| id | 学習結果 | L1 | ステアの L1 | スロットルの L1 | L1(last) | 介入の L1 | r1_app | nonzone | 振れ幅の比 | 低速ブレーキ | TRT p99 [ms] | engine | sanity | 門 |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        gate = '基準' if r['id'] == 'B' else ('合格' if r.get('gates_pass') else '不合格 ' + ','.join(r.get('gate_fail_reasons') or []))
        a('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['id'], r['run'], f(r.get('l1_best_us')), f(r.get('l1_steer_best_us')), f(r.get('l1_thr_best_us')),
            f(r.get('l1_last_us')), f(r.get('l1_interv_us')), f(r.get('r1_app_thr_diff_us'), '%+.1f'),
            f(r.get('nonzone_thr_diff_us'), '%+.1f'), f(r.get('steer_amp_ratio'), '%.3f'),
            f(r.get('brake_lowspeed_frac'), '%.4f'), f(r.get('trt_p99_ms'), '%.2f'),
            f(r.get('engine_ok')), f(r.get('sanity_ok')), gate))
    a('')
    a('- 門は protocol の 5 つ + A1(家族 label_r1: 5 番目をステアの L1 の差 < 10 に置き換え、低速ブレーキ ≤ 0.01)。')
    a('- B との差(L1): ' + '・'.join('%s %+.1f(ステア %+.1f)' % (r['id'], r['l1_best_us'] - by['B']['l1_best_us'],
                                                              r['l1_steer_best_us'] - by['B']['l1_steer_best_us'])
                                    for r in rows if r['id'] != 'B' and r.get('l1_best_us') is not None) + ' µs。')
    a('')
    a('## 2. nonzone_thr_diff_us(区間の外)')
    a('')
    z = doc['zones']
    a('- 区間の外 = primary のうち v1.1 の判定が none %d・unknown %d の **%d フレーム**(R1_app %d・R1_turn %d・GM %d・DE %d は入れない)。' % (
        z['primary_by_zone']['none'], z['primary_by_zone']['unknown'], z['nonzone_frames'], z['primary_by_zone']['R1_app'],
        z['primary_by_zone']['R1_turn'], z['primary_by_zone']['GM'], z['primary_by_zone']['DE']))
    a('- 区間の判定は make_bench_tables.py と同じやり方(zones2.py)。学習のシーンの数が tables_card.json と一致: %s。'
      % ('はい' if z['check_train_counts_same_as_card'] else '**いいえ**'))
    a('')
    a('| id | nonzone(行 0) | 90% 区間(シーン) | B との差 | 90% 区間 | 行 2 | 10 行 | 予測の平均 | 人の平均 | none だけ | unknown だけ | primary 全体(行 0) |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        q = r.get('ref_nonzone') or {}
        ci = q.get('ci90_scene_boot_us') or [None, None]
        cb = q.get('vs_B_ci90_scene_boot_us') or [None, None]
        bz = q.get('by_zone_row0_us') or {}
        a('| %s | %s | %s〜%s | %s | %s〜%s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['id'], f(r.get('nonzone_thr_diff_us'), '%+.1f'), f(ci[0], '%+.1f'), f(ci[1], '%+.1f'),
            f(q.get('vs_B_us'), '%+.1f'), f(cb[0], '%+.1f'), f(cb[1], '%+.1f'), f(q.get('row2_us'), '%+.1f'),
            f(q.get('allrows_us'), '%+.1f'), f(q.get('pred_thr_row0_us'), '%.1f'), f(q.get('human_thr_row0_us'), '%.1f'),
            f((bz.get('none') or {}).get('thr_diff_us'), '%+.1f'), f((bz.get('unknown') or {}).get('thr_diff_us'), '%+.1f'),
            f(q.get('primary_all_row0_us'), '%+.1f')))
    a('')
    a('- 単位は µs。正 = 人より緩める(PWM は前進が小さい値。中立 1498)。90% 区間は 11 シーンの復元抽出(参考。門・選び方に使わない)。')
    a('')
    a('## 3. A10 の C1 の決め方(計算しただけ。まだ当てていない)')
    a('')
    a('- 決まり: 対象 = {SLOW・SLOW1・SLOW-SEC0・SLOW-SEC1} のうち A1 の門に合格し、r1_app_thr_diff_us ≥ +20.0。'
      'nonzone_thr_diff_us(小数第 1 位)の小さい順 → r1_app の大きい順 → id の順。')
    if c1['targets']:
        a('- 対象の並び: ' + ' → '.join('%s(nonzone %+.1f・r1_app %+.1f)' % (t['id'], t['nonzone_thr_diff_us_r1'], t['r1_app_thr_diff_us'])
                                    for t in c1['targets']) + '。')
    if c1['excluded']:
        a('- 対象外: ' + '・'.join('%s(%s)' % kv for kv in c1['excluded'].items()) + '。')
    a('- **決まりの答え: C1 = %s(%s)**%s。' % (c1['pick'], c1.get('pick_run'), '(対象が空なので SLOW のまま)' if c1['empty_fallback'] else ''))
    a('- data/bench_candidates.json は変えていない(C1 は今も SLOW = bench_20261021-slow-s0)。入れ替えは後の段で。')
    a('')
    a('## 4. 確かめ')
    a('')
    chk = doc['check_vs_20261021']
    a('- B・SLOW の欄を 10/21 の desk_bench.json と同じ値で作れた: %s(%d 個の値を比べた)。' % ('はい' if chk['pass'] else '**いいえ** %s' % chk['diffs'][:3], chk['compared']))
    cc_p, rc_p = HERE / 'logs' / 'crosscheck_B.json', HERE / 'logs' / 'recount2.json'
    if cc_p.exists():
        cc = json.load(open(cc_p, encoding='utf-8'))
        a('- 予測の道具の照らし合わせ: B を予測し直し、10/12 の保存との差は最大 %s µs(上限 0.01、%s。%s)。'
          % (cc.get('max_abs_us'), '合格' if cc.get('pass') else '**不合格**', cc.get('checked_at')))
    if rc_p.exists():
        rc = json.load(open(rc_p, encoding='utf-8'))
        a('- 別のコード(recount2.py。desk_lib・desk_eval を読まない)での数え直し: %s(C1 の答えも %s)。'
          % ('全部一致' if rc.get('pass') else '**食い違いあり**', rc.get('a10_c1_recount', {}).get('pick')))
    a('- 区間の判定に使った future_valid(clean_driving)と、primary の future_valid(ai_mix)が違うフレーム: primary の中に 0。')
    a('- 予測は `pred/`(新しい 3 本の best・last、sanity_<id>.npz、区間 zones_heldout.npz、まとめ desk_inputs_20261022.npz)。')
    a('- 数: primary %d・介入 %d・R1 の手前 1 秒 %d(起点 %d)・保留の低速 %d・区間の外 %d。' % (
        doc['counts']['primary'], doc['counts']['interv'], doc['counts']['r1_pre1s_frames'], doc['counts']['r1_anchors_used'],
        doc['counts']['lowspeed_heldout'], doc['counts']['nonzone']))
    a('')
    if NOTES.exists():
        a('## 5. 見えたこと')
        a('')
        a(NOTES.read_text(encoding='utf-8').strip())
        a('')
    (OUT / 'desk_bench_20261022.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    doc = json.load(open(OUT / 'desk_bench_20261022.json', encoding='utf-8'))
    p = figure(doc)
    md(doc, p)
    print('書きました', p, OUT / 'desk_bench_20261022.md')


if __name__ == '__main__':
    main()
