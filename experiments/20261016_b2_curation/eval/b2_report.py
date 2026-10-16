#!/usr/bin/env python3
"""B2 の評価の図と文書(CPU。evaluation.json・result.json・engine の json を読むだけ)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261016_b2_curation/eval/b2_report.py all

書くもの(/home/jetson/jetformer_reports/b2_curation_20261016/):
  fig/*.png        日本語(Noto Sans CJK JP)・文字 16 pt 以上・2160×1350 ピクセル
  training.md      学習の記録(B2 の種 0・1、B との比較、ログの最初の 4 行の確かめ)
  evaluation.md    机上の評価(全部のゲートの値と合否・参考の値・予想との答え合わせ)
  aba_card_b2.md   来週の実車のカード(固定した card_aba_b2.md の写しに、机上の結果とエンジンの結果を埋めたもの)
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPORT = Path('/home/jetson/jetformer_reports/b2_curation_20261016')
FIG = REPORT / 'fig'
REPO = Path('/home/jetson/jetformer')
RUNS = REPO / 'data' / 'runs'
FIGSIZE, DPI = (14.4, 9.0), 150
# 参照のパレット(dataviz の既定。1 = 青、2 = 橙)・文字は文字の色
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
GRAY, LGRAY, INK, INK2, GRID, SURF = '#52514e', '#b9b8b2', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
GOOD, BAD = '#008300', '#e34948'
CLS = {'better': '良い', 'equivalent': '同等', 'worse': '悪い', None: '—'}
EJ = {'fp16_vs_fp32_le_2p5us': 'FP32 との差 ≤ 2.5 µs', 'finite': 'NaN・Inf なし', 'run_numpy_median_le_6ms': '中央値 ≤ 6.0 ms',
      'run_numpy_p99_le_8ms': 'p99 ≤ 8.0 ms', 'peak_mem_le_4096mb': 'メモリ ≤ 4,096 MB', 'median_le_B_plus_0p5': '中央値 ≤ B + 0.5 ms',
      'listed_and_genre_trained': '一覧に出てジャンル g を学習', 'drivable': '走行できる印', 'ref_same_as_B': '中心・中立が B と同じ'}
GATE_JA = {'G3-R1-held': 'R1 保留(速い 25 通過)', 'G3-R1-train': 'R1 学習のログ(速い 24 通過)',
           'G3-R1-slow': 'R1 遅い進入(43 通過)', 'G3-DE': 'DE(65 通過)', 'G3-straight': '直線(542 行)',
           'G3-no-collapse': '崩れなし', 'G3-L1': '保留の精度(L1)', 'G3-intervention': '介入の場面'}
NAMES_JA = {'B': 'モデル B', 'A_s0': 'A 種 0', 'A_s1': 'A 種 1', 'A_s2': 'A 種 2', 'A_s3': 'A 種 3',
            'B2_s0': 'B2 種 0', 'B2_s1': 'B2 種 1', 'DROP_s0': '外すだけ 種 0', 'RELABEL_s0': '付け替えだけ 種 0'}
RUN = {'B': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'B2_s0': 'b2cur_20261016-s0', 'B2_s1': 'b2cur_20261016-s1',
       'A_s0': 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
       'A_s1': 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
       'DROP_s0': 'b2cur_20261016-droponly-s0', 'RELABEL_s0': 'b2cur_20261016-relabelonly-s0'}


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1.0,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True})
    return plt


def _save(fig, name):
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def load():
    return json.loads((REPORT / 'evaluation.json').read_text(encoding='utf-8'))


def f1(x, nd=1, sign=False):
    if x is None:
        return '—'
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return '—'
    s = ('%+.' + str(nd) + 'f') % v if sign else ('%.' + str(nd) + 'f') % v
    return s.replace('-', '−')


def ok(b):
    return '**合格**' if b else '**不合格**'


# ---------------------------------------------------------------- 図

def fig_r1(res):
    """R1 の手前 1 秒の予測スロットルの差 D(+ = B より遅い)。3 つの集合 × {A の種 0〜3、B2 g=5、B2 g=4}"""
    plt = _setup()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sets = [('R1_held', 'R1 保留\n(速く近づいた 25 通過)'), ('R1_train', 'R1 学習のログ\n(速く近づいた 24 通過)'),
            ('R1_slow', 'R1 保留\n(遅く近づいた 43 通過)')]
    x = np.arange(len(sets))
    w = 0.13
    a_names = ['A_s0', 'A_s1', 'A_s2', 'A_s3']
    for k, an in enumerate(a_names):
        vals = [res['gate_detail']['5'][s]['D_A'].get(an) for s, _ in sets]
        ax.bar(x - 2.5 * w + k * w, vals, w * 0.9, color=LGRAY, edgecolor=SURF, linewidth=2,
               label='A の種 0〜3(選別なし・陰性対照)' if k == 0 else None)
    for k, (g, col, lab) in enumerate(((5, BLUE, 'B2(ジャンル 5)'), (4, ORANGE, 'B2(ジャンル 4)'))):
        d = res['gate_detail'][str(g)]
        vals = [d[s]['D'] for s, _ in sets]
        lo = [d[s]['D'] - d[s]['ci95'][0] for s, _ in sets]
        hi = [d[s]['ci95'][1] - d[s]['D'] for s, _ in sets]
        xs = x + 1.5 * w + k * w * 1.1
        ax.bar(xs, vals, w, color=col, edgecolor=SURF, linewidth=2, label=lab)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt='none', ecolor=INK2, elinewidth=2, capsize=6)
        for xx, v, l_, h_ in zip(xs, vals, lo, hi):
            top = v + h_
            ax.text(xx, top + 0.8, f1(v, 1, True), ha='center', va='bottom', fontsize=16, color=INK)
    ax.axhline(30, color=INK2, lw=2, ls='--')
    ax.text(-0.45, 29, '速い通過の合格線 +30 µs', ha='left', va='top', fontsize=16, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(-10, color=INK2, lw=1.5, ls=':')
    ax.text(-0.45, -9, '遅い通過の下限 −10 µs', ha='left', va='bottom', fontsize=16, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in sets])
    ax.set_ylabel('R1 の手前 1 秒の予測スロットル\nB との差 D [µs](+ = 遅い)')
    ax.set_title('R1 の手前のスロットル: B2 は B より弱まったか(誤差の線 = 通過の選び直しの 95% 区間)')
    ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.0, 0.90))
    fig.tight_layout()
    return _save(fig, 'eval_r1_throttle.png')


def fig_r1_passes(res):
    """通過ごと: 進入 eRPM と B2 − B(種の平均、走行のジャンル)"""
    plt = _setup()
    g = str(res['drive_genre'] or 5)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, (key, title) in zip(axes, (('HAI_r1', '保留(人 12 + AI 56 通過)'), ('T_r1', '学習のログの人(61 通過)'))):
        s = res['secondary']['all_passes'][key]
        e = np.array([p['entry_erpm'] for p in s['passes']], float)
        d = np.array(s['per_pass_mean_delta'], float)
        dropped = np.array([p['dropped_b2'] for p in s['passes']], bool)
        ai = np.array([p['set'] == 'AI' for p in s['passes']], bool)
        ax.axvline(5500, color=LGRAY, lw=2, ls='--')
        ax.axhline(0, color=INK2, lw=1)
        ax.axhline(30, color=INK2, lw=1.5, ls=':')
        ax.scatter(e[~dropped & ~ai], d[~dropped & ~ai], s=110, color=BLUE, edgecolor=SURF, linewidth=2, label='人の通過')
        if ai.any():
            ax.scatter(e[ai], d[ai], s=110, color=AQUA, marker='D', edgecolor=SURF, linewidth=2, label='AI 走行の通過')
        if dropped.any():
            ax.scatter(e[dropped], d[dropped], s=150, color=ORANGE, marker='s', edgecolor=SURF, linewidth=2,
                       label='B2 の学習から外した通過')
        ax.set_title(title)
        ax.set_ylim(min(-25.0, float(np.nanmin(d)) - 3), max(48.0, float(np.nanmax(d)) + 18))
        ax.set_xlabel('実際の進入 eRPM')
        ax.legend(loc='upper left', frameon=False, fontsize=16)
    axes[0].set_ylabel('B2 − B の予測スロットル [µs]\n(R1 の手前 1 秒、+ = 遅い)')
    fig.suptitle('通過ごとの差(B2 は 2 本の種の平均・ジャンル %s、B はジャンル 5)。点線 = +30 µs、破線 = 5500 eRPM' % g,
                 fontsize=20)
    fig.tight_layout()
    return _save(fig, 'eval_r1_passes.png')


def fig_l1(res):
    """保留の精度 L1(best、own_labels): 主指標・rest・選別の窓・介入"""
    plt = _setup()
    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE)
    names = ['A_s0', 'A_s1', 'A_s2', 'A_s3', 'B', 'B2_s0', 'B2_s1']
    cols = [LGRAY] * 4 + [GRAY, BLUE, BLUE]
    titles = {'primary': '主指標\n(4,670 フレーム)', 'rest': '選別の窓の外\n(4,424)', 'curation_windows': '選別の窓の中\n(246)',
              'intervention_iv_all': '介入の場面\n(1,570)'}
    V = res['L1_values']['own']['best']
    for ax, m in zip(axes, ('primary', 'rest', 'curation_windows', 'intervention_iv_all')):
        v = [V[n][m] for n in names]
        ax.bar(range(len(names)), v, 0.75, color=cols, edgecolor=SURF, linewidth=2)
        lo, hi = min(v), max(v)
        pad = max(0.6, (hi - lo) * 0.6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(['A0', 'A1', 'A2', 'A3', 'B', 'B2\n種0', 'B2\n種1'], fontsize=16, rotation=0)
        c = res['L1']['own'][m]['vs_A']
        ax.set_title('%s\nvs A: %s(%s µs)' % (titles[m], CLS[c['final_class']], f1(c['worsening_best_us'], 2, True)),
                     fontsize=18)
        for k, vv in enumerate(v):
            ax.text(k, vv, f1(vv, 1), ha='center', va='bottom', fontsize=16, rotation=90, color=INK)
    axes[0].set_ylabel('L1 [µs](小さいほど人に近い、best)')
    fig.suptitle('保留の精度(10/13 の規則。B2 = 自分の付け替えのジャンル。A0〜A3 = A の種 0〜3(灰)、B = モデル B(濃い灰)、青 = B2)',
                 fontsize=19)
    fig.tight_layout()
    return _save(fig, 'eval_l1.png')


def fig_gates(res):
    """ゲートの合否(ジャンル 5・4)"""
    plt = _setup()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis('off')
    names = ['G3-R1-held', 'G3-R1-train', 'G3-R1-slow', 'G3-DE', 'G3-straight', 'G3-no-collapse', 'G3-L1', 'G3-intervention']
    vals = {g: res['gates'][g]['gates'] for g in ('5', '4')}
    val_txt = {
        'G3-R1-held': lambda g: 'D = %s µs' % f1(res['gate_detail'][g]['R1_held']['D'], 1, True),
        'G3-R1-train': lambda g: 'D = %s µs' % f1(res['gate_detail'][g]['R1_train']['D'], 1, True),
        'G3-R1-slow': lambda g: 'D = %s µs' % f1(res['gate_detail'][g]['R1_slow']['D'], 1, True),
        'G3-DE': lambda g: 'ステア %s・スロットル %s µs' % (f1(res['gate_detail'][g]['DE_steer']['D'], 1, True),
                                                    f1(res['gate_detail'][g]['DE_throttle']['D'], 1, True)),
        'G3-straight': lambda g: '低下 %s%%' % f1(100 * vals[g]['G3-straight']['drop'], 1),
        'G3-no-collapse': lambda g: '偏り %s µs' % f1(vals[g]['G3-no-collapse']['fwd_bias_mean'], 1, True),
        'G3-L1': lambda g: 'vs A %s(%s µs)' % (CLS[res['L1']['own']['primary']['vs_A']['final_class']],
                                               f1(res['L1']['own']['primary']['vs_A']['worsening_best_us'], 2, True)),
        'G3-intervention': lambda g: 'vs A %s' % CLS[res['L1']['own']['intervention_iv_all']['vs_A']['final_class']],
    }
    y0 = 0.88
    X = (0.36, 0.69)
    ax.text(0.0, 0.97, 'ゲート', fontsize=20, color=INK, weight='bold')
    for k, g in enumerate(('5', '4')):
        ax.text(X[k], 0.97, 'B2 をジャンル %s で' % g, fontsize=20, color=INK, weight='bold')
    for i, n in enumerate(names):
        y = y0 - i * 0.095
        ax.text(0.0, y, GATE_JA[n], fontsize=19, color=INK, va='center')
        for k, g in enumerate(('5', '4')):
            p = vals[g][n]['pass']
            ax.text(X[k], y + 0.012, ('✓ 合格' if p else '× 不合格'), fontsize=19, color=GOOD if p else BAD, va='center',
                    weight='bold')
            ax.text(X[k], y - 0.033, val_txt[n](g), fontsize=16, color=INK2, va='center')
    y = y0 - len(names) * 0.095 - 0.01
    for k, g in enumerate(('5', '4')):
        go = res['gates'][g]['desk_go']
        ax.text(X[k], y, 'desk_go(%s) = %s' % (g, '合格' if go else '不合格'), fontsize=20,
                color=GOOD if go else BAD, weight='bold', va='center')
    ax.text(0.0, y, '机上の結論', fontsize=20, color=INK, weight='bold', va='center')
    fig.suptitle('B2 の机上のゲート(すべて B2 の 2 本の種の平均。B はジャンル 5)', fontsize=22)
    return _save(fig, 'eval_gates.png')


def fig_training(res):
    """学習の曲線(検証の誤差)"""
    plt = _setup()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style = {'B': (GRAY, '-', 'モデル B'), 'A_s0': (LGRAY, '-', 'A 種 0'), 'A_s1': (LGRAY, '--', 'A 種 1'),
             'B2_s0': (BLUE, '-', 'B2 種 0'), 'B2_s1': (ORANGE, '-', 'B2 種 1')}
    for n, (c, ls, lab) in style.items():
        rj = RUNS / RUN[n] / 'result.json'
        if not rj.exists():
            continue
        r = json.loads(rj.read_text(encoding='utf-8'))
        h = r.get('history') or []
        ep = [x['epoch'] for x in h]
        v = [x['val_us'] for x in h]
        ax.plot(ep, v, color=c, ls=ls, lw=2.5, label='%s(最良 %s エポック目 %s µs)' % (lab, r.get('best_epoch'),
                                                                                f1(r.get('best_val_us'), 1)))
    ax.set_xlabel('エポック')
    ax.set_ylabel('検証の誤差 [µs]')
    ax.set_ylim(45, 70)
    ax.set_title('学習の検証の誤差(B2 は外す表を通した検証 1,476、B・A は 1,559 サンプル)', fontsize=20)
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    return _save(fig, 'train_val_curves.png')


def fig_engine(res):
    """エンジンの速さ(run_numpy・エンジンだけ・走行ループの infer_ms)"""
    lp = res['engine'].get('loop') or {}
    if not lp:
        return None
    plt = _setup()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = [n for n in ('B', 'B2_s0', 'B2_s1') if n in lp]
    items = [('run_numpy_median', 'run_numpy 中央値'), ('run_numpy_p99', 'run_numpy p99'), ('engine_median', 'エンジンだけ 中央値'),
             ('infer_ms_median', '走行ループの infer_ms 中央値')]
    x = np.arange(len(items))
    w = 0.35
    for k, n in enumerate(names):
        v = [lp[n].get(key) or 0 for key, _ in items]
        ax.bar(x + (k - (len(names) - 1) / 2) * w, v, w * 0.92, color=[GRAY, BLUE, ORANGE][k], edgecolor=SURF, linewidth=2,
               label=NAMES_JA[n])
        for xx, vv in zip(x + (k - (len(names) - 1) / 2) * w, v):
            ax.text(xx, vv + 0.08, f1(vv, 2), ha='center', va='bottom', fontsize=16, color=INK)
    ax.set_ylim(0, 7.2)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in items])
    ax.set_ylabel('ミリ秒')
    ax.set_title('FP16 エンジンの速さ(同じ時間帯に測定)。ゲート: run_numpy 中央値 ≤ 6.0 ms・p99 ≤ 8.0 ms', fontsize=20)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    return _save(fig, 'eval_engine.png')


def figures(res):
    out = []
    for fn in (fig_gates, fig_r1, fig_r1_passes, fig_l1, fig_training, fig_engine):
        p = fn(res)
        if p:
            out.append(str(p))
            print(p)
    return out


# ---------------------------------------------------------------- 文書

def _read(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def training_md(res=None):
    proto = _read(REPORT / 'protocol.json')
    exp = proto['model']['expected_train_log']
    rows = []
    R = {}
    for n in ('B2_s0', 'B2_s1', 'B', 'A_s0', 'A_s1', 'DROP_s0', 'RELABEL_s0'):
        r = _read(RUNS / RUN[n] / 'result.json')
        if r:
            R[n] = r
    lines = ['# 学習の記録 — B2(b2_curation_20261016)', '',
             '- **依頼**(ユーザー、10/16 18:00ごろ):「今週は走行できないので、ほかにできることは？」。選ばれた「4 データを選び直して B2 を学習」の、GPU の学習の段です。',
             '- 固定した `protocol.json` の `model.train_cmd` の文字列のまま、`experiments/20261016_b2_curation/train_b2.py run 0 1` で学習しました(jf-heavy --lock big で 1 つずつ)。',
             '- 実車・PCA9685・I2C、`jetformer/drive.py`・走行の既定には触れていません。生データと既存のデータセットも書き換えていません。git commit はしていません。',
             '- 学習結果の名前は protocol で固定した **`b2cur_20261016-s0`・`b2cur_20261016-s1`** です(この段の指示文の `b2_20261016-s<seed>` ではなく、固定した名前を使いました。中身は同じです)。',
             '', '## 1. 結果', '',
             '| | B2 種 0 | B2 種 1 | モデル B(比べる相手) |', '|---|---|---|---|']

    def g(n, k, fmt=lambda v: str(v)):
        v = (R.get(n) or {}).get(k)
        return fmt(v) if v is not None else '—'
    for lab, k, fm in (('状態', 'state', str), ('エポック', 'epochs_done', str), ('学習サンプル', 'train_samples', lambda v: '{:,}'.format(v)),
                       ('検証サンプル', 'val_samples', lambda v: '{:,}'.format(v)),
                       ('ジャンル別の学習サンプル(4 / 5)', 'train_mode_counts', lambda v: '{:,} / {:,}'.format(v.get('4'), v.get('5'))),
                       ('衝突の前後として外したフレーム(学習 / 検証)', 'impact_masked_frames',
                        lambda v: '{:,} / {:,}'.format(v['train'], v['val'])),
                       ('最良のエポック', 'best_epoch', str), ('最良の検証の誤差', 'best_val_us', lambda v: '%.1f µs' % v),
                       ('ジャンル別の検証の誤差(4 / 5)', 'best_val_per_mode_us', lambda v: '%.1f / %.1f µs' % (v.get('4'), v.get('5'))),
                       ('学習の時間', 'duration_s', lambda v: '%.0f 分' % (v / 60) if v > 600 else '—(再開の分だけ記録)'),
                       ('GPU のメモリの最大', 'peak_mem_mb', lambda v: '{:,.0f} MB'.format(v))):
        lines.append('| %s | %s | %s | %s |' % (lab, g('B2_s0', k, fm), g('B2_s1', k, fm), g('B', k, fm)))
    lines += ['', '- B2 の検証の誤差は、外す表を通した検証(1,476 サンプル)なので、B の値(1,559 サンプル)と直接は比べられません。'
              '精度の比較は、外さない保留の 4,670 フレームで行いました(`evaluation.md` の 5 章)。',
              '- B の学習の時間 200 秒は、電池切れの後の再開の分だけの値です(result.json の note)。', '']
    lines += ['## 2. 学習のログの最初の 4 行(protocol の expected_train_log と比べた)', '',
              '| 行 | protocol | 種 0 | 種 1 |', '|---|---|---|---|']
    for n in ('B2_s0', 'B2_s1'):
        pass
    for lab, want, k in (('衝突の前後として外したフレーム', '%d / %d' % (exp['impact_masked_printed']['train'], exp['impact_masked_printed']['val']),
                          lambda r: '%d / %d' % (r['impact_masked_frames']['train'], r['impact_masked_frames']['val'])),
                         ('drop で外した', '%d / %d' % (exp['dropped_printed']['train'], exp['dropped_printed']['val']),
                          lambda r: '%d / %d' % (r['rl_options']['dropped_frames']['train'], r['rl_options']['dropped_frames']['val'])),
                         ('学習 / 検証のサンプル', '%d / %d' % (exp['train_samples'], exp['val_samples']),
                          lambda r: '%d / %d' % (r['train_samples'], r['val_samples'])),
                         ('ジャンル別(4 / 5)', '%d / %d' % (exp['genre_counts_train']['4'], exp['genre_counts_train']['5']),
                          lambda r: '%d / %d' % (r['train_mode_counts']['4'], r['train_mode_counts']['5']))):
        lines.append('| %s | %s | %s | %s |' % (lab, want, k(R['B2_s0']) if 'B2_s0' in R else '—',
                                                k(R['B2_s1']) if 'B2_s1' in R else '—'))
    st = _read(REPORT / 'logs' / 'train_state.json') or {}
    lines += ['', '- 両方の種で 4 行とも protocol と同じでした。表の sha1(drop_b2 `%s`・remap_b2 `%s`)も、result.json と今のファイルで同じです。'
              % ((R.get('B2_s0') or {}).get('rl_options', {}).get('sha1', {}).get('drop_mask'),
                 (R.get('B2_s0') or {}).get('rl_options', {}).get('sha1', {}).get('mode_remap_npz')),
              '- 道具の確かめ(`train_b2.py` の check_result): 種 0 = %s、種 1 = %s。' % (
                  (st.get('0') or {}).get('result_check'), (st.get('1') or {}).get('result_check')), '']
    lines += ['## 3. 途中で起きたこと', '']
    dev = (REPORT / 'deviations.md').read_text(encoding='utf-8')
    mem_lines = [ln for ln in dev.splitlines() if 'train_b2.py' in ln]
    if mem_lines:
        lines.append('- メモリの上限: 種 0 は、1 エポック目の前に上限 3G で止まりました(10/15 の v0-s2 と同じ)。protocol の on_cgroup_kill のとおり、'
                     '上限を 3872M にして最初から学習し直しました(計算は変わりません)。`deviations.md` に書きました。')
        if any('s1' in ln for ln in mem_lines):
            lines.append('- 種 1 も同じく上限を上げました。')
        else:
            lines.append('- 種 1 は上限 3G のまま最後まで学習できました。')
    lines += ['- 電源断はありませんでした。', '']
    lines += ['## 4. 学習の曲線', '', '![学習の曲線](fig/train_val_curves.png)', '',
              '- 最良のエポック: B2 種 0 = %s、種 1 = %s(B = %s)。B2 は学習サンプルが 5.7%% 少なく、ジャンル 4 が多い構成です。' % (
                  g('B2_s0', 'best_epoch'), g('B2_s1', 'best_epoch'), g('B', 'best_epoch')),
              '- 学習の誤差(train)は最後まで下がり、検証の誤差は 15〜20 エポック目からほぼ横ばいです(A・B と同じ形)。', '']
    if 'DROP_s0' in R or 'RELABEL_s0' in R:
        lines += ['## 5. 参考の腕(判定しない)', '']
        for n in ('DROP_s0', 'RELABEL_s0'):
            r = R.get(n)
            if r:
                lines.append('- %s(`%s`): 学習 %s・検証 %s サンプル、ジャンル別 %s、最良 %s エポック目 %.1f µs' % (
                    NAMES_JA[n], RUN[n], '{:,}'.format(r['train_samples']), '{:,}'.format(r['val_samples']),
                    r['train_mode_counts'], r['best_epoch'], r['best_val_us']))
        lines.append('')
    lines += ['## %d. ファイル' % (6 if ('DROP_s0' in R or 'RELABEL_s0' in R) else 5), '',
              '| ファイル | 中身 |', '|---|---|',
              '| `/home/jetson/jetformer/data/runs/b2cur_20261016-s0/`・`-s1/` | best.pth・last.pth・jetformer.onnx・model_config.json・result.json・run.json・train.log |',
              '| `/home/jetson/jetformer/experiments/20261016_b2_curation/train_b2.py` | 学習の段の道具(待つ・run.json・最初の 4 行の確かめ・再開・メモリの上限の引き上げ) |',
              '| `logs/train_driver.log`・`logs/train_state.json` | 学習の道具のログと状態 |', '']
    text = '\n'.join(lines)
    (REPORT / 'training.md').write_text(text, encoding='utf-8')
    return text


def _gate_row(res, gname, g):
    G = res['gates'][g]['gates'][gname]
    d = res['gate_detail'][g]
    if gname == 'G3-R1-held':
        s = d['R1_held']
        v = 'D = %s µs(種 0 %s・種 1 %s、区間 %s〜%s、D − max D_A = %s、last %s)' % (
            f1(s['D'], 1, True), f1(s['D_seeds']['0'], 1, True), f1(s['D_seeds']['1'], 1, True), f1(s['ci95'][0], 1, True),
            f1(s['ci95'][1], 1, True), f1(G['D_minus_maxA'], 1, True), f1(s.get('D_last'), 1, True))
    elif gname == 'G3-R1-train':
        s = d['R1_train']
        v = 'D = %s µs(種 0 %s・種 1 %s、D − max D_A = %s)' % (f1(s['D'], 1, True), f1(s['D_seeds']['0'], 1, True),
                                                           f1(s['D_seeds']['1'], 1, True), f1(G['D_minus_maxA'], 1, True))
    elif gname == 'G3-R1-slow':
        v = 'D = %s µs' % f1(d['R1_slow']['D'], 1, True)
    elif gname == 'G3-DE':
        v = 'ステア %s(下限 %s)・スロットル %s(下限 %s)µs' % (f1(d['DE_steer']['D'], 1, True), f1(G['threshold_steer_us'], 1),
                                                       f1(d['DE_throttle']['D'], 1, True), f1(G['threshold_throttle_us'], 1))
    elif gname == 'G3-straight':
        v = '前進の強さ B2 %s・B %s µs → 低下 %s%%' % (f1(G['fwd_arm_mean'], 1), f1(G['fwd_ref'], 1), f1(100 * G['drop'], 2))
    elif gname == 'G3-no-collapse':
        v = 'fwd_bias %s µs・向きの正解率 %s(B %s)' % (f1(G['fwd_bias_mean'], 1, True), f1(G['sign_acc_mean'], 3),
                                                f1(G['sign_acc_ref'], 3))
    elif gname == 'G3-L1':
        p, r = res['L1']['own']['primary']['vs_A'], res['L1']['own']['rest']['vs_A']
        v = '主指標 vs A %s(%s µs)・rest %s(%s µs)' % (CLS[p['final_class']], f1(p['worsening_best_us'], 2, True),
                                                   CLS[r['final_class']], f1(r['worsening_best_us'], 2, True))
    else:
        c = res['L1']['own']['intervention_iv_all']['vs_A']
        v = 'vs A %s(%s µs)' % (CLS[c['final_class']], f1(c['worsening_best_us'], 2, True))
    return v


def reading_lines(res):
    """0.1 読み(結果を見た後の記述。判定は変えない)"""
    d5, d4 = res['gate_detail']['5'], res['gate_detail']['4']
    sec = res['secondary']
    td = sec['T_dropped_vs_kept_fast']
    da = d5['R1_held']['D_A']
    st5 = res['gates']['5']['gates']['G3-straight']
    st4 = res['gates']['4']['gates']['G3-straight']
    l1p = res['L1']['own']['primary']['vs_A']
    ps = sec['per_scene_L1_own_best'].get('20261010-152707-0001') or {}
    g44 = sec['genres_HAI_fast'].get('B2_g4_vs_B_g4') or {}
    L = ['### 0.1 読み(結果を見た後の記述。判定は変えていません)', '',
         '1. **選別の効きは、外した通過の上だけに小さく出ました。** 学習のログで B2 が外した速い 11 通過では、B2 は B より %s µs 遅く出しました'
         '(区間 %s〜%s。A の種 0〜3 は %s µs)。B2 も学んだ速い 13 通過では %s µs でした。'
         % (f1(td['dropped']['D'], 1, True), f1(td['dropped']['ci95'][0], 1, True), f1(td['dropped']['ci95'][1], 1, True),
            '・'.join(f1(v, 1, True) for v in td['dropped']['D_A'].values()), f1(td['kept']['D'], 1, True)),
         '2. **保留の速い通過には広がりませんでした。** 保留の 25 通過(ジャンル 5)で %s µs です。いちばん速い AI の進入(8,000 eRPM 以上の 4 通過)'
         'では B2 が平均 +21 µs 遅く出しましたが、A の種も +3〜+18 µs 遅く出していて(B がそこで強く出しすぎているだけ)、選別の効きとは言えません。'
         '5,500〜7,000 eRPM の 20 通過では平均 −5.9 µs で、B とほぼ同じか少し強めでした(`fig/eval_r1_passes.png`)。'
         % f1(d5['R1_held']['D'], 1, True),
         '3. **種を替えただけのずれのほうが大きいです。** A の種 0〜3 の B とのずれは %s µs で、種 2 は %s µs でした。今の机上の物差しでは、'
         '±20 µs くらいまでは種のばらつきに埋もれます。'
         % ('・'.join(f1(v, 1, True) for v in da.values()), f1(da.get('A_s2'), 1, True)),
         '4. **ジャンルの付け替えは、R1 だけでなく走り全体の速さを動かしました。** ジャンル 5 は遅いログだけになり、直線の前進が %s%% 弱まり'
         '(直線のゲートで不合格)、DE のスロットルも %s µs 遅くなりました。ジャンル 4 は R1 の手前で一様に約 +10 µs 遅い(速い %s・遅い %s µs)一方、'
         '直線では %s%% 強く、R1 に限った効きではありません。B もジャンル 4 にそろえて比べると、B2 は %s µs(少し強め)でした。'
         % (f1(100 * st5['drop'], 1), f1(d5['DE_throttle']['D'], 1, True), f1(d4['R1_held']['D'], 1, True),
            f1(d4['R1_slow']['D'], 1, True), f1(-100 * st4['drop'], 1), f1(g44.get('D'), 1, True)),
         '5. **精度は落ちていません。** 保留の L1 は A と同等(%s µs)、介入の場面も同等、崩れもありません。外した窓は全体の 5%% なので、'
         '平均の精度にはほとんど出ません。付け替えた 152707-0001 では B2 の L1 が B より大きい(B %s・B2 %s / %s µs)のが目立つ点です。'
         % (f1(l1p['worsening_best_us'], 2, True), f1(ps.get('B'), 1), f1(ps.get('B2_s0'), 1), f1(ps.get('B2_s1'), 1)),
         '6. **結論**: 「外すだけ」の選別(学習サンプルの 5.7%)では、R1 の手前の癖は机上で測れるほど変わりませんでした。B のままにし、'
         'protocol の no-go の次の手の (c) 安全層 v1 の速度上限(シミュレーターで試験中)を先に進めるのが筋です。(a) 新しい「遅く入って速く出る」データ、'
         '(b) 重みの調整は、種を 4 本以上にして測る必要があります。', '']
    return L


def evaluation_md(res):
    proto = _read(REPORT / 'protocol.json')
    g5, g4 = res['gates']['5'], res['gates']['4']
    gd = res['drive_genre']
    eng = res.get('engine') or {}
    car = eng.get('car_run')
    L = []
    L += ['# 机上の評価 — B2(b2_curation_20261016)', '',
          '- **依頼**(ユーザー、10/16 18:00ごろ):「今週は走行できないので、ほかにできることは？」。選ばれた「4 データを選び直して B2 を学習」の、机上の評価です。',
          '- 判定は、学習の前に固定した `protocol.json`(sha256 `%s…`)のゲートのとおりです。固定の後に変えたことは `deviations.md` にあります。' % res['protocol_sha256'][:12],
          '- 実車は走らせていません。PCA9685・I2C には触れていません(走行ループの時間の確認は、偽の PCA9685 で行いました)。`drive.py`・走行の既定は変えていません。', '']
    L += ['## 0. 結論', '']
    if gd is not None:
        L.append('- **desk_go(%d) に合格しました。** 来週の実車のカード(B → B2 → B)に、B2 をジャンル **%d** で載せられます。' % (gd, gd))
    else:
        L.append('- **desk_go は、ジャンル 5 でも 4 でも不合格でした(no-go)。** B2 は実車に出しません。モデル B のままにします。')
    L.append('- ジャンル 5: %s(落ちたゲート: %s)。ジャンル 4: %s(落ちたゲート: %s)。' % (
        '合格' if g5['desk_go'] else '不合格', '・'.join(GATE_JA[x] for x in g5['failed']) or 'なし',
        '合格' if g4['desk_go'] else '不合格', '・'.join(GATE_JA[x] for x in g4['failed']) or 'なし'))
    s5, s4 = res['gate_detail']['5']['R1_held'], res['gate_detail']['4']['R1_held']
    L.append('- いちばん大事な問い「R1 の手前で、B2 のスロットルは B より弱まるか」: 保留の速く近づいた 25 通過で、'
             'ジャンル 5 は **%s µs**、ジャンル 4 は **%s µs** でした(合格線 +30 µs、+ = 遅い)。A の種 0〜3(選別なし)のずれは %s µs でした。' % (
                 f1(s5['D'], 1, True), f1(s4['D'], 1, True),
                 '・'.join(f1(v, 1, True) for v in s5['D_A'].values())))
    if eng.get('gate'):
        e0 = (eng['gate'].get('B2_s0') or {})
        L.append('- エンジン(B2 種 0、FP16): %s。run_numpy 中央値 %s ms(B %s ms)、FP32 との差の最大 %s µs。' % (
            '合格' if e0.get('pass') else '不合格',
            f1(((eng['checks'].get('B2_s0') or {}).get('latency_ms') or {}).get('run_numpy_median'), 2),
            f1(eng.get('B_run_numpy_same_session_ms'), 2),
            f1(((eng['checks'].get('B2_s0') or {}).get('accuracy') or {}).get('actions_max_us'), 2)))
    L.append('- 机上の結論は、実車の改善を示しません(10/12 の規則 7)。最終の判断は実車のカード 3(B → B2 → B)です。')
    L += ['', '![ゲートの一覧](fig/eval_gates.png)', '']
    L += reading_lines(res)
    L += ['## 1. 確かめたこと(評価の前)', '',
          '| 確かめ | 結果 |', '|---|---|']
    cc = res.get('crosscheck') or {}
    rp_ = res.get('repro') or {}
    ti = res['tables_info']
    L.append('| 既にある A・B の予測で、固定した基準値(L1 の 5 本 × 4 集合 × best・last、陰性対照の窓の値)を作り直す | %s(L1 の差の最大 %s µs、陰性対照の違い %d 件) |' % (
        '合格' if rp_.get('pass') else '不合格', f1(rp_.get('L_max_abs_diff_rounded'), 4), len(rp_.get('negative_control_diffs') or [])))
    if cc:
        L.append('| crosscheck: B の best を新しい道具で予測し直し、10/12 の B.npz と比べる | %s(全部の行: 最大 %s µs。part 0 だけ: 最大 %s µs、差は最後の半端なバッチ(%s 行)の中だけ)|' % (
            '合格' if cc.get('pass') else '不合格', f1((cc.get('full') or {}).get('max_abs_us'), 4),
            f1((cc.get('part0') or {}).get('max_abs_us'), 4), (cc.get('part0') or {}).get('last_batch_size')))
    L.append('| 窓の表の数(protocol の counts_expected) | 同じ(R1: H 12・AI 56・T 61 通過、DE: H 10・AI 55・T 55、直線 %d 行)。窓の表 %s 行・key `%s` |' % (
        ti['straight_frames'], '{:,}'.format(ti['window_rows']), ti['window_key']))
    L.append('| 予測するジャンル | 保留の表の B2 は own_labels(152405-0002・152707-0001 の %s 行 = 4)。窓の表はジャンル 5・4 を全部の行に |' % '{:,}'.format(ti['own_label_rows']))
    L.append('| 道具を固定した後に変えたか(eval_sha256.txt) | %s |' % ('変えていない' if not res['tools_changed_after_freeze'] else '変えた: %s' % res['tools_changed_after_freeze']))
    L.append('| 中心・中立(model_config) | 全部のモデルで 1441 / 1496 µs%s |' % ('' if not res.get('bad_refs') else '(違うもの: %s)' % res['bad_refs']))
    L += ['', '- crosscheck は、protocol の字義(part 0 だけ)では 0.080 µs の差で合いませんでした。原因と扱いは `deviations.md` にあります(判定への影響はありません)。', '']
    L += ['## 2. ゲート(B2 は 2 本の種の平均。B はジャンル 5、A の種 0〜3 もジャンル 5)', '',
          '| ゲート | 合格の条件 | ジャンル 5 | ジャンル 4 |', '|---|---|---|---|']
    cond = {'G3-R1-held': 'D ≥ +30・両方の種 ≥ +20・区間の下限 > 0・D − max D_A ≥ +15・last ≥ +20',
            'G3-R1-train': 'D ≥ +30・両方の種 ≥ +20・D − max D_A ≥ +15', 'G3-R1-slow': 'D ≥ −10',
            'G3-DE': 'ステア・スロットルとも D ≥ min(−15, min D_A − 5)', 'G3-straight': '低下 ≤ 5%',
            'G3-no-collapse': 'fwd_bias ≥ −30・正解率 ≥ B − 0.01', 'G3-L1': 'NOT(rest が悪い かつ > 2.798)・主指標の悪化 < 10',
            'G3-intervention': 'NOT(悪い かつ > 5.822)'}
    for n in ('G3-R1-held', 'G3-R1-train', 'G3-R1-slow', 'G3-DE', 'G3-straight', 'G3-no-collapse', 'G3-L1', 'G3-intervention'):
        L.append('| %s | %s | %s: %s | %s: %s |' % (GATE_JA[n], cond[n], ok(g5['gates'][n]['pass']), _gate_row(res, n, '5'),
                                                ok(g4['gates'][n]['pass']), _gate_row(res, n, '4')))
    L.append('| **desk_go** | 全部合格 | %s | %s |' % (ok(g5['desk_go']), ok(g4['desk_go'])))
    L += ['', '- 走行のジャンルの決め方(固定): ジャンル 5 で desk_go ならジャンル 5、だめでジャンル 4 で desk_go ならジャンル 4、どちらもだめなら no-go。',
          '- 結果: **%s**。' % ('ジャンル %d' % gd if gd is not None else 'no-go'), '']
    # 部品の詳細
    PJ = {'D_ge_30': 'D ≥ +30', 'each_seed_ge_20': '両方の種 ≥ +20', 'ci_low_gt_0': '区間の下限 > 0',
          'minus_maxA_ge_15': 'D − max D_A ≥ +15', 'last_ge_20': 'last ≥ +20', 'D_ge_minus10': 'D ≥ −10',
          'steer_ok': 'ステア', 'throttle_ok': 'スロットル'}
    L += ['### 2.1 R1・DE のゲートの部品(○ = 満たした、× = 満たさない)', '', '| ゲート | ジャンル 5 | ジャンル 4 |', '|---|---|---|']
    for n in ('G3-R1-held', 'G3-R1-train', 'G3-R1-slow', 'G3-DE'):
        cells = []
        for g in ('5', '4'):
            parts = res['gates'][g]['gates'][n]['parts']
            cells.append('・'.join('%s %s' % (PJ.get(k, k), '○' if v else '×') for k, v in parts.items()))
        L.append('| %s | %s | %s |' % (GATE_JA[n], cells[0], cells[1]))
    L += ['', '![R1 の手前のスロットル](fig/eval_r1_throttle.png)', '', '![通過ごとの差](fig/eval_r1_passes.png)', '']
    # 集合ごとの値
    L += ['### 2.2 集合ごとの値(µs、+ = B より遅い / 左)', '',
          '| 集合 | 通過 | B の平均の出力 | D(B2 g=5) | D(B2 g=4) | A 種 0 | A 種 1 | A 種 2 | A 種 3 |', '|---|---|---|---|---|---|---|---|---|']
    for key, lab in (('R1_held', 'R1 保留・速い(スロットル)'), ('R1_train', 'R1 学習のログ・速い'), ('R1_slow', 'R1 保留・遅い'),
                     ('DE_steer', 'DE(ステア)'), ('DE_throttle', 'DE(スロットル)')):
        a, b = res['gate_detail']['5'][key], res['gate_detail']['4'][key]
        L.append('| %s | %d | %s | %s | %s | %s |' % (lab, a['n'], f1(a['u_B_mean'], 1), f1(a['D'], 1, True), f1(b['D'], 1, True),
                                                      ' | '.join(f1(a['D_A'].get(x), 1, True) for x in ('A_s0', 'A_s1', 'A_s2', 'A_s3'))))
    L.append('')
    # L1
    L += ['## 3. 保留の精度(L1、10/13 の規則)', '', '![保留の精度](fig/eval_l1.png)', '',
          '| 集合 | ジャンル | vs A(best の悪化・分類) | vs B(best の悪化・分類) | 種の帯(5 本の範囲 × 2) |', '|---|---|---|---|---|']
    for lab in ('own', 'aligned'):
        for m, mj in (('primary', '主指標 4,670'), ('rest', '選別の窓の外 4,424'), ('curation_windows', '選別の窓の中 246'),
                      ('intervention_iv_all', '介入 1,570')):
            a, b = res['L1'][lab][m]['vs_A'], res['L1'][lab][m]['vs_B']
            L.append('| %s | %s | %s µs・%s(閾値 %s) | %s µs・%s | %s µs |' % (
                mj, 'own_labels(主)' if lab == 'own' else 'aligned(参考)', f1(a['worsening_best_us'], 2, True),
                CLS[a['final_class']], f1(a['best']['threshold_us'], 3), f1(b['worsening_best_us'], 2, True),
                CLS[b['final_class']], f1(a['fresh_band_best_us'], 3)))
    V = res['L1_values']['own']['best']
    L += ['', 'L1(best、own_labels)の値 [µs]:', '', '| | ' + ' | '.join(NAMES_JA[n] for n in ('A_s0', 'A_s1', 'A_s2', 'A_s3', 'B', 'B2_s0', 'B2_s1')) + ' |',
          '|---|' + '---|' * 7]
    for m, mj in (('primary', '主指標'), ('rest', '選別の窓の外'), ('curation_windows', '選別の窓の中'), ('intervention_iv_all', '介入')):
        L.append('| %s | %s |' % (mj, ' | '.join(f1(V[n][m], 2) for n in ('A_s0', 'A_s1', 'A_s2', 'A_s3', 'B', 'B2_s0', 'B2_s1'))))
    L += ['', '- 読み(先に決めた読み方): **%s**' % res['gates']['5']['gates']['G3-L1']['reading'],
          '- 感度(参考): vs B で同じゲートを見ると、L1 %s・介入 %s。' % (
              '合格' if res['sensitivity_L1_vs_B']['gate']['pass'] else '不合格',
              '合格' if res['sensitivity_L1_vs_B']['intervention']['pass'] else '不合格'), '']
    ib = res.get('intervention_by_class') or {}
    if ib:
        L += ['介入の場面の種類別の L1(best、判定には使わない)[µs]:', '', '| 種類 | フレーム | A 種 0 | A 種 1 | B | B2 種 0 | B2 種 1 |', '|---|---|---|---|---|---|---|']
        for k, v in ib.items():
            L.append('| %s | %d | %s |' % (k, v['frames'], ' | '.join(f1(v.get(n), 1) for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1'))))
        L.append('')
    # 崩れ
    nm = res['no_collapse_metrics']
    L += ['## 4. 崩れなし・直線', '',
          '| | ' + ' | '.join(NAMES_JA[n] for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1')) + ' |', '|---|' + '---|' * 5,
          '| fwd_bias(3 フレーム先の予測 − 人)[µs] | %s |' % ' | '.join(f1(nm[n]['fwd_bias_exact'], 1, True) for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1')),
          '| 向きの正解率 | %s |' % ' | '.join(f1(nm[n]['steer_sign_acc'], 3) for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1')), '']
    sec = res['secondary']
    for g in ('5', '4'):
        G = res['gates'][g]['gates']['G3-straight']
        L.append('- 直線 542 行(ジャンル %s): 前進の強さ B2 %s µs(種 0 %s・種 1 %s)、B %s µs → 低下 %s%%(≤ 5%% で合格)。SW・HS の直線 %d 行(計画 G3 の字義、参考): 低下 %s%%。' % (
            g, f1(G['fwd_arm_mean'], 1), f1(G['fwd_arm_seeds'][0], 1), f1(G['fwd_arm_seeds'][1], 1), f1(G['fwd_ref'], 1),
            f1(100 * G['drop'], 2), sec['straight_sw_hs']['g' + g]['frames'], f1(100 * sec['straight_sw_hs']['g' + g]['drop'], 2)))
    L.append('- 参考: A の種 0〜3 の直線の低下(ジャンル 5、B に対して): %s%%。' % '・'.join(
        f1(100 * v, 2) for v in sec['straight_542_A'].values()))
    L.append('')
    # エンジン
    L += ['## 5. エンジンとタイミング(B2 種 0、FP16)', '']
    if eng.get('gate'):
        L += ['![エンジンの速さ](fig/eval_engine.png)', '', '| 項目 | B2 種 0 | B2 種 1 | モデル B |', '|---|---|---|---|']
        ch = eng['checks']

        def ev_(n, f):
            try:
                return f(ch[n])
            except (KeyError, TypeError):
                return '—'
        for lab, f in (('FP16 と FP32 の差の最大 [µs]', lambda c: f1(c['accuracy']['actions_max_us'], 3)),
                       ('NaN・Inf なし', lambda c: 'はい' if c['accuracy']['finite'] else 'いいえ'),
                       ('run_numpy 中央値 / p99 [ms]', lambda c: '%s / %s' % (f1(c['latency_ms']['run_numpy_median'], 3),
                                                                           f1(c['latency_ms']['run_numpy_p99'], 3))),
                       ('エンジンだけ 中央値 [ms]', lambda c: f1(c['latency_ms']['engine_median'], 3)),
                       ('学習の GPU のメモリの最大 [MB]', lambda c: f1(c['peak_mem_mb'], 0)),
                       ('走行の一覧(drive.list_models)', lambda c: 'ある' if c['listed'] else 'ない'),
                       ('trained_modes', lambda c: str(c['model_info']['trained_modes'])),
                       ('中心・中立 [µs]', lambda c: '%s / %s' % (f1(c['model_info']['steer_center_us'], 0),
                                                              f1(c['model_info']['throttle_neutral_us'], 0))),
                       ('エンジンの大きさ [MB]', lambda c: f1(c['engine_mb'], 1))):
            L.append('| %s | %s | %s | %s |' % (lab, ev_('B2_s0', f), ev_('B2_s1', f), ev_('B', f)))
        lp = eng.get('loop') or {}
        L.append('| 走行ループ(偽の PCA9685)の infer_ms 中央値 [ms] | %s | %s | %s |' % tuple(
            f1((lp.get(n) or {}).get('infer_ms_median'), 3) for n in ('B2_s0', 'B2_s1', 'B')))
        L.append('| 走行ループの HOLD・エラー | %s | %s | %s |' % tuple(
            ('%s・%s' % ((lp.get(n) or {}).get('holds'), (lp.get(n) or {}).get('error') or 'なし')) if n in lp else '—'
            for n in ('B2_s0', 'B2_s1', 'B')))
        L.append('| ループの出力 = 同じ入力で直接 run_numpy(差の最大 µs) | %s | %s | %s |' % tuple(
            f1((lp.get(n) or {}).get('loop_vs_direct_max_us'), 4) for n in ('B2_s0', 'B2_s1', 'B')))
        L.append('| 走行ループ・カメラ → PWM の見積もり [ms] | %s | %s | %s |' % tuple(
            ('%s・%s' % (f1((lp.get(n) or {}).get('loop_estimate_ms'), 2), f1((lp.get(n) or {}).get('cam_to_pwm_estimate_ms'), 1)))
            if n in lp else '—' for n in ('B2_s0', 'B2_s1', 'B')))
        L += ['', '- ゲート(固定): FP32 との差 ≤ 2.5 µs、run_numpy 中央値 ≤ 6.0 ms・p99 ≤ 8.0 ms、学習の GPU のメモリ ≤ 4,096 MB、NaN・Inf なし、'
              '中央値 ≤ B(2.377 ms、固定の値)+ 0.5 ms、走行の一覧に出てジャンル g を学習している、中心・中立が B と同じ。',
              '- 結果: B2 種 0 = %s。部品: %s。' % (ok((eng['gate'].get('B2_s0') or {}).get('pass')),
                                             '・'.join('%s %s' % (EJ.get(k, k), '○' if v else '×') for k, v in ((eng['gate'].get('B2_s0') or {}).get('parts') or {}).items())),
              '- 同じ時間帯に測った B の run_numpy 中央値(%s ms)と比べても、B2 は +0.5 ms の内でした: %s。' % (
                  f1(eng.get('B_run_numpy_same_session_ms'), 3),
                  '○' if ((eng['gate'].get('B2_s0') or {}).get('vs_B_same_session') or {}).get('parts', {}).get('median_le_B_plus_0p5') else '×'),
              '- 走行の予算: %s。B2 の見積もりは走行ループ %s ms・カメラ → PWM %s ms で、1 フレームの 33.3 ms に十分収まります。' % (
                  eng.get('budget_note', ''), f1(((eng.get('loop') or {}).get('B2_s0') or {}).get('loop_estimate_ms'), 2),
                  f1(((eng.get('loop') or {}).get('B2_s0') or {}).get('cam_to_pwm_estimate_ms'), 1)),
              '- 偽の PCA9685 の走行ループ(記録したフレームを 30 fps で流す)の fps・遅れは、B と B2 で同じでした(机上の流し方による値で、実車の値ではありません)。',
              '- 実車の dry run(車輪を浮かせて 60 秒)での確かめは、カードの手順 0 に残します(HOLD 0・infer_ms 中央値 ≤ B + 1 ms・カメラ 29 fps 以上・遅れ中央値 ≤ 35 ms)。', '']
    else:
        L += ['- まだ作っていません。', '']
    # 参考
    L += ['## 6. 参考の値(判定しない)', '', '走行のジャンル %s で計算(no-go のときはジャンル 5)。' % sec['genre_for_reference'], '',
          '| 集合 | 通過 | D [µs] | 種 0 / 種 1 | A の種 0〜3 |', '|---|---|---|---|---|']

    def srow(lab, s):
        if not s:
            return
        L.append('| %s | %d | %s(区間 %s〜%s) | %s / %s | %s |' % (
            lab, s['n'], f1(s['D'], 1, True), f1(s['ci95'][0], 1, True), f1(s['ci95'][1], 1, True), f1(s['D_seeds']['0'], 1, True),
            f1(s['D_seeds'].get('1'), 1, True), '・'.join(f1(v, 1, True) for v in (s.get('D_A') or {}).values())))
    srow('全部の通過・保留(H ∪ AI、計画 G3 の字義)', sec['all_passes']['HAI_r1'])
    srow('全部の通過・学習のログ(T)', sec['all_passes']['T_r1'])
    srow('6200 以上だけ・保留', sec['ge6200']['HAI'])
    srow('6200 以上だけ・学習のログ', sec['ge6200']['T'])
    srow('速い・保留の人(H)だけ', sec['H_vs_AI_fast']['H'])
    srow('速い・保留の AI だけ', sec['H_vs_AI_fast']['AI'])
    srow('速い・学習のログ・B2 が外した通過', sec['T_dropped_vs_kept_fast']['dropped'])
    srow('速い・学習のログ・B2 も学んだ通過', sec['T_dropped_vs_kept_fast']['kept'])
    srow('窓 0.4 秒(10/12)・保留の速い', sec['windows']['w04_HAI_fast'])
    srow('窓 2 秒・保留の速い', sec['windows']['w2_HAI_fast'])
    srow('窓 0.4 秒・学習のログの速い', sec['windows']['w04_T_fast'])
    srow('窓 2 秒・学習のログの速い', sec['windows']['w2_T_fast'])
    srow('last.pth の B2(B は best)・保留の速い', sec['last_HAI_fast'])
    for k, lab in (('B2_aligned_vs_B_aligned', 'aligned のジャンル(全部のモデル)'), ('B2_own_vs_B_aligned', 'B2 = own_labels・B = aligned'),
                   ('B2_g4_vs_B_g5', 'B2 = 4・B = 5'), ('B2_g5_vs_B_g5', 'B2 = 5・B = 5'), ('B2_g4_vs_B_g4', 'B2 = 4・B = 4')):
        srow('ジャンル: %s・保留の速い' % lab, sec['genres_HAI_fast'].get(k))
    L.append('')
    est = sec['entry_estimate']
    L += ['見積もった進入 eRPM(rl_evaluate.entry_estimate と同じ式: 実際の進入 + 定常の対応表での差。閉ループの影響は入らない粗い見積もり):', '',
          '| 集合 | 通過 | 実際 ≥ 5500 / ≥ 6200 | B ≥ 5500 / ≥ 6200(中央値) | B2 種 0 | B2 種 1 |', '|---|---|---|---|---|---|']
    for k, v in est.items():
        L.append('| %s | %d | %s%% / %s%% | %s%% / %s%%(%s) | %s%% / %s%%(%s) | %s%% / %s%%(%s) |' % (
            '保留(H ∪ AI)' if k == 'HAI' else '学習のログ(T)', v['passes'], f1(100 * v['actual_share_ge5500'], 0),
            f1(100 * v['actual_share_ge6200'], 0),
            *[x for n in ('B', 'B2_s0', 'B2_s1') for x in (f1(100 * v[n]['share_ge5500'], 0), f1(100 * v[n]['share_ge6200'], 0),
                                                         f1(v[n]['median'], 0))]))
    L.append('')
    dh = sec['de_vs_human']
    L += ['DE の出口の予測ステアと人の差(計画 G3「人の −20 µs に近づく」。B2 は DE を直す手ではないので判定しない):', '']
    for k, v in dh.items():
        L.append('- %s(%d 通過): 人の平均 %s µs。B − 人 %s・B2 種 0 − 人 %s・B2 種 1 − 人 %s µs。' % (
            '保留の人' if k == 'H' else '学習のログの人', v['passes'], f1(v['human_exec_steer_mean'], 1), f1(v['B_minus_human'], 1, True),
            f1(v['B2_s0_minus_human'], 1, True), f1(v['B2_s1_minus_human'], 1, True)))
    bg = sec['by_genre_aligned']
    L += ['', 'ジャンル別(aligned の mode)の L1・fwd_bias(best、own_labels):', '', '| ジャンル | フレーム | ' + ' | '.join(NAMES_JA[n] for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1')) + ' |',
          '|---|---|' + '---|' * 5]
    for k, v in bg.items():
        L.append('| %s | %d | %s |' % (k, v['frames'], ' | '.join('%s(%s)' % (f1(v[n]['L1'], 1), f1(v[n]['fwd_bias'], 1, True))
                                                                  for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1'))))
    ps = sec['per_scene_L1_own_best']
    L += ['', 'シーン別の L1(best、own_labels)[µs]:', '', '| シーン | フレーム | ' + ' | '.join(NAMES_JA[n] for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1')) + ' |',
          '|---|---|' + '---|' * 5]
    for s, v in ps.items():
        L.append('| %s | %d | %s |' % (s, v['frames'], ' | '.join(f1(v[n], 1) for n in ('A_s0', 'A_s1', 'B', 'B2_s0', 'B2_s1'))))
    L.append('')
    arms = sec.get('optional_arms') or {}
    if arms:
        L += ['参考の腕(種 0 だけ・判定しない。R1 の効きが「外したこと」と「付け替え」のどちらから来たかの目安):', '']
        for n, a in arms.items():
            L.append('- %s: 保留の速い D = %s µs(g=5)/ %s µs(g=4)、学習のログの速い D = %s / %s µs、主指標 L1 %s µs・rest %s µs。' % (
                NAMES_JA[n], f1((a.get('R1_held_fast_g5') or {}).get('D'), 1, True), f1((a.get('R1_held_fast_g4') or {}).get('D'), 1, True),
                f1((a.get('R1_train_fast_g5') or {}).get('D'), 1, True), f1((a.get('R1_train_fast_g4') or {}).get('D'), 1, True),
                f1(a.get('L1_primary_own_best'), 2), f1(a.get('L1_rest_own_best'), 2)))
        L.append('')
    else:
        L += ['- 参考の腕(外すだけ・付け替えだけ)は、本体を先に済ませるため学習していません(protocol の「遅れたら最初に削る」)。', '']
    # 予想との答え合わせ
    pv = res['predictions_vs_results']
    L += ['## 7. 結果を見る前の予想との答え合わせ', '', '| 項目 | 予想(固定) | 結果 |', '|---|---|---|']
    L.append('| R1 保留 | %s | D = %s(g=5)・%s(g=4)µs、%s / %s |' % (pv['G3-R1-held']['predicted'], f1(pv['G3-R1-held']['D_g5'], 1, True),
                                                                     f1(pv['G3-R1-held']['D_g4'], 1, True), '合格' if pv['G3-R1-held']['pass_g5'] else '不合格',
                                                                     '合格' if pv['G3-R1-held']['pass_g4'] else '不合格'))
    L.append('| R1 学習のログ | %s | D = %s(g=5)・%s(g=4)µs |' % (pv['G3-R1-train']['predicted'], f1(pv['G3-R1-train']['D_g5'], 1, True),
                                                             f1(pv['G3-R1-train']['D_g4'], 1, True)))
    L.append('| R1 遅い進入 | %s | D = %s(g=5)・%s(g=4)µs |' % (pv['G3-R1-slow']['predicted'], f1(pv['G3-R1-slow']['D_g5'], 1, True),
                                                           f1(pv['G3-R1-slow']['D_g4'], 1, True)))
    L.append('| DE | %s | ステア %s・スロットル %s µs(g=5) |' % (pv['G3-DE']['predicted'], f1(pv['G3-DE']['steer_D_g5'], 1, True),
                                                         f1(pv['G3-DE']['thr_D_g5'], 1, True)))
    L.append('| 直線 | %s | 低下 %s%%(g=5)・%s%%(g=4) |' % (pv['G3-straight']['predicted'], f1(100 * pv['G3-straight']['drop_g5'], 2),
                                                       f1(100 * pv['G3-straight']['drop_g4'], 2)))
    L.append('| L1 | %s | vs A %s(%s µs)、rest %s |' % (pv['G3-L1']['predicted'], CLS[pv['G3-L1']['class_vs_A']],
                                                       f1(pv['G3-L1']['worsening_best_us'], 2, True), CLS[pv['G3-L1']['rest_class_vs_A']]))
    L.append('| 介入 | %s | vs A %s |' % (pv['G3-intervention']['predicted'], CLS[pv['G3-intervention']['class_vs_A']]))
    L.append('| エンジン | %s | run_numpy 中央値 %s ms |' % (pv['G3-engine']['predicted'], f1(pv['G3-engine']['run_numpy_median'], 3)))
    L.append('| どちらかのジャンルで desk_go | %s | %s |' % (pv['desk_go_any_g']['predicted'],
                                                    'ジャンル %s' % pv['desk_go_any_g']['result'] if pv['desk_go_any_g']['result'] else 'no-go'))
    L.append('')
    L += ['## 8. 注意', '',
          '- 机上の値は「人や AI が実際に走った状態の画像」に対するモデルの出力です。B2 が実際に走れば状態が変わる(閉ループ)ので、実車の進入速度の変化は、この値からは決まりません。',
          '- 判定はゲートの積で、走行のジャンルは 2 つまでです。偽陽性は小さいものの 0 ではありません。',
          '- 保留の人の速い通過は 4 しかありません。保留の AI の 21 通過は、10/10 の first_trial がいろいろなゲインで走った状態です。',
          '- 走行ループの時間は偽の PCA9685 で測った値と、B の実車の値に差を足した見積もりです。実車での確かめはカードの dry run です。', '',
          '## 9. ファイル', '',
          '| ファイル | 中身 |', '|---|---|',
          '| `evaluation.json` | 全部の値(ゲート・部品・通過ごとの差・参考の値・エンジン) |',
          '| `fig/eval_gates.png`・`eval_r1_throttle.png`・`eval_r1_passes.png`・`eval_l1.png`・`eval_engine.png`・`train_val_curves.png` | 図 |',
          '| `eval_sha256.txt` | 窓の表の key・表と道具の sha256(予測・評価の前に固定) |',
          '| `/home/jetson/jetformer/experiments/20261016_b2_curation/eval/b2_eval.py`・`b2_lib.py`・`test_b2_lib.py`・`b2_report.py` | 評価の道具・計算・試験・図と文書 |',
          '| `/home/jetson/jetformer/data/zoo/b2_curation_20261016/` | 窓の表・予測(pred/)・crosscheck・repro・エンジンの確認(engine/) |', '']
    text = '\n'.join(L)
    (REPORT / 'evaluation.md').write_text(text, encoding='utf-8')
    return text


def aba_card(res):
    """固定した card_aba_b2.md(sha256 は protocol.sha256)は書き換えず、写しに机上の結果を埋めて aba_card_b2.md に書く"""
    src = (REPORT / 'card_aba_b2.md').read_text(encoding='utf-8')
    gd = res['drive_genre']
    eng = res.get('engine') or {}
    car = eng.get('car_run')
    g5, g4 = res['gates']['5'], res['gates']['4']
    s = res['gate_detail'][str(gd or 5)]['R1_held']
    e0 = (eng.get('checks') or {}).get('B2_s0') or {}
    rows = {
        'desk_go(5) / desk_go(4)': '%s / %s' % ('合格' if g5['desk_go'] else '不合格(%s)' % '・'.join(GATE_JA[x] for x in g5['failed']),
                                                '合格' if g4['desk_go'] else '不合格(%s)' % '・'.join(GATE_JA[x] for x in g4['failed'])),
        '走らせるジャンル g': ('**%d**' % gd) if gd is not None else '**なし(no-go。B2 は実車に出さない)**',
        '走らせる学習結果': (('`%s`(エンジン合格)' % car) if gd is not None else
                             ('なし(no-go)。エンジンは `b2cur_20261016-s0` で作り、ゲートに合格(参考)')) if car else
                            ('`b2cur_20261016-s0`(エンジン: %s)' % ('不合格' if eng.get('gate') else 'まだ')),
        'R1 の手前 1 秒の予測スロットル(B2 − B、速く近づいた通過)': '%s µs(保留の 25 通過、ジャンル %s。合格線 +30)' % (f1(s['D'], 1, True), gd or 5),
        'エンジン(run_numpy 中央値・FP32 との差)': '%s ms(B %s ms)・%s µs' % (
            f1((e0.get('latency_ms') or {}).get('run_numpy_median'), 2), f1(eng.get('B_run_numpy_same_session_ms'), 2),
            f1((e0.get('accuracy') or {}).get('actions_max_us'), 2)),
    }
    out = []
    for ln in src.splitlines():
        m = re.match(r'^\| (.+?) \| (.+?) \|$', ln)
        if m and m.group(1) in rows:
            ln = '| %s | %s |' % (m.group(1), rows[m.group(1)])
        out.append(ln)
    text = '\n'.join(out)
    text = text.replace('## 机上の結果(学習と判定の後に埋める)', '## 机上の結果(%s に埋めた。詳しくは evaluation.md)' % res['created_at'][:16])
    head = ['> この版は、固定したカード `card_aba_b2.md`(sha256 は `protocol.sha256` にあり、書き換えない)の写しに、'
            '机上の結果とエンジンの結果を埋めたものです。手順・止める決まり・判定は固定のまま変えていません。', '']
    if gd is None:
        head += ['> **机上の判定は no-go でした。このカードは走りません**(モデル B のまま。理由と次の手は `evaluation.md` の 0 章)。',
                 '> 来週の実車は、計画どおりカード 1(B の基準 60 周以上)を先に走り、B2 の枠は計画の項目 13(カード 2: B → B + 速度上限 → B。'
                 '上限を drive.py に入れるのは G1 合格とユーザーの了承のあと)に回すのが筋です。B2 を規則の外で試すかどうかは、ユーザーが決めます'
                 '(試すなら探索の扱いで、このカードの判定は参考になります)。', '']
    lines = text.splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.startswith('# ')), 0)
    rest = lines[i + 1:]
    while rest and not rest[0].strip():
        rest = rest[1:]
    lines = lines[:i + 1] + [''] + head + rest
    extra = ['', '## 付録: 机上で確かめたタイミング(走る前の dry run の目安)', '']
    lp = eng.get('loop') or {}
    if lp:
        for n in ('B', 'B2_s0'):
            if n in lp:
                v = lp[n]
                extra.append('- %s: run_numpy 中央値 %s ms・p99 %s ms、走行ループ(偽の PCA9685)の infer_ms 中央値 %s ms、HOLD %s。' % (
                    NAMES_JA[n], f1(v.get('run_numpy_median'), 2), f1(v.get('run_numpy_p99'), 2), f1(v.get('infer_ms_median'), 2),
                    v.get('holds')))
        extra.append('- dry run の合格線「infer_ms の中央値が B + 1 ms 以下」は、机上では差 %s ms でした。' % f1(
            (lp.get('B2_s0') or {}).get('infer_ms_median', 0) - (lp.get('B') or {}).get('infer_ms_median', 0)
            if lp.get('B2_s0') and lp.get('B') and lp['B2_s0'].get('infer_ms_median') is not None
            and lp['B'].get('infer_ms_median') is not None else None, 2, True))
    else:
        extra.append('- まだ測っていません。')
    extra += ['- 走行の一覧では、B2 は `b2cur_20261016-s0` の名前で出ます。推奨のモデル(drive_recommended)・走行の設定(drive_settings)は B のままで、変えていません。'
              '走る日に画面で B2 を選び、ジャンル g を選びます。', '']
    text = '\n'.join(lines + extra)
    (REPORT / 'aba_card_b2.md').write_text(text, encoding='utf-8')
    return text


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('training', 'all'):
        training_md()
        print('training.md')
    if cmd in ('figures', 'docs', 'all'):
        res = load()
        if cmd in ('figures', 'all'):
            figures(res)
        evaluation_md(res)
        aba_card(res)
        print('evaluation.md・aba_card_b2.md')
    if cmd == 'train-fig':
        print(fig_training(None))


if __name__ == '__main__':
    main()
