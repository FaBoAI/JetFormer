#!/usr/bin/env python3
"""学習の段のまとめ: train/state.json と各学習結果の result.json → training.md と図 fig/train_val_curves.png(CPU)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_real_accuracy/train/training_report.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'eval'))
import realacc_driver as D  # noqa: E402

ARM_JA = {'ENS': 'ENS(新しい種)', 'SIZE': 'SIZE(データの量)', 'ROT': 'ROT(位相を回す)', 'GAINL': 'GAINL(任意)',
          'SIZE_EQ': 'SIZE-EQ(固定の外の参考)'}


def load(p, d=None):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return d


def fig_curves(rows, out):
    import realacc_figures as F
    plt = F._setup()
    fig, axes = plt.subplots(1, 2, figsize=F.FIGSIZE, sharey=True)
    cols = {'v0': F.BLUE, 'msf': F.PURPLE, 'sizeH': F.GREEN, 'sizeQ': F.ORANGE, 'sizeT': F.GRAY, 'rot': F.RED, 'gainl': F.INK2,
            'sizeHeq': F.GREEN}
    max_ep = 30
    for name, res in rows:
        h = res.get('history') or []
        if not h:
            continue
        key = name.replace(D.PREFIX, '').split('-')[0]
        ax = axes[0] if key in ('v0', 'msf') else axes[1]
        max_ep = max(max_ep, h[-1]['epoch'])
        ax.plot([x['epoch'] for x in h], [x['val_us'] for x in h], color=cols.get(key, F.GRAY), lw=2,
                ls=':' if key == 'sizeHeq' else ('-' if name.endswith(('s0', 's2')) else '--'))
        ax.text(h[-1]['epoch'] + 0.3, h[-1]['val_us'], name.replace(D.PREFIX, ''), fontsize=16, va='center',
                color=cols.get(key, F.GRAY))
    axes[0].set_title('ENS の 4 本(学習の検証)')
    axes[1].set_title('SIZE・ROT・GAINL(学習の検証)')
    for ax in axes:
        ax.set_xlabel('エポック')
    axes[0].set_xlim(0, 37)
    axes[1].set_xlim(0, max_ep + 9)
    axes[0].set_ylabel('学習の道具の検証の L1 [µs](間引き 3 の 1,559 サンプル)')
    axes[0].set_ylim(40, 80)
    fig.tight_layout()
    return F._save(fig, out, 'train_val_curves.png')


def main():
    st = load(D.STATE_PATH, {}) or {}
    proto = json.loads(D.PROTOCOL.read_text(encoding='utf-8'))
    steps = D.build_steps(proto) + D.build_posthoc_steps(proto)      # 後ろ = 固定の外の参考の腕 SIZE-EQ
    L = ['# 学習の段(real_accuracy_20261015)', '',
         '- 書いた時刻: %s(`experiments/20261015_real_accuracy/train/training_report.py`)' % D.now(),
         '- 道具: `experiments/20261015_real_accuracy/train/realacc_driver.py all`(protocol.json の gpu_order の順・GPU は 1 つずつ・'
         '電源断のあとは同じコマンドで続きから)。ログ: `logs/driver.log`、各学習の `data/runs/<名前>/train.log`',
         '- 学習のコマンドは protocol.json の arms.*.train の文字列そのまま。足したのはメモリの扱いだけの包み(THP なし・MALLOC_ARENA_MAX 2・'
         'expandable_segments・choom 1000・全体の空きメモリの見張り。deviations.md)',
         '- 10/15 23:46 ごろ〜10/16 10:03 に電源が落ちた。gainl-s1(8 エポック目まで)と、まだ始めていなかった SIZE-EQ の 3 本は削った。'
         '済んでいた gainl-s0・sizeHeq-s0 は 10/16 に `train/resume_20261016.py` で確かめ・予測した(deviations.md 10/16)', '',
         '| 学習結果 | 腕 | 学習 / 検証のサンプル | best のエポック | 学習の検証 best / last [µs] | 最後の学習の L1 [µs] | 試行・再開 | '
         'メモリの上限 | 時間 | 確かめ | last.pth |', '|---|---|---|---|---|---|---|---|---|---|---|']
    rows = []
    for s in steps:
        if s['type'] != 'train':
            continue
        name = s['name']
        run = D.RUNS / name
        res = load(run / 'result.json') or {}
        rs = (st.get('runs') or {}).get(name) or {}
        if not res or res.get('state') != 'done':
            cut = rs.get('cut') or any(s['group'] in c or name in c for c in st.get('cuts') or [])
            L.append('| %s | %s | — | — | — | — | %d・%d | — | — | %s | — |' % (
                name.replace(D.PREFIX, ''), ARM_JA.get(s['arm'], s['arm']), len(rs.get('attempts') or []), rs.get('resumes', 0),
                ('削った(%s)' % rs['cut'] if isinstance(rs.get('cut'), str) else '削った') if cut else 'まだ'))
            continue
        rows.append((name, res))
        h = res.get('history') or []
        caps = sorted({a.get('mem_cap') for a in rs.get('attempts') or [] if a.get('mem_cap')})
        L.append('| %s | %s | %s / %s | %s | %s / %s | %s | %d・%d | %s | %s | %s | %s |' % (
            name.replace(D.PREFIX, ''), ARM_JA.get(s['arm'], s['arm']), res.get('train_samples'), res.get('val_samples'),
            res.get('best_epoch'), res.get('best_val_us'), h[-1]['val_us'] if h else '—', h[-1]['train_us'] if h else '—',
            len(rs.get('attempts') or []), rs.get('resumes', 0), '・'.join(caps) or '—',
            '%.0f 分' % (res['duration_s'] / 60) if res.get('duration_s') else '—', rs.get('check', '—'),
            '消した(予測を確かめた後)' if rs.get('last_pth_deleted') else ('あり' if (run / 'last.pth').exists() else '—')))
    L += ['', '## 段', '']
    for k, v in (st.get('steps') or {}).items():
        L.append('- %s: `%s`' % (k, json.dumps(v, ensure_ascii=False)))
    if st.get('cuts'):
        L += ['', '## 削った段(protocol の cut_order_if_late・disk)', ''] + ['- %s' % c for c in st['cuts']]
    rot = []
    for n in ('rot-s0', 'rot-s1'):
        p = D.RUNS / (D.PREFIX + n) / 'rot_log.jsonl'
        if p.exists():
            ph = [json.loads(x)['phase'] for x in p.read_text().splitlines() if x.strip()]
            rot.append('- %s: 位相の並び(エポック 1〜)%s' % (n, ''.join(str(x) for x in ph)))
    if rot:
        L += ['', '## ROT の位相(rot_log.jsonl)', ''] + rot
    L += ['', '## 試行の記録(state.json)', '']
    for name, rs in (st.get('runs') or {}).items():
        for a in rs.get('attempts') or []:
            L.append('- %s: %s 開始・%s・上限 %s・終了コード %s%s%s' % (
                name.replace(D.PREFIX, ''), a.get('started'), '再開(%d エポック目まで済み)' % a['resume_from_epoch']
                if a.get('resume_from_epoch') else '最初から', a.get('mem_cap'), a.get('exit', '(道具の外で終わった)'),
                '・メモリの上限で止まった' if a.get('memory_cap_hit') else '', '・' + a['note'] if a.get('note') else ''))
    figp = None
    if rows:
        try:
            figp = fig_curves(rows, D.REPORT / 'fig')
        except Exception as e:      # noqa: BLE001  図が作れなくても表は書く
            L.append('\n(図を作れませんでした: %s)' % e)
    if figp:
        L += ['', '![学習の検証の曲線](fig/%s)' % Path(figp).name]
    L += ['', '空き容量: %.1f GB(書いた時刻)' % D.disk_free_gb(), '']
    (D.REPORT / 'training.md').write_text('\n'.join(L), encoding='utf-8')
    print(D.REPORT / 'training.md')


if __name__ == '__main__':
    main()
