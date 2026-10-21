#!/usr/bin/env python3
"""ベンチマークレポートの雛形(2026-10-21、まとめの段): tools/bench_report.py の出力に、実車の部分の空の表と印を足す

    python -I make_benchmark_template.py                 # 既定: bench_report.py で作り直してから足す
    python -I make_benchmark_template.py --no-regen      # 今の README.md に足すだけ

- 机上の部分(3 章)は bench_report.py が desk_bench.json と bench_candidates.json から埋める。
- 実車の部分(0・2・5・7・8 章)は、まだ記録が無いので空欄の表にする(形は plan/bench_spec.md)。
- 本番のレポートは 10/23・10/24 の夜に tools/bench_nightly.sh が ~/jetformer_reports/real_bench_20261023/ に書く。この雛形は使わない。
標準ライブラリだけ。車・装置・GPU は使わない。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
PREP = Path('/home/jetson/jetformer_reports/bench_prep_20261021')
OUT = PREP / 'benchmark'
BENCH = JF / 'data' / 'bench_candidates.json'
DESK = PREP / 'desk' / 'desk_bench.json'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
MARK = '<!-- template: make_benchmark_template.py -->'


def regen():
    subprocess.run([PY, str(JF / 'tools' / 'bench_report.py'), '--day', '20261023', '--out', str(OUT)], cwd=str(JF), check=True)


def insert_after(text, head, add):
    i = text.index(head)
    j = text.index('\n## ', i + len(head))
    body = text[i + len(head):j].rstrip('\n')
    return text[:i] + head + body + '\n\n' + add.rstrip('\n') + '\n' + text[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-regen', action='store_true')
    a = ap.parse_args()
    if not a.no_regen:
        regen()
    readme = OUT / 'README.md'
    s = readme.read_text(encoding='utf-8')
    if MARK in s:
        print('すでに足してある: %s' % readme)
        return 0
    bench = json.loads(BENCH.read_text(encoding='utf-8'))
    desk = {r['id']: r for r in json.loads(DESK.read_text(encoding='utf-8'))['candidates']}
    cands = {c['id']: c for c in bench['candidates']}
    day1 = bench['days'][0]['blocks']
    plan_blocks = {}
    for b in day1:
        plan_blocks.setdefault(b['id'], []).append(str(b['block']))

    # 冒頭の印
    banner = '\n'.join([
        MARK,
        '> **雛形(2026-10-21)**: 机上の部分(3 章)は埋まっています。実車の部分(0・2・5〜9 章)は空欄です。',
        '> 10/23・10/24 の夜に `bash tools/bench_nightly.sh 20261023`(2 日目は `20261024 --day1 20261023`)が、',
        '> 本番のレポートを `~/jetformer_reports/real_bench_20261023/` に書きます。この雛形は書き換えません。',
        '> 9 章の「記録に無い」は、まだ実車の記録が無いためです。',
        ''])
    first_nl = s.index('\n')
    s = s[:first_nl + 1] + '\n' + banner + s[first_nl + 1:]
    s = s.replace('- 日: \n', '- 日: 2026-10-23(金)・2026-10-24(土)(予定)\n', 1)
    s = s.replace('- ソフト: jetformer の git commit `30b71e8`\n',
                  '- ソフト: jetformer の git commit `30b71e8`(10/21 の変更は作業ツリーにあり、まだ commit していない。'
                  'エンジンの sha256 は下の表)\n', 1)

    # 0 結論
    s = insert_after(s, '## 0. 結論', '- (10/24 の夜に書く)「X が B より良い」・「B のまま」(次に試す候補)・「足りない」のどれか 1 つ。\n'
                     '- 推奨のモデルを変えるかは、ユーザーが決める。')

    # 2 条件: 予定のブロック
    rows = ['', '- 1 日目の予定(data/bench_candidates.json。走ったら上の表に記録が入る)', '',
            '| 日 | ブロック | id | run | 電池(予定) | V_before | V_after | 説明欄の check | memo |', '|---|---|---|---|---|---|---|---|---|']
    batt = {1: '#1', 2: '#1', 3: '#2', 4: '#2', 5: '#3', 6: '#3', 7: '#4', 8: '#4'}
    for b in day1:
        rows.append('| 10/23 | %d | %s | %s | %s | | | | |' % (b['block'], b['id'], cands[b['id']]['run'], batt.get(b['block'], '')))
    rows.append('| 10/24 | (10/23 の夜に決める) | B・T1・T2・B・T2・T1・B | | | | | | |')
    s = insert_after(s, '## 2. 条件', '\n'.join(rows[1:]))

    # 3 1 日目のブロック(予定)を入れる
    for cid, c in cands.items():
        old = '| %s | %s | %s | %s | %s | — | %s |' % (cid, c.get('label', ''), (c.get('family') or '—'), c.get('run', ''),
                                                      c.get('hypothesis', ''), c.get('risk', '—'))
        new = old.replace(' | — | %s |' % c.get('risk', '—'), ' | %s(予定) | %s |' % ('・'.join(plan_blocks.get(cid, [])) or '—', c.get('risk', '—')))
        s = s.replace(old, new, 1)
    # 3 入らなかった候補の机上の値
    ns = bench.get('selection', {}).get('not_selected', [])
    t = ['- 入らなかった候補の机上の値(desk_bench.json)', '',
         '| id | 表示名 | 家族 | l1_best_us | l1_steer_best_us | l1_interv_us | r1_app_thr_diff_us | steer_amp_ratio | trt_p99_ms | brake_lowspeed_frac | 門 |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for x in ns:
        r = desk.get(x['id'], {})
        t.append('| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.3f | %.2f | %.4f | %s |' % (
            x['id'], r.get('label', ''), r.get('family') or '—', r['l1_best_us'], r['l1_steer_best_us'], r['l1_interv_us'],
            r['r1_app_thr_diff_us'], r['steer_amp_ratio'], r['trt_p99_ms'], r['brake_lowspeed_frac'],
            '合格' if r.get('gates_pass') else '不合格'))
    s = insert_after(s, '## 3. 候補の表(机上の指標)', '\n'.join(t))

    # 5 順位表(空)
    head1 = ('| 順位 | ブロック | id | 周 | 失敗/周 [90%] | Δ失敗/周 [90%] | 衝突/周 | 強い衝突/周 | R1 強/通過 | GM/通過 | DE/通過 | '
             '失敗の交代/周 | ラップ中央値 | Δラップ [90%] | S [90%] | 印 |')
    sep1 = '|' + '---|' * 16
    r5 = ['### 5.1 1 日目(10/23)— 上位 2 本を選ぶ', '', head1, sep1,
          '| (B) | %s | B | | | 0 | | | | | | | | 0 | 0 | 基準 |' % '・'.join(plan_blocks.get('B', []))]
    for b in day1:
        if b['id'] != 'B':
            r5.append('| | %d | %s | | | | | | | | | | | | | |' % (b['block'], b['id']))
    r5 += ['', '- 印: T1・T2・足りない(周 < 12)・安全で中止・回れない・はっきり悪い・R1 の枠で入れ替え。',
           '- λ = 0.5・2.0 の順位は参考(選びに使わない。直し A4)。「1 日目の上位 2 本は、B より良いとは言わない」。', '',
           '### 5.2 2 日目(10/24)— ベストモデル', '',
           '| id | 周 | 失敗/周 [90%] | Δ失敗/周 [90%] | R1 強/通過 | ラップ中央値 | Δラップ [90%] | S [90%] | 再起動/4分 | 条件 1〜5 | 結論 |',
           '|---|---|---|---|---|---|---|---|---|---|---|',
           '| B | | | 0 | | | 0 | 0 | | — | 基準 |', '| T1 | | | | | | | | | | |', '| T2 | | | | | | | | | | |']
    s = insert_after(s, '## 5. 結果の順位表', '\n'.join(r5))

    # 7 電池(空)
    r7 = ['| 電池 | 日 | ブロック(予定) | V_before | V_after | 減り | ラップ中央値 | 失敗/周 | 再起動 |', '|---|---|---|---|---|---|---|---|---|',
          '| #6 | 10/23 | 自由走行・dry run | | | | | | |', '| #1 | 10/23 | 1 B・2 SLOW | | | | | | |', '| #2 | 10/23 | 3 ENS・4 B | | | | | | |',
          '| #3 | 10/23 | 5 AUG・6 ACT | | | | | | |', '| #4 | 10/23 | 7 WGT・8 B | | | | | | |', '| #5 | 10/23 | ブレーキの測定 | | | | | | |',
          '', '- B の周から当てた lap = a + b·V の b(秒/V)と 90% 区間、直しを使ったか: (夜に書く)']
    s = insert_after(s, '## 7. 電池ごと', '\n'.join(r7))

    # 8 時間のずれ(空)
    r8 = ['| 日 | B のブロック | 周 | ラップ中央値 | 失敗/周 | R1 強/通過 | 電圧(前→後) |', '|---|---|---|---|---|---|---|',
          '| 10/23 | 1 | | | | | |', '| 10/23 | 4 | | | | | |', '| 10/23 | 8 | | | | | |', '| 10/24 | 1・4・7(・9) | | | | | |',
          '', '- 「ずれあり」の判定(直し A7: B の最初と最後で、ラップか失敗/周の差の 90% 区間が 0 を含まない)と、内挿を使ったか: (夜に書く)']
    s = insert_after(s, '## 8. 時間のずれ(B の 3 回)', '\n'.join(r8))

    readme.write_text(s, encoding='utf-8')
    print('雛形を書いた: %s' % readme)
    return 0


if __name__ == '__main__':
    sys.exit(main())
