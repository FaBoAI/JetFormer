"""10/24 速さのチューニング: 走ったあとの早い評価(2〜3 分)。今日の HHMM 以後の AI 走行(dry run でない)を集計して、
段(tune)ごとに ベストラップ・きれいな周の中央値・衝突・R1 の進入・失敗の交代・電圧を出す。

    python -I quick_eval.py 1600            # 16:00 以後
    python -I quick_eval.py 1600 --no-run   # 集計(bench_nightly)を回し直さず、前の scorecard.json だけ読む

読めなかった本(scorecard の per_log が error だけ。電源断で npz が全部 0 バイトなど)は「(読めない: …)」の行にして飛ばす
(2026-10-26 fix2)。
"""
import glob
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

JF = Path('/home/jetson/jetformer')
RAW = JF / 'data' / 'raw'
DESK = '/home/jetson/jetformer_reports/prep_20261022/desk/desk_bench_merged.json'


def kv(desc, k):
    m = re.search(r'(?:^|\s)%s=(\S+)' % re.escape(k), desc or '')
    return m.group(1) if m else None


def main():
    since = sys.argv[1] if len(sys.argv) > 1 else '0000'
    day = next((a.split('=')[1] for a in sys.argv if a.startswith('--day=')), time.strftime('%Y%m%d'))
    out = Path('/home/jetson/jetformer_reports/bench_%s_tune/since_%s' % (day, since))
    logs = []
    for d in sorted(RAW.glob('%s-*' % day)):
        if d.name.split('-')[1][:4] < since:
            continue
        try:
            j = json.loads((d / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        dr = j.get('drive') or {}
        if not dr or dr.get('dry_run'):
            continue
        logs.append((d.name, j))
    if not logs:
        print('この時刻より後の AI 走行がありません')
        return
    if '--no-run' not in sys.argv:
        out.mkdir(parents=True, exist_ok=True)
        cmd = ['bash', str(JF / 'tools/bench_nightly.sh'), day, '--out', str(out / 'nightly'), '--report', str(out / 'report'), '--',
               '--desk', DESK, '--logs'] + [n for n, _ in logs] + ['--k-first', '0', '--k-later', '0', '--min-laps', '1']
        r = subprocess.run(cmd, cwd=str(JF), capture_output=True, text=True)
        (out / 'nightly.log').write_text(r.stdout + r.stderr, encoding='utf-8')
    sc = json.loads((out / 'nightly' / 'scorecard' / 'scorecard.json').read_text(encoding='utf-8'))
    rows = []
    print('| No. | 段 | モデル | 速さ | ジャンル | AI 秒 | 通過 | きれいな周 | ベスト | 中央値 | 衝突(強) | R1 進入 中央値 | 失敗の交代 | 電池 V前→後 | メモ |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, j in logs:
        L = sc['per_log'].get(name)
        desc = j.get('description') or ''
        p = (j.get('drive') or {}).get('params') or {}
        if not L:
            print('| %s | (集計なし) | | | | | | | | | | | | | |' % kv(desc, 'ai_no'))
            continue
        if 'error' in L or not all(k in L for k in ('passes', 'laps', 'ai_s', 'takeovers', 'cand')):
            # 読めなかった本(電源断で npz が全部 0 バイトなど。scorecard の per_log は error だけ)。止まらずに次の本へ
            # (2026-10-26 fix2 R2-rec-4: 前は KeyError で止まり、そのあとの本と「段ごと」が出なかった)
            print('| %s | (読めない: %s) %s | | | | | | | | | | | | | |' % (
                kv(desc, 'ai_no'), str(L.get('error') or '欄が足りない')[:80].replace('|', '/'), name))
            continue
        ps = [x for x in L['passes'] if x.get('ai')]
        clean = [x['lap_s'] for x in L['laps'] if x.get('clean')]
        ce = L.get('collision_events') or []
        ent = [x['entry_erpm'] for x in ps if x.get('entry_erpm')]
        row = dict(no=kv(desc, 'ai_no'), tune=kv(desc, 'tune') or kv(desc, 'block'), cand=L['cand'], g=p.get('throttle_gain'),
                   mode=j.get('mode_id'), ai=L['ai_s'], n=len(ps), clean=clean, col=len(ce), strong=sum(1 for e in ce if e.get('strong')),
                   ent=ent, to=L['takeovers']['failure'], bat=kv(desc, 'battery'), vb=kv(desc, 'V_before'), va=kv(desc, 'V_after'),
                   memo=(re.search(r'memo=(.*)$', desc) or [None, ''])[1])
        rows.append(row)
        print('| %s | %s | %s | %.2f | %s | %.0f | %d | %d | %s | %s | %d(%d) | %s | %d | %s %s→%s | %s |' % (
            row['no'], row['tune'], row['cand'], row['g'] or 0, row['mode'], row['ai'], row['n'], len(clean),
            '%.2f' % min(clean) if clean else '—', '%.2f' % st.median(clean) if clean else '—', row['col'], row['strong'],
            int(st.median(ent)) if ent else '—', row['to'], row['bat'], row['vb'], row['va'], (row['memo'] or '')[:30]))
    # 段ごと
    print('\n| 段(モデル・速さ) | 本数 | きれいな周 | ベスト | 中央値 | 8 秒台の周 | 衝突/周相当 | 強い衝突 | R1 進入 中央値 |')
    print('|---|---|---|---|---|---|---|---|---|')
    groups = {}
    for r in rows:
        groups.setdefault((r['cand'], r['g'], r['mode']), []).append(r)
    for (cand, g, mode), rs in sorted(groups.items(), key=lambda kvp: (kvp[0][0], kvp[0][1] or 0)):
        clean = [x for r in rs for x in r['clean']]
        ent = [x for r in rs for x in r['ent']]
        n = sum(r['n'] for r in rs)
        col = sum(r['col'] for r in rs)
        print('| %s・%.2f%s | %d | %d | %s | %s | %d | %s | %d | %s |' % (
            cand, g or 0, '・ジャンル%s' % mode if mode != 5 else '', len(rs), len(clean), '%.2f' % min(clean) if clean else '—',
            '%.2f' % st.median(clean) if clean else '—', sum(1 for x in clean if x < 9.0), '%.2f' % (col / n) if n else '—',
            sum(r['strong'] for r in rs), int(st.median(ent)) if ent else '—'))
    print('\n(衝突は label の数え方。「通過」= AI が R1 を回った回数 ≒ 周。8 秒台 = 9.0 秒未満のきれいな周。R1 の進入 6,200 eRPM 以上は 10/10 で強い衝突 6/9)')


if __name__ == '__main__':
    main()
