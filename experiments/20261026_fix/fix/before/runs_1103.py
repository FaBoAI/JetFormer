#!/usr/bin/env python3
"""段 plan(2026-10-26): その日の AI 走行の「説明欄の本」と「log.json の実際の設定」を並べ、計画(bench_candidates.json の
その日のブロックの expect)と違う本に印を付ける。読むだけ(標準ライブラリ)。data/raw は変えない。

    python3 -I runs_1103.py                 # 2026-11-03
    python3 -I runs_1103.py --day 20261024  # 別の日(expect が無い日は並べるだけ)

- 説明欄は scene.json(走り終わりのメモが入ったもの)を先に、無ければ log.json。
- 実際の設定 = log.json の drive.params。launch_boost が無い記録は「—」(発進の前進の無いコード)。
- 違い = expect の値が null でない項目で、実際の値と違うもの。違う本は数えない(手順書 7 章)。
"""
import glob
import json
import re
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
RAW = JF / 'data' / 'raw'
KEYS = ('throttle_gain', 'speed_ceiling', 'ceil_left_mps', 'speed_auto', 'launch_boost', 'launch_us', 'steer_trim_us')


def kv(desc, k):
    m = re.search(r'(?:^|\s)%s=(\S*)' % re.escape(k), (desc or '').split(' memo=')[0])
    return m.group(1) if m else None


def desc_of(d, j):
    for p in sorted(d.glob('*/scene.json')):
        try:
            s = json.loads(p.read_text(encoding='utf-8')).get('description')
            if s:
                return s
        except (OSError, ValueError):
            continue
    return j.get('description') or ''


def diffs(expect, params):
    out = []
    for k, v in (expect or {}).items():
        if v is None:
            continue
        a = params.get(k)
        if a is None:
            if k == 'launch_boost' and v == 0:
                continue                      # 発進の前進の無いコード = OFF と同じ
            out.append('%s 無し(計画 %s)' % (k, v))
        elif abs(float(a) - float(v)) > 1e-6:
            out.append('%s %s(計画 %s)' % (k, a, v))
    return out


def model_diff(bench, b, run):
    """ブロックの候補のモデルと、実際に走ったモデル(ブロックが分からなければ見ない)"""
    if not b:
        return []
    want = next((c.get('run') for c in bench.get('candidates') or [] if c.get('id') == b.get('id')), None)
    return [] if not want or want == run else ['モデル %s(計画 %s)' % (run, want)]


def main():
    day = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--day'), '20261103').replace('-', '')
    date = '%s-%s-%s' % (day[:4], day[4:6], day[6:8])
    bench = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    blocks = {b.get('block'): b for d in bench.get('days', []) if d.get('date') == date for b in d.get('blocks') or []}
    rows = []
    for d in sorted(Path(p) for p in glob.glob(str(RAW / ('%s-*' % day)))):
        try:
            j = json.loads((d / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        dr = j.get('drive') or {}
        if not dr or dr.get('dry_run'):
            continue
        desc = desc_of(d, j)
        p = dr.get('params') or {}
        try:
            bno = int(kv(desc, 'block') or 0)
        except ValueError:
            bno = 0
        b = blocks.get(bno) if kv(desc, 'day') == date else None
        rows.append({'log': d.name, 'ai_no': kv(desc, 'ai_no'), 'block': bno, 'run_id': (b or {}).get('run_id'),
                     'tune': kv(desc, 'tune'), 'cand': kv(desc, 'cand'), 'run': dr.get('run'),
                     'params': {k: p.get(k) for k in KEYS}, 'V_before': kv(desc, 'V_before'),
                     'diffs': diffs((b or {}).get('expect'), p) + model_diff(bench, b, dr.get('run'))})
    print('| No. | 記録 | 本 | 説明欄の tune | cand | 速度 | 天井 | 左 | 自動 | 発進 | V前 | 計画との違い |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    f = lambda x: '—' if x is None else ('%g' % x if isinstance(x, (int, float)) else str(x))
    for r in rows:
        q = r['params']
        print('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['ai_no'] or '—', r['log'], r['run_id'] or '—', r['tune'] or '—', r['cand'] or '—', f(q['throttle_gain']),
            f(q['speed_ceiling']), f(q['ceil_left_mps']), f(q['speed_auto']), f(q['launch_boost']), r['V_before'] or '—',
            '・'.join(r['diffs']) or 'なし'))
    print('\n%d 本・計画と違う %d 本' % (len(rows), sum(1 for r in rows if r['diffs'])))
    if '--json' in sys.argv:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
