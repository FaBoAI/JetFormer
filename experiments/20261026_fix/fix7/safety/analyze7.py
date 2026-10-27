"""直す 7-1 回目・係 safety: exp7.py の出力(out/exp7_<くみ>.json)を表にする。出力 out/analyze7.txt(と標準出力)
列: 組の数・OFF が直す前とビット単位で同じ組・形ごとに「0 m 近く(< 0.1 m)」「escape が前の壁を押した(> 0.05 秒)」
「後ろの当たり」の組の数、下がった距離の最小・中央、OFF との差(形 − OFF)の最小・最大、OFF より 0.05 m を超えて短い・長い組の数
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def table(kind):
    p = HERE / 'out' / ('exp7_%s.json' % kind)
    if not p.exists():
        return ['%s: (まだ無い)' % kind]
    d = json.loads(p.read_text())
    cells = d['cells']
    vs = d['variants']
    lines = ['## %s(%d 組。%s 秒)' % (kind, len(cells), d.get('elapsed_s'))]
    if any('same_off' in r for r in cells.values()):
        lines.append('OFF(両方 0)が直す前の写しとビット単位で同じ: %d / %d' % (sum(1 for r in cells.values() if r.get('same_off')),
                                                                     len(cells)))
    if any('same_on_f7b' in r for r in cells.values()):
        lines.append('今の ON が直す前の写しの ON とビット単位で同じ: %d / %d' % (
            sum(1 for r in cells.values() if r.get('same_on_f7b')), len(cells)))
    lines.append('| 形 | 0 m 近く | escape の壁押し | 後ろの当たり | 下がった 最小 / 中央 | 形 − OFF 最小 / 最大 | OFF より短い / 長い(0.05 m 超) |')
    lines.append('|---|---|---|---|---|---|---|')
    for v in vs:
        backs = [r[v]['back'] if r[v]['back'] is not None else 0.0 for r in cells.values()]
        diffs = [((r[v]['back'] or 0.0) - (r['off']['back'] or 0.0)) for r in cells.values()] if 'off' in vs else [0.0]
        lines.append('| %s | %d | %d | %d | %.3f / %.3f | %+.3f / %+.3f | %d / %d |' % (
            v, sum(1 for b in backs if b < 0.1), sum(1 for r in cells.values() if (r[v]['esc_push'] or 0) > 0.05),
            sum(1 for r in cells.values() if r[v]['rear_hits']), min(backs), statistics.median(backs), min(diffs), max(diffs),
            sum(1 for x in diffs if x < -0.05), sum(1 for x in diffs if x > 0.05)))
    return lines


def main(kinds):
    out = []
    for k in kinds or ['glitch50', 'glitch100', 'burst', 'spring', 'gapslow', 'brake']:
        out += table(k) + ['']
    txt = '\n'.join(out)
    (HERE / 'out' / 'analyze7.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1:])
