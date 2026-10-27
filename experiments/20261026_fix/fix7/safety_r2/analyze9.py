"""直す 7-2 回目・係 safety: exp9.py の出力(out/exp9_<くみ>.json)を表にする。出力 out/analyze9.txt(と標準出力)
列: 組の数・OFF が直す前とビット単位で同じ組・今の ON が直す前の ON とビット単位で同じ組・形ごとに「0 m 近く(< 0.1 m)・escape が
前の壁を押した(> 0.05 秒)」= 失敗の組(うち跳ね 2 つの組)・後ろの当たりの組と最大の速さ・下がった距離の最小 / 中央・OFF との差
(形 − OFF)の最小 / 最大・OFF より 0.05 m を超えて短い / 長い組
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def table(kind):
    p = HERE / 'out' / ('exp9_%s.json' % kind)
    if not p.exists():
        return ['## %s: (まだ無い)' % kind]
    d = json.loads(p.read_text())
    cells = d['cells']
    vs = d['variants']
    lines = ['## %s(%d 組。%s 秒)' % (kind, len(cells), d.get('elapsed_s'))]
    if any('same_off' in r for r in cells.values()):
        lines.append('- OFF(両方 0)が直す前の写し(7-1 回目のあと)とビット単位で同じ: %d / %d' % (
            sum(1 for r in cells.values() if r.get('same_off')), len(cells)))
    if any('same_on_r1' in r for r in cells.values()):
        lines.append('- 今の ON が直す前の ON とビット単位で同じ: %d / %d' % (
            sum(1 for r in cells.values() if r.get('same_on_r1')), len(cells)))
    lines.append('')
    lines.append('| 形 | 失敗(0 m 近く・escape の壁押し) | うち跳ね 2 つ | 後ろの当たり(組・最大 m/s) | 下がった 最小 / 中央 | 形 − OFF 最小 / 最大 | OFF より短い / 長い |')
    lines.append('|---|---|---|---|---|---|---|')
    for v in vs:
        backs = [r[v]['back'] if r[v]['back'] is not None else 0.0 for r in cells.values()]
        fails = [k for k, r in cells.items() if (r[v]['back'] or 0.0) < 0.1 and (r[v]['esc_push'] or 0) > 0.05]
        two = [k for k in fails if cells[k].get('tag', {}).get('two')]
        rh = [r[v]['rear_max_v'] for r in cells.values() if r[v]['rear_hits']]
        diffs = [((r[v]['back'] or 0.0) - (r['off']['back'] or 0.0)) for r in cells.values()] if 'off' in vs else [0.0]
        lines.append('| %s | %d | %d | %d・%.2f | %.3f / %.3f | %+.3f / %+.3f | %d / %d |' % (
            v, len(fails), len(two), len(rh), max(rh, default=0.0), min(backs), statistics.median(backs),
            min(diffs), max(diffs), sum(1 for x in diffs if x < -0.05), sum(1 for x in diffs if x > 0.05)))
    return lines


def main(kinds):
    if not kinds:
        kinds = sorted(p.stem[5:] for p in (HERE / 'out').glob('exp9_*.json'))
    out = []
    for k in kinds:
        out += table(k) + ['']
    txt = '\n'.join(out)
    (HERE / 'out' / 'analyze9.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1:])
