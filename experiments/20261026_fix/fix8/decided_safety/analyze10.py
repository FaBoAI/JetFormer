"""fix8 係 safety-decided: exp10 の出力(out/exp10_*.json)をまとめる → 標準出力と out/analyze10.json

場面の分け方(ON と OFF の両方の走りで見る。発進(AI 側にした時刻)から最初の周の終わり(+10 秒)まで):
  A 前提の場面 = 当たり 0・AI 側の中立止め 0・リカバリー / 中止のフレーム 0
  B 当たり 0・中立止めあり・リカバリー / 中止 0
  C 当たり 0・リカバリーか中止がどちらかの走りにある
  D 当たりあり
違った場面は、最初に違ったフレームの時刻(発進から)・違った欄・出来事と、⑤ の列の代わり(起点まで to_r1・動く move_0p2 / 0p4・
LH の入口の速さ v_lh・空転の窓の前進 f_slipwin・最初の周の当たり)の ON と OFF を出す。
    python -I analyze10.py
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
GRIDS = ['plain', 'drop', 'vskip', 'glitch', 'burst', 'tof', 'fuzz8101', 'fuzz8102']


def load(grid):
    cells = {}
    parts = sorted(OUT.glob('exp10_%s.json' % grid)) + sorted(OUT.glob('exp10_%s_[0-9].json' % grid))
    for p in parts:
        cells.update(json.loads(p.read_text(encoding='utf-8'))['cells'])
    return cells


def klass(r):
    on, off = r['on'], r['off']
    hits = bool(on['hits'] or off['hits'])
    holds = bool(on['ai_holds'] or off['ai_holds'])
    rec = (on['rec_frames'] + on['giveup_frames'] + off['rec_frames'] + off['giveup_frames']) > 0
    if hits:
        return 'D'
    if rec:
        return 'C'
    if holds:
        return 'B'
    return 'A'


COLS = ('to_r1', 'move_0p2', 'move_0p4', 'v_lh', 'f_slipwin')


def delta(on, off):
    out = {}
    for k in COLS:
        a, b = on.get(k), off.get(k)
        if a != b:
            out[k] = [a, b]
    if len(on['hits']) != len(off['hits']):
        out['hits'] = [len(on['hits']), len(off['hits'])]
    for k in ('giveup_frames', 'rec_frames'):
        if on[k] != off[k]:
            out[k] = [on[k], off[k]]
    return out


def main():
    summary = {}
    diffs = []
    guard_nohit = collections.Counter()
    for g in GRIDS:
        cells = load(g)
        if not cells:
            print(g, 'なし')
            continue
        cnt = collections.Counter()
        same = collections.Counter()
        for key, c in cells.items():
            ai = c['scene']['ai_from']
            for lb in ('lb0', 'lb1'):
                r = c[lb]
                k = klass(r)
                cnt[k] += 1
                same[k] += bool(r['same'])
                if not r['on']['hits'] and not r['off']['hits']:
                    for w in r['on']['guards']:
                        guard_nohit[(g, k, w)] += 1
                if not r['same']:
                    cm = r['cmp']
                    t_first = min([x['t'] for x in (cm['pca_first'], cm['row_first']) if x and x.get('t') is not None],
                                  default=None)
                    diffs.append({'grid': g, 'key': key, 'lb': lb, 'class': k,
                                  't_from_ai': None if t_first is None else round(t_first - ai, 3),
                                  'row_cols': (cm['row_first'] or {}).get('col'), 'ev_first': cm['ev_first'],
                                  'pca_first': cm['pca_first'],
                                  'guards_on': r['on']['guards'], 'guards_off': r['off']['guards'],
                                  'ai_holds_on': r['on']['ai_holds'][:12], 'delta': delta(r['on'], r['off']),
                                  'on': {k2: r['on'][k2] for k2 in COLS + ('giveup_frames', 'rec_frames', 'x_end')},
                                  'off': {k2: r['off'][k2] for k2 in COLS + ('giveup_frames', 'rec_frames', 'x_end')},
                                  'scene': {k2: v for k2, v in c['scene'].items() if k2 in (
                                      'drops', 'veh_skip', 'tof_skip', 'glitch', 'spikes', 'm0', 'volt', 'ai_from', 'front0',
                                      'vmax', 'vcorner')}})
        summary[g] = {'scenes': len(cells), 'runs_pairs': sum(cnt.values()),
                      'by_class': {k: [same[k], cnt[k]] for k in sorted(cnt)}}
        print('%-9s 場面 %4d  ON/OFF の組 %4d  ' % (g, len(cells), sum(cnt.values())) +
              '  '.join('%s %d/%d 同じ' % (k, same[k], cnt[k]) for k in sorted(cnt)))
    tot = collections.Counter()
    tot_same = collections.Counter()
    for g, s in summary.items():
        for k, (a, b) in s['by_class'].items():
            tot[k] += b
            tot_same[k] += a
    print('合計 ' + '  '.join('%s %d/%d 同じ' % (k, tot_same[k], tot[k]) for k in sorted(tot)))
    print('当たりの無い走りの ON の見張りの出来事:', dict(guard_nohit) or 'なし')
    print('違った組 %d' % len(diffs))
    kinds = collections.Counter()
    for d in diffs:
        ev = d['ev_first']
        evk = None
        if ev and isinstance(ev.get('on'), list) and ev['on'] and isinstance(ev['on'][0], str):
            evk = (ev['on'][0], (ev['on'][1] or {}).get('what') if isinstance(ev['on'][1], dict) else None)
        kinds[(d['grid'], d['class'], tuple(sorted(d['delta'])), tuple(d['guards_on'][:2]))] += 1
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print('  %4d  %s' % (v, k))
    (OUT / 'analyze10.json').write_text(json.dumps({'summary': summary, 'total': {k: [tot_same[k], tot[k]] for k in tot},
                                                     'guard_events_no_hit': {'|'.join(map(str, k)): v for k, v in guard_nohit.items()},
                                                     'diffs': diffs}, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
