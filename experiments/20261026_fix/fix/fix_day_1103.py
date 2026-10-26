#!/usr/bin/env python3
"""直し(2026-10-26、反証の major 1): data/bench_candidates.json の 2026-11-03 の日の「押したあとに入れるもの」を直す。

    python3 -I fix_day_1103.py            # 違いを表示するだけ(書かない)
    python3 -I fix_day_1103.py --write    # 控え(data/bench_candidates.json.bak-20261026-fix)が今のファイルと同じなら書く

- 発進の前進は API ではなく、ブロックを押したあとに画面のスイッチで入れる(app.js の benchSetFields は、押したときに
  サーバーが 1 だと launch_boost を pending にして OFF を保ち、走行開始で 0 を送るため)。
- 新しい中身は ../plan/add_day_1103.py の day_1103()(1 つの元)。変えてよいのは 11/3 の日の note と、ブロックの
  after_press・if・expect だけ。block・id・params・mode_id・tune・run_id・form・gain_rule(画面と bench_report が読む項目を含む)
  は 1 つも変えない。今の日(10/23・10/24)・候補・基準・ほかの項目も変えない。changes に 1 行足す。
標準ライブラリだけ。車・基板・jetformer-web には触れない。
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
BENCH = JF / 'data' / 'bench_candidates.json'
BACKUP = JF / 'data' / 'bench_candidates.json.bak-20261026-fix'
GEN = JF / 'experiments' / '20261026_fix' / 'plan' / 'add_day_1103.py'
DATE = '2026-11-03'
MAY_CHANGE = ('after_press', 'if', 'expect')          # ブロックで変えてよい項目(画面・bench_report は読まない)


def load_gen():
    spec = importlib.util.spec_from_file_location('add_day_1103', GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def change_entry():
    return {'at': '2026-10-26 13:4x',
            'what': '2026-11-03 の回 3・回 4(12〜23)の after_press・if・expect と日の note を直した'
                    '(発進の前進は押したあとに画面のスイッチで入れる。API では入れない。expect に zone_caps 0)。'
                    'block・id・params・tune などは変えていない',
            'why': '反証(fix_20261026/verify.md 3 章 1)。API で入れると、ブロックを押したときサーバーが 1 なら画面が pending にして'
                   '走行開始で 0 を送り、ON の本が黙って OFF になる。控え data/bench_candidates.json.bak-20261026-fix'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(cur, gen):
    new = copy.deepcopy(cur)
    idx = [i for i, d in enumerate(new.get('days') or []) if d.get('date') == DATE]
    if len(idx) != 1:
        raise SystemExit('%s の日が 1 つでない(%d)' % (DATE, len(idx)))
    want = gen.day_1103()
    old_day = new['days'][idx[0]]
    if [b['block'] for b in old_day['blocks']] != [b['block'] for b in want['blocks']]:
        raise SystemExit('ブロックの並びが違う(書きません)')
    blocks = []
    for ob, wb in zip(old_day['blocks'], want['blocks']):
        for k in set(ob) | set(wb):
            if k in MAY_CHANGE:
                continue
            if ob.get(k) != wb.get(k):
                raise SystemExit('ブロック %s の %s が違う(変えてはいけない項目。書きません)' % (ob['block'], k))
        blocks.append(wb)
    for k in set(old_day) | set(want):
        if k not in ('note', 'blocks') and old_day.get(k) != want.get(k):
            raise SystemExit('日の %s が違う(書きません)' % k)
    new['days'][idx[0]] = dict(old_day, note=want['note'], blocks=blocks)
    new.setdefault('changes', []).append(change_entry())
    return new


def check_same_except(old, new):
    """11/3 の日の note と after_press・if・expect、changes の最後のほかは同じか"""
    probs = []
    for k in set(old) | set(new):
        if k in ('days', 'changes'):
            continue
        if old.get(k) != new.get(k):
            probs.append('項目 %s が変わった' % k)
    if len(old['days']) != len(new['days']):
        probs.append('days の数が変わった')
    for od, nd in zip(old['days'], new['days']):
        if od.get('date') != DATE:
            if od != nd:
                probs.append('%s の日が変わった' % od.get('date'))
            continue
        for k in set(od) | set(nd):
            if k not in ('note', 'blocks') and od.get(k) != nd.get(k):
                probs.append('11/3 の %s が変わった' % k)
        for ob, nb in zip(od['blocks'], nd['blocks']):
            for k in set(ob) | set(nb):
                if k not in MAY_CHANGE and ob.get(k) != nb.get(k):
                    probs.append('11/3 ブロック %s の %s が変わった' % (ob.get('block'), k))
        if len(od['blocks']) != len(nd['blocks']):
            probs.append('11/3 のブロックの数が変わった')
    if new['changes'][:-1] != old.get('changes', []):
        probs.append('今の changes が変わった')
    return probs


def main():
    write = '--write' in sys.argv
    cur = json.loads(BENCH.read_text(encoding='utf-8'))
    new = build(cur, load_gen())
    probs = check_same_except(cur, new)
    if probs:
        raise SystemExit('止めます: ' + '・'.join(probs))
    od = next(d for d in cur['days'] if d['date'] == DATE)
    nd = next(d for d in new['days'] if d['date'] == DATE)
    n = 0
    if od['note'] != nd['note']:
        n += 1
        print('note:\n  前 %s\n  後 %s' % (od['note'], nd['note']))
    for ob, nb in zip(od['blocks'], nd['blocks']):
        for k in MAY_CHANGE:
            if ob.get(k) != nb.get(k):
                n += 1
                print('%2d %s %s:\n  前 %s\n  後 %s' % (ob['block'], ob['run_id'], k, json.dumps(ob.get(k), ensure_ascii=False),
                                                     json.dumps(nb.get(k), ensure_ascii=False)))
    print('違い %d 項目' % n)
    if not write:
        print('(表示だけ。書くときは --write)')
        return
    if not BACKUP.exists() or sha(BACKUP) != sha(BENCH):
        raise SystemExit('控え %s が無いか、今のファイルと違います(書きません)' % BACKUP)
    tmp = str(BENCH) + '.tmp-fix'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(tmp, BENCH)
    back = json.loads(BENCH.read_text(encoding='utf-8'))
    probs = check_same_except(cur, back)
    if probs or back != new:
        raise SystemExit('書いたあとの確かめで違い: ' + '・'.join(probs or ['中身が違う']))
    print('書きました: %s(sha256 %s)。控え %s(sha256 %s)' % (BENCH, sha(BENCH), BACKUP, sha(BACKUP)))


if __name__ == '__main__':
    main()
