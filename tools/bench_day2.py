#!/usr/bin/env python3
"""10/23 の夜: 2 日目(10/24)のブロックを data/bench_candidates.json の days[1] に書く(2026-10-21 Fix-all、V-protocol minor 12)

    python tools/bench_day2.py --report data/reports/bench_20261023/report.json --plan B          # 見るだけ(書かない)
    python tools/bench_day2.py --t1 SLOW --t2 ENS --plan B --write                                 # T1・T2 を手で
    python tools/bench_day2.py --report ... --plan A --a-conditions-ok --write                     # A 案(B+上限)

並び(plan/protocol.json day2): T1・T2 が 2 本 → B・T1・T2・B・T2・T1・B。1 本 → B・T1・B・T1・B。0 本 → B・B・B。
  A 案: そのあとに B+上限(id 'B+cap'、モデル B、K = 3)・B を足す。B+上限の候補の行(run = B の run)も candidates に足す。
        A 案の条件(10/23 の B の記録で G1 に合格・ユーザーがもう一度了承・コースの配置が 10/10 と同じ)を人が確かめたことを
        --a-conditions-ok で示さないと書かない。区間の速度上限のスイッチは、ブロックを押しても ON にならない(走行タブで入れる)。
  B 案: T1・T2 のブロックの数える時間を 300 秒(counted_ai_s。B は 240 秒のまま)。
T1・T2 は --report(bench_report.py の report.json の 1 日目の ranking)か、--t1・--t2 で渡す。
書くのは days[1](2 日目)と、A 案の B+上限の行だけ。1 日目・基準の値・ほかの候補は変えない。--write のときだけ書き、
前のファイルを data/bench_candidates.json.<時刻>.bak に控える。読み取りだけの部分は標準ライブラリだけ。
"""

import argparse
import copy
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / 'data' / 'bench_candidates.json'
CAP_ID = 'B+cap'
DAY2_DATE = '2026-10-24'
T_COUNTED_B_PLAN = 300


def finalists_from_report(path):
    """bench_report.py の report.json → (T1, T2, λ)。1 日目の ranking が無ければ ValueError"""
    rep = json.loads(Path(path).read_text(encoding='utf-8'))
    for d in rep.get('days') or []:
        rk = d.get('ranking')
        if rk:
            return rk.get('T1'), rk.get('T2'), rk.get('lambda')
    raise ValueError('report.json に 1 日目の順位(ranking)がありません: %s' % path)


def sequence(t1, t2):
    if t1 and t2:
        return ['B', t1, t2, 'B', t2, t1, 'B']
    if t1:
        return ['B', t1, 'B', t1, 'B']
    return ['B', 'B', 'B']


def build_day2(bench, t1, t2, plan, date=DAY2_DATE, lam=None):
    """bench(dict)の写しに 2 日目を書いたもの → (新しい dict, 文の一覧)。id がおかしければ ValueError"""
    if plan not in ('A', 'B'):
        raise ValueError('plan は A か B')
    t1 = t1 or None
    t2 = t2 or None
    if t2 and not t1:
        raise ValueError('T2 だけで T1 が無い')
    known = {c.get('id'): c for c in bench.get('candidates') or []}
    if 'B' not in known:
        raise ValueError('candidates に B がありません')
    for t in (t1, t2):
        if t is not None and (t not in known or t == 'B' or t == CAP_ID):
            raise ValueError('T1・T2 は B 以外の候補の id: %s(候補: %s)' % (t, '・'.join(k for k in known if k != 'B')))
    if t1 and t1 == t2:
        raise ValueError('T1 と T2 が同じ: %s' % t1)
    new = copy.deepcopy(bench)
    seq = sequence(t1, t2)
    if plan == 'A':
        seq = seq + [CAP_ID, 'B']
    blocks = []
    for i, cid in enumerate(seq):
        b = {'block': i + 1, 'id': cid}
        if plan == 'B' and cid not in ('B', CAP_ID):
            b['counted_ai_s'] = T_COUNTED_B_PLAN
        blocks.append(b)
    n = len([t for t in (t1, t2) if t])
    note = '10/23 の夜に決めた: T1 = %s・T2 = %s(protocol ranking.day1_top2%s)。%s 案' % (
        t1 or 'なし', t2 or 'なし', '、λ = %s' % lam if lam is not None else '', plan)
    if plan == 'A':
        note += '。B+上限のブロックは走行タブのスイッチで区間の速度上限を ON にする(ブロックを押しても ON にならない)。K = 3'
    else:
        note += '。T1・T2 のブロックは数える時間 300 秒(画面の AI の時間 6:00 で止める)'
    if n < 2:
        note += '。1 日目に選べたのが %d 本(protocol day2.if_fewer_finalists)' % n
    day = {'date': date, 'plan': plan, 'blocks': blocks, 'note': note}
    days = [d for d in new.get('days') or [] if isinstance(d, dict)]
    idx = next((i for i, d in enumerate(days) if d.get('date') == date), None)
    if idx is None:
        days.append(day)
    else:
        days[idx] = day
    new['days'] = days
    msgs = []
    if plan == 'A' and CAP_ID not in known:
        new['candidates'].append({
            'id': CAP_ID, 'label': 'B+上限(区間の速度上限 ON)', 'run': known['B'].get('run'),
            'hypothesis': '区間の速度上限 v1.1(R1 2.5・GM 2.2・DE 3.0 m/s、位相が分からないとき 3.0)を ON にした B。初見(約 20 周)',
            'risk': 'mid', 'family': 'cap', 'slot': 'A 案',
            'selected_by': 'protocol day2.cap_plan_A(10/23 の B の記録で G1 合格・ユーザーの再了承・コースが 10/10 と同じ)'})
        msgs.append('candidates に %s(run = B の run)を足した' % CAP_ID)
    msgs.append('2 日目 %s: %s' % (date, '・'.join('%d %s' % (b['block'], b['id']) for b in blocks)))
    return new, msgs


def write_atomic(path, data, stamp=None):
    path = Path(path)
    stamp = stamp or datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    bak = path.with_name(path.name + '.%s.bak' % stamp)
    bak.write_bytes(path.read_bytes())
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    os.replace(tmp, path)
    return bak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bench', default=str(BENCH_PATH))
    ap.add_argument('--report', default=None, help='bench_report.py の report.json(1 日目の ranking の T1・T2 を使う)')
    ap.add_argument('--t1', default=None)
    ap.add_argument('--t2', default=None)
    ap.add_argument('--plan', required=True, choices=('A', 'B'))
    ap.add_argument('--a-conditions-ok', action='store_true',
                    help='A 案の条件 3 つ(G1 合格・ユーザーの再了承・コースが同じ)を人が確かめた')
    ap.add_argument('--date', default=DAY2_DATE)
    ap.add_argument('--write', action='store_true', help='書く(無ければ見るだけ)')
    a = ap.parse_args(argv)
    if a.plan == 'A' and not a.a_conditions_ok:
        print('A 案は、条件 3 つ(10/23 の B の記録で G1 合格・ユーザーがもう一度了承・コースの配置が 10/10 と同じ)を確かめてから '
              '--a-conditions-ok を付けてください。そろわなければ --plan B', file=sys.stderr)
        return 2
    lam = None
    try:
        if a.report:
            t1, t2, lam = finalists_from_report(a.report)
            if a.t1 or a.t2:
                print('--report と --t1・--t2 は一緒に使えません', file=sys.stderr)
                return 2
        else:
            t1, t2 = a.t1, a.t2
        bench = json.loads(Path(a.bench).read_text(encoding='utf-8'))
        new, msgs = build_day2(bench, t1, t2, a.plan, a.date, lam)
    except (OSError, ValueError) as e:
        print('書けません: %s' % e, file=sys.stderr)
        return 2
    for m in msgs:
        print(m)
    print(json.dumps([d for d in new['days'] if d.get('date') == a.date][0], ensure_ascii=False, indent=1))
    if a.write:
        bak = write_atomic(a.bench, new)
        print('書きました: %s(前のものは %s)。画面のパネルは次の読み込みで変わります' % (a.bench, bak))
    else:
        print('(見るだけ。書くときは --write)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
