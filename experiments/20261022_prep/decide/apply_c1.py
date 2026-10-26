"""段 Decide(2026-10-23): A10 で C1 が変わったときだけ data/bench_candidates.json の C1 を替える(標準ライブラリだけ)

    python3 -I apply_c1.py --decision logs/decide_c1.json [--write]

- 替えるのは 2 か所だけ: candidates の C1 の行(SLOW)と days[0](2026-10-23)の block 2 の id。ほかは 1 字も変えない
  (書いたあと、その 2 か所を除いて元と同じかを確かめる。違えば元に戻す)。
- 控え: data/bench_candidates.json.bak-20261023(すでにあれば、今のファイルと同じ中身のときだけ続ける)。
- --write が無ければ、替えた中身を表示するだけ。
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

BENCH = Path('/home/jetson/jetformer/data/bench_candidates.json')
BAK = Path('/home/jetson/jetformer/data/bench_candidates.json.bak-20261023')
OLD_ID, OLD_RUN = 'SLOW', 'bench_20261021-slow-s0'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(d):
    return json.dumps(d, ensure_ascii=False, indent=1) + '\n'         # 元のファイルと同じ書き方(indent 1・最後に改行)


def strip(d):
    """比べるために、替える 2 か所を除いた形"""
    e = copy.deepcopy(d)
    e['candidates'] = [c for c in e['candidates'] if c.get('slot') != 'C1']
    for day in e['days']:
        if day.get('date') == '2026-10-23':
            day['blocks'] = [b for b in day['blocks'] if b.get('block') != 2]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decision', required=True)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    dec = json.load(open(a.decision, encoding='utf-8'))
    if not dec.get('all_checks_ok'):
        print('decide_c1.json の確かめが通っていません。替えません'); return 2
    if not dec.get('changed'):
        print('C1 は変わりません(%s)。替えません' % dec.get('c1_id')); return 0
    new_id, new_run = dec['c1_id'], dec['c1_run']
    row = next(t for t in dec['table'] if t['id'] == new_id)

    text = BENCH.read_text(encoding='utf-8')
    old = json.loads(text)
    assert dump(old) == text, '元のファイルの書き方が indent 1 ではありません(書き直すと形が変わる)'
    c1 = [i for i, c in enumerate(old['candidates']) if c.get('slot') == 'C1']
    if len(c1) == 1 and old['candidates'][c1[0]]['id'] == new_id and old['candidates'][c1[0]]['run'] == new_run:
        print('すでに替えてあります(C1 = %s・%s)。何もしません' % (new_id, new_run)); return 0
    assert len(c1) == 1 and old['candidates'][c1[0]]['id'] == OLD_ID and old['candidates'][c1[0]]['run'] == OLD_RUN, \
        '今の C1 が SLOW(%s)ではありません' % OLD_RUN
    d0 = [d for d in old['days'] if d.get('date') == '2026-10-23']
    assert len(d0) == 1
    b2 = [b for b in d0[0]['blocks'] if b.get('block') == 2]
    assert len(b2) == 1 and b2[0]['id'] == OLD_ID, 'block 2 が SLOW ではありません'
    assert all(c['id'] != new_id for c in old['candidates']), '%s はすでに candidates にあります' % new_id

    new = copy.deepcopy(old)
    i = c1[0]
    oldrow = old['candidates'][i]
    new['candidates'][i] = {
        'id': new_id,
        'label': oldrow['label'],                                   # 「遅く入る型」のまま
        'run': new_run,
        'hypothesis': ('R1 の課題への対策(区間だけ緩める版): 上限 v1.1 を記録に再生し、上限がかかる区間(R1_app・R1_turn・GM・DE)'
                       'の行だけスロットルのラベルを上限の出力に付け替えて学ぶ(区間の外は人のまま。種 1)→ R1 の手前で自分で緩め、'
                       '区間の外では SLOW より遅くならないか'),
        'risk': 'low',                                              # protocol risk_rule: 0.90 ≤ 0.988 ≤ 1.10・r1_app +26.1 ≥ −10・p99 2.45 ≤ 8.0・jetformer_v0
        'family': 'label_r1',
        'slot': 'C1',
        'selected_by': ('A10(plan/protocol_amendments.md、2026-10-23): 対象 = A1 の門に合格かつ r1_app_thr_diff_us ≥ +20.0 → '
                        'nonzone_thr_diff_us の小さい順で 1 番目(%.1f µs。SLOW1 %.1f・SLOW %.1f。SLOW-SEC0 は門 brake_lowspeed_frac で外れた)'
                        % (row['nonzone_thr_diff_us'],
                           next(t for t in dec['table'] if t['id'] == 'SLOW1')['nonzone_thr_diff_us'],
                           next(t for t in dec['table'] if t['id'] == 'SLOW')['nonzone_thr_diff_us'])),
    }
    for b in [d for d in new['days'] if d.get('date') == '2026-10-23'][0]['blocks']:
        if b.get('block') == 2:
            b['id'] = new_id
    assert strip(new) == strip(old), '替える 2 か所の外が変わっています'
    out = dump(new)
    print('--- 替える C1 の行:')
    print(json.dumps(new['candidates'][i], ensure_ascii=False, indent=1))
    print('--- 10/23 の並び:', '・'.join(b['id'] for b in [d for d in new['days'] if d.get('date') == '2026-10-23'][0]['blocks']))
    if not a.write:
        print('(--write が無いので書いていません)'); return 0

    before = sha(BENCH)
    if BAK.exists():
        if sha(BAK) != before:
            print('控え %s がすでにあり、今のファイルと中身が違います。替えません' % BAK); return 3
    else:
        shutil.copy2(BENCH, BAK)
    assert sha(BAK) == before
    tmp = BENCH.with_suffix('.json.tmp-decide')
    tmp.write_text(out, encoding='utf-8')
    json.loads(tmp.read_text(encoding='utf-8'))
    shutil.copymode(BENCH, tmp)
    tmp.replace(BENCH)
    chk = json.loads(BENCH.read_text(encoding='utf-8'))
    if strip(chk) != strip(old) or BENCH.read_text(encoding='utf-8') != out:
        shutil.copy2(BAK, BENCH)
        print('書いたものが合いません。控えから戻しました'); return 4
    print('書きました: %s(控え %s、sha256 %s → %s)' % (BENCH, BAK, before[:12], sha(BENCH)[:12]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
