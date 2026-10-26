#!/usr/bin/env python3
"""段 plan(2026-10-26): data/bench_candidates.json の days に 2026-11-03 の日を足す。

    python3 -I add_day_1103.py            # 足した形を表示するだけ(書かない)
    python3 -I add_day_1103.py --write    # 控え(data/bench_candidates.json.bak-20261026-plan)が今のファイルと同じなら書く

- 並びは fix_20261026/fix_plan.md 5 章(回 1 基準と床 → 回 2 天井 + 速さ → 回 3 発進の前進の OFF/ON → 回 4 本番の形)。
- 今の日(2026-10-23・2026-10-24)・候補・基準・ほかの項目は変えない(書いたあとに確かめる)。changes に 1 行足す。
- ブロックの上書き(params)は throttle_gain だけ。速さの天井・自動合わせ・左旋回の天井・発進の前進はブロックでは入らない
  (web/bench.py の _block_overrides と app.js の benchSetFields・readDriveParams。check_bench_day.py で確かめる)。
  ブロックを押したあとに、画面のスイッチ(天井・自動合わせ・発進の前進)か、止まっているときの /api/drive/params(左 4.2)で入れる。
  2026-10-26 直し(fix.md): 発進の前進は API で入れない。ブロックを押したときサーバーが 1 だと、画面は launch_boost を
  「まだ送っていない」(pending)にして OFF を保ち、走行開始で 0 を送る(app.js benchSetFields)。だからスイッチで入れる。
  今のファイルを直すのは ../fix/fix_day_1103.py(この day_1103() を使う)。
- expect = 走るときの実際の設定(log.json の drive.params)の計画。null は「その場で決める」。runs_1103.py が照らし合わせる。
  bench.py・bench_report.py は block・id・params・mode_id・tune・counted_ai_s しか読まないので、expect・form などは画面に影響しない。
標準ライブラリだけ。車・基板・jetformer-web には触れない。
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
BENCH = JF / 'data' / 'bench_candidates.json'
BACKUP = JF / 'data' / 'bench_candidates.json.bak-20261026-plan'
DATE = '2026-11-03'
SAFE_GAIN = 0.90          # 押したときの速度ゲイン(電圧の表のいちばん低い値。満充電 8.30 V 以上の P1)。走る前に表で入れ直す

P1 = {'speed_ceiling': 0, 'speed_auto': 0, 'launch_boost': 0}
C41 = {'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0}
C41A = dict(C41, speed_auto=1, auto_target_mps=4.27)
C42 = {'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.2, 'speed_auto': None, 'launch_boost': 0}
BEST2 = {'speed_ceiling': None, 'ceil_left_mps': None, 'speed_auto': None}      # 回 2 のよい形(その場で決める)


def blk(no, run_id, tune, form, gain_rule, expect, after_press, cond=None):
    b = {'block': no, 'id': 'B', 'params': {'throttle_gain': SAFE_GAIN}, 'tune': tune,
         'run_id': run_id, 'form': form, 'gain_rule': gain_rule, 'expect': expect, 'after_press': after_press}
    if cond:
        b['if'] = cond
    return b


def day_1103():
    B = []
    # 回 1・パック A「基準と床」
    B.append(blk(1, '1-1', '1-1・P1確認・天井OFF・4〜6周', '4〜6周', 'P1 の電圧の表', P1, 'ゲインを電圧の表に'))
    B.append(blk(2, '1-2', '1-2・P1・床1本目・止まって3周', '止まって3周', 'P1 の表 + 1-1 の p90 で直す', P1, 'ゲイン'))
    B.append(blk(3, '1-3', '1-3・P1・床2本目・止まって3周', '止まって3周', '1-2 と同じ', P1, 'ゲイン'))
    B.append(blk(4, '1-4', '1-4・天井4.1・同じゲイン・4〜6周', '4〜6周', '1-3 と同じ', C41,
                 'ゲイン → 天井のスイッチ ON'))
    B.append(blk(5, '1-5', '1-5・P1・天井OFF・4〜6周(A/B/AのA)', '4〜6周', '1-3 と同じ', P1, 'ゲイン(天井は押すと OFF)'))
    # 回 2・パック B「天井 + 速さ・自動合わせ・左 4.2」
    B.append(blk(6, '2-1', '2-1・天井4.1・目安ゲイン・4〜6周', '4〜6周', '1-4 の走り終わりの目安「使わないなら」(上げるのは 1.10 まで)',
                 C41, 'ゲイン → 天井 ON'))
    B.append(blk(7, '2-2', '2-2・P1・天井OFF・4〜6周(A)', '4〜6周', 'P1 の電圧の表', P1, 'ゲイン'))
    B.append(blk(8, '2-3', '2-3・天井4.1+自動8.12・4〜6周', '4〜6周', '2-1 の目安「次も自動合わせなら」', C41A,
                 'ゲイン → 天井 ON → 自動合わせ ON(ファームが無ければ説明欄に V_before=)'))
    B.append(blk(9, '2-4', '2-4・天井4.1+自動8.12・くり返し', '4〜6周', '2-3 の目安', C41A, '2-3 と同じ'))
    B.append(blk(10, '2-5', '2-5・天井4.2(条件つき)・4〜6周', '4〜6周', '2-1〜2-4 のよい方と同じ', C42,
                 'API で ceil_left_mps 4.2(止まっているとき)→ ゲイン → 天井 ON(自動合わせは 2-1〜2-4 のよい方と同じ)',
                 cond='2-1〜2-4 で SW の出口・LH の当たり 0・強い衝突 0・周がまだ 8.22 秒より遅いときだけ'))
    B.append(blk(11, '2-6', '2-6・天井4.1に戻す・4〜6周(4.2のA)', '4〜6周', '2-5 と同じ', dict(C42, ceil_left_mps=4.1),
                 'API で ceil_left_mps 4.1 に戻す → ゲイン → 天井 ON', cond='2-5 を走ったときだけ'))
    # 回 3・パック C「発進の前進の OFF/ON 比べ」(protocol_v3 14 章。段 launch の実装・試験・再起動・ユーザーの了承のあと)
    # 発進の前進は、押したあとに画面のスイッチで入れる(API では入れない。2026-10-26 直し fix.md)。区間の速度上限は OFF
    # (ON だと前進が約 184 µs に弱まる。build_launch.md 7 章)。前を 1.5 m 空ける・走行開始は CH3 を人側のまま
    for i in range(8):
        on = i % 2 == 1
        exp = dict(BEST2, zone_caps=0, launch_boost=1 if on else 0)
        if on:
            exp.update(launch_us=250, launch_mps=2.5, launch_max_s=1.2)
        B.append(blk(12 + i, '3-%d' % (i + 1),
                     '3-%d・発進%s・回2のよい形・止まって1〜2周' % (i + 1, 'ON(250・2.5)' if on else 'OFF'),
                     '止まって1〜2周', '回 2 のよい形と同じ', exp,
                     ('ゲイン → 回 2 のよい形のスイッチ → 発進の前進のスイッチ ON(確認の文に答える。API では入れない)。'
                      '区間の速度上限は OFF のまま・前を 1.5 m 空ける。走行開始の確認の文に「発進の前進: ON」の行があること'
                      if on else
                      'ゲイン → 回 2 のよい形のスイッチ。発進の前進は押すと OFF(スイッチに触れない)。'
                      '区間の速度上限は OFF のまま。走行開始の確認の文に「発進の前進」の行が無いこと'),
                     cond=('発進の前進が使えるとき(再起動のあとスイッチが押せる・車輪を浮かせた確かめ・ユーザーの了承)。'
                           '使えなければ OFF の 3-1・3-3・3-5・3-7 だけ' if on else None)))
    # 回 4・パック A'(または D)「止まってから 3 周の本番の形」
    for i in range(4):
        exp = dict(BEST2, zone_caps=0, launch_boost=None)
        B.append(blk(20 + i, '4-%d' % (i + 1), '4-%d・本番の形・止まって3周%s' % (i + 1, '・予備' if i == 3 else ''),
                     '止まって3周', 'その日いちばんよい形(runbook 6 章)', exp,
                     'その日いちばんよい形のゲイン・スイッチ。発進の前進は回 3 で「使ってよい」かつユーザーが決めたときだけ、'
                     '押したあとにスイッチで ON(API では入れない・前を 1.5 m・区間の速度上限 OFF)。'
                     '走行開始の確認の文の「発進の前進」の行で ON/OFF を読む',
                     cond='4-1〜4-3 で 25 秒の判定がまだのとき' if i == 3 else None))
    return {
        'date': DATE,
        'plan': 'fix_20261026',
        'note': ('回1 基準と床 → 回2 天井+速さ → 回3 発進の前進 OFF/ON → 回4 本番の形(止まって3周)。'
                 '押すと速度 0.90・天井・自動合わせ・発進の前進は OFF。ゲインは電圧の表で入れ直し、'
                 '天井・自動合わせ・発進の前進は押したあとに画面のスイッチで、左4.2 は止まっているときに API で入れる'
                 '(手順書 fix_20261026/runbook_20261103.md)'),
        'blocks': B,
    }


def change_entry():
    return {'at': '2026-10-26 11:5x',
            'what': 'days に 2026-11-03 を足した(23 ブロック。どれも B・押したときの速度 0.90。今の日・候補・基準は変えていない)',
            'why': 'fix_20261026 の段 plan(fix_plan.md 5 章の並び・runbook_20261103.md)。控え data/bench_candidates.json.bak-20261026-plan'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(cur):
    if any(d.get('date') == DATE for d in cur.get('days') or []):
        raise SystemExit('もう %s の日があります(書きません)' % DATE)
    new = copy.deepcopy(cur)
    new['days'].append(day_1103())
    new.setdefault('changes', []).append(change_entry())
    return new


def same_except_new(old, new):
    """今の日・候補・基準・ほかの項目が同じで、足したのは days の最後と changes の最後だけか"""
    probs = []
    for k in old:
        if k in ('days', 'changes'):
            continue
        if old[k] != new.get(k):
            probs.append('項目 %s が変わった' % k)
    for k in new:
        if k not in old:
            probs.append('項目 %s が増えた' % k)
    if new['days'][:-1] != old['days']:
        probs.append('今の days が変わった')
    if new['changes'][:-1] != old.get('changes', []):
        probs.append('今の changes が変わった')
    return probs


def main():
    write = '--write' in sys.argv
    text = BENCH.read_text(encoding='utf-8')
    cur = json.loads(text)
    new = build(cur)
    probs = same_except_new(cur, new)
    if probs:
        raise SystemExit('止めます: ' + '・'.join(probs))
    d = new['days'][-1]
    print('足す日 %s: %d ブロック' % (d['date'], len(d['blocks'])))
    for b in d['blocks']:
        print('  %2d %-4s %s | params %s | expect %s%s' % (b['block'], b['run_id'], b['tune'], b['params'],
                                                       json.dumps(b['expect'], ensure_ascii=False),
                                                       (' | 条件: ' + b['if']) if b.get('if') else ''))
    if not write:
        print('(表示だけ。書くときは --write)')
        return
    if not BACKUP.exists() or sha(BACKUP) != sha(BENCH):
        raise SystemExit('控え %s が無いか、今のファイルと違います(書きません)' % BACKUP)
    tmp = str(BENCH) + '.tmp-plan'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(tmp, BENCH)
    back = json.loads(BENCH.read_text(encoding='utf-8'))
    probs = same_except_new(cur, back)
    if probs or back != new:
        raise SystemExit('書いたあとの確かめで違い: ' + '・'.join(probs or ['中身が違う']))
    print('書きました: %s(sha256 %s)。控え %s(sha256 %s)' % (BENCH, sha(BENCH), BACKUP, sha(BACKUP)))


if __name__ == '__main__':
    main()
