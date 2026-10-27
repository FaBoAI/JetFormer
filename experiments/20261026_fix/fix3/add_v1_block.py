#!/usr/bin/env python3
"""係 e5(2026-10-26): E5 で決まった 11/3 の新しいモデルの 1 枠を、data/bench_candidates.json の 2026-11-03 の日に足す台本。

    P=/home/jetson/miniconda/envs/jetracer/bin/python
    $P -I experiments/20261026_fix/fix3/add_v1_block.py --dry-run          # 差分を出すだけ(既定。書かない)
    $P -I experiments/20261026_fix/fix3/add_v1_block.py --dry-run --run v3_20261026-v3d-aug2-e60-s0 --label V1   # 名前を手で
  書くのは main session(ユーザーの了承のあと):
    $P -I experiments/20261026_fix/fix3/add_v1_block.py --write --user-approval "<ユーザーの言葉>"
      控え data/bench_candidates.json.bak-<今日>-e5 を作り(あれば同じ中身のときだけ)、書いたあとに読み直して照らす。

足すもの(protocol_v3 12 章 slot_20261103_v3.on_track の順。1 本 4〜6 周。設定は B と同じ):
  候補  id V3(label「v3 の 1 枠(V1 など)」・run = 1 枠の学習)を candidates の最後に
  ブロック 24〜27 を 2026-11-03 の blocks の最後に(手順書の並びの一番後ろ = fix_plan.md 5 章の「残り」):
    24 5-1 V3 表のゲイン − 0.05(安全の確かめ。強い衝突 0 で次へ。押したときの速度 0.85)
    25 5-2 V3 表どおり
    26 5-3 B  表どおり(A/B/A の B)
    27 5-4 V3 表どおり(A/B/A の 2 本目の A)
  changes に 1 行
今の日(2026-10-23・2026-10-24)・11/3 の 1〜23・今の候補・基準・ほかの項目は変えない(足したあとに照らす。違えば止める)。
1 枠が無い(B のまま・決まらない・まだ)なら何も足さない。標準ライブラリだけ(走れるかの確かめに jetformer.drive.model_info を
読めれば読む。読めなければファイルで見る)。車・基板・jetformer-web には触れない。
"""
import argparse
import copy
import difflib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

JF = Path('/home/jetson/jetformer')
BENCH = JF / 'data' / 'bench_candidates.json'
RUNS = JF / 'data' / 'runs'
E5_PROPOSED = Path('/home/jetson/jetformer_reports/fix_20261026/e5_seal_20261101/e5_proposed.json')
DATE = '2026-11-03'
CAND_ID = 'V3'
FIRST_BLOCK = 24
SAFE_GAIN = 0.90          # 押したときの速度(add_day_1103.py と同じ。電圧の表のいちばん低い値)
STEP = 0.05               # 1 本目は表 − 0.05(protocol_v3 12 章)。画面のスライダーも 0.05 刻み
GENRE = 5
ENGINE = 'jetformer_fp16.engine'
EXPECT = {'speed_ceiling': None, 'ceil_left_mps': None, 'speed_auto': 0, 'launch_boost': 0, 'zone_caps': 0}
COND = ('E5 で 1 枠が決まり、ユーザーの了承があるとき。時間が残ったとき(fix_plan.md 5 章の「残り」。残す順は'
        '本番の形・発進の比べ・自動合わせ・左 4.2 のあと)')
STOP = ('止める: qualify_plan 3.7(4G 以上 2 回・8G 以上 1 回・R1 の 3 軸 7G 以上 1 回か本の中央 4.5G 以上・同じ場所で 2 本続けて)。'
        '止まって 5 秒出ないときはその本をやめる')


def slot_from(path):
    """e5_open_seal.py judge の案(e5_proposed.json)→ (学習の名前, 枠の印, 状態, 作り物か)"""
    p = json.loads(Path(path).read_text(encoding='utf-8'))
    e5 = p.get('E5') or {}
    return e5.get('slot'), e5.get('slot_label'), e5.get('status'), bool(p.get('dry_run'))


def blk(no, cid, run_id, tune, gain, gain_rule, after_press, cond):
    return {'block': no, 'id': cid, 'params': {'throttle_gain': gain}, 'tune': tune, 'run_id': run_id, 'form': '4〜6周',
            'gain_rule': gain_rule, 'expect': dict(EXPECT), 'after_press': after_press, 'if': cond}


def new_blocks(label):
    tag = 'v3の1枠(%s)' % label
    common = ('ゲインを電圧の表で入れ直す(下の決まり)→ 天井は B の本と同じ(回 2 で天井 4.1 が使えたなら押したあとにスイッチ ON・'
              '左 4.1)。自動合わせ・発進の前進・区間の速度上限は OFF のまま(押すと OFF)。走行開始の確認の文でモデル・速度ゲイン・'
              '天井の行を読む。各本で電圧(前・後)を説明欄に。' + STOP)
    return [
        blk(FIRST_BLOCK, CAND_ID, '5-1', '5-1・%s・表のゲイン−0.05・4〜6周(安全の確かめ)' % tag, round(SAFE_GAIN - STEP, 2),
            'P1 の電圧の表 − 0.05(押したときは 0.85)', common + '。強い衝突(4G 以上)0 なら次へ。あれば新しいモデルはやめる', COND),
        blk(FIRST_BLOCK + 1, CAND_ID, '5-2', '5-2・%s・表どおり・4〜6周' % tag, SAFE_GAIN, 'P1 の電圧の表', common,
            '5-1 で強い衝突 0 のとき。' + COND),
        blk(FIRST_BLOCK + 2, 'B', '5-3', '5-3・B・表どおり・4〜6周(A/B/AのB)', SAFE_GAIN, 'P1 の電圧の表', common,
            '5-2 を走ったとき。' + COND),
        blk(FIRST_BLOCK + 3, CAND_ID, '5-4', '5-4・%s・表どおり・4〜6周(A/B/Aの2本目のA)' % tag, SAFE_GAIN, 'P1 の電圧の表',
            common + '。使う決まり: 5-2・5-4 の 2 本続けて 3 周の窓の中央値が P1・P2 の良い方より 0.3 秒以上速く、'
                     'DE と R1 の衝突/通過が増えていない(qualify_plan 3.8 の P3)。満たさなければ B のまま', '5-3 を走ったとき。' + COND),
    ]


def candidate(run, label, status):
    return {'id': CAND_ID, 'label': 'v3 の 1 枠(%s)' % label, 'run': run,
            'hypothesis': '学習 v3(原因 A の直し: 交代の直後・低速のブレーキの正解を外す/差し替える)。直線を B にそろえる直しなので、'
                          'B より速くなる見込みは無い(protocol_v3 12 章)。走らせるのは主に次の学習の材料のため',
            'risk': 'mid', 'family': 'v3', 'slot': '11/3 の新しいモデルの 1 枠(E5)',
            'selected_by': 'protocol_v3 E5(experiments/20261026_fix/fix3/e5_open_seal.py judge: 11/1 に封の T・V の拒否権 → '
                           'BF2-1 → V1 → V2 の順の最初 = %s。%s)' % (label, status)}


def change_entry(run, label, approval):
    return {'at': time.strftime('%Y-%m-%d %H:%M'),
            'what': '2026-11-03 にブロック %d〜%d(v3 の 1 枠 %s = %s を表 − 0.05 → 表どおり → B → 新しいモデル)と候補 %s を足した'
                    '(今の日・11/3 の 1〜23・今の候補・基準は変えていない)' % (FIRST_BLOCK, FIRST_BLOCK + 3, label, run, CAND_ID),
            'why': 'protocol_v3 12 章(E5 の 1 枠の走らせ方)。ユーザーの了承: %s' % approval}


def build(cur, run, label, status, approval='(まだ。--dry-run の差分)'):
    days = cur.get('days') or []
    idx = [i for i, d in enumerate(days) if d.get('date') == DATE]
    if len(idx) != 1:
        raise SystemExit('%s の日が 1 つではありません(%d)' % (DATE, len(idx)))
    if any(c.get('id') == CAND_ID for c in cur.get('candidates') or []):
        raise SystemExit('もう候補 %s があります(書きません)' % CAND_ID)
    if any(c.get('run') == run for c in cur.get('candidates') or []):
        raise SystemExit('もう同じ学習の候補があります: %s' % run)
    day = days[idx[0]]
    nos = [b.get('block') for b in day.get('blocks') or []]
    if nos != list(range(1, FIRST_BLOCK)):
        raise SystemExit('%s のブロックが 1〜%d ではありません(%s)。足す場所が決まらない' % (DATE, FIRST_BLOCK - 1, nos))
    new = copy.deepcopy(cur)
    new['candidates'].append(candidate(run, label, status))
    nd = [d for d in new['days'] if d.get('date') == DATE][0]
    nd['blocks'] += new_blocks(label)
    new.setdefault('changes', []).append(change_entry(run, label, approval))
    return new


def same_except_new(old, new):
    """足したのは candidates の最後・11/3 の blocks の最後の 4 つ・changes の最後だけか"""
    probs = []
    for k in set(old) | set(new):
        if k in ('days', 'candidates', 'changes'):
            continue
        if old.get(k) != new.get(k):
            probs.append('項目 %s が変わった' % k)
    if new['candidates'][:-1] != old.get('candidates', []) or new['candidates'][-1].get('id') != CAND_ID:
        probs.append('今の候補が変わった')
    if len(new['days']) != len(old['days']):
        probs.append('日の数が変わった')
    for od, nd in zip(old['days'], new['days']):
        if od.get('date') != DATE:
            if od != nd:
                probs.append('日 %s が変わった' % od.get('date'))
            continue
        if {k: v for k, v in od.items() if k != 'blocks'} != {k: v for k, v in nd.items() if k != 'blocks'}:
            probs.append('%s の日の項目が変わった' % DATE)
        if nd['blocks'][:len(od['blocks'])] != od['blocks'] or len(nd['blocks']) != len(od['blocks']) + 4:
            probs.append('%s の 1〜%d が変わったか、足したのが 4 つでない' % (DATE, len(od['blocks'])))
    if new['changes'][:-1] != old.get('changes', []):
        probs.append('今の changes が変わった')
    return probs


def drivable(run):
    """走れるか(画面と同じ drive.model_info。読めなければファイルで)→ (True/False/None, 理由)"""
    try:
        sys.path.insert(0, str(JF))
        from jetformer import drive
        info = drive.model_info(run)
        if info is None:
            return False, 'TensorRT エンジンが無い(%s/%s)' % (RUNS / run, ENGINE)
        if not info.get('drivable', True):
            return False, info.get('why_not') or '走行できないモデル'
        if GENRE not in (info.get('trained_modes') or []):
            return False, 'ジャンル %d を学習していない' % GENRE
        return True, 'drive.model_info: 走れる(ジャンル %d・%s)' % (GENRE, info.get('policy_title'))
    except Exception as e:      # noqa: BLE001
        d = RUNS / run
        if not (d / ENGINE).exists():
            return False, 'エンジンが無い(drive を読めない: %s)' % type(e).__name__
        try:
            cfg = json.loads((d / 'model_config.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False, 'model_config.json が読めない'
        if GENRE not in (cfg.get('trained_modes') or []):
            return False, 'ジャンル %d を学習していない' % GENRE
        return None, 'エンジンと model_config はある(drive を読めないので精度の確かめは見ていない: %s)' % type(e).__name__


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1) + '\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='差分を出すだけ(既定)')
    ap.add_argument('--write', action='store_true', help='書く(main session。ユーザーの了承のあと)')
    ap.add_argument('--user-approval', default=None, help='--write のときだけ: ユーザーの了承の言葉')
    ap.add_argument('--proposed', default=str(E5_PROPOSED), help='e5_open_seal.py judge の e5_proposed.json')
    ap.add_argument('--run', default=None, help='1 枠の学習の名前(e5_proposed.json を使わないとき)')
    ap.add_argument('--label', default=None, help='枠の印(V1・V2・BF2-1)')
    ap.add_argument('--bench', default=str(BENCH))
    ap.add_argument('--allow-dry-proposal', action='store_true', help='作り物の封の案でも差分を出す(試し)')
    ap.add_argument('--allow-not-drivable', action='store_true', help='走れないモデルでも書く(エンジンを 11/2 までに作るとき)')
    args = ap.parse_args(argv)
    if args.write and args.dry_run:
        ap.error('--write と --dry-run は一緒に使えません')
    bench = Path(args.bench)
    if args.run:
        run, label, status, dry = args.run, args.label or 'V1', '名前を手で渡した', False
    else:
        pp = Path(args.proposed)
        if not pp.exists():
            print('1 枠の案がまだありません(%s)。11/1 の e5_open_seal.py judge のあと。足すものは無し' % pp)
            return 0
        run, label, status, dry = slot_from(pp)
        if dry and not args.allow_dry_proposal:
            raise SystemExit('作り物の封の案です(%s)。試しなら --allow-dry-proposal' % pp)
    if not run:
        print('1 枠は無し(%s)。B のまま。bench_candidates.json には何も足さない' % (status or '—'))
        return 0
    text = bench.read_text(encoding='utf-8')
    cur = json.loads(text)
    approval = args.user_approval if args.write else '(まだ。--dry-run の差分)'
    new = build(cur, run, label, status, approval)
    probs = same_except_new(cur, new)
    if probs:
        raise SystemExit('止めます: ' + '・'.join(probs))
    ok, why = drivable(run)
    old_t, new_t = dumps(cur), dumps(new)
    diff = list(difflib.unified_diff(old_t.splitlines(), new_t.splitlines(), str(bench), str(bench) + '(足したあと)', lineterm='', n=1))
    print('1 枠: %s = %s(%s)' % (label, run, status))
    print('走れるか: %s(%s)' % ({True: 'はい', False: 'いいえ', None: '分からない'}[ok], why))
    print('足すブロック(%s):' % DATE)
    for b in new_blocks(label):
        print('  %2d %s %-3s %s | 押したときの速度 %.2f | %s' % (b['block'], b['run_id'], b['id'], b['tune'], b['params']['throttle_gain'],
                                                       b['gain_rule']))
    print('--- 差分(%d 行) ---' % len(diff))
    print('\n'.join(diff))
    if not args.write:
        print('(差分だけ。書いていない。書くのは main session: ユーザーの了承のあと --write --user-approval "<言葉>")')
        return 0
    if not (args.user_approval or '').strip():
        raise SystemExit('--write には --user-approval "<ユーザーの言葉>" が要ります')
    if ok is False and not args.allow_not_drivable:
        raise SystemExit('走れないモデルなので書きません(%s)。エンジンを作るなら --allow-not-drivable' % why)
    backup = bench.with_name(bench.name + '.bak-%s-e5' % time.strftime('%Y%m%d'))
    if backup.exists():
        if sha(backup) != sha(bench):
            raise SystemExit('控え %s があり、今のファイルと違います(書きません)' % backup)
    else:
        backup.write_bytes(bench.read_bytes())
    tmp = str(bench) + '.tmp-e5'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(new_t)
    os.replace(tmp, bench)
    back = json.loads(bench.read_text(encoding='utf-8'))
    probs = same_except_new(cur, back)
    if probs or back != new:
        raise SystemExit('書いたあとの確かめで違い: ' + '・'.join(probs or ['中身が違う']))
    print('書きました: %s(sha256 %s)。控え %s(sha256 %s)' % (bench, sha(bench), backup, sha(backup)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
