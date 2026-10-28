#!/usr/bin/env python3
"""段 plan(2026-10-26): その日の AI 走行の「説明欄の本」と「log.json の実際の設定」を並べ、計画(bench_candidates.json の
その日のブロックの expect)と違う本に印を付ける。読むだけ(標準ライブラリ)。data/raw は変えない。

    python3 -I runs_1103.py                 # 2026-11-03
    python3 -I runs_1103.py --day 20261024  # 別の日(expect が無い日は並べるだけ)

- 説明欄は scene.json(走り終わりのメモが入ったもの)を先に、無ければ log.json。
- 実際の設定 = log.json の drive.params。launch_boost が無い記録は「—」(発進の前進の無いコード)。
- 違い = expect の値が null でない項目で、実際の値と違うもの。違う本は数えない(手順書 7 章)。
- 発進の前進の出来事 = 記録の events.jsonl の launch_boost(始めた・始めなかった・終わりの理由)。計画が ON で
  始めた回が 0 の本にも印を付ける(ON として数えない。10/26 直し fix.md)。
- 2026-10-26 fix2(直す 2 回目):
  - 計画が ON でなくても、実際に launch_boost = 1 で走って始めた回が 0 の本に印(回 4 は計画が null = その日に決める。R2-rec-6)。
  - 回の形: 計画が null の項目(その日に決める。回 3 は天井・左・自動、回 4 は発進の前進も)は、同じ回の本どうしで同じはず。
    回の最初の本と違う本に印(「回 4 の形が 4-1 と違う」)。
  - 走行中に変えた設定(state2-2): events.jsonl の params・log.json の drive.params_end・AI_CMD の天井・自動合わせが途中から
    OFF・発進の前進の終わり 10(OFF にした)。あれば「途中で変えた」の印(その本は数えない。手順書 7 章)。AI_CMD は numpy が
    あるときだけ読む(無ければ events と log.json だけ)。
- 2026-10-26 critic: log.json が読めない記録(電源断の 0 バイトなど)は表に入れず、最後に「読めない記録」の行で名前を出す
  (前は黙って外した。反証 1 回目 R4)。
- 2026-10-26 fix3 直す 1 回目 r1-rec3-5: 試験の選択肢(rec_back_guard・rec_hold_keep・launch_no_kick。起動のたびに OFF)。
  同じ回の中で回の最初の本と ON/OFF が違う本に印(「回 3 の試験の選択肢が 3-1(No.12)と違う」。再起動で OFF に戻ったときなど)。
  どれかが ON の本があれば、表の下に「試験の選択肢: 1-1(No.1)から 後ろの見張り → 2-3(No.8)から なし(前の本と違う)」の行。
  JSON の行に options。どれも OFF(か欄が無い)なら表・印は前と同じ。
- 2026-10-27 fix5(反証 4 回目 t3): リカバリーの下がる量(rev_s・rev_us。10/23・24 は 0.4 秒・300 µs)を KEYS に足した(JSON の行の
  params)。approvals 10 の (b)(下がる量を小さくする)を選んだ日に読む: 0.4 秒・300 µs でない本があれば表の下に「リカバリーの
  下がる量: 1-1(No.1)から 0.30 秒・250 µs → …」の 1 行。同じ回の中で回の最初の本と違う本に印(「回 2 の下がる量が 2-1(No.6)と
  違う」。基準の設定・ブロックで 0.4・300 に戻ったときなど)。どれも 0.4・300(か欄が無い)なら表・印は前と同じ。
- 2026-10-28 fix8 tools-decided(fix7/fix_docs_r1.md 4 章 3): 試験の選択肢の印の後ろの文を「(再起動で OFF に戻った? 3 章で止めて OFF
  にした本 X から先なら数える。手順書 3・7 章)」に(OPT_MARK_NOTE。前は「(再起動で OFF に戻った? 手順書 7 章)」。手順書 3 章の
  「わざと OFF にした日」も分かる文に)。表の下の試験の選択肢の行(options_line)の本が「—」の記録(run_id が無い = 説明欄に block が
  無い。朝の発進の前進の確かめを収集したまま走った試し)は今まで通り区切りに並べる(手順書 3 章 5・3 章「わざと OFF にした日」・8.1 ② が
  その読み方を書き、tests/test_fix7_docs_r2.py::test_g3 が固定している。計画の無い日は全部の本が「—」なので外すと行が消える)。ただし
  本「—」から始まった区切りに回の本が続いたら、その区切りの後ろに「回の本は X(No.Y)から」を足す(昼に本「—」のあと起こし直しのまま
  回の本を走った日に、区切りが本「—」の名前だけになり、8.1 ② の朝の読み方と読み違えうるため。朝の手順どおりの行は前と同じ)。
- 2026-10-28 fix8 ui r1(反証 8-1 回目 runs-opt-mark-on-the-on-runs): 試験の選択肢の印(round_options)は、回の最初の本ではなく、
  その回の計画の選択肢(回の中で ON の選択肢がいちばん多い本。同じ数なら先の本)と比べる。前は回の最初の本が OFF の日(回の切れ目の
  起き直し・入れ忘れのまま 1 本走り、2 本目から入れ直した)に、印と「計画と違う N 本」が ON で走った正しい本に付き、OFF の本に付かな
  かった(表の下の「回の本は X から」とも食い違った)。起き直しも 3 章の「止めて OFF」も ON → OFF なので、OFF で走った本に印。回の途中
  から OFF になった形・どれも同じ形の出力は前と同じ。回の形(round_form)・下がる量(round_rev)の印は前のまま(回の最初の本と比べる)。
"""
import glob
import json
import re
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
RAW = JF / 'data' / 'raw'
KEYS = ('throttle_gain', 'speed_ceiling', 'ceil_left_mps', 'speed_auto', 'launch_boost', 'launch_us', 'steer_trim_us',
        'zone_caps', 'rev_s', 'rev_us')
# リカバリーの下がる量(drive.py の既定・bench_candidates.json の基準 = 10/23・24 の値。2026-10-27 fix5 t3)。計画(expect)には無いので、
# round_rev・rev_line で読む(どれもこの値か欄が無ければ、表・印は前と同じ)
REV_KEYS = ('rev_s', 'rev_us')
REV_USUAL = (0.4, 300.0)
# 試験の選択肢(drive.py。どれも既定 0・起動のたびに 0。2026-10-26 fix3 直す 1 回目 r1-rec3-5)。計画(expect)には無いので KEYS とは
# 別に読む: 同じ回の中でその回の計画の選択肢(ON がいちばん多い本。fix8 ui r1)と違えば印(A/B/A に混ぜない)、ON の本があれば表の下に
# 1 行(どれも OFF なら出さない = 前と同じ)
OPT_KEYS = ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick')
OPT_NAMES = {'rec_back_guard': '後ろの見張り', 'rec_hold_keep': 'リカバリーを残す', 'launch_no_kick': '見送りで補助なし'}
# 試験の選択肢の印の後ろの文(round_options)。2026-10-28 fix8 tools-decided(fix7/fix_docs_r1.md 4 章 3): 前は「(再起動で OFF に戻った?
# 手順書 7 章)」だけで、手順書 3 章の「わざと OFF にした日」(最初のリカバリーがおかしくて止めて OFF・紙に本 X)の本 X から先の印も
# 起こし直しに見えた。頭の「(再起動で OFF に戻った? 」は同じ(手順書 3 章が引く「(再起動で OFF に戻った? …)」)
OPT_MARK_NOTE = '(再起動で OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。手順書 3・7 章)'


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


def launch_events(d):
    """記録の events.jsonl の launch_boost の出来事 → {'start': n, 'skip': [理由], 'end': [理由]}(無ければ空)"""
    out = {'start': 0, 'skip': [], 'end': []}
    try:
        lines = (d / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('kind') != 'launch_boost':
            continue
        ev = e.get('event')
        if ev == 'start':
            out['start'] += 1
        elif ev in ('skip', 'end'):
            out[ev].append(str(e.get('reason') or e.get('why') or ''))
    return out


def _on(v):
    try:
        return int(float(v)) == 1
    except (TypeError, ValueError):
        return False


def midrun(d, params, drv):
    """走行中に変えた設定 [文](events.jsonl の params・log.json の params_end・AI_CMD の ceil_on・sa_on・launch_end)"""
    out, seen = [], set()

    def add(k, a, b):
        if k not in seen:
            seen.add(k)
            out.append('%s %s→%s' % (k, a, b))
    try:
        lines = (d / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('kind') == 'params' and isinstance(e.get('changed'), dict):
            for k, ab in sorted(e['changed'].items()):
                add(k, ab[0], ab[1])
    for k, v in sorted((drv.get('params_end') or {}).items()):
        a = params.get(k)
        try:
            same = a is not None and abs(float(a) - float(v)) <= 1e-9
        except (TypeError, ValueError):
            same = a == v
        if not same:
            add(k, a, v)
    try:
        import numpy as np
    except ImportError:                 # 標準ライブラリだけのとき: AI_CMD は読まない
        return out
    for sc in sorted(d.glob('*/AI_CMD.npz')):
        try:
            if sc.stat().st_size == 0:
                continue
            with np.load(sc) as z:
                ai = np.asarray(z['select'], bool) & ~np.asarray(z['dry_run'], bool) if 'select' in z else None
                if ai is None or not ai.any():
                    continue
                for k, col in (('speed_ceiling', 'ceil_on'), ('speed_auto', 'sa_on')):
                    if _on(params.get(k)) and col in z and (~np.asarray(z[col], bool) & ai).any():
                        add(k, 1, 0)
                if 'launch_end' in z and (np.asarray(z['launch_end']) == 10).any():
                    add('launch_boost', 1, 0)
        except (OSError, ValueError, KeyError, EOFError):
            continue
    return out


def round_form(rows, blocks):
    """回の形の印: 計画が null の項目(その日に決める)は、同じ回の本どうしで同じはず。回の最初の本と違う本に印を足す"""
    first = {}
    for r in rows:
        b = blocks.get(r['block']) if r.get('run_id') else None
        if not b:
            continue
        rnd = str(r['run_id']).split('-')[0]
        keys = sorted(k for k, v in (b.get('expect') or {}).items() if v is None and k in KEYS)
        if not keys:
            continue
        if rnd not in first:
            first[rnd] = r
            continue
        f = first[rnd]
        diff = ['%s %s(%s は %s)' % (k, r['params'].get(k), f['run_id'], f['params'].get(k)) for k in keys
                if r['params'].get(k) != f['params'].get(k)]
        if diff:
            r['diffs'].append('回 %s の形が %s(No.%s)と違う: %s' % (rnd, f['run_id'], f['ai_no'] or '—', '・'.join(diff)))


def opt_on(r):
    """その本で ON の試験の選択肢(OPT_KEYS の順。欄が無い・0 は OFF)"""
    return tuple(k for k in OPT_KEYS if _on((r.get('options') or {}).get(k)))


def round_options(rows):
    """試験の選択肢の印(r1-rec3-5): 同じ回の本どうしで同じはず。その回の計画の選択肢 = 回の中で ON の選択肢がいちばん多い本
    (同じ数なら先の本)と違う本に印を足す(計画の expect に無い項目)。

    2026-10-28 fix8 ui r1(反証 8-1 回目 runs-opt-mark-on-the-on-runs): 前は回の最初の本と比べたので、回の最初の本が OFF の日
    (回の切れ目で起き直し・入れ忘れのまま 1 本走り、2 本目の前に入れ直した)は、印「回 N の試験の選択肢が …(再起動で OFF に戻った? …)」
    と「計画と違う N 本」が ON で走った正しい本の側に付き、OFF で走った本には付かなかった(表の下の「回の本は X から」の行とも食い違った)。
    試験の選択肢が計画と違う向きは、起き直し(drive.RESET_ON_BOOT)も手順書 3 章の「止めて OFF」も ON → OFF なので、その回で ON が
    いちばん多い本を計画の側にする = OFF で走った本に印。回の途中から OFF になった形(前から正しかった形)の印は前と同じ"""
    rounds = {}
    for r in rows:
        if r.get('run_id'):
            rounds.setdefault(str(r['run_id']).split('-')[0], []).append(r)
    for rnd, rs in rounds.items():
        f = max(rs, key=lambda x: len(opt_on(x)))       # 同じ数なら先の本(max は最初のものを返す)
        b = opt_on(f)
        for r in rs:
            if r is f:
                continue
            a = opt_on(r)
            diff = ['%s %s(%s は %s)' % (OPT_NAMES[k], 'ON' if k in a else 'OFF', f['run_id'], 'ON' if k in b else 'OFF')
                    for k in OPT_KEYS if (k in a) != (k in b)]
            if diff:
                r['diffs'].append('回 %s の試験の選択肢が %s(No.%s)と違う: %s%s' % (
                    rnd, f['run_id'], f['ai_no'] or '—', '・'.join(diff), OPT_MARK_NOTE))


def options_line(rows):
    """どれかの本で試験の選択肢が ON なら、ON/OFF が変わったところを並べた 1 行(どれも OFF なら None)"""
    if not any(opt_on(r) for r in rows):
        return None
    segs, prev, dash = [], None, None
    for r in rows:
        o = opt_on(r)
        if o != prev:
            segs.append(['%s(No.%s)から %s' % (r.get('run_id') or r['log'], r.get('ai_no') or '—',
                                             '・'.join(OPT_NAMES[k] for k in o) or 'なし'),
                         [] if prev is None else ['前の本と違う']])
            prev = o
            dash = segs[-1] if not r.get('run_id') else None
        elif dash is not None and r.get('run_id'):
            # 2026-10-28 fix8 tools-decided: 本「—」(run_id の無い本)から始まった区切りに回の本が続いたら、その最初の回の本も書く
            # (起こし直しのあとの最初の本が本「—」だと、回の本の名前が出ず、手順書 8.1 ② の「本「—」から始まる「なし」の区切り」と
            # 読み違えうる。朝の確かめ(本「—」のあと選択肢を入れて回の本が別の区切りになる)の行は前と同じ)
            dash[1].append('回の本は %s(No.%s)から' % (r['run_id'], r.get('ai_no') or '—'))
            dash = None
    return ('試験の選択肢(log.json の走行開始の値。起動のたびに OFF に戻る。再起動のあとは朝の設定を入れ直す・手順書 7 章): '
            + ' → '.join(h + ('(%s)' % '。'.join(n) if n else '') for h, n in segs))


def rev_of(r):
    """その本の下がる量 (rev_s, rev_us)(log.json の走行開始の値。欄が無い・読めない項目は None)"""
    out = []
    for k in REV_KEYS:
        try:
            v = (r.get('params') or {}).get(k)
            out.append(None if v is None else round(float(v), 6))
        except (TypeError, ValueError):
            out.append(None)
    return tuple(out)


def rev_text(v):
    s, us = v
    return '%s 秒・%s µs' % ('—' if s is None else '%.2f' % s, '—' if us is None else '%.0f' % us)


def round_rev(rows):
    """下がる量の印(fix5 t3): 同じ回の本どうしで同じはず。回の最初の本と違う本に印を足す(どちらも欄がある本だけ比べる)"""
    first = {}
    for r in rows:
        if not r.get('run_id') or None in rev_of(r):
            continue
        rnd = str(r['run_id']).split('-')[0]
        if rnd not in first:
            first[rnd] = r
            continue
        f = first[rnd]
        a, b = rev_of(r), rev_of(f)
        if a != b:
            diff = ['%s %g(%s は %g)' % (k, x, f['run_id'], y) for k, x, y in zip(REV_KEYS, a, b) if x != y]
            r['diffs'].append('回 %s の下がる量が %s(No.%s)と違う: %s(基準の設定・ブロックで基準の値に戻った? approvals 10)' % (
                rnd, f['run_id'], f['ai_no'] or '—', '・'.join(diff)))


def rev_line(rows):
    """0.4 秒・300 µs でない本があれば、下がる量が変わったところを並べた 1 行(どれも 0.4・300 か欄が無ければ None)"""
    vals = [rev_of(r) for r in rows]
    if not any(v != REV_USUAL and v != (None, None) for v in vals):
        return None
    segs, prev = [], None
    for r, v in zip(rows, vals):
        if v != prev:
            segs.append('%s(No.%s)から %s%s' % (r.get('run_id') or r['log'], r.get('ai_no') or '—', rev_text(v),
                                              '' if prev is None else '(前の本と違う)'))
            prev = v
    return ('リカバリーの下がる量(rev_s・rev_us。log.json の走行開始の値。10/23・24 は %s。approvals 10): ' % rev_text(REV_USUAL)
            + ' → '.join(segs))


def launch_text(ev):
    parts = []
    if ev['start']:
        parts.append('始め %d' % ev['start'])
    if ev['end']:
        parts.append('終わり ' + '/'.join(ev['end']))
    if ev['skip']:
        parts.append('始めない ' + '/'.join(sorted(set(ev['skip']))))
    return '・'.join(parts) or '—'


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
    rows, unread = [], []
    for d in sorted(Path(p) for p in glob.glob(str(RAW / ('%s-*' % day)))):
        try:
            j = json.loads((d / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            # 2026-10-26 critic(反証 1 回目 R4): 電源断などで log.json が 0 バイト・壊れた記録を黙って外さない。名前を出す
            if d.is_dir():
                unread.append('%s(%s)' % (d.name, 'log.json が 0 バイト' if (d / 'log.json').is_file()
                                          and (d / 'log.json').stat().st_size == 0 else type(e).__name__))
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
                     'params': {k: p.get(k) for k in KEYS}, 'options': {k: p.get(k) for k in OPT_KEYS},
                     'V_before': kv(desc, 'V_before'),
                     'launch': launch_events(d), 'midrun': midrun(d, p, dr),
                     'diffs': diffs((b or {}).get('expect'), p) + model_diff(bench, b, dr.get('run'))})
        # 計画が ON の本、または実際に ON で走った本(回 4 の計画は null)で、始めた回が 0 → ON として数えない
        if (((b or {}).get('expect') or {}).get('launch_boost') == 1 or _on(p.get('launch_boost'))) \
                and rows[-1]['launch']['start'] == 0:
            rows[-1]['diffs'].append('発進の前進が始まっていない(%s)' % launch_text(rows[-1]['launch']))
        if rows[-1]['midrun']:
            rows[-1]['diffs'].append('途中で変えた: %s' % '・'.join(rows[-1]['midrun']))
    round_form(rows, blocks)
    round_options(rows)
    round_rev(rows)
    print('| No. | 記録 | 本 | 説明欄の tune | cand | 速度 | 天井 | 左 | 自動 | 発進 | 発進の出来事 | V前 | 計画との違い |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    f = lambda x: '—' if x is None else ('%g' % x if isinstance(x, (int, float)) else str(x))
    for r in rows:
        q = r['params']
        print('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['ai_no'] or '—', r['log'], r['run_id'] or '—', r['tune'] or '—', r['cand'] or '—', f(q['throttle_gain']),
            f(q['speed_ceiling']), f(q['ceil_left_mps']), f(q['speed_auto']), f(q['launch_boost']), launch_text(r['launch']),
            r['V_before'] or '—', '・'.join(r['diffs']) or 'なし'))
    ol = options_line(rows)
    if ol:
        print('\n' + ol)
    rl = rev_line(rows)
    if rl:
        print('\n' + rl)
    print('\n%d 本・計画と違う %d 本' % (len(rows), sum(1 for r in rows if r['diffs'])))
    if unread:
        print('読めない記録 %d 本(表に入れていない。AI 走行かどうかも分からない。紙のメモ・動画と照らす): %s'
              % (len(unread), '・'.join(unread)))
    if '--json' in sys.argv:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
