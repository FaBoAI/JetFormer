"""fix8 係 docs 直す 8-2 回目(f8r2-docs-paper-trial-skip-drops-faults・rehearsal-r2-unmarked-rerun-skip-drops-hit)。直さない。

本物の ⑤ collect.py(今の版。sha256 を出す)と本物の car_check を、作り物の回 3 で流し、手順書の新しい決まり
「当たった・空転した・記録が欠けた(car_check の衝突・空転・欠けの行が出た)ON の本は --skip しない(試し・走り直しでも)。そのまま数える
= 回 3 は満たさない」のとおりにすると「満たさない」のまま、前の書き方(紙の試し・印の無い走り直しは例外なく --skip)だと「使ってよい」に
動くことを確かめる。不具合の無い本は --skip しても「使ってよい」のまま(外してよい)。
  trial  : tests/test_fix8_tools_decided.mtday の回 3(OFF 4・ON 3 + 3-8 = ON。紙に「試し」= 説明欄は書かない)。
           3-8 の形 hit(最初の周の 3.0 秒に 6G)・spin(空転の比 ≈ 1.8)・gap(最初の周のシーン 0002 の IMU.npz が無い)・clean
  rerun  : 反証の票 A(refute_rehearsal_r2_A/rerun_skip.py)と同じ回 3 の 3-2(ON): 1 周目の SW のあと 17.5 m で止まり 0.5 秒で人側
           (リカバリーの前 = 印の無い走り直し)→ 6 秒 → AI 側(走り直し)。走り直しの最初の周の形 hit(LH 21.5 m で 4.5G)・spin(走り直しの
           AI 側から 1.2 秒の IMU を 0.55 倍)・gap(走り直しの最初の周のシーン 0002 の IMU.npz が無い)・clean
読み方 path: doc = 新しい手順書(不具合のある ON の本は --skip しない・無い本は --skip)/ old = 前の書き方(いつも --skip)/
             plain = --skip しない。⑤ は手順書 8.1 と同じ --ignore-gain。car_check は不具合の本だけ(--no-json)。
作り物の日は fix8/tmp_docs_r2/skip/ に作って消す。
    jf-heavy … python -B skip_check_r2.py → 標準出力・out/skip_check_r2.json
"""
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
import test_fix8_tools_decided as T  # noqa: E402
import test_fix7_tools as t7  # noqa: E402
import test_fix7_tools_r2 as t72  # noqa: E402
import car_check as cc  # noqa: E402

TMP = JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_docs_r2' / 'skip'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
TRIAL = {'hit': ('nospin', {'hits': [(3.0, 6.0)]}), 'spin': ('spin', {}), 'gap': ('gap', {}), 'clean': ('nospin', {})}


def verdict(j):
    """回 3 の組の答え(比べのある組だけ)"""
    out = []
    for g in j.get('summary') or []:
        c = g.get('compare') or {}
        if c:
            out.append({'all_ok': c.get('all_ok'), 'false_checks': sorted(k for k, v in (c.get('checks') or {}).items() if v is False),
                        'd_to_r1_s': c.get('d_to_r1_s')})
    return out


def word(v):
    return '・'.join('使ってよい' if g['all_ok'] else '満たさない' for g in v) or '?'


def car_check_text(D, log):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = cc.main([log, '--raw', str(D.raw), '--labels', str(D.tmp / 'nolab_cc'), '--no-json'])
    return rc, buf.getvalue() + err.getvalue()


def cc_flags(txt):
    """car_check の文から: 衝突が 0 回でない・発進の空転の比の行の頭が「!」か「比が出ない」・記録が欠けている"""
    lines = txt.splitlines()
    coll = [x for x in lines if '衝突(AI・' in x]
    slip = [x for x in lines if '発進の空転の比(' in x]
    n = re.search(r'衝突\(AI・[^)]*\): (\d+) 回', coll[0]) if coll else None
    return {'collision': bool(n) and int(n.group(1)) > 0,
            'slip': bool(slip) and (slip[0].startswith('!') or any('比が出ない発進は' in x for x in lines)),
            'missing': '記録が欠けている' in txt,
            'lines': [x.strip()[:160] for x in coll + slip + [x for x in lines if '記録が欠けている' in x][:1]]}


def rerun_book(D, block, prm, shape):
    """refute_rehearsal_r2_A/rerun_skip.py の rear_book と同じ(late = False)。shape で走り直しの最初の周の不具合"""
    base = t7.launch_keys(1.0, 0.15, 0.3)
    t_b = t7.time_at_distance(base + [(200.0, 2.0)], 17.1, 200.0)
    t_stop = t_b + 0.4
    t_h = t_stop + 0.5
    t_res = round(t_h + 6.0, 2)
    dur = t_res + 32.0
    vk = base + [(t_b, 2.0), (t_stop, 0.0), (t_res + 0.15, 0.0), (t_res + 0.45, 2.0), (dur, 2.0)]
    hits = [(t7.time_at_distance(vk, 21.5, dur), 4.5)] if shape == 'hit' else []
    log = D.add(block, prm, 0.15, 0.3, dur=dur, vk=vk, ai=[(1.0, t_h), (t_res, dur - 1.0)], launch=[(1.0, 0.5), (t_res, 0.5)],
                hits=hits)
    if shape == 'spin':
        t72.scale_imu(D.raw, log, t_res, t_res + 1.2, 0.55)
    elif shape == 'gap':
        (D.raw / log / ('%s-0002' % log) / 'IMU.npz').unlink()
    return log, {'t_stop': round(t_stop, 3), 't_human': round(t_h, 3), 't_res': t_res}


def rerun_day(root, shape):
    D = t7.Day(root)
    D.add(12, t7.PRM, 0.30, 0.6)
    log, info = rerun_book(D, 13, t7.ON, shape)
    D.add(14, t7.PRM, 0.31, 0.6)
    D.add(15, t7.ON, 0.20, 0.3)
    D.add(16, t7.PRM, 0.32, 0.6)
    D.add(17, t7.ON, 0.28, 0.3)
    D.add(18, t7.PRM, 0.33, 0.6)
    D.add(19, t7.ON, 0.31, 0.3)
    return D, log, info


def main():
    res = {'collect_sha256': hashlib.sha256(COLLECT.read_bytes()).hexdigest(), 'trial': [], 'rerun': []}
    if TMP.exists():
        shutil.rmtree(TMP)
    k = 0
    try:
        for shape, (sh, kw) in TRIAL.items():
            k += 1
            D, log = T.mtday(TMP / ('t%02d' % k), 'none', sh, **kw)
            rc, txt = car_check_text(D, log)
            fl = cc_flags(txt)
            faulty = fl['collision'] or fl['slip'] or fl['missing']
            row = {'shape': shape, 'log': log, 'car_check_rc': rc, 'car_check': fl, 'faulty_by_car_check': faulty}
            for path in ('plain', 'old', 'doc'):
                skip = path == 'old' or (path == 'doc' and not faulty)
                j, out, _ = T.runc(D, 'new', list(T.IG) + (['--skip', log] if skip else []), 'r2t%d%s' % (k, path))
                row[path] = {'skip': log if skip else None, 'verdict': verdict(j), 'word': word(verdict(j))}
            res['trial'].append(row)
            print('trial %-5s car_check %s → doc %s(skip %s)・old %s・plain %s' % (
                shape, {x: fl[x] for x in ('collision', 'slip', 'missing')}, row['doc']['word'], row['doc']['skip'] is not None,
                row['old']['word'], row['plain']['word']), flush=True)
            shutil.rmtree(D.tmp, ignore_errors=True)
        for shape in ('hit', 'spin', 'gap', 'clean'):
            k += 1
            D, log, info = rerun_day(TMP / ('r%02d' % k), shape)
            rc, txt = car_check_text(D, log)
            fl = cc_flags(txt)
            j0, out0, _ = T.runc(D, 'new', list(T.IG), 'r2r%dp' % k)
            rows = [r for r in j0['launches'] if r['log'] == log]
            t2 = [r['t_ai'] for r in rows if (r.get('start_no') or 1) == 2][0]
            sk = '%s@%.2f' % (log, t2)
            # 走り直しの発進の行(⑤ の発進ごとの行)の不具合
            r2 = [r for r in rows if (r.get('start_no') or 1) == 2][0]
            row = {'shape': shape, 'log': log, 'info': info, 'skip_arg': sk, 'after_abort': r2.get('after_abort'),
                   'rerun_row': {x: r2.get(x) for x in ('t_ai', 'launch_state', 'collisions_first_lap', 'slip', 'first_lap_gap')},
                   'car_check_rc': rc, 'car_check': fl}
            faulty = fl['collision'] or fl['slip'] or fl['missing']
            row['faulty_by_car_check'] = faulty
            row['plain'] = {'skip': None, 'verdict': verdict(j0), 'word': word(verdict(j0))}
            for path in ('old', 'doc'):
                skip = path == 'old' or (path == 'doc' and not faulty)
                j, out, _ = T.runc(D, 'new', list(T.IG) + (['--skip', sk] if skip else []), 'r2r%d%s' % (k, path))
                row[path] = {'skip': sk if skip else None, 'verdict': verdict(j), 'word': word(verdict(j))}
            res['rerun'].append(row)
            print('rerun %-5s 印 %s car_check %s → doc %s(skip %s)・old %s・plain %s' % (
                shape, r2.get('after_abort'), {x: fl[x] for x in ('collision', 'slip', 'missing')}, row['doc']['word'],
                row['doc']['skip'] is not None, row['old']['word'], row['plain']['word']), flush=True)
            shutil.rmtree(D.tmp, ignore_errors=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print('collect.py sha256', res['collect_sha256'][:12])
    (HERE / 'out' / 'skip_check_r2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
