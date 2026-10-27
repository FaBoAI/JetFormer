#!/usr/bin/env python3
"""直す 1 回目の前と後で、今の状態(段 1 の前・v3 の予測 0 本・lock2 なし)の答えが同じか(OFF のときの同じ)。
前 = tmp_fix1/base(直す前の台本で出した)、後 = tmp_fix1/new(直した台本)。numpy も torch も使わない。読むだけ。

    python3 -I compare_round1.py <前の場所> <後の場所>

同じでなくてよいもの(この台本が外す): 時刻・秒・台本の sha256・足した欄(下の ADDED)・注の先頭に足した lock2 の注
"""
import json
import sys
from pathlib import Path

VOLATILE = {'made_at', 'seconds', 'eval_v3_sha256', 'desk_v3_sha256', 'script_sha256', 'desk_script_sha256', 'written_at',
            'checked_at', 'desk_made_at', 'desk_path', 'path', 'tab_file'}
ADDED = {'excluded', 'lock2_gate', 'train_state', 'complete', 'lock2', 'lock2_scripts_same', 'lock2_scripts_diff',
         'exclude_why', 'provisional_keep'}


def strip(v, path=''):
    if isinstance(v, dict):
        return {k: strip(x, path + '/' + k) for k, x in v.items() if k not in VOLATILE and k not in ADDED}
    if isinstance(v, list):
        return [strip(x, path) for x in v]
    if isinstance(v, str):
        return v.replace('/tmp_fix1/base/', '/X/').replace('/tmp_fix1/new/', '/X/')
    return v


def diff(a, b, path='', out=None):
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append('%s: 型 %s ≠ %s' % (path, type(a).__name__, type(b).__name__))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append('%s/%s: %s にだけある' % (path, k, '前' if k in a else '後'))
            else:
                diff(a[k], b[k], path + '/' + k, out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append('%s: 長さ %d ≠ %d' % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, '%s[%d]' % (path, i), out)
    elif a != b:
        out.append('%s: %r ≠ %r' % (path, a, b))
    return out


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main(base, new):
    base, new = Path(base), Path(new)
    res = {}
    # 段 0 の 8(check): 確かめの 1,170 個・自己試験・27 本の門・足りない値
    a, b = strip(load(base / 'check' / 'check_v3.json')), strip(load(new / 'check' / 'check_v3.json'))
    res['check_v3.json'] = diff(a, b)
    # 机上の欄(desk report): 27 本の欄。注の先頭の lock2 の注は lock2 が無いので出ない
    a, b = strip(load(base / 'desk' / 'desk_v3.json')), strip(load(new / 'desk' / 'desk_v3.json'))
    res['desk_v3.json'] = diff(a, b)
    # 判定(空回し・本物)
    for d in ('dry', 'real'):
        for f in ('decisions_v3_proposed.json', 'eval_v3.json'):
            a, b = load(base / d / f), load(new / d / f)
            for x in (a, b):                     # 「lock2 が無い」の注と理由(本物だけ)は足したので、それ以外を比べる
                E = x['judge']['decisions'] if f == 'eval_v3.json' else x
                if f == 'eval_v3.json':
                    x['judge']['notes'] = [n for n in x['judge']['notes'] if 'lock2_v3' not in n]
                for k in ('E2', 'E3'):
                    E[k]['why'] = E[k]['why'].replace('lock2_v3 の外・', '').replace('lock2_v3 の外', '')
            res['%s/%s' % (d, f)] = diff(strip(a), strip(b))
    ok = all(not v for v in res.values())
    for k, v in res.items():
        print('%-40s %s' % (k, '同じ' if not v else '違う %d 個: %s' % (len(v), ' / '.join(v[:6]))))
    print('答え: %s' % ('全部同じ(時刻・sha256・足した欄・lock2 の注を除く)' if ok else '違いあり'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
