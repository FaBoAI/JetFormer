"""読むだけ: 合わせた前の係の試験(tests/test_fix7_docs_r2.py・test_fix7_docs_final.py・test_fix8_docs_decided.py・test_fix8_docs_r1.py)が
弱まっていないかを、本物の文書は変えずに読んだ中身の写しを壊して、試験の関数をそのまま呼んで確かめる(fix8 docs r1 の mutate_tests.py と
同じ形)。

(a) 合わせる前の試験(before_tests/*.orig)は今の文書で落ちる(= 合わせる要があった)。合わせた試験は通る。
(b) この組の SUPERSEDE を 1 つずつ戻すと(今の文書の Y を X に)、その前の台本の合わせた「済」の試験が落ちる。
(c) 前の台本の new(この組の書き換えのあとの形)を 1 つずつ壊すと(終わりの 1 字を ★ に)、その台本の合わせた「済」の試験が落ちる
    (この組の revert・effective が前の係の直しの見落としを作っていない)。
(d) h4・j1 の合わせた言葉: approvals 14 の 2 行目の「では起きていない(出す値が直す前と全部同じ」を消すと、h4・j1 とも落ちる。

使い方: python3 -I -B mutate_tests.py [--quick](--quick は (c) を台本ごとに 5 つずつ)
"""
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
T = JF / 'tests'
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(T))


def _load(name, path):
    import importlib.machinery
    spec = importlib.util.spec_from_file_location(name, str(path), loader=importlib.machinery.SourceFileLoader(name, str(path)))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


s1 = _load('mt_s1', HERE / 'docs_fix8_sync1.py')
mods = s1.prev_mods()
fix8, r81 = mods['decided'], mods['r81']
tests = {'r2': _load('mt_t_r2', T / 'test_fix7_docs_r2.py'), 'final': _load('mt_t_final', T / 'test_fix7_docs_final.py'),
         'decided': _load('mt_t_dec', T / 'test_fix8_docs_decided.py'), 'r81': _load('mt_t_r81', T / 'test_fix8_docs_r1.py')}
origs = {'r2': _load('mt_o_r2', HERE / 'before_tests' / 'test_fix7_docs_r2.py.orig'),
         'final': _load('mt_o_final', HERE / 'before_tests' / 'test_fix7_docs_final.py.orig'),
         'decided': _load('mt_o_dec', HERE / 'before_tests' / 'test_fix8_docs_decided.py.orig'),
         'r81': _load('mt_o_r81', HERE / 'before_tests' / 'test_fix8_docs_r1.py.orig')}
LIVE = s1.read_all()
F4 = ('runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md')


def call(k, mod, texts, tmp):
    """台本 k の「済」の試験を texts で呼ぶ。通れば None・落ちれば例外の名前"""
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        if k == 'r2':
            if 's1' in mod.test_script_edits_all_done.__code__.co_varnames:
                mod.test_script_edits_all_done(texts, mods['r2'], fix8, s1)
            else:
                mod.test_script_edits_all_done(texts, mods['r2'], fix8)
        elif k == 'final':
            live4 = {f: texts[f] for f in F4}
            fn = mod.test_script_edits_all_done_and_earlier_intact
            if 's1' in fn.__code__.co_varnames:
                fn(live4, mods['final'], fix8, r81, s1, tmp)
            else:
                fn(live4, mods['final'], fix8, r81, tmp)
        elif k == 'decided':
            fn = mod.test_script_edits_all_done
            if 's1' in fn.__code__.co_varnames:
                fn(texts, fix8, r81, s1)
            else:
                fn(texts, fix8, r81)
        elif k == 'r81':
            fn = mod.test_script_edits_all_done
            if 's1' in fn.__code__.co_varnames:
                fn(texts, r81, s1, tmp)
            else:
                fn(texts, r81, tmp)
        return None
    except AssertionError:          # 落ちた = 見つけた
        return 'AssertionError'
    except Exception as e:          # noqa: BLE001  試験の呼び方の誤りなど(見つけたに数えない)
        return 'ほかの例外 %s: %s' % (type(e).__name__, str(e)[:80])


def main():
    quick = '--quick' in sys.argv
    t0 = time.time()
    out = []
    with tempfile.TemporaryDirectory(prefix='mt_sync1_', dir=str(JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_docs_sync1')) as tmp:
        tmp = Path(tmp)
        # (a)
        for k in tests:
            ta = time.time()
            a_new = call(k, tests[k], LIVE, tmp / ('a_new_' + k))
            out.append('(a) %-8s 合わせた試験の時間 %.1f 秒' % (k, time.time() - ta))
            a_old = call(k, origs[k], LIVE, tmp / ('a_old_' + k))
            out.append('(a) %-8s 合わせた試験: %s / 合わせる前(.orig): %s' % (k, '通る' if a_new is None else '落ちる ' + a_new,
                                                                       '通る' if a_old is None else '落ちる ' + a_old))
        # (b)
        nb, cb = 0, 0
        for s, f, i, x, y in s1.SUPERSEDE:
            if s not in tests:
                continue
            br = dict(LIVE, **{f: LIVE[f].replace(y, x, 1)})
            assert br[f] != LIVE[f]
            r = call(s, tests[s], br, tmp / ('b_%d' % nb))
            nb += 1
            cb += r is not None
            out.append('(b) %-8s %s %d を戻す → %s' % (s, f, i, ('落ちる ' + r) if r else '**通る(見落とし)**'))
        out.append('(b) 落ちた %d / %d' % (cb, nb))
        # (c)
        nc, cc, kinds = 0, 0, {}
        for k in tests:
            n_k = 0
            for f, edits in mods[k].ALL:
                if f not in LIVE:
                    continue
                for i in range(len(edits)):
                    if quick and n_k >= 5:
                        break
                    eff = s1.effective(k, f, i, s1.prev_new(mods, k, f, i))
                    if LIVE[f].count(eff) != 1:
                        out.append('(c) %s %s %d: 今の文書に new が 1 回でない(%d)' % (k, f, i, LIVE[f].count(eff)))
                        continue
                    j = len(eff.rstrip('\n |')) - 1
                    bad = eff[:j] + '★' + eff[j + 1:]
                    br = dict(LIVE, **{f: LIVE[f].replace(eff, bad, 1)})
                    r = call(k, tests[k], br, tmp / ('c_%s_%d' % (k, nc)))
                    nc += 1
                    n_k += 1
                    cc += r is not None
                    kinds[r] = kinds.get(r, 0) + 1
                    if r is None:
                        out.append('(c) %-8s %s %d を壊す → **通る(見落とし)**' % (k, f, i))
            out.append('(c) %-8s ここまで %d 個' % (k, n_k))
        out.append('(c) 落ちた %d / %d(落ち方 %s)' % (cc, nc, kinds))
        # (d)
        w = 'では起きていない(出す値が直す前と全部同じ'
        ap = LIVE['fix3/approvals.md']
        br = dict(LIVE, **{'fix3/approvals.md': ap.replace(w, 'では起きていない(', 1)})
        h4 = tests['final'].h4_problems(br)
        j1 = tests['r81'].j1_problems(br)
        out.append('(d) 14 の 2 行目の「%s」を消す → h4 %s・j1 %s' % (w, '落ちる' if h4 else '**通る**', '落ちる' if j1 else '**通る**'))
        out.append('(d) 今の文書 → h4 %s・j1 %s' % ('通る' if not tests['final'].h4_problems(LIVE) else '落ちる',
                                                    '通る' if not tests['r81'].j1_problems(LIVE) else '落ちる'))
    out.append('(%.0f 秒)' % (time.time() - t0))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
