"""読むだけ: 合わせた前の係の試験(tests/test_fix7_docs_r1.py・r2・final・test_fix8_docs_decided.py・r1・sync1)が弱まっていないかを、
本物の文書は変えずに読んだ中身の写しを壊して、試験の関数をそのまま呼んで確かめる(fix8 docs 同期 1 の mutate_tests.py と同じ形)。

(a) 合わせる前の試験(before_tests/*.orig)は今の文書で落ちる(= 合わせる要があった)。合わせた試験は通る。
(b) この組の SUPERSEDE を 1 つずつ戻すと(今の文書の Y を X に)、その前の台本の合わせた「済」の試験が落ちる。
(c) 前の台本の new(この組の書き換えのあとの今の形)を 1 つずつ壊すと(終わりの 1 字を ★ に)、その台本の合わせた「済」の試験が落ちる
    (この組の revert・effective が前の係の直しの見落としを作っていない)。
(d) g2 の合わせた所: approvals のいちばん上の「前の今(10/27 17:3x。」の行から「**2 は済み**(10/27 17:16:28 に再起動。**10/27 19:36 から
    の fix7 安全の直しのため」を消すと g2 が落ちる。
使い方: python3 -B mutate_tests_r2.py [--quick](--quick は (c) を台本ごとに 5 つずつ)→ 標準出力(out/mutate_tests_r2.txt に写す)
"""
import importlib.machinery
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
    spec = importlib.util.spec_from_file_location(name, str(path), loader=importlib.machinery.SourceFileLoader(name, str(path)))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


r82 = _load('mt2_r82', HERE / 'docs_fix8_r2.py')
mods = r82.prev_mods()
NAMES = {'r1': 'test_fix7_docs_r1.py', 'r2': 'test_fix7_docs_r2.py', 'final': 'test_fix7_docs_final.py',
         'decided': 'test_fix8_docs_decided.py', 'r81': 'test_fix8_docs_r1.py', 's1': 'test_fix8_docs_sync1.py'}
tests = {k: _load('mt2_t_' + k, T / v) for k, v in NAMES.items()}
origs = {k: _load('mt2_o_' + k, HERE / 'before_tests' / (v + '.orig')) for k, v in NAMES.items()}
LIVE = r82.read_all()


def call(k, mod, texts, tmp):
    """台本 k の「済」の試験を texts で呼ぶ(引数は名前で渡す)。通れば None・落ちれば例外の名前"""
    tmp.mkdir(parents=True, exist_ok=True)
    fn = getattr(mod, 'test_script_edits_all_done_and_earlier_intact' if k == 'final' else 'test_script_edits_all_done')
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    pool = {'live': texts, 'script': mods[k], 'fix8': mods['decided'], 'r81': mods['r81'], 's1': mods['s1'], 'r82': r82, 'tmp_path': tmp}
    if k == 'decided':
        pool['script'] = mods['decided']
    try:
        fn(**{n: pool[n] for n in names})
        return None
    except AssertionError:          # 落ちた = 見つけた
        return 'AssertionError'
    except Exception as e:          # noqa: BLE001  試験の呼び方の誤りなど(見つけたに数えない)
        return 'ほかの例外 %s: %s' % (type(e).__name__, str(e)[:80])


def main():
    quick = '--quick' in sys.argv
    t0 = time.time()
    out = []
    root = JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_docs_r2'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mt_r2_', dir=str(root)) as tmp:
        tmp = Path(tmp)
        for k in tests:                                                     # (a)
            a_new = call(k, tests[k], LIVE, tmp / ('a_new_' + k))
            a_old = call(k, origs[k], LIVE, tmp / ('a_old_' + k))
            out.append('(a) %-8s 合わせた試験: %s / 合わせる前(.orig): %s' % (k, '通る' if a_new is None else '落ちる ' + a_new,
                                                                       '通る' if a_old is None else '落ちる ' + a_old))
        nb, cb = 0, 0                                                       # (b)
        for s, f, i, x, y in r82.SUPERSEDE:
            br = dict(LIVE, **{f: LIVE[f].replace(y, x, 1)})
            assert br[f] != LIVE[f]
            r = call(s, tests[s], br, tmp / ('b_%d' % nb))
            nb += 1
            cb += r is not None
            out.append('(b) %-8s %s %d を戻す → %s' % (s, f, i, ('落ちる ' + r) if r else '**通る(見落とし)**'))
        out.append('(b) 落ちた %d / %d' % (cb, nb))
        nc, cc, kinds = 0, 0, {}                                            # (c)
        for k in tests:
            n_k = 0
            for f, edits in mods[k].ALL:
                if f not in LIVE:
                    continue
                for i in range(len(edits)):
                    if quick and n_k >= 5:
                        break
                    eff = r82.effective(k, f, i, r82.prev_new(mods, k, f, i))
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
        ap = LIVE['fix3/approvals.md']                                      # (d)
        w = '**2 は済み**(10/27 17:16:28 に再起動。**10/27 19:36 からの fix7 安全の直しのため'
        assert ap.count(w) == 1
        br = dict(LIVE, **{'fix3/approvals.md': ap.replace(w, '**2 は済み**(', 1)})
        g2 = tests['r2'].g2_problems(br, tmp / 'd')
        out.append('(d) 「前の今(10/27 17:3x)」の行の「%s…」を消す → g2 %s / 今の文書 → g2 %s' % (
            w[:20], '落ちる' if g2 else '**通る**', '通る' if not tests['r2'].g2_problems(LIVE, tmp / 'd0') else '落ちる'))
    out.append('(%.0f 秒)' % (time.time() - t0))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
