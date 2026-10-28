"""読むだけ: 合わせた前の係の試験(tests/test_fix7_docs_r1.py・r2・final・test_fix8_docs_decided.py・r1・sync1・r2)が弱まっていないかを、
本物の文書は変えずに読んだ中身の写しを壊して、試験の関数をそのまま呼んで確かめる(fix8 docs r2 の mutate_tests_r2.py と同じ形。2026-10-28 fix8 docs 仕上げ)。

(a) 合わせる前の試験(before_tests/*.orig)は今の文書で落ちる(= 合わせる要があった)。合わせた試験は通る。
(b) この組の SUPERSEDE を 1 つずつ戻すと(今の文書の Y を X に)、その前の台本の合わせた「済」の試験が落ちる。
(c) 前の台本の new(この組の書き換えのあとの今の形)を 1 つずつ壊すと(終わりの 1 字を ★ に)、その台本の合わせた「済」の試験が落ちる
    (この組の revert・effective が前の係の直しの見落としを作っていない)。
(d) 合わせた所: 同期 1 の k4(3.1 の行の 409 の文の時刻だけを 04:54 に戻す = 2 か所の時刻がそろわない)・k0 と fix8 docs r2 の m0
    (todo の題をどちらの時刻でもない時刻に)は落ち、今の文書では通る。
使い方: python3 -B mutate_tests_final.py [--quick](--quick は (c) を台本ごとに 5 つずつ)→ 標準出力(out/mutate_tests_final.txt に写す)
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


fin = _load('mtf_fin', HERE / 'docs_fix8_final.py')
mods = fin.prev_mods()
r82 = mods['r82']
NAMES = {'r1': 'test_fix7_docs_r1.py', 'r2': 'test_fix7_docs_r2.py', 'final': 'test_fix7_docs_final.py',
         'decided': 'test_fix8_docs_decided.py', 'r81': 'test_fix8_docs_r1.py', 's1': 'test_fix8_docs_sync1.py',
         'r82': 'test_fix8_docs_r2.py'}
tests = {k: _load('mtf_t_' + k, T / v) for k, v in NAMES.items()}
origs = {k: _load('mtf_o_' + k, HERE / 'before_tests' / (v + '.orig')) for k, v in NAMES.items()}
LIVE = fin.read_all()


def call(k, mod, texts, tmp):
    """台本 k の「済」の試験を texts で呼ぶ(引数は名前で渡す)。通れば None・落ちれば例外の名前"""
    tmp.mkdir(parents=True, exist_ok=True)
    fn = getattr(mod, 'test_script_edits_all_done_and_earlier_intact' if k == 'final' else 'test_script_edits_all_done')
    names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
    pool = {'live': texts, 'script': mods[k], 'fix8': mods['decided'], 'r81': mods['r81'], 's1': mods['s1'], 'r82': r82, 'fin': fin,
            'tmp_path': tmp}
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
    root = JF / 'experiments' / '20261026_fix' / 'fix8' / 'tmp_docs_final'
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mt_fin_', dir=str(root)) as tmp:
        tmp = Path(tmp)
        for k in tests:                                                     # (a)
            a_new = call(k, tests[k], LIVE, tmp / ('a_new_' + k))
            a_old = call(k, origs[k], LIVE, tmp / ('a_old_' + k))
            out.append('(a) %-8s 合わせた試験: %s / 合わせる前(.orig): %s' % (k, '通る' if a_new is None else '落ちる ' + a_new,
                                                                       '通る' if a_old is None else '落ちる ' + a_old))
        nb, cb = 0, 0                                                       # (b)
        for s, f, i, x, y in fin.SUPERSEDE:
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
                    eff = fin.effective(k, f, i, fin.prev_new(mods, k, f, i))
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
        rb, td = LIVE['runbook_20261103.md'], LIVE['todo_user_1026.md']                     # (d)
        w = '今のファイルは 10/28 11:02 に直したもの)」→ 読み直す'
        assert rb.count(w) == 1
        br = dict(LIVE, **{'runbook_20261103.md': rb.replace(w, w.replace('11:02', '04:54'), 1)})
        k4 = tests['s1'].k4_problems(br)
        out.append('(d) 同期 1 の k4: 3.1 の 409 の文の時刻だけ 04:54 に → %s / 今の文書 → %s' % (
            '落ちる' if k4 else '**通る**', '通る' if not tests['s1'].k4_problems(LIVE) else '落ちる'))
        t0_ = td.splitlines()[0]
        br = dict(LIVE, **{'todo_user_1026.md': td.replace(t0_, '# ユーザーが 11/3 までにすること(10/28 12:0x 現在)', 1)})
        k0, m0 = tests['s1'].k0_problems(br), tests['r82'].m0_problems(br)
        out.append('(d) todo の題をどちらの時刻でもない 12:0x に → 同期 1 の k0 %s・fix8 docs r2 の m0 %s / 今の文書 → k0 %s・m0 %s' % (
            '落ちる' if k0 else '**通る**', '落ちる' if m0 else '**通る**', '通る' if not tests['s1'].k0_problems(LIVE) else '落ちる',
            '通る' if not tests['r82'].m0_problems(LIVE) else '落ちる'))
    out.append('(%.0f 秒)' % (time.time() - t0))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
