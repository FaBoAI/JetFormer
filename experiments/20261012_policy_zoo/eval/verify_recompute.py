"""保存した予測(data/zoo/policy_zoo_20261012/pred・pert)から評価をやり直し(CPU。予測はしない)、保存済みの
eval/zoo_eval.json・eval/tables.md と一致するかを確かめる。書くのは eval/verify/ だけ。

    CUDA_VISIBLE_DEVICES= python /home/jetson/jetformer_reports/policy_zoo_20261012/eval/verify_recompute.py
"""
import datetime
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tools')
import policy_zoo as pz  # noqa: E402

HERE = Path(__file__).resolve().parent
REP = HERE.parent
OUT = HERE / 'verify'


def compare(a, b, path, diffs):
    """2 つの json の木を比べる(created_at・seconds は除く。数は相対 1e-6 まで同じとみなす)"""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ('created_at', 'seconds'):
                continue
            if k not in a or k not in b:
                diffs.append([path + '/' + k, 'ない側あり', k in a, k in b])
                continue
            compare(a[k], b[k], path + '/' + k, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append([path, '長さ', len(a), len(b)])
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, '%s[%d]' % (path, i), diffs)
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-6 * max(1.0, abs(a)):
            diffs.append([path, a, b])
    elif a != b:
        diffs.append([path, a, b])


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    res = pz.evaluate_all(str(REP))
    with open(OUT / 'zoo_eval_recomputed.json', 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    saved = json.load(open(HERE / 'zoo_eval.json', encoding='utf-8'))
    diffs = []
    compare(res, saved, '', diffs)
    spec = importlib.util.spec_from_file_location('make_tables', HERE / 'make_tables.py')
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    t_new = mt.tables(res)
    (OUT / 'tables_recomputed.md').write_text(t_new, encoding='utf-8')
    out = {'checked_at': datetime.datetime.now().isoformat(timespec='seconds'), 'seconds': round(time.time() - t0, 1),
           'zoo_eval_diffs': len(diffs), 'zoo_eval_diff_examples': diffs[:20],
           'tables_identical': t_new == (HERE / 'tables.md').read_text(encoding='utf-8'),
           'verdict': res['verdict'], 'note': '保存した予測から CPU で評価をやり直した結果と、保存済みの結果の比較(予測はやり直していない)'}
    with open(OUT / 'verify.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ('zoo_eval_diffs', 'tables_identical', 'seconds')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
