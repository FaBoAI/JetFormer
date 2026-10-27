"""反証 7-1・docs F1: 手順書 3 章の「その日の最初のリカバリーがおかしければ、止めて OFF にし、その本から先は OFF」(11/3 の (c) の
決め「おかしければ止めて OFF」)をしたあとの本を、② runs_1103 がどう出すか、本物の runs_1103.py の関数で見る(作り物の行だけ。
data/raw は読まない・書かない)。手順書 8.1 ② の読み方(「回 N の試験の選択肢が … と違う」の印は起こし直しで OFF に戻った本 → 7 章)
・4 章 6・4 章 10 の表と並べる。

    cd ~/jetformer && python3 -I experiments/20261026_fix/fix7/review_docs_r1/demo_f1_off_on_purpose.py
"""
import importlib.util

P = '/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py'
spec = importlib.util.spec_from_file_location('runs_1103_demo', P)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

ON = {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
OFF = {'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}
# 11/3: 朝に両方 ON。1-2 の最初のリカバリーで 1.2 m 下がった → 手順書 3 章どおり人側 → 止めて OFF → 1-3 から OFF(紙に書いた)
plan = [('1-1', 2, ON), ('1-2', 3, ON), ('1-3', 4, OFF), ('1-4', 5, OFF), ('1-5', 6, OFF), ('2-1', 7, OFF)]
rows = [{'log': '20261103-1%05d' % i, 'run_id': rid, 'ai_no': str(no), 'options': dict(o), 'diffs': []}
        for i, (rid, no, o) in enumerate(plan)]
R.round_options(rows)
for r in rows:
    print('%s No.%s 計画との違い: %s' % (r['run_id'], r['ai_no'], '・'.join(r['diffs']) or 'なし'))
print()
print(R.options_line(rows))
