#!/bin/bash
# 反証 7-2・docs: 手順書 8.1 のコマンドの塊を、① が落ちた(手順書は「① が落ちても ②〜⑤ は回せる」と書く)ときに上から順に流すと
# どうなるかを見る。python3 は「① は終了コード 1・ほかは開くファイルがあるかだけ見る」作り物の関数に置き換える(本物の道具は流さない・
# 何も書かない)。cd と && の動きは本物の bash。mkdir -p $R も流さない(作業場所の外に作らないため)。
python3() {
  if [[ "$*" == *quick_eval.py* ]]; then echo "  (① quick_eval が落ちた = 終了コード 1)"; return 1; fi
  local f="$2"
  if [ -f "$f" ]; then echo "  ok: $f を開けた(cwd $PWD)"; else echo "  python3: can't open file '$PWD/$f': [Errno 2] No such file or directory"; return 2; fi
}
cd ~/jetformer
R=/nonexistent_demo
echo "# ①"; cd experiments/20261024_tune && python3 -I quick_eval.py 0000 --day=20261103 && cd ~/jetformer
echo "# ②"; python3 -I experiments/20261026_fix/plan/runs_1103.py
echo "# ③"; python3 -I tools/standing_3lap.py --day 20261103 --md $R/standing_3lap.md --json $R/standing_3lap.json
echo "# ④"; python3 -I tools/quick_eval_by_params.py 0000 --day=20261103 --out $R/by_params
echo "# ⑤"; cd experiments/20261026_fix/diag_launch && echo "  (ここで jf-heavy … collect.py)" && cd ~/jetformer
echo "cwd の最後: $PWD"
