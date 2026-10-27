#!/bin/bash
# 反証 7-1・docs F3・F5(読むだけ)
cd /home/jetson/jetformer
echo "== F3: e5_open_seal.py plan(既定 = 読むだけ)の答え"
/home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix3/e5_open_seal.py plan 2>&1 | grep -E '"V1"|"V2"|"final"|"open"|答え'
echo "== F3: 文書の approvals 6(候補 0 のとき封を開けるか)を「答えがまだ」に数えている行"
grep -n "答えがまだなのは\|^| 6 |" /home/jetson/jetformer_reports/fix_20261026/fix3/approvals.md | cut -c1-140
grep -n "4・6〜11・13\|候補が 0 のとき封を開けるか" /home/jetson/jetformer_reports/fix_20261026/todo_user_1026.md | cut -c1-140
echo "== F5: 10/27 の pytest 一式の終わりの空きと作業場所"
tail -4 /home/jetson/jetformer_reports/fix_20261026/fix6/pytest_full_20261027.txt
echo "== F5: 文書と disk_check の数"
grep -n "3.9 GB" /home/jetson/jetformer_reports/fix_20261026/README.md /home/jetson/jetformer_reports/fix_20261026/fix3/approvals.md | cut -c1-160
grep -n "PYTEST_GB = " experiments/20261026_fix/fix3/disk/disk_check.py
timeout 120 python3 -I experiments/20261026_fix/fix3/disk/disk_check.py 2>&1 | grep -E "今の空き|pytest 一式の最中"
