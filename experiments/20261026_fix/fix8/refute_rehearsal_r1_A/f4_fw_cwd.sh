#!/bin/bash
# 指摘 rehearsal-fw-cwd の確かめ(打ち消しの票 A)。手順書 3 章と todo 1 章の文をそのまま抜き出し、作り物の家(~)と ~/jetformer で流す。
# fw_vbat_check.py は GET だけ(装置は開かない)。--seconds 1 で短く。
set -u
HERE=/home/jetson/jetformer/experiments/20261026_fix/fix8/refute_rehearsal_r1_A
RB=/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md
TD=/home/jetson/jetformer_reports/fix_20261026/todo_user_1026.md
FAKE=$HERE/tmp_f4/home; rm -rf "$HERE/tmp_f4"; mkdir -p "$FAKE"
CMD=$(grep -n '電圧: ファームがあれば' "$RB" | head -1 | sed -E 's/^([0-9]+):.*`(python3 tools\/fw_vbat_check.py [^`]*)`.*/\1|\2/')
echo "手順書の行と文: $CMD"
CMD_TODO=$(grep -n 'FaBo のファームを書き込む' "$TD" | head -1 | sed -E 's/^([0-9]+):.*`(python3 tools\/fw_vbat_check.py [^`]*)`.*/\1|\2/')
echo "todo の行と文: $CMD_TODO"
RUN=$(echo "${CMD#*|}" | sed 's/<LiPo チェッカーの値>/8.30/')
echo "流す文(値を入れた): $RUN --seconds 1"
echo "--- 3 章でこの行より前にある cd(3 章の頭 〜 この行)"
S=$(grep -n '^## 3\. 朝の準備' "$RB" | cut -d: -f1); E=${CMD%%|*}
sed -n "${S},${E}p" "$RB" | grep -o '`[^`]*cd [^`]*`' || echo "(cd を含むコマンドは無い)"
echo "--- 作り物の家(HOME=$FAKE)の ~ で"
( cd "$FAKE" && HOME="$FAKE" bash -c "$RUN --seconds 1"; echo "終了コード $?" ) 2>&1 | sed "s#$FAKE#<作り物の家>#g"
echo "--- ~/jetformer で(本物のサーバーへ GET だけ)"
( cd /home/jetson/jetformer && PYTHONDONTWRITEBYTECODE=1 bash -c "$RUN --seconds 1" | tail -6; echo "終了コード ${PIPESTATUS[0]}" ) 2>&1
echo "--- 8.2 の ⑦ の塊の頭"
grep -n -B6 'python3 tools/fw_vbat_check.py --seconds 10' "$RB" | grep -E 'cd ~/jetformer|fw_vbat'
rm -rf "$HERE/tmp_f4"
