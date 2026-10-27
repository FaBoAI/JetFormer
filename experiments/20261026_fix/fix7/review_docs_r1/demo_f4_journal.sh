#!/bin/bash
# 反証 7-1・docs F4: todo_user_1026.md 4 章の「3. 起きたか・エラーが無いか」のコマンド(手順書 3.1 が 11/3 の朝にもう一度と指す)を
# そのまま流す(読むだけ)。今のサーバー(10/27 17:16:28 起動)にエラーが無くても、-n 60 が 10/10 まで届いて古い Traceback を出すかを見る
echo "== 文書のコマンドそのまま"
systemctl is-active jetformer-web; journalctl -u jetformer-web -n 60 --no-pager | grep -n "Traceback\|Error" || echo "エラーの行なし"
echo "== 60 行の最初と最後の日付"
journalctl -u jetformer-web -n 60 --no-pager | grep -v '^--' | sed -n '1p;$p' | cut -c1-120
echo "== 今の起動(ActiveEnterTimestamp)からだけ見たとき"
S=$(systemctl show -p ActiveEnterTimestamp --value jetformer-web); echo "起動 $S"
journalctl -u jetformer-web --since "$S" --no-pager | grep -n "Traceback\|Error" || echo "エラーの行なし"
