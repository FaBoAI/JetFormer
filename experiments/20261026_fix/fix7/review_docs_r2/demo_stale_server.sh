#!/bin/bash
# 反証 7-2・docs: 手順書 3.1 の表(と todo 4 章の 3・4)で、今のサーバーが「ディスクの今のコード」で起きているかを見分けられるか(読むだけ。GET と systemctl show・stat だけ)
echo "## todo 4 章 4 の GET(文書のコマンドそのまま)"
curl -s http://127.0.0.1:8080/api/drive/status | python3 -c "import json,sys; s=json.load(sys.stdin); p=s['params']; print({k: p.get(k) for k in ('launch_boost','rec_back_guard','rec_hold_keep','launch_no_kick','speed_ceiling','speed_auto','zone_caps','ceil_left_mps','launch_us')}, 'boot_at', s.get('boot_at'))"
echo "## 3.1 の 11/3 の朝の journal の形"
systemctl is-active jetformer-web; systemctl show -p NRestarts --value jetformer-web; journalctl -u jetformer-web -b --no-pager | grep -n "Traceback\|Error" || echo "エラーの行なし"
echo "## 起動の時刻とコードの時刻"
B=$(curl -s http://127.0.0.1:8080/api/drive/status | python3 -c "import json,sys; print(int(json.load(sys.stdin)['boot_at']))")
echo "boot_at $B ($(date -d @$B '+%m/%d %H:%M:%S'))  ActiveEnterTimestamp $(systemctl show -p ActiveEnterTimestamp --value jetformer-web)"
for f in jetformer/drive.py jetformer/assist.py jetformer/web/server.py; do
  m=$(stat -c %Y /home/jetson/jetformer/$f); echo "$f 更新 $(date -d @$m '+%m/%d %H:%M:%S')  起動より $([ $m -gt $B ] && echo 新しい = 動いていないコード || echo 古い)"
done
echo "## fix7 安全の直しの印(ディスク)"
grep -c "class EscMayReverse\|back_no_esc\|early_none" /home/jetson/jetformer/jetformer/assist.py
echo "## Jetson の今の起動(-b の起点)"
journalctl --list-boots --no-pager | tail -1
