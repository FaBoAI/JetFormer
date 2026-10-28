#!/bin/bash
# 反証 8-1 rehearsal: 手順書 3.1 と todo 4 章の systemctl・journalctl のコマンドを、作り物の systemctl・journalctl(fakebin)で流す。
# 本物の systemd・journal には触れない。コマンドの文は手順書と todo から抜き出す(手で写さない)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RB=/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md
TD=/home/jetson/jetformer_reports/fix_20261026/todo_user_1026.md
export PATH="$HERE/fakebin:$PATH"
export TZ=Asia/Tokyo
J=$(mktemp); export FAKE_JLOG=$(mktemp)
# 手順書 3.1 の文中の 11/3 の朝のコマンド(`systemctl is-active …` から「エラーの行なし"」まで)
MORNING=$(python3 -I -c "import re,sys; s=open(sys.argv[1],encoding='utf-8').read(); print(re.search(r'\`(systemctl is-active jetformer-web;[^\`]*)\`', s).group(1))" "$RB")
NEWER=$(grep -A1 '^# 起動のあとに直したコード' "$RB" | tail -1)
echo "== 手順書 3.1 の朝のコマンド: $MORNING"
printf 'Nov 03 08:01:02 jetson python[901]: INFO started\n' > "$J"; export FAKE_JOURNAL="$J"
bash -c "$MORNING"
printf 'Nov 03 08:01:02 jetson python[901]: Traceback (most recent call last):\nNov 03 08:01:02 jetson python[901]: RuntimeError: x\n' > "$J"
echo "-- (journal に Traceback がある形)"; bash -c "$MORNING"
echo "== 手順書 3.1 の「起動のあとに直したコード」: $NEWER"
for aet in "Wed 2026-10-28 00:53:23 JST" "Wed 2026-10-28 06:00:00 JST" ""; do
  echo "-- ActiveEnterTimestamp='$aet'"; FAKE_AET="$aet" bash -c "$NEWER"
done
echo "== todo 4 章の 3(--since)"
T3=$(grep '^sleep 10; systemctl is-active' "$TD" | sed 's/^sleep 10; //')
printf 'Nov 03 08:01:02 jetson python[901]: INFO started\n' > "$J"; bash -c "$T3"; echo "journalctl の引数: $(tail -1 "$FAKE_JLOG")"
rm -f "$J" "$FAKE_JLOG"
