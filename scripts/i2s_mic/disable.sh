#!/bin/bash
# enable.sh で追加した起動エントリを外し、元の DEFAULT に戻す(sudo ./disable.sh → 再起動)
set -euo pipefail
CONF=/boot/extlinux/extlinux.conf
LABEL=jetformer-i2smic
PREV=/boot/extlinux/.jetformer-i2smic-previous-default
[ "$(id -u)" = 0 ] || { echo "sudo で実行してください"; exit 1; }
[ -f "$PREV" ] || { echo "enable.sh で追加した設定がありません"; exit 1; }
python3 - "$CONF" "$LABEL" "$(cat "$PREV")" <<'PY'
import datetime, re, shutil, sys
conf, label, prev = sys.argv[1:4]
lines = open(conf).read().split('\n')
shutil.copy2(conf, conf + '.bak-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
out, skip = [], False
for l in lines:
    m = re.match(r'^\s*LABEL\s+(\S+)', l)
    if m:
        skip = m.group(1) == label
    if re.match(r'^\s*DEFAULT\s', l):
        l = 'DEFAULT %s' % prev
    if not skip:
        out.append(l)
open(conf, 'w').write('\n'.join(out).rstrip('\n') + '\n')
print('元に戻しました: DEFAULT=%s' % prev)
PY
rm -f "$PREV"
echo "再起動すると元の設定で起動します: sudo reboot"
