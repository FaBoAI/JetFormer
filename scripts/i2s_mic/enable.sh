#!/bin/bash
# JetFormer基板のI2Sマイク用に、40ピンヘッダのI2S2(pin12/35/38)を有効にする起動設定を追加する。
#   sudo ./enable.sh   → 再起動後から有効
# 今の起動エントリ(DEFAULT)をそのまま複製し、デバイスツリーだけ「I2S2有効版」に差し替えた
# エントリ jetformer-i2smic を作って DEFAULT にする。元のエントリは残るので、起動メニュー
# (シリアル/画面)から選べば元に戻せる。元に戻すときは sudo ./disable.sh
set -euo pipefail
cd "$(dirname "$0")"
CONF=/boot/extlinux/extlinux.conf
LABEL=jetformer-i2smic
[ "$(id -u)" = 0 ] || { echo "sudo で実行してください"; exit 1; }
dtc -@ -q -I dts -O dtb -o jetformer-i2s2-mic.dtbo jetformer-i2s2-mic.dts
python3 - "$CONF" "$LABEL" <<'PY'
import datetime, os, re, shutil, subprocess, sys
conf, label = sys.argv[1], sys.argv[2]
lines = open(conf).read().split('\n')
default = next((l.split()[1] for l in lines if re.match(r'^\s*DEFAULT\s', l)), None)
if default == label:
    sys.exit('すでに %s が既定です(変更なし)' % label)
blocks, cur = {}, None
for i, l in enumerate(lines):
    m = re.match(r'^\s*LABEL\s+(\S+)', l)
    if m:
        cur = m.group(1); blocks[cur] = [i, i]
    elif cur is not None:
        blocks[cur][1] = i
if default not in blocks:
    sys.exit('DEFAULT のエントリが見つかりません')
if label in blocks:
    sys.exit('%s のエントリが既にあります。disable.sh で外してから実行してください' % label)
s, e = blocks[default]
block = lines[s:e + 1]
while block and not block[-1].strip():
    block.pop()
fdt = next((l.split()[1] for l in block if re.match(r'^\s*FDT\s', l)), None)
if not fdt or not os.path.exists(fdt):
    sys.exit('FDT が見つかりません: %s' % fdt)
new_fdt = re.sub(r'\.dtb$', '', fdt) + '-i2smic.dtb'
subprocess.run(['fdtoverlay', '-i', fdt, '-o', new_fdt, 'jetformer-i2s2-mic.dtbo'], check=True)
out = subprocess.run(['fdtget', new_fdt, '/bus@0/pinmux@2430000/exp-header-pinmux/hdr40-pin12', 'nvidia,function'],
                     capture_output=True, text=True).stdout.strip()
if out != 'i2s2':
    sys.exit('作成したデバイスツリーの確認に失敗しました')
backup = conf + '.bak-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
shutil.copy2(conf, backup)
nb = []
for l in block:
    if re.match(r'^\s*LABEL\s', l):
        l = 'LABEL %s' % label
    elif re.match(r'^\s*MENU LABEL', l):
        l = '      MENU LABEL JetFormer I2S mic (40pin I2S2)'
    elif re.match(r'^\s*FDT\s', l):
        l = '      FDT %s' % new_fdt
    nb.append(l)
lines = [('DEFAULT %s' % label) if re.match(r'^\s*DEFAULT\s', l) else l for l in lines]
lines += [''] + nb + ['']
open(conf, 'w').write('\n'.join(lines))
open('/boot/extlinux/.jetformer-i2smic-previous-default', 'w').write(default + '\n')
print('起動設定を追加しました: DEFAULT=%s (元: %s、バックアップ: %s)' % (label, default, backup))
print('デバイスツリー: %s' % new_fdt)
PY
echo "再起動すると有効になります: sudo reboot"
