#!/usr/bin/env python3
"""V-drive: 既定値(DEFAULT_PARAMS・PARAM_LIMITS・RESET_ON_BOOT)と起動時の値を、変更前の木と今の木で別のプロセスで読んで比べる。
data/drive_settings.json(ステア 1.5・first_trial)をそのまま読ませ、起動直後の params が同じか(zone_caps を除く)・zone_caps 0 か。
zone_caps 1 を書いた設定ファイルでも起動で 0 か。ファイルは一時フォルダに写して使う(本物は読むだけ)。

    python -I v_defaults.py <木> <一時フォルダ>
"""
import json
import shutil
import sys
from pathlib import Path

tree, tmp = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(tree))
from jetformer import drive  # noqa: E402

assert Path(drive.__file__).resolve().parent.parent == tree.resolve()
tmp.mkdir(parents=True, exist_ok=True)
real = Path('/home/jetson/jetformer/data/drive_settings.json')
out = {'tree': str(tree), 'DEFAULT_PARAMS': drive.DEFAULT_PARAMS, 'PARAM_LIMITS': {k: list(v) for k, v in drive.PARAM_LIMITS.items()},
       'RESET_ON_BOOT': list(drive.RESET_ON_BOOT)}
p = tmp / 'drive_settings_copy.json'
shutil.copy(real, p)
drive.SETTINGS_PATH = p
d = drive.Driver(hub=None, recorder=None)
out['boot_params_real_file'] = d.params
out['boot_last'] = d.last
q = tmp / 'drive_settings_zone1.json'
js = json.loads(real.read_text())
js['params'].update(zone_caps=1, tcp_mode=2, speed_scale=1.5)
q.write_text(json.dumps(js))
drive.SETTINGS_PATH = q
out['boot_params_zone1_file'] = drive.Driver(hub=None, recorder=None).params
print(json.dumps(out, ensure_ascii=False))
