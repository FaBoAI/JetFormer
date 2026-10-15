#!/usr/bin/env python3
"""準備の段のまとめ: prepare_check.py の結果(prepare_check.json)に、CPU の動作確認(smoke/<腕>.json)・道具の sha256・
ディスクの空き・腕のコマンドを足して書き直す(書き終えてから置き換える)

    python3 -I finalize_prepare.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
OUT = REPORT / 'prepare_check.json'
TOOLS = ['make_combined.py', 'build_arms.sh', 'prepare_check.py', 'diverse_train.py', 'nothp.py', 'export_bundle.py',
         'check_bundle.py', 'http_load_check.py', 'finalize_prepare.py']
ARMS = ['S-DIV', 'S-MOD', 'C-DIV']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    d = json.loads(OUT.read_text(encoding='utf-8'))
    smoke = {}
    for arm in ARMS:
        p = REPORT / 'smoke' / ('%s.json' % arm)
        s = json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
        smoke[arm] = None if s is None else {
            'ok': s['ok'], 'kill_after_epoch1': s['kill_after_epoch1'], 'started': s['started'], 'ended': s['ended'],
            'real_command': s['real_command'], 'smoke_overrides': s['smoke_overrides'],
            'phases': [{k: ph[k] for k in ('cmd_extra', 'exit', 'seconds', 'killed', 'last_epoch_after')} for ph in s['phases']],
            'result': s['result'], 'model_config': s['model_config'], 'onnx_bytes': s['onnx_bytes']}
    d['smoke'] = smoke
    d['smoke_note'] = ('CPU(CUDA_VISIBLE_DEVICES 空)・jf-heavy --mem 1500M --lock none で、本番と同じ train.py の引数のうち '
                       '--epochs 2・--batch 8・--workers 0 に替え --max-steps 2 を足したもの。出力は smoke/runs(data/runs に作らない。'
                       '終わったら消した)。S-DIV と C-DIV は 1 エポック目の last.pth のあとで kill -9(電源断のまね)→ 同じコマンドに '
                       '--resume で 2 エポック目を続けた')
    d['tools_sha256'] = {t: sha256(HERE / t) for t in TOOLS if (HERE / t).exists()}
    du = shutil.disk_usage('/home/jetson')
    d['disk_free_gb_at_finalize'] = round(du.free / 2 ** 30, 2)
    d['finalized_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    dry = subprocess.run(['/home/jetson/miniconda/envs/jetracer/bin/python', str(HERE / 'diverse_train.py'), 'dryrun',
                          'S-DIV/s0', 'S-MOD/s0', 'C-DIV/s0', 'C-DIV/s1'], capture_output=True, text=True, cwd='/home/jetson/jetformer')
    d['train_commands'] = dry.stdout.strip().splitlines()
    d['ok']['smoke_all_arms'] = all(v and v['ok'] for v in smoke.values())
    d['all_ok'] = bool(all(d['ok'].values()))
    tmp = Path(str(OUT) + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, OUT)
    print(json.dumps({'all_ok': d['all_ok'], 'ok': d['ok'], 'disk_free_gb': d['disk_free_gb_at_finalize']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
