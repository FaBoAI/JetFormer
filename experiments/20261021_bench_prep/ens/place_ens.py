#!/usr/bin/env python3
"""段 S4(2026-10-21): 候補 ENS(2本の平均)を走行タブの一覧に置く。車は動かさない・PWM を出さない・装置に触れない。

10/15 に作った 1 つのエンジン(experiments/20261015_real_accuracy/ens/ens_p2_fp16.engine = realacc_20261015-v0-s2 と
realacc_20261015-msf-s2 の actions の平均。精度・速さのゲートに合格)を、data/runs/bench_20261021-ens-p2 に置く。
エンジンは作り直さない(写すだけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 2500 -- nice -n 10 \\
        /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261021_bench_prep/ens/place_ens.py all

  小コマンド(all は順に別のプロセスで。出来ているものは飛ばす = 電源断のあとはもう一度 all でよい):
    place   CPU(torch): エンジン・ONNX を写す、model_config.json・run.json・result.json を書く、2 本の best.pth を
            1 つ(キー ms.0.*・ms.1.*)にまとめた best.pth を書く。読み直して(runtime.load_run)10/15 の ref.npz の
            64 フレームで 2 本の平均と同じか(差 ≤ 0.01 µs)→ logs/place_check.json
    check   GPU: 10/13 の engine_load_check.py(走行タブと同じ手順・他の候補と同じ確認)→ logs/engine_load_check.json、
            それを data/runs/<ENS>/engine_check.json(drive.engine_check_of の形)に写す
    verify  torch なし: drive.model_info で drivable True・engine_check の accuracy_ok True・check_io の合格を確かめる
            → logs/verify.json(合格で終了コード 0)
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
LOGS = HERE / 'logs'
RUNS = REPO / 'data' / 'runs'
SRC = REPO / 'experiments' / '20261015_real_accuracy' / 'ens'
SRC_ENGINE, SRC_ONNX, SRC_CFG, SRC_REF, SRC_GATE = (SRC / 'ens_p2_fp16.engine', SRC / 'ens_p2.onnx', SRC / 'model_config.json',
                                                    SRC / 'ref.npz', SRC / 'gate.json')
NAME = 'bench_20261021-ens-p2'
DST = RUNS / NAME
MEMBERS = ('realacc_20261015-v0-s2', 'realacc_20261015-msf-s2')
ENGINE_FILE, ONNX_FILE = 'jetformer_fp16.engine', 'jetformer.onnx'
LOAD_CHECK = REPO / 'experiments' / '20261013_sim_cotrain' / 'train' / 'engine_load_check.py'
GATE_US = 2.5
REF_TOL_US = 0.01
NOTE = ('実車ベンチ(10/23)の候補 ENS(2本の平均): realacc_20261015-v0-s2(TransFuser軽量型)と realacc_20261015-msf-s2'
        '(TransFuser多段型)の予測の平均を 1 つにまとめたエンジン(10/15 に作成・ゲート合格)。机上の門とユーザーの了承の前は走らせない')
PY = sys.executable


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def write_json(path, obj):
    path = Path(path)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def copy_same(src, dst):
    """src を dst に写す(tmp → 置き換え・更新時刻も写す)。大きさと中身が同じなら何もしない → 写したか"""
    if dst.exists() and dst.stat().st_size == src.stat().st_size and sha256(dst) == sha256(src):
        return False
    tmp = Path(str(dst) + '.tmp')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


# ---------------------------------------------------------------- place(CPU・torch)
def cmd_place():
    import numpy as np
    import torch
    from jetformer.policies import ens_average, ens_config, runtime
    gate = read_json(SRC_GATE) or {}
    src_cfg = read_json(SRC_CFG) or {}
    if not gate.get('all_gates') or list(src_cfg.get('members') or []) != list(MEMBERS):
        raise SystemExit('10/15 の ENS のエンジンがゲートに合格していないか、メンバーが違います: %s' % SRC_GATE)
    if gate.get('engine') != str(SRC_ENGINE):
        raise SystemExit('gate.json のエンジンの場所が違います: %s' % gate.get('engine'))
    DST.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = {'started_at': now(), 'name': NAME, 'members': list(MEMBERS)}
    # 1. エンジン・ONNX(写すだけ。作り直さない)
    log['engine_copied'] = copy_same(SRC_ENGINE, DST / ENGINE_FILE)
    log['onnx_copied'] = copy_same(SRC_ONNX, DST / ONNX_FILE)
    log['engine_sha256'] = sha256(DST / ENGINE_FILE)
    # 2. model_config.json(入出力は 10/15 に ONNX から読んだもの = エンジンの元)
    io = {k: src_cfg[k] for k in ('inputs', 'input_shapes', 'outputs', 'output_shapes')}
    cfg = ens_config.make_config(MEMBERS, RUNS, io=io, engine=src_cfg.get('engine'), note=NOTE)
    cfg['source'] = {'engine': str(SRC_ENGINE), 'onnx': str(SRC_ONNX), 'gate': str(SRC_GATE),
                     'built_by': str(SRC / 'ens_gate.py')}
    if read_json(DST / 'model_config.json') != cfg:
        write_json(DST / 'model_config.json', cfg)
    # 3. best.pth(2 本の best.pth をまとめる。キーは ens_average.EnsAverage と ens_gate.make_average_module と同じ)
    pth = DST / 'best.pth'
    if not pth.exists():
        models = [runtime.load_run(RUNS / n, 'best', device='cpu')[0].float().eval() for n in MEMBERS]
        sd = ens_average.from_members(models).state_dict()
        tmp = str(pth) + '.tmp'
        torch.save(sd, tmp)
        os.replace(tmp, pth)
        log['best_pth_written'] = True
        del models
    # 4. 読み直して、10/15 の 2 本の平均(ref.npz)と同じか
    torch.set_num_threads(4)
    model, rcfg, spec, pcfg = runtime.load_run(DST, 'best', device='cpu')
    z = np.load(SRC_REF)
    frames = {k[3:]: z[k] for k in z.files if k.startswith('in_')}
    got = np.concatenate([runtime.predict(model, {k: v[s:s + 8] for k, v in frames.items()})['actions']
                          for s in range(0, len(z['avg']), 8)])
    diff = float(np.abs(got - z['avg']).max() * 500.0)
    log.update(policy=spec.name, policy_title=spec.title, frames=int(len(got)), ref_max_diff_us=round(diff, 5),
               ref_ok=bool(diff <= REF_TOL_US), params_m=runtime.params_m(model))
    # 5. run.json・result.json(学習タブ・走行タブの一覧用。学習はしていないので L1 は 10/15 の評価の値を参考に書く)
    ev = read_json(Path('/home/jetson/jetformer_reports/real_accuracy_20261015/evaluation.json')) or {}
    p2 = (ev.get('per_variant') or {}).get('P2') or {}
    run = read_json(DST / 'run.json') or {
        'name': NAME, 'created_at': now(), 'policy': 'ens_average', 'policy_cfg': cfg['policy_cfg'],
        'dataset': cfg.get('dataset'), 'stride': cfg.get('stride'), 'command_input': cfg.get('command_input'),
        'mask_impacts': cfg.get('mask_impacts'), 'build_engine': False, 'members': list(MEMBERS),
        'note': NOTE, 'experiment': 'bench_prep_20261021', 'arm': 'ENS',
        'command': 'experiments/20261021_bench_prep/ens/place_ens.py all'}
    if not (DST / 'run.json').exists():
        write_json(DST / 'run.json', run)
    res = {'state': 'done', 'dataset': cfg.get('dataset'), 'stride': cfg.get('stride'), 'policy': 'ens_average',
           'policy_cfg': cfg['policy_cfg'], 'command_input': cfg.get('command_input'),
           'steer_center_us': cfg.get('steer_center_us'), 'throttle_neutral_us': cfg.get('throttle_neutral_us'),
           'best_val_us': None, 'members': list(MEMBERS), 'params_m': log['params_m'],
           'holdout_l1_us': {'best': (p2.get('best') or {}).get('primary'), 'last': (p2.get('last') or {}).get('primary'),
                             'source': '/home/jetson/jetformer_reports/real_accuracy_20261015/evaluation.json per_variant.P2'
                                       '(保留の 4,670 フレーム・10/13 の規則。エンジンは best の 2 本の平均)'},
           'note': '学習はしていない(2 本の学習結果の平均)。best_val_us は学習の検証誤差の欄なので空のまま'}
    if read_json(DST / 'result.json') != res:
        write_json(DST / 'result.json', res)
    log['finished_at'] = now()
    write_json(LOGS / 'place_check.json', log)
    print(json.dumps(log, ensure_ascii=False), flush=True)
    if not log['ref_ok']:
        raise SystemExit('まとめた best.pth の出力が 10/15 の 2 本の平均と合いません(最大 %.4f µs)' % diff)


# ---------------------------------------------------------------- check(GPU)
def engine_check_from(load, engine_path):
    """engine_load_check.py の結果 → drive.engine_check_of の形(write_engine_check.py と同じ欄)"""
    acc, lat = load.get('accuracy') or {}, load.get('latency_ms') or {}
    st = engine_path.stat()
    return {
        'ok': bool(load.get('all_gates')), 'accuracy_ok': bool(acc.get('gate_actions_2p5us') and acc.get('finite')),
        'speed_ok': bool(lat.get('gate_run_numpy')), 'actions_max_us': acc.get('actions_max_us'),
        'actions_p99_us': acc.get('actions_p99_us'), 'gate_us': GATE_US,
        'run_numpy_median_ms': lat.get('run_numpy_median'), 'run_numpy_p99_ms': lat.get('run_numpy_p99'),
        'checked_at': load['checked_at'], 'note': NOTE,
        'engine': {'file': ENGINE_FILE, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
        'source': str(LOGS / 'engine_load_check.json'),
        'method': '実フレーム 64 個(clean_driving_20261011 の val から等間隔)で TensorRT(run_numpy)と PyTorch FP32'
                  '(CPU・まとめた best.pth = 2 本の best.pth の平均)の actions の差。速さは 100 回の空回しの後 600 回'
                  '(experiments/20261013_sim_cotrain/train/engine_load_check.py)'}


def cmd_check():
    eng = DST / ENGINE_FILE
    out = LOGS / 'engine_load_check.json'
    dst = DST / 'engine_check.json'
    import datetime
    load = read_json(out)
    fresh = False
    if load and load.get('checked_at'):
        checked = datetime.datetime.strptime(load['checked_at'], '%Y-%m-%d %H:%M:%S').timestamp()
        fresh = eng.stat().st_mtime <= checked and (DST / 'best.pth').stat().st_mtime <= checked
    if not fresh:
        r = subprocess.run([PY, '-u', str(LOAD_CHECK), str(DST), str(out)], cwd=str(REPO))
        if r.returncode != 0:
            raise SystemExit('engine_load_check.py が失敗しました(終了コード %d)' % r.returncode)
        load = read_json(out)
    else:
        print('check: 確認済み(%s)' % load['checked_at'])
    ec = engine_check_from(load, eng)
    if read_json(dst) != ec:
        write_json(dst, ec)
    print(json.dumps({k: ec[k] for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'run_numpy_median_ms',
                                          'run_numpy_p99_ms')}, ensure_ascii=False), flush=True)
    if not ec['ok']:
        raise SystemExit('エンジンの確認に落ちました: %s' % out)


# ---------------------------------------------------------------- verify(torch なし)
def cmd_verify():
    from jetformer import drive, policies
    info = drive.model_info(NAME)
    load = read_json(LOGS / 'engine_load_check.json') or {}
    cio = load.get('check_io') or {}
    res = {'checked_at': now(), 'name': NAME, 'model_info': None, 'torch_loaded': 'torch' in sys.modules}
    if info is not None:
        res['model_info'] = {k: info.get(k) for k in ('policy', 'policy_title', 'drivable', 'why_not', 'trained_modes',
                                                      'steer_center_us', 'throttle_neutral_us', 'command_input',
                                                      'engine_precision', 'engine_check')}
        res['in_list_models'] = any(m['name'] == NAME for m in drive.list_models())
    res['check_io'] = {'ok': cio.get('ok'), 'why': cio.get('why')}
    res['registered'] = policies.get('ens_average').name == 'ens_average'
    res['not_in_training_choices'] = 'ens_average' not in policies.names()
    res['ok'] = bool(info and info['drivable'] and (info.get('engine_check') or {}).get('accuracy_ok') is True
                     and cio.get('ok') is True and res['registered'] and res['not_in_training_choices']
                     and res.get('in_list_models') and not res['torch_loaded'])
    write_json(LOGS / 'verify.json', res)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if not res['ok']:
        raise SystemExit('確かめに落ちました: %s' % (LOGS / 'verify.json'))


def cmd_all():
    for sub, env in (('place', {'CUDA_VISIBLE_DEVICES': ''}), ('check', {}), ('verify', {'CUDA_VISIBLE_DEVICES': ''})):
        r = subprocess.run([PY, '-u', __file__, sub], cwd=str(REPO), env=dict(os.environ, **env))
        if r.returncode != 0:
            raise SystemExit('%s が失敗しました(終了コード %d)' % (sub, r.returncode))


def main():
    cmds = {'place': cmd_place, 'check': cmd_check, 'verify': cmd_verify, 'all': cmd_all}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]]()


if __name__ == '__main__':
    main()
