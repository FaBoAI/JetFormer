#!/usr/bin/env python3
"""学習結果 1 本 → シミュレーターのブリッジ用の一式(bundle)を書き出す(CPU。data/runs の元のファイルは変えない)

    cd ~/jetformer && python -u experiments/20261014_sim_dart_margin/train/export_bundle.py <data/runs/学習結果> <一式のフォルダー>

sim_dart_20261014 の写し(元: experiments/20261013_sim_rl_bestlap/train/export_bundle.py)。この作業の protocol.json の eval.bundle は
「ベストラップの export_bundle.py(写し)で ONNX・model_config.json・io_spec.json の ZIP。中心 1442 / 1498 µs、遅延 1・先読み 1・
ゲイン 1.0・ジャンル 8」なので、直す欄はベストラップの protocol.json の eval_bundle をそのまま使う(sha256 を固定して読むだけ)。
元からの変更は 3 つ: (1) パス(この作業の報告のフォルダー)、(2) 2 つのデータセットを合わせた腕(A1)は model_config の dataset が
フォルダーの名前(A1)になるので、参照の出力の val は run.json の data_path から読む、(3) README と simulator_bundle の作業名。
ベストラップの protocol.json の eval_bundle / design.md 6.1 節のとおり:
  - jetformer.onnx      学習結果の ONNX をそのまま写す(sha256 が同じことを確かめる)
  - model_config.json   学習結果の model_config に次を足す/直す
                          steer_center_us 1442・throttle_neutral_us 1498(記録の PWM の換算と同じ。学習が出す値は
                          「止まっているフレームの中央値」で、シミュレーターのデータでは中心と違う)
                          作者の shortcut-rc-finetuned と同じ simulator_physics・simulator_sensors・simulator_world_version・
                          simulator_sensor_profile・evaluation_route・required_route_event
                          simulator_execution {control_delay_frames: 1}
                          simulator_calibration(作者の値で mode_id 8・steer_gain 1.0・throttle_gain 1.0・lookahead 1)
                          recommended_inference {steering_gain 1.0, throttle_gain 1.0, lookahead 1}
                          simulator_bundle(元の学習結果・直した欄・学習が出した中心値の記録。アダプターは読まない)
  - io_spec.json        jetformer.model_export.io_spec を、直した model_config で作る(アダプターは埋め込みの model_config と
                        model_config.json が同じことを確かめる)
  - calibration.json    ブリッジが一式の既定として使うのと同じ較正(参考。ZIP にも入れるがブリッジは読まない)
  - README.md           model_export.readme + この一式の注意
  - <一式のフォルダー>.zip  <名前>/ の下に上の 5 つ(ブリッジの /api/models/bundle にそのまま渡せる形)
  - export.json         各ファイルの sha256・直した欄・ジャンルの数
さらに、確かめ用に PyTorch(CPU・best.pth)の出力を、モデル自身のデータの val から等間隔の 16 フレームで計算して
<一式のフォルダー>/../_check/<名前>_torch_ref.npz に保存する(check_bundle.py が ONNX の出力と比べる)。
"""

import datetime
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
PROTOCOL = REPORT / 'protocol.json'                     # この作業の決まり(sha256 を記録するだけ)
BESTLAP_PROTOCOL = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/protocol.json')   # 直す欄(読むだけ)
BESTLAP_PROTOCOL_SHA256 = '1f249211704c0065b87645d5bf93d959fcd65205fd4c2f4e535a1c310e81b8b6'
EXPERIMENT = 'sim_dart_20261014'
DATASETS = REPO / 'data' / 'datasets'
N_REF = 16
SIM_MODE = 8
STEER_CENTER_US = 1442.0
THROTTLE_NEUTRAL_US = 1498.0
BUNDLE_FILES = ('jetformer.onnx', 'model_config.json', 'io_spec.json', 'calibration.json', 'README.md')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def external_reference(proto):
    """作者の shortcut-rc-finetuned の model_config(sha256 を protocol と比べてから読む)"""
    ref = proto['files']['external_ref_model_config']
    if sha256(ref['path']) != ref['sha256']:
        raise SystemExit('外部の参照の model_config.json の sha256 が protocol と違います: %s' % ref['path'])
    return read_json(ref['path'])


def patched_config(cfg, ext, proto, run_name, src_sha):
    """学習結果の model_config → ブリッジ用(protocol.json eval_bundle.model_config_patch)"""
    eb = proto['eval_bundle']['model_config_patch']
    out = json.loads(json.dumps(cfg))
    if int(out['num_modes']) > 9 or SIM_MODE not in out['trained_modes']:
        raise SystemExit('ジャンルが合いません: num_modes %s・trained_modes %s(ブリッジは 0〜8・ジャンル 8 を使う)' % (
            out['num_modes'], out['trained_modes']))
    trained_center = (out.get('steer_center_us'), out.get('throttle_neutral_us'))
    out['steer_center_us'] = float(eb['steer_center_us'])
    out['throttle_neutral_us'] = float(eb['throttle_neutral_us'])
    if (out['steer_center_us'], out['throttle_neutral_us']) != (STEER_CENTER_US, THROTTLE_NEUTRAL_US):
        raise SystemExit('protocol の中心値が想定と違います')
    for key in eb['copy_from_external_ref']:
        out[key] = json.loads(json.dumps(ext[key]))
    out['simulator_execution'] = dict(eb['simulator_execution'])
    cal = json.loads(json.dumps(ext['simulator_calibration']))
    cal.update({'mode_id': SIM_MODE, 'steer_gain': 1.0, 'throttle_gain': 1.0, 'lookahead': 1})
    out['simulator_calibration'] = cal
    out['recommended_inference'] = dict(eb['recommended_inference'])
    out['simulator_bundle'] = {
        'experiment': EXPERIMENT,
        'source_run': run_name,
        'source_model_config_sha256': src_sha,
        'patched_keys': ['steer_center_us', 'throttle_neutral_us'] + list(eb['copy_from_external_ref'])
                        + ['simulator_execution', 'simulator_calibration', 'recommended_inference'],
        'trained_steer_center_us': trained_center[0],
        'trained_throttle_neutral_us': trained_center[1],
        'note': 'シミュレーター専用(ジャンル 8)。実車では使わない。中心値は記録の PWM の換算(1442 / 1498 µs)に直した',
        'protocol_sha256': sha256(PROTOCOL),
        'patch_from': {'protocol': str(BESTLAP_PROTOCOL), 'sha256': BESTLAP_PROTOCOL_SHA256, 'key': 'eval_bundle.model_config_patch'},
    }
    return out, trained_center


def dataset_dir(run_dir, cfg):
    """学習に使ったデータのフォルダー。data/datasets/<名前> が無ければ(合わせた腕)run.json の data_path"""
    d = DATASETS / cfg['dataset']
    if (d / 'splits.json').exists():
        return d
    run = read_json(Path(run_dir) / 'run.json')
    d = REPO / run['data_path']
    if not (d / 'splits.json').exists() or d.name != cfg['dataset']:
        raise SystemExit('学習のデータのフォルダーが見つかりません: %s' % d)
    return d


def torch_reference(run_dir, cfg, out_npz):
    """モデル自身のデータの val から等間隔に 16 フレーム(衝突 2.5G の前後を除く・学習と同じ正規化)と PyTorch FP32(CPU)の出力"""
    import torch
    from jetformer.train_data import JetFormerDataset
    from jetformer.policies import runtime
    torch.set_num_threads(2)
    ddir = dataset_dir(run_dir, cfg)
    ds = JetFormerDataset(str(ddir), 'val', stride=1, use_command=cfg.get('command_input', True),
                          mask_impacts=True, impact_g=2.5)
    idx = np.linspace(0, len(ds) - 1, N_REF).round().astype(int)
    cols = [ds[int(i)] for i in idx]
    names = ('image', 'tof', 'imu', 'state', 'mode')
    frames = {n: np.stack([c[k].numpy() for c in cols]).astype(np.float32) for k, n in enumerate(names)}
    target = np.stack([c[5].numpy() for c in cols]).astype(np.float32)
    if frames['mode'].shape[1] != cfg['num_modes'] or not (frames['mode'].argmax(1) == SIM_MODE).all():
        raise SystemExit('val のジャンルが合いません: %s' % sorted(set(frames['mode'].argmax(1).tolist())))
    model, _, _, _ = runtime.load_run(run_dir, 'best', device='cpu')
    acts = []
    for s in range(0, N_REF, 4):
        o = runtime.predict(model, {k: v[s:s + 4] for k, v in frames.items()})
        acts.append(o['actions'])
    actions = np.concatenate(acts).astype(np.float32)
    with open(ddir / 'splits.json', encoding='utf-8') as f:
        val_names = json.load(f)['val']
    picks = np.array([[val_names[ds.index[int(i)][0]], str(int(ds.index[int(i)][1]))] for i in idx])
    tmp = str(out_npz) + '.tmp.npz'
    np.savez_compressed(tmp, actions=actions, target=target, picks=picks, **frames)
    os.replace(tmp, out_npz)
    l1 = float(np.abs(actions - target).mean() * 500.0)
    return {'frames': N_REF, 'dataset': cfg['dataset'], 'dataset_dir': str(ddir), 'val_frames': len(ds), 'torch_l1_vs_label_us': round(l1, 2),
            'npz': str(out_npz), 'npz_sha256': sha256(out_npz)}


def main(run_dir, bundle_dir):
    from jetformer import model_export
    run_dir, bundle_dir = Path(run_dir).resolve(), Path(bundle_dir).resolve()
    name = run_dir.name
    if bundle_dir.name != name:
        raise SystemExit('一式のフォルダーの名前は学習結果と同じにします: %s / %s' % (bundle_dir.name, name))
    if sha256(BESTLAP_PROTOCOL) != BESTLAP_PROTOCOL_SHA256:
        raise SystemExit('ベストラップの protocol.json の sha256 が固定したものと違います')
    proto = read_json(BESTLAP_PROTOCOL)
    ext = external_reference(proto)
    cfg = read_json(run_dir / 'model_config.json')
    src_cfg_sha = sha256(run_dir / 'model_config.json')
    staging = bundle_dir.parent / '_staging'
    work = staging / name                      # model_export.io_spec は <RUNS_DIR>/<名前>/ を読むので、名前のフォルダーで作る
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # ---- ONNX(写して sha256 を確かめる)
    shutil.copyfile(run_dir / 'jetformer.onnx', work / 'jetformer.onnx')
    onnx_sha = sha256(run_dir / 'jetformer.onnx')
    if sha256(work / 'jetformer.onnx') != onnx_sha:
        raise SystemExit('ONNX の写しの sha256 が違います')
    # ---- model_config(直したもの)
    pc, trained_center = patched_config(cfg, ext, proto, name, src_cfg_sha)
    write_json(work / 'model_config.json', pc)
    # ---- io_spec(直した model_config から。model_export の RUNS_DIR を作業場所に向ける)
    saved = model_export.RUNS_DIR
    model_export.RUNS_DIR = staging
    try:
        spec = model_export.io_spec(name)
        readme = model_export.readme(name, spec)
    finally:
        model_export.RUNS_DIR = saved
    if spec['model_config'] != pc:
        raise SystemExit('io_spec の埋め込みの model_config が直したものと違います')
    write_json(work / 'io_spec.json', spec)
    # ---- 較正(ブリッジが一式の既定に使うのと同じ: simulator_calibration + recommended_inference)
    cal = dict(pc['simulator_calibration'])
    cal.update({'steer_gain': pc['recommended_inference']['steering_gain'],
                'throttle_gain': pc['recommended_inference']['throttle_gain'],
                'lookahead': pc['recommended_inference']['lookahead']})
    write_json(work / 'calibration.json', cal)
    note = '''
## この一式について(sim_dart_20261014)
- シミュレーター専用(ジャンル 8)のモデルです。実車では使いません。
- 元の学習結果: `data/runs/%s`(この一式のために元のファイルは変えていません)
- `model_config.json` は、ブリッジで走らせるために次を足し/直しました(ベストラップの protocol.json の eval_bundle と同じ):
  中心値 1442 / 1498 µs(学習が出した値は %s / %s µs)、作者の `shortcut-rc-finetuned` と同じ物理・センサーの欄、
  制御の遅れ 1 フレーム、ジャンル 8・ゲイン 1.0・先読み 1。
- 走行モード(ジャンル)は 8 です。ブリッジは長さ %d の one-hot の 8 番目を 1 にして渡します。
- ZIP(`%s.zip`)をシミュレーターの「モデルの一式」から読み込めます(ブリッジの /api/models/bundle)。
''' % (name, trained_center[0], trained_center[1], pc['num_modes'], name)
    (work / 'README.md').write_text(readme + note, encoding='utf-8')
    # 前の一式と入れ替える(途中で止まっても、半端な一式を正式の名前で残さない)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    os.replace(work, bundle_dir)
    # ---- ZIP(<名前>/ の下。ブリッジは同じフォルダーの ONNX・model_config・io_spec を使い、ほかは無視する)
    zpath = bundle_dir.with_name(name + '.zip')
    ztmp = bundle_dir.with_name(name + '.zip.tmp')
    stamp = datetime.datetime.now().timetuple()[:6]
    with zipfile.ZipFile(ztmp, 'w', allowZip64=True) as z:
        for fn in BUNDLE_FILES:
            info = zipfile.ZipInfo('%s/%s' % (name, fn), date_time=stamp)
            info.compress_type = zipfile.ZIP_STORED if fn.endswith('.onnx') else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(bundle_dir / fn, 'rb') as f:
                z.writestr(info, f.read())
    os.replace(ztmp, zpath)
    # ---- PyTorch の参照の出力
    check_dir = bundle_dir.parent / '_check'
    check_dir.mkdir(exist_ok=True)
    ref = torch_reference(run_dir, cfg, check_dir / ('%s_torch_ref.npz' % name))
    exp = {'run': name, 'run_dir': str(run_dir), 'exported_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'files': {fn: {'sha256': sha256(bundle_dir / fn), 'bytes': (bundle_dir / fn).stat().st_size} for fn in BUNDLE_FILES},
           'zip': str(zpath), 'zip_sha256': sha256(zpath), 'zip_bytes': zpath.stat().st_size,
           'source': {'onnx_sha256': onnx_sha, 'model_config_sha256': src_cfg_sha},
           'num_modes': pc['num_modes'], 'trained_modes': pc['trained_modes'], 'mode_id': SIM_MODE,
           'trained_center_us': list(trained_center), 'bundle_center_us': [STEER_CENTER_US, THROTTLE_NEUTRAL_US],
           'patched_keys': pc['simulator_bundle']['patched_keys'], 'torch_ref': ref}
    write_json(bundle_dir / 'export.json', exp)
    print(json.dumps({'run': name, 'zip': str(zpath), 'zip_bytes': exp['zip_bytes'], 'num_modes': pc['num_modes'],
                      'trained_center_us': list(trained_center), 'torch_l1_vs_label_us': ref['torch_l1_vs_label_us']},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
