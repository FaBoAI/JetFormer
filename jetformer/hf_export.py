"""データセットを Hugging Face Hub に上げる(ブラウザの Dataset Viewer で見られる形にする)

リポジトリ(種類 dataset)に上げるもの:
  data/train-00000-of-0000N.parquet …   1行 = カメラ1フレーム。画像・ToFのカラー画像は Viewer にそのまま表示される
  data/validation-00000-of-0000N.parquet
  nuscenes/<name>.tar                    元の nuScenes 形式(tools/train.py でそのまま学習に使える。省略可)
  README.md                              データセットカード(列・センサー・ジャンルの説明、読み込み方)

Hub へのログインは Jetson で1回: /home/jetson/miniconda/envs/jetracer/bin/hf auth login
"""

import datetime
import json
import math
import os
import shutil

import cv2
import numpy as np

from . import config
from .recorder import depth_colormap

ROWS_PER_SHARD = 1000          # 1ファイル約30MB(画像 + ToF)
SPLITS = (('train', 'train'), ('val', 'validation'))     # (splits.json の名前, Hub での名前)


def features():
    import datasets
    f32 = datasets.Value('float32')
    return datasets.Features({
        'scene': datasets.Value('string'),
        'frame_index': datasets.Value('int32'),
        'timestamp_us': datasets.Value('int64'),
        'mode_id': datasets.Value('int16'),
        'mode_name': datasets.Value('string'),
        'description': datasets.Value('string'),
        'image': datasets.Image(),
        'tof_image': datasets.Image(),
        'steering_us': f32, 'throttle_us': f32, 'mode_us': f32, 'erpm': f32,
        'vehicle_valid': datasets.Value('bool'),
        'future_valid': datasets.Value('bool'),
        'future_actions_us': datasets.List(datasets.List(f32)),
        'imu': datasets.List(datasets.List(f32)),
        'tof_raw': datasets.Array2D(shape=(42, 54), dtype='uint16'),
        'tof_age_ms': f32,
        'vehicle_age_ms': f32,
    })


def _scene_table(root, name):
    path = root / ('v1.0-%s' % name) / 'scene.json'
    with open(path, encoding='utf-8') as f:
        return {s['name']: s for s in json.load(f)}


def _rows(root, scene, scene_meta, modes):
    """1シーン分の行(カメラ1フレーム = 1行)"""
    a = np.load(root / 'aligned' / ('%s.npz' % scene))
    n = len(a['timestamps_us'])
    mode = a['mode'] if 'mode' in a.files else np.zeros(n, np.int16)
    for i in range(n):
        with open(root / str(a['cam_files'][i]), 'rb') as f:
            jpg = f.read()
        tof = a['tof'][i]
        ok, tof_jpg = cv2.imencode('.jpg', depth_colormap(tof, 216, 168, vmin=0, vmax=2000), [cv2.IMWRITE_JPEG_QUALITY, 85])
        v = a['vehicle'][i]
        yield {
            'scene': scene, 'frame_index': i, 'timestamp_us': int(a['timestamps_us'][i]),
            'mode_id': int(mode[i]), 'mode_name': modes.get(int(mode[i]), str(int(mode[i]))),
            'description': scene_meta.get('description', ''),
            'image': {'bytes': jpg, 'path': None}, 'tof_image': {'bytes': tof_jpg.tobytes(), 'path': None},
            'steering_us': float(v[0]), 'throttle_us': float(v[1]), 'mode_us': float(v[2]), 'erpm': float(v[3]),
            'vehicle_valid': bool(a['vehicle_valid'][i]), 'future_valid': bool(a['future_valid'][i]),
            'future_actions_us': a['future_actions'][i].tolist(), 'imu': a['imu'][i].tolist(),
            'tof_raw': tof.astype(np.uint16), 'tof_age_ms': float(a['tof_age_ms'][i]),
            'vehicle_age_ms': float(a['vehicle_age_ms'][i]) if 'vehicle_age_ms' in a.files else 0.0,
        }


def build_parquet(name, out_dir, progress=None):
    """data/<split>-xxxxx-of-xxxxx.parquet を作る。戻り値 {Hubでの分割名: フレーム数}"""
    import datasets
    root = config.DATASET_DIR / name
    with open(root / 'splits.json', encoding='utf-8') as f:
        splits = json.load(f)
    with open(root / 'dataset_card.json', encoding='utf-8') as f:
        card = json.load(f)
    modes = {m['id']: m['name'] for m in card.get('modes', [])}
    scenes = _scene_table(root, name)
    feats = features()
    data_dir = out_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    total = sum(len(np.load(root / 'aligned' / ('%s.npz' % s))['timestamps_us']) for k, _ in SPLITS for s in splits.get(k, []))
    done, counts = 0, {}
    for key, hub_name in SPLITS:
        names = splits.get(key, [])
        n_rows = sum(len(np.load(root / 'aligned' / ('%s.npz' % s))['timestamps_us']) for s in names)
        if not n_rows:
            continue
        n_shards = max(1, math.ceil(n_rows / ROWS_PER_SHARD))
        shard, batch = 0, []

        def flush():
            nonlocal shard, batch
            if not batch:
                return
            ds = datasets.Dataset.from_list(batch, features=feats)
            ds.to_parquet(str(data_dir / ('%s-%05d-of-%05d.parquet' % (hub_name, shard, n_shards))))
            shard += 1
            batch = []
        for s in names:
            for row in _rows(root, s, scenes.get(s, {}), modes):
                batch.append(row)
                done += 1
                if len(batch) >= math.ceil(n_rows / n_shards):
                    flush()
                if progress and done % 100 == 0:
                    progress(0.6 * done / total, 'Parquetを作成中 %d / %d フレーム' % (done, total))
        flush()
        counts[hub_name] = n_rows
    return counts


def make_card(name, repo_id, counts, include_nuscenes, license_id='other'):
    """README.md(データセットカード)"""
    root = config.DATASET_DIR / name
    with open(root / 'dataset_card.json', encoding='utf-8') as f:
        card = json.load(f)
    scenes = _scene_table(root, name)
    modes = card.get('modes', [])
    mode_frames = card.get('mode_frames', {})
    yaml = ['---', 'license: %s' % license_id, 'task_categories:', '- robotics',
            'tags:', '- jetformer', '- rc-car', '- autonomous-driving', '- time-of-flight', '- imitation-learning',
            'pretty_name: JetFormer %s' % name, 'configs:', '- config_name: default', '  data_files:']
    for _, hub_name in SPLITS:
        if hub_name in counts:
            yaml += ['  - split: %s' % hub_name, '    path: data/%s-*.parquet' % hub_name]
    yaml.append('---')
    mode_rows = '\n'.join('| %d | %s | %s |' % (m['id'], m['name'], mode_frames.get(str(m['id']), 0)) for m in modes)
    scene_rows = '\n'.join('| %s | %s | %s | %.1f |' % (s, v.get('mode_name', ''), v.get('description', '') or '-', v.get('duration_s', 0))
                           for s, v in sorted(scenes.items()))
    nus = ('\n## 元の形式(学習用)\n\n`nuscenes/%s.tar` に、JetFormerの学習(`tools/train.py`)でそのまま使える nuScenes 形式の'
           'データセット(表・画像・ToFの16bit PNG・`aligned/*.npz`)が入っています。\n\n```bash\n'
           'hf download %s --repo-type dataset --include "nuscenes/*" --local-dir .\n'
           'tar xf nuscenes/%s.tar\npython tools/train.py --data %s --epochs 30 --stride 3\n```\n' % (name, repo_id, name, name)) \
        if include_nuscenes else ''
    return '\n'.join(yaml) + '''

# JetFormer データセット: %(name)s

RCカー(Jetson Orin Nano)で収集した、カメラ・ToF(VL53L9)・IMU・操作値(プロポ)のデータです。
1行がカメラの1フレーム(30fps)で、その時刻のToF・IMU・操作値と、1〜10フレーム先の操作値(学習の正解)が並んでいます。
収集は20秒前後の「走行(シーン)」単位で、走行ジャンル(駐車・高速走行など)が付いています。

- 作成: %(created)s / シーン数 %(scenes)d / フレーム数 %(frames)d(学習 %(train)s・検証 %(val)s)
- JetFormer の管理画面で収集・作成

## 読み込み

```python
from datasets import load_dataset
ds = load_dataset("%(repo)s", split="train")
row = ds[0]
row["image"]          # カメラ画像(PIL.Image, 224x224)
row["steering_us"]    # その時刻のステアリング [µs]
```

## 列

| 列 | 内容 |
|---|---|
| image | 前方カメラ(IMX219、224×224、JPEG) |
| tof_image | ToFの距離を色にした画像(近い=赤、遠い=青、0〜2000mm、無効=黒)。見る用 |
| tof_raw | ToFの生の値 [42×54] uint16。下位13bit=距離mm(0x1FFF=無効)、上位3bit=信頼度0〜7 |
| steering_us / throttle_us / mode_us | プロポの CH1(ステアリング)/ CH2(スロットル)/ CH3(モード)のパルス幅 [µs] |
| erpm | モーターの電気角回転数(符号=回転方向) |
| imu | 直前10サンプル(100Hz)の [加速度xyz m/s², 角速度xyz rad/s] |
| future_actions_us | 1〜10フレーム先の [ステアリング, スロットル] µs(学習の正解) |
| vehicle_valid / future_valid | 操作値が有効か(プロポ受信中・人が操縦・100ms以内)。偽の行は学習に使わない |
| mode_id / mode_name | 走行ジャンル(下の表) |
| scene / frame_index / timestamp_us | シーン名・シーン内の番号・時刻(UNIX µs) |
| description | シーンの説明 |

## 走行ジャンル

| ID | 名前 | フレーム数 |
|---|---|---|
%(mode_rows)s

## シーン

| シーン | ジャンル | 説明 | 長さ[秒] |
|---|---|---|---|
%(scene_rows)s
%(nus)s''' % {'name': name, 'created': card.get('created_at', ''), 'scenes': card.get('scenes', 0), 'frames': card.get('frames', 0),
              'train': counts.get('train', 0), 'val': counts.get('validation', 0), 'repo': repo_id,
              'mode_rows': mode_rows, 'scene_rows': scene_rows, 'nus': nus}


def stage(name, repo_id, include_nuscenes=True, license_id='other', progress=None):
    """上げるファイルを data/exports/hf_<name>/ に用意する。戻り値 (フォルダ, 分割ごとのフレーム数)"""
    out = config.EXPORT_DIR / ('hf_%s' % name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = build_parquet(name, out, progress)
    if include_nuscenes:
        from .dataset import export_dataset
        tar = config.EXPORT_DIR / ('%s.tar' % name)
        if not tar.exists():
            if progress:
                progress(0.62, 'nuScenes形式のtarを作成中')
            export_dataset(name)
        (out / 'nuscenes').mkdir()
        try:
            os.link(tar, out / 'nuscenes' / tar.name)      # 同じディスクならコピーせずリンク
        except OSError:
            shutil.copy2(tar, out / 'nuscenes' / tar.name)
    with open(out / 'README.md', 'w', encoding='utf-8') as f:
        f.write(make_card(name, repo_id, counts, include_nuscenes, license_id))
    return out, counts


def login_status():
    """Jetsonに Hugging Face のログイン情報があるか"""
    try:
        from huggingface_hub import whoami
        w = whoami()
        return {'logged_in': True, 'user': w.get('name'), 'orgs': [o.get('name') for o in w.get('orgs', [])]}
    except Exception as e:
        return {'logged_in': False, 'error': type(e).__name__,
                'hint': '/home/jetson/miniconda/envs/jetracer/bin/hf auth login でログインしてください'}


def upload(name, repo_id, private=True, include_nuscenes=True, license_id='other', progress=None):
    """データセットを Hub に上げる。戻り値 {'url', 'repo_id', 'private', 'counts'}"""
    from huggingface_hub import HfApi
    if '/' not in repo_id or repo_id.count('/') != 1:
        raise ValueError('リポジトリ名は「ユーザー名/名前」で指定してください')
    out, counts = stage(name, repo_id, include_nuscenes, license_id, progress)
    api = HfApi()
    if progress:
        progress(0.7, 'Hugging Face にアップロード中(%s)' % repo_id)
    api.create_repo(repo_id, repo_type='dataset', private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type='dataset',
                      commit_message='JetFormer dataset %s (%s)' % (name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M')),
                      delete_patterns=['data/*.parquet'])            # 前回の分割数が違っても古いファイルが残らないように
    info = {'repo_id': repo_id, 'url': 'https://huggingface.co/datasets/%s' % repo_id, 'private': bool(private),
            'counts': counts, 'nuscenes': bool(include_nuscenes),
            'uploaded_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(config.DATASET_DIR / name / 'hf_upload.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    shutil.rmtree(out, ignore_errors=True)
    if progress:
        progress(1.0, 'アップロードしました')
    return info

