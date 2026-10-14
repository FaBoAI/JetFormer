"""確認の段(review、2026-10-14): RL のデータセットの ZIP を、今の JetFormer のコードで「別のデータ置き場」に取り込み直して確かめる

    JETFORMER_DATA=<空のフォルダー> python -I reimport_check.py <ZIP> <出力 JSON>

本物の data/ には何も書かない(JETFORMER_DATA で置き場を変える。modes.json は本物の写しを置く)。
確かめること: 取り込める・一覧に出る(dataset.list_datasets)・シーンタブに 60 シーン展開される(dataset.list_scenes)・
ジャンル 8・タグ simulator + rl-bestlap・学習に使えるフレーム 35,400・強化学習モードの学習除外(label_mask)が当たらない・
学習の読み込み(train_data.JetFormerDataset)で 1 サンプル読める。
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, '/home/jetson/jetformer')


def main():
    zpath, out = sys.argv[1], sys.argv[2]
    data = os.environ['JETFORMER_DATA']
    assert os.path.realpath(data) != os.path.realpath('/home/jetson/jetformer/data'), '本物の data/ には取り込まない'
    from jetformer import config, dataset, dataset_import  # noqa: E402
    assert str(config.DATA_DIR) == data
    t0 = time.time()
    prog = []
    info = dataset_import.import_archive(zpath, simulator=True, progress=lambda p, m: prog.append((round(p, 3), m)))
    name = info['name'] if isinstance(info, dict) else info
    tags_before = info.get('tags') if isinstance(info, dict) else None
    card = dataset.set_dataset_tags(name, ['simulator', 'rl-bestlap'])
    listed = [d for d in dataset.list_datasets() if d.get('name') == name]
    scenes = [s for s in dataset.list_scenes() if (s.get('imported_from') or {}).get('dataset') == name]
    res = {'archive': zpath, 'archive_sha256': hashlib.sha256(open(zpath, 'rb').read()).hexdigest(), 'data_dir': data,
           'seconds': round(time.time() - t0, 1), 'name': name, 'listed': len(listed) == 1,
           'card': {k: card.get(k) for k in ('scenes', 'frames', 'training_frames', 'train_scenes', 'val_scenes', 'num_modes', 'mode_frames', 'tags', 'source')},
           'label_mask_applied': 'label_mask' in card, 'tags_after_import_before_set': tags_before,
           'scenes_tab': {'count': len(scenes), 'modes': sorted({s.get('mode_id') for s in scenes}),
                          'frames_cam': sum((s.get('counts') or {}).get('CAM_FRONT', 0) for s in scenes),
                          'unset_segments': sum(1 for s in scenes for sg in s.get('mode_segments') or [] if sg.get('mode_id') == 0)},
           'progress_last': prog[-1] if prog else None}
    # 学習の読み込みで 1 サンプル(CPU)
    try:
        from jetformer import train_data
        ds = train_data.JetFormerDataset(str(config.DATASET_DIR / name), split='train')
        smp = ds[0]
        res['train_reader'] = {'train_samples': len(ds), 'sample_keys': sorted(smp.keys()) if isinstance(smp, dict) else str(type(smp))}
    except Exception as e:  # noqa: BLE001
        res['train_reader'] = {'error': repr(e)[:300]}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])


if __name__ == '__main__':
    main()
