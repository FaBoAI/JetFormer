#!/usr/bin/env python3
"""2 つのデータセットを合わせた学習用のフォルダーを作る(protocol.json の arms.combined_datasets)

    python -I make_combined.py <出力フォルダー> <データセット 1> <データセット 2> [...]

data/datasets には新しいデータセットを作らない。出力フォルダーに次を置く:
  aligned/<シーン>.npz           元のデータセットの aligned/<シーン>.npz へのシンボリックリンク(絶対パス)
  samples/<センサー>/<シーン>     元の samples/<センサー>/<シーン> へのシンボリックリンク
  splits.json                    元の splits.json の train・val・test をつなげたもの(順番は引数の順)
  dataset_card.json              1 つ目のカードの写し + combined(元のデータセット・シーン数)。ジャンルの表は全部同じことを確かめる
  combined.json                  作ったときの記録(元の splits.json・dataset_card.json の sha256)
元のデータセットは読むだけ。シーン名が重なったら止める。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    out = Path(argv[1])
    srcs = [Path(a).resolve() for a in argv[2:]]
    if out.exists() and any(out.iterdir()):
        raise SystemExit('出力フォルダー %s はもう中身があります(作り直すときは消してから)' % out)
    cards = [json.loads((s / 'dataset_card.json').read_text(encoding='utf-8')) for s in srcs]
    modes0 = cards[0].get('modes')
    for s, c in zip(srcs, cards):
        if c.get('modes') != modes0:
            raise SystemExit('%s のジャンルの表が 1 つ目と違います' % s.name)
    splits = {'train': [], 'val': [], 'test': []}
    seen = set()
    (out / 'aligned').mkdir(parents=True)
    rec = {'schema': 'jf-sim-dart-combined/v1', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'sources': []}
    for s in srcs:
        sp = json.loads((s / 'splits.json').read_text(encoding='utf-8'))
        sensors = sorted(p.name for p in (s / 'samples').iterdir() if p.is_dir())
        n_links = 0
        for split in ('train', 'val', 'test'):
            for scene in sp.get(split, []):
                if scene in seen:
                    raise SystemExit('シーン名 %s が重なっています' % scene)
                seen.add(scene)
                splits[split].append(scene)
                src_npz = s / 'aligned' / (scene + '.npz')
                if not src_npz.exists():
                    raise SystemExit('%s がありません' % src_npz)
                os.symlink(src_npz, out / 'aligned' / (scene + '.npz'))
                for sensor in sensors:
                    d = s / 'samples' / sensor / scene
                    if d.exists():
                        (out / 'samples' / sensor).mkdir(parents=True, exist_ok=True)
                        os.symlink(d, out / 'samples' / sensor / scene)
                        n_links += 1
        rec['sources'].append({'dataset': s.name, 'path': str(s), 'splits_sha256': sha256(s / 'splits.json'),
                               'card_sha256': sha256(s / 'dataset_card.json'),
                               'scenes': {k: len(sp.get(k, [])) for k in ('train', 'val', 'test')},
                               'sample_dir_links': n_links, 'relabel': (cards[srcs.index(s)].get('relabel') or {}).get('schema')})
    card = dict(cards[0])
    card['name'] = out.name
    card['scenes'] = sum(len(v) for v in splits.values())
    card['train_scenes'] = len(splits['train'])
    card['val_scenes'] = len(splits['val'])
    card['test_scenes'] = len(splits['test'])
    for k in ('frames', 'tof_frames', 'training_frames'):
        if all(isinstance(c.get(k), (int, float)) for c in cards):
            card[k] = sum(c[k] for c in cards)
    card.pop('relabel', None)
    card.pop('scene_log', None)
    card['combined'] = {'note': '学習用に 2 つのデータセットを合わせたフォルダー(sim_dart_20261014 の腕 %s)。'
                                '中身は元のデータセットへのシンボリックリンク。data/datasets には置かない' % out.name,
                        'sources': [r['dataset'] for r in rec['sources']],
                        'relabel_of_sources': {r['dataset']: r['relabel'] for r in rec['sources']}}
    (out / 'splits.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
    rec['splits'] = {k: len(v) for k, v in splits.items()}
    (out / 'combined.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(rec['splits'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
