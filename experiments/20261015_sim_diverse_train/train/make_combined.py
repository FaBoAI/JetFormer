#!/usr/bin/env python3
"""いくつかのデータセットを合わせた学習用のフォルダーを作る(protocol.json の arms_Q1.combined_folders・arms_Q2.C-DIV.folder)

    python -I make_combined.py --out <出力フォルダー> [--modes-from <dataset_card.json>] [--note <文>] -- <元 1>[@分け方] <元 2>[@分け方] ...

  分け方: 元の splits.json のどの分け(train・val・test)を入れるか。省くと 3 つとも(元と同じ分けに入れる)。
          例 `.../sim-rl-fastest_20261014-1125@train` = 学習の 12 シーンだけ(検証の 3 シーンは入れない。C-DIV のシミュレーター)

data/datasets には新しいデータセットを作らない。出力フォルダーに次を置く:
  aligned/<シーン>.npz           元のデータセットの aligned/<シーン>.npz へのシンボリックリンク(絶対パス)
  samples/<センサー>/<シーン>     元の samples/<センサー>/<シーン> へのシンボリックリンク(画像は写さない)
  splits.json                    入れた分けをつなげたもの(順番は引数の順・元の splits.json の順)
  dataset_card.json              ジャンルの表(--modes-from のもの。無ければ 1 つ目のもの)+ combined(元・シーン数・フレーム数)
  combined.json                  作ったときの記録(元の splits.json・dataset_card.json の sha256、入れた分け、シーンの数)

10/14 の experiments/20261014_sim_dart_margin/train/make_combined.py の写しを直したもの。変えたこと:
  (1) ジャンルの表は、名前の書き方がデータセットで違う(「駐車1」と「駐車(1)」)ので、表がまったく同じかではなく、
      id 0〜8 がそろっていることだけを確かめる(protocol.json の arms_Q1.combined_folders)。
  (2) 元ごとに入れる分けを選べる(C-DIV はシミュレーターの学習の 12 シーンだけ)。
  (3) --modes-from でジャンルの表を指定できる(C-DIV は simreal_mix_20261013 と同じ 0〜8 の表。protocol の arms_Q2.C-DIV.genres)。
  (4) 電源断に備えて <出力>.tmp に作ってから名前を替える(半端なフォルダーを残さない。.tmp が残っていたら消して作り直す)。
  (5) カードのフレーム数・ジャンル別のフレーム数は、入れたシーンの aligned から数え直す(分けを選ぶと元のカードの合計と違うため)。
元のデータセットは読むだけ。シーン名が重なったら止める。
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test')
NEED_MODE_IDS = set(range(9))          # ジャンル 0〜8(ジャンル数 9)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_source(text):
    if '@' in text:
        path, which = text.rsplit('@', 1)
        which = tuple(w for w in which.split(',') if w)
        bad = [w for w in which if w not in SPLITS]
        if bad:
            raise SystemExit('分け方 %s が分かりません(train・val・test)' % bad)
    else:
        path, which = text, SPLITS
    return Path(path).resolve(), which


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modes-from', default=None, help='ジャンルの表を取る dataset_card.json(既定: 1 つ目の元のもの)')
    ap.add_argument('--note', default='', help='カードの combined.note に書く文')
    ap.add_argument('--out', required=True, help='出力フォルダー')
    ap.add_argument('sources', nargs='+')
    args = ap.parse_args(argv[1:])
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit('出力フォルダー %s はもう中身があります(作り直すときは消してから)' % out)
    tmp = out.with_name(out.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)          # 前の試みの半端なもの(シンボリックリンクだけ。元のデータは消えない)
    srcs = [parse_source(a) for a in args.sources]
    cards = [json.loads((s / 'dataset_card.json').read_text(encoding='utf-8')) for s, _ in srcs]
    mode_tables = {}
    for (s, _), c in zip(srcs, cards):
        ids = {int(m['id']) for m in c.get('modes', [])}
        if args.modes_from:
            # 表を指定したとき(C-DIV): 元の表の id が指定の表(0〜8)の中にあればよい(実機の表は 0〜5 だけ)
            if not ids <= NEED_MODE_IDS:
                raise SystemExit('%s のジャンルの表の id %s が 0〜8 の外です' % (s.name, sorted(ids - NEED_MODE_IDS)))
        elif not NEED_MODE_IDS <= ids:
            raise SystemExit('%s のジャンルの表に id %s がありません' % (s.name, sorted(NEED_MODE_IDS - ids)))
        mode_tables[s.name] = {int(m['id']): m['name'] for m in c['modes']}
    if args.modes_from:
        modes = json.loads(Path(args.modes_from).read_text(encoding='utf-8'))['modes']
        modes_src = str(Path(args.modes_from).resolve())
    else:
        modes = cards[0]['modes']
        modes_src = str(srcs[0][0] / 'dataset_card.json')
    if {int(m['id']) for m in modes} != NEED_MODE_IDS:
        raise SystemExit('使うジャンルの表(%s)の id が 0〜8 ではありません' % modes_src)
    names0 = {int(m['id']): m['name'] for m in modes}
    name_diffs = {src: {str(i): [names0.get(i), n] for i, n in tab.items() if names0.get(i) != n}
                  for src, tab in mode_tables.items()}
    splits = {k: [] for k in SPLITS}
    seen = set()
    (tmp / 'aligned').mkdir(parents=True)
    rec = {'schema': 'jf-sim-diverse-combined/v1', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'modes_from': modes_src, 'mode_name_differences': {k: v for k, v in name_diffs.items() if v}, 'sources': []}
    frames_total, mode_frames = 0, {}
    for (s, which), card in zip(srcs, cards):
        sp = json.loads((s / 'splits.json').read_text(encoding='utf-8'))
        sensors = sorted(p.name for p in (s / 'samples').iterdir() if p.is_dir())
        n_links, n_frames, src_mode_frames = 0, 0, {}
        for split in which:
            for scene in sp.get(split, []):
                if scene in seen:
                    raise SystemExit('シーン名 %s が重なっています' % scene)
                seen.add(scene)
                splits[split].append(scene)
                src_npz = s / 'aligned' / (scene + '.npz')
                if not src_npz.exists():
                    raise SystemExit('%s がありません' % src_npz)
                os.symlink(src_npz, tmp / 'aligned' / (scene + '.npz'))
                with np.load(src_npz) as z:
                    mode = z['mode'] if 'mode' in z.files else np.zeros(len(z['timestamps_us']), np.int16)
                n_frames += len(mode)
                for m, c in zip(*np.unique(mode, return_counts=True)):
                    src_mode_frames[str(int(m))] = src_mode_frames.get(str(int(m)), 0) + int(c)
                for sensor in sensors:
                    d = s / 'samples' / sensor / scene
                    if d.exists():
                        (tmp / 'samples' / sensor).mkdir(parents=True, exist_ok=True)
                        os.symlink(d, tmp / 'samples' / sensor / scene)
                        n_links += 1
        frames_total += n_frames
        for m, c in src_mode_frames.items():
            mode_frames[m] = mode_frames.get(m, 0) + c
        rec['sources'].append({'dataset': s.name, 'path': str(s), 'splits_included': list(which),
                               'splits_sha256': sha256(s / 'splits.json'), 'card_sha256': sha256(s / 'dataset_card.json'),
                               'scenes_in_source': {k: len(sp.get(k, [])) for k in SPLITS},
                               'scenes_included': {k: len(sp.get(k, [])) if k in which else 0 for k in SPLITS},
                               'frames_included': n_frames, 'mode_frames_included': src_mode_frames,
                               'sample_dir_links': n_links, 'relabel': (card.get('relabel') or {}).get('schema'),
                               'label_mask': (card.get('label_mask') or {}).get('masked_frames')})
    card = {'name': out.name, 'version': cards[0].get('version'), 'modes': modes, 'num_modes': len(modes),
            'scenes': sum(len(v) for v in splits.values()), 'train_scenes': len(splits['train']),
            'val_scenes': len(splits['val']), 'test_scenes': len(splits['test']), 'frames': frames_total,
            'mode_frames': dict(sorted(mode_frames.items(), key=lambda kv: int(kv[0]))),
            'sensors': cards[0].get('sensors'), 'source': 'combined',
            'combined': {'note': args.note or ('学習用にいくつかのデータセットを合わせたフォルダー(sim_diverse_train_20261015 の腕 %s)。'
                                               '中身は元のデータセットへのシンボリックリンク。data/datasets には置かない' % out.name),
                         'sources': [r['dataset'] for r in rec['sources']],
                         'splits_included': {r['dataset']: r['splits_included'] for r in rec['sources']},
                         'relabel_of_sources': {r['dataset']: r['relabel'] for r in rec['sources']},
                         'label_mask_of_sources': {r['dataset']: r['label_mask'] for r in rec['sources']},
                         'modes_from': modes_src}}
    (tmp / 'splits.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8')
    (tmp / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
    rec['splits'] = {k: len(v) for k, v in splits.items()}
    rec['frames'] = frames_total
    rec['mode_frames'] = card['mode_frames']
    (tmp / 'combined.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    if out.exists():
        out.rmdir()                  # 空のフォルダー(上で中身が無いことを確かめた)
    os.replace(tmp, out)
    print(json.dumps({'out': str(out), 'splits': rec['splits'], 'frames': frames_total, 'mode_frames': card['mode_frames']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
