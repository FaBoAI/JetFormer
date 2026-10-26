#!/usr/bin/env python3
"""封の 14 本と protocol_v2 の固定を照らす(2026-10-25)。読むだけ。numpy も使わない。

    python3 -I /home/jetson/jetformer/experiments/20261025_train_v2/check_seal_v2.py

確かめること:
  1. protocol_v2.sha256 に書いたファイルの sha256 が今も同じか
  2. 封の 14 本のラベル(npz・json)・raw の log.json・シーンの一覧の sha256 が protocol_v2.json と同じか
     (中身は開かない。ファイルの sha256 とディレクトリの名前だけ)
  3. 封の記録が、作ったデータセット dataset_v2_20261025 の source_scenes に無いか(データセットがあるときだけ)
段 0 の前・lock2 の前・11/1 に封を開ける前に動かす。どれかが違えば 1 で終わる。
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
DATA = Path('/home/jetson/jetformer/data')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    bad = []
    for line in open(REP / 'protocol_v2.sha256', encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        want, path = line.split(None, 1)
        p = Path(path) if path.startswith('/') else REP / path
        if not p.exists() or sha(p) != want:
            bad.append('固定のファイルが違う: %s' % path)
    proto = json.load(open(REP / 'protocol_v2.json', encoding='utf-8'))
    lines = []
    for log in proto['seal']['logs']:
        ent = proto['seal']['per_log'][log]
        raw = DATA / 'raw' / log
        scenes = sorted(d for d in os.listdir(raw) if (raw / d).is_dir())
        got = (sha(DATA / 'labels' / ('%s.npz' % log)), sha(DATA / 'labels' / ('%s.json' % log)), sha(raw / 'log.json'),
               hashlib.sha256(('\n'.join(scenes) + '\n').encode('utf-8')).hexdigest())
        want = (ent['labels_npz_sha256'], ent['labels_json_sha256'], ent['raw_log_json_sha256'], ent['scene_list_sha256'])
        if got != want:
            bad.append('封の記録が変わった: %s' % log)
        lines.append('%s %s %s %s %s' % ((log,) + got))
    manifest = hashlib.sha256(('\n'.join(lines) + '\n').encode('utf-8')).hexdigest()
    if manifest != proto['seal']['manifest_sha256']:
        bad.append('封の一覧の sha256 が違う')
    card = DATA / 'datasets' / 'dataset_v2_20261025' / 'dataset_card.json'
    ds_note = 'データセットはまだ無い'
    if card.exists():
        c = json.load(open(card, encoding='utf-8'))
        src = json.dumps(c.get('source_scenes', c), ensure_ascii=False)
        hit = [log for log in proto['seal']['logs'] if log in src]
        ds_note = 'source_scenes に封の記録: %d 本' % len(hit)
        if hit:
            bad.append('データセットに封の記録がある: %s' % ' '.join(hit))
    print('封の一覧の sha256:', manifest)
    print(ds_note)
    if bad:
        print('\n'.join(bad))
        return 1
    print('すべて同じ(固定のファイル・封の 14 本)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
