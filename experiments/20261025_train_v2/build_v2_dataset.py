#!/usr/bin/env python3
"""データセット dataset_v2_20261025 を作る(2026-10-25。protocol_v2.json の tables.dataset のとおり)。

tools/build_ai_dataset.py(jetformer の道具)をそのまま呼ぶ。前と後に確かめて、外れたら止まる(終わりの値 1)。
  前: protocol の固定(sha256sum -c)・--ai-logs は protocol の 46 本(省かない)・封の 14 本が入っていない・
      ラベルがある・ジャンル 9 が data/modes.json にある・同じ名前のデータセットが無い
  後: 封の記録が source_scenes に 0・検証(val)が H1 の 11 シーンと同じ・splits の学習と検証が重ならない
GPU・torch は使わない(numpy と cv2)。data/raw・data/labels は読むだけ(画像は os.link で写す = build_dataset のまま)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 3G --need-mb 1500 -- nice -n 10 \
      /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/build_v2_dataset.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL = REP / 'protocol_v2.json'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
NAME = 'dataset_v2_20261025'


def fail(msg):
    print('止める: ' + msg, flush=True)
    sys.exit(1)


def main():
    r = subprocess.run(['sha256sum', '-c', 'protocol_v2.sha256'], cwd=str(REP), capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        fail('protocol の固定が合いません')
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    ds = p['tables']['dataset']
    assert ds['name'] == NAME
    ai_logs = [x for v in ds['ai_logs'].values() for x in v]
    sealed = list(p['seal']['logs'])
    if len(ai_logs) != 46 or len(set(ai_logs)) != 46:
        fail('--ai-logs が 46 本ではありません: %d' % len(ai_logs))
    if set(ai_logs) & set(sealed):
        fail('封の記録が --ai-logs にあります: %s' % sorted(set(ai_logs) & set(sealed)))
    for lg in ai_logs:
        if not (ROOT / 'data/labels' / (lg + '.npz')).exists() or not (ROOT / 'data/labels' / (lg + '.json')).exists():
            fail('ラベルがありません: %s' % lg)
        if not (ROOT / 'data/raw' / lg).is_dir():
            fail('記録がありません: %s' % lg)
    modes = json.loads((ROOT / 'data/modes.json').read_text(encoding='utf-8'))
    if not any(m['id'] == 9 for m in modes):
        fail('ジャンル 9 が data/modes.json にありません(先に add_genre9.py)')
    if (ROOT / 'data/datasets' / NAME).exists():
        fail('同じ名前のデータセットがあります: %s' % NAME)

    cmd = [PY, '-I', 'tools/build_ai_dataset.py', '--name', NAME, '--human-from', 'clean_driving_20261011',
           '--ai-logs'] + ai_logs + ['--val-ai-logs', '--keep-ai-log-human', '--label-gains', 'eval',
                                     '--eval-steer-gain', '1.0', '--eval-throttle-gain', '1.0']
    print('コマンド:', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        fail('build_ai_dataset.py が %d で終わりました' % r.returncode)

    root = ROOT / 'data/datasets' / NAME
    card = json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    src = card['source_scenes']
    bad = [s for s in src if s.rsplit('-', 1)[0] in set(sealed)]
    if bad:
        fail('封の記録のシーンが source_scenes にあります: %s' % bad[:5])
    h1 = p['holdouts']['H1']['scenes']
    if sorted(splits['val']) != sorted(h1):
        fail('検証が H1 の 11 シーンと違います: %s' % splits['val'])
    if set(splits['train']) & set(splits['val']):
        fail('学習と検証が重なります')
    if card['ai_mix']['ai_logs'] != ai_logs or card['ai_mix']['val_ai_logs']:
        fail('カードの ai_logs・val_ai_logs が違います')
    print('作った後の確かめ: OK(シーン %d・学習 %d・検証 %d・封 0)' % (len(src), len(splits['train']), len(splits['val'])))


if __name__ == '__main__':
    main()
