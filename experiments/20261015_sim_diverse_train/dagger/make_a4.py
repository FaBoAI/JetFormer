#!/usr/bin/env python3
"""A4(DAgger)の学習データと train/a4_plan.json を作る(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 6)

    python3 -I make_a4.py --base-arm S-DIV --dagger-dataset <取り込んで付け替えたデータセットの名前>

  - 元の腕(S-DIV か S-MOD)の合わせたフォルダーの元(combined.json の sources、同じ分け)+ DAgger の 12 シーン(全部 train)を、
    学習の段の make_combined.py(写し)で train/data/A4/ にシンボリックリンクで合わせる(data/datasets に新しいデータセットは作らない)
  - DAgger のデータセットが付け替え済み(dataset_card.json の relabel)であることを確かめる
  - <レポート>/train/a4_plan.json = {ok, data, init(元の腕の best.pth), base_arm, dagger_dataset, ...}(学習の道具 diverse_train.py が読む)
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261015_sim_diverse_train')
REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data' / 'datasets'
RUNS = {'S-DIV': 'simdiv_20261015-SDIV-s0', 'S-MOD': 'simdiv_20261015-SMOD-s0'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-arm', choices=list(RUNS), required=True)
    ap.add_argument('--dagger-dataset', required=True)
    a = ap.parse_args()
    dd = DS / a.dagger_dataset
    card = json.loads((dd / 'dataset_card.json').read_text())
    if not card.get('relabel'):
        sys.exit('DAgger のデータセットが付け替えられていません(dataset_card.json の relabel が無い)')
    base = json.loads((EXP / 'train' / 'data' / a.base_arm / 'combined.json').read_text())
    srcs = []
    for s in base['sources']:
        inc = s['splits_included']
        srcs.append(s['path'] + ('' if set(inc) == {'train', 'val', 'test'} else '@' + ','.join(inc)))
    sp = json.loads((dd / 'splits.json').read_text())
    if sp.get('val') or sp.get('test'):
        sys.exit('DAgger のデータセットに val・test があります(全部 train のはず)')
    srcs.append(str(dd))
    out = EXP / 'train' / 'data' / 'A4'
    cmd = [sys.executable, '-I', str(EXP / 'train' / 'make_combined.py'), '--out', str(out), '--modes-from', base['modes_from'],
           '--note', f'A4(DAgger): {a.base_arm} のデータ + DAgger の 12 シーン(全部 train)'] + srcs
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    if r.returncode != 0:
        sys.exit('make_combined.py が失敗しました')
    best = REPO / 'data' / 'runs' / RUNS[a.base_arm] / 'best.pth'
    if not best.exists():
        sys.exit('元の腕の best.pth がありません: ' + str(best))
    plan = {'ok': True, 'data': str(out.relative_to(REPO)), 'init': str(best.relative_to(REPO)), 'base_arm': a.base_arm, 'base_run': RUNS[a.base_arm],
            'dagger_dataset': a.dagger_dataset, 'dagger_relabel': card['relabel'], 'made': datetime.datetime.now().isoformat(timespec='seconds'),
            'recipe': 'protocol.json dagger.procedure 6: 元の腕の best.pth から、元の腕のデータ + DAgger の 12 シーンで 10 エポック(lr 1e-4、ほかは同じ。--init)'}
    (Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015/train/a4_plan.json')).write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
