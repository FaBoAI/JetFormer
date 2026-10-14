"""昇格の段(2026-10-15、r3/promote): 前の版の方策で記録した強化学習モードのデータセット(no_wall r2・recover r1)のカードに、
置き換えたことと分かっている問題を書き足す(データ・ラベル・npz は変えない。dataset_card.json に rl_superseded を足すだけ)。

  - sim-rl-recover_20261014-1025・-1125: 離脱復活 r1(7720f275…、乱し v1)。ゲートを外したシーンがある(確認 recover r4 の数え直し:
    1025 は 2/3、1125 は 6/15。記録の段・README の「2/15」は誤り)。r3/rc4/promote/recorded_scene_check_existing.json
  - sim-rl-no_wall_20261014-1025・-1125: 絶対壁にぶつからない r2(153c709c…、シールド v3)。記録したシーンは決まり(すき間 8 cm 以上など)を満たすが、
    方策は確認 r2 で不合格(参考)。r4 に置き換えた
変える前のカードは r3/promote/before/dataset_cards/<名前>.json。書き込みは一時ファイル → fsync → 置き換え。

    python3 -I r3/promote/note_old_datasets.py [--dry-run]
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

DS = Path('/home/jetson/jetformer/data/datasets')
BEFORE = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/promote/before/dataset_cards')
CHECK = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc4/promote/recorded_scene_check_existing.json')
NOW = datetime.now().isoformat(timespec='seconds')


def gate_miss_scenes():
    d = json.loads(CHECK.read_text(encoding='utf-8'))
    out = {}
    for r in d['results']:
        if r.get('accept'):
            continue
        p = Path(r['dir'])
        out.setdefault(p.parts[-4] if 'simulation' in p.parts else p.parent.name, []).append({'scene': r['scene'], 'gate_misses': r.get('gate_misses'),
                                                                                              'passed_gates_report': r.get('passed_gates_report')})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    gm = gate_miss_scenes()
    notes = {}
    for name in ('sim-rl-recover_20261014-1025', 'sim-rl-recover_20261014-1125'):
        bad = gm.get(name, [])
        card = json.loads((DS / name / 'dataset_card.json').read_text(encoding='utf-8'))
        n = card.get('scenes')
        notes[name] = {'schema': 'jf-sim-rl-superseded/v1', 'at': NOW, 'mode': 'recover', 'recorded_with_policy': (card.get('rl') or {}).get('policy_sha256'),
                       'replaced_by': {'round': 'r4', 'policy_sha256': '68e321e40309e6fa0d8c05cc3419ca8ef8a1d2c3191aeb91411b3a1cd93990ef', 'state': 'accepted'},
                       'gate_miss_scenes': bad,
                       'note_ja': ('このデータセットは離脱復活 r1 の方策(不合格(参考)・乱し v1)で記録した。2026-10-15 に r4(合格)へ置き換えた。'
                                   '%d シーン中 %d シーンでシミュレーターのゲートを外している(確認 recover r4 の数え直し。乱しのあとにゲートの外側を通り、'
                                   'sim は 1 周後に同じゲートを通るまで周を数えない)。外した所の操作も学習に使われる(乱した操作そのものは学習から外してある)。'
                                   '使うなら r4 で記録し直したデータセットを優先すること' % (n, len(bad)))}
    for name in ('sim-rl-no_wall_20261014-1025', 'sim-rl-no_wall_20261014-1125'):
        card = json.loads((DS / name / 'dataset_card.json').read_text(encoding='utf-8'))
        notes[name] = {'schema': 'jf-sim-rl-superseded/v1', 'at': NOW, 'mode': 'no_wall', 'recorded_with_policy': (card.get('rl') or {}).get('policy_sha256'),
                       'replaced_by': {'round': 'r4', 'policy_sha256': 'e632d96bfc71fca15eb11a4500a62f697908d583b84f6b7242a460e3e58b879d', 'state': 'accepted'},
                       'note_ja': ('このデータセットは「絶対壁にぶつからない」r2 の方策(シールド v3、確認 r2 で不合格(参考))で記録した。記録したシーンは'
                                   '記録の決まり(衝突 0・すき間 8 cm 以上・止まりなし)を満たしているが、2026-10-15 に r4(合格、シールド v5)へ置き換えた')}
    BEFORE.mkdir(parents=True, exist_ok=True)
    for name, note in notes.items():
        p = DS / name / 'dataset_card.json'
        card = json.loads(p.read_text(encoding='utf-8'))
        if not (BEFORE / (name + '.json')).exists():
            shutil.copyfile(p, BEFORE / (name + '.json'))
        card['rl_superseded'] = note
        print(name, '→', note['note_ja'][:90], '…', len(note.get('gate_miss_scenes') or []))
        if a.dry_run:
            continue
        tmp = p.with_name('.dataset_card.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(card, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
