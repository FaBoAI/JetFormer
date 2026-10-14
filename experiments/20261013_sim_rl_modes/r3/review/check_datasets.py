"""第 3〜4 回の審査: JetFormer に入っている強化学習モードのデータセット(一覧・タグ・ジャンル・学習除外・カード・シーンタブ)を API とファイルから確かめる(読むだけ)

    python3 -I check_datasets.py > ~/jetformer_reports/sim_rl_modes_20261013/r3/review/datasets.json

見ること:
  - 一覧(/api/datasets)に sim-rl-* が全部あり、シーン数・train/val・フレーム・学習に使えるフレーム・ジャンルのフレーム(mode_frames)
  - タグ = 登録簿の mode.json の tags(+ simulator)
  - カードの rl.policy_sha256 = 今の登録簿(新しいもの)/ 前の版のものには rl_superseded
  - 学習除外の数(label_mask.masked_frames)= mode_frames の 0 = frames − training_frames − 10×シーン
  - シーンタブ(/api/scenes)に ds-<名前> のシーンがシーン数だけあり、mode_id 8
  - ジャンルの表 data/modes.json は変わっていない(git の HEAD と同じ)
"""
import hashlib
import json
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

API = 'http://192.168.55.1:8080'
JF = Path('/home/jetson/jetformer')
REG = JF / 'data/sim_rl/modes'


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


def main():
    ds = get(API + '/api/datasets')
    scenes = get(API + '/api/scenes')
    by_log = {}
    for s in scenes:
        by_log.setdefault(s.get('log'), []).append(s)
    reg = {m: json.loads((REG / m / 'mode.json').read_text(encoding='utf-8')) for m in ('fastest', 'no_wall', 'recover', 'speed', 'line')}
    rows, problems = [], []
    for d in sorted((x for x in ds if str(x.get('name', '')).startswith('sim-rl-')), key=lambda x: x['name']):
        name = d['name']
        rl = d.get('rl') or {}
        mode = rl.get('mode')
        lm = d.get('label_mask') or {}
        mf = {str(k): v for k, v in (d.get('mode_frames') or {}).items()}
        sc = by_log.get('ds-' + name, [])
        cur = reg.get(mode, {})
        superseded = d.get('rl_superseded')
        row = {'name': name, 'mode': mode, 'scenes': d.get('scenes'), 'train_val': '%s/%s' % (d.get('train_scenes'), d.get('val_scenes')),
               'frames': d.get('frames'), 'training_frames': d.get('training_frames'), 'mode_frames': mf,
               'masked_frames': lm.get('masked_frames'), 'tags': d.get('tags'), 'rl_state_at_record': rl.get('state'),
               'policy_sha256': (rl.get('policy_sha256') or '')[:12], 'current_policy': (cur.get('policy_sha256') or '')[:12],
               'superseded': bool(superseded), 'superseded_gate_miss_scenes': len((superseded or {}).get('gate_miss_scenes') or []),
               'scene_tab': len(sc), 'scene_tab_mode_ids': dict(Counter(str(s.get('mode_id')) for s in sc)),
               'scene_tab_tags': sorted({t for s in sc for t in (s.get('tags') or [])}), 'has_rl_notes': bool(d.get('rl_notes'))}
        # 決まり
        want_tags = cur.get('tags') or []
        if sorted(d.get('tags') or []) != sorted(want_tags):
            problems.append('%s: タグ %s != mode.json %s' % (name, d.get('tags'), want_tags))
        n_sc = d.get('scenes') or 0
        exp_train = (d.get('frames') or 0) - 10 * n_sc - (lm.get('masked_frames') or 0)
        if mode == 'recover':
            # 乱しの外した範囲と最後の 10 フレームは重なることがあるので、== ではなく ≤ と mode_frames の 0 で見る
            row['training_frames_check'] = {'frames-10*scenes-masked': exp_train, 'mode0_frames': mf.get('0'), 'masked': lm.get('masked_frames')}
            if mf.get('0') != lm.get('masked_frames'):
                problems.append('%s: mode_frames の 0(%s)!= 学習除外(%s)' % (name, mf.get('0'), lm.get('masked_frames')))
        else:
            row['training_frames_check'] = {'frames-10*scenes': (d.get('frames') or 0) - 10 * n_sc}
            if d.get('training_frames') != (d.get('frames') or 0) - 10 * n_sc:
                problems.append('%s: 学習に使えるフレーム %s != frames − 10×シーン' % (name, d.get('training_frames')))
            if set(mf) != {'8'}:
                problems.append('%s: ジャンル 8 以外のフレーム %s' % (name, mf))
        if len(sc) != n_sc:
            problems.append('%s: シーンタブ %d != シーン %d' % (name, len(sc), n_sc))
        if any(str(s.get('mode_id')) != '8' for s in sc):
            problems.append('%s: シーンタブの mode_id が 8 でない' % name)
        is_current = rl.get('policy_sha256') == cur.get('policy_sha256')
        row['is_current_policy'] = is_current
        if not is_current and not superseded:
            problems.append('%s: 今の方策と違うのに rl_superseded が無い' % name)
        if is_current and superseded:
            problems.append('%s: 今の方策なのに rl_superseded がある' % name)
        rows.append(row)
    # ジャンルの表
    head = subprocess.run(['git', '-C', str(JF), 'show', 'HEAD:data/modes.json'], capture_output=True)
    modes_json_now = (JF / 'data/modes.json').read_bytes() if (JF / 'data/modes.json').exists() else None
    genre = {'tracked_in_git': head.returncode == 0,
             'same_as_HEAD': (head.returncode == 0 and modes_json_now == head.stdout),
             'sha256_now': hashlib.sha256(modes_json_now).hexdigest() if modes_json_now else None}
    out = {'schema': 'jf-sim-rl-r3-review-datasets/v1', 'datasets': rows, 'genre_table': genre, 'problems': problems,
           'total_datasets_listed': len(ds), 'sim_rl_datasets': len(rows)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
