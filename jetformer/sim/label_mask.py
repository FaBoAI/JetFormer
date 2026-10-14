"""シミュレーターの強化学習モードのシーンで、学習に使わない操作(info.jf.l = 0)を正解から外す(取り込みのとき)

強化学習モード(離脱復活など)の教師は、わざと乱した操作のフレームに action_info.jf.l = 0 の印を付ける(frames.jsonl)。
乱した操作を正解にしないため、取り込み(dataset_import.import_archive。/api/datasets/import・/api/sim/datasets/<id>/import が使う)で
次のように aligned/<シーン>.npz を書き換える(sim_rl_modes の modes.json common.masking・protocol definitions.mask_rule):

- フレーム j の操作が学習しないもの(l = 0)なら、それを正解(観測 k の正解は k+1〜k+10 の操作)に含む観測 k を外す。
  さらに観測 j 自身も外す(k ∈ [j − 10, j])。乱しの区間 [ps, pe) なら k ∈ [ps − 10, pe − 1]。
  (2026-10-14 統合の段: 設計の [ps − 10, pe − 2] から 1 つ広げた。シーンタブに展開したシーンからデータセットを作り直すと、
  正解を時刻(int(33333.33 µs) おき)で探すため、sim の 33,333/33,334 µs おきのフレームでは正解が 1 フレーム前にずれることがあり、
  観測 pe − 1 の最初の正解が乱した操作 pe − 1 になってしまう(確認 recover r2)。1 つ広げればどちらの数え方でも乱した操作が正解に入らない)
- 外した観測は future_valid[k] = False・mode[k] = 0(未設定 = 学習しない。シーンタブへの展開でも残る)。vehicle_valid は変えない。
- シーンごとの記録は、集約の形(simulation/scenes/<シーン>/)と、1 シーンの ZIP の形(simulation/frames.jsonl・session.json。
  シミュレーターの「JetFormerへ送る」やブリッジの 1 シーンの取り込み)の両方を読む。
- **安全側(fail closed)**: 強化学習モードのシーン(session.json の context.teacher.schema が jetformer-sim-rl-teacher/v1、
  または frames.jsonl のどこかに action_info.jf がある)では、jf・jf.l の無い行、l が 0/1 でない行、frames.jsonl に無いフレームを
  「学習しない」として外す。frames.jsonl が読めなければシーンの全部を外す。ブラウザが info を落としても乱した操作が正解にならない。
- 強化学習モードでないシーン(RC 教師など。jf がどこにも無い)は変えない。

dataset_card.json に label_mask = {schema 'jetformer-sim-label-mask/v1', source, horizon 10, fail_closed, masked_frames, per_scene, …} を書く。
"""

import json
import os
from pathlib import Path

import numpy as np

TEACHER_SCHEMA = 'jetformer-sim-rl-teacher/v1'
CARD_SCHEMA = 'jetformer-sim-label-mask/v1'
SOURCE = 'frames.jsonl action_info.jf.l'
HORIZON = 10
RULE = 'k ∈ [j − 10, j]'          # l = 0 の操作 j に対して外す観測 k(乱しの区間 [ps, pe) なら [ps − 10, pe − 1])


def mask_from_learn(learn, horizon=HORIZON):
    """learn[j] = 1(学習に使う)/ 0(使わない)→ 外す観測の印 mask[k](True = 外す)。k ∈ [j − horizon, j]"""
    learn = np.asarray(learn).astype(bool)
    mask = np.zeros(learn.shape[0], bool)
    for j in np.nonzero(~learn)[0]:
        mask[max(0, j - horizon):j + 1] = True
    return mask


def scene_dir(root, scene, single=False):
    """シーンの記録のフォルダー。集約の形は simulation/scenes/<シーン>/、1 シーンの ZIP の形(single)は simulation/"""
    d = Path(root) / 'simulation' / 'scenes' / scene
    if d.is_dir() or not single:
        return d
    return Path(root) / 'simulation'


def scene_teacher(root, scene, single=False):
    """シーンの session.json の context.teacher(無ければ None)"""
    p = scene_dir(root, scene, single) / 'session.json'
    try:
        t = json.loads(p.read_text(encoding='utf-8')).get('context', {}).get('teacher')
    except (OSError, ValueError, AttributeError):
        return None
    return t if isinstance(t, dict) else None


def read_frames(path):
    """frames.jsonl の行(dict)。読めなければ None"""
    try:
        rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except (OSError, ValueError):
        return None


def _jf(row):
    info = row.get('action_info') if isinstance(row, dict) else None
    jf = info.get('jf') if isinstance(info, dict) else None
    return jf if isinstance(jf, dict) else None


def learn_from_rows(rows, n):
    """frames.jsonl の行 → 長さ n の learn(1 = 学習に使う)。行は frame の番号で並べる(無ければ行の順)。
    安全側: jf・l が無い行、l が 0/1 でない行、無いフレームは 0。戻り値 (learn, 数え)"""
    learn = np.zeros(n, np.int8)
    seen = np.zeros(n, bool)
    stats = {'rows': 0, 'missing_jf': 0, 'bad_l': 0, 'l0': 0, 'missing_frames': 0, 'out_of_range': 0}
    for i, row in enumerate(rows or []):
        stats['rows'] += 1
        f = row.get('frame', i) if isinstance(row, dict) else i
        if not isinstance(f, int) or isinstance(f, bool) or not 0 <= f < n:
            stats['out_of_range'] += 1
            continue
        jf = _jf(row)
        if jf is None:
            stats['missing_jf'] += 1
            l = 0
        elif jf.get('l') == 1 and not isinstance(jf.get('l'), bool):
            l = 1
        elif jf.get('l') == 0 and not isinstance(jf.get('l'), bool):
            l = 0
            stats['l0'] += 1
        else:
            stats['bad_l'] += 1
            l = 0
        if seen[f]:                       # 同じフレームが 2 回あれば、どちらかが 0 なら 0
            learn[f] = min(int(learn[f]), l)
        else:
            learn[f] = l
            seen[f] = True
    stats['missing_frames'] = int((~seen).sum())
    return learn, stats


def _is_rl_scene(teacher, rows):
    if teacher and (teacher.get('schema') == TEACHER_SCHEMA or str(teacher.get('id', '')).startswith('jf-rl-')):
        return True
    return any(_jf(r) is not None for r in rows or [])


def _save_npz(path, arrays):
    tmp = path.with_name(path.name + '.mask-tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def apply_to_dataset(root, horizon=HORIZON):
    """展開したデータセット(root)の強化学習モードのシーンに学習除外を入れる。
    強化学習モードのシーンが無ければ何もしないで None。あれば dataset_card.json に label_mask を書き、その中身を返す"""
    root = Path(root)
    try:
        splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None                       # 形の確かめ(validate_dataset)が拒否する
    scenes = []
    for v in (splits.values() if isinstance(splits, dict) else []):
        if isinstance(v, list):
            scenes += [s for s in v if isinstance(s, str) and s not in scenes]
    per_scene, warnings = {}, []
    total_masked = 0
    single = len(scenes) == 1             # 1 シーンの ZIP は simulation/ の直下に記録がある
    for s in scenes:
        npz = root / 'aligned' / ('%s.npz' % s)
        if not npz.exists():
            continue                      # validate_dataset がエラーにする
        teacher = scene_teacher(root, s, single)
        fpath = scene_dir(root, s, single) / 'frames.jsonl'
        rows = read_frames(fpath) if fpath.exists() else None
        if not _is_rl_scene(teacher, rows):
            continue
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        n = len(arrays.get('timestamps_us', arrays.get('future_valid', [])))
        if rows is None:
            learn, st = np.zeros(n, np.int8), {'rows': 0, 'missing_jf': 0, 'bad_l': 0, 'l0': 0, 'missing_frames': n, 'out_of_range': 0}
            warnings.append('%s: frames.jsonl が読めないので、全フレームを学習から外しました(強化学習モードのシーン)' % s)
        else:
            learn, st = learn_from_rows(rows, n)
            if st['missing_jf'] or st['bad_l'] or st['missing_frames']:
                warnings.append('%s: info.jf の無い・おかしいフレーム %d を学習しないものとして扱いました'
                                % (s, st['missing_jf'] + st['bad_l'] + st['missing_frames']))
        mask = mask_from_learn(learn, horizon) if rows is not None else np.ones(n, bool)   # 読めなければ全部
        fv = arrays['future_valid'].astype(bool) if 'future_valid' in arrays else None
        newly = int((mask & fv).sum()) if fv is not None else int(mask.sum())
        if fv is not None:
            fv = fv.copy()
            fv[mask] = False
            arrays['future_valid'] = fv
        if 'mode' in arrays:
            md = arrays['mode'].copy()
            md[mask] = 0
            arrays['mode'] = md
        _save_npz(npz, arrays)
        total_masked += int(mask.sum())
        per_scene[s] = {'frames': int(n), 'masked': int(mask.sum()), 'newly_masked': newly, 'learn0_frames': int((learn == 0).sum()),
                        'teacher_mode': (teacher or {}).get('mode'), 'teacher_id': (teacher or {}).get('id'), **st}
    if not per_scene:
        return None
    card_path = root / 'dataset_card.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    # 学習に使えるフレームの数(カードの表示用)を数え直す。元の値は label_mask.before に残す
    before = {'training_frames': card.get('training_frames'), 'mode_frames': card.get('mode_frames')}
    tf, mf = 0, {}
    for s in scenes:
        p = root / 'aligned' / ('%s.npz' % s)
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            if 'future_valid' in z.files:
                tf += int(z['future_valid'].astype(bool).sum())
            if 'mode' in z.files:
                u, c = np.unique(z['mode'].astype(int), return_counts=True)
                for m, k in zip(u.tolist(), c.tolist()):
                    mf[str(m)] = mf.get(str(m), 0) + int(k)
    if 'training_frames' in card:
        card['training_frames'] = tf
    if isinstance(card.get('mode_frames'), dict):
        card['mode_frames'] = {k: mf.get(k, 0) for k in sorted(set(card['mode_frames']) | set(mf), key=lambda x: int(x))}
    lm = {'schema': CARD_SCHEMA, 'source': SOURCE, 'horizon': horizon, 'fail_closed': True,
          'rule': 'l = 0 の操作 j を正解(k+1〜k+10)に含む観測 k と観測 j 自身(k ∈ [j − 10, j])を future_valid = False・mode = 0。'
                  'jf の無い行は l = 0 とみなす', 'rule_id': RULE,
          'masked_frames': total_masked, 'scenes': len(per_scene), 'per_scene': per_scene, 'before': before, 'warnings': warnings[:50]}
    card['label_mask'] = lm
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
    return lm
