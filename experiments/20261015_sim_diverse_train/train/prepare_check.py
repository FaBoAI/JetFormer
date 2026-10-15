#!/usr/bin/env python3
"""準備の段の確かめ(CPU だけ): 腕の学習データ(合わせたフォルダー)が protocol.json のとおりか

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -I experiments/20261015_sim_diverse_train/train/prepare_check.py

tools/train.py と同じ JetFormerDataset・同じ設定(間引き 3・衝突の前後を外す 2.5G・今の操作値なし)で読み込み、次を確かめて
/home/jetson/jetformer_reports/sim_diverse_train_20261015/prepare_check.json に書く(書き終えてから置き換える):

  元のデータセット(7 つのシミュレーター + 実機)
    - シーン数・1 シーン 600 フレーム・フレーム数がカードと同じ・aligned のジャンル(シミュレーター 8(と離脱復活の 0)、実機 4・5)
    - 5 モードはカードの rl.state = accepted。no_wall・recover の 1025・1125(合格前)を使っていない
    - 離脱復活: frames.jsonl の action_info.jf.l から外すフレーム(l = 0 の操作 j について k ∈ [j − 10, j])を数え直し、
      aligned のジャンル 0 のフレームとまったく同じ・future_valid = False・合わせて 1,623
    - PWM の約束: 各データセット 1 シーンで、frames.jsonl の操作(command_si)から PWM を 2 つの約束(1441 / 1496 と 1442 / 1498)で
      作り直し、(a) 自分の約束で記録の PWM(applied_pwm_us)と同じ、(b) 2 つの約束の差が舵 ≤ 1 µs・スロットル ≤ 2 µs(5 µs を超えたら付け替え)、
      (c) aligned の正解(future_actions)が記録の PWM と同じ(D1 は付け替え済みなので参考)
  腕(S-DIV・S-MOD・C-DIV)
    - シーン数・学習 / 検証のサンプル数が protocol.json の見込みと同じ、元ごとに数えた合計と同じ(サンプルの (シーン, フレーム) の表も同じ)
    - ジャンル 0 のサンプル 0・ジャンル数 9・ジャンル別のサンプル数・離脱復活の外したフレームが 1 つも入っていない
    - シンボリックリンクがすべて元のデータセットのファイルを指す・全フレームの画像のパスが開ける・画像を写していない(フォルダーが小さい)
  C-DIV だけ
    - 実機の検証 1,559 サンプルが A(clean_driving_20261011)・C(simreal_mix_20261013)の検証とまったく同じ (シーン, フレーム)
    - 実機の学習 5,912 サンプルが A の学習と同じ・衝突の前後として外したフレーム(学習 3,522・検証 1,083)が A と同じ
    - 実機の aligned のリンク先が clean_driving_20261011 のファイルそのもの(同じ i ノード)
    - シミュレーターは 5 モードの学習の 12 シーンずつ(60 シーン)・学習サンプル 11,391・検証に入っていない
    - action_reference(止まっている間の操作の中央値)を記録(A・C は [1441.0, 1496.0])
車は動かさない・PCA9685 に触らない。元のデータセットは読むだけ。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '/home/jetson/jetformer')
import numpy as np  # noqa: E402

from jetformer.train_data import JetFormerDataset  # noqa: E402

REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data' / 'datasets'
HERE = Path(__file__).resolve().parent
ARMS_DIR = HERE / 'data'
REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA = REPORT / 'protocol.sha256'
OUT = REPORT / 'prepare_check.json'
KW = dict(stride=3, use_command=False, mask_impacts=True, impact_g=2.5)     # tools/train.py と同じ(arms_Q1.common_recipe)

MODES5 = ['sim-rl-fastest_20261014-1125', 'sim-rl-speed_20261014-1125', 'sim-rl-no_wall_20261015-0302',
          'sim-rl-recover_20261015-0302', 'sim-rl-line_20261015-0302']
D1 = 'jetformer-simulator-rl-dart-fast-10min_20261014-062216_4446c73d'
D2 = 'jetformer-simulator-rl-margin-line-10min_20261014-070934_48d04e6b'
REAL = 'clean_driving_20261011'
C_MIX = 'simreal_mix_20261013'
RECOVER = 'sim-rl-recover_20261015-0302'
NOT_USED = ['sim-rl-no_wall_20261014-1025', 'sim-rl-recover_20261014-1025', 'sim-rl-no_wall_20261014-1125',
            'sim-rl-recover_20261014-1125']
A_RUNS = ['clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30']
C_RUNS = ['simreal_mix_20261013-cotrain-s0', 'simreal_mix_20261013-cotrain-s1']
# 腕 → [(元のデータセット, 入れる分け)]
ARMS = {
    'S-DIV': [(d, ('train', 'val')) for d in MODES5 + [D1, D2]],
    'S-MOD': [(d, ('train', 'val')) for d in MODES5],
    'C-DIV': [(REAL, ('train', 'val'))] + [(d, ('train',)) for d in MODES5],
}
SAMPLE_MAX_FOLDER_MB = 5            # 合わせたフォルダーはリンクと JSON だけ(画像を写していれば数百 MB になる)
PWM_LIMIT_STEER, PWM_LIMIT_THR, PWM_RELABEL_US = 1.0, 2.0, 5.0


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ds_index(root, split):
    """JetFormerDataset(train.py と同じ設定)で読み込み → (サンプルの (シーン, フレーム) の表, 数えたもの)"""
    ds = JetFormerDataset(str(root), split, color_jitter=(split == 'train'), **KW)
    sp = load_json(Path(root) / 'splits.json')[split]
    loaded = [s for s in sp if (Path(root) / 'aligned' / (s + '.npz')).exists()]
    names = []
    for a in ds.scenes:     # ds.scenes は splits.json の順に、読めたシーンだけ(aligned の cam_files からシーン名を取る)
        names.append(str(a['cam_files'][0]).split('/')[2])
    idx = [(names[k], int(i)) for k, i in ds.index]
    info = {'samples': len(ds), 'scenes': len(ds.scenes), 'scenes_in_split': len(sp), 'scenes_loadable': len(loaded),
            'impact_masked': ds.impact_masked, 'num_modes': ds.num_modes,
            'mode_counts': {str(k): v for k, v in sorted(ds.mode_counts().items())}}
    ref = ds.action_reference() if split == 'train' else None
    if ref is not None:
        info['action_reference'] = list(ref)
    del ds
    return idx, info


# ---------------------------------------------------------------- 元のデータセット
L0 = {}                               # 離脱復活の乱した操作(l = 0)のフレーム {シーン: set}


def recover_masked_frames():
    """離脱復活の外すフレームを frames.jsonl の action_info.jf.l から数え直す → {シーン: set(フレーム)}, 確かめ"""
    root = DS / RECOVER
    card = load_json(root / 'dataset_card.json')
    horizon = int(card['label_mask']['horizon'])
    out, per_scene, ok = {}, {}, True
    L0.clear()
    scenes = [s for k in ('train', 'val', 'test') for s in load_json(root / 'splits.json').get(k, [])]
    for scene in scenes:
        rows = [json.loads(line) for line in (root / 'simulation' / 'scenes' / scene / 'frames.jsonl').open(encoding='utf-8')]
        n_rows = len(rows)
        l0 = [int(r['frame']) for r in rows if int(((r.get('action_info') or {}).get('jf') or {}).get('l', 0)) == 0]
        with np.load(root / 'aligned' / (scene + '.npz')) as z:
            mode = z['mode'].astype(int)
            fv = z['future_valid'].astype(bool)
        n = len(mode)
        m = set()
        for j in l0:
            m.update(k for k in range(j - horizon, j + 1) if 0 <= k < n)
        al0 = set(np.nonzero(mode == 0)[0].tolist())
        same = (m == al0)
        fv_false = all(not fv[k] for k in m)
        ok &= same and fv_false
        out[scene] = m
        L0[scene] = set(l0)
        per_scene[scene] = {'rows': n_rows, 'frames': n, 'l0_actions': len(l0), 'masked_recount': len(m),
                            'aligned_mode0': len(al0), 'same': same, 'future_valid_false': fv_false,
                            'card_masked': card['label_mask']['per_scene'].get(scene, {}).get('masked')}
    total = sum(len(v) for v in out.values())
    res = {'rule': card['label_mask']['rule_id'], 'horizon': horizon, 'scenes': len(scenes), 'masked_recount': total,
           'card_masked': card['label_mask']['masked_frames'], 'per_scene': per_scene,
           'ok': bool(ok and total == card['label_mask']['masked_frames'] == 1623)}
    return out, res


def pwm_from_command(steer, speed, cal, center, neutral):
    """シミュレーターの操作(舵 rad・速さ m/s)→ PWM [µs](収集器と同じ区分線形: 中心から両端 1000 / 2000 まで)"""
    lo, hi = float(cal['pwm_min_us']), float(cal['pwm_max_us'])
    s = float(cal.get('steering_sign', 1)) * steer / float(cal['max_steering_rad'])
    s = max(-1.0, min(1.0, s))
    ps = center + s * ((hi - center) if s >= 0 else (center - lo))
    f = speed / float(cal['max_speed_mps']) if speed >= 0 else speed / float(cal['max_reverse_speed_mps'])
    f = max(-1.0, min(1.0, f))
    d = f * (1.0 if float(cal.get('erpm_forward_sign', -1)) > 0 else -1.0)
    pt = neutral + d * ((hi - neutral) if d >= 0 else (neutral - lo))
    return ps, pt


def pwm_check(dataset):
    """1 シーン(学習の最初)で PWM の約束を確かめる"""
    root = DS / dataset
    scene = load_json(root / 'splits.json')['train'][0]
    sdir = root / 'simulation' / 'scenes' / scene
    sess = load_json(sdir / 'session.json')
    cal = sess['calibration']
    own = (float(sess['model_config']['steer_center_us']), float(sess['model_config']['throttle_neutral_us']))
    conv = {'1441/1496': (1441.0, 1496.0), '1442/1498': (1442.0, 1498.0)}
    other_key = '1442/1498' if own == (1441.0, 1496.0) else '1441/1496'
    rows = [json.loads(line) for line in (sdir / 'frames.jsonl').open(encoding='utf-8')]
    applied = np.array([r['applied_pwm_us'] for r in rows], np.float64)
    own_pwm = np.array([pwm_from_command(r['command_si']['steering'], r['command_si']['speed'], cal, *own) for r in rows])
    oth_pwm = np.array([pwm_from_command(r['command_si']['steering'], r['command_si']['speed'], cal, *conv[other_key])
                        for r in rows])
    rebuild_err = np.abs(own_pwm - applied).max(axis=0)
    cross = np.abs(oth_pwm - own_pwm).max(axis=0)
    # aligned の正解(future_actions[k, h])が記録の PWM のどのフレームか(h = 0 が k + o)
    with np.load(root / 'aligned' / (scene + '.npz')) as z:
        fa = z['future_actions'].astype(np.float64)
        fv = z['future_valid'].astype(bool) & (z['mode'] != 0)
    n, H = fa.shape[0], fa.shape[1]
    label = {}
    for o in (0, 1):
        errs = []
        for k in np.nonzero(fv)[0]:
            if k + o + H - 1 < len(applied):
                errs.append(np.abs(fa[k] - applied[k + o:k + o + H]).max())
        label[str(o)] = round(float(max(errs)), 4) if errs else None
    best_o = min((o for o in label if label[o] is not None), key=lambda o: label[o])
    relabelled = bool(load_json(root / 'dataset_card.json').get('relabel'))
    ok_conv = bool(cross[0] <= PWM_LIMIT_STEER + 1e-6 and cross[1] <= PWM_LIMIT_THR + 1e-6)
    return {'scene': scene, 'own_convention': '%g/%g' % own, 'other_convention': other_key,
            'calibration': {k: cal[k] for k in ('max_steering_rad', 'max_speed_mps', 'max_reverse_speed_mps', 'pwm_min_us',
                                                'pwm_max_us', 'steering_sign', 'erpm_forward_sign')},
            'rebuild_vs_applied_max_us': [round(float(x), 4) for x in rebuild_err],
            'cross_convention_max_us': [round(float(x), 4) for x in cross],
            'label_vs_applied_max_us_by_offset': label, 'label_offset': int(best_o), 'relabelled_dataset': relabelled,
            'min_speed_mps': round(min(r['command_si']['speed'] for r in rows), 4),
            'ok_rebuild': bool(rebuild_err.max() <= 0.01), 'ok_within_1_2us': ok_conv,
            'needs_relabel_over_5us': bool(cross.max() > PWM_RELABEL_US),
            'ok_label_equals_applied': (bool(label[best_o] is not None and label[best_o] <= 0.01) if not relabelled else None)}


def source_check(name):
    root = DS / name
    card = load_json(root / 'dataset_card.json')
    sp = load_json(root / 'splits.json')
    scenes = [s for k in ('train', 'val', 'test') for s in sp.get(k, [])]
    lens, modes = [], {}
    for s in scenes:
        with np.load(root / 'aligned' / (s + '.npz')) as z:
            m = z['mode']
            lens.append(len(m))
            for v, c in zip(*np.unique(m, return_counts=True)):
                modes[str(int(v))] = modes.get(str(int(v)), 0) + int(c)
    sim = name != REAL
    res = {'scenes': {k: len(sp.get(k, [])) for k in ('train', 'val', 'test')}, 'frames': int(sum(lens)),
           'card_frames': card.get('frames'), 'frames_per_scene': sorted(set(lens)) if sim else [min(lens), max(lens)],
           'aligned_mode_frames': dict(sorted(modes.items(), key=lambda kv: int(kv[0]))),
           'card_mode_frames': {k: v for k, v in (card.get('mode_frames') or {}).items() if v},
           'rl_state': (card.get('rl') or {}).get('state') if isinstance(card.get('rl'), dict) else None,
           'relabel': (card.get('relabel') or {}).get('schema'),
           'label_mask_frames': (card.get('label_mask') or {}).get('masked_frames'),
           'mode_ids': sorted(int(m['id']) for m in card.get('modes', [])),
           'mode_names': {str(m['id']): m['name'] for m in card.get('modes', [])}}
    allowed = {'0', '8'} if sim else {'4', '5'}
    res['ok'] = bool(res['frames'] == card.get('frames') and set(modes) <= allowed
                     and (not sim or res['frames_per_scene'] == [600])
                     and (name not in MODES5 or res['rl_state'] == 'accepted')
                     and (name != D1 or res['relabel'] == 'jf-sim-dart-relabel/v1'))
    return res


# ---------------------------------------------------------------- 腕
def links_check(arm_root, sources):
    """シンボリックリンクが元のデータセットのファイル・フォルダーを指すか・全フレームの画像のパスが開けるか"""
    src_roots = {str((DS / s).resolve()) for s, _ in sources}
    bad, n_aligned, n_dirs, n_imgs, missing_imgs = [], 0, 0, 0, []
    for p in sorted((arm_root / 'aligned').iterdir()):
        n_aligned += 1
        tgt = os.path.realpath(p)
        if not p.is_symlink() or not os.path.isfile(tgt) or not any(tgt.startswith(r + '/aligned/') for r in src_roots):
            bad.append(str(p))
    for sensor in sorted((arm_root / 'samples').iterdir()):
        for p in sorted(sensor.iterdir()):
            n_dirs += 1
            tgt = os.path.realpath(p)
            if not p.is_symlink() or not os.path.isdir(tgt) or not any(tgt.startswith(r + '/samples/') for r in src_roots):
                bad.append(str(p))
    sp = load_json(arm_root / 'splits.json')
    for split in ('train', 'val', 'test'):
        for s in sp.get(split, []):
            with np.load(arm_root / 'aligned' / (s + '.npz')) as z:
                files = z['cam_files']
            for f in files:
                n_imgs += 1
                if not os.path.exists(arm_root / str(f)):
                    missing_imgs.append(str(f))
    size = 0
    for dp, dn, fn in os.walk(arm_root):        # リンクはたどらない(リンクそのものの大きさ)
        for f in fn:
            size += os.lstat(os.path.join(dp, f)).st_size
    return {'aligned_links': n_aligned, 'sample_dir_links': n_dirs, 'bad_links': bad[:20], 'n_bad_links': len(bad),
            'images_checked': n_imgs, 'images_missing': len(missing_imgs), 'missing_examples': missing_imgs[:5],
            'folder_bytes_without_following_links': size,
            'ok': bool(not bad and not missing_imgs and size < SAMPLE_MAX_FOLDER_MB * 2 ** 20)}


def arm_check(arm, proto, src_idx, src_info, recover_mask):
    root = ARMS_DIR / arm
    sources = ARMS[arm]
    exp_q = proto['arms_Q1'][arm] if arm in proto['arms_Q1'] else proto['arms_Q2'][arm]
    res = {'path': str(root), 'combined_json': load_json(root / 'combined.json'), 'splits': {}, 'checks': {},
           'files_sha256': {fn: sha256(root / fn) for fn in ('splits.json', 'dataset_card.json', 'combined.json')}}
    sp = load_json(root / 'splits.json')
    # 期待するシーンの並び(元の splits.json の順・引数の順)
    want_scenes = {'train': [], 'val': [], 'test': []}
    for s, which in sources:
        ssp = load_json(DS / s / 'splits.json')
        for k in which:
            want_scenes[k] += ssp.get(k, [])
    res['checks']['scene_lists_equal_sources'] = bool(all(sp.get(k, []) == want_scenes[k] for k in want_scenes))
    for split in ('train', 'val'):
        idx, info = ds_index(root, split)
        want = []
        for s, which in sources:
            if split in which:
                want += src_idx[(s, split)]
        info['sum_of_sources'] = len(want)
        info['index_equal_sources'] = (idx == want)
        info['impact_masked_sum_of_sources'] = sum(src_info[(s, split)]['impact_masked'] for s, w in sources if split in w)
        info['genre0_samples'] = int(info['mode_counts'].get('0', 0))
        rm = [(s, f) for s, f in idx if s in recover_mask and f in recover_mask[s]]
        info['recover_masked_frames_in_index'] = len(rm)
        # 正解の窓(k + 1〜k + 10。PWM の確かめの label_offset 1)と観測 k 自身に、乱した操作が 1 つも入っていない
        bad_win = [(s, f) for s, f in idx if s in L0 and any(f <= j <= f + 10 for j in L0[s])]
        info['samples_with_perturbed_action_in_label_window'] = len(bad_win)
        info['recover_samples_checked'] = sum(1 for s, _ in idx if s in L0)
        per_src = {}
        for s, which in sources:
            if split in which:
                per_src[s] = len(src_idx[(s, split)])
        info['samples_per_source'] = per_src
        res['splits'][split] = info
        res['_idx_' + split] = idx
    tr, va = res['splits']['train'], res['splits']['val']
    exp_scenes = exp_q.get('train_val_scenes')
    if arm == 'C-DIV':
        exp_scenes = [41 + 60, 11]
        exp_samples = {'train': exp_q['stride3_samples_expected']['train'], 'val': exp_q['stride3_samples_expected']['val']}
        exp_modes = {'4': 2228, '5': 3684, '8': exp_q['stride3_samples_expected']['sim_train']}
    else:
        exp_samples = exp_q['stride3_samples_expected']
        exp_modes = {'8': exp_samples['train']}
    c = res['checks']
    c['scenes_expected'] = bool([tr['scenes'], va['scenes']] == list(exp_scenes) and tr['scenes'] == tr['scenes_in_split']
                                and va['scenes'] == va['scenes_in_split'])
    c['samples_expected'] = bool(tr['samples'] == exp_samples['train'] and va['samples'] == exp_samples['val'])
    c['samples_equal_sum_of_sources'] = bool(tr['samples'] == tr['sum_of_sources'] and va['samples'] == va['sum_of_sources'])
    c['index_equal_sources'] = bool(tr['index_equal_sources'] and va['index_equal_sources'])
    c['impact_masked_equal_sources'] = bool(tr['impact_masked'] == tr['impact_masked_sum_of_sources']
                                            and va['impact_masked'] == va['impact_masked_sum_of_sources'])
    c['genre0_samples_zero'] = bool(tr['genre0_samples'] == 0 and va['genre0_samples'] == 0)
    c['num_modes_9'] = bool(tr['num_modes'] == 9 and va['num_modes'] == 9)
    c['train_mode_counts_expected'] = bool(tr['mode_counts'] == exp_modes)
    c['recover_masked_frames_excluded'] = bool(tr['recover_masked_frames_in_index'] == 0
                                               and va['recover_masked_frames_in_index'] == 0)
    c['recover_label_windows_clean'] = bool(tr['samples_with_perturbed_action_in_label_window'] == 0
                                            and va['samples_with_perturbed_action_in_label_window'] == 0)
    c['not_used_datasets_absent'] = bool(not any(n in [s for s, _ in sources] for n in NOT_USED))
    res['links'] = links_check(root, sources)
    c['links_ok'] = res['links']['ok']
    if arm in ('S-DIV', 'S-MOD'):
        per_mode = [tr['samples_per_source'][d] for d in MODES5]
        res['per_mode_train'] = per_mode
        c['per_mode_train_expected'] = bool(per_mode == proto['data']['accepted_modes']['stride3_samples']['per_mode_train'])
        if arm == 'S-DIV':
            c['D1_D2_samples_expected'] = bool(
                tr['samples_per_source'][D1] == proto['data']['dart_D1']['stride3_samples']['train']
                and va['samples_per_source'][D1] == proto['data']['dart_D1']['stride3_samples']['val']
                and tr['samples_per_source'][D2] == proto['data']['margin_D2']['stride3_samples']['train']
                and va['samples_per_source'][D2] == proto['data']['margin_D2']['stride3_samples']['val'])
            share = {k: round(v / tr['samples'], 3) for k, v in
                     {'5 モード': sum(per_mode), 'D1 DART': tr['samples_per_source'][D1],
                      'D2 余裕の線': tr['samples_per_source'][D2]}.items()}
            res['source_share_of_train_samples'] = share
            c['source_share_expected'] = bool(share == proto['arms_Q1']['S-DIV']['source_share_of_train_samples'])
        card = load_json(root / 'dataset_card.json')
        c['modes_table_from_first_source'] = bool(card['modes'] == load_json(DS / MODES5[0] / 'dataset_card.json')['modes'])
    return res


def cdiv_extra(res, src_idx, src_info, proto):
    """C-DIV だけの確かめ(実機の分けが A と同じ・リンク先・シミュレーターの分け・action_reference)"""
    c = res['checks']
    tr_idx, va_idx = res['_idx_train'], res['_idx_val']
    real_val = src_idx[(REAL, 'val')]
    real_train = src_idx[(REAL, 'train')]
    real_scenes = set(s for k in ('train', 'val') for s in load_json(DS / REAL / 'splits.json')[k])
    a_res = [load_json(REPO / 'data' / 'runs' / r / 'result.json') for r in A_RUNS]
    c_res = [load_json(REPO / 'data' / 'runs' / r / 'result.json') for r in C_RUNS]
    # C(simreal_mix_20261013)の検証の (シーン, フレーム)
    c_val_idx, c_val_info = ds_index(DS / C_MIX, 'val')
    extra = {
        'real_val_samples': len(va_idx), 'A_val_samples': [r['val_samples'] for r in a_res],
        'C_val_samples': [r['val_samples'] for r in c_res] + [c_val_info['samples']],
        'real_train_samples_in_CDIV': sum(1 for s, _ in tr_idx if s in real_scenes),
        'A_train_samples': [r['train_samples'] for r in a_res],
        'sim_train_samples_in_CDIV': sum(1 for s, _ in tr_idx if s not in real_scenes),
        'sim_val_samples_in_CDIV': sum(1 for s, _ in va_idx if s not in real_scenes),
        'A_impact_masked': [r['impact_masked_frames'] for r in a_res],
        'CDIV_impact_masked': {'train': res['splits']['train']['impact_masked'], 'val': res['splits']['val']['impact_masked']},
        'A_train_mode_counts': [r['train_mode_counts'] for r in a_res],
        'C_train_mode_counts': [r['train_mode_counts'] for r in c_res],
    }
    c['real_val_same_frames_as_A'] = bool(va_idx == real_val and len(va_idx) == 1559
                                          and all(r['val_samples'] == 1559 for r in a_res))
    c['val_same_frames_as_C'] = bool(va_idx == c_val_idx)
    c['real_train_same_frames_as_A'] = bool([x for x in tr_idx if x[0] in real_scenes] == real_train
                                            and len(real_train) == 5912 and all(r['train_samples'] == 5912 for r in a_res))
    c['impact_masked_same_as_A'] = bool(all(r['impact_masked_frames'] == extra['CDIV_impact_masked'] for r in a_res))
    c['sim_train_samples_11391'] = bool(extra['sim_train_samples_in_CDIV'] == 11391)
    c['no_sim_in_val'] = bool(extra['sim_val_samples_in_CDIV'] == 0)
    # 実機の aligned のリンク先が clean_driving_20261011 のファイルそのもの
    root = ARMS_DIR / 'C-DIV'
    same_files, diff = 0, []
    for s in sorted(real_scenes):
        a = root / 'aligned' / (s + '.npz')
        b = DS / REAL / 'aligned' / (s + '.npz')
        if a.is_symlink() and os.path.samefile(a, b):
            same_files += 1
        else:
            diff.append(s)
        for sensor in ('CAM_FRONT', 'TOF_FRONT'):
            if (DS / REAL / 'samples' / sensor / s).exists() and not os.path.samefile(root / 'samples' / sensor / s,
                                                                                         DS / REAL / 'samples' / sensor / s):
                diff.append(s + '/' + sensor)
    extra['real_aligned_same_files'] = same_files
    extra['real_link_differences'] = diff
    c['real_aligned_same_files_as_clean_driving'] = bool(same_files == 52 and not diff)
    # シミュレーターのシーン = 5 モードの学習の 12 シーンずつ(モードの順)
    want = [s for d in MODES5 for s in load_json(DS / d / 'splits.json')['train']]
    got = [s for s in load_json(root / 'splits.json')['train'] if s not in real_scenes]
    c['sim_scenes_are_5mode_train_12_each'] = bool(got == want and len(got) == 60)
    sim_val = [s for d in MODES5 for s in load_json(DS / d / 'splits.json')['val']]
    c['5mode_val_scenes_not_in_CDIV'] = bool(not (set(sim_val) & set(s for k in ('train', 'val', 'test')
                                                                        for s in load_json(root / 'splits.json').get(k, []))))
    # ジャンルの表 = simreal_mix_20261013 と同じ
    c['modes_table_same_as_C'] = bool(load_json(root / 'dataset_card.json')['modes']
                                      == load_json(DS / C_MIX / 'dataset_card.json')['modes'])
    # action_reference
    extra['action_reference_CDIV'] = res['splits']['train'].get('action_reference')
    extra['action_reference_A_C'] = [[r['steer_center_us'], r['throttle_neutral_us']] for r in a_res + c_res]
    extra['action_reference_real_only'] = src_info[(REAL, 'train')].get('action_reference')
    c['action_reference_recorded'] = extra['action_reference_CDIV'] is not None
    extra['action_reference_same_as_A_C'] = bool(extra['action_reference_CDIV'] == [1441.0, 1496.0])
    res['cdiv'] = extra


def main():
    t0 = time.time()
    sha = sha256(PROTOCOL)
    if sha != PROTOCOL_SHA.read_text(encoding='utf-8').split()[0]:
        raise SystemExit('protocol.json の sha256 が固定したものと違います。止めます')
    proto = load_json(PROTOCOL)
    out = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'protocol_sha256': sha, 'settings': KW,
           'tool': str(Path(__file__).resolve()), 'tool_sha256': sha256(__file__), 'sources': {}, 'pwm': {}, 'arms': {},
           'ok': {}}
    log('元のデータセット')
    for name in MODES5 + [D1, D2, REAL]:
        out['sources'][name] = source_check(name)
    log('離脱復活の外すフレームの数え直し')
    recover_mask, out['recover_mask'] = recover_masked_frames()
    log('PWM の約束')
    for name in MODES5 + [D1, D2]:
        out['pwm'][name] = pwm_check(name)
    log('元ごとのサンプル')
    src_idx, src_info = {}, {}
    for name in MODES5 + [D1, D2, REAL]:
        for split in ('train', 'val'):
            src_idx[(name, split)], src_info[(name, split)] = ds_index(DS / name, split)
        out['sources'][name]['samples'] = {sp: src_info[(name, sp)] for sp in ('train', 'val')}
    for arm in ('S-DIV', 'S-MOD', 'C-DIV'):
        log('腕 %s' % arm)
        res = arm_check(arm, proto, src_idx, src_info, recover_mask)
        if arm == 'C-DIV':
            cdiv_extra(res, src_idx, src_info, proto)
        for k in [k for k in res if k.startswith('_idx_')]:
            del res[k]
        res['ok'] = bool(all(res['checks'].values()))
        out['arms'][arm] = res
    ok = out['ok']
    ok['sources_frames_genres_accepted'] = all(v['ok'] for v in out['sources'].values())
    ok['recover_mask_recount_1623'] = out['recover_mask']['ok']
    ok['pwm_rebuild_equals_applied'] = all(v['ok_rebuild'] for v in out['pwm'].values())
    ok['pwm_conventions_within_1_2us'] = all(v['ok_within_1_2us'] for v in out['pwm'].values())
    ok['pwm_no_relabel_needed'] = not any(v['needs_relabel_over_5us'] for v in out['pwm'].values())
    ok['pwm_labels_equal_applied_not_relabelled'] = all(v['ok_label_equals_applied'] for v in out['pwm'].values()
                                                        if v['ok_label_equals_applied'] is not None)
    for arm, r in out['arms'].items():
        ok['arm_' + arm] = r['ok']
    out['all_ok'] = bool(all(ok.values()))
    out['seconds'] = round(time.time() - t0, 1)
    write_json(OUT, out)
    print(json.dumps({'all_ok': out['all_ok'], 'ok': ok,
                      'arms': {a: {'checks': r['checks'], 'train': r['splits']['train']['samples'],
                                   'val': r['splits']['val']['samples']} for a, r in out['arms'].items()}},
                     ensure_ascii=False, indent=1))
    return 0 if out['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
