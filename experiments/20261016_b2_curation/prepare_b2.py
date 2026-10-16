#!/usr/bin/env python3
"""B2 の準備の段(CPU だけ): protocol.json の curation を実装し、GPU の学習の前に確かめる

    cd ~/jetformer
    CPU="/home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 env CUDA_VISIBLE_DEVICES="
    PY=/home/jetson/miniconda/envs/jetracer/bin/python
    $CPU $PY -u experiments/20261016_b2_curation/prepare_b2.py derived   # ジャンルを直したデータセット(ハードリンク)
    $CPU $PY -u experiments/20261016_b2_curation/prepare_b2.py checks    # 表の作り直し・ジャンル別の数・外したフレームの一覧・保留の分け方
    $PY -u experiments/20261016_b2_curation/prepare_b2.py smoke          # tools/train.py の CPU の動作確認(各段を jf-heavy の CPU で起動)
    $PY experiments/20261016_b2_curation/prepare_b2.py check-log <train.log>   # 学習の最初の数行が protocol と同じか(GPU の段でも使う)

作るもの(どれも自分のフォルダーだけ。生データ・既存のデータセットは書き換えない):
  data/derived/b2_curation_20261016/clean_driving_20261011_b2genre/
      ジャンルを直したデータセット。元の clean_driving_20261011 の全ファイルへのハードリンク。新しく書くのは
      152405・152707 の aligned 12 個(mode 5 → 4)・v1.0-*/scene.json・dataset_card.json だけ。
      remap_b2.npz と同じ付け替えを、データセットの形にしたもの(--mode-remap-npz を読まない道具でも B2 のジャンルになる)。
      固定した学習のコマンド(protocol.json の model.train_cmd)は、元のデータセット + remap_b2.npz のまま変えない。
      どちらで学習しても、学習・検証のサンプルとジャンルが同じになることを checks で確かめる。
      注意: リンクしたファイルは元のデータセットと中身を共有する。どちらのファイルも、その場で書き換えないこと。
  data/derived/b2_curation_20261016/derived_manifest.json   作り方・リンクの数・新しいファイルの sha256
  data/derived/b2_curation_20261016/dropped_frames.csv      外したフレームの一覧(理由・どの通過か・B で使っていたか)
  /home/jetson/jetformer_reports/b2_curation_20261016/data/prepare_checks.json   確かめた結果
  /home/jetson/jetformer_reports/b2_curation_20261016/smoke/                       CPU の動作確認のログと結果
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
sys.path.insert(0, str(HERE))
import make_b2_tables as M  # noqa: E402

REPORT = Path('/home/jetson/jetformer_reports/b2_curation_20261016')
PROTOCOL = REPORT / 'protocol.json'
PROTOCOL_SHA = REPORT / 'protocol.sha256'
CHECKS_JSON = REPORT / 'data' / 'prepare_checks.json'
SMOKE_DIR = REPORT / 'smoke'
OUT_DIR = M.OUT_DIR                                     # data/derived/b2_curation_20261016
SRC_NAME = M.DATASET                                    # clean_driving_20261011
DERIVED_NAME = SRC_NAME + '_b2genre'
DERIVED_ROOT = OUT_DIR / DERIVED_NAME
MANIFEST = OUT_DIR / 'derived_manifest.json'
DROPPED_CSV = OUT_DIR / 'dropped_frames.csv'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
JF_HEAVY = '/home/jetson/bin/jf-heavy'
REASON_NAME = {1: '速い進入(6200 eRPM 以上)の手前の窓 [ta-2, ta+1]', 2: 'R1 で当たった通過(通過ごと)'}
A_RUNS = {'A_s0': 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
          'A_s1': 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
          'B': 'clean_driving_20261011-nocmd-s3-20261011-024626',
          'A_s2': 'realacc_20261015-v0-s2', 'A_s3': 'realacc_20261015-v0-s3'}
PRED_TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
JST = datetime.timezone(datetime.timedelta(hours=9))


def now_jst():
    return datetime.datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S')


def sha256(path):
    return M.sha256(path)


def save_json(path, obj):
    M._save_json(path, obj)


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


# ================================================================ 1. ジャンルを直したデータセット(ハードリンク)

def relabel_mode(mode, to=M.RELABEL_TO):
    """aligned の mode [N] → 付け替えた mode。未設定(0)はそのまま(学習に使わないフレームのまま)、ほかは to"""
    mode = np.asarray(mode)
    out = mode.copy()
    out[mode != 0] = to
    return out.astype(mode.dtype)


def _is_relabel_scene(scene, relabel_logs):
    return str(scene)[:15] in relabel_logs


def _dst_rel(rel, ver_src, ver_dst):
    """元のデータセットの相対パス → 作るデータセットの相対パス(v1.0-<名前> のフォルダー名だけ変える)"""
    parts = Path(rel).parts
    if parts and parts[0] == ver_src:
        return str(Path(ver_dst, *parts[1:]))
    return str(rel)


def derived_plan(src_root, relabel_logs=M.RELABEL_LOGS):
    """作るデータセットのファイルの一覧: (元の相対パス, 作る相対パス, 'link' | 'aligned' | 'scene' | 'card')"""
    src_root = Path(src_root)
    card = json.loads((src_root / 'dataset_card.json').read_text(encoding='utf-8'))
    ver_src = card['version']
    ver_dst = 'v1.0-' + derived_name(card['name'])
    plan = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames.sort()
        for fn in sorted(filenames):
            rel = str((Path(dirpath) / fn).relative_to(src_root))
            dst = _dst_rel(rel, ver_src, ver_dst)
            if rel == 'dataset_card.json':
                kind = 'card'
            elif rel == str(Path(ver_src, 'scene.json')):
                kind = 'scene'
            elif Path(rel).parts[0] == 'aligned' and _is_relabel_scene(Path(rel).stem, relabel_logs):
                kind = 'aligned'
            else:
                kind = 'link'
            plan.append((rel, dst, kind))
    return plan


def derived_name(src_name):
    return src_name + '_b2genre'


def _write_npz_compressed(path, arrays):
    """元と同じ np.savez_compressed(キーの順も同じ)。書き終えてから置き換える"""
    path = Path(path)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _relabeled_scene_table(scenes, relabel_logs, to, mode_name):
    """nuScenes の scene.json の表 → 付け替えたシーンの mode_id・mode_name・mode_segments を直したもの"""
    out = []
    for s in scenes:
        s = dict(s)
        if _is_relabel_scene(s.get('name', ''), relabel_logs):
            if s.get('mode_id', 0) != 0:
                s['mode_id'], s['mode_name'] = to, mode_name
            segs = []
            for g in s.get('mode_segments') or []:
                g = dict(g)
                if g.get('mode_id', 0) != 0:
                    g['mode_id'], g['mode_name'] = to, mode_name
                segs.append(g)
            if 'mode_segments' in s:
                s['mode_segments'] = segs
        out.append(s)
    return out


def build_derived(src_root=None, dst_root=None, relabel_logs=M.RELABEL_LOGS, to=M.RELABEL_TO, manifest_path=None,
                  created_at=None):
    """ジャンルを直したデータセットを作る(済んでいて確かめに通れば何もしない)。

    作り方: <dst>.partial に作ってから名前を変える(電源断で半端なデータセットを残さない。半端な .partial は消して作り直す。
    中身はリンクと自分で書いたファイルだけなので、消しても元のデータセットは変わらない)。
    リンクできないとき(別のファイルシステムなど)は、コピーせずに止める(ディスクを使わない約束)。"""
    from jetformer import config
    src_root = Path(src_root or (config.DATASET_DIR / SRC_NAME))
    dst_root = Path(dst_root or DERIVED_ROOT)
    manifest_path = Path(manifest_path or MANIFEST)
    if dst_root.exists():
        v = verify_derived(src_root, dst_root, relabel_logs, to)
        if v['ok']:
            print('作ってあります(確かめに通りました): %s' % dst_root)
            return v
        raise SystemExit('作ってあるデータセットが確かめに通りません(手で調べてください。自動では消しません): %s\n%s'
                         % (dst_root, json.dumps(v['problems'][:10], ensure_ascii=False)))
    src_card = json.loads((src_root / 'dataset_card.json').read_text(encoding='utf-8'))
    ver_src = src_card['version']
    new_name = derived_name(src_card['name'])
    ver_dst = 'v1.0-' + new_name
    mode_name = {m['id']: m['name'] for m in src_card.get('modes', [])}.get(to, str(to))
    plan = derived_plan(src_root, relabel_logs)
    changed_src = [rel for rel, _, kind in plan if kind != 'link']
    before = {rel: sha256(src_root / rel) for rel in changed_src}       # 元のファイルが変わらないことを後で確かめる
    tmp = dst_root.with_name(dst_root.name + '.partial')
    if tmp.exists():
        print('前回の半端なフォルダーを消して作り直します: %s' % tmp)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_link = 0
    for rel, dst, kind in plan:
        out = tmp / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == 'link':
            os.link(src_root / rel, out)               # 失敗したら止める(コピーしない)
            n_link += 1
    # 付け替えるシーンの aligned(mode だけ変える。ほかの配列は同じ値・同じ順)
    relabeled = {}
    mode_frames = {}
    for rel, dst, kind in plan:
        if Path(rel).parts[0] != 'aligned':
            continue
        with np.load(src_root / rel, allow_pickle=False) as z:
            if kind == 'aligned':
                arrays = {k: z[k] for k in z.files}
                old = arrays['mode']
                arrays['mode'] = relabel_mode(old, to)
                _write_npz_compressed(tmp / dst, arrays)
                relabeled[Path(rel).stem] = {'frames': int(len(old)), 'from': {str(k): int(v) for k, v in
                                                                              zip(*np.unique(old, return_counts=True))},
                                             'to': {str(k): int(v) for k, v in
                                                    zip(*np.unique(arrays['mode'], return_counts=True))}}
                m = arrays['mode']
            else:
                m = z['mode'] if 'mode' in z.files else np.zeros(0, np.int16)
        for k, v in zip(*np.unique(m, return_counts=True)):
            mode_frames[str(int(k))] = mode_frames.get(str(int(k)), 0) + int(v)
    # nuScenes の scene.json(コンパクトな JSON。元の書き方と同じ)
    scenes = json.loads((src_root / ver_src / 'scene.json').read_text(encoding='utf-8'))
    _write_text(tmp / ver_dst / 'scene.json',
                json.dumps(_relabeled_scene_table(scenes, relabel_logs, to, mode_name), ensure_ascii=False))
    # データセットのカード
    card = dict(src_card)
    card['name'] = new_name
    card['version'] = ver_dst
    card['created_at'] = created_at or datetime.datetime.now().isoformat(timespec='seconds')
    card['mode_frames'] = dict(sorted(mode_frames.items(), key=lambda kv: int(kv[0])))
    card['derived_from'] = {
        'dataset': src_card['name'], 'created_at': src_card.get('created_at'),
        'how': '元のデータセットの全ファイルへのハードリンク。新しく書いたのは付け替えたシーンの aligned・%s/scene.json・'
               'dataset_card.json だけ(リンクしたファイルは元と中身を共有する。その場で書き換えないこと)' % ver_dst,
        'tool': 'experiments/20261016_b2_curation/prepare_b2.py derived', 'experiment': 'b2_curation_20261016'}
    card['genre_fix'] = {
        'logs': list(relabel_logs), 'to': to, 'to_name': mode_name, 'scenes': sorted(relabeled),
        'why': '生ログの log.json の mode_id は 4(高速走行)で、シーンの scene.json だけが 5 だった。前進 eRPM の中央値・p90 も'
               'ジャンル 4 の夜のログより速い(protocol.json の curation.genre_fix)',
        'same_as': 'data/derived/b2_curation_20261016/remap_b2.npz(--mode-remap-npz)と同じ付け替え',
        'not_changed': 'splits.json(学習・検証のシーンの分け方)・画像・ToF・操作値・未来の操作値・時刻は元と同じ'}
    save_json(tmp / 'dataset_card.json', card)
    after = {rel: sha256(src_root / rel) for rel in changed_src}
    if before != after:
        raise SystemExit('元のデータセットのファイルが変わりました(ありえない): %s' % [r for r in before if before[r] != after[r]])
    os.replace(tmp, dst_root)
    v = verify_derived(src_root, dst_root, relabel_logs, to)
    manifest = {
        'created_at': card['created_at'], 'src': str(src_root), 'dst': str(dst_root), 'name': new_name,
        'linked_files': n_link, 'new_files': {dst: sha256(dst_root / dst) for _, dst, kind in plan if kind != 'link'},
        'relabeled_scenes': relabeled, 'mode_frames': card['mode_frames'], 'src_mode_frames': src_card.get('mode_frames'),
        'src_changed_files_sha256_unchanged': before, 'verify': {k: v[k] for k in v if k != 'problems'},
        'problems': v['problems']}
    save_json(manifest_path, manifest)
    if not v['ok']:
        raise SystemExit('作ったデータセットが確かめに通りません: %s' % json.dumps(v['problems'][:10], ensure_ascii=False))
    print('作りました: %s(リンク %d・新しいファイル %d)' % (dst_root, n_link, len(manifest['new_files'])))
    return v


def verify_derived(src_root, dst_root, relabel_logs=M.RELABEL_LOGS, to=M.RELABEL_TO):
    """作ったデータセットを確かめる: (1) リンクのファイルは元と同じ inode(= 同じ中身)、(2) 新しいファイルは別の inode、
    (3) 付け替えた aligned は mode 以外の配列が元と同じで mode = relabel_mode(元)、(4) 余分・足りないファイルが無い、
    (5) splits.json が元と同じ inode(学習・検証の分け方が同じ)"""
    src_root, dst_root = Path(src_root), Path(dst_root)
    plan = derived_plan(src_root, relabel_logs)
    problems = []
    n_link = n_new = 0
    expected = set()
    for rel, dst, kind in plan:
        expected.add(dst)
        p = dst_root / dst
        if not p.exists():
            problems.append({'missing': dst})
            continue
        same = os.stat(p).st_ino == os.stat(src_root / rel).st_ino and os.stat(p).st_dev == os.stat(src_root / rel).st_dev
        if kind == 'link':
            n_link += 1
            if not same:
                problems.append({'not_linked': dst})
            continue
        n_new += 1
        if same:
            problems.append({'linked_but_should_be_new': dst})
        if kind == 'aligned':
            with np.load(src_root / rel, allow_pickle=False) as a, np.load(p, allow_pickle=False) as b:
                if list(a.files) != list(b.files):
                    problems.append({'keys_differ': dst})
                    continue
                for k in a.files:
                    x, y = a[k], b[k]
                    if k == 'mode':
                        if not np.array_equal(relabel_mode(x, to), y) or x.dtype != y.dtype:
                            problems.append({'mode_wrong': dst})
                    elif x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                        problems.append({'array_differs': dst, 'key': k})
    for dirpath, _, filenames in os.walk(dst_root):
        for fn in filenames:
            rel = str((Path(dirpath) / fn).relative_to(dst_root))
            if rel not in expected:
                problems.append({'extra': rel})
    sp_same = (dst_root / 'splits.json').exists() and os.stat(dst_root / 'splits.json').st_ino == os.stat(src_root / 'splits.json').st_ino
    if not sp_same:
        problems.append({'splits_not_same_inode': True})
    # scene.json: 付け替えたシーンだけ mode が to、ほかは元と同じ。カード: 名前・版・ジャンルの表
    src_card = json.loads((src_root / 'dataset_card.json').read_text(encoding='utf-8'))
    ver_src, ver_dst = src_card['version'], 'v1.0-' + derived_name(src_card['name'])
    try:
        a = json.loads((src_root / ver_src / 'scene.json').read_text(encoding='utf-8'))
        b = json.loads((dst_root / ver_dst / 'scene.json').read_text(encoding='utf-8'))
        name_to = {m['id']: m['name'] for m in src_card.get('modes', [])}.get(to, str(to))
        if b != _relabeled_scene_table(a, relabel_logs, to, name_to) or len(a) != len(b):
            problems.append({'scene_json_differs': True})
        if not any(_is_relabel_scene(s['name'], relabel_logs) and s.get('mode_id') == to for s in b):
            problems.append({'scene_json_not_relabeled': True})
        card = json.loads((dst_root / 'dataset_card.json').read_text(encoding='utf-8'))
        if (card.get('name') != derived_name(src_card['name']) or card.get('version') != ver_dst
                or card.get('modes') != src_card.get('modes') or 'genre_fix' not in card):
            problems.append({'card_wrong': True})
    except (OSError, ValueError, KeyError) as e:
        problems.append({'meta_unreadable': str(e)})
    return {'ok': not problems, 'linked': n_link, 'new': n_new, 'splits_same_inode': bool(sp_same), 'problems': problems}


# ================================================================ 2. 外したフレームの一覧(理由・どの通過か)

def attribute_drops(t_s, reason, passes):
    """外したフレームごとに、どの通過の窓か(make_b2_tables.drop_reason と同じ窓)。
    戻り値: 通過の番号 [N](passes の添え字。外していないフレームは -1)。窓が重なるときは起点 ta がいちばん近い通過"""
    t = np.asarray(t_s, np.float64)
    reason = np.asarray(reason)
    out = np.full(len(t), -1, np.int64)
    best = np.full(len(t), np.inf)
    for j, p in enumerate(passes):
        ta = float(p['ta_s'])
        e = p.get('entry_erpm')
        wins = []
        if e is not None and float(e) >= M.FAST_ERPM:
            wins.append((M.REASON_FAST, (t >= ta - M.BEFORE_S) & (t <= ta + M.AFTER_S)))
        if p.get('r1_collision'):
            end = M.collided_end(ta, p.get('tg_s'), p.get('r1_collision_t_s') or [])
            wins.append((M.REASON_COLL, (t >= ta - M.BEFORE_S) & (t <= end)))
        for r, w in wins:
            hit = w & (reason == r) & (np.abs(t - ta) < best)
            out[hit] = j
            best[hit] = np.abs(t - ta)[hit]
    return out


def pass_label(p):
    fast = p.get('entry_erpm') is not None and float(p['entry_erpm']) >= M.FAST_ERPM
    return 'fast+collision' if fast and p.get('r1_collision') else 'collision' if p.get('r1_collision') else 'fast'


# ================================================================ 3. 学習のログの確かめ(GPU の段でも使う)

LOG_PATTERNS = {
    'impact': re.compile(r'衝突の前後として外したフレーム: 学習 (\d+) / 検証 (\d+)'),
    'drop': re.compile(r'フレームの表: drop で外した 学習 (\d+) / 検証 (\d+)、衝突の前後の除外から戻した 学習 (\d+) / 検証 (\d+)'),
    'samples': re.compile(r'学習 (\d+) サンプル / 検証 (\d+) サンプル / 間引き 1/(\d+) / ジャンル数 (\d+) / device=(\w+)'),
    'genres': re.compile(r'ジャンル別の学習サンプル数: (\{.*\})'),
}


def parse_train_log(text):
    """tools/train.py の最初の数行を読む → {'impact': [学習, 検証], 'drop': [学習, 検証, 戻した学習, 戻した検証],
    'samples': [学習, 検証, 間引き, ジャンル数, device], 'genres': {名前: 数}}(無い行は入れない)"""
    import ast
    out = {}
    for line in text.splitlines():
        for k, pat in LOG_PATTERNS.items():
            if k in out:
                continue
            m = pat.search(line)
            if not m:
                continue
            if k == 'genres':
                out[k] = {str(a): int(b) for a, b in ast.literal_eval(m.group(1)).items()}
            elif k == 'samples':
                out[k] = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)]
            else:
                out[k] = [int(x) for x in m.groups()]
    return out


def compare_train_log(parsed, expected, mode_names, full=True):
    """読んだ行と protocol.json の model.expected_train_log を比べる → (合っているか, 違い [文字列])。
    full=False は --max-steps の動作確認(サンプル数・ジャンル別の行は切り詰めた数なので見ない)"""
    diffs = []
    want = {'impact': [expected['impact_masked_printed']['train'], expected['impact_masked_printed']['val']],
            'drop': [expected['dropped_printed']['train'], expected['dropped_printed']['val'], 0, 0]}
    if full:
        want['samples'] = [expected['train_samples'], expected['val_samples']]
        want['genres'] = {mode_names.get(int(k), k): int(v) for k, v in expected['genre_counts_train'].items()}
    for k, w in want.items():
        got = parsed.get(k)
        if got is None:
            diffs.append('%s の行がありません' % k)
            continue
        if k == 'samples':
            got = got[:2]
        if got != w:
            diffs.append('%s: ログ %s、protocol %s' % (k, got, w))
    return not diffs, diffs


def config_dataset_root():
    from jetformer import config
    return config.DATASET_DIR / SRC_NAME


def mode_names_from_card(root):
    card = json.loads((Path(root) / 'dataset_card.json').read_text(encoding='utf-8'))
    return {int(m['id']): m['name'] for m in card.get('modes', [])}


# ================================================================ 4. 学習のコマンド(protocol の文字列から作る)

def split_protocol_cmd(cmd):
    """protocol の train_cmd('cd ~/jetformer && <前置き> ... tools/train.py <引数>')→ (前置きの語, train.py の引数の語)"""
    head, _, rest = cmd.partition('&&')
    if 'cd ~/jetformer' not in head:
        raise ValueError('train_cmd が cd ~/jetformer で始まっていません')
    words = shlex.split(rest)
    i = words.index('tools/train.py')
    return words[:i + 1], words[i + 1:]


def set_arg(args, name, value=None):
    """引数の語の列で、--name の値を変える(無ければ足す)。value=None はフラグ(値なし)"""
    args = list(args)
    if name in args:
        i = args.index(name)
        if value is None:
            return args
        args[i + 1] = str(value)
        return args
    return args + ([name] if value is None else [name, str(value)])


def drop_arg(args, name, has_value=True):
    args = list(args)
    if name in args:
        i = args.index(name)
        del args[i:i + (2 if has_value else 1)]
    return args


def cpu_prefix(prefix, mem='1500M'):
    """GPU の前置き(jf-heavy --mem 3G --lock big --need-mb 3200 -- ... env ...)→ CPU の前置き(上限 1500M・鍵なし・CUDA を見せない)。
    nice・nothp.py・choom・MALLOC_ARENA_MAX の包みはそのまま(同じ包みで動くかも確かめる)"""
    p = list(prefix)
    if p[0] != JF_HEAVY:
        raise ValueError('前置きが jf-heavy で始まっていません: %s' % p[0])
    j = p.index('--')
    opts = p[1:j]
    opts = set_arg(opts, '--mem', mem)
    opts = set_arg(opts, '--lock', 'none')
    opts = set_arg(opts, '--need-mb', '1500')
    rest = p[j:]
    k = rest.index('env')
    rest = rest[:k + 1] + ['CUDA_VISIBLE_DEVICES='] + rest[k + 1:]
    return [p[0]] + opts + rest


def smoke_cmd(protocol, seed='0', out=None, data=None, drop_remap=True, epochs=None, max_steps=None, batch=None,
              workers=None, resume=False, mem='1500M'):
    prefix, args = split_protocol_cmd(protocol['model']['train_cmd'][seed])
    args = set_arg(args, '--out', out)
    if data:
        args = set_arg(args, '--data', data)
    if not drop_remap:
        args = drop_arg(args, '--mode-remap-npz')
    for name, v in (('--epochs', epochs), ('--max-steps', max_steps), ('--batch', batch), ('--workers', workers)):
        if v is not None:
            args = set_arg(args, name, v)
    if resume:
        args = set_arg(args, '--resume')
    return cpu_prefix(prefix, mem) + args


# ================================================================ 5. checks

def _git(*args):
    return subprocess.run(['git', '-C', str(REPO)] + list(args), capture_output=True, text=True).stdout.strip()


def _datasets(root, drop=None, remap=None):
    """tools/train.py と同じ読み込み(色の揺らぎなし)の学習・検証のデータセット"""
    from jetformer.train_data import JetFormerDataset, load_frame_table
    rl = {}
    if drop:
        rl['drop'] = load_frame_table(drop, 'drop')
    if remap:
        rl['mode_remap'] = load_frame_table(remap, 'mode')
    return {sp: JetFormerDataset(str(root), sp, stride=M.STRIDE, use_command=False, mask_impacts=True, impact_g=M.IMPACT_G, **rl)
            for sp in ('train', 'val')}


def _ds_summary(ds):
    return {sp: {'samples': len(d), 'impact_masked': d.impact_masked, 'dropped': d.dropped,
                 'genres': {str(k): v for k, v in sorted(d.mode_counts().items())},
                 'action_reference': list(d.action_reference())} for sp, d in ds.items()}


def _index_modes(d):
    return [(k, i, d.mode_of(k, i)) for k, i in d.index]


def check_tables(protocol):
    """make_b2_tables を一時フォルダーに作り直し、固定した表と同じ sha256 か"""
    frozen = protocol['files']
    with tempfile.TemporaryDirectory(prefix='b2tables_', dir=str(OUT_DIR)) as td:
        card = M.build(out_dir=Path(td))
        got = {n: sha256(Path(td) / n) for n in ('passes_b2.json', 'drop_b2.npz', 'remap_b2.npz')}
        frozen_card = json.loads((OUT_DIR / 'curation_card.json').read_text(encoding='utf-8'))
        same_card = all(card[k] == frozen_card[k] for k in ('rules', 'passes', 'dropped_passes', 'counts', 'per_scene'))
    want = {'passes_b2.json': frozen['passes_b2']['sha256'], 'drop_b2.npz': frozen['drop_b2']['sha256'],
            'remap_b2.npz': frozen['remap_b2']['sha256']}
    now = {n: sha256(OUT_DIR / n) for n in want}
    return {'rebuilt_sha256_equal': got == want, 'files_on_disk_equal_protocol': now == want, 'card_contents_equal': same_card,
            'sha256': want}


def dropped_frames_table(protocol):
    """外したフレームの一覧(CSV の行)と、通過ごと・理由ごとの数"""
    from jetformer import config
    from jetformer.train_data import impact_mask
    root = config.DATASET_DIR / SRC_NAME
    splits = json.loads((root / 'splits.json').read_text())
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}
    passes = json.loads((OUT_DIR / 'passes_b2.json').read_text(encoding='utf-8'))
    human = [p for p in passes if p['log'] in M.HUMAN_LOGS]
    with np.load(OUT_DIR / 'drop_b2.npz', allow_pickle=False) as z:
        d_scene, d_frame, d_drop, d_reason = z['scene'].astype(str), z['frame'], z['drop'], z['reason']
    rows = []
    per_pass = {}
    mism = 0
    for s in sorted(split_of):
        sel = d_scene == s
        order = np.argsort(d_frame[sel])
        fr, dr, rs = d_frame[sel][order], d_drop[sel][order], d_reason[sel][order]
        with np.load(root / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
            a = {k: z[k] for k in ('timestamps_us', 'mode', 'future_valid', 'vehicle_valid', 'imu')}
        n = len(a['timestamps_us'])
        if len(fr) != n or not np.array_equal(fr, np.arange(n)):
            raise SystemExit('drop_b2.npz のフレームがシーン %s の全フレームでありません' % s)
        ps = [p for p in human if p['log'] == s[:15]]
        if not ps:
            mism += int(dr.sum())                          # 通過の無いログのシーンに外すフレームがあってはいけない
            continue
        t0 = ps[0]['t0_us']
        t = (a['timestamps_us'] - t0) / 1e6
        again = M.drop_reason(t, ps)                       # 表を作った規則で作り直して一致を確かめる(全部のシーン)
        mism += int((again != rs).sum()) + int((dr != (rs > 0)).sum())
        if not dr.any():
            continue
        who = attribute_drops(t, rs, ps)
        imp = impact_mask(a['timestamps_us'], a['imu'], M.IMPACT_G)
        ok = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
        used_B = M.used_samples(a['future_valid'], a['vehicle_valid'], a['mode'], imp)
        g_b2 = M.remap_value(s)
        for i in np.nonzero(dr)[0]:
            p = ps[who[i]] if who[i] >= 0 else None
            rows.append({'scene': s, 'split': split_of[s], 'frame': int(i), 'timestamp_us': int(a['timestamps_us'][i]),
                         't_log_s': round(float(t[i]), 3), 'reason': int(rs[i]), 'reason_name': REASON_NAME[int(rs[i])],
                         'pass_log': p['log'] if p else '', 'pass_ta_s': p['ta_s'] if p else '',
                         't_minus_ta_s': round(float(t[i] - p['ta_s']), 3) if p else '',
                         'pass_entry_erpm': p['entry_erpm'] if p else '', 'pass_label': pass_label(p) if p else '',
                         'pass_r1_collision': int(bool(p['r1_collision'])) if p else '',
                         'pass_r1_strong': int(bool(p['r1_strong'])) if p else '',
                         'usable_frame': int(ok[i]), 'impact_masked_2p5g': int(imp[i]), 'stride_kept': int(i % M.STRIDE == 0),
                         'sample_in_B': int(used_B[i]), 'genre_B': int(a['mode'][i]),
                         'genre_B2': int(g_b2 if g_b2 >= 0 else a['mode'][i])})
            if p is not None:
                key = '%s@%.3f' % (p['log'], p['ta_s'])
                q = per_pass.setdefault(key, {'log': p['log'], 'ta_s': p['ta_s'], 'entry_erpm': p['entry_erpm'],
                                              'label': pass_label(p), 'r1_strong': bool(p['r1_strong']),
                                              'frames': 0, 'usable_frames': 0, 'samples_removed': {'train': 0, 'val': 0},
                                              'scenes': []})
                q['frames'] += 1
                q['usable_frames'] += int(ok[i])
                q['samples_removed'][split_of[s]] += int(used_B[i])
                if s not in q['scenes']:
                    q['scenes'].append(s)
    return rows, per_pass, mism


def write_csv(path, rows):
    path = Path(path)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    _write_text(path, buf.getvalue())


def check_heldout(protocol, b_counts):
    """保留の分け方が A と同じか: splits.json・A/B の result.json・heldout.json・pred_table の key"""
    from jetformer import config
    root = config.DATASET_DIR / SRC_NAME
    splits = json.loads((root / 'splits.json').read_text())
    card = json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))
    created = datetime.datetime.fromisoformat(card['created_at'])
    newest = max(os.stat(Path(dp) / f).st_mtime for dp, _, fs in os.walk(root) for f in fs)
    runs = {}
    for k, r in A_RUNS.items():
        res = json.loads((config.DATA_DIR / 'runs' / r / 'result.json').read_text(encoding='utf-8'))
        run = json.loads((config.DATA_DIR / 'runs' / r / 'run.json').read_text(encoding='utf-8'))
        runs[k] = {'run': r, 'dataset': res['dataset'], 'train_samples': res['train_samples'], 'val_samples': res['val_samples'],
                   'impact_masked_frames': res['impact_masked_frames'], 'train_mode_counts': res['train_mode_counts'],
                   'created_at': run.get('created_at'),
                   'same_as_now': (res['dataset'] == SRC_NAME and res['train_samples'] == b_counts['train']['samples']
                                   and res['val_samples'] == b_counts['val']['samples']
                                   and res['impact_masked_frames'] == {'train': b_counts['train']['impact_masked'],
                                                                       'val': b_counts['val']['impact_masked']}
                                   and res['train_mode_counts'] == b_counts['train']['genres'])}
    held = json.loads((config.DATA_DIR / 'rl' / 'ai_mix_20261011' / 'eval' / 'heldout.json').read_text(encoding='utf-8'))
    import rl_evaluate as ev
    tab = ev.pred_table()
    return {
        'splits_sha256': sha256(root / 'splits.json'), 'train_scenes': len(splits['train']), 'val_scenes': len(splits['val']),
        'val_scene_list': splits['val'], 'train_val_overlap': sorted(set(splits['train']) & set(splits['val'])),
        'dataset_files_newest_mtime': datetime.datetime.fromtimestamp(newest).isoformat(timespec='seconds'),
        'dataset_created_at': card['created_at'],
        'dataset_unchanged_since_created': newest <= created.timestamp() + 60,
        'runs': runs, 'all_runs_same_split_and_counts': all(v['same_as_now'] for v in runs.values()),
        'heldout_human_clean_scenes_equal_val': sorted(held['human_clean_scenes']) == sorted(splits['val']),
        'pred_table_key': tab['key'], 'pred_table_key_equal': tab['key'] == PRED_TABLE_KEY == held['pred_table_key'],
        'pred_table_rows': int(len(tab['part'])),
        'b2_scene_split': '同じ splits.json(学習 41・検証 11 シーン)。外す表と付け替えの表はシーンを動かさない(フレームを外す・ジャンルを変えるだけ)',
        'relabeled_val_scenes': [s for s in splits['val'] if M.remap_value(s) >= 0],
    }


def run_checks():
    protocol = load_protocol()
    from jetformer import config
    t0 = time.time()
    res = {'created_at': now_jst(), 'protocol_sha256': sha256(PROTOCOL)}
    # 0) 固定の確かめ(protocol.sha256 の全部の行・コードの版)
    bad = []
    for line in PROTOCOL_SHA.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        h, p = line.split(None, 1)
        p = p.strip()
        path = Path(p) if p.startswith('/') else REPORT / p
        if sha256(path) != h:
            bad.append(p)
    res['frozen_files_ok'] = not bad
    res['frozen_files_changed'] = bad
    head = _git('rev-parse', '--short', 'HEAD')
    dirty = _git('status', '--porcelain', '--', 'tools/train.py', 'jetformer/train_data.py', 'jetformer/model.py',
                 'jetformer/policies', 'tools/offline_rl.py', 'tools/label_events.py', 'tools/rl_evaluate.py', 'data/modes.json')
    res['code'] = {'head': head, 'head_equal_protocol': head == protocol['code_version']['jetformer_git_head'],
                   'relevant_uncommitted': dirty.splitlines()}
    print('[checks] 固定のファイル %s・コード %s' % ('OK' if not bad else bad, head), flush=True)
    # 1) 表の作り直し(通過の見つけ方・外す表・付け替えの表)
    res['tables'] = check_tables(protocol)
    print('[checks] 表の作り直し: %s' % res['tables'], flush=True)
    # 2) ジャンル別の数(B = 表なし、B2 = 元 + 表、B2' = ジャンルを直したデータセット + 外す表)
    src = config.DATASET_DIR / SRC_NAME
    dsB = _datasets(src)
    cB = _ds_summary(dsB)
    del dsB
    dsB2 = _datasets(src, OUT_DIR / 'drop_b2.npz', OUT_DIR / 'remap_b2.npz')
    cB2 = _ds_summary(dsB2)
    exp = protocol['model']['expected_train_log']
    cur = protocol['curation']['counts']
    res['counts'] = {'B': cB, 'B2': cB2}
    res['counts_equal_protocol'] = {
        'B': (cB['train']['samples'], cB['val']['samples']) == (cur['B_train_samples'], cur['B_val_samples'])
        and cB['train']['genres'] == cur['genre_samples_train']['B'] and cB['val']['genres'] == cur['genre_samples_val']['B'],
        'B2': (cB2['train']['samples'], cB2['val']['samples']) == (exp['train_samples'], exp['val_samples'])
        and cB2['train']['genres'] == {k: exp['genre_counts_train'][k] for k in sorted(exp['genre_counts_train'])}
        and cB2['val']['genres'] == cur['genre_samples_val']['B2']
        and [cB2['train']['dropped'], cB2['val']['dropped']] == [exp['dropped_printed']['train'], exp['dropped_printed']['val']]
        and [cB2['train']['impact_masked'], cB2['val']['impact_masked']] == [exp['impact_masked_printed']['train'],
                                                                             exp['impact_masked_printed']['val']]}
    res['genre_table'] = genre_table(src)
    print('[checks] 数: B %s / B2 %s' % (res['counts_equal_protocol']['B'], res['counts_equal_protocol']['B2']), flush=True)
    # 3) ジャンルを直したデータセットが B2 と同じか
    if DERIVED_ROOT.exists():
        v = verify_derived(src, DERIVED_ROOT)
        dsD = _datasets(DERIVED_ROOT, OUT_DIR / 'drop_b2.npz')
        cD = _ds_summary(dsD)
        eq = {}
        for sp in ('train', 'val'):
            eq[sp] = {'index_and_genre_equal': _index_modes(dsB2[sp]) == _index_modes(dsD[sp]),
                      'counts_equal': cD[sp] == cB2[sp]}
            rng = np.random.default_rng(0)
            pick = rng.choice(len(dsD[sp]), size=min(24, len(dsD[sp])), replace=False)
            same = True
            for n in pick:
                x, y = dsB2[sp][int(n)], dsD[sp][int(n)]
                same &= all(bool((u == w).all()) for u, w in zip(x, y))
            eq[sp]['tensors_equal_24_samples'] = bool(same)
        res['derived'] = {'path': str(DERIVED_ROOT), 'verify': {k: v[k] for k in v if k != 'problems'},
                          'problems': v['problems'][:20], 'counts': cD, 'equal_to_B2': eq,
                          'all_equal': v['ok'] and all(all(e.values()) for e in eq.values())}
        del dsD
        print('[checks] ジャンルを直したデータセット: %s' % res['derived']['all_equal'], flush=True)
    else:
        res['derived'] = {'path': str(DERIVED_ROOT), 'missing': True, 'all_equal': False}
    del dsB2
    # 4) 外したフレームの一覧
    rows, per_pass, mism = dropped_frames_table(protocol)
    write_csv(DROPPED_CSV, rows)
    by_reason = {}
    for r in rows:
        k = str(r['reason'])
        q = by_reason.setdefault(k, {'name': REASON_NAME[r['reason']], 'frames': 0, 'usable_frames': 0,
                                     'samples_in_B': {'train': 0, 'val': 0}})
        q['frames'] += 1
        q['usable_frames'] += r['usable_frame']
        q['samples_in_B'][r['split']] += r['sample_in_B']
    dropped_passes = sorted(per_pass.values(), key=lambda q: (q['log'], q['ta_s']))
    frozen_passes = {(p['log'], p['ta_s']) for p in protocol['curation']['dropped_passes']}
    res['dropped'] = {
        'csv': str(DROPPED_CSV), 'csv_sha256': sha256(DROPPED_CSV), 'frames': len(rows), 'by_reason': by_reason,
        'unattributed_frames': sum(1 for r in rows if r['pass_log'] == ''), 'rule_recompute_mismatch': mism,
        'passes': dropped_passes, 'passes_equal_protocol': {(q['log'], q['ta_s']) for q in dropped_passes} == frozen_passes,
        'samples_removed': {'train': sum(q['samples_removed']['train'] for q in dropped_passes),
                            'val': sum(q['samples_removed']['val'] for q in dropped_passes)}}
    print('[checks] 外したフレーム %d(理由別 %s)' % (len(rows), {k: v['frames'] for k, v in by_reason.items()}), flush=True)
    # 5) 保留の分け方
    res['heldout'] = check_heldout(protocol, cB)
    print('[checks] 保留: A と同じ %s・pred_table %s' % (res['heldout']['all_runs_same_split_and_counts'],
                                                     res['heldout']['pred_table_key_equal']), flush=True)
    res['all_ok'] = bool(res['frozen_files_ok'] and res['code']['head_equal_protocol'] and not res['code']['relevant_uncommitted']
                         and all(res['tables'][k] for k in ('rebuilt_sha256_equal', 'files_on_disk_equal_protocol',
                                                            'card_contents_equal'))
                         and all(res['counts_equal_protocol'].values()) and res['derived'].get('all_equal')
                         and res['dropped']['unattributed_frames'] == 0 and res['dropped']['rule_recompute_mismatch'] == 0
                         and res['dropped']['passes_equal_protocol']
                         and res['dropped']['samples_removed'] == {'train': cur['removed_train_samples'],
                                                                   'val': cur['removed_val_samples']}
                         and res['heldout']['all_runs_same_split_and_counts'] and res['heldout']['pred_table_key_equal']
                         and res['heldout']['heldout_human_clean_scenes_equal_val']
                         and res['heldout']['dataset_unchanged_since_created'] and not res['heldout']['train_val_overlap'])
    res['seconds'] = round(time.time() - t0, 1)
    save_json(CHECKS_JSON, res)
    print('[checks] 全部 %s(%.0f 秒)→ %s' % ('OK' if res['all_ok'] else 'NG', res['seconds'], CHECKS_JSON), flush=True)
    return res


def genre_table(src):
    """分け方 × ジャンルごとのフレーム数(全部・使えるフレーム・外した・衝突の除外・学習サンプル)。B と B2 の両方"""
    from jetformer.train_data import impact_mask
    splits = json.loads((Path(src) / 'splits.json').read_text())
    with np.load(OUT_DIR / 'drop_b2.npz', allow_pickle=False) as z:
        d_scene, d_frame, d_drop = z['scene'].astype(str), z['frame'], z['drop']
    tab = {}
    for sp in ('train', 'val'):
        for s in splits[sp]:
            with np.load(Path(src) / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                a = {k: z[k] for k in ('timestamps_us', 'mode', 'future_valid', 'vehicle_valid', 'imu')}
            sel = d_scene == s
            drop = np.zeros(len(a['mode']), bool)
            drop[d_frame[sel]] = d_drop[sel]
            imp = impact_mask(a['timestamps_us'], a['imu'], M.IMPACT_G)
            ok = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
            for model, g_map, use_drop in (('B', None, False), ('B2', M.remap_value(s), True)):
                mode = a['mode'] if g_map is None or g_map < 0 else np.where(a['mode'] != 0, g_map, 0)
                okd = ok & ~drop if use_drop else ok
                used = M.used_samples(a['future_valid'], a['vehicle_valid'], a['mode'], imp, drop if use_drop else None)
                for g in np.unique(mode[ok]) if ok.any() else []:
                    gm = mode == g
                    c = tab.setdefault(model, {}).setdefault(sp, {}).setdefault(str(int(g)), {
                        'scenes': 0, 'frames': 0, 'usable_frames': 0, 'dropped_usable': 0, 'impact_masked': 0, 'samples': 0})
                    c['scenes'] += 1
                    c['frames'] += int(gm.sum())
                    c['usable_frames'] += int((ok & gm).sum())
                    c['dropped_usable'] += int((ok & drop & gm).sum()) if use_drop else 0
                    c['impact_masked'] += int((okd & imp & gm).sum())
                    c['samples'] += int((used & gm).sum())
    return tab


# ================================================================ 6. smoke(tools/train.py の CPU の動作確認)

def _run_logged(cmd, log, kill_after_epoch=None, timeout=1800):
    """コマンドを動かし、出力を log に書く。kill_after_epoch=N なら、N エポック目の行と last.pth が出たら
    プロセスのグループごと SIGKILL(電源断のまね)"""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w', encoding='utf-8') as f:
        f.write('==== %s 開始: %s\n' % (now_jst(), ' '.join(shlex.quote(c) for c in cmd)))
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        killed = False
        while p.poll() is None:
            time.sleep(2)
            if kill_after_epoch is not None and not killed:
                text = log.read_text(encoding='utf-8', errors='replace')
                if re.search(r'^epoch +%d ' % kill_after_epoch, text, re.M):
                    time.sleep(1)
                    os.killpg(p.pid, signal.SIGKILL)
                    killed = True
            if time.time() - t0 > timeout:
                os.killpg(p.pid, signal.SIGKILL)
                f.write('\n[smoke] 時間切れで止めました\n')
                break
        rc = p.wait()
        f.write('==== %s 終了 rc=%s%s\n' % (now_jst(), rc, '(SIGKILL で止めた = 電源断のまね)' if killed else ''))
    return rc, killed, log.read_text(encoding='utf-8', errors='replace')


def run_smoke():
    """CPU で tools/train.py を 4 回動かす(protocol の train_cmd の文字列から、上限・鍵・CUDA・出力先・エポックだけ変える)
      S1 数: 元のデータセット + 表、--epochs 0(学習しない)。最初の 4 行が protocol の expected_train_log と同じか
      S2 数: ジャンルを直したデータセット + 外す表(付け替えの表なし)、--epochs 0。S1 と同じ数か
      S3 学習: 元 + 表、--epochs 2 --max-steps 2 --batch 8 --workers 2。1 エポック目の後で SIGKILL(電源断のまね)
      S4 再開: S3 と同じ引数 + --resume。2 エポック目から続け、best.pth・ONNX・model_config.json・result.json を書くか"""
    from jetformer import config
    protocol = load_protocol()
    exp = protocol['model']['expected_train_log']
    names = mode_names_from_card(config.DATASET_DIR / SRC_NAME)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    res = {'created_at': now_jst(), 'note': 'CPU(CUDA_VISIBLE_DEVICES=)・jf-heavy --mem 1500M --lock none。'
           '出力先は smoke/ の下(data/runs には書かない = 走行の一覧に出ない)。重いファイル(pth・onnx)は確かめた後に消す', 'steps': {}}
    out = lambda n: str(SMOKE_DIR / n)                                             # noqa: E731
    steps = [
        ('S1_counts', smoke_cmd(protocol, out=out('S1_counts'), epochs=0, workers=0), None, True),
        ('S2_derived_counts', smoke_cmd(protocol, out=out('S2_derived_counts'),
                                        data=str(DERIVED_ROOT.relative_to(REPO)), drop_remap=False, epochs=0, workers=0),
         None, True),
        ('S3_train_kill', smoke_cmd(protocol, out=out('S3_train'), epochs=2, max_steps=2, batch=8, workers=2), 1, False),
        ('S4_resume', smoke_cmd(protocol, out=out('S3_train'), epochs=2, max_steps=2, batch=8, workers=2, resume=True),
         None, False),
    ]
    for name, cmd, kill, full in steps:
        if name == 'S3_train_kill' and Path(out('S3_train')).exists():
            shutil.rmtree(out('S3_train'))                   # 前回の動作確認の残り(自分の smoke/ の下だけ)
        print('[smoke] %s: %s' % (name, ' '.join(shlex.quote(c) for c in cmd)), flush=True)
        t0 = time.time()
        rc, killed, text = _run_logged(cmd, SMOKE_DIR / (name + '.log'), kill_after_epoch=kill)
        parsed = parse_train_log(text)
        ok, diffs = compare_train_log(parsed, exp, names, full=full)
        st = {'cmd': cmd, 'rc': rc, 'killed': killed, 'seconds': round(time.time() - t0, 1), 'parsed': parsed,
              'log_matches_protocol': ok, 'diffs': diffs, 'device_cpu': parsed.get('samples', [None] * 5)[4] == 'cpu'}
        if name == 'S3_train_kill':
            st['last_pth_after_kill'] = (Path(out('S3_train')) / 'last.pth').exists()
            st['epoch1_line'] = bool(re.search(r'^epoch +1 ', text, re.M))
            st['ok'] = ok and killed and st['last_pth_after_kill'] and st['epoch1_line'] and st['device_cpu']
        elif name == 'S4_resume':
            d = Path(out('S3_train'))
            r = json.loads((d / 'result.json').read_text(encoding='utf-8')) if (d / 'result.json').exists() else {}
            mc = json.loads((d / 'model_config.json').read_text(encoding='utf-8')) if (d / 'model_config.json').exists() else {}
            st['resumed_line'] = bool(re.search(r'再開: 1 エポック目まで済み', text))
            st['files'] = {f: (d / f).exists() for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json', 'result.json')}
            st['result'] = {k: r.get(k) for k in ('state', 'epochs_done', 'best_epoch', 'train_samples', 'val_samples',
                                                   'impact_masked_frames', 'steer_center_us', 'throttle_neutral_us', 'seed')}
            st['result_rl_options'] = r.get('rl_options')
            st['model_config'] = {k: mc.get(k) for k in ('num_modes', 'trained_modes', 'train_mode_counts', 'steer_center_us',
                                                           'throttle_neutral_us', 'command_input', 'mask_impacts', 'policy',
                                                           'dataset')}
            sha1 = (r.get('rl_options') or {}).get('sha1', {})
            want_sha1 = {k: hashlib.sha1(Path(REPO / v).read_bytes()).hexdigest()[:12]
                         for k, v in (('drop_mask', 'data/derived/b2_curation_20261016/drop_b2.npz'),
                                      ('mode_remap_npz', 'data/derived/b2_curation_20261016/remap_b2.npz'))}
            st['sha1_equal'] = sha1 == want_sha1
            st['ok'] = (ok and rc == 0 and st['resumed_line'] and all(st['files'].values()) and r.get('state') == 'done'
                        and r.get('epochs_done') == 2 and sorted(mc.get('trained_modes') or []) == [4, 5]
                        and mc.get('steer_center_us') == 1441.0 and mc.get('throttle_neutral_us') == 1496.0
                        and mc.get('command_input') is False and st['sha1_equal']
                        and (r.get('rl_options') or {}).get('dropped_frames') == exp['dropped_printed']
                        and r.get('impact_masked_frames') == exp['impact_masked_printed'] and st['device_cpu'])
        else:
            d = Path(out(name))
            r = json.loads((d / 'result.json').read_text(encoding='utf-8')) if (d / 'result.json').exists() else {}
            st['result_state'] = r.get('state')
            st['action_reference'] = [r.get('steer_center_us'), r.get('throttle_neutral_us')]
            st['ok'] = ok and rc == 0 and r.get('state') == 'stopped_no_model' and st['device_cpu']
        res['steps'][name] = st
        print('[smoke] %s: %s(%.0f 秒)%s' % (name, 'OK' if st['ok'] else 'NG', st['seconds'], diffs or ''), flush=True)
        save_json(SMOKE_DIR / 'smoke.json', res)
    s1, s2 = res['steps']['S1_counts']['parsed'], res['steps']['S2_derived_counts']['parsed']
    res['derived_same_as_tables'] = s1 == s2
    res['all_ok'] = all(st['ok'] for st in res['steps'].values()) and res['derived_same_as_tables']
    # 重いファイルを消す(ログ・result.json・model_config.json は残す)
    removed = []
    for d in SMOKE_DIR.iterdir():
        if d.is_dir():
            for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'last.pth.tmp'):
                if (d / f).exists():
                    (d / f).unlink()
                    removed.append(str((d / f).relative_to(SMOKE_DIR)))
    res['removed_heavy_files'] = removed
    save_json(SMOKE_DIR / 'smoke.json', res)
    print('[smoke] 全部 %s → %s' % ('OK' if res['all_ok'] else 'NG', SMOKE_DIR / 'smoke.json'), flush=True)
    return res


# ================================================================ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('derived')
    sub.add_parser('checks')
    sub.add_parser('smoke')
    p = sub.add_parser('check-log')
    p.add_argument('log')
    p.add_argument('--partial', action='store_true', help='--max-steps の動作確認(サンプル数の行は見ない)')
    args = ap.parse_args()
    if args.cmd == 'derived':
        build_derived()
    elif args.cmd == 'checks':
        r = run_checks()
        sys.exit(0 if r['all_ok'] else 1)
    elif args.cmd == 'smoke':
        r = run_smoke()
        sys.exit(0 if r['all_ok'] else 1)
    elif args.cmd == 'check-log':
        from jetformer import config
        exp = load_protocol()['model']['expected_train_log']
        parsed = parse_train_log(Path(args.log).read_text(encoding='utf-8', errors='replace'))
        ok, diffs = compare_train_log(parsed, exp, mode_names_from_card(config.DATASET_DIR / SRC_NAME), full=not args.partial)
        print(json.dumps({'ok': ok, 'diffs': diffs, 'parsed': parsed}, ensure_ascii=False))
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
