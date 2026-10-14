"""別の環境(シミュレーター・エミュレーターなど)で作ったデータセットを取り込む

PC からブラウザでまとめてアップロードしたもの(PUT /api/datasets/import)や、tools/upload_datasets.py が送ったものを、
data/datasets/<名前>/ に展開する。形はエクスポート(dataset.export_dataset)と同じ「先頭のフォルダ = データセット名」の
tar(.tar・.tar.gz・.tgz)か zip。先頭のフォルダが無く、dataset_card.json がいちばん上にある tar・zip も、カードの name で取り込む。
Mac の Finder や tar が入れる __MACOSX/・.DS_Store・._* は無視する。

取り込む前に、学習(jetformer/train_data.py の JetFormerDataset)が読むものがそろっているかを確かめる(validate_dataset)。
足りなければ取り込まない(途中のフォルダに展開してから置き換えるので、半端なデータセットは残らない)。
"""

import datetime
import json
import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from . import config, dataset

# 学習で読む aligned/*.npz の配列: (名前, 1フレームあたりの形。None = 長さだけ確かめる)
REQUIRED_ARRAYS = [('timestamps_us', None), ('cam_files', None), ('tof', (42, 54)), ('imu', (10, 6)), ('vehicle', (4,)),
                   ('vehicle_valid', None), ('future_actions', (10, 2)), ('future_valid', None)]
JUNK_PREFIXES = ('__MACOSX/',)
JUNK_NAMES = ('.DS_Store', 'Thumbs.db')
ARCHIVE_SUFFIXES = ('.tar', '.tar.gz', '.tgz', '.zip')


def _junk(name):
    p = PurePosixPath(name)
    return name.startswith(JUNK_PREFIXES) or p.name in JUNK_NAMES or p.name.startswith('._')


def _safe_parts(name):
    """アーカイブの中のパスを確かめて部品に分ける(絶対パス・..・空は拒否)"""
    name = name.replace('\\', '/')
    if name.startswith('/') or (len(name) > 1 and name[1] == ':'):
        raise ValueError('絶対パスが入っています: %s' % name)
    parts = [x for x in name.split('/') if x not in ('', '.')]
    if '..' in parts:
        raise ValueError('「..」を含むパスが入っています: %s' % name)
    return parts


def _members(path):
    """(種類, [(部品, ディレクトリか, 取り出す関数)])。リンク・デバイスは拒否。
    種類はファイルの先頭で決める(zipfile.is_zipfile は、最後に .npz(中身は zip)が入った tar も zip とみなしてしまう)"""
    with open(path, 'rb') as f:
        head = f.read(4)
    if head[:2] == b'PK' and zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        out = []
        for info in z.infolist():
            if _junk(info.filename):
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError('リンクは取り込めません: %s' % info.filename)
            parts = _safe_parts(info.filename)
            if parts:
                out.append((parts, info.is_dir(), (lambda i=info: z.open(i))))
        return 'zip', out, z
    if tarfile.is_tarfile(path):
        t = tarfile.open(path, 'r:*')
        out = []
        for m in t.getmembers():
            if _junk(m.name):
                continue
            if not (m.isfile() or m.isdir()):
                raise ValueError('ファイルとフォルダ以外(リンクなど)は取り込めません: %s' % m.name)
            parts = _safe_parts(m.name)
            if parts:
                out.append((parts, m.isdir(), (lambda mm=m: t.extractfile(mm))))
        return 'tar', out, t
    raise ValueError('tar(.tar・.tar.gz)か zip ではありません')


def _layout(members):
    """データセットの名前と、アーカイブの中の先頭(取り除く部品の数)を決める"""
    files = [p for p, is_dir, _ in members if not is_dir]
    if any(p == ['dataset_card.json'] for p in files):
        return None, 0                                       # 先頭のフォルダ無し(カードの name を使う)
    tops = {p[0] for p in files}
    if len(tops) != 1:
        raise ValueError('いちばん上のフォルダが1つではありません(%s)。1つのデータセットを1つのファイルにしてください'
                         % ', '.join(sorted(tops)[:5]))
    top = tops.pop()
    if [top, 'dataset_card.json'] not in files:
        raise ValueError('%s/dataset_card.json がありません(JetFormer のデータセットの形ではありません)' % top)
    return top, 1


def validate_dataset(root, sample=20):
    """学習で読むものがそろっているか。(エラーのリスト, 注意のリスト, 要約)"""
    root = Path(root)
    errors, warnings = [], []
    try:
        card = json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        return ['dataset_card.json を読めません: %s' % e], warnings, {}
    try:
        splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
        scenes = list(splits.get('train', [])) + list(splits.get('val', []))
    except (OSError, ValueError, AttributeError) as e:
        return ['splits.json を読めません: %s' % e], warnings, {}
    if not splits.get('train'):
        errors.append('splits.json の train が空です')
    if not card.get('modes'):
        warnings.append('dataset_card.json に modes(走行ジャンルの表)がありません。ジャンルは「未設定」だけになります')
    else:
        local = {m['id']: m['name'] for m in dataset.modes.load()}
        diff = [m for m in card['modes'] if m.get('id') in local and m.get('name') != local[m['id']]]
        if diff:
            warnings.append('ジャンルの名前がこの Jetson と違います(ID は同じものとして学習します): %s'
                            % ', '.join('%s=%s' % (m['id'], m['name']) for m in diff[:4]))
    frames, usable, img_checked = 0, 0, 0
    rng = np.random.default_rng(0)
    for s in scenes:
        p = root / 'aligned' / ('%s.npz' % s)
        if not p.exists():
            errors.append('aligned/%s.npz がありません' % s)
            continue
        try:
            with np.load(p, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
        except Exception as e:                                          # noqa: BLE001 (壊れたファイルは何でも起こりうる)
            errors.append('aligned/%s.npz を読めません: %s' % (s, e))
            continue
        n = len(a.get('timestamps_us', []))
        for key, shape in REQUIRED_ARRAYS:
            if key not in a:
                errors.append('%s.npz に %s がありません' % (s, key))
            elif len(a[key]) != n:
                errors.append('%s.npz の %s の長さ(%d)が timestamps_us(%d)と違います' % (s, key, len(a[key]), n))
            elif shape is not None and tuple(a[key].shape[1:]) != shape:
                errors.append('%s.npz の %s の形が %s です(%s が要る)' % (s, key, tuple(a[key].shape[1:]), shape))
        if 'mode' not in a:
            warnings.append('%s.npz に mode がありません(ジャンル未設定 = 学習に使われません)' % s)
        if any(e.startswith(s + '.npz') for e in errors):
            continue
        frames += n
        ok = a['future_valid'].astype(bool) & a['vehicle_valid'].astype(bool)
        ok &= (a['mode'] != 0) if 'mode' in a else False      # mode が無いと学習では全部「未設定」になり使われない
        usable += int(ok.sum())
        if n:
            for i in rng.choice(n, size=min(sample, n), replace=False):
                f = root / str(a['cam_files'][i])
                img_checked += 1
                if not f.exists():
                    errors.append('%s の画像がありません: %s' % (s, a['cam_files'][i]))
                    break
    if scenes and frames and not usable:
        errors.append('学習に使えるフレームがありません(future_valid・vehicle_valid・mode を確かめてください)')
    summary = {'scenes': len(scenes), 'frames': frames, 'usable_frames': usable, 'images_checked': img_checked}
    return errors[:20], warnings[:20], summary


def import_archive(path, simulator=True, source_name=None, progress=None, to_scenes=True):
    """アーカイブを data/datasets/<名前>/ に取り込む。戻り値 {'name', 'summary', 'warnings', 'tags', 'source', 'scenes'}
    to_scenes: シーンタブで見られるよう、シーンの形にも展開する(scene_import.dataset_to_scenes)"""
    path = Path(path)
    kind, members, handle = _members(path)
    try:
        top, strip = _layout(members)
        if top is None:
            with [f for p, d, f in members if p == ['dataset_card.json']][0]() as fh:
                top = json.load(fh).get('name') or ''
        name = dataset_name_ok(top)
        dest = config.DATASET_DIR / name
        if dest.exists():
            return already_imported(name, source_name or path.name, path.stat().st_size, progress, to_scenes)
        stage = config.DATASET_DIR / ('.import-%s' % uuid.uuid4().hex[:8])
        root = stage / name
        root.mkdir(parents=True)
        try:
            files = [(p, f) for p, d, f in members if not d]
            for k, (parts, opener) in enumerate(files):
                rel = parts[strip:]
                if not rel:
                    continue
                out = root.joinpath(*rel)
                if root.resolve() not in out.resolve().parents:
                    raise ValueError('データセットの外に出るパスがあります: %s' % '/'.join(parts))
                out.parent.mkdir(parents=True, exist_ok=True)
                with opener() as src, open(out, 'wb') as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                if progress and k % 200 == 0:
                    progress(0.1 + 0.7 * k / max(len(files), 1), '展開中 %d / %d' % (k, len(files)))
            card, summary, warnings, lmask = check_staged(root, name, progress)
            os.replace(root, dest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    finally:
        handle.close()
    return after_import(name, dest, card, summary, warnings, lmask, source_name or path.name, path.stat().st_size, kind,
                        simulator, progress, to_scenes)


def check_staged(root, name, progress=None):
    """展開した(作業フォルダーの)データセットの学習除外・形の確かめ・カードの名前。戻り値 (card, summary, warnings, label_mask)。
    だめなら ValueError(取り込まない)。シミュレーターの強化学習モードのまとめての取り込み(sim/rl_import.py)も使う"""
    if progress:
        progress(0.85, '中身を確かめています')
    # シミュレーターの強化学習モードのシーンは、乱した操作(info.jf.l = 0)を正解から外す。jf の無い行も外す(安全側)。
    # 強化学習モードでないシーンは変えない(sim/label_mask.py)
    from .sim import label_mask
    try:
        lmask = label_mask.apply_to_dataset(root)
    except Exception as e:                                      # noqa: BLE001 (外せないなら取り込まない)
        raise ValueError('学習から外すフレームを決められません: %s' % e)
    errors, warnings, summary = validate_dataset(root)
    if errors:
        raise ValueError('学習に使える形ではありません: ' + ' / '.join(errors[:6]))
    if lmask:
        summary['label_masked_frames'] = lmask['masked_frames']
        warnings = warnings + lmask['warnings'][:10]
    card_path = root / 'dataset_card.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    if card.get('name') != name:                    # カードの名前をフォルダの名前にそろえる
        card['name'] = name
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
    return card, summary, warnings, lmask


def rl_tags(lmask):
    """強化学習モードのシーンのタグ(sim-rl・sim-rl-<モード> と、登録簿のモードのタグ)。学習除外の結果の teacher_mode から決める"""
    if not lmask:
        return []
    tags = []
    for v in (lmask.get('per_scene') or {}).values():
        m = v.get('teacher_mode')
        if not isinstance(m, str) or not m.replace('_', '').isalnum() or len(m) > 32:
            continue
        tags += ['sim-rl', 'sim-rl-%s' % m]
        try:
            from .sim import rl_modes
            tags += [t for t in rl_modes.load_mode(m).get('tags', []) if isinstance(t, str)]
        except Exception:                                               # noqa: BLE001 (登録簿に無いモードでもタグは付ける)
            pass
    return tags


def after_import(name, dest, card, summary, warnings, lmask, file_name, size, kind, simulator, progress=None, to_scenes=True,
                 extra_tags=None, extra_info=None):
    """置き換えた後: タグ(+ 強化学習モードのタグ)・import_source.json・シーンタブへの展開"""
    tags = list(card.get('tags') or []) + (['simulator'] if simulator else []) + rl_tags(lmask) + list(extra_tags or [])
    card = dataset.set_dataset_tags(name, tags)
    info = {'name': name, 'file': file_name, 'bytes': size, 'archive': kind,
            'simulator': bool(simulator), 'imported_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'summary': summary, 'warnings': warnings}
    if extra_info:
        info.update(extra_info)
    scenes = expand_scenes(name, progress, info['warnings']) if to_scenes else None
    info['scenes'] = scenes
    with open(dest / 'import_source.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    if progress:
        progress(1.0, '取り込みました: %s(%d シーン・学習に使えるフレーム %d%s)' % (
            name, summary['scenes'], summary['usable_frames'],
            '・シーンタブに %d シーン' % len(scenes['created']) if scenes and scenes.get('created') else ''))
    return dict(info, tags=card['tags'], source=card['source'])


def already_imported(name, file_name, size, progress=None, to_scenes=True):
    """同じ名前のデータセットがあるとき。同じファイル(名前と大きさ)から取り込んだものなら「取り込み済み」として成功にし、
    シーンタブにまだ出ていなければ展開する。違うものなら、どれが入っているかを書いてエラーにする"""
    dest = config.DATASET_DIR / name
    try:
        prev = json.loads((dest / 'import_source.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        prev = None
    if not prev or prev.get('file') != file_name or int(prev.get('bytes', -1)) != int(size):
        how = ('(%s に %s から取り込み済み)' % (prev.get('imported_at'), prev.get('file'))) if prev else ''
        raise FileExistsError('同じ名前のデータセット「%s」があります%s。中身を入れ替えるときは、データセットタブで削除してから'
                              '取り込んでください' % (name, how))
    card = json.loads((dest / 'dataset_card.json').read_text(encoding='utf-8'))
    log = card.get('scene_log')
    scenes = None
    warnings = list(prev.get('warnings') or [])
    if to_scenes and not (log and (config.RAW_DIR / log).is_dir()):
        scenes = expand_scenes(name, progress, warnings)
    if progress:
        progress(1.0, 'すでに取り込み済みです(%s)%s' % (prev.get('imported_at'),
                      '。シーンタブに %d シーンを展開しました' % len(scenes['created']) if scenes and scenes.get('created') else ''))
    return dict(prev, already_imported=True, scenes=scenes, warnings=warnings, tags=card.get('tags', []),
                source=card.get('source', 'real'))


def expand_scenes(name, progress=None, warnings=None):
    """シーンの形に展開する。失敗しても取り込みは取り消さず、注意に書く"""
    from . import scene_import
    try:
        return scene_import.dataset_to_scenes(name, progress, 0.9, 0.99)
    except Exception as e:                                              # noqa: BLE001
        if warnings is not None:
            warnings.append('シーンタブへの展開に失敗しました: %s' % e)
        return None


def dataset_name_ok(name):
    """データセット名に使えるか(英数字・_・-)"""
    if not name or not all(c.isalnum() and c.isascii() or c in '_-' for c in name):
        raise ValueError('データセット名に使えない文字があります: %r(英数字・_・- だけ)' % name)
    return name
