"""シミュレーターで 1 シーンずつ記録した ZIP を、JetFormer の 1 つのデータセットにまとめて取り込む(強化学習モードの収集プラン)

ブリッジの「1つのZIPにまとめる」(qa/combine_scene_datasets.py)は空き 20 GiB + 元の 2 倍を要求するので、プランのモードごとの
シーン ZIP(data/sim/runtime/datasets/<セッション>/…zip)を JetFormer がじかにまとめる。形はブリッジの集約と同じ:

    aligned/<シーン>.npz・samples/…・can_bus/…   そのまま(シーンごとに名前が違う)
    simulation/scenes/<シーン>/{session.json, frames.jsonl, evaluation-report.json}
    simulation/collection.json                   まとめの記録(セッション ID・シーン・開始・契約)
    v1.0-<名前>/*.json                            nuScenes 風の表(token で重ねる)
    dataset_card.json・splits.json・manifest.json・DATASET_README.txt

集約と同じ主な確かめ: 1 シーン・600 フレーム・30 Hz・連続周回・衝突 0・教師の印(合成・demo)・future_valid は最後の 10 だけ false・
manifest の sha256・契約(コースの版・経路・センサー・教師(context.teacher)・物理・換算・ジャンル表)がシーン間で同じ・
同じ開始姿勢が train と val にまたがらない。そのあと dataset_import と同じ道(学習除外 → 形の確かめ → タグ → シーンタブ)を通る。
"""

import datetime
import hashlib
import io
import json
import os
import re
import shutil
import uuid
import zipfile
from pathlib import PurePosixPath

import numpy as np

from .. import config, dataset_import
from . import datasets

COLLECTION_SCHEMA = 'jetformer-sim-rl-collection/v1'
SCHEMA = 'course-sim-jetformer-dataset-v1'
TABLES = ('scene', 'log', 'sensor', 'calibrated_sensor', 'sample', 'sample_data', 'ego_pose', 'category', 'attribute',
          'visibility', 'map', 'instance', 'sample_annotation')
CONTRACT_CONTEXT = ('world_version', 'route', 'sensor_profile', 'sensors', 'ego_sensor_profile', 'teacher')
CONFIG_PER_SCENE = ('seed', 'noiseStd')         # シーンごとに変えてよい設定(種・合成の雑音の場面)
MAX_SCENES = 200
FRAMES = 600                                    # 1 シーン = 20 秒・30 Hz(テストでは小さくする)


class ImportError_(ValueError):
    pass


def _json(data):
    return json.loads(data.decode('utf-8') if isinstance(data, bytes) else data)


def _stable(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def audit_scene(path, sid, expected_frames=None):
    """1 シーンの ZIP を確かめる。戻り値はまとめに使う情報"""
    expected_frames = expected_frames or FRAMES
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(set(names)) != len(names):
            raise ImportError_('%s: ZIP の中に同じ名前があります' % sid[:8])
        for i in infos:
            p = PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ((i.external_attr >> 16) & 0o170000) == 0o120000:
                raise ImportError_('%s: 安全でないパスがあります' % sid[:8])
        cards = [n for n in names if PurePosixPath(n).name == 'dataset_card.json']
        if len(cards) != 1:
            raise ImportError_('%s: データセットが 1 つの ZIP ではありません' % sid[:8])
        prefix = cards[0][:-len('dataset_card.json')]
        if any(not n.startswith(prefix) for n in names):
            raise ImportError_('%s: データセットの外のファイルがあります' % sid[:8])
        files = [n for n in names if not n.endswith('/')]

        def read(rel):
            return z.read(prefix + rel)
        card, splits, session = _json(read('dataset_card.json')), _json(read('splits.json')), _json(read('simulation/session.json'))
        manifest = _json(read('manifest.json'))
        if card.get('synthetic') is not True or card.get('training_label_mode') != 'synthetic_controller' or card.get('controller') != 'demo':
            raise ImportError_('%s: 教師の合成ラベル(demo・自動ラベル)の記録ではありません' % sid[:8])
        if manifest.get('schema') != SCHEMA or session.get('schema') != SCHEMA:
            raise ImportError_('%s: 知らない形のシーンです' % sid[:8])
        sp = [(k, s) for k in ('train', 'val', 'test') for s in splits.get(k, [])]
        if len(sp) != 1 or card.get('scenes') != 1:
            raise ImportError_('%s: 1 シーンの ZIP ではありません' % sid[:8])
        split, scene = sp[0]
        if not isinstance(scene, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', scene):
            raise ImportError_('%s: シーンの名前がおかしいです' % sid[:8])
        mf = {it['path']: it for it in manifest.get('files', [])}
        if set(mf) != {n[len(prefix):] for n in files if n != prefix + 'manifest.json'}:
            raise ImportError_('%s: ファイルの一覧が manifest と違います' % sid[:8])
        for rel, it in mf.items():
            data = read(rel)
            if len(data) != it['bytes'] or hashlib.sha256(data).hexdigest() != it['sha256']:
                raise ImportError_('%s: 中身が manifest の sha256 と違います: %s' % (sid[:8], rel))
        with np.load(io.BytesIO(read('aligned/%s.npz' % scene)), allow_pickle=False) as d:
            a = {k: d[k] for k in d.files}
        n = len(a['timestamps_us'])
        if n != expected_frames or card.get('frames') != n:
            raise ImportError_('%s: フレーム数が %d です(%d が要る)' % (sid[:8], n, expected_frames))
        rel_t = a['timestamps_us'] - a['timestamps_us'][0]
        if not np.all(np.abs(rel_t - np.rint(np.arange(n) * 1e6 / 30)) <= 1):
            raise ImportError_('%s: カメラの時刻が 30 Hz で続いていません' % sid[:8])
        if not np.array_equal(a['future_valid'].astype(bool), np.arange(n) < n - 10):
            raise ImportError_('%s: future_valid が「最後の 10 フレームだけ false」ではありません' % sid[:8])
        if not a['vehicle_valid'].all() or len(set(a['mode'].tolist())) != 1:
            raise ImportError_('%s: ラベルの印かジャンルがそろっていません' % sid[:8])
        rows = [_json(x) for x in read('simulation/frames.jsonl').splitlines() if x.strip()]
        if len(rows) != n or any(r.get('frame') != rows[0].get('frame', 0) + i for i, r in enumerate(rows)):
            raise ImportError_('%s: frames.jsonl のフレームが続いていません' % sid[:8])
        report = _json(read('simulation/evaluation-report.json'))
        cfg, ctx = session.get('config') or {}, session.get('context') or {}
        if (cfg.get('episodeMode') != 'continuous-laps-v1' or report.get('outcome') != {'terminal': True, 'success': False,
                                                                                         'reason': 'collection_duration_complete'}):
            raise ImportError_('%s: 20 秒の連続周回を走り切ったシーンではありません' % sid[:8])
        if (report.get('metrics') or {}).get('collisions') != 0:
            raise ImportError_('%s: 衝突があります' % sid[:8])
        if cfg.get('frictionMultiplier', 1) != 1:
            raise ImportError_('%s: 低摩擦の場面は強化学習モードでは使いません' % sid[:8])
        contract = {k: ctx.get(k) for k in CONTRACT_CONTEXT}
        contract.update(config={k: v for k, v in cfg.items() if k not in CONFIG_PER_SCENE}, calibration=session.get('calibration'),
                        model_config=session.get('model_config'), modes=card.get('modes'))
        start = rows[0].get('pose_pre_action')
        tables = {}
        for t in TABLES:
            try:
                tables[t] = _json(read('%s/%s.json' % (card['version'], t)))
            except KeyError:
                tables[t] = []
        return {'path': path, 'sid': sid, 'prefix': prefix, 'card': card, 'session': session, 'report': report, 'split': split,
                'scene': scene, 'frames': n, 'training_frames': int(a['future_valid'].sum()), 'contract': contract,
                'start': start, 'start_key': _stable(start), 'tables': tables, 'members': [x[len(prefix):] for x in files],
                'teacher': ctx.get('teacher') or {}, 'scenario': report.get('scenario'), 'seed': cfg.get('seed'),
                'laps': (report.get('metrics') or {}).get('completed_laps'), 'bytes': os.path.getsize(path)}


def import_scenes(session_ids, name, tags=(), card_extra=None, progress=None, to_scenes=True, delete_sources=False, scene_meta=None):
    """シーン ZIP(セッション ID の並び)を 1 つのデータセット name に取り込む。戻り値は dataset_import.import_archive と同じ形 + rl"""
    if not session_ids or len(session_ids) > MAX_SCENES or len(set(session_ids)) != len(session_ids):
        raise ImportError_('シーンは 1〜%d 個の重ならないセッション ID にしてください' % MAX_SCENES)
    name = dataset_import.dataset_name_ok(name)
    dest = config.DATASET_DIR / name
    if dest.exists():
        raise FileExistsError('同じ名前のデータセット「%s」があります' % name)
    arcs = []
    for sid in session_ids:
        p, info = datasets.archive_path(sid)
        arcs.append((p, sid))
    total = sum(p.stat().st_size for p, _ in arcs)
    free = shutil.disk_usage(str(config.DATA_DIR if config.DATA_DIR.exists() else config.DATASET_DIR.parent)).free
    if free < 2.2 * total + (1 << 30):
        raise ImportError_('空き容量が足りません(ZIP %.1f GB の 2.2 倍 + 1 GB が必要、空き %.1f GB)' % (total / 1e9, free / 1e9))
    srcs = []
    for k, (p, sid) in enumerate(arcs):
        if progress:
            progress(0.02 + 0.13 * k / len(arcs), 'シーンを確かめています %d / %d' % (k + 1, len(arcs)))
        srcs.append(audit_scene(p, sid))
    for f in ('scene', 'start_key'):
        if len({s[f] for s in srcs}) != len(srcs):
            raise ImportError_('同じシーン・同じ開始姿勢が 2 つあります(%s)' % f)
    c0 = _stable(srcs[0]['contract'])
    for s in srcs[1:]:
        if _stable(s['contract']) != c0:
            diff = [k for k in s['contract'] if _stable(s['contract'][k]) != _stable(srcs[0]['contract'][k])]
            raise ImportError_('シーンの間で契約(%s)が違います。同じモード・同じ換算で記録したシーンだけをまとめてください' % '・'.join(diff))
    teacher = srcs[0]['teacher']
    stage = config.DATASET_DIR / ('.import-%s' % uuid.uuid4().hex[:8])
    root = stage / name
    root.mkdir(parents=True)
    try:
        tables = {t: {} for t in TABLES}
        for k, s in enumerate(srcs):
            if progress:
                progress(0.15 + 0.6 * k / len(srcs), 'まとめています %d / %d' % (k + 1, len(srcs)))
            with zipfile.ZipFile(s['path']) as z:
                for rel in s['members']:
                    if rel.startswith(('aligned/', 'samples/', 'can_bus/')):
                        out = root / rel
                    elif rel.startswith('simulation/'):
                        out = root / 'simulation' / 'scenes' / s['scene'] / rel[len('simulation/'):]
                    else:
                        continue
                    if out.exists():
                        raise ImportError_('まとめるときにファイルの名前が重なりました: %s' % rel)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(s['prefix'] + rel) as src, open(out, 'wb') as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
            for t, rows in s['tables'].items():
                for row in rows if isinstance(rows, list) else []:
                    tok = row.get('token') if isinstance(row, dict) else None
                    if tok is None:
                        continue
                    if tok in tables[t] and tables[t][tok] != row:
                        raise ImportError_('表 %s の token が重なりました' % t)
                    tables[t][tok] = row
        card = json.loads(json.dumps(srcs[0]['card']))
        splits = {k: [s['scene'] for s in srcs if s['split'] == k] for k in ('train', 'val', 'test')}
        frames = sum(s['frames'] for s in srcs)
        mode_frames = {}
        for s in srcs:
            for k, v in (s['card'].get('mode_frames') or {}).items():
                mode_frames[k] = mode_frames.get(k, 0) + int(v)
        version = 'v1.0-%s' % name
        card.update(name=name, version=version, scenes=len(srcs), train_scenes=len(splits['train']), val_scenes=len(splits['val']),
                    test_scenes=len(splits['test']), frames=frames, tof_frames=frames, hours=frames / 30 / 3600,
                    training_frames=sum(s['training_frames'] for s in srcs), source_scenes=[s['scene'] for s in srcs],
                    collection_schema=COLLECTION_SCHEMA, scene_duration_s=srcs[0]['frames'] / 30, continuous_frames_per_scene=srcs[0]['frames'],
                    split_scope='シーン(20 秒の連続した走り)ごとの分け方。開始姿勢は train と val で重ならない(同じコース・同じ教師の方策)',
                    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), mode_frames=mode_frames,
                    tags=list(card.get('tags') or []) + list(tags))
        card['notes'] = list(card.get('notes') or []) + [
            'JetFormer がシーン ZIP をまとめて取り込んだもの(sim/rl_import.py)。各シーンの記録は simulation/scenes/<シーン>/。',
            '強化学習モードのシーンでは、乱した操作(frames.jsonl の action_info.jf.l = 0)を正解に含むフレームを学習から外してある(dataset_card.json の label_mask)。']
        if card_extra:
            card.update(card_extra)
        scenes_pub = [{'source_session_id': s['sid'], 'scene': s['scene'], 'split': s['split'], 'frames': s['frames'],
                       'start': s['start'], 'scenario': s['scenario'], 'seed': s['seed'], 'completed_laps': s['laps'],
                       **((scene_meta or {}).get(s['sid']) or {})} for s in srcs]
        (root / version).mkdir()
        for t, rows in tables.items():
            (root / version / ('%s.json' % t)).write_text(json.dumps(list(rows.values()), ensure_ascii=False), encoding='utf-8')
        (root / 'simulation' / 'collection.json').write_text(json.dumps(
            {'schema': COLLECTION_SCHEMA, 'scenes': scenes_pub, 'contract': srcs[0]['contract'], 'teacher': teacher,
             'source_labels_unchanged': True, 'merged_by': 'jetformer/sim/rl_import.py'}, ensure_ascii=False, indent=1), encoding='utf-8')
        (root / 'dataset_card.json').write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
        (root / 'splits.json').write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding='utf-8')
        (root / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'collection_schema': COLLECTION_SCHEMA, 'frames': frames,
                                                         'scenes': scenes_pub}, ensure_ascii=False, indent=1), encoding='utf-8')
        (root / 'DATASET_README.txt').write_text(
            'JetFormer のシミュレーター(強化学習モード)で集めた合成データセット\n'
            '%d シーン(各 20 秒・600 フレーム・30 Hz)。ジャンル ID は 8(シミュレーター)。\n'
            '乱した操作を正解に含むフレームは future_valid = false・mode = 0(未設定)にしてある(dataset_card.json の label_mask)。\n'
            'splits.json の train / val はシーン単位。すべて合成データで、実車の性能の保証ではありません。\n' % len(srcs), encoding='utf-8')
        card2, summary, warnings, lmask = dataset_import.check_staged(root, name, progress)
        os.replace(root, dest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    info = dataset_import.after_import(name, dest, card2, summary, warnings, lmask, '%d シーン ZIP' % len(srcs), total, 'scene-zips',
                                       True, progress, to_scenes,
                                       extra_info={'session_ids': list(session_ids), 'collection_schema': COLLECTION_SCHEMA})
    deleted = []
    if delete_sources:
        for _, sid in arcs:
            try:
                datasets.delete(sid)
                deleted.append(sid)
            except Exception:                                           # noqa: BLE001 (消せなくても取り込みは成功)
                pass
    info['deleted_sources'] = deleted
    info['label_mask'] = {k: (lmask or {}).get(k) for k in ('masked_frames', 'scenes', 'rule_id')} if lmask else None
    return info
