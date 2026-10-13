"""シミュレーターのコースの登録簿(jetformer/sim/courses.py)のテスト"""

import copy
import hashlib
import io
import json
import zipfile

import pytest

import _sim_testlib as L
from jetformer.sim import courses, install, paths, worldhash


@pytest.fixture
def rel(tmp_path, monkeypatch):
    L.install_fake_release(tmp_path, monkeypatch)
    return install.active_release()


@pytest.fixture
def known(rel, monkeypatch):
    """偽のリリースの dist を、既知の v3 として扱う(本物の sha256 は作れないため)"""
    k = copy.deepcopy(courses.KNOWN)
    k[0]['scene_sha256'] = hashlib.sha256(L.SCENE_JS).hexdigest()
    k[0]['world_sha256'] = hashlib.sha256(L.WORLD_JS).hexdigest()
    k[1]['scene_sha256'] = hashlib.sha256(L.V2_SCENE_JS).hexdigest()
    k[1]['world_sha256'] = hashlib.sha256(L.V2_WORLD_JS).hexdigest()
    monkeypatch.setattr(courses, 'KNOWN', k)
    courses._release_courses_cache.clear()
    return k


def upload(tmp_path, files, name='c.zip', replace=False, top=None):
    p = tmp_path / name
    p.write_bytes(L.course_zip(files, top=top))
    return courses.import_package(p, name, replace=replace)


def jf_course(world=None, **meta):
    c = {'format': 'jetformer-sim-course/v1', 'name': '自宅コース 1', 'world_version': 'home-course-1-v1'}
    c.update(meta)
    return {'course.json': c, 'physics-world.json': world or L.simple_world()}


def test_release_and_snapshot_courses(rel, known):
    r = courses.list_courses()
    ids = [c['id'] for c in r['courses']]
    assert ids == ['default', 'indoor-course-eval-2-shortcut'] and r['default'] == 'default'
    d, v2 = r['courses']
    assert d['kind'] == 'release' and d['name'] == 'Default(標準コース)' and d['world_hash'] == '0a464384'
    assert d['features']['collection_plan'] and 'speed-rc' in d['features']['teachers'] and not d['deletable']
    assert v2['kind'] == 'snapshot' and v2['world_hash'] == '547d7e80' and v2['name'].startswith('旧コース v2')
    assert v2['features'] == {'teachers': ['slow', 'speed'], 'collection_plan': False}
    c = courses.get_course('indoor-course-eval-2-shortcut')
    assert c.file_for('scene.js') == rel.root / 'qa/course-snapshots/indoor-course-eval-2-shortcut/scene.js'
    assert courses.get_course('default').file_for('world.js') == rel.dist / 'world.js'
    assert courses.get_course('default').asset_for('carpet.jpg') == rel.dist / 'assets' / 'carpet.jpg'


def test_unknown_default_and_tampered_snapshot(rel, tmp_path):
    d = courses.get_course('default')
    assert d.world_hash is None and d.world_version == 'indoor-course-eval-3-turf-board-removed' and d.warnings
    snap = rel.root / 'qa/course-snapshots/indoor-course-eval-2-shortcut/scene.js'
    import os
    os.chmod(snap, 0o644)
    snap.write_bytes(b'// tampered\n')
    courses._release_courses_cache.clear()
    assert courses.get_course('indoor-course-eval-2-shortcut') is None      # sha256 が合わないスナップショットは出さない


def test_upload_jetformer_format_data_course(rel, known, tmp_path):
    files = jf_course(visual={'floor_texture': 'floor.jpg', 'surface_textures': {'shortcut': 'black-mat.jpg'}, 'colors': {'rail': '#FFFFFF'}})
    files['assets/floor.jpg'] = L.tiny_jpg()
    files['notes/other.bin'] = b'x'
    r = upload(tmp_path, files, top='home-course-1')
    c = r['course']
    assert not r['already_registered'] and c['kind'] == 'data' and c['renderer'] == 'generic' and c['deletable']
    assert c['id'] == 'home-course-1-v1' and c['world_hash'] == worldhash.world_hash(L.simple_world())
    assert c['stats']['gates'] == 6 and c['has_shortcut'] and c['size_m'] == [5.0, 4.0]
    assert any('使わないファイル' in w for w in c['warnings'])
    d = paths.courses_dir() / c['id']
    meta = json.loads((d / 'course.json').read_text())
    assert meta['visual']['colors']['rail'] == '#ffffff' and meta['source']['format'] == 'jetformer'
    assert (d / 'assets' / 'floor.jpg').is_file() and not (d / 'scene.js').exists()
    co = courses.get_course(c['id'])
    assert co.file_for('scene.js') == paths.STATIC / 'course-scene.js'
    assert co.asset_for('floor.jpg') == d / 'assets' / 'floor.jpg'
    assert co.asset_for('turf.jpg') == rel.dist / 'assets' / 'turf.jpg'      # ないものは dist から
    # 同じものをもう一度 → 登録済み
    assert upload(tmp_path, files, top='home-course-1')['already_registered']
    # 一覧の並び: default、snapshot、アップロード
    assert [x['id'] for x in courses.list_courses()['courses']][-1] == 'home-course-1-v1'


def test_course_data_module_is_safe_json(rel, tmp_path):
    files = jf_course(name='</script><b>&x')
    c = upload(tmp_path, files)['course']
    text = courses.course_data_module(c['id'])
    assert text.startswith('export default {') and text.rstrip().endswith('};')
    assert '<' not in text and '>' not in text and '&' not in text
    data = json.loads(text[len('export default '):].rstrip().rstrip(';'))
    assert data['name'] == '</script><b>&x' and data['world']['spawn'] == L.simple_world()['spawn']
    assert set(data['physics']) == {'segments', 'disks', 'surfaces', 'bounds'}
    # ブラウザ側の makeWorld(buildCourse().physics) と同じ形で hash が登録時と一致する
    world = dict(data['physics'], **data['world'])
    assert worldhash.world_hash(world) == c['world_hash']
    assert courses.course_data_module('default') is None


def test_snapshot_format_with_untrusted_js_is_data(rel, tmp_path):
    w = L.simple_world()
    raw = json.dumps(w).encode()
    man = {'world_version': 'snap-x-1', 'world_hash': worldhash.world_hash(w),
           'files': {'physics-world.json': hashlib.sha256(raw).hexdigest(), 'scene.js': hashlib.sha256(b'evil()').hexdigest(),
                     'world.js': hashlib.sha256(b'x').hexdigest()}}
    r = upload(tmp_path, {'manifest.json': man, 'physics-world.json': raw, 'scene.js': b'evil()', 'world.js': b'x'})
    c = r['course']
    assert c['kind'] == 'data' and any('JS は実行しません' in x for x in c['warnings'])
    assert not (paths.courses_dir() / c['id'] / 'scene.js').exists()


def test_trusted_js_when_matching_release_files(rel, known, tmp_path):
    w = L.v2_world()
    raw = json.dumps(w).encode()
    man = {'world_version': 'indoor-course-eval-2-copy', 'world_hash': '547d7e80',
           'files': {'physics-world.json': hashlib.sha256(raw).hexdigest(), 'scene.js': hashlib.sha256(L.V2_SCENE_JS).hexdigest(),
                     'world.js': hashlib.sha256(L.V2_WORLD_JS).hexdigest()}}
    r = upload(tmp_path, {'manifest.json': man, 'physics-world.json': raw, 'scene.js': L.V2_SCENE_JS, 'world.js': L.V2_WORLD_JS})
    c = r['course']
    assert c['kind'] == 'trusted-js' and c['renderer'] == 'simulator'
    assert any('同じ形のコース' in x for x in c['warnings'])
    co = courses.get_course(c['id'])
    assert co.file_for('scene.js') == paths.courses_dir() / c['id'] / 'scene.js'


def test_manifest_sha_and_hash_mismatch(rel, tmp_path):
    w = L.simple_world()
    raw = json.dumps(w).encode()
    with pytest.raises(courses.CourseError, match='sha256'):
        upload(tmp_path, {'manifest.json': {'world_version': 'a-1', 'files': {'physics-world.json': '0' * 64}}, 'physics-world.json': raw})
    with pytest.raises(courses.CourseError, match='world_hash'):
        upload(tmp_path, {'manifest.json': {'world_version': 'a-1', 'world_hash': 'deadbeef'}, 'physics-world.json': raw})


def bad(world_change):
    w = L.simple_world()
    world_change(w)
    return w


@pytest.mark.parametrize('change,msg', [
    (lambda w: w['segments'][0].update(kind='laser'), 'kind'),
    (lambda w: w['segments'][0].update(radius=0), 'radius'),
    (lambda w: w['segments'][0].update(height=6), 'height'),
    (lambda w: w['disks'][0].update(x=60), 'disks'),
    (lambda w: w['surfaces'].append(dict(w['surfaces'][0])), '重複'),
    (lambda w: w['surfaces'][0].update(mu=5), 'mu'),
    (lambda w: w.update(surfaces=[]), 'surfaces'),
    (lambda w: w['bounds'].update(minX=3), 'bounds'),
    (lambda w: w.update(route=w['route'][:2]), 'route'),
    (lambda w: w['route'].insert(1, [-1.49, .75]), '近すぎ'),
    (lambda w: w['route'].append([-1.5, .75]), '最初の点'),
    (lambda w: w['route'][0].__setitem__(0, -2.45), 'bounds の内側'),
    (lambda w: w['requiredCrossings'][0].update(surface_id='nope'), 'surface_id'),
    (lambda w: w['requiredCrossings'][0].update(minX=-3), '路面'),
    (lambda w: w['requiredCrossings'][0].update(direction=2), 'direction'),
    (lambda w: w.pop('spawn'), 'spawn'),
    (lambda w: w['spawn'].update(x=-1.0, z=0.02), '近すぎ'),
    (lambda w: w['spawn'].update(yaw=4), 'yaw'),
    (lambda w: w['segments'][0].update(a=[0, True]), '有限の数'),
])
def test_validation_errors(rel, tmp_path, change, msg):
    with pytest.raises(courses.CourseError) as e:
        upload(tmp_path, jf_course(world=bad(change)))
    assert any(msg in x for x in e.value.errors), e.value.errors


def test_nan_and_deep_json_rejected(rel, tmp_path):
    raw = json.dumps(L.simple_world()).replace('0.0095', 'NaN', 1)
    with pytest.raises(courses.CourseError, match='NaN'):
        upload(tmp_path, jf_course(world=None) | {'physics-world.json': raw})
    deep = {'format': 'jetformer-sim-course/v1', 'name': 'x', 'world_version': 'x-1', 'description': [[[[[[[[[['x']]]]]]]]]]}
    with pytest.raises(courses.CourseError, match='深すぎ'):
        upload(tmp_path, {'course.json': deep, 'physics-world.json': L.simple_world()})


def test_spawn_near_wall_warns(rel, tmp_path):
    w = L.simple_world()
    w['spawn'] = {'x': -0.5, 'z': .2, 'yaw': 0}           # 中央のレール(z=0)から 0.19 m
    c = upload(tmp_path, jf_course(world=w))['course']
    assert any('spawn が壁や柱に近い' in x for x in c['warnings'])


@pytest.mark.parametrize('meta,msg', [
    ({'id': 'default'}, '予約語'), ({'id': 'A'}, 'id'), ({'world_version': 'bad version'}, 'world_version'),
    ({'name': ''}, 'name'), ({'name': 'x' * 41}, 'name'), ({'visual': {'floor_texture': 'nothere.jpg'}}, 'floor_texture'),
    ({'visual': {'colors': {'rail': 'red'}}}, 'colors'),
])
def test_meta_errors(rel, tmp_path, meta, msg):
    with pytest.raises(courses.CourseError) as e:
        upload(tmp_path, jf_course(**meta))
    assert any(msg in x for x in e.value.errors), e.value.errors


def test_world_version_unique_and_replace(rel, tmp_path):
    upload(tmp_path, jf_course())
    w = L.simple_world()
    w['segments'][4]['b'] = [1.2, 0]
    with pytest.raises(courses.CourseError) as e:
        upload(tmp_path, jf_course(world=w))
    assert e.value.status == 409 and 'world_version' in str(e.value)
    r = upload(tmp_path, jf_course(world=w), replace=True)
    assert r['course']['world_hash'] == worldhash.world_hash(w)
    # 同じ id で別の world_version
    with pytest.raises(courses.CourseError) as e:
        upload(tmp_path, jf_course(world_version='home-course-1-v2', id='home-course-1-v1'))
    assert e.value.status == 409


def test_textures_reencoded_and_checked(rel, tmp_path):
    files = jf_course()
    files['assets/wall.png'] = L.tiny_png()
    c = upload(tmp_path, files)['course']
    data = (paths.courses_dir() / c['id'] / 'assets' / 'wall.png').read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    for name, payload, msg in [('assets/fake.jpg', L.tiny_png(), '拡張子'), ('assets/x.jpg', b'<svg/>', '拡張子'),
                               ('assets/Bad Name.jpg', L.tiny_jpg(), '名前'), ('assets/x.jpeg', L.tiny_jpg(), '名前')]:
        f = jf_course(world_version='t-%d' % len(name))
        f[name] = payload
        with pytest.raises(courses.CourseError, match=msg):
            upload(tmp_path, f)
    huge = L.tiny_png(w=5000, h=2)
    f = jf_course(world_version='t-huge')
    f['assets/huge.png'] = huge
    with pytest.raises(courses.CourseError, match='px'):
        upload(tmp_path, f)


def test_simulator_release_zip_rejected(rel, tmp_path):
    with pytest.raises(courses.CourseError, match='シミュレーター本体'):
        upload(tmp_path, {'COMPLETE-MANIFEST.json': {}, 'bridge.py': 'x', 'physics-world.json': L.simple_world()})


def test_zip_attacks(rel, tmp_path):
    for name in ('../x/physics-world.json', '/physics-world.json'):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            z.writestr(name, json.dumps(L.simple_world()))
        p = tmp_path / 'atk.zip'
        p.write_bytes(buf.getvalue())
        with pytest.raises(courses.CourseError):
            courses.import_package(p, 'atk.zip')
    p = tmp_path / 'notzip.zip'
    p.write_bytes(b'hello')
    with pytest.raises(courses.CourseError, match='ZIP'):
        courses.import_package(p, 'notzip.zip')
    with pytest.raises(courses.CourseError, match='physics-world'):
        upload(tmp_path, {'course.json': jf_course()['course.json']})


def test_default_setting_and_delete(rel, known, tmp_path):
    c = upload(tmp_path, jf_course())['course']
    courses.set_default(c['id'])
    assert courses.get_default() == c['id']
    assert courses.list_courses()['courses'][-1]['is_default']
    with pytest.raises(courses.CourseError) as e:
        courses.delete_course('default')
    assert e.value.status == 409
    with pytest.raises(courses.CourseError):
        courses.delete_course('indoor-course-eval-2-shortcut')
    courses.delete_course(c['id'])
    assert courses.get_default() == 'default' and courses.get_course(c['id']) is None
    with pytest.raises(courses.CourseError) as e:
        courses.set_default('nothing-here')
    assert e.value.status == 404
    (paths.courses_dir() / 'settings.json').write_text('{broken')
    assert courses.get_default() == 'default'


def test_export_then_reupload_keeps_hash(rel, known, tmp_path):
    data = courses.export_package('default')
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {'default/course.json', 'default/physics-world.json', 'default/README.txt'} <= names
    p = tmp_path / 'exp.zip'
    p.write_bytes(data)
    r = courses.import_package(p, 'exp.zip')
    assert r['already_registered'] and r['course']['id'] == 'default'      # 同じ world_version・同じ形
    # 編集して world_version を変えると新しいコース
    course = json.loads(z.read('default/course.json'))
    world = json.loads(z.read('default/physics-world.json'))
    course.update(world_version='my-course-1', id='my-course-1')
    course.pop('world_hash')
    world['segments'][0]['height'] = 0.2
    c = upload(tmp_path, {'course.json': course, 'physics-world.json': world})['course']
    assert c['world_hash'] != '0a464384' and c['kind'] == 'data'
    exp2 = zipfile.ZipFile(io.BytesIO(courses.export_package(c['id'])))
    w2 = json.loads(exp2.read('my-course-1/physics-world.json'))
    assert worldhash.world_hash(w2) == c['world_hash']
    v2 = zipfile.ZipFile(io.BytesIO(courses.export_package('indoor-course-eval-2-shortcut')))
    assert worldhash.world_hash(json.loads(v2.read('indoor-course-eval-2-shortcut/physics-world.json'))) == '547d7e80'


def test_v3_physics_passes_validation():
    clean, warnings = courses.validate_world(L.v3_world())
    assert worldhash.world_hash(clean) == '0a464384' and not [w for w in warnings if 'spawn' in w]
    clean, _ = courses.validate_world(L.v2_world())
    assert worldhash.world_hash(clean) == '547d7e80'


# ---------------------------------------------------------------- review で追加したもの
def test_trusted_js_needs_the_release_pair(rel, known, tmp_path):
    """JS は「同じフォルダーにそろった scene.js と world.js の組」が導入済みの版と一致するときだけ使う。
    版のほかのファイル(app.js など)や、組を入れ替えたものを scene.js として動かさせない"""
    assert rel.trusted_js == {(hashlib.sha256(L.SCENE_JS).hexdigest(), hashlib.sha256(L.WORLD_JS).hexdigest()),
                              (hashlib.sha256(L.V2_SCENE_JS).hexdigest(), hashlib.sha256(L.V2_WORLD_JS).hexdigest())}
    w = L.v2_world()
    raw = json.dumps(w).encode()
    app_js = L.release_files()['dist/app.js']
    for i, (scene, world) in enumerate(((app_js, L.V2_WORLD_JS), (L.V2_WORLD_JS, L.V2_SCENE_JS), (L.SCENE_JS, L.V2_WORLD_JS))):
        man = {'world_version': 'mixed-js-%d' % i, 'world_hash': '547d7e80',
               'files': {'physics-world.json': hashlib.sha256(raw).hexdigest(), 'scene.js': hashlib.sha256(scene).hexdigest(),
                         'world.js': hashlib.sha256(world).hexdigest()}}
        c = upload(tmp_path, {'manifest.json': man, 'physics-world.json': raw, 'scene.js': scene, 'world.js': world})['course']
        assert c['kind'] == 'data' and c['renderer'] == 'generic', (i, c['kind'])
        assert not (paths.courses_dir() / c['id'] / 'scene.js').exists()
    # 登録済みの trusted-js でも、あとで JS を差し替えたら簡易描画に落とす
    man = {'world_version': 'indoor-course-eval-2-copy', 'world_hash': '547d7e80',
           'files': {'physics-world.json': hashlib.sha256(raw).hexdigest(), 'scene.js': hashlib.sha256(L.V2_SCENE_JS).hexdigest(),
                     'world.js': hashlib.sha256(L.V2_WORLD_JS).hexdigest()}}
    c = upload(tmp_path, {'manifest.json': man, 'physics-world.json': raw, 'scene.js': L.V2_SCENE_JS, 'world.js': L.V2_WORLD_JS})['course']
    assert c['kind'] == 'trusted-js'
    (paths.courses_dir() / c['id'] / 'scene.js').write_bytes(app_js)
    co = courses.get_course(c['id'])
    assert co.kind == 'data' and co.file_for('scene.js') == paths.STATIC / 'course-scene.js'


def test_concurrent_uploads_of_same_course_do_not_500(rel, tmp_path):
    """同じコースを 2 つ同時に送っても OSError(os.replace の ENOTEMPTY)にならない(登録はロックで 1 つずつ)"""
    import threading
    p = tmp_path / 'same.zip'
    p.write_bytes(L.course_zip(jf_course()))
    results, errors = [], []

    def go():
        try:
            results.append(courses.import_package(p, 'same.zip'))
        except Exception as e:          # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=go) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errors, errors
    assert sorted(r['already_registered'] for r in results) == [False, True, True, True]
    assert [c.id for c in courses.list_course_objects() if c.deletable] == ['home-course-1-v1']


def test_sweep_stale_after_power_loss(rel, tmp_path):
    """電源断で残った作業ファイル: 古い .upload-*・.tmp-*・.del-* は消し、置き換え途中の .old-* は元に戻す。新しいものは残す"""
    import os
    import time
    c = upload(tmp_path, jf_course())['course']
    base = paths.courses_dir()
    # 置き換えの途中(前のコースを .old- にした直後)で止まった状態を作る
    os.replace(base / c['id'], base / '.old-0123456789ab')
    assert courses.get_course(c['id']) is None
    old = time.time() - 3600
    (base / '.upload-aaaaaaaaaaaa.zip').write_bytes(b'x' * 100)
    (base / '.tmp-bbbbbbbbbbbb').mkdir()
    (base / '.tmp-bbbbbbbbbbbb' / 'course.json').write_text('{}')
    (base / '.del-cccccccccccc').mkdir()
    for n in ('.upload-aaaaaaaaaaaa.zip', '.tmp-bbbbbbbbbbbb', '.del-cccccccccccc'):
        os.utime(base / n, (old, old))
    (base / '.upload-new000000000.zip').write_bytes(b'y')          # 受け取り中(新しい)
    r = courses.sweep_stale()
    assert r['restored'] == [c['id']] and r['removed'] == 3
    assert courses.get_course(c['id']) is not None and courses.get_course(c['id']).world_hash == c['world_hash']
    assert sorted(os.listdir(base)) == ['.upload-new000000000.zip', c['id']]
    # 置き換えが終わっていた(新しいコースがある)なら .old- は消すだけ
    (base / '.old-feedfeedfeed').mkdir()
    (base / '.old-feedfeedfeed' / 'course.json').write_text(json.dumps({'format': 'jetformer-sim-course-meta/v1', 'id': c['id']}))
    r = courses.sweep_stale()
    assert r == {'removed': 1, 'restored': []} and courses.get_course(c['id']) is not None


def test_delete_and_default_are_durable(rel, tmp_path):
    import os
    c = upload(tmp_path, jf_course())['course']
    courses.set_default(c['id'])
    base = paths.courses_dir()
    assert json.loads((base / 'settings.json').read_text()) == {'default': c['id']}
    assert not (base / 'settings.json.tmp').exists()
    courses.delete_course(c['id'])
    assert courses.get_default() == 'default'
    assert [n for n in os.listdir(base) if n.startswith('.')] == []
