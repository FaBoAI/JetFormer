"""シミュレーター本体の ZIP の導入(jetformer/sim/install.py)のテスト"""

import hashlib
import json
import os
import stat
import zipfile

import pytest

import _sim_testlib as L
from jetformer import config
from jetformer.sim import install, paths


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    inbox = tmp_path / 'inbox'
    inbox.mkdir()
    monkeypatch.setenv('JETFORMER_SIM_INBOX', str(inbox))
    return inbox


def test_install_extracts_verified_files_and_skips_big_data(env):
    sha = L.make_release_zip(env / 'sim.zip', extra={'最初にお読みください.txt': 'はじめに'.encode()})
    r = install.install_release('sim.zip')
    rid = '20261013-144013-%s' % sha[:8]
    assert r['release_id'] == rid and r['activated'] and not r['already_installed']
    root = paths.releases_dir() / rid
    assert (root / 'bridge.py').is_file() and (root / 'dist' / 'index.html').is_file()
    assert (root / 'COMPLETE-MANIFEST.json').is_file()
    assert (root / '最初にお読みください.txt').is_file()                  # 最上位の説明は入れて警告
    assert any('最初にお読みください.txt' in w for w in r['warnings'])
    assert not (root / 'dataset-downloads').exists() and not (root / 'runtime').exists()
    assert set(r['skipped']) == {'dataset-downloads/big.zip', 'runtime/datasets/x/evaluation-report.json'}
    assert stat.S_IMODE(os.stat(root / 'bridge.py').st_mode) == 0o644
    assert install.active_id() == rid
    rel = install.active_release()
    assert rel.files['dist/scene.js'] == hashlib.sha256(L.SCENE_JS).hexdigest()
    assert hashlib.sha256(L.SCENE_JS).hexdigest() in rel.trusted_sha256
    rec = json.loads((paths.releases_dir() / ('%s.json' % rid)).read_text())
    assert rec['default_course']['world_version'] == 'indoor-course-eval-3-turf-board-removed'
    assert not list(paths.releases_dir().glob('.tmp-*'))


def test_install_again_is_already_installed(env):
    L.make_release_zip(env / 'sim.zip')
    install.install_release('sim.zip')
    r = install.install_release('sim.zip')
    assert r['already_installed'] is True


def test_record_missing_reinstalls(env):
    L.make_release_zip(env / 'sim.zip')
    rid = install.install_release('sim.zip')['release_id']
    (paths.releases_dir() / ('%s.json' % rid)).unlink()
    r = install.install_release('sim.zip')
    assert r['already_installed'] is False and (paths.releases_dir() / ('%s.json' % rid)).is_file()


def test_sha_mismatch_aborts_and_leaves_nothing(env):
    L.make_release_zip(env / 'sim.zip', mutate={'dist/app.js': b"import { buildCourse } from './sceNe.js';\n"})
    with pytest.raises(install.InstallError, match='sha256'):
        install.install_release('sim.zip')
    assert install.list_releases() == [] and install.active_id() is None
    assert not list(paths.releases_dir().glob('.tmp-*'))


def test_unlisted_python_is_skipped(env):
    L.make_release_zip(env / 'sim.zip', extra={'evil.py': b'import os\n', 'dist/extra.js': b'alert(1)'})
    r = install.install_release('sim.zip')
    root = paths.releases_dir() / r['release_id']
    assert not (root / 'evil.py').exists() and not (root / 'dist' / 'extra.js').exists()
    assert {'evil.py', 'dist/extra.js'} <= set(r['skipped'])


def test_missing_manifest_file_aborts(env):
    L.make_release_zip(env / 'sim.zip', drop=('dist/sim-core.js',))
    with pytest.raises(install.InstallError, match='ZIP にありません'):
        install.install_release('sim.zip')


def test_required_file_missing(env):
    files = L.release_files()
    del files['training/speed_physics.py']
    L.make_release_zip(env / 'sim.zip', files=files)
    with pytest.raises(install.InstallError, match='speed_physics'):
        install.install_release('sim.zip')


@pytest.mark.parametrize('name', ['course-simulator/../evil.py', '/abs.py', 'course-simulator/a\\b.py', 'course-simulator/c:d.py'])
def test_bad_paths_rejected(env, name):
    L.make_release_zip(env / 'sim.zip', entries=[(zipfile.ZipInfo(name), b'x')])
    with pytest.raises(install.InstallError):
        install.install_release('sim.zip')


def test_symlink_rejected(env):
    info = zipfile.ZipInfo('course-simulator/link')
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    L.make_release_zip(env / 'sim.zip', entries=[(info, b'/etc/passwd')])
    with pytest.raises(install.InstallError, match='シンボリックリンク'):
        install.install_release('sim.zip')


def test_case_duplicate_rejected(env):
    L.make_release_zip(env / 'sim.zip', extra={'README.MD': b'dup'})
    with pytest.raises(install.InstallError, match='重複'):
        install.install_release('sim.zip')


def test_encrypted_and_unknown_compression_rejected(env):
    from jetformer.sim import ziputil
    info = zipfile.ZipInfo('course-simulator/secret.txt')
    info.flag_bits |= 0x1                    # zipfile では書けないので、中央ディレクトリの判定を直接確かめる
    with pytest.raises(ziputil.ZipCheckError, match='暗号化'):
        ziputil.entry_type(info)
    info = zipfile.ZipInfo('course-simulator/x.txt')
    info.compress_type = zipfile.ZIP_LZMA
    with pytest.raises(ziputil.ZipCheckError, match='圧縮方式'):
        ziputil.entry_type(info)
    # ZIP の中のバイトを書き換えて、暗号化の印を立てたエントリーを作る
    L.make_release_zip(env / 'sim.zip', extra={'secret.txt': b'x'})
    raw = bytearray((env / 'sim.zip').read_bytes())
    name = b'course-simulator/secret.txt'
    for sig in (b'PK\x03\x04', b'PK\x01\x02'):
        i = 0
        while True:
            i = raw.find(sig, i)
            if i < 0:
                break
            off = 30 if sig == b'PK\x03\x04' else 46
            if raw[i + off:i + off + len(name)] == name:
                flag = 6 if sig == b'PK\x03\x04' else 8
                raw[i + flag] |= 1
            i += 4
    (env / 'sim.zip').write_bytes(bytes(raw))
    with pytest.raises(install.InstallError, match='暗号化'):
        install.install_release('sim.zip')


def test_wrong_schema_and_two_tops(env):
    files = L.release_files()
    L.make_release_zip(env / 'a.zip', files=files, manifest=L.manifest_for(files, schema='other'))
    with pytest.raises(install.InstallError, match='schema'):
        install.install_release('a.zip')
    L.make_release_zip(env / 'b.zip', entries=[(zipfile.ZipInfo('other-top/x.txt'), b'x')])
    with pytest.raises(install.InstallError, match='1 つのフォルダー'):
        install.install_release('b.zip')


def test_size_limit(env, monkeypatch):
    monkeypatch.setattr(install, 'MAX_TOTAL', 1000)
    L.make_release_zip(env / 'sim.zip')
    with pytest.raises(install.InstallError, match='2 GiB'):
        install.install_release('sim.zip')


def test_inbox_listing_and_name_only(env):
    L.make_release_zip(env / 'sim.zip')
    (env / 'notes.zip').write_bytes(L.course_zip({'a.txt': 'x'}))
    (env / 'broken.zip').write_bytes(b'not a zip')
    (env / 'readme.txt').write_text('x')
    os.symlink(env / 'sim.zip', env / 'link.zip')
    items = {i['file']: i for i in install.list_inbox()}
    assert set(items) == {'sim.zip', 'notes.zip', 'broken.zip'}
    assert items['sim.zip']['kind'] == 'simulator' and not items['sim.zip']['installed']
    assert items['notes.zip']['kind'] == 'other' and items['broken.zip']['problem']
    for bad in ('../sim.zip', '/tmp/sim.zip', 'sub/sim.zip', 'link.zip', 'readme.txt', 'nothing.zip'):
        with pytest.raises(install.InstallError):
            install.install_release(bad)
    install.install_release('sim.zip')
    assert {i['file']: i for i in install.list_inbox()}['sim.zip']['installed']


def test_activation_waits_while_bridge_running(env):
    L.make_release_zip(env / 'a.zip')
    first = install.install_release('a.zip')['release_id']
    files = L.release_files()
    files['README.md'] = b'# new version\n'
    L.make_release_zip(env / 'b.zip', files=files, manifest=L.manifest_for(files, created_at='2026-11-01T10:00:00+09:00'))
    r = install.install_release('b.zip', bridge_running=lambda: True)
    assert r['activated'] is False and 'ブリッジ' in r['message']
    assert install.active_id() == first
    with pytest.raises(install.InstallError) as e:
        install.activate(r['release_id'], bridge_running=True)
    assert e.value.status == 409
    install.activate(r['release_id'])
    assert install.active_id() == r['release_id']
    with pytest.raises(install.InstallError, match='使っている版'):
        install.delete_release(r['release_id'])
    install.delete_release(first)
    assert [x['id'] for x in install.list_releases()] == [r['release_id']]


def test_op_lock_blocks_parallel_install(env):
    L.make_release_zip(env / 'sim.zip')
    assert paths.OP_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(install.InstallError) as e:
            install.install_release('sim.zip')
        assert e.value.status == 409
    finally:
        paths.OP_LOCK.release()


def test_js_pairs_and_durable_records(tmp_path, monkeypatch):
    """review で追加: コースとして実行してよい JS の組(同じフォルダーの scene.js・world.js)と、記録の書き方(.tmp を残さない)"""
    L.install_fake_release(tmp_path, monkeypatch)
    rel = install.active_release()
    assert rel.trusted_js == {(hashlib.sha256(L.SCENE_JS).hexdigest(), hashlib.sha256(L.WORLD_JS).hexdigest()),
                              (hashlib.sha256(L.V2_SCENE_JS).hexdigest(), hashlib.sha256(L.V2_WORLD_JS).hexdigest())}
    assert install.js_pairs({'a/scene.js': '1', 'b/world.js': '2', 'c/scene.js': '3', 'c/world.js': '4'}) == {('3', '4')}
    assert not list(paths.sim_dir().glob('*.tmp')) and not list(paths.releases_dir().glob('*.tmp'))
    assert json.loads(paths.active_file().read_text()) == {'release': rel.id}
