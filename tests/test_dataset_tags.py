"""データセットのタグ(シミュレーター)のテスト: 自動の判定・付け直し・Hugging Face のカードのタグ・Hub の検索の絞り込み"""

import json
import types

import pytest

from jetformer import config, dataset, hf_export


def test_normalize_tags_merges_simulator_aliases():
    assert dataset.normalize_tags(['シミュレーター', 'sim', ' indoor ', '', 'Emulator', 'indoor']) == ['simulator', 'indoor']


@pytest.mark.parametrize('meta,want', [
    ({'tags': ['indoor']}, False), ({'tags': ['シミュレーター']}, True), ({'tags': ['SIM']}, True),
    ({'source': 'simulator'}, True), ({}, False)])
def test_is_sim_scene(meta, want):
    assert dataset.is_sim_scene(meta) is want


@pytest.mark.parametrize('n,n_sim,tags,want', [
    (5, 0, [], 'real'), (5, 5, [], 'simulator'), (5, 2, [], 'mixed'), (5, 0, ['simulator'], 'simulator')])
def test_data_source(n, n_sim, tags, want):
    assert dataset.data_source(n, n_sim, tags) == want


def _card(tmp_path, monkeypatch, **extra):
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path)
    root = tmp_path / 'ds'
    (root / 'v1.0-ds').mkdir(parents=True)
    card = {'name': 'ds', 'scenes': 4, 'frames': 100, 'created_at': '2026-10-12T23:00:00', 'modes': [], 'mode_frames': {}}
    card.update(extra)
    (root / 'dataset_card.json').write_text(json.dumps(card), encoding='utf-8')
    (root / 'splits.json').write_text('{"train": [], "val": []}', encoding='utf-8')
    (root / 'v1.0-ds' / 'scene.json').write_text('[]', encoding='utf-8')
    return root


def test_set_dataset_tags_toggles_simulator(tmp_path, monkeypatch):
    root = _card(tmp_path, monkeypatch)
    c = dataset.set_dataset_tags('ds', ['indoor', 'シミュレーター'])
    assert c['tags'] == ['indoor', 'simulator'] and c['source'] == 'simulator'
    c = dataset.set_dataset_tags('ds', ['indoor'])
    assert c['tags'] == ['indoor'] and c['source'] == 'real'
    assert json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))['source'] == 'real'
    listed = {d['name']: d for d in dataset.list_datasets()}
    assert listed['ds']['tags'] == ['indoor'] and listed['ds']['source'] == 'real'


def test_simulator_tag_cannot_be_removed_when_sim_scenes_inside(tmp_path, monkeypatch):
    _card(tmp_path, monkeypatch, sim_scenes=2, tags=['simulator'], source='mixed')
    c = dataset.set_dataset_tags('ds', [])
    assert c['tags'] == ['simulator'] and c['source'] == 'mixed'


def test_old_dataset_defaults_to_real(tmp_path, monkeypatch):
    _card(tmp_path, monkeypatch)
    d = dataset.list_datasets()[0]
    assert d['tags'] == [] and d['source'] == 'real'


def test_set_dataset_tags_rejects_bad_names(tmp_path, monkeypatch):
    _card(tmp_path, monkeypatch)
    for bad in ('../x', 'a/b', 'missing'):
        with pytest.raises(KeyError):
            dataset.set_dataset_tags(bad, ['simulator'])


def test_hub_tags_and_card(tmp_path, monkeypatch):
    assert hf_export.hub_tags({'source': 'real', 'tags': ['indoor']}) == hf_export.BASE_TAGS + ['indoor']
    tags = hf_export.hub_tags({'source': 'simulator', 'tags': ['simulator', '屋内', 'Night Run']})
    assert tags[:5] == hf_export.BASE_TAGS and 'simulator' in tags and 'synthetic' in tags and 'night-run' in tags
    assert len(tags) == len(set(tags))
    _card(tmp_path, monkeypatch, tags=['simulator'], source='simulator')
    md = hf_export.make_card('ds', 'someone/jetformer-ds', {'train': 80, 'validation': 20}, include_nuscenes=False)
    head = md.split('---')[1]
    assert '- simulator' in head and '- synthetic' in head and '- jetformer' in head
    assert 'データの種類: シミュレーター' in md


def _fake_api(monkeypatch, items):
    calls = {}

    class Api(object):
        def list_datasets(self, **kw):
            calls.update(kw)
            return [types.SimpleNamespace(id=i, tags=t, private=False, last_modified=None, downloads=0) for i, t in items]
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    return calls


def test_list_hub_datasets_kind_filter(monkeypatch):
    items = [('a/real', ['jetformer', 'rc-car']), ('b/sim', ['jetformer', 'simulator', 'synthetic', 'license:other'])]
    calls = _fake_api(monkeypatch, items)
    allx = hf_export.list_hub_datasets(kind='all')
    assert [d['repo_id'] for d in allx] == ['a/real', 'b/sim'] and allx[1]['simulator'] and 'license:other' not in allx[1]['tags']
    assert [d['repo_id'] for d in hf_export.list_hub_datasets(kind='real')] == ['a/real']
    hf_export.list_hub_datasets(kind='simulator')
    assert calls['filter'] == ['jetformer', 'simulator']


def test_broken_dataset_is_listed_and_deletable(tmp_path, monkeypatch):
    """作成の途中で電源が切れて dataset_card.json が空になったデータセットがあっても、一覧は止まらず、削除できる"""
    _card(tmp_path, monkeypatch)
    broken = tmp_path / 'half_built'
    (broken / 'aligned').mkdir(parents=True)
    (broken / 'dataset_card.json').write_text('', encoding='utf-8')
    (tmp_path / '.building-other').mkdir()                         # 作成中のフォルダは一覧に出さない
    (tmp_path / '.building-other' / 'dataset_card.json').write_text('{"name": "other"}', encoding='utf-8')
    listed = {d['name']: d for d in dataset.list_datasets()}
    assert set(listed) == {'ds', 'half_built'} and listed['half_built']['broken'] and not listed['ds'].get('broken')
    dataset.delete_dataset('half_built')
    assert not broken.exists()
    for bad in ('../x', '.building-other', ''):
        with pytest.raises(KeyError):
            dataset.delete_dataset(bad)
