"""シミュレーターの world hash(sim-core.js の normalizedWorld + FNV-1a を Python に移したもの)のテスト"""

import copy
import json
import os

import pytest

from jetformer.sim import worldhash as wh

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'sim')


def load(name):
    with open(os.path.join(FIX, name), encoding='utf-8') as f:
        return json.load(f)


def test_known_course_hashes():
    # シミュレーター本体(node で sim-core.js)と同じ値: v2 はスナップショットの manifest、v3 は Default
    assert wh.world_hash(load('v2-physics-world.json')) == '547d7e80'
    assert wh.world_hash(load('v3-physics-world.json')) == '0a464384'
    assert load('v2-manifest.json')['world_hash'] == '547d7e80'


def test_describe_stats_and_canonical():
    d = wh.describe(load('v3-physics-world.json'))
    assert d['stats'] == {'segments': 51, 'disks': 94, 'surfaces': 5, 'gates': 33, 'crossings': 1}
    assert len(d['canonical_sha256']) == 64
    assert d['canonical_sha256'] != wh.describe(load('v2-physics-world.json'))['canonical_sha256']


@pytest.mark.parametrize('value,text', [
    (0, '0'), (-0.0, '0'), (1, '1'), (1.0, '1'), (-2.5, '-2.5'), (0.1, '0.1'), (1e-7, '1e-7'), (1.5e-7, '1.5e-7'),
    (0.000001, '0.000001'), (1e21, '1e+21'), (1e20, '100000000000000000000'), (123456789012345680000, '123456789012345680000'),
    (0.30000000000000004, '0.30000000000000004'), (5e-324, '5e-324'), (1.7976931348623157e308, '1.7976931348623157e+308'),
    (float('nan'), 'null'), (float('inf'), 'null'), (2 ** 53, '9007199254740992'), (100, '100'), (.0095, '0.0095'),
])
def test_js_number(value, text):
    assert wh.js_number(value) == text


def test_js_stringify_escapes_like_json_stringify():
    assert wh.js_stringify({'a': 'x"y\\z\n', 'b': [True, None, 1.5]}) == '{"a":"x\\"y\\\\z\\n","b":[true,null,1.5]}'
    assert wh.js_stringify('\x01') == '"\\u0001"'
    assert wh.js_stringify('日本') == '"日本"'


def test_fnv_uses_utf16_first_unit_per_code_point():
    # 「😀」(U+1F600)は charCodeAt(0) = 0xD83D だけを使う
    assert wh.fnv1a32('😀') == wh.fnv1a32('\ud83d')
    assert wh.fnv1a32('') == '811c9dc5'


def test_post_foot_becomes_rectangle_and_zero_height_dropped():
    w = {'disks': [{'x': 1, 'z': 2, 'radius': .11, 'height': .03, 'kind': 'post-foot'},
                   {'x': 1, 'z': 2, 'radius': .02, 'height': 0, 'kind': 'post'}],
         'segments': [{'a': [0, 0], 'b': [1, 0], 'radius': 3, 'height': .1}]}
    n = wh.normalized_world(w)
    assert n['disks'] == [{'id': 'disk-0', 'x': 1, 'z': 2, 'radius': .11, 'height': .03, 'kind': 'post-foot',
                           'shape': 'rectangle', 'w': .16, 'd': .15}]
    assert n['segments'][0]['radius'] == 2.0            # 0..2 にはさむ
    assert n['segments'][0]['kind'] == 'rail'           # kind がなければ rail


def test_route_duplicates_and_closing_point_removed():
    w = {'route': [[0, 0], [0, 0], [1, 0], [1, 1], [0, 0]], 'bounds': {'minX': 1, 'maxX': 0}}
    n = wh.normalized_world(w)
    assert n['route'] == [[0, 0], [1, 0], [1, 1]]
    assert n['bounds'] == {'minX': -20, 'maxX': 20, 'minZ': -20, 'maxZ': 20}
    assert n['spawn'] == {'x': 0, 'z': 0, 'yaw': 0.0}


def test_key_order_does_not_change_hash_but_values_do():
    w = load('v3-physics-world.json')
    shuffled = {k: w[k] for k in reversed(list(w))}
    assert wh.world_hash(shuffled) == '0a464384'
    w2 = copy.deepcopy(w)
    w2['segments'][0]['height'] = 0.12
    assert wh.world_hash(w2) != '0a464384'
    assert wh.canonical_sha256(w2) != wh.canonical_sha256(w)
