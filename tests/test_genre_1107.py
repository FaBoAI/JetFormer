"""ジャンルの付け替え(2026-11-07): 0 未設定・1〜3 駐車・4 練習コース 人間・5 練習コース AI・6 予選コース 人間・7 予選コース AI・
8 シミュレーター(ユーザーの決めで番号も振り直した)。

- tools/retag_genres.py: ジャンルの表・付け替えの表・書く・戻す(作り物の data/ で。本物の data/ は触らない)
- app.js: 選ぶ欄・走行タブのジャンルの名前はモデルの表から・シーンの詳細の保存で黙って 0 にしない(node)
- modes.update がほかの欄を消さない
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import retag_genres as R  # noqa: E402

STATIC = os.path.join(ROOT, 'jetformer', 'web', 'static')
NODE = shutil.which('node')
OLD_TABLE = [                                            # 11/7 の朝までの表
    {'id': 0, 'name': '未設定', 'description': 'ジャンル未設定(学習には使わない)'},
    {'id': 1, 'name': '駐車(1)', 'description': ''}, {'id': 2, 'name': '駐車(2)', 'description': ''},
    {'id': 3, 'name': '駐車(3)', 'description': ''}, {'id': 4, 'name': '高速走行', 'description': ''},
    {'id': 5, 'name': '低速走行', 'description': ''}, {'id': 6, 'name': '低速走行・良', 'description': 'x'},
    {'id': 7, 'name': '高速走行・良', 'description': 'y'}, {'id': 8, 'name': 'シミュレーター', 'description': 's'},
    {'id': 9, 'name': 'AI・きれいな速い周', 'description': 'z'},
]
MID_TABLE = [dict(m, retired=True) if 4 <= m['id'] <= 9 else dict(m) for m in OLD_TABLE] + [   # 11/7 の昼に仮に使った表
    {'id': 10, 'name': '練習コース 人間', 'description': ''}, {'id': 11, 'name': '練習コース AI', 'description': ''},
    {'id': 12, 'name': '予選コース 人間', 'description': ''}, {'id': 13, 'name': '予選コース AI', 'description': ''}]
NEW_NAMES = ['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '練習コース 人間', '練習コース AI', '予選コース 人間', '予選コース AI', 'シミュレーター']


# ---------------------------------------------------------------- ジャンルの表
def test_merged_modes_renumbers():
    for src in (OLD_TABLE, MID_TABLE):
        new = R.merged_modes(src)
        assert [m['id'] for m in new] == list(range(9)) and [m['name'] for m in new] == NEW_NAMES
        assert not any('retired' in m for m in new)
        assert new[8]['description'] == 's'                 # 引き継ぐ物(0〜3・8)は前の説明のまま
        assert R.merged_modes(new) == new                   # 何度通しても同じ
    assert OLD_TABLE[4] == {'id': 4, 'name': '高速走行', 'description': ''}      # 元の表を書き換えない
    with pytest.raises(SystemExit):
        R.merged_modes([dict(m, name='別') if m['id'] == 1 else m for m in OLD_TABLE])    # 駐車(1) の名前が違う
    with pytest.raises(SystemExit):
        R.merged_modes(MID_TABLE + [{'id': 14, 'name': '駐車(4)', 'description': ''}])   # 画面から足したジャンルは黙って消さない


def test_modes_update_keeps_other_fields(tmp_path, monkeypatch):
    from jetformer import modes
    monkeypatch.setattr(modes, 'MODES_FILE', tmp_path / 'modes.json')
    modes._save(R.merged_modes(MID_TABLE))
    modes.update(6, description='メモ')
    assert modes.get(6) == {'id': 6, 'name': '予選コース 人間', 'description': 'メモ'}
    assert modes.validate(4) == 4 and modes.add('駐車(4)')['id'] == 9        # AI 走行は 4(L-A の番号)で記録できる
    with pytest.raises(ValueError):
        modes.validate(12)                                   # 仮の 12 はもう無い


# ---------------------------------------------------------------- 付け替え(作り物の data/)
def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _log(data, log, mode, drive=None, scenes=((20.0, None),), broken=False, before=None):
    d = os.path.join(data, 'raw', log)
    os.makedirs(d, exist_ok=True)
    if broken:
        open(os.path.join(d, 'log.json'), 'w').close()      # 空の log.json(本物にも 3 本ある)
    else:
        lj = {'logfile': log, 'mode_id': mode, 'mode_name': 'x', 'drive': drive, 'location': '場所'}
        if before is not None:
            lj['mode_id_before'] = before
        _w(os.path.join(d, 'log.json'), lj)
    for i, (dur, segs) in enumerate(scenes):
        s = '%s-%04d' % (log, i + 1)
        segs = segs or [{'start_s': 0.0, 'end_s': dur, 'mode_id': mode, 'mode_name': 'x'}]
        sj = {'name': s, 'log': log, 'mode_id': segs[0]['mode_id'], 'mode_name': 'x', 'mode_segments': segs, 'duration_s': dur,
              'drive': drive}
        if before is not None:
            sj['mode_id_before'] = before
        _w(os.path.join(d, s, 'scene.json'), sj)


DRIVE = {'run': 'v2bgx_20261105-la', 'dry_run': False, 'params': {'steer_gain': 1.1}}


@pytest.fixture
def data(tmp_path):
    """11/7 の昼の状態(仮の 10〜13)と、明日の形(AI は L-A の 4 で記録)を混ぜた作り物"""
    d = str(tmp_path / 'data')
    _w(os.path.join(d, 'modes.json'), MID_TABLE)
    _log(d, '20261010-120000', 10, before=4)                            # 練習コース・人(仮 10)→ 4
    _log(d, '20261010-130000', 11, DRIVE, scenes=((20.0, None), (12.0, None)), before=5)   # 練習コース・AI(仮 11)→ 5
    _log(d, '20261106-170000', 12, before=7)                            # 予選コース・人(仮 12)→ 6
    _log(d, '20261106-171000', 13, DRIVE, before=4)                     # 予選コース・AI(仮 13)→ 7(前の番号 4 は残す)
    _log(d, '20261010-140000', 1)                                       # 駐車 → そのまま
    _log(d, 'ds-sim-x_20261014-1025', 8)                                # シミュレーター → 飛ばす
    _log(d, '20261023-200000', 11, DRIVE, broken=True, before=5)        # log.json が空・シーンは AI → 5(log.json は触らない)
    _log(d, '20261010-150000', 10, scenes=((20.0, [                     # 区間あり: 10 → 4・未設定と駐車はそのまま・境目もそのまま
        {'start_s': 0.0, 'end_s': 5.0, 'mode_id': 10}, {'start_s': 5.002, 'end_s': 10.0, 'mode_id': 0},
        {'start_s': 10.002, 'end_s': 20.0, 'mode_id': 1}]),))
    _log(d, '20261108-110000', 4, DRIVE)                                # 明日の AI(L-A の 4 で記録)→ 7・前の番号 4
    _log(d, '20261108-111000', 6)                                       # 明日の人(6 を選んだ)→ そのまま
    _log(d, '20261108-112000', 4)                                       # 明日の人で 4(練習コース 人間)を選んでしまった → 6
    _log(d, '20261106-183709', 1)                                       # 11/6 は駐車の印でも予選コース 人間
    _log(d, '20261108-120000', 2)                                       # 11/8 の駐車の印はそのまま(決まりは 11/6 だけ)
    os.makedirs(os.path.join(d, 'raw', '20261010-160000', '20261010-160000-0001'))   # scene.json が無いシーン → 飛ばす
    _w(os.path.join(d, 'raw', '20261010-160000', 'log.json'), {'mode_id': 11, 'mode_name': 'x', 'drive': DRIVE})
    return d


def _sj(data, scene):
    return json.load(open(os.path.join(data, 'raw', scene.rsplit('-', 1)[0], scene, 'scene.json'), encoding='utf-8'))


def _lj(data, log):
    return json.load(open(os.path.join(data, 'raw', log, 'log.json'), encoding='utf-8'))


def _digest(root):
    out = {}
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


def test_plan_rules(data):
    pl = R.plan(os.path.join(data, 'raw'))
    new = {r['scene']: (r['action'], r['new']) for r in pl['scenes']}
    assert new['20261010-120000-0001'] == ('retag', 4) and new['20261010-130000-0002'] == ('retag', 5)
    assert new['20261106-170000-0001'] == ('retag', 6) and new['20261106-171000-0001'] == ('retag', 7)
    assert new['20261108-110000-0001'] == ('retag', 7) and new['20261108-112000-0001'] == ('retag', 6)
    assert new['20261108-111000-0001'] == ('keep', 6) and new['20261010-140000-0001'] == ('keep', 1)
    assert new['20261106-183709-0001'] == ('retag', 6) and new['20261108-120000-0001'] == ('keep', 2)
    assert new['20261023-200000-0001'] == ('retag', 5)                  # log.json が空なら scene.json の drive で見る
    assert not any(r['log'].startswith('ds-') for r in pl['scenes'])
    skipped = {s.get('log') or s.get('scene'): s['why'] for s in pl['skipped']}
    assert 'ds-sim-x_20261014-1025' in skipped and '20261023-200000' in skipped and '20261010-160000-0001' in skipped
    lg = {r['log']: r for r in pl['logs']}
    assert (lg['20261106-171000']['new'], lg['20261108-110000']['new'], lg['20261010-140000']['action']) == (7, 7, 'keep')
    assert '20261023-200000' not in lg                                    # 空の log.json は書かない
    day = R.plan(os.path.join(data, 'raw'), '20261108')
    assert {r['log'] for r in day['scenes']} == {'20261108-110000', '20261108-111000', '20261108-112000', '20261108-120000'}
    other = R.plan(os.path.join(data, 'raw'), '20261010', '予選コース')    # --course: その日を予選コースとして分ける
    assert {r['scene']: r['new'] for r in other['scenes']}['20261010-120000-0001'] == 6


def test_apply_then_undo_restores_bytes(data, tmp_path):
    from jetformer import dataset
    before = _digest(data)
    b1, b2 = str(tmp_path / 'b_modes'), str(tmp_path / 'b_raw')
    with pytest.raises(SystemExit):                                       # 表をそろえる前は apply しない
        R.main(['apply', '--backup', b2, '--data', data, '--force'])
    assert R.main(['modes', '--backup', b1, '--data', data, '--force']) == 0
    mt = json.load(open(os.path.join(data, 'modes.json'), encoding='utf-8'))
    assert [m['name'] for m in mt] == NEW_NAMES
    assert R.main(['apply', '--backup', b2, '--data', data, '--force']) == 0
    s = _sj(data, '20261106-171000-0001')
    assert s['mode_id'] == 7 and s['mode_segments'] == [{'start_s': 0.0, 'end_s': 20.0, 'mode_id': 7}]
    assert s['mode_id_before'] == 4 and 'mode_name' not in s              # 最初の番号(走ったモデルの番号)はそのまま
    t = _sj(data, '20261108-110000-0001')
    assert t['mode_id'] == 7 and t['mode_id_before'] == 4 and t['mode_segments_before'][0]['mode_id'] == 4
    lj = _lj(data, '20261108-110000')
    assert (lj['mode_id'], lj['mode_name'], lj['mode_id_before'], lj['mode_name_before']) == (7, '予選コース AI', 4, 'x')
    assert lj['location'] == '場所' and lj['drive'] == DRIVE                # ほかの欄はそのまま
    m = _sj(data, '20261010-150000-0001')
    assert [(g['start_s'], g['end_s'], g['mode_id']) for g in m['mode_segments']] == [
        (0.0, 5.0, 4), (5.002, 10.0, 0), (10.002, 20.0, 1)]
    cam_t = (np.arange(0, 20.0, 1 / 30) * 1e6).astype(np.int64)          # 学習のフレームのジャンル(dataset._frame_modes)
    fm = dataset._frame_modes(m, cam_t)
    assert set(np.unique(fm[cam_t < 4.9e6])) == {4} and set(np.unique(fm[(cam_t > 5.1e6) & (cam_t < 9.9e6)])) == {0}
    assert set(np.unique(fm[cam_t > 10.1e6])) == {1}
    assert _sj(data, '20261010-140000-0001')['mode_id'] == 1 and _sj(data, '20261108-111000-0001')['mode_id'] == 6
    assert _sj(data, '20261108-112000-0001')['mode_id'] == 6 and _sj(data, '20261106-183709-0001')['mode_id'] == 6
    assert _sj(data, '20261108-120000-0001')['mode_id'] == 2
    assert open(os.path.join(data, 'raw', '20261023-200000', 'log.json')).read() == ''
    assert _sj(data, '20261023-200000-0001')['mode_id'] == 5
    assert _sj(data, 'ds-sim-x_20261014-1025-0001')['mode_id'] == 8
    once = _digest(data)
    assert R.main(['apply', '--backup', b2, '--data', data, '--force']) == 0     # 2 回目は何も変えない
    assert _digest(data) == once
    man = json.load(open(os.path.join(b2, 'manifest.json'), encoding='utf-8'))
    assert all(os.path.exists(e['backup']) for e in man['files'].values())
    # 戻す: apply のあとで人が変えたファイルは戻さない
    edited_p = os.path.join(data, 'raw', '20261010-120000', '20261010-120000-0001', 'scene.json')
    _w(edited_p, dict(json.load(open(edited_p, encoding='utf-8')), description='あとで変えた'))
    R.main(['undo', '--backup', b2, '--data', data])
    R.main(['undo', '--backup', b1, '--data', data])
    after = _digest(data)
    changed = sorted(k for k in before if before[k] != after.get(k))
    assert changed == ['raw/20261010-120000/20261010-120000-0001/scene.json']
    assert json.load(open(edited_p, encoding='utf-8'))['description'] == 'あとで変えた'


def test_undo_refuses_table_that_lacks_ids_in_use(tmp_path):
    d = str(tmp_path / 'data')
    _w(os.path.join(d, 'modes.json'), OLD_TABLE)                         # 0〜9 の表
    _log(d, '20261010-120000', 10)                                       # 記録は 10(0〜9 の表に無い番号)を使っている
    b = str(tmp_path / 'b')
    R.Writer(b, d).write(os.path.join(d, 'modes.json'), MID_TABLE, 'x')  # 表を 0〜13 にした、という記録
    done, kept = R.undo(b, d)
    assert done == 0 and kept and '10' in kept[0]
    assert json.load(open(os.path.join(d, 'modes.json'), encoding='utf-8')) == MID_TABLE
    assert R.undo(b, d, force=True)[0] == 1


def test_apply_refuses_when_scene_changed_after_plan(data, tmp_path):
    R.main(['modes', '--backup', str(tmp_path / 'b1'), '--data', data, '--force'])
    pl = R.plan(os.path.join(data, 'raw'))
    p = os.path.join(data, 'raw', '20261106-170000', '20261106-170000-0001', 'scene.json')
    _w(p, dict(json.load(open(p, encoding='utf-8')), mode_id=1, mode_segments=[{'start_s': 0.0, 'end_s': 20.0, 'mode_id': 1}]))
    w = R.Writer(str(tmp_path / 'b2'), data)
    with pytest.raises(SystemExit):
        R.apply(pl, R.names_of(R.merged_modes(MID_TABLE)), w, {'at': 'x'})


# ---------------------------------------------------------------- 画面(node)
SEL_STUB = r"""
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
function mkSel() {
  const s = { _v: '', options: [], _html: '', selectedIndex: -1 };
  Object.defineProperty(s, 'innerHTML', { get() { return this._html; }, set(h) {
    this._html = h;
    this.options = [...h.matchAll(/<option value="([^"]*)"[^>]*>([^<]*)<\/option>/g)].map((m) => ({ value: m[1], textContent: m[2] }));
    this._v = this.options.length ? this.options[0].value : ''; this.selectedIndex = this.options.length ? 0 : -1; } });
  Object.defineProperty(s, 'value', { get() { return this._v; }, set(v) {
    const i = this.options.findIndex((o) => o.value === String(v)); this.selectedIndex = i; this._v = i >= 0 ? String(v) : ''; } });
  return s;
}
const __el = {};
const $ = (q) => (__el[q] = __el[q] || mkSel());
const toast = () => {};
const renderModes = () => {}; const renderScenes = () => {};
"""


def _app_js():
    return open(os.path.join(STATIC, 'app.js'), encoding='utf-8').read()


def _defs(js, names):
    out = []
    for n in names:
        m = re.search(r'^(?:async )?function %s\(.*?^}\n' % n, js, re.S | re.M) or \
            re.search(r'^const %s = .*?;\n(?=\S|\n)' % n, js, re.S | re.M)      # 次の行が字下げでない・空行の前まで
        assert m, n
        out.append(m.group(0))
    return '\n'.join(out)


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_genre_selects_and_drive_names():
    js = _app_js()
    modes = [dict(m, scenes=0, minutes=0) for m in R.merged_modes(MID_TABLE)]
    code = SEL_STUB + _defs(js, ['MODE_COLORS', 'modeColor', 'modeName', 'modeRetired', 'modeOptionLabel',
                                 'fillModeSelect', 'loadModes', 'DRIVE_GENRE_ALIAS', 'driveGenreName', 'drModeText']) + r"""
const state = { modes: [], tab: 'record' };
const api = async () => MODES;
const MODES = %s;
(async () => {
  await loadModes();
  const ids = (q) => $(q).options.map((o) => o.value);
  const r1 = { rec: ids('#rec-mode'), recV: $('#rec-mode').value, bulk: ids('#bulk-mode'), m: $('#m-mode').options.map((o) => o.textContent) };
  $('#rec-mode').value = '6';
  await loadModes();
  const r2 = $('#rec-mode').value;
  // 走行タブ: L-A の表(4 高速走行・5 低速走行)を言い換えた選ぶ欄。名前は modes.json(4 = 練習コース 人間)でなく、選ぶ欄の字から
  const opts = [{ id: 4, name: '高速走行' }, { id: 5, name: '低速走行' }]
    .map((x) => `<option value="${x.id}">${driveGenreName(x.name, x.id)}</option>`).join('');
  $('#dr-mode').innerHTML = opts;
  $('#dr-mode').value = '4';
  const d4 = drModeText();
  const names = [driveGenreName('高速走行', 4), driveGenreName('低速走行', 5), driveGenreName('予選コース 人間', 6), driveGenreName('', 9)];
  process.stdout.write(JSON.stringify({ r1, r2, d4, names, colors: [0, 1, 2, 3, 4, 5, 6, 7, 8].map(modeColor) }));
})();
""" % json.dumps(modes, ensure_ascii=False)
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    r1 = got['r1']
    assert r1['rec'] == [str(i) for i in range(9)] and r1['recV'] == '1'      # 既定は前と同じ駐車(1)
    assert r1['bulk'] == [str(i) for i in range(9)]
    assert r1['m'][6] == '6: 予選コース 人間' and not any('(旧)' in t for t in r1['m'])
    assert got['r2'] == '6'
    assert got['d4'] == '予選コース AI'                                        # 「練習コース 人間」「4: 高速走行」と出さない
    assert got['names'] == ['予選コース AI', '練習コース AI', '予選コース 人間', 'ジャンル 9']   # 新しい表のモデルはそのまま
    assert len(set(got['colors'])) == 9


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_scene_save_does_not_zero_unknown_genre():
    js = _app_js()
    m = re.search(r"\$\('#m-save'\)\.addEventListener\('click', async \(\) => \{\n(.*?)\n\}\);", js, re.S)
    assert m
    body = m.group(1)
    code = SEL_STUB + r"""
const state = { modal: { name: 's1', mode_id: 13, mode_mixed: false } };
let sent = null;
const api = async (method, url, body) => { sent = body; return { mode_id: 13, mode_mixed: false, mode_segments: [] }; };
const parseTags = (s) => s.split(',').map((x) => x.trim()).filter(Boolean);
const sceneBadges = () => ''; const loadScenes = () => {}; const confirm = () => true;
$('#m-desc').value = 'd'; $('#m-tags').value = 't';
$('#m-mode').innerHTML = '<option value="0">0</option><option value="4">4</option>';   // 古い画面: 13 が表に無い
$('#m-mode').value = '13';                                                                   // → 空になる
(async () => {
""" + body + r"""
  process.stdout.write(JSON.stringify(sent));
})();
"""
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    sent = json.loads(r.stdout)
    assert set(sent) == {'description', 'tags'}                               # mode_id を送らない(黙って 0 = 未設定にしない)


def test_static_files():
    html = open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read()
    assert '4 練習コース 人間・5 練習コース AI・6 予選コース 人間・7 予選コース AI' in html and 'tools/retag_genres.py apply' in html
    js = _app_js()
    assert "fillModeSelect(rec, null, { hideRetired: true })" in js
    assert "ジャンル: ${drModeText()}" in js and "ジャンルを「${drModeText()}」に切り替えました" in js
    # 走行タブのジャンルはモデルが学んだ表から(L-A は自分の表の 4 で走る)。見せる名前だけ言い換える(番号なし・学んだ物だけ)
    assert "modeSel.innerHTML = (m.modes || []).filter((x) => x.id !== 0 && m.trained_modes.includes(x.id))" in js
    assert "<option value=\"${x.id}\">${esc(driveGenreName(x.name, x.id))}</option>" in js
    assert js.count('esc(driveGenreName(names[id] || modeName(+id), id))') == 3
    if NODE:
        r = subprocess.run([NODE, '--check', os.path.join(STATIC, 'app.js')], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
