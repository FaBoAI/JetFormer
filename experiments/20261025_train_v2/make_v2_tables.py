#!/usr/bin/env python3
"""T1: データセット v2 のフレームの表を作る(2026-10-25。plan.md 2.3・protocol_v2.json の tables・bfast_recipe)。

numpy・OpenCV(読み込みだけ)。torch・GPU は使わない。data/raw・data/labels は読むだけ。
書くのは data/derived/dataset_v2_20261025/ の 7 つだけ:
  drop_v2h.npz・drop_v2m.npz・weights_v2m.npz・remap_v2m.npz・drop_bfast.npz・weights_bfast.npz・tables_card.json

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 3G --need-mb 1500 -- nice -n 10 \
      /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/make_v2_tables.py

止まる(終わりの値 1。表を書かない):
  protocol の固定が合わない・封の 14 本のどれかが source_scenes にある・--ai-logs(カードの ai_logs)が protocol の 46 本と違う・
  検証が H1 の 11 シーンと違う・介入の窓(intervention_val_20261011 の 29)のフレームと学習に残すフレームが重なる・
  手本の周が protocol の一覧と違う

マスク(plan_checks.py と同じ。inventory.py の process_log の写しに足して、ラベルの行ごとに取り出す):
  新しい人 = 棚卸しの (a) 人の走行・衝突の前後(ラベルの near_collision)でない・ジャンル 4・5・clean_driving に無い・
            10/10 の介入の窓 [t, t+5 秒) の外
  残す新しい人 = 新しい人 かつ(走りの区間: 前後 1 秒の前進 eRPM の中央値 >= 3,000 か 交代の直後 3 秒)
  手本の周(B-FAST・v2-M 重み 1.0) = protocol の bfast_recipe.teacher_laps(10/24 2 回目の B のきれいな 9.0 秒未満、10 周)
  速めの周(v2-M 重み 0.5) = protocol の tables.v2m_half_laps(10/23 の B のきれいな 9.0〜9.5 秒、4 周)
  周のフレーム = R1 の起点 a 〜 次の起点 b の [a, b) の AI 側のフレーム(inventory の周と同じ)
  B-FAST の R1 の窓 = 手本の周のフレームで、その記録の R1 の起点 ta(le.course_anchors = offline_rl.course_passes の ta)の
            [ta − 1, ta + 1) → 重み 0.5

表(どれも scene・frame・値。データセットの全シーン(学習 + 検証)の全フレーム。検証のシーンは drop=False・重み 1・付け替え -1):
  drop_v2h    学習シーン: clean_driving の学習 41 シーンは全部残す(B と同じ)。AI の記録のシーンは「残す新しい人」だけ残す
  drop_v2m    v2-H + 手本の周 + 速めの周
  weights_v2m 速めの周 0.5、ほか 1.0
  remap_v2m   手本の周・速めの周 → 9、ほか −1(そのまま)
  drop_bfast  clean_driving の学習 41 シーン + 手本の周(新しい人・10/23 の周は入れない)
  weights_bfast 手本の周の R1 の窓 0.5、ほか 1.0
  drop には reason(int8)も入れる: 0 残す 1 検証 2 AI の記録の外したフレーム(人) 3 同(AI) 4 ラベルなし
学習の衝突の前後(--impact-g 2.5)・future_valid・vehicle_valid・ジャンル 0・間引き 3 は train.py が掛ける(表では外さない)。
"""
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
HERE = Path('/home/jetson/jetformer/experiments/20261025_train_v2')
INV_PY = Path('/home/jetson/jetformer/experiments/20261025_dataset_v2/inventory.py')
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL = REP / 'protocol_v2.json'
NAME = 'dataset_v2_20261025'
DS = ROOT / 'data/datasets' / NAME
OUTD = ROOT / 'data/derived' / NAME
RUN_ERPM, RUN_K = 3000.0, 61          # plan_checks.py と同じ
R1_WIN_S = 1.0
W_HALF, W_R1 = 0.5, 0.5
REMAP_ID = 9
SESS_TEACH = '10/24 2 回目(16:01〜)'
REASON = {'keep': 0, 'val': 1, 'ai_log_human': 2, 'ai_log_ai': 3, 'no_label': 4}


def fail(msg):
    print('止める: ' + msg, flush=True)
    sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- inventory の写し
EXTRA = '''
    # ---- make_v2_tables で足した取り出し(ここから。plan_checks.py の EXTRA と同じマスク)
    _e = -erpm
    if n >= PC_RUN_K:
        from numpy.lib.stride_tricks import sliding_window_view as _swv
        _h = PC_RUN_K // 2
        _pad = np.concatenate([np.full(_h, np.nan), _e, np.full(_h, np.nan)])
        _med = np.nanmedian(_swv(_pad, PC_RUN_K), axis=1)
    else:
        _med = np.full(n, np.nanmedian(_e) if n else np.nan)
    _run = _med >= PC_RUN_ERPM
    _ivw = np.zeros(n, bool)
    for _w in PC_IV.get(log, []):
        _ivw |= (t >= _w) & (t < _w + 5.0)
    _new = a_mask & ~near & ~base_tr & ~base_va & np.isin(mode, (4, 5)) & ~_ivw
    PC_OUT[log] = {'session': session, 't_s': t.copy(), 't_us': t_us.copy(), 'ai': ai.copy(), 'hum': hum.copy(),
                   'new': _new, 'run': _run, 'tk': b_mask.copy(), 'keep_new': _new & (_run | b_mask),
                   'near': near.copy(), 'ivw': _ivw, 'mode': mode.copy(), 'idx_in': idx_in.copy(),
                   'anchors': np.asarray(anchors, float).copy(), 'laps': [dict(x) for x in laps],
                   'base_tr': base_tr.copy(), 'base_va': base_va.copy()}
    # ---- make_v2_tables で足した取り出し(ここまで)
'''


def fast_scene_span(sdir):
    """明るさの画像を読まない scene_brightness(plan_checks.py と同じ)"""
    try:
        with np.load(sdir / 'CAM_FRONT.npz', allow_pickle=False) as z:
            ts = z['timestamps_us'].astype(np.int64)
    except Exception:
        names = sorted(os.listdir(sdir / 'CAM_FRONT')) if (sdir / 'CAM_FRONT').is_dir() else []
        names = [x for x in names if x.endswith('.jpg') and x[:-4].isdigit()]
        ts = np.array([int(x[:-4]) for x in names], np.int64)
    idx = np.arange(0, len(ts), 3)
    span = (int(ts[0]), int(ts[-1])) if len(ts) else None
    nan = np.full(len(idx), np.nan)
    return (ts[idx] if len(ts) else ts), nan, nan.copy(), len(ts), span


def iv_windows():
    card = json.load(open(ROOT / 'data/datasets/intervention_val_20261011/dataset_card.json', encoding='utf-8'))
    out = {}
    for w in card['intervention']['windows']:
        if w.get('included'):
            out.setdefault(w['log'], []).append(float(w['t_s']))
    return out


def per_log_masks(logs):
    spec = importlib.util.spec_from_file_location('inv_copy_t1', str(INV_PY))
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    inv.scene_brightness = fast_scene_span
    src = inspect.getsource(inv.process_log)
    anchor = "    rec['candidates'] = {"
    if src.count(anchor) != 1:
        fail('inventory.process_log の形が変わった')
    src = src.replace(anchor, EXTRA + anchor)
    ns = inv.__dict__
    ns.update(PC_RUN_K=RUN_K, PC_RUN_ERPM=RUN_ERPM, PC_IV=iv_windows(), PC_OUT={})
    exec(compile(src, str(INV_PY) + ' (make_v2_tables の写し)', 'exec'), ns)
    modes = inv.mode_table()
    membership = inv.dataset_membership()
    all_logs = sorted(d for d in os.listdir(inv.RAW) if d[:8] in inv.DAYS and (inv.RAW / d).is_dir())
    sess = inv.sessions(all_logs)
    for lg in logs:
        ns['process_log'](lg, membership, modes, sess.get(lg))
        if lg not in ns['PC_OUT']:
            fail('%s の取り出しができません(ラベルか記録が読めない)' % lg)
        print('  マスク', lg, sess.get(lg), flush=True)
    return ns['PC_OUT'], inv


# ---------------------------------------------------------------- 周
def lap_rows(d, t_start, lap_s):
    """記録の取り出し d の、起点 t_start の周 [a, b) の AI 側の行のマスクと周の情報"""
    lp = [x for x in d['laps'] if abs(x['t_start'] - t_start) < 2e-3]
    if len(lp) != 1:
        fail('周 %.3f が見つかりません' % t_start)
    lp = lp[0]
    A = d['anchors']
    j = int(np.argmin(np.abs(A - t_start)))
    if abs(A[j] - t_start) > 2e-3 or j + 1 >= len(A):
        fail('周 %.3f の起点が見つかりません' % t_start)
    a, b = float(A[j]), float(A[j + 1])
    if abs((b - a) - lap_s) > 2e-3:
        fail('周 %.3f の長さが %.3f(protocol は %.3f)' % (t_start, b - a, lap_s))
    m = (d['t_s'] >= a) & (d['t_s'] < b)
    if not d['ai'][m].all():
        fail('周 %.3f に AI 側でないフレームがあります' % t_start)
    return m, lp, a, b


def select_laps(po, rule):
    """規則で周を選び直す(protocol の一覧と同じかを確かめるため)"""
    out = []
    for lg, d in sorted(po.items()):
        for x in d['laps']:
            if x['who'] == 'ai' and x['clean'] and not x['suspect'] and x.get('model') == 'B' and rule(lg, d, x):
                out.append((lg, round(x['t_start'], 3), round(x['lap_s'], 3), x['samples']))
    return out


def save_table(path, scenes, frames, key, values, extra=None):
    arrs = {'scene': np.array(scenes), 'frame': np.asarray(frames, np.int32), key: values}
    if extra:
        arrs.update(extra)
    tmp = str(path) + '.tmp.npz'
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def main():
    t0 = datetime.now()
    r = subprocess.run(['sha256sum', '-c', 'protocol_v2.sha256'], cwd=str(REP), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        fail('protocol の固定が合いません')
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    tb = p['tables']
    sealed = set(p['seal']['logs'])
    ai_logs = [x for v in tb['dataset']['ai_logs'].values() for x in v]
    h1 = sorted(p['holdouts']['H1']['scenes'])
    card = json.loads((DS / 'dataset_card.json').read_text(encoding='utf-8'))
    splits = json.loads((DS / 'splits.json').read_text(encoding='utf-8'))
    base_card = json.loads((ROOT / 'data/datasets/clean_driving_20261011/dataset_card.json').read_text(encoding='utf-8'))
    base_splits = json.loads((ROOT / 'data/datasets/clean_driving_20261011/splits.json').read_text(encoding='utf-8'))
    base_train = set(base_splits['train'])

    # 止める条件
    if card.get('built_by') != 'tools/build_ai_dataset.py':
        fail('build_ai_dataset.py で作ったデータセットではありません')
    if card['ai_mix']['ai_logs'] != ai_logs:
        fail('カードの ai_logs が protocol の 46 本と違います')
    bad = [s for s in card['source_scenes'] if s.rsplit('-', 1)[0] in sealed]
    if bad:
        fail('封の記録が source_scenes にあります: %s' % bad[:5])
    if sorted(splits['val']) != h1:
        fail('検証が H1 の 11 シーンと違います')
    if card['ai_mix']['val_ai_logs']:
        fail('val_ai_logs が空ではありません')
    miss_base = [s for s in base_card['source_scenes'] if s not in card['source_scenes']]
    if miss_base:
        fail('clean_driving のシーンが足りません: %s' % miss_base)

    # ラベルの行ごとのマスク(AI の記録 46 本)
    print('inventory の写しでマスクを取り出す(46 本)', flush=True)
    po, inv = per_log_masks(ai_logs)

    # 周: protocol の一覧と、規則で選び直したもの
    teach_p = [(x['log'], round(x['t_start_s'], 3), round(x['lap_s'], 3), x['samples']) for x in p['bfast_recipe']['teacher_laps']]
    half_p = [(x['log'], round(x['t_start_s'], 3), round(x['lap_s'], 3), x['samples']) for x in tb['v2m_half_laps']]
    teach_r = select_laps(po, lambda lg, d, x: d['session'] == SESS_TEACH and x['lap_s'] < 9.0)
    half_r = select_laps(po, lambda lg, d, x: lg.startswith('20261023') and 9.0 <= x['lap_s'] < 9.5)
    other_fast = select_laps(po, lambda lg, d, x: x['lap_s'] < 9.5 and not lg.startswith('20261023')
                             and d['session'] != SESS_TEACH)
    teach_half_24 = select_laps(po, lambda lg, d, x: d['session'] == SESS_TEACH and 9.0 <= x['lap_s'] < 9.5)
    sub9_23 = select_laps(po, lambda lg, d, x: lg.startswith('20261023') and x['lap_s'] < 9.0)
    if sorted(teach_r) != sorted(teach_p):
        fail('手本の周が protocol の一覧と違います: 規則 %s / protocol %s' % (teach_r, teach_p))
    if sorted(half_r) != sorted(half_p):
        fail('速めの周が protocol の一覧と違います: 規則 %s / protocol %s' % (half_r, half_p))

    lapmask = {lg: {'teach': np.zeros(len(d['t_s']), bool), 'half': np.zeros(len(d['t_s']), bool),
                    'r1win': np.zeros(len(d['t_s']), bool)} for lg, d in po.items()}
    lap_info = {'teacher': [], 'half': []}
    for kind, lst in (('teach', teach_p), ('half', half_p)):
        for lg, ts, ls, smp in lst:
            d = po[lg]
            m, lp, a, b = lap_rows(d, ts, ls)
            lapmask[lg][kind] |= m
            lap_info['teacher' if kind == 'teach' else 'half'].append(
                {'log': lg, 't_start_s': ts, 'lap_s': ls, 'label_rows': int(m.sum()), 'inventory_samples': smp,
                 'anchor_a_s': round(a, 3), 'anchor_b_s': round(b, 3)})
    for lg, d in po.items():
        w = np.zeros(len(d['t_s']), bool)
        for ta in d['anchors']:
            w |= (d['t_s'] >= ta - R1_WIN_S) & (d['t_s'] < ta + R1_WIN_S)
        lapmask[lg]['r1win'] = w & lapmask[lg]['teach']

    # 棚卸しの数(ラベルの行・間引き 3 = plan_checks と同じ数え方)
    def smp(m, d):
        return int(np.sum(m & (d['idx_in'] % 3 == 0)))
    inv_counts = {'keep_new_by_session': {}, 'new_by_session': {}}
    for lg, d in po.items():
        if lg.startswith('20261010'):
            g = '10/10 介入の 8 本(窓の外)' if lg in iv_windows() else '10/10 ほかの AI 走行 3 本'
        elif lg.startswith('20261023'):
            g = '10/23'
        else:
            g = '10/24 2 回目'
        for key, mm in (('keep_new_by_session', d['keep_new']), ('new_by_session', d['new'])):
            inv_counts[key][g] = inv_counts[key].get(g, 0) + smp(mm, d)
    inv_counts['keep_new'] = sum(inv_counts['keep_new_by_session'].values())
    inv_counts['teacher'] = sum(smp(lapmask[lg]['teach'], po[lg]) for lg in po)
    inv_counts['half'] = sum(smp(lapmask[lg]['half'], po[lg]) for lg in po)
    inv_counts['teacher_r1win'] = sum(smp(lapmask[lg]['r1win'], po[lg]) for lg in po)
    print('棚卸しの数え方:', json.dumps(inv_counts, ensure_ascii=False), flush=True)

    # 介入の検証セットの実際のフレーム(記録ごとの時刻)。窓 [t, t+5) の外でも、このフレームは学習に残さない
    iv_root = ROOT / 'data/datasets/intervention_val_20261011'
    iv_card = json.loads((iv_root / 'dataset_card.json').read_text(encoding='utf-8'))
    iv_ts, iv_off = {}, {}
    for s in iv_card['source_scenes']:
        with np.load(iv_root / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
            iv_ts.setdefault(s.rsplit('-', 1)[0], []).append(z['timestamps_us'].astype(np.int64))
            iv_off.setdefault(s.rsplit('-', 1)[0], []).append(z['t_since_takeover_s'].astype(np.float64))
    iv_ts = {k: np.concatenate(v) for k, v in iv_ts.items()}
    iv_off = {k: np.concatenate(v) for k, v in iv_off.items()}
    iv_excl = {'frames': 0, 'samples_label_stride3': 0, 't_since_takeover_s': [], 'by_log': {}}

    # 表(データセットの全シーン・全フレーム)
    rows = {k: [] for k in ('scene', 'frame', 'v2h', 'v2m', 'bf', 'w_v2m', 'w_bf', 'remap', 'reason')}
    stat = {'frames': {}, 'scenes': {'base_train': 0, 'val': 0, 'ai_log': 0}}
    ivt = {}
    for sc in sorted(set(splits['train']) | set(splits['val'])):
        with np.load(DS / 'aligned' / (sc + '.npz'), allow_pickle=False) as z:
            n = len(z['timestamps_us'])
            src = z['source'].astype(np.int64) if 'source' in z.files else np.full(n, -1)
            li = z['label_index'].astype(np.int64) if 'label_index' in z.files else np.full(n, -1)
        keep_h = np.zeros(n, bool)
        keep_m = np.zeros(n, bool)
        keep_b = np.zeros(n, bool)
        w_m = np.ones(n, np.float32)
        w_b = np.ones(n, np.float32)
        remap = np.full(n, -1, np.int16)
        reason = np.full(n, REASON['keep'], np.int8)
        lg = sc.rsplit('-', 1)[0]
        if sc in splits['val']:
            stat['scenes']['val'] += 1
            keep_h[:] = keep_m[:] = keep_b[:] = True      # 検証は drop=False(H1 をそのまま)
            reason[:] = REASON['val']
        elif sc in base_train:
            stat['scenes']['base_train'] += 1
            keep_h[:] = keep_m[:] = keep_b[:] = True
        else:
            if lg not in po:
                fail('学習のシーン %s が clean_driving でも AI の記録 46 本でもありません' % sc)
            stat['scenes']['ai_log'] += 1
            d, lm = po[lg], lapmask[lg]
            has = li >= 0
            k = np.clip(li, 0, None)
            if has.any() and k.max() >= len(d['t_s']):
                fail('%s: label_index がラベルの行の外です' % sc)
            # aligned の時刻とラベルの時刻が同じか(行の対応の確かめ)
            with np.load(DS / 'aligned' / (sc + '.npz'), allow_pickle=False) as z:
                ts = z['timestamps_us'].astype(np.int64)
            if has.any() and not np.array_equal(ts[has], d['t_us'][k[has]]):
                fail('%s: aligned の時刻と label_index の行の時刻が違います' % sc)
            new = has & d['keep_new'][k] & (src == 0)
            te = has & lm['teach'][k] & (src == 1)
            ha = has & lm['half'][k] & (src == 1)
            r1 = has & lm['r1win'][k] & (src == 1)
            if lg in iv_ts:
                hit = np.isin(ts, iv_ts[lg])
                ex = hit & (new | te | ha)
                if ex.any():
                    iv_excl['frames'] += int(ex.sum())
                    iv_excl['samples_label_stride3'] += int(np.sum(ex & has & (d['idx_in'][k] % 3 == 0)))
                    iv_excl['by_log'][lg] = iv_excl['by_log'].get(lg, 0) + int(ex.sum())
                    pos = np.searchsorted(np.sort(iv_ts[lg]), ts[ex])
                    srt = np.argsort(iv_ts[lg])
                    iv_excl['t_since_takeover_s'] += [round(float(x), 3) for x in iv_off[lg][srt][pos]]
                    if (ex & (te | ha)).any():
                        fail('%s: 手本・速めの周のフレームが介入の検証セットにあります' % sc)
                new &= ~hit
                te &= ~hit
                ha &= ~hit
                r1 &= ~hit
            keep_h = new
            keep_m = new | te | ha
            keep_b = te.copy()
            w_m[ha] = W_HALF
            w_b[r1] = W_R1
            remap[te | ha] = REMAP_ID
            reason[:] = np.where(~has, REASON['no_label'],
                                 np.where(src == 1, REASON['ai_log_ai'], REASON['ai_log_human']))   # 外したときの理由
            for key, mm in (('new', new), ('teach', te), ('half', ha), ('r1win', r1)):
                stat['frames'][key] = stat['frames'].get(key, 0) + int(mm.sum())
            ivt.setdefault(lg, []).append((ts[keep_m | keep_b], sc))
        rows['scene'] += [sc] * n
        rows['frame'].append(np.arange(n, dtype=np.int32))
        rows['v2h'].append(keep_h)
        rows['v2m'].append(keep_m)
        rows['bf'].append(keep_b)
        rows['w_v2m'].append(w_m)
        rows['w_bf'].append(w_b)
        rows['remap'].append(remap)
        rows['reason'].append(reason)
    cat = {k: (np.concatenate(v) if k != 'scene' else v) for k, v in rows.items()}

    # 介入の検証セットのフレームとの重なり(時刻が同じフレーム)。上で外したので 0 のはず(守り)
    overlap = 0
    for s in iv_card['source_scenes']:
        lg = s.rsplit('-', 1)[0]
        if lg not in ivt:
            continue
        with np.load(iv_root / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
            it = z['timestamps_us'].astype(np.int64)
        for kt, _ in ivt[lg]:
            overlap += int(np.isin(kt, it).sum())
    if overlap:
        fail('介入の検証セットのフレームと、学習に残すフレームが %d 重なります' % overlap)

    # 書く
    OUTD.mkdir(parents=True, exist_ok=True)
    scenes, frames = cat['scene'], cat['frame']
    note = np.array('make_v2_tables.py 2026-10-25 / %s' % NAME)
    val_set = set(splits['val'])
    is_val = np.array([s in val_set for s in scenes])

    def reason_of(keep):
        return np.where(keep, np.where(is_val, REASON['val'], REASON['keep']), cat['reason']).astype(np.int8)
    files = {
        'drop_v2h.npz': ('drop', ~cat['v2h'], {'reason': reason_of(cat['v2h'])}),
        'drop_v2m.npz': ('drop', ~cat['v2m'], {'reason': reason_of(cat['v2m'])}),
        'weights_v2m.npz': ('weight', cat['w_v2m'].astype(np.float32), {'note': note}),
        'remap_v2m.npz': ('mode', cat['remap'].astype(np.int16), {'note': note}),
        'drop_bfast.npz': ('drop', ~cat['bf'], {'reason': reason_of(cat['bf'])}),
        'weights_bfast.npz': ('weight', cat['w_bf'].astype(np.float32), {'note': note}),
    }
    # 検証のシーンは drop=False・重み 1・付け替え −1(H1 を変えない)
    for fn, (key, val, extra) in files.items():
        v = val[is_val]
        ok = (~v.astype(bool)).all() if key == 'drop' else ((v == 1.0).all() if key == 'weight' else (v == -1).all())
        if not ok:
            fail('%s: 検証のシーンの値が既定ではありません' % fn)
    out_files = {}
    for fn, (key, val, extra) in files.items():
        if key == 'drop':
            val = val.astype(bool)
        save_table(OUTD / fn, scenes, frames, key, val, extra)
        out_files[fn] = {'sha256': sha(OUTD / fn), 'rows': int(len(frames)), 'key': key}
    tables_card = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'script': 'experiments/20261025_train_v2/make_v2_tables.py',
        'script_sha256': sha(Path(__file__)),
        'dataset': NAME,
        'dataset_card_sha256': sha(DS / 'dataset_card.json'),
        'splits_sha256': sha(DS / 'splits.json'),
        'protocol_v2_sha256': sha(PROTOCOL),
        'rules': {
            'new_human_keep': '棚卸しの (a)・near_collision でない・ジャンル 4・5・clean_driving に無い・介入の窓 [t, t+5) の外 かつ '
                              '(前後 1 秒の前進 eRPM の中央値 >= 3000 か 交代の直後 3 秒)。人側(source 0)のフレームだけ',
            'teacher_laps': 'protocol bfast_recipe.teacher_laps(10 周)。周 [a, b) の AI 側(source 1)のフレーム',
            'half_laps': 'protocol tables.v2m_half_laps(4 周)。重み 0.5',
            'r1_window': '手本の周のフレームで R1 の起点の [ta − 1, ta + 1) → B-FAST の重み 0.5',
            'base': 'clean_driving_20261011 の学習 41 シーンは全フレーム drop=False(B と同じ)',
            'val': '検証 11 シーン(H1)は drop=False・重み 1・付け替え −1',
            'not_in_tables': '衝突の前後(--impact-g 2.5)・future_valid・vehicle_valid・ジャンル 0・間引き 3 は train.py が掛ける',
            'reason_codes': {str(v): k for k, v in REASON.items()},
        },
        'files': out_files,
        'scenes': stat['scenes'],
        'frames_kept_before_train_filters': {
            'base_train_all_frames': int(sum(1 for s in scenes if s in base_train)),
            'new_human': stat['frames'].get('new', 0), 'teacher': stat['frames'].get('teach', 0),
            'half': stat['frames'].get('half', 0), 'teacher_r1_window': stat['frames'].get('r1win', 0)},
        'inventory_counts_label_rows_stride3': inv_counts,
        'laps': lap_info,
        'lap_rule_check': {
            'teacher_rule_equals_protocol': True, 'half_rule_equals_protocol': True,
            'b_clean_9_0_to_9_5_in_teacher_session': teach_half_24,
            'b_clean_sub9_on_1023': sub9_23,
            'b_clean_sub9_5_other_sessions_in_ai_logs': other_fast},
        'interv_overlap_frames': overlap,
        'interv_excluded_beyond_window': dict(iv_excl, note='plan_checks の窓 [t_s, t_s+5) の外にあった介入の検証セットのフレーム。'
                                                            '学習に残さない(protocol の止める条件「介入の窓と学習のフレームが重なる」を守るため)'),
        'sealed_in_source_scenes': 0,
        'seconds': round((datetime.now() - t0).total_seconds(), 1),
    }
    tmp = OUTD / 'tables_card.json.tmp'
    tmp.write_text(json.dumps(tables_card, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, OUTD / 'tables_card.json')
    print(json.dumps({k: tables_card[k] for k in ('files', 'scenes', 'frames_kept_before_train_filters',
                                                   'inventory_counts_label_rows_stride3', 'lap_rule_check',
                                                   'interv_overlap_frames', 'interv_excluded_beyond_window', 'seconds')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
