"""走行タブの「ベンチマークの候補」と「基準の設定」の API(server.py が include_router する。server.py は import しない)

    GET /api/bench/candidates   data/bench_candidates.json を読み、候補ごとに走れるか(drive.model_info)を付けて返す

- 読むだけ。ファイル・走行の設定(data/drive_settings.json)・推奨のモデルは書かない。走行も始めない。
- ファイルが無いときは exists False(画面はパネルを出さない)。基準の値は 10/18 plan.md 3.4 の値(fallback_baseline)を返す。
- 壊れているときは exists True・ok False と理由(画面はパネルに理由を出す)。基準の値は 3.4 の値。
- 基準の値の params は drive.DEFAULT_PARAMS に、ファイルの値を重ねたもの(「ほかは DEFAULT_PARAMS」)。
  知らない項目は捨て、範囲の外は drive.PARAM_LIMITS に丸めて warnings に書く。
- 区間の速度上限(zone_caps)は、ファイルが 1 でも画面は ON にしない(ON はスイッチで、確かめてから)。
"""

import json
import math
import time

from fastapi import APIRouter

from .. import config, drive

BENCH_PATH = config.DATA_DIR / 'bench_candidates.json'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'           # モデル B(data/drive_recommended.json と同じ)
DESCRIPTION_TEMPLATE = ('bench=real_bench_20261023 day={date} block={block} cand={id} model={run} '
                        'battery=#__ V_before=__ V_after=__ check=__ memo=')
RISKS = ('low', 'mid', 'high')

router = APIRouter()


def fallback_baseline():
    """10/18 plan.md 3.4 の基準の値: モデル B・ステア 1.0・速度 1.0・先読み 3・ジャンル 5・ToF ガード 1(記録だけ)・
    重ね合わせ 0・speed_cap 0・区間の速度上限 OFF。ほかは drive.DEFAULT_PARAMS"""
    params = dict(drive.DEFAULT_PARAMS)
    params.update(steer_gain=1.0, throttle_gain=1.0, lookahead=3, tof_guard=1, ensemble_j=0, speed_cap=0.0, zone_caps=0)
    return {'run': B_RUN, 'mode_id': 5, 'params': params, 'source': '10/18 plan.md 3.4'}


def _baseline(raw, warnings):
    base = fallback_baseline()
    if raw is None:
        return base
    if not isinstance(raw, dict):
        warnings.append('baseline が辞書ではありません(10/18 plan.md 3.4 の値を使います)')
        return base
    base['source'] = 'data/bench_candidates.json'
    if raw.get('mode_id') is not None:
        try:
            base['mode_id'] = int(raw['mode_id'])
        except (TypeError, ValueError):
            warnings.append('baseline.mode_id が整数ではありません(5 を使います)')
    params = raw.get('params') or {}
    if not isinstance(params, dict):
        warnings.append('baseline.params が辞書ではありません(10/18 plan.md 3.4 の値を使います)')
        params = {}
    base['params'] = dict(drive.DEFAULT_PARAMS)              # ファイルに書いていない項目は DEFAULT_PARAMS
    for k, v in params.items():
        if k not in drive.PARAM_LIMITS:
            warnings.append('baseline.params の %s は走行の設定にありません(使いません)' % k)
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = float('nan')
        if not math.isfinite(x):
            warnings.append('baseline.params の %s が数値ではありません(既定 %s を使います)' % (k, drive.DEFAULT_PARAMS[k]))
            continue
        lo, hi = drive.PARAM_LIMITS[k]
        if not lo <= x <= hi:
            warnings.append('baseline.params の %s = %s は範囲 %s〜%s の外です(丸めます)' % (k, v, lo, hi))
            x = min(max(x, lo), hi)
        base['params'][k] = int(round(x)) if isinstance(drive.DEFAULT_PARAMS[k], int) else x
    if base['params'].get('zone_caps'):
        warnings.append('baseline.params の zone_caps が 1 です。画面の「基準の設定」は区間の速度上限を ON にしません'
                        '(ON にするのは、G1 に合格し、ユーザーが了承したあと。スイッチで入れてください)')
    return base


def _check_run(run, mode_id, cache):
    """走れるか: (drivable, why_not, policy_title)。drive.start と同じ確かめ(エンジン・drivable・ジャンル)"""
    if not isinstance(run, str) or not run.strip():
        return False, '学習結果(run)が書かれていません', None
    run = run.strip()
    if run not in cache:
        cache[run] = drive.model_info(run)
    info = cache[run]
    if info is None:
        return False, '走行に使える学習結果がありません(TensorRT エンジンが必要): %s' % run, None
    if not info.get('drivable', True):
        return False, info.get('why_not') or '走行できないモデルです', info.get('policy_title')
    if mode_id not in (info.get('trained_modes') or []):
        return False, 'このモデルはジャンル %d を学習していません' % mode_id, info.get('policy_title')
    return True, None, info.get('policy_title')


def _description(template, date, block, cid, run):
    out = template
    for k, v in (('{date}', date), ('{block}', block), ('{id}', cid), ('{run}', run)):
        out = out.replace(k, str(v))
    return out


def load_candidates(path=None):
    """data/bench_candidates.json → 画面の表示用(壊れていても例外を出さない)"""
    path = path or BENCH_PATH
    out = {'exists': False, 'ok': False, 'error': None, 'warnings': [], 'path': str(path),
           'today': time.strftime('%Y-%m-%d'), 'baseline': fallback_baseline(), 'candidates': [], 'days': []}
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        return out
    except OSError as e:
        out.update(exists=True, error='読めません: %s' % e)
        return out
    out['exists'] = True
    try:
        data = json.loads(text)
    except ValueError as e:
        out['error'] = 'JSON として読めません: %s' % e
        return out
    if not isinstance(data, dict):
        out['error'] = '中身が辞書ではありません'
        return out
    warnings = out['warnings']
    if data.get('version') != 1:
        warnings.append('version が 1 ではありません(%r)。版 1 として読みます' % data.get('version'))
    for k in ('version', 'bench', 'protocol_sha256', 'created_at'):
        out[k] = data.get(k)
    base = _baseline(data.get('baseline'), warnings)
    tpl = data.get('description_template')
    out['description_template'] = tpl if isinstance(tpl, str) and tpl.strip() else DESCRIPTION_TEMPLATE

    cache, cands, by_id = {}, [], {}
    for c in data.get('candidates') or []:
        if not isinstance(c, dict) or not str(c.get('id') or '').strip():
            warnings.append('id の無い候補を飛ばしました')
            continue
        cid = str(c['id']).strip()
        if cid in by_id:
            warnings.append('候補の id %s が重なっています(最初のものを使います)' % cid)
            continue
        run = str(c.get('run') or '').strip()
        ok, why, title = _check_run(run, base['mode_id'], cache)
        risk = str(c.get('risk') or '').strip()
        item = {'id': cid, 'label': str(c.get('label') or cid), 'run': run, 'hypothesis': str(c.get('hypothesis') or ''),
                'risk': risk if risk in RISKS else (risk or None), 'drivable': ok, 'why_not': why, 'policy_title': title}
        cands.append(item)
        by_id[cid] = item
    out['candidates'] = cands
    b = by_id.get('B')
    if b and b['run']:
        base['run'] = b['run']                  # 基準の設定のモデル = 候補 B
    out['baseline'] = base

    days = []
    for d in data.get('days') or []:
        if not isinstance(d, dict):
            continue
        date = str(d.get('date') or '')
        blocks = []
        for i, blk in enumerate(d.get('blocks') or []):
            if not isinstance(blk, dict):
                continue
            cid = str(blk.get('id') or '').strip()
            try:
                no = int(blk.get('block'))
            except (TypeError, ValueError):
                no = i + 1
            c = by_id.get(cid)
            item = {'block': no, 'id': cid, 'run': c['run'] if c else '', 'label': c['label'] if c else cid,
                    'risk': c['risk'] if c else None, 'drivable': bool(c and c['drivable']),
                    'why_not': (c['why_not'] if c else '候補の表(candidates)に無い id です: %s' % cid)}
            item['description'] = _description(out['description_template'], date, no, cid, item['run']) if c else ''
            blocks.append(item)
        days.append({'date': date, 'note': str(d.get('note') or ''), 'blocks': blocks})
    out['days'] = days
    out['ok'] = True
    return out


@router.get('/api/bench/candidates')
def api_bench_candidates():
    """ベンチマークの候補(data/bench_candidates.json)と基準の値。走れないモデルには理由(why_not)"""
    return load_candidates()
