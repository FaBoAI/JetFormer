"""走行タブの「ベンチマークの候補」と「基準の設定」の API(server.py が include_router する。server.py は import しない)

    GET /api/bench/candidates   data/bench_candidates.json を読み、候補ごとに走れるか(drive.model_info)を付けて返す
    POST /api/bench/after       走り終わりのメモ(電池・電圧・止めた理由)を、その記録のシーンの説明に書く(2026-10-24)

- 読むだけ。ファイル・走行の設定(data/drive_settings.json)・推奨のモデルは書かない。走行も始めない。
- ファイルが無いときは exists False(画面はパネルを出さない)。基準の値は 10/18 plan.md 3.4 の値(fallback_baseline)を返す。
- 壊れているときは exists True・ok False と理由(画面はパネルに理由を出す)。基準の値は 3.4 の値。
- 基準の値の params は drive.DEFAULT_PARAMS に、ファイルの値を重ねたもの(「ほかは DEFAULT_PARAMS」)。
  知らない項目は捨て、範囲の外は drive.PARAM_LIMITS に丸めて warnings に書く。
- 区間の速度上限(zone_caps)は、ファイルが 1 でも画面は ON にしない(ON はスイッチで、確かめてから)。
  速さの天井(speed_ceiling)・速さのゲインの自動合わせ(speed_auto)も同じ(2026-10-25)。発進の前進(launch_boost)も同じ
  (2026-10-26)。ブロックの上書きでは捨てる(launch_us などの値は上書きできる。ON にしなければ効かない)。
  基準(baseline)に 1 と書けば、計画の照らし合わせ(plan_check)では「計画どおり ON」とみなす(画面は ON にしない)。
- 電池の電圧(2026-10-25): 走行の開始・終わりに基板の電圧を説明欄に入れる決まり(vbat_before_description・
  vbat_after_description。server.py が呼ぶ)。V_before・V_after が空(__・無し)なら基板の値を入れて V_src=board、
  人の数があればそのまま残して Vb_before・Vb_after に基板の値。走行中の 1 秒ごとの中央値の最小は Vb_min。
  読み方: V_x が基板の値なのは「V_src=board かつ Vb_x が無い」とき(Vb_x があれば V_x は人の値)。
  シーンの説明の書き換え(走り終わりのメモ・電圧)は DESC_LOCK で 1 つずつ。
"""

import json
import math
import os
import re
import threading
import time

from fastapi import APIRouter, Body, HTTPException

from .. import config, dataset, drive

BENCH_PATH = config.DATA_DIR / 'bench_candidates.json'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'           # モデル B(data/drive_recommended.json と同じ)
DESCRIPTION_TEMPLATE = ('bench=real_bench_20261023 day={date} block={block} cand={id} model={run} '
                        'battery=#__ V_before=__ V_after=__ check=__ memo=')
RISKS = ('low', 'mid', 'high')
# AI 走行の番号(2026-10-23 ユーザーの依頼「AI走行にNoを付与、1から開始」)。モーターに出力する走行(dry run でない)を始めるたびに、
# その日の 1 から数える。説明欄の先頭に ai_no=N を足す(道具の parse_description が読む key=value の形)
AI_NO_PATH = config.DATA_DIR / 'ai_run_no.json'
AI_NO_LOCK = threading.Lock()

router = APIRouter()


def next_ai_no(path=None, today=None):
    """その日の次の AI 走行の番号(書かない)。ファイルが無い・壊れている・日付が違うなら 1"""
    today = today or time.strftime('%Y-%m-%d')
    try:
        with open(path or AI_NO_PATH, encoding='utf-8') as f:
            doc = json.load(f)
        if doc.get('date') == today:
            return int(doc.get('last', 0)) + 1
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return 1


def commit_ai_no(n, path=None, today=None):
    """番号 n を使ったことを書く(走行が始まったあとだけ呼ぶ)"""
    path = path or AI_NO_PATH
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'date': today or time.strftime('%Y-%m-%d'), 'last': int(n)}, f)
    os.replace(tmp, path)


def with_ai_no(description, n):
    """説明欄の先頭に ai_no=N を足す"""
    return ('ai_no=%d %s' % (int(n), description or '')).strip()


# ---------------------------------------------------------------- 説明欄の欄(key=value。memo= より後ろは全部 memo)
def _split_memo(desc):
    m = re.search(r'(?:^|\s)memo=(.*)$', desc or '')
    return ((desc or '')[:m.start()] if m else (desc or '')).strip(), (m.group(1).strip() if m else None)


def get_field(desc, key):
    head, memo = _split_memo(desc)
    if key == 'memo':
        return memo
    m = re.search(r'(?:^|\s)%s=(\S*)' % re.escape(key), head)
    return m.group(1) if m else None


def set_field(desc, key, value):
    """欄 key を value にする(無ければ memo= の前に足す)。memo は add_memo で"""
    head, memo = _split_memo(desc)
    tok = '%s=%s' % (key, value)
    pat = re.compile(r'(^|\s)%s=\S*' % re.escape(key))
    head = pat.sub(lambda m: m.group(1) + tok, head, count=1) if pat.search(head) else (head + ' ' + tok).strip()
    return head if memo is None else ('%s memo=%s' % (head, memo)).strip()


def add_memo(desc, text):
    """memo の先頭に text を足す(同じ文がもうあれば足さない)"""
    head, memo = _split_memo(desc)
    if not text or (memo and text in memo):
        return desc
    return ('%s memo=%s' % (head, text if not memo else '%s。%s' % (text, memo))).strip()


# ---------------------------------------------------------------- 計画と違う走行の印(2026-10-24、10/23 の課題)
# 10/23 は公式 8 本のあと、ブロックを押し直さずにモデル・設定を替えて 11 本走り、説明欄が block=1 cand=B のまま残って
# B の成績に混ざった(assign_20261023.json で手で直した)。走行開始のときに、説明欄のブロックの計画(候補のモデル)と
# 基準の設定を比べ、違えば cand=EXP・official=0 にして memo に違いを書く(順位に入れない)。ステアは「この日のステア」で
# 変えてよいので違いにしないが、基準と違えば steer= の欄に残す。
PLAN_KEYS = ('throttle_gain', 'lookahead', 'tof_guard', 'ensemble_j', 'speed_cap', 'zone_caps', 'tcp_mode',
             'speed_ceiling', 'speed_auto')
# 画面の「基準の設定」・ブロックでは ON にしないスイッチ(ON はスイッチで、確かめてから。1 走行 = 1 設定)
# 発進の前進(launch_boost、2026-10-26)も同じ: 基準の設定・ブロックの上書きでは ON にしない(スイッチで、確かめてから)
SWITCH_KEYS = ('zone_caps', 'speed_ceiling', 'speed_auto', 'launch_boost')
SWITCH_NAMES = {'zone_caps': '区間の速度上限', 'speed_ceiling': '速さの天井', 'speed_auto': '速さのゲインの自動合わせ',
                'launch_boost': '発進の前進'}
EXP_ID = 'EXP'


def plan_check(description, run, mode_id, params, data=None):
    """戻り値 (説明欄, 違いの一覧)。説明欄に bench= と cand= が無い(ベンチマークでない)なら、そのまま返す"""
    desc = description or ''
    cand, block, day = get_field(desc, 'cand'), get_field(desc, 'block'), get_field(desc, 'day')
    if not get_field(desc, 'bench') or not cand or cand == EXP_ID:
        return desc, []
    data = data if data is not None else load_candidates()
    if not data.get('ok'):
        return desc, []
    base = data.get('baseline') or fallback_baseline()
    planned = None
    for d in data.get('days') or []:
        if d.get('date') == day:
            for b in d.get('blocks') or []:
                if str(b.get('block')) == str(block):
                    planned = b
    exp_run = (planned or {}).get('run') or next((c['run'] for c in data.get('candidates') or [] if c.get('id') == cand), None)
    diffs = []
    if planned is not None and planned.get('id') != cand:
        diffs.append('ブロック %s の計画は %s(説明欄は %s)' % (block, planned.get('id'), cand))
    if exp_run and run and run != exp_run:
        diffs.append('モデル %s(計画 %s)' % (run, exp_run))
    if mode_id is not None and base.get('mode_id') is not None and int(mode_id) != int(base['mode_id']):
        diffs.append('ジャンル %s(基準 %s)' % (mode_id, base['mode_id']))
    bp, p = base.get('params') or {}, params or {}
    for k in PLAN_KEYS:
        if k in p and k in bp:
            try:
                if abs(float(p[k]) - float(bp[k])) > 1e-6:
                    diffs.append('%s %s(基準 %s)' % (k, p[k], bp[k]))
            except (TypeError, ValueError):
                continue
    try:
        if 'steer_gain' in p and abs(float(p['steer_gain']) - float(bp.get('steer_gain', 1.0))) > 1e-6:
            desc = set_field(desc, 'steer', '%g' % float(p['steer_gain']))
    except (TypeError, ValueError):
        pass
    if diffs:
        desc = set_field(desc, 'cand', EXP_ID)
        if get_field(desc, 'official') is not None:
            desc = set_field(desc, 'official', '0')
        desc = add_memo(desc, '計画と違うので試し(元 cand=%s): %s' % (cand, '・'.join(diffs)))
    return desc, diffs


# ---------------------------------------------------------------- 走り終わりのメモ(2026-10-24、10/23 の課題)
STOP_REASONS = ('240秒走った', '安全で中止', '回れない', 'B止め', '電池', 'その他')
DESC_LOCK = threading.Lock()       # シーンの説明の書き換え(走り終わりのメモ・基板の電圧)を 1 つずつ


def blank(v):
    """説明欄の値が空か(無い・空・'__'・'#__')"""
    return v is None or not str(v).strip() or set(str(v).strip()) <= set('_#')


def _board_key(key):
    return {'V_before': 'Vb_before', 'V_after': 'Vb_after'}[key]


def after_description(desc, battery=None, v_before=None, v_after=None, reason=None, memo=None):
    """説明欄に走り終わりの欄を入れる。空の値は変えない。
    基板の値で埋めた V_before・V_after(V_src=board で Vb_x が無い)を人の違う値で替えるときは、基板の値を Vb_x に残す"""
    out = desc or ''
    if battery not in (None, ''):
        out = set_field(out, 'battery', '#' + str(battery).lstrip('#'))
    for k, v in (('V_before', v_before), ('V_after', v_after)):
        if v not in (None, ''):
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise ValueError('%s が数値ではありません: %r' % (k, v))
            cur, bk = get_field(out, k), _board_key(k)
            if get_field(out, 'V_src') == 'board' and not blank(cur) and blank(get_field(out, bk)):
                try:
                    if abs(float(cur) - x) > 1e-6:
                        out = set_field(out, bk, cur)          # 基板の値を残す(V_x は人の値になる)
                except ValueError:
                    pass
            out = set_field(out, k, '%g' % x)
    text = '・'.join(x for x in (reason, (memo or '').strip()) if x)
    return add_memo(out, text) if text else out


# ---------------------------------------------------------------- 基板の電圧を説明欄へ(2026-10-25、spec 4.2 節)
def _fill_voltage(desc, key, v):
    """V_x が空なら基板の値と V_src=board、人の数があれば Vb_x に基板の値(Vb_x が空のときだけ)。
    V_x がもう同じ基板の値(V_src=board で ±0.005 V)なら何もしない(同じ値を 2 回書いても変わらない)"""
    txt = '%.2f' % float(v)
    cur = get_field(desc, key)
    if blank(cur):
        out = set_field(desc, key, txt)
        return out if get_field(out, 'V_src') == 'board' else set_field(out, 'V_src', 'board')
    bk = _board_key(key)
    if not blank(get_field(desc, bk)):
        return desc
    try:
        if get_field(desc, 'V_src') == 'board' and abs(float(cur) - float(v)) < 0.005:
            return desc
    except ValueError:
        pass
    return set_field(desc, bk, txt)


def vbat_before_description(desc, v):
    """走行開始: 止まっているときの基板の電圧 v [V](None なら何もしない)を V_before(または Vb_before)に"""
    if v is None or not math.isfinite(float(v)):
        return desc or ''
    return _fill_voltage(desc or '', 'V_before', v)


def vbat_after_description(desc, v_after=None, v_min=None):
    """走行の終わり: 止まってからの基板の電圧を V_after(または Vb_after)、走行中の 1 秒ごとの中央値の最小を Vb_min に。
    空の欄だけ埋める(人の数・前に書いた値は変えない)"""
    out = desc or ''
    if v_after is not None and math.isfinite(float(v_after)):
        out = _fill_voltage(out, 'V_after', v_after)
    if v_min is not None and math.isfinite(float(v_min)) and blank(get_field(out, 'Vb_min')):
        out = set_field(out, 'Vb_min', '%.2f' % float(v_min))
    return out


def log_scenes(log):
    """記録 log のシーンの名前(scene.json があるもの)"""
    root = config.RAW_DIR / log
    return sorted(p.parent.name for p in root.glob('*/scene.json')) if root.is_dir() else []


def write_vbat_after(log, v_after=None, v_min=None):
    """記録 log のすべてのシーンの説明に、走り終わりの基板の電圧を書く(server.py が走行の終わりに呼ぶ)。書いたシーンの数"""
    if not re.fullmatch(r'\d{8}-\d{6}', str(log or '')):
        raise ValueError('記録の名前が正しくありません: %r' % log)
    if v_after is None and v_min is None:
        return 0
    root = config.RAW_DIR / log
    try:
        base = json.loads((root / 'log.json').read_text(encoding='utf-8')).get('description', '')
    except (OSError, ValueError):
        base = ''
    n = 0
    with DESC_LOCK:
        for name in log_scenes(log):
            try:
                cur = json.loads((root / name / 'scene.json').read_text(encoding='utf-8')).get('description') or base
            except (OSError, ValueError):
                continue
            out = vbat_after_description(cur, v_after, v_min)
            if out != cur:
                dataset.update_scene(name, description=out)
                n += 1
    return n


def fallback_baseline():
    """10/18 plan.md 3.4 の基準の値: モデル B・ステア 1.0・速度 1.0・先読み 3・ジャンル 5・ToF ガード 1(記録だけ)・
    重ね合わせ 0・speed_cap 0・区間の速度上限 OFF。ほかは drive.DEFAULT_PARAMS"""
    params = dict(drive.DEFAULT_PARAMS)
    params.update(steer_gain=1.0, throttle_gain=1.0, lookahead=3, tof_guard=1, ensemble_j=0, speed_cap=0.0, zone_caps=0,
                  start_kick=1)                    # 発進の補助(2026-10-23 ユーザーの依頼)
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
    if base['params'].get('launch_boost'):
        warnings.append('baseline.params の launch_boost が 1 です。画面の「基準の設定」は発進の前進を ON にしません'
                        '(スイッチで入れてください)')
    for k in ('speed_ceiling', 'speed_auto'):
        if base['params'].get(k):
            warnings.append('baseline.params の %s が 1 です。画面の「基準の設定」は%sを ON にしません'
                            '(スイッチで入れてください。計画の照らし合わせでは ON の計画とみなします)' % (k, SWITCH_NAMES[k]))
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


def _block_overrides(blk, warnings, where):
    """ブロックの上書き → ({'params': {...}, 'mode_id': n}, tune の文字)。知らない項目は捨て、範囲の外は丸めて warnings に"""
    ov = {}
    raw = blk.get('params')
    if isinstance(raw, dict):
        params = {}
        for k, v in raw.items():
            if k not in drive.PARAM_LIMITS or k in SWITCH_KEYS:
                warnings.append('%s: 上書きできない項目 %s を捨てました' % (where, k)
                                + ('(%sはスイッチで入れてください)' % SWITCH_NAMES[k] if k in SWITCH_KEYS else ''))
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                warnings.append('%s: %s が数値ではありません' % (where, k))
                continue
            lo, hi = drive.PARAM_LIMITS[k]
            if not math.isfinite(x) or x < lo or x > hi:
                warnings.append('%s: %s=%r を範囲 %s〜%s に丸めました' % (where, k, v, lo, hi))
                x = min(max(x, lo), hi) if math.isfinite(x) else lo
            params[k] = x
        if params:
            ov['params'] = params
    if blk.get('mode_id') is not None:
        try:
            ov['mode_id'] = int(blk['mode_id'])
        except (TypeError, ValueError):
            warnings.append('%s: mode_id が整数ではありません' % where)
    tune = str(blk.get('tune') or '').strip()
    return ov, tune


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
            # ブロックごとの上書き(2026-10-24 速さのチューニング): params(速度ゲインなど)・mode_id(ジャンル)・tune(表示名)
            ov, tune = _block_overrides(blk, warnings, '%s のブロック %s' % (date, no))
            if ov:
                item.update(ov)
                if 'mode_id' in ov and c:
                    ok, why, _ = _check_run(c['run'], ov['mode_id'], cache)
                    item['drivable'], item['why_not'] = bool(ok), why
            if tune:
                item['tune'] = tune
            if c and (ov or tune):
                tag = ' '.join((['tune=%s' % tune.replace(' ', '_')] if tune else [])
                               + ['%s=%s' % (k, v) for k, v in sorted((ov.get('params') or {}).items())]
                               + (['mode=%d' % ov['mode_id']] if 'mode_id' in ov else []))
                desc = item['description']
                i = desc.find('memo=')
                item['description'] = (desc[:i] + tag + ' ' + desc[i:]) if i >= 0 else (desc + ' ' + tag).strip()
            blocks.append(item)
        days.append({'date': date, 'note': str(d.get('note') or ''), 'blocks': blocks})
    out['days'] = days
    out['ok'] = True
    return out


@router.get('/api/bench/candidates')
def api_bench_candidates():
    """ベンチマークの候補(data/bench_candidates.json)と基準の値。走れないモデルには理由(why_not)"""
    return load_candidates()


@router.post('/api/bench/after')
def api_bench_after(body: dict = Body(...)):
    """走り終わりのメモを、記録 log のすべてのシーンの説明に書く(夜の集計は section_scorecard.effective_description で読む)"""
    log = str(body.get('log') or '').strip()
    if not re.fullmatch(r'\d{8}-\d{6}', log):
        raise HTTPException(400, '記録の名前が正しくありません: %r' % log)
    reason = body.get('reason') or None
    if reason is not None and reason not in STOP_REASONS:
        raise HTTPException(400, '止めた理由は %s のどれか' % '・'.join(STOP_REASONS))
    root = config.RAW_DIR / log
    scenes = log_scenes(log)
    if not scenes:
        raise HTTPException(404, '記録 %s のシーンがありません' % log)
    try:
        base = json.loads((root / 'log.json').read_text(encoding='utf-8')).get('description', '')
    except (OSError, ValueError):
        base = ''
    out = None
    with DESC_LOCK:                        # 走行の終わりの基板の電圧の書き込み(server.py)と重ならないように
        for name in scenes:
            try:
                cur = json.loads((root / name / 'scene.json').read_text(encoding='utf-8')).get('description') or base
                out = after_description(cur, body.get('battery'), body.get('V_before'), body.get('V_after'), reason,
                                        body.get('memo'))
            except ValueError as e:
                raise HTTPException(400, str(e))
            dataset.update_scene(name, description=out)
    return {'ok': True, 'log': log, 'scenes': len(scenes), 'description': out}
