"""シミュレーターの「強化学習モード」(2026-10-14、sim_rl_modes の統合)

強化学習(RL)の方策で教師データの走りを変え、データセットにばらつきを出すモード(最速・絶対壁にぶつからない・離脱復活・速さいろいろ など)を、
JetFormer のシミュレーターのページで選べるようにする。ここはサーバー側(登録簿の読み込み・方策の配信・収集プラン・進み具合)。

登録簿(RL の学習側の発行スクリプトが書く。JetFormer は読んで確かめるだけ):

    data/sim_rl/modes/<key>/mode.json     モードの説明・記録の決まり・タグ(schema jf-sim-rl-mode/v1)
    data/sim_rl/modes/<key>/status.json   合否(accepted 合格 / not_accepted 不合格(参考)/ pending 確認待ち / not_published 出さない)
    data/sim_rl/modes/<key>/policy.json   方策の重み(sha256 が mode.json の policy_sha256 と同じこと)
    data/sim_rl/modes/<key>/eval.json     元の sim-core.js で試験した結果(画面の数字はここから。JetFormer は表示するだけ)
    data/sim_rl/modes/<key>/card.json     画面に出す要約(schema jf-sim-rl-mode-card/v1)
    data/sim_rl/modes/<key>/scenes.json   記録に使うシーンの組(Node で前もって走らせて確かめたもの。schema jf-sim-rl-scene-pool/v1)
    data/sim_rl/plans/<id>.json           収集プラン(モード × 分)と進み具合

ジャンルは全部 8(シミュレーター)のまま。モードはタグ(sim-rl・sim-rl-<key> など)とデータセット名 sim-rl-<key>_<日時> で分ける。
環境変数 JETFORMER_SIM_RL=0 で、この仕組み全体(ページへの差し込みと API)を止められる。
"""

import datetime
import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

from .. import config
from . import paths

KEY_RE = re.compile(r'^[a-z][a-z0-9_]{0,31}$')
PLAN_RE = re.compile(r'^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$')
SCENE_ID_RE = re.compile(r'^s[0-9]{3}(r[0-9]{1,2})?$')
MODE_SCHEMA = 'jf-sim-rl-mode/v1'
STATUS_SCHEMA = 'jf-sim-rl-mode-status/v1'
CARD_SCHEMA = 'jf-sim-rl-mode-card/v1'
POOL_SCHEMA = 'jf-sim-rl-scene-pool/v1'
PLAN_SCHEMA = 'jf-sim-rl-plan/v1'
POLICY_SCHEMAS = ('rl-bestlap-policy/v1', 'jf-sim-rl-mode-policy/v1')
SELECTABLE_STATES = ('accepted', 'not_accepted', 'pending')
STATE_LABEL = {'accepted': '合格', 'not_accepted': '不合格(参考)', 'pending': '確認待ち', 'not_published': '出していません(試験で不合格)',
               'training': '学習中', 'untested': '未試験'}
RL_COURSES = ('default',)                 # RL の観測は Default(v3・33 ゲート)の参照ループ専用
SCENES_PER_MINUTE = 3
MAX_MINUTES = 20
DEFAULT_MINUTES = 5
SCENARIO_ROTATION = ('baseline', 'baseline', 'noise', 'dim')   # modes.json recording_common.scenario_rotation(2 : 1 : 1)
BLOCKED_SCENARIOS = ('low-friction',)                            # μ × 0.4 は RL が学習・試験した物理の範囲の外
MAX_POLICY_BYTES = 16 << 20
MAX_JSON_BYTES = 8 << 20
SPARE_RATIO = 0.25                        # 記録で外れたシーンの替え(プランのシーン数の 25% + 2)

_lock = threading.RLock()
_sha_cache = {}


class RlError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def enabled():
    return os.environ.get('JETFORMER_SIM_RL', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def rl_dir():
    return Path(config.DATA_DIR) / 'sim_rl'


def modes_dir():
    return rl_dir() / 'modes'


def plans_dir():
    return rl_dir() / 'plans'


def course_supported(cid):
    return cid in RL_COURSES


# ---------------------------------------------------------------- 登録簿
def _read_json(p, limit=MAX_JSON_BYTES):
    p = Path(p)
    if p.is_symlink() or not p.is_file():
        return None
    if p.stat().st_size > limit:
        raise RlError('%s が大きすぎます' % p.name)
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise RlError('%s を読めません: %s' % (p.name, e))


def file_sha256(p):
    """sha256(大きさと mtime で覚える)"""
    p = Path(p)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    hit = _sha_cache.get(key)
    if hit is None:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        hit = h.hexdigest()
        if len(_sha_cache) > 64:
            _sha_cache.clear()
        _sha_cache[key] = hit
    return hit


def _mode_dir(key):
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise RlError('モードの名前の形が違います', 404)
    d = modes_dir() / key
    if d.is_symlink() or not d.is_dir():
        raise RlError('モードがありません: %s' % key, 404)
    return d


def load_mode(key):
    """登録簿の 1 モードを読んで確かめる。戻り値は画面・API 用の dict(problems があれば選べない)"""
    d = _mode_dir(key)
    problems, warnings = [], []             # problems があると選べない。warnings は表示だけ
    try:
        mode = _read_json(d / 'mode.json')
    except RlError as e:
        mode, problems = None, [str(e)]
    if not isinstance(mode, dict) or mode.get('schema') != MODE_SCHEMA or mode.get('key') != key:
        return {'key': key, 'name': key, 'selectable': False, 'state': 'broken', 'label': '壊れています',
                'problems': problems or ['mode.json が無いか、形が違います']}
    try:
        status = _read_json(d / 'status.json') or {}
    except RlError as e:
        status, problems = {}, problems + [str(e)]
    state = status.get('state') if status.get('schema') == STATUS_SCHEMA else None
    if state not in STATE_LABEL:
        problems.append('status.json が無いか、形が違います')
        state = 'untested'
    card, pool, ev = None, None, None
    for name in ('card.json', 'scenes.json', 'eval.json'):
        try:
            v = _read_json(d / name)
        except RlError as e:
            problems.append(str(e))
            v = None
        if name == 'card.json':
            card = v if isinstance(v, dict) and v.get('schema') == CARD_SCHEMA else None
        elif name == 'scenes.json':
            pool = v if isinstance(v, dict) and v.get('schema') == POOL_SCHEMA and v.get('mode') == key else None
        else:
            ev = v if isinstance(v, dict) else None
    pol = d / 'policy.json'
    policy = None
    if state in SELECTABLE_STATES:
        if pol.is_symlink() or not pol.is_file():
            problems.append('policy.json がありません')
        elif pol.stat().st_size > MAX_POLICY_BYTES:
            problems.append('policy.json が大きすぎます')
        else:
            sha = file_sha256(pol)
            if sha != mode.get('policy_sha256'):
                problems.append('policy.json の sha256 が mode.json の policy_sha256 と違います(差し替えの途中か、壊れています)')
            else:
                policy = {'sha256': sha, 'bytes': pol.stat().st_size}
        if pool is not None and pool.get('policy_sha256') != mode.get('policy_sha256'):
            warnings.append('記録のシーンの組(scenes.json)が別の方策で作られています。収集プランは作れません(モードは選べます)')
            pool = None
    course = mode.get('course') or 'default'
    selectable = state in SELECTABLE_STATES and policy is not None and not problems and course in RL_COURSES
    pool_n = len(pool.get('scenes') or []) if pool else 0
    rec = mode.get('recording') or {}
    return {
        'key': key, 'name': mode.get('name_ja') or key, 'short': mode.get('short_ja') or '', 'order': mode.get('order', 99),
        'priority': mode.get('priority'), 'course': course, 'state': state,
        'label': ('学習中 %d%%' % round(100 * float(status['progress'])) if state == 'training' and isinstance(status.get('progress'), (int, float))
                  else STATE_LABEL.get(state, state)),
        'failed': status.get('failed') or [], 'pending': status.get('pending') or [], 'updated_at': status.get('updated_at'),
        'selectable': selectable, 'problems': problems, 'warnings': warnings, 'policy': policy, 'policy_sha256': mode.get('policy_sha256'),
        'obs_wrapper': mode.get('obs_wrapper'), 'runtime': mode.get('runtime') or {}, 'tags': list(mode.get('tags') or []),
        'scenario_blocklist': sorted(set(BLOCKED_SCENARIOS) | set(mode.get('scenario_blocklist') or [])),
        'scenario_note': mode.get('scenario_note_ja'), 'policy_source': mode.get('policy_source'),
        'card': card, 'planned_time_s': (card or {}).get('lap_from_stop_s'),
        'pool': {'scenes': pool_n, 'acceptance_rate': pool.get('acceptance_rate'), 'rules': pool.get('rules'),
                 'max_minutes': min(MAX_MINUTES, pool_n // SCENES_PER_MINUTE)} if pool else None,
        'plan_ok': selectable and pool_n >= SCENES_PER_MINUTE,
        'start_distribution': rec.get('start_distribution'),
        'dataset_card': mode.get('dataset_card'),
        'ui_note': mode.get('ui_note_ja'),          # 手で走らせるときの注意(2026-10-15 昇格の段。止まったまま動かない開始など)
        'eval_created_at': (ev or {}).get('created_at'),
    }


def list_modes(with_datasets=True):
    out = []
    base = modes_dir()
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.name.startswith('.') or not KEY_RE.match(d.name) or d.is_symlink() or not d.is_dir():
                continue
            try:
                out.append(load_mode(d.name))
            except RlError as e:
                out.append({'key': d.name, 'name': d.name, 'selectable': False, 'state': 'broken', 'label': '壊れています', 'problems': [str(e)]})
    out.sort(key=lambda m: (m.get('order', 99), m['key']))
    if with_datasets:
        ds = collected_datasets()
        for m in out:
            m['datasets'] = ds.get(m['key'], {'count': 0, 'scenes': 0, 'minutes': 0.0, 'names': []})
    return {'enabled': enabled(), 'courses': list(RL_COURSES), 'modes': out, 'scenes_per_minute': SCENES_PER_MINUTE,
            'max_minutes': MAX_MINUTES, 'default_minutes': DEFAULT_MINUTES}


def mode_for_page(key):
    """ページ(teacher.js・sim-rl.js)に渡すモードの情報(選べるものだけ)"""
    m = load_mode(key)
    if not m['selectable']:
        raise RlError('このモードは選べません: %s' % ('・'.join(m['problems']) or m['label']), 409)
    return m


def policy_file(key):
    m = load_mode(key)
    if not m['selectable']:
        raise RlError('このモードの方策は配れません: %s' % ('・'.join(m['problems']) or m['label']), 409)
    return _mode_dir(key) / 'policy.json', m['policy']['sha256']


def collected_datasets():
    """集めたデータセット(タグ sim-rl-<key>)の数・シーン・分。カードだけを読む"""
    out = {}
    base = Path(config.DATASET_DIR)
    if not base.is_dir():
        return out
    for d in base.iterdir():
        if d.name.startswith('.') or not d.is_dir():
            continue
        p = d / 'dataset_card.json'
        try:
            if p.stat().st_size > MAX_JSON_BYTES:
                continue
            card = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        tags = card.get('tags') or []
        for t in tags:
            if isinstance(t, str) and t.startswith('sim-rl-'):
                k = t[len('sim-rl-'):]
                e = out.setdefault(k, {'count': 0, 'scenes': 0, 'minutes': 0.0, 'names': []})
                if card.get('rl_superseded'):
                    # 前の版の方策で記録したデータセット(カードに rl_superseded。2026-10-15 昇格の段)は別に数える
                    o = e.setdefault('superseded', {'count': 0, 'names': []})
                    o['count'] += 1
                    o['names'].append(d.name)
                    continue
                e['count'] += 1
                sc = card.get('scenes') or 0
                sc = len(sc) if isinstance(sc, list) else int(sc or 0)
                e['scenes'] += sc
                e['minutes'] = round(e['minutes'] + float(card.get('frames') or 0) / 30.0 / 60.0, 1)
                e['names'].append(d.name)
    return out


# ---------------------------------------------------------------- 収集プラン
def _now():
    return datetime.datetime.now()


def _plan_path(pid):
    if not isinstance(pid, str) or not PLAN_RE.match(pid):
        raise RlError('プランの ID の形が違います', 404)
    return plans_dir() / ('%s.json' % pid)


def _scene_fields(sc, split, name, scenario):
    """アプリの連続試験の 1 条件(JetFormer が決める値だけ。センサー・物理などはページが RC_COLLECTION_PLAN[0] から足す)"""
    st = sc['start']
    return {'name': name, 'dataset-split': split, 'seed': int(sc['seed']), 'start-gate': int(st['gate']),
            'spawn-offset': float(st['left_offset_m']), 'spawn-yaw': float(st['left_yaw_deg']), 'scenario': scenario,
            'controller': 'demo', 'teacher-profile': 'speed-rc', 'episode-mode': 'continuous-laps-v1', 'limit': 20, 'dataset-mode': 8}


def _scene_entry(sid, key, sc, split, name, scenario, pool_index):
    return {'id': sid, 'mode': key, 'name': name, 'split': split, 'scenario': scenario, 'fields': _scene_fields(sc, split, name, scenario),
            'params': dict(sc.get('params') or {}), 'pool_index': pool_index,
            'expect': {'min_clearance': sc.get('min_clearance'), 'valid_laps': sc.get('valid_laps'), 'laps': sc.get('laps'),
                       'actions_sha256': sc.get('actions_sha256'), 'events': sc.get('events'), 'learn0_frames': sc.get('learn0_frames')}}


def _plan_serial():
    d = plans_dir()
    try:
        return sum(1 for p in d.iterdir() if p.suffix == '.json')
    except OSError:
        return 0


def create_plan(body):
    """body = {course, minutes: {key: 分}, val_ratio, seed, delete_sources}"""
    if not enabled():
        raise RlError('強化学習モードは止められています(JETFORMER_SIM_RL=0)', 409)
    if not isinstance(body, dict):
        raise RlError('プランの指定が違います')
    course = body.get('course') or 'default'
    if not course_supported(course):
        raise RlError('強化学習モードは Default コースだけで使えます(RL の観測が Default の参照ループ専用のため)')
    minutes = body.get('minutes')
    if not isinstance(minutes, dict) or not minutes:
        raise RlError('モードごとの分(minutes)を指定してください')
    try:
        val_ratio = float(body.get('val_ratio', 0.2))
    except (TypeError, ValueError):
        raise RlError('val_ratio は数にしてください')
    if not 0.0 <= val_ratio <= 0.5:
        raise RlError('val_ratio は 0〜0.5 にしてください')
    serial = _plan_serial()
    seed = body.get('seed', serial)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1_000_000:
        raise RlError('seed は 0〜999999 の整数にしてください')
    plan_modes, scenes, order = {}, [], []
    want = []
    for key, mins in minutes.items():
        if isinstance(mins, bool) or not isinstance(mins, int) or not 0 <= mins <= MAX_MINUTES:
            raise RlError('%s の分は 0〜%d の整数にしてください' % (key, MAX_MINUTES))
        if mins:
            want.append((key, mins))
    if not want:
        raise RlError('どのモードも 0 分です')
    loaded = []
    for key, mins in want:
        m = load_mode(key)
        if not m['plan_ok']:
            raise RlError('%s はプランに使えません(%s)' % (m['name'], '・'.join(m['problems']) or m['label'] if not m['selectable']
                                                  else '記録のシーンの組がありません'), 409)
        loaded.append((m, mins))
    loaded.sort(key=lambda x: (x[0]['order'], x[0]['key']))
    stamp = _now()
    for m, mins in loaded:
        key = m['key']
        pool = _read_json(_mode_dir(key) / 'scenes.json')['scenes']
        n = mins * SCENES_PER_MINUTE
        n_spare = int(n * SPARE_RATIO) + 2
        if n > len(pool):
            raise RlError('%s の記録のシーンの組は %d シーン(%d 分)までです' % (m['name'], len(pool), len(pool) // SCENES_PER_MINUTE), 409)
        n_spare = min(n_spare, len(pool) - n)
        off = (seed * 37) % len(pool)
        pick = [(off + j) % len(pool) for j in range(n + n_spare)]
        n_val = int(round(n * val_ratio))
        # val は等間隔に(15 シーンで 0.2 なら 5 本ごとの 5 本目 = 3 本)
        val_idx = set(int((k + 1) * n / n_val) - 1 for k in range(n_val)) if n_val else set()
        counters = {'train': 0, 'val': 0}
        for j in range(n):
            split = 'val' if j in val_idx else 'train'
            name = 'rl-%s-%s-%02d' % (key, split, counters[split])
            counters[split] += 1
            sid = 's%03d' % len(scenes)
            scenario = SCENARIO_ROTATION[j % len(SCENARIO_ROTATION)]
            scenes.append(_scene_entry(sid, key, pool[pick[j]], split, name, scenario, pick[j]))
            order.append(sid)
        plan_modes[key] = {
            'name': m['name'], 'state': m['state'], 'label': m['label'], 'policy_sha256': m['policy_sha256'], 'minutes': mins,
            'scenes': n, 'train': n - n_val, 'val': n_val, 'tags': m['tags'],
            'dataset_name': 'sim-rl-%s_%s' % (key, stamp.strftime('%Y%m%d-%H%M')),
            'spares': [{'pool_index': i, 'scene': pool[i]} for i in pick[n:]], 'spares_used': 0,
            'dataset_card': m.get('dataset_card'),
        }
    pid = '%s-%s' % (stamp.strftime('%Y%m%d-%H%M%S'), uuid.uuid4().hex[:6])
    plan = {'schema': PLAN_SCHEMA, 'id': pid, 'created_at': stamp.isoformat(timespec='seconds'), 'course': course, 'seed': seed,
            'serial': serial, 'val_ratio': val_ratio, 'minutes': {m['key']: mins for m, mins in loaded},
            'delete_sources': bool(body.get('delete_sources', False)), 'modes': plan_modes, 'scenes': scenes, 'order': order,
            'state': 'ready', 'progress': {}, 'imports': {}, 'calibration': None, 'updated_at': stamp.isoformat(timespec='seconds'),
            'url': '/sim/c/%s/?rlplan=%s' % (course, pid)}
    _save_plan(plan)
    return plan


def _save_plan(plan):
    d = plans_dir()
    d.mkdir(parents=True, exist_ok=True)
    plan['updated_at'] = _now().isoformat(timespec='seconds')
    paths.write_durable(_plan_path(plan['id']), json.dumps(plan, ensure_ascii=False, indent=1))


def get_plan(pid):
    p = _plan_path(pid)
    plan = _read_json(p)
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise RlError('プランがありません: %s' % pid, 404)
    return plan


def plan_summary(plan):
    """一覧・カード用の短い形"""
    prog = plan.get('progress') or {}
    by_mode = {}
    for sid in plan.get('order', []):
        sc = next((s for s in plan['scenes'] if s['id'] == sid), None)
        if not sc:
            continue
        e = by_mode.setdefault(sc['mode'], {'planned': 0, 'done': 0, 'failed': 0, 'replaced': 0})
        st = (prog.get(sid) or {}).get('status')
        if not sid[4:]:
            e['planned'] += 1
        if st == 'done':
            e['done'] += 1
        elif st == 'failed':
            e['failed'] += 1
        if sid[4:]:
            e['replaced'] += 1
    return {'id': plan['id'], 'created_at': plan.get('created_at'), 'updated_at': plan.get('updated_at'), 'state': plan.get('state'),
            'course': plan.get('course'), 'minutes': plan.get('minutes'), 'url': plan.get('url'), 'modes': by_mode,
            'imports': {k: {kk: v.get(kk) for kk in ('state', 'dataset', 'error', 'job_id', 'scenes')} for k, v in (plan.get('imports') or {}).items()},
            'message': plan.get('message')}


def list_plans(limit=20):
    d = plans_dir()
    out = []
    try:
        files = sorted((p for p in d.iterdir() if p.suffix == '.json' and PLAN_RE.match(p.stem)), reverse=True)[:limit]
    except OSError:
        files = []
    for p in files:
        try:
            out.append(plan_summary(get_plan(p.stem)))
        except RlError:
            continue
    return out


PROGRESS_STATES = ('recording', 'done', 'failed', 'skipped')
PLAN_STATES = ('ready', 'running', 'stopped', 'finished', 'error')


def update_progress(pid, body):
    """ページの実行係からの進み具合。body = {scene, status, session_id, frames, collisions, laps, reason, error, replace,
    plan_state, message, calibration}。replace = true で、外れたシーンの次にそのモードの替えを入れる"""
    if not isinstance(body, dict):
        raise RlError('進み具合の形が違います')
    with _lock:
        plan = get_plan(pid)
        sid = body.get('scene')
        replaced = None
        if sid is not None:
            sc = next((s for s in plan['scenes'] if s['id'] == sid), None) if isinstance(sid, str) and SCENE_ID_RE.match(sid) else None
            if sc is None:
                raise RlError('プランにないシーンです: %s' % sid, 404)
            st = body.get('status')
            if st not in PROGRESS_STATES:
                raise RlError('status は %s のどれかにしてください' % '・'.join(PROGRESS_STATES))
            rec = {'status': st, 'at': _now().isoformat(timespec='seconds')}
            for k, typ in (('session_id', str), ('frames', int), ('collisions', int), ('reason', str), ('error', str), ('valid_laps', int)):
                v = body.get(k)
                if isinstance(v, typ) and not isinstance(v, bool):
                    rec[k] = v[:500] if isinstance(v, str) else v
            if isinstance(body.get('laps'), list):
                rec['laps'] = [float(x) for x in body['laps'][:10] if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if rec.get('session_id') and not re.match(r'^[0-9a-f]{32}$', rec['session_id']):
                raise RlError('session_id の形が違います')
            prev = plan['progress'].get(sid) or {}
            if prev.get('replaced_by'):
                rec['replaced_by'] = prev['replaced_by']
            plan['progress'][sid] = rec
            if st == 'failed' and body.get('replace') and not prev.get('replaced_by'):
                replaced = _insert_spare(plan, sc)
                if replaced:
                    plan['progress'][sid]['replaced_by'] = replaced['id']
        ps = body.get('plan_state')
        if ps is not None:
            if ps not in PLAN_STATES:
                raise RlError('plan_state は %s のどれかにしてください' % '・'.join(PLAN_STATES))
            plan['state'] = ps
        if isinstance(body.get('message'), str):
            plan['message'] = body['message'][:500]
        cal = body.get('calibration')
        if isinstance(cal, dict):
            c = {k: cal.get(k) for k in ('policy_loaded', 'policy_name', 'steer_center_us', 'throttle_neutral_us', 'source')}
            if plan.get('calibration') is None:
                plan['calibration'] = c
        _save_plan(plan)
        return {'plan': plan, 'replacement': replaced}


def _insert_spare(plan, failed):
    key = failed['mode']
    md = plan['modes'][key]
    if md['spares_used'] >= len(md['spares']):
        return None
    sp = md['spares'][md['spares_used']]
    md['spares_used'] += 1
    base = failed['id'][:4]
    k = 1
    while any(s['id'] == '%sr%d' % (base, k) for s in plan['scenes']):
        k += 1
    sid = '%sr%d' % (base, k)
    # 替えの名前は「元の名前-r<k>」。末尾の -r<数> だけを外す(split('-r') だと rl-recover-… が「rl」で切れていた。審査の段 2026-10-14)
    name = '%s-r%d' % (re.sub(r'-r\d+$', '', failed['name']), k)
    entry = _scene_entry(sid, key, sp['scene'], failed['split'], name, failed['scenario'], sp['pool_index'])
    plan['scenes'].append(entry)
    i = plan['order'].index(failed['id'])
    plan['order'].insert(i + 1, sid)
    return entry


def mode_session_ids(plan, key):
    """プランの中の、そのモードの記録できたシーン(順番どおり)。[(シーン, session_id)]"""
    out = []
    for sid in plan.get('order', []):
        sc = next((s for s in plan['scenes'] if s['id'] == sid), None)
        rec = (plan.get('progress') or {}).get(sid) or {}
        if sc and sc['mode'] == key and rec.get('status') == 'done' and rec.get('session_id'):
            out.append((sc, rec['session_id']))
    return out


def mode_failed_session_ids(plan, key):
    """プランの中の、そのモードの外れたシーン(衝突など)の session_id。取り込みのあと元のシーンを消すときに一緒に消す"""
    out = []
    for sc in plan.get('scenes', []):
        rec = (plan.get('progress') or {}).get(sc['id']) or {}
        if sc['mode'] == key and rec.get('status') == 'failed' and rec.get('session_id'):
            out.append(rec['session_id'])
    return out


def set_import_state(pid, key, **kw):
    with _lock:
        plan = get_plan(pid)
        cur = plan.setdefault('imports', {}).get(key) or {}
        cur.update({k: v for k, v in kw.items()})
        cur['at'] = _now().isoformat(timespec='seconds')
        plan['imports'][key] = cur
        _save_plan(plan)
        return cur
