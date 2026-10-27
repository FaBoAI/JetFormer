#!/usr/bin/env python3
"""区間の成績表(U1、2026-10-21): 1 走行・複数走行の「衝突/周・区間の通過あたりの衝突・ラップ・交代・再起動」を 1 コマンドで出す

    python tools/section_scorecard.py 20261010-172910 20261010-173445            # → data/reports/scorecard_<名前>/
    python tools/section_scorecard.py --day 20261023 --out DIR                     # その日の AI のログ全部
    python tools/section_scorecard.py --day 20261023 --bench data/bench_candidates.json   # 候補の id ごと(説明欄の cand=)
    python tools/section_scorecard.py LOGS --assign assign.json                    # 説明欄を上書きして割り当てる(模擬・直し用)

読み取りだけ(numpy、GPU なし)。生ログ(data/raw)とラベル(data/labels)は変えない。ラベルが無いログは
tools/label_events.py の label_log をその場で呼ぶ(書き出さない)。出力は scorecard.json と scorecard.md。

出すものは 2 つ:
  手順の指標(Markdown の 0 章・JSON の groups.*.protocol): bench_prep_20261021/plan/protocol.json の metrics の名前と定義。
    数える区間 = K 番目の AI の R1 の起点 ta_K の 0.5 秒前から AI の時間 counted_ai_s(既定 240 秒)まで。
    周 j = [ta_j − 0.5, 次の AI の R1 の起点 − 0.5)。区間の AI の衝突は全部数える(立て直し中も)。
    失敗/周 = (衝突 + stuck・preemptive の交代)÷ 周。基板の再起動のあった周は分子も分母も除く。
    再起動/4 分 = ブロックの時間(最初の AI のフレームの 0.5 秒前 → 数える区間の終わり。240 秒に届かなければ記録の終わり)の
    基板の再起動の全部 ÷(その中の AI の時間 / 240 秒)。慣らし・最後の半端な周・周が 0 本のブロックも入る(10/21 V-tools の直し)。
    R1 の旋回中のブレーキ(cap_brake_in_r1_turn)= ON のブレーキのフレームのうち、後から見た起点 ta → ta + 0.5 秒のもの。
    通過あたりの率には、正確な二項(Clopper-Pearson)の 90% 区間も並べる(ci90_exact。順位・判定はブートストラップの区間)。
    90% 区間 = 周を並びのまま 3 周ずつのかたまりにし、走行ごとにかたまりを復元抽出するブートストラップ(既定 10,000 回、種 = その日)。
  ログ全体の数え方(1〜4 章。10/10 の数と比べるための参考): 下のとおり。

ログ全体の数え方(どれも tools/label_events.py・tools/offline_rl.py の course_passes と同じ規則):
  AI の時間   ラベルの ai(CH3 が AI 側)かつ dry_run でないフレームの dt の和
  周相当      AI が走っている間(|eRPM| > 200)のジャイロの向きの変化 ÷ 360°(左回り)。衝突/周・交代/周の分母
  衝突        既定(rule=label)= label_events の衝突(水平 2.5G・0.6 秒でまとめる)のうち AI 側のもの。
              参考(rule=cmap)= collision_map.md と同じ |‖a‖ − 1G| ≥ 2.5G(0.6 秒でまとめる・1 サンプルだけのものは除く)。
              どちらも「立て直し中」(失敗による交代のあと、人が AI に戻してから 2.5 秒以内)の衝突は数えず、別に出す
              (collision_map.md の「回復」3 件と同じもの)。強い衝突 = 4G 以上(label の strong)
  通過        R1 の起点(label_events.course_anchors、1 周に 1 回の大きな右ヘアピンの入口)ごと。AI の通過 = 起点の時刻に AI 側。
              偽の起点(衝突で右へ回ったもの)は外す(label_events.reject_false_anchors。外したものは r1_anchors_rejected、
              ラベルのファイルが外す前のものなら衝突・交代・フレームの場所を付け直す = loc_relabeled)。
              R1 の衝突 = 起点 −0.5 秒〜緑マット出口 −0.3 秒(出口が無ければ起点 +3.5 秒)の場所 R1 の衝突、
              GM = 起点 〜 出口 +0.3 秒の場所 GM、DE = 出口 −0.3 〜 +2.3 秒の場所 DE(通過の数え方は course_passes と同じ。
              通過あたりの衝突は rule=label の全衝突で数える = passes.json と同じ)
  R1 の進入   起点の手前 0.4 秒の最大の前進 eRPM。5,500 未満 / 5,500〜6,200 / 6,200 以上の帯ごとに 強い衝突/通過
  ラップ      R1 の起点 → 次の起点(5〜16 秒・ずっと AI・0.2 秒を超える途切れなし)。きれいな周 = その間に AI の衝突・交代・
              リカバリー・基板の再起動が無い。区間の時間 = その周の中で、ラベルの場所(loc_code)ごとのフレームの dt の和
              (R1+GM・後半 DE+SW+HS・LH は explore_control.md 0.1 の表と同じまとめ)。LapCounter の周(走行画面と同じ)も出す
  交代        AI → 人(CH3)。失敗による交代 = after_collision・stuck・preemptive(label_events の分類)
  基板の再起動 VEHICLE.board_t_us が戻った回数(label_events の board_reboots)。AI 側で起きたものも数える
  信頼区間    衝突/周などの率は、件数のポアソンの正確な 90% 区間 ÷ 周相当。通過あたりは Clopper-Pearson の 90% 区間
  説明欄      走行開始の「説明」欄の bench=… day=… block=… cand=… model=… battery=#… V_before=… V_after=… check=… memo=…
              (data/bench_candidates.json の description_template)。cand・battery・block ごとにまとめる
  速さの制御  走行ごとに speed_ceiling・speed_auto(log.json の drive.params)と、AI_CMD の天井(ceil_*)・自動合わせ(sa_*)の
              要約(speed_ctrl。天井が出力を変えた・影で弱める/中立・左旋回の割合、m_fb の終わり・使った周、印)
  電圧        vbat: VEHICLE.npz の vbat_v(基板)→ scene.json の stats.vbat → 説明欄の V_before・V_after の順
"""

import argparse
import json
import math
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from jetformer import config  # noqa: E402
import label_events as le  # noqa: E402

SCHEMA = 'jf-section-scorecard/v1'
KV = le.KV
G = le.G
CMAP_G = 2.5                 # collision_map.md の |‖a‖ − 1G| のしきい値
RESTART_S = 2.5              # 失敗による交代のあと、人が AI に戻してからこの秒数の衝突は「立て直し中」
MOVING_ERPM = 200
ENTRY_BINS = [(0, 5500, '<5500'), (5500, 6200, '5500-6200'), (6200, 1e9, '>=6200')]
LAP_MIN_S, LAP_MAX_S = 5.0, 16.0
LOCS = ['R1', 'GM', 'DE', 'LH', 'HS', 'other', 'unknown']
SECTION_GROUPS = {'R1+GM': ('R1', 'GM'), 'DE+SW+HS': ('DE', 'other', 'HS'), 'LH': ('LH',)}
WARMUP_DEFAULT = 2           # 手順の慣らしの R1 の起点の数 K(その日に初めてのモデルは 3)
COUNTED_AI_S = 240.0         # 手順の数える AI の時間
BOOT_N = 10000
BOOT_CHUNK = 3
LAMBDA_S = 1.0               # 順位の点数 S = Δ失敗/周 + λ × Δラップの中央値
DESC_KEYS = ('bench', 'day', 'block', 'cand', 'model', 'battery', 'V_before', 'V_after', 'check', 'memo')
CONF = 0.90


# ------------------------------------------------------------------ 小さな統計
def _logpmf_pois(k, mu):
    return -mu + k * math.log(mu) - math.lgamma(k + 1) if mu > 0 else (0.0 if k == 0 else -math.inf)


def pois_cdf(k, mu):
    if k < 0:
        return 0.0
    if mu <= 0:
        return 1.0
    return min(1.0, sum(math.exp(_logpmf_pois(i, mu)) for i in range(int(k) + 1)))


def _bisect(f, lo, hi, it=80):
    """f(lo) と f(hi) の符号が違うときの根"""
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def poisson_ci(k, conf=CONF):
    """件数 k の平均の正確な(Garwood)両側の区間"""
    a = (1 - conf) / 2
    k = int(k)
    hi_b = k + 20 + 10 * math.sqrt(k + 1)
    lo = 0.0 if k == 0 else _bisect(lambda mu: (1 - pois_cdf(k - 1, mu)) - a, 1e-9, hi_b)
    hi = _bisect(lambda mu: pois_cdf(k, mu) - a, 1e-9, hi_b)
    return lo, hi


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    s = 0.0
    for i in range(int(k) + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p)
                      + (n - i) * math.log(1 - p))
    return min(1.0, s)


def cp_ci(k, n, conf=CONF):
    """Clopper-Pearson の両側の区間"""
    if n <= 0:
        return None, None
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else _bisect(lambda p: (1 - binom_cdf(k - 1, n, p)) - a, 1e-12, 1 - 1e-12)
    hi = 1.0 if k == n else _bisect(lambda p: binom_cdf(k, n, p) - a, 1e-12, 1 - 1e-12)
    return lo, hi


def rate(k, denom, conf=CONF):
    """件数 / 周相当 と 90% 区間"""
    if not denom or denom <= 0:
        return {'n': int(k), 'per': None, 'ci90': [None, None]}
    lo, hi = poisson_ci(k, conf)
    return {'n': int(k), 'per': _r(k / denom, 3), 'ci90': [_r(lo / denom, 3), _r(hi / denom, 3)]}


def prop(k, n, conf=CONF):
    lo, hi = cp_ci(k, n, conf)
    return {'k': int(k), 'n': int(n), 'p': _r(k / n, 3) if n else None, 'ci90': [_r(lo, 3), _r(hi, 3)],
            'text': '%d/%d' % (k, n)}


def _r(x, n=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if math.isfinite(x) else None


def _med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return _r(float(np.median(xs)), 3) if xs else None


def _min(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return _r(min(xs), 3) if xs else None


# ------------------------------------------------------------------ 説明欄
def parse_description(s):
    """走行開始の説明欄 → dict。'bench=x day=2026-10-23 block=2 cand=MSF model=… battery=#3 V_before=8.2 V_after=7.6
    check=ok memo=自由な文' の形。memo= より後ろは全部 memo。'__' や空の値は None。"""
    out = {}
    if not s:
        return out
    s = str(s)
    m = re.search(r'(?:^|\s)memo=(.*)$', s)
    if m:
        out['memo'] = m.group(1).strip() or None
        s = s[:m.start()]
    for k, v in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=(\S*)', s):
        v = v.strip()
        if not v or set(v) <= set('_#'):
            v = None
        elif k == 'battery':
            v = v.lstrip('#') or None
        elif k == 'block':
            try:
                v = int(v)
            except ValueError:
                pass
        out[k] = v
    return out


DESC_ORDER = ('bench', 'day', 'block', 'cand', 'model', 'battery', 'V_before', 'V_after', 'check')


def _desc_text(d):
    """parse_description の dict → 説明欄の文(memo は最後)"""
    keys = [k for k in DESC_ORDER if k in d] + [k for k in d if k not in DESC_ORDER and k != 'memo']
    parts = ['%s=%s' % (k, ('#%s' % d[k] if k == 'battery' else d[k]) if d[k] is not None else '__') for k in keys]
    if d.get('memo'):
        parts.append('memo=%s' % d['memo'])
    return ' '.join(parts)


def _free_text(s):
    """説明欄の key=値 でも memo= でもない文(「安全で中止」を memo= の外に書いたときなど)"""
    s = str(s or '')
    m = re.search(r'(?:^|\s)memo=(.*)$', s)
    if m:
        s = s[:m.start()]
    return ' '.join(re.sub(r'[A-Za-z_][A-Za-z0-9_]*=\S*', ' ', s).split())


def scene_descriptions(log, raw_dir=None):
    """記録の中のシーン(data/raw/<log>/<scene>/scene.json)の説明 [(更新時刻, シーン名, 説明)]"""
    root = Path(raw_dir or config.RAW_DIR) / log
    out = []
    if not root.is_dir():
        return out
    for sj in sorted(root.glob('*/scene.json')):
        try:
            d = json.loads(sj.read_text(encoding='utf-8'))
            out.append((sj.stat().st_mtime, sj.parent.name, str(d.get('description') or '')))
        except (OSError, ValueError):
            continue
    return out


def effective_description(log, base, raw_dir=None):
    """走ったあとの直しを入れた説明欄(10/21 Fix-all、V-protocol major 3)。

    記録の説明(log.json・ラベル)は走行開始のときの文で、走ったあとに書く V_after・「安全で中止」・「回れない」は
    シーンの説明(PATCH /api/scenes/<名前> = scene.json)にしか入らない。そこで、log.json と違う説明のシーンを
    「直し」とみなし、値の入った欄(V_after=7.6 など。'__' や空は直しではない)で記録の説明を上書きする。
    直しのシーンが複数あれば更新の新しい順に優先し、同じ欄に違う値があれば conflicts に残す。memo と key=値 の外の文は
    全部をつなぐ(安全で中止・回れないの言葉を落とさない)。返り値 (説明欄の文, {'edited_scenes', 'conflicts', 'changed'})"""
    base = str(base or '')
    edits = [(t, n, d) for t, n, d in scene_descriptions(log, raw_dir) if d.strip() and d.strip() != base.strip()]
    info = {'edited_scenes': [n for _, n, _ in sorted(edits, key=lambda x: x[1])], 'conflicts': [], 'changed': False}
    if not edits:
        return base, info
    merged = parse_description(base)
    seen = {}
    memos = [merged['memo']] if merged.get('memo') else []
    extra = [x for x in [_free_text(base)] if x]
    for t, n, d in sorted(edits, key=lambda x: (x[0], x[1])):          # 古い直し → 新しい直し(新しい方が上書き)
        e = parse_description(d)
        for k, v in e.items():
            if k == 'memo' or v is None:
                continue
            if k in seen and seen[k][1] != v:
                info['conflicts'].append('%s: %s=%s(%s)と %s=%s(%s)→ 新しい直しの %s を使った' % (
                    k, k, seen[k][1], seen[k][0], k, v, n, v))
            seen[k] = (n, v)
            merged[k] = v
        if e.get('memo') and e['memo'] not in memos:
            memos.append(e['memo'])
        ft = _free_text(d)
        if ft and ft not in extra:
            extra.append(ft)
    note = ' / '.join(memos + extra)
    merged['memo'] = note or None
    out = _desc_text(merged)
    info['changed'] = out.strip() != base.strip()
    return (out if info['changed'] else base), info


def load_bench(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def assign_candidate(desc, run, bench):
    """説明欄の cand= を優先。無ければ説明欄の model= か記録の run で候補の run と照合"""
    if desc.get('cand'):
        return desc['cand'], 'description'
    if bench:
        for c in bench.get('candidates', []):
            r = c.get('run')
            if r and (desc.get('model') == r or run == r):
                return c['id'], 'run'
    return None, None


# ------------------------------------------------------------------ 1 本のログ
# 記録・ラベルのファイルが読めないときの例外(0 バイト・書きかけ・壊れた zip・JSON。電源断の記録)
UNREADABLE = (OSError, ValueError, EOFError, zipfile.BadZipFile)


def _labels(log, labels_dir, raw_dir):
    """ラベル → (info, lab, 元)。ファイルがあれば読み、無ければ記録からその場で作る(書き出さない)。
    2026-10-26 fix2: ラベルのファイルが読めなければ記録から作り直す。記録の log.json が読めない(0 バイト。20261023-204825)
    ときは落ちずに注意を出し、(error 付きの info, None, 元) を返す(score_log・car_check は「読めなかった」にする)"""
    pj, pn = Path(labels_dir) / (log + '.json'), Path(labels_dir) / (log + '.npz')
    if pj.exists() and pn.exists():
        try:
            info = json.loads(pj.read_text(encoding='utf-8'))
            with np.load(pn) as z:
                lab = {k: z[k] for k in z.files}
            return info, lab, 'file'
        except UNREADABLE as e:
            print('注意: %s のラベルのファイルが読めません(%s: %s)。記録から作り直します' % (log, type(e).__name__, e),
                  file=sys.stderr)
    rd = Path(raw_dir or config.RAW_DIR) / log
    if rd.is_dir():         # 記録の log.json(0 バイト = 電源断。label_log は npz の壊れは飛ばすが log.json は読めないと落ちる)
        try:                # 記録の dir が無い(名前の間違い)は今まで通り例外(呼び手が捕まえる)
            json.loads((rd / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            msg = '記録の log.json が読めません(%s: %s)' % (type(e).__name__, str(e)[:120])
            print('注意: %s の%s。飛ばします' % (log, msg), file=sys.stderr)
            return {'log': log, 'error': msg}, None, 'computed'
    info, lab = le.label_log(log, raw_dir)
    return info, lab, 'computed'


def cmap_collisions(it, accel, thr=CMAP_G, merge_s=le.IMPACT_MERGE_S):
    """collision_map.md(cmap_src/ld.py)と同じ |‖a‖ − 1G| の衝突。0.6 秒以内は 1 件、前後 50ms に 1G 超が 1 つだけなら除く"""
    g = np.abs(np.linalg.norm(accel.astype(np.float64), axis=1) - G) / G
    ev = []
    for i in np.nonzero(g > thr)[0]:
        if ev and it[i] - ev[-1]['t'] <= merge_s:
            if g[i] > ev[-1]['g']:
                ev[-1]['g'] = float(g[i])
            continue
        near = (np.abs(it - it[i]) <= 0.05) & (g > 1.0)
        ev.append({'t': float(it[i]), 'g': float(g[i]), 'suspect': bool(near.sum() <= 1)})
    return [e for e in ev if not e['suspect']]


def score_log(log, labels_dir=None, raw_dir=None, desc_override=None, bench=None, warmup_anchors=None, counted_ai_s=None,
              labels=None, loaded=None, extras=None):
    """1 本のログの成績(数と分母。率は aggregate で出す)。warmup_anchors・counted_ai_s を渡すと、手順の数える区間
    (K 番目の AI の R1 の起点の 0.5 秒前から AI の時間 counted_ai_s 秒まで)の周の表 laps_protocol も作る。
    labels = (info, lab, 元) と loaded = le.load_log の結果を渡すと読み直さない(tools/car_check.py)。
    extras に dict を渡すと、使った起点・緑マット出口・IMU の時刻・向き・平滑したヨー・場所(loc_code)・出来事を入れて返す"""
    labels_dir = Path(labels_dir or config.DATA_DIR / 'labels')
    raw_dir = Path(raw_dir or config.RAW_DIR)
    info, lab, lab_src = labels if labels is not None else _labels(log, labels_dir, raw_dir)
    if desc_override is not None:
        desc_raw = desc_override
    else:                                   # 走ったあとにシーンの説明で直した欄(V_after など)も入れる
        desc_raw, _ = effective_description(log, info.get('description', ''), raw_dir)
    desc = parse_description(desc_raw)
    run = info.get('run')
    cand, cand_src = assign_candidate(desc, run, bench)
    res = {'log': log, 'description': desc_raw, 'desc': desc, 'cand': cand, 'cand_source': cand_src, 'run': run,
           'mode_id': info.get('mode_id'), 'params_at_start': info.get('params_at_start'), 'labels': lab_src,
           'ai_log': bool(info.get('ai_log')), 'problems': list(info.get('problems') or [])}
    if lab is None or 'events' not in info:
        res['error'] = info.get('error', 'ラベルがありません')
        return res
    t = lab['t_s'].astype(np.float64)
    n = len(t)
    ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)
    dt = lab['dt_s'].astype(np.float64)
    assist = lab['assist']
    ev = info['events']
    L = loaded if loaded is not None else le.load_log(log, raw_dir)
    V, I = L['VEHICLE'], L['IMU']
    t0 = info['t0_us']
    sec = lambda x: (np.asarray(x, np.int64) - t0) / 1e6      # noqa: E731
    res['ai_s'] = _r(dt[ai].sum(), 2)
    res['duration_s'] = info.get('duration_s')
    if V is None or I is None:
        res['error'] = 'VEHICLE・IMU がありません'
        return res
    vt = sec(V['timestamps_us'])
    verpm = V['erpm'].astype(np.float64)
    it = sec(I['timestamps_us'])
    gz = I['gyro'][:, 2].astype(np.float64)
    still = (np.abs(np.interp(it, vt, verpm)) < 50) & (np.abs(gz) < 5)
    hd, gzu = le.heading_deg(it, gz, still)
    # R1 の起点。偽の起点(衝突で右へ回ったもの)は外す(label_events.reject_false_anchors、spec 5.1)。
    # 人/AI はラベルの ai(CH3。dry run も AI 側)、衝撃はラベルの衝突(水平 4G 以上)。label_log と同じ結果になる
    anchors, gm_exit, gzs, rejected = le.checked_anchors(it, hd, gzu, vt, verpm, t, lab['ai'].astype(bool),
                                                         ev.get('collisions', []))
    anchors = np.asarray(anchors, float)
    gm_exit = np.asarray(gm_exit, float)
    res['r1_anchors_rejected'] = rejected
    res['anchor_rule'] = le.ANCHOR_RULE
    loc_code = lab['loc_code'].astype(int)
    res['loc_relabeled'] = False
    if rejected and info.get('anchor_rule') != le.ANCHOR_RULE:
        # ラベルのファイルが偽の起点を外す前に作られた → 衝突・交代の場所とフレームの場所を、外したあとの起点で付け直す
        ev, loc_code = relocate_events(ev, t, anchors, gm_exit, it, hd, gzs)
        res['loc_relabeled'] = True
    if extras is not None:
        extras.update(anchors=anchors, gm_exit=gm_exit, it=it, hd=hd, gzs=gzs, gzu=gzu, loc_code=loc_code, ev=ev,
                      vt=vt, verpm=verpm)

    # AI 側かを IMU の時刻で(最後のフレームの値。0.2 秒を超えてフレームが無ければ AI ではない)
    fi = np.clip(np.searchsorted(t, it, side='right') - 1, 0, n - 1)
    ai_imu = ai[fi] & (it - t[fi] <= le.GAP_S) & (it >= t[0])
    moving = np.abs(np.interp(it, vt, verpm)) > MOVING_ERPM
    dti = np.clip(np.diff(it, append=it[-1]), 0, 0.2)
    lap_eq = float(np.sum(gzu * dti * (ai_imu & moving)) / 360.0)
    res['lap_equiv'] = _r(abs(lap_eq), 3)
    res['r1_anchors'] = int(len(anchors))

    # 立て直し中の窓(失敗による交代 → 戻し → +RESTART_S)
    tks = ev.get('takeovers', [])
    restart = []
    for hb in ev.get('handbacks', []):
        prev = [x for x in tks if x['t_s'] <= hb['t_s']]
        if prev and prev[-1].get('failure'):
            restart.append((hb['t_s'], hb['t_s'] + RESTART_S))
    in_restart = lambda tc: any(a <= tc <= b for a, b in restart)      # noqa: E731

    def frame_at(tc):
        return int(np.clip(np.searchsorted(t, tc, side='right') - 1, 0, n - 1))

    # 衝突(label の規則)
    coll = []
    for c in ev.get('collisions', []):
        if not c.get('ai'):
            continue
        k = c['frame'] if c.get('frame') is not None else frame_at(c['t_s'])
        cls = 'restart' if in_restart(c['t_s']) else ('recovery' if int(assist[k]) == 2 else 'main')
        coll.append({'t_s': c['t_s'], 'loc': c['loc'], 'loc_detail': c.get('loc_detail'), 'g': c['g'],
                     'strong': bool(c.get('strong')), 'outcome': c.get('outcome'), 'class': cls,
                     'speed_before_mps': c.get('speed_before_mps')})
    # 衝突(collision_map の規則)
    cm = []
    for e in cmap_collisions(it, I['accel']):
        k = frame_at(e['t'])
        if not (t[0] <= e['t'] <= t[-1] + le.GAP_S) or not ai[k]:
            continue
        det = int(le.locate([e['t']], anchors, gm_exit, it, hd, gzs)[0])
        cls = 'restart' if in_restart(e['t']) else ('recovery' if int(assist[k]) == 2 else 'main')
        cm.append({'t_s': _r(e['t'], 3), 'g': _r(e['g'], 2), 'loc': le.LOC_NAMES[le._DETAIL_TO_LOC[det]], 'class': cls})

    def by_loc(rows):
        d = {k: 0 for k in LOCS}
        for c in rows:
            d[c['loc'] if c['loc'] in d else 'unknown'] += 1
        return d
    main_l = [c for c in coll if c['class'] == 'main']
    main_c = [c for c in cm if c['class'] == 'main']
    res['collisions'] = {'label': {'main': len(main_l), 'strong': sum(c['strong'] for c in main_l),
                                   'restart': sum(c['class'] == 'restart' for c in coll),
                                   'recovery': sum(c['class'] == 'recovery' for c in coll), 'by_loc': by_loc(main_l)},
                         'cmap': {'main': len(main_c), 'restart': sum(c['class'] == 'restart' for c in cm),
                                  'recovery': sum(c['class'] == 'recovery' for c in cm), 'by_loc': by_loc(main_c)}}
    res['collision_events'] = coll
    res['collision_events_cmap'] = cm

    # 通過(course_passes と同じ規則。衝突は label の全衝突)
    cs_all = ev.get('collisions', [])
    passes = []
    for j, ta in enumerate(anchors):
        tg = gm_exit[j] if j < len(gm_exit) else np.nan
        w = (vt >= ta - 0.4) & (vt <= ta)
        entry = float(np.max(-verpm[w])) if w.any() else np.nan
        hi_r1 = tg - 0.3 if np.isfinite(tg) else ta + 3.5
        c_r1 = [c for c in cs_all if c['loc'] == 'R1' and ta - 0.5 <= c['t_s'] <= hi_r1]
        c_gm = [c for c in cs_all if c['loc'] == 'GM' and np.isfinite(tg) and ta - 0.5 <= c['t_s'] <= tg + 0.3]
        c_de = [c for c in cs_all if c['loc'] == 'DE' and np.isfinite(tg) and tg - 0.3 <= c['t_s'] <= tg + 2.3]
        k = int(np.clip(np.searchsorted(t, ta), 0, n - 1))
        passes.append({'ta_s': _r(ta, 3), 'tg_s': _r(tg, 3) if np.isfinite(tg) else None, 'ai': bool(ai[k]),
                       'entry_erpm': _r(entry, 0), 'r1_collision': bool(c_r1), 'r1_strong': any(c['strong'] for c in c_r1),
                       'gm_collision': bool(c_gm), 'de_collision': bool(c_de), 'gm_exit': bool(np.isfinite(tg))})
    res['passes'] = passes

    # ラップ(R1 の起点 → 次の起点)と区間の時間
    tk_t = np.array([x['t_s'] for x in tks])
    rb_t = np.array([b['t_last_s'] for b in ev.get('board_reboots', [])])
    ai_coll_t = np.array([c['t_s'] for c in coll])
    _, Df = le.cumulative_distance(vt, verpm)
    laps = []
    for j in range(len(anchors) - 1):
        a, b = float(anchors[j]), float(anchors[j + 1])
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        dur = b - a
        tm = t[m]
        gaps = np.any(np.diff(tm) > le.GAP_S) or (tm[0] - a > le.GAP_S) or (b - tm[-1] > le.GAP_S)
        if not (LAP_MIN_S <= dur <= LAP_MAX_S) or not ai[m].all() or gaps:
            continue
        clean = (not np.any((ai_coll_t >= a) & (ai_coll_t < b)) and not np.any((tk_t >= a) & (tk_t < b))
                 and not np.any(assist[m] == 2) and not np.any((rb_t >= a - 0.5) & (rb_t < b)))
        sect = {}
        for code, name in enumerate(le.LOC_NAMES):
            if code == 0:
                continue
            sect[name] = _r(dt[m & (loc_code == code)].sum(), 3)
        laps.append({'t_start': _r(a, 3), 'lap_s': _r(dur, 3), 'clean': bool(clean), 'sections_s': sect,
                     'dpsi_deg': _r(float(np.interp(b, it, hd) - np.interp(a, it, hd)), 1),
                     'dist_m': _r(float(np.interp(b, vt, Df) - np.interp(a, vt, Df)), 2)})
    res['laps'] = laps
    lc = [x for x in ev.get('laps', []) if x.get('ai_only') and not x.get('first')]
    res['lapcounter'] = {'ai_laps': len(lc), 'clean_laps': sum(1 for x in lc if x.get('clean')),
                         'clean_times_s': [x['time_s'] for x in lc if x.get('clean')]}

    # 交代・再起動・そのほか
    res['takeovers'] = {'all': len(tks), 'failure': sum(1 for x in tks if x.get('failure')),
                        'by_class': {c: sum(1 for x in tks if x.get('class') == c) for c in le.TAKEOVER_CLASSES},
                        'by_loc': {k: sum(1 for x in tks if x.get('failure') and x.get('loc') == k) for k in LOCS}}
    rbs = ev.get('board_reboots', [])
    res['board_reboots'] = {'all': len(rbs), 'ai': sum(1 for b in rbs if ai[frame_at(b['t_last_s'])])}
    c = info.get('counts', {})
    res['recoveries'] = c.get('recoveries', 0)
    res['holds_ai'] = c.get('holds_ai', 0)
    res['guard_warnings_ai'] = c.get('guard_warnings_ai', 0)
    res['restart_windows'] = [[_r(a, 2), _r(b, 2)] for a, b in restart]
    K = WARMUP_DEFAULT if warmup_anchors is None else int(warmup_anchors)
    budget = COUNTED_AI_S if counted_ai_s is None else float(counted_ai_s)
    inf_ms = None
    A = L.get('AI_CMD')
    if A is not None and 'infer_ms' in A:
        at = sec(A['timestamps_us'])
        inf_ms = (at, A['infer_ms'].astype(np.float64))
    res['window'], res['laps_protocol'] = protocol_laps(t, ai, dt, assist, ev, anchors, gm_exit, vt, verpm, K, budget, inf_ms)
    # 基板の再起動/4 分の元(V-tools 4 章 1): ブロックの時間 = AI の走り始め(最初の AI のフレームの 0.5 秒前。慣らしも入れる)から
    # 数える区間の終わり(240 秒に届かなければ記録の終わり)まで。その中の再起動を全部(交代のあいだに人が運転していたときも。
    # 数えた周の行と同じ数え方)数え、その中の AI の時間で割る。数えた周の中だけにすると、慣らし・最後の半端な周・区間の後の
    # 再起動と、周が 0 本のブロックの再起動が消える
    w = res['window']
    rbs_all = ev.get('board_reboots', [])
    if ai.any():
        s0 = float(t[ai][0]) - 0.5
        s1 = float(w['end_s']) if w.get('budget_reached') and w.get('end_s') is not None else float('inf')
        w['block_span_s'] = [_r(s0, 3), None if not math.isfinite(s1) else _r(s1, 3)]
        w['reboots_block'] = sum(1 for b in rbs_all if s0 <= b['t_last_s'] < s1)
        w['ai_s_block'] = _r(float(dt[ai & (t < s1)].sum()), 2)
    else:
        w['reboots_block'], w['ai_s_block'] = 0, 0.0
    w['reboots_ai_log'] = res['board_reboots']['ai']            # 参考: ログ全体の AI 側(フレームが AI)の再起動
    w['reboots_all_log'] = res['board_reboots']['all']          # 参考: ログ全体(人が運転していたときも)
    if A is not None and 'cap_mode' in A:
        res['window']['cap'] = cap_counts(sec(A['timestamps_us']), A, t, ai, res['window'].get('start_s'),
                                          res['window'].get('end_s'), anchors, gm_exit)
    # 速さの天井・自動合わせ(AI_CMD の ceil_*・sa_*、log.json の drive.params)と電圧(spec 5.3)
    res['speed_ctrl'] = speed_ctrl_summary(A, t0, t, ai, res.get('params_at_start'))
    res['vbat'] = vbat_summary(V, t0, log, raw_dir, desc)
    del L
    return res


def relocate_events(ev, t, anchors, gm_exit, it, hd, gzs):
    """衝突・交代の場所(loc・loc_detail)とフレームの場所(loc_code)を、与えた起点で付け直す(元の ev は変えない)"""
    ev = dict(ev)

    def fix(rows):
        out = []
        for x in rows or []:
            x = dict(x)
            det = int(le.locate([x['t_s']], anchors, gm_exit, it, hd, gzs)[0])
            x['loc'] = le.LOC_NAMES[le._DETAIL_TO_LOC[det]]
            if 'loc_detail' in x:
                x['loc_detail'] = le.LOC_DETAIL[det]
            out.append(x)
        return out
    for k in ('collisions', 'collision_suspects', 'bumps', 'takeovers'):
        if k in ev:
            ev[k] = fix(ev[k])
    loc_code = le._DETAIL_TO_LOC[le.locate(t, anchors, gm_exit, it, hd, gzs)].astype(int)
    return ev, loc_code


# ------------------------------------------------------------------ 速さの天井・自動合わせ・電圧の要約(spec 2.6・3.9・4.2・5.3)
CEIL_MODES = {-2: 'failed', -1: 'off', 0: 'pass', 1: 'weak', 2: 'neutral'}
SA_EVENTS = {0: 'none', 1: 'corrected', 2: 'lap_not_used', 3: 'no_raise_ceiling', 4: 'no_raise_r1_shock',
             5: 'no_raise_after_collision', 6: 'deadband', 7: 'false_anchor', 8: 'abandoned_collision'}
SA_USED = (1, 3, 4, 5, 6)            # 周の速さで直しを決めた周(上げないも不感帯も「使った周」)
SA_MFB_EDGES = (0.90, 1.10)
CEIL_SWITCH_MAX_PER_S = 10           # 天井の出し方(そのまま・弱めた・中立)の入れ替わりがこれより多い 1 秒があれば印
AUTO_EDGE_LAPS = 3                   # m_fb が端に続けて張り付いた周の数の印
AUTO_FLIP_LAPS = 3                   # 上げ下げの向きが続けて入れ替わった回数の印
VBAT_EDGE_S = 2.0                    # 走り始め・終わりの電圧 = 最初・最後の 2 秒の中央値


def _runmax(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def speed_ctrl_summary(A, t0, t, ai, params=None):
    """AI_CMD の速さの天井(ceil_*)・自動合わせ(sa_*)の要約と、log.json の drive.params の speed_ceiling・speed_auto。
    ai はラベルのフレームの AI(dry run を除いたもの)。欄の無い古い記録は recorded = False"""
    p = params or {}
    out = {'speed_ceiling': p.get('speed_ceiling'), 'speed_auto': p.get('speed_auto'),
           'ceil_recorded': bool(A is not None and 'ceil_mode' in A), 'auto_recorded': bool(A is not None and 'sa_m' in A),
           'flags': []}
    if A is None or not (out['ceil_recorded'] or out['auto_recorded']) or not len(t):
        return out
    at = (np.asarray(A['timestamps_us'], np.int64) - int(t0)) / 1e6
    o = np.argsort(at, kind='stable')
    at = at[o]
    k = np.clip(np.searchsorted(t, at, side='right') - 1, 0, len(t) - 1)
    aif = np.asarray(ai, bool)[k] & (at >= t[0]) & (np.abs(at - t[k]) <= le.GAP_S)
    if 'dry_run' in A:
        aif &= ~np.asarray(A['dry_run'], bool)[o]
    n_ai = int(aif.sum())
    out['ai_frames'] = n_ai

    def col(name, default, dtype):
        return np.asarray(A[name], dtype)[o] if name in A else np.full(len(at), default, dtype)
    if out['ceil_recorded']:
        mode = col('ceil_mode', -1, np.int64)
        on = col('ceil_on', False, bool)
        applied = col('ceil_applied', False, bool)
        left = col('ceil_left', False, bool)
        ev_ = aif & (mode >= 0)
        n_ev = int(ev_.sum())
        frac = lambda m, d: _r(float(m.sum()) / d, 4) if d else None      # noqa: E731
        out['ceil'] = {'on_frames': int((aif & on).sum()), 'eval_frames': n_ev,
                       'applied_frames': int((aif & applied).sum()), 'applied_frac': frac(aif & applied, n_ai),
                       'weak_frac': frac(ev_ & (mode == 1), n_ev), 'neutral_frac': frac(ev_ & (mode == 2), n_ev),
                       'limit_frac': frac(ev_ & (mode >= 1), n_ev), 'left_frac': frac(ev_ & left, n_ev),
                       'failed_frames': int((aif & (mode == -2)).sum())}
        # 出し方の入れ替わり(続いた計算のフレームどうし)。いちばん多い 1 秒の回数
        idx = np.nonzero(ev_)[0]
        sw = []
        for a_, b_ in zip(idx[:-1], idx[1:]):
            if at[b_] - at[a_] <= le.GAP_S and mode[a_] != mode[b_]:
                sw.append(at[b_])
        sw = np.asarray(sw)
        mx = int(max((np.searchsorted(sw, x + 1.0, side='left') - j for j, x in enumerate(sw)), default=0))
        out['ceil']['switches'] = int(len(sw))
        out['ceil']['switch_max_per_s'] = mx
        if mx > CEIL_SWITCH_MAX_PER_S:
            out['flags'].append('ceil_switch_gt%d_per_s' % CEIL_SWITCH_MAX_PER_S)
        if out['ceil']['failed_frames']:
            out['flags'].append('ceil_failed')
    if out['auto_recorded']:
        m = col('sa_m', np.nan, np.float64)
        mfb = col('sa_m_fb', np.nan, np.float64)
        mff = col('sa_m_ff', np.nan, np.float64)
        gain = col('sa_gain', np.nan, np.float64)
        on = col('sa_on', False, bool)
        src = col('sa_src', 0, np.int64)
        vv = col('sa_v', np.nan, np.float64)
        evc = col('sa_event', 0, np.int64)
        lp90 = col('sa_lap_p90', np.nan, np.float64)
        ls = col('sa_lap_s', np.nan, np.float64)
        last = lambda x: _r(float(x[aif & np.isfinite(x)][-1]), 4) if (aif & np.isfinite(x)).any() else None   # noqa: E731
        close = np.nonzero(aif & np.isfinite(lp90))[0]
        laps = []
        for i in close:
            j = min(i + 1, len(at) - 1) if i + 1 < len(at) and aif[min(i + 1, len(at) - 1)] else i
            laps.append({'t_s': _r(at[i], 3), 'p90_mps': _r(lp90[i], 3), 'lap_s': _r(ls[i], 3),
                         'event': SA_EVENTS.get(int(evc[i]), str(int(evc[i]))),
                         'm_fb': _r(mfb[j], 4) if np.isfinite(mfb[j]) else None})
        evs = {}
        for i in np.nonzero(aif & (evc > 0))[0]:
            nm = SA_EVENTS.get(int(evc[i]), str(int(evc[i])))
            evs[nm] = evs.get(nm, 0) + 1
        fbs = [x['m_fb'] for x in laps if x['m_fb'] is not None]
        edge = [any(abs(v - e) <= 1e-3 for e in SA_MFB_EDGES) for v in fbs]
        d = [b - a for a, b in zip(fbs[:-1], fbs[1:]) if abs(b - a) > 1e-6]
        flips = [np.sign(x) != np.sign(y) for x, y in zip(d[:-1], d[1:])]
        srcs = src[aif]
        out['auto'] = {'on_frames': int((aif & on).sum()), 'm_end': last(m), 'm_ff_end': last(mff), 'm_fb_end': last(mfb),
                       'gain_end': last(gain), 'src': int(np.bincount(srcs[srcs >= 0]).argmax()) if (srcs >= 0).any() else None,
                       'v_first': _r(float(vv[aif & np.isfinite(vv)][0]), 3) if (aif & np.isfinite(vv)).any() else None,
                       'laps': laps, 'events': evs,
                       'laps_used': sum(1 for i in close if int(evc[i]) in SA_USED),
                       'laps_skipped': sum(1 for i in close if int(evc[i]) not in SA_USED),
                       'edge_run_max': _runmax(edge), 'flip_run_max': _runmax(flips),
                       'applied_frames': int((aif & col('sa_applied', False, bool)).sum())}
        if out['auto']['edge_run_max'] >= AUTO_EDGE_LAPS:
            out['flags'].append('auto_edge_%dlaps' % AUTO_EDGE_LAPS)
        if out['auto']['flip_run_max'] >= AUTO_FLIP_LAPS:
            out['flags'].append('auto_flip_%dlaps' % AUTO_FLIP_LAPS)
    return out


def _volt(x):
    """説明欄の電圧の文('8.31'・'8.31V')→ float。読めなければ None"""
    if x is None:
        return None
    m = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*[Vv]?\s*$', str(x))
    if not m:
        return None
    v = float(m.group(1))
    return v if 3.0 <= v <= 20.0 else None


def _scene_vbat(log, raw_dir):
    """シーン(scene.json)の stats.vbat(最初・最後・最小・中央)を記録の順につなぐ。無ければ None"""
    root = Path(raw_dir or config.RAW_DIR) / log
    rows = []
    for sj in sorted(root.glob('*/scene.json')) if root.is_dir() else []:
        try:
            st = (json.loads(sj.read_text(encoding='utf-8')).get('stats') or {}).get('vbat')
        except (OSError, ValueError):
            continue
        if not isinstance(st, dict):
            continue
        g = lambda *ks: next((float(st[k]) for k in ks if st.get(k) is not None), None)    # noqa: E731
        rows.append((g('first', 'start'), g('last', 'end'), g('min'), g('median', 'p50')))
    rows = [r for r in rows if any(x is not None for x in r)]
    if not rows:
        return None
    firsts = [r[0] for r in rows if r[0] is not None]
    lasts = [r[1] for r in rows if r[1] is not None]
    mins = [r[2] for r in rows if r[2] is not None]
    meds = [r[3] for r in rows if r[3] is not None]
    return {'start_v': _r(firsts[0], 3) if firsts else None, 'end_v': _r(lasts[-1], 3) if lasts else None,
            'min_v': _r(min(mins), 3) if mins else None, 'median_v': _r(float(np.median(meds)), 3) if meds else None}


def vbat_summary(V, t0, log=None, raw_dir=None, desc=None):
    """電圧の要約。上から順に: VEHICLE.npz の vbat_v(基板。最初・最後の 2 秒の中央値・1 秒ごとの中央値の最小)→
    scene.json の stats.vbat → 説明欄の V_before・V_after(Vb_before・Vb_after・Vb_min)。source = board / scene / description / None"""
    d = desc or {}
    out = {'source': None, 'start_v': None, 'end_v': None, 'min_v': None, 'median_v': None, 'n': 0,
           'V_before': _volt(d.get('V_before')), 'V_after': _volt(d.get('V_after')),
           'Vb_before': _volt(d.get('Vb_before')), 'Vb_after': _volt(d.get('Vb_after')), 'Vb_min': _volt(d.get('Vb_min'))}
    if V is not None and 'vbat_v' in V and t0 is not None:
        v = np.asarray(V['vbat_v'], np.float64)
        ok = np.isfinite(v) & (v > 0)
        if 'vbat_ok' in V:
            ok &= np.asarray(V['vbat_ok'], bool)
        if ok.any():
            tt = (np.asarray(V['timestamps_us'], np.int64)[ok] - int(t0)) / 1e6
            vv = v[ok]
            o = np.argsort(tt, kind='stable')
            tt, vv = tt[o], vv[o]
            b = np.floor((tt - tt[0])).astype(int)
            meds = [float(np.median(vv[b == i])) for i in np.unique(b) if (b == i).sum() >= 5]
            out.update(source='board', n=int(len(vv)), start_v=_r(float(np.median(vv[tt <= tt[0] + VBAT_EDGE_S])), 3),
                       end_v=_r(float(np.median(vv[tt >= tt[-1] - VBAT_EDGE_S])), 3),
                       min_v=_r(min(meds), 3) if meds else _r(float(np.min(vv)), 3), median_v=_r(float(np.median(vv)), 3))
            return out
    sc = _scene_vbat(log, raw_dir) if log else None
    if sc:
        out.update(source='scene', **sc)
        return out
    vb = out['Vb_before'] if out['Vb_before'] is not None else out['V_before']
    va = out['Vb_after'] if out['Vb_after'] is not None else out['V_after']
    if vb is not None or va is not None:
        out.update(source='description', start_v=vb, end_v=va, min_v=out['Vb_min'])
    return out


def _cap_codes():
    """記録の cap_zone・cap_mode の番号(jetformer/safety_caps.py の ZONES・CAP_MODES の並び)。読めなければ今の並び"""
    try:
        from jetformer import safety_caps as _sc
        return _sc.ZONES.index('R1_turn'), _sc.CAP_MODES.index('brake'), tuple(_sc.CAP_MODES.index(x) for x in ('cap', 'neutral', 'brake'))
    except Exception:
        return 3, 3, (1, 2, 3)


CAP_ZONE_R1_TURN, CAP_MODE_BRAKE, CAP_MODES_LIMITED = _cap_codes()   # 番号: −1 動かしていない・−2 計算の失敗
TURN_NO_BRAKE_S = 0.5                # R1 の旋回中 = 後から見た起点 ta → ta + 0.5 秒(cap_replay の TURN_NO_BRAKE と同じ)


def cap_counts(at, A, t, ai, start, end, anchors=(), gm_exit=()):
    """区間の速度上限の記録(S1 の AI_CMD.cap_*)の数。

    - ai_frames・on_frames・applied_frames・shadow_*: 数える区間の AI のフレーム(区間が無ければ 0)
    - on_brake_frames_in_r1_turn(**B+上限で 0 であること**、protocol の cap_brake_in_r1_turn): ON(cap_on)でブレーキを
      出したフレームのうち、時刻が R1 の旋回中(ジャイロから後で見た起点 ta → ta + 0.5 秒)のもの。ログの AI のフレーム全部
      (慣らしも入れる)。層の区間 cap_zone == R1_turn では数えない(層はブレーキを R1_app の中でしか出さないので、
      位相が遅れて実際の旋回中に出ても 0 になる。V-tools 4 章 2)
    - on_brake_frames_in_r1_turn_to_tg: 同じで、旋回中を ta → GM の出口 tg − 0.3 秒(無ければ ta + 3.5 秒)にしたもの
    - *_zone: 層の区間で数えたもの(参考。作りの上でいつも 0)"""
    o = np.argsort(at, kind='stable')
    at = at[o]
    k = np.clip(np.searchsorted(t, at, side='right') - 1, 0, len(t) - 1)
    ai_f = ai[k] & (at >= t[0])
    m = (at >= start) & (at < end) & ai_f if start is not None and end is not None else np.zeros(len(at), bool)
    mode = A['cap_mode'][o].astype(int)
    zone = A['cap_zone'][o].astype(int) if 'cap_zone' in A else np.full(len(at), -1)
    on = A['cap_on'][o].astype(bool) if 'cap_on' in A else np.zeros(len(at), bool)
    brake = mode == CAP_MODE_BRAKE
    onset = brake & ~np.concatenate([[False], brake[:-1]])
    applied = A['cap_applied'][o].astype(bool) if 'cap_applied' in A else np.zeros(len(at), bool)
    in_turn = np.zeros(len(at), bool)
    in_turn_tg = np.zeros(len(at), bool)
    for j, ta in enumerate(np.asarray(anchors, float)):
        tg = float(gm_exit[j]) if j < len(gm_exit) and np.isfinite(gm_exit[j]) else None
        in_turn |= (at >= ta) & (at <= ta + TURN_NO_BRAKE_S)
        in_turn_tg |= (at >= ta) & (at <= (tg - 0.3 if tg is not None else ta + 3.5))
    zturn = zone == CAP_ZONE_R1_TURN
    return {'ai_frames': int(m.sum()), 'ai_frames_log': int(ai_f.sum()), 'on_frames': int((m & on).sum()),
            'on_frames_log': int((ai_f & on).sum()), 'applied_frames': int((m & applied).sum()),
            'shadow_cap_frames': int((m & np.isin(mode, CAP_MODES_LIMITED)).sum()),
            'shadow_brake_starts': int((m & onset).sum()),
            'shadow_brake_frames_in_r1_turn': int((m & brake & in_turn).sum()),
            'on_brake_frames_in_r1_turn': int((ai_f & brake & on & in_turn).sum()),
            'on_brake_starts_in_r1_turn': int((ai_f & onset & on & in_turn).sum()),
            'on_brake_frames_in_r1_turn_to_tg': int((ai_f & brake & on & in_turn_tg).sum()),
            'on_brake_frames_in_r1_turn_window': int((m & brake & on & in_turn).sum()),
            'r1_anchors': int(len(anchors)),
            'shadow_brake_in_r1_turn_zone': int((m & onset & zturn).sum()),
            'on_brake_frames_in_r1_turn_zone': int((ai_f & brake & on & zturn).sum()),
            'failed_frames': int((m & (mode == -2)).sum())}


def protocol_laps(t, ai, dt, assist, ev, anchors, gm_exit, vt, verpm, K, budget, inf_ms=None):
    """plan/protocol.json の metrics.definitions の周(数える区間の中の AI の R1 の起点 ta_j、周 j = [ta_j − 0.5, 次の AI の
    起点 − 0.5))と、その中の出来事。返り値 (区間の情報, 周の行のリスト)"""
    n = len(t)

    def frame_at(tc):
        return int(np.clip(np.searchsorted(t, tc, side='right') - 1, 0, n - 1))
    ai_idx = [j for j, ta in enumerate(anchors) if t[0] <= ta <= t[-1] and ai[frame_at(ta)]]
    win = {'warmup_anchors': K, 'counted_ai_s': budget, 'ai_anchors': len(ai_idx)}
    if len(ai_idx) < max(K, 1):
        win.update(start_s=None, end_s=None, ai_s=0.0, note='AI の R1 の起点が K 個に足りない(慣らしで終わった)')
        return win, []
    start = float(anchors[ai_idx[K - 1]] - 0.5) if K >= 1 else float(t[0])
    m = t >= start
    cum = np.cumsum(np.where(m & ai, dt, 0.0))
    over = np.nonzero(cum >= budget)[0]
    end = float(t[over[0]] + dt[over[0]]) if len(over) else float(t[-1] + dt[-1])
    win.update(start_s=_r(start, 3), end_s=_r(end, 3), ai_s=_r(min(budget, float(cum[-1])), 2),
               budget_reached=bool(len(over)))
    inw = (t >= start) & (t < end)
    cs = [c for c in ev.get('collisions', []) if c.get('ai')]
    tks = ev.get('takeovers', [])
    hbs = ev.get('handbacks', [])
    recs = [r for r in ev.get('recoveries', []) if r.get('ai')]
    holds = [h for h in ev.get('holds', []) if h.get('ai')]
    rbs = ev.get('board_reboots', [])
    within = lambda x, a, b: a <= x < b      # noqa: E731
    rows = []
    sel_idx = [j for j in ai_idx[K - 1:] if anchors[j] - 0.5 < end]
    ai_time_cum = np.cumsum(np.where(inw & ai, dt, 0.0))
    for q, j in enumerate(sel_idx):
        ta = float(anchors[j])
        tg = float(gm_exit[j]) if j < len(gm_exit) and np.isfinite(gm_exit[j]) else None
        nxt_ai = sel_idx[q + 1] if q + 1 < len(sel_idx) else None
        a = ta - 0.5
        b_full = float(anchors[nxt_ai] - 0.5) if nxt_ai is not None else None
        truncated = b_full is None or b_full > end
        b = end if truncated else b_full
        if truncated and not (tg is not None and tg + 2.3 <= end):
            continue
        fm = (t >= a) & (t < b)
        col = [c for c in cs if within(c['t_s'], a, b)]
        tk = [x for x in tks if within(x['t_s'], a, b)]
        rb = [x for x in rbs if within(x['t_last_s'], a, b)]
        hi_r1 = tg - 0.3 if tg is not None else ta + 3.5
        c_r1 = [c for c in cs if c['loc'] == 'R1' and ta - 0.5 <= c['t_s'] <= hi_r1]
        c_gm = [c for c in cs if tg is not None and c['loc'] == 'GM' and ta <= c['t_s'] <= tg + 0.3]
        c_de = [c for c in cs if tg is not None and c['loc'] == 'DE' and tg - 0.3 <= c['t_s'] <= tg + 2.3]
        w = (vt >= ta - 0.4) & (vt <= ta)
        entry = float(np.max(-verpm[w])) if w.any() else None
        # ラップ: 次の起点(全部の起点の次)がこの周の次の AI の起点で、そのあいだがずっと AI・交代/戻し/HOLD/再起動なし
        lap_s = None
        if j + 1 < len(anchors) and nxt_ai == j + 1 and not truncated:
            tb = float(anchors[j + 1])
            mm = (t >= ta) & (t < tb)
            bad = (any(within(x['t_s'], ta, tb) for x in tks) or any(within(x['t_s'], ta, tb) for x in hbs)
                   or any(within(x['t_s'], ta, tb) or within(ta, x['t_s'], x['end_s'] + 1e-9) for x in ev.get('holds', []))
                   or any(within(x['t_last_s'], ta - 0.5, tb) for x in rbs))
            if mm.any() and ai[mm].all() and not bad:
                lap_s = tb - ta
        ai_only = bool(fm.any() and ai[fm].all())
        n_fail = sum(1 for x in tk if x.get('failure'))
        n_sp = sum(1 for x in tk if x.get('class') in ('stuck', 'preemptive'))
        n_rec = sum(1 for x in recs if within(x['t_s'], a, b))
        mid = 0.5 * (a + b)
        k_mid = int(np.clip(np.searchsorted(t, mid), 0, n - 1))
        row = {'ta': _r(ta, 3), 'tg': _r(tg, 3), 't_from': _r(a, 3), 't_to': _r(b, 3), 'truncated': bool(truncated),
               'reboot': bool(rb), 'n_reboots': len(rb), 'collisions': len(col), 'strong': sum(1 for c in col if c.get('strong')),
               'collision_locs': [c['loc'] for c in col],
               'tk_all': len(tk), 'tk_failure': n_fail, 'tk_stuck_preemptive': n_sp, 'recoveries': n_rec,
               'holds': sum(1 for h in holds if within(h['t_s'], a, b)),
               'incidents': len(col) + n_sp,
               'r1_coll': bool(c_r1), 'r1_strong': any(c.get('strong') for c in c_r1),
               'gm_coll': (bool(c_gm) if tg is not None else None), 'de_coll': (bool(c_de) if tg is not None else None),
               'entry_erpm': _r(entry, 0), 'lap_time_s': _r(lap_s, 3),
               'clean': bool(ai_only and not col and not n_fail and not n_rec),
               't_r1gm_s': _r(tg - ta, 3) if tg is not None else None,
               't_rest_s': _r(float(anchors[j + 1]) - tg, 3) if (tg is not None and nxt_ai == j + 1 and not truncated) else None,
               'ai_s_at_mid': _r(float(ai_time_cum[k_mid]), 2)}
        if inf_ms is not None:
            at, im = inf_ms
            mm2 = (at >= a) & (at < b) & np.isfinite(im)
            row['infer_ms'] = [round(float(x), 1) for x in im[mm2]] if mm2.any() else []
        rows.append(row)
    win['laps_rows'] = len(rows)
    win['holds_in_window'] = sum(1 for h in holds if start <= h['t_s'] < end)
    win['reboots_in_window'] = sum(1 for x in rbs if start <= x['t_last_s'] < end)
    # 区間の中の最後の交代(戻しが後に無ければ「走行の終わりの止め方」の候補。240 秒に届かないブロックで失敗に入ることがある。
    # V-tools 4 章 8。数え方は変えず、bench_report が deviations に書く)
    tk_in = [x for x in tks if start <= x['t_s'] < end]
    if tk_in:
        lt = tk_in[-1]
        win['last_takeover'] = {'t_s': lt['t_s'], 'class': lt.get('class'), 'failure': bool(lt.get('failure')),
                                'counted_as_incident': lt.get('class') in ('stuck', 'preemptive'),
                                'handback_after': any(h['t_s'] > lt['t_s'] for h in hbs),
                                'in_counted_lap': any(r['t_from'] <= lt['t_s'] < r['t_to'] for r in rows)}
    return win, rows


# ------------------------------------------------------------------ 手順の指標(protocol.json の metrics)
RATE_KEYS = {  # 名前: (分子の欄, 分母の欄)
    'incidents_per_lap': ('incidents', 'one'),
    'collisions_per_lap': ('collisions', 'one'),
    'strong_collisions_per_lap': ('strong', 'one'),
    'failure_takeovers_per_lap': ('tk_failure', 'one'),
    'takeovers_per_lap': ('tk_all', 'one'),
    'r1_strong_per_pass': ('r1_strong', 'one'),
    'r1_coll_per_pass': ('r1_coll', 'one'),
    'gm_coll_per_pass': ('gm_coll', 'has_tg'),
    'de_coll_per_pass': ('de_coll', 'has_tg'),
    'clean_lap_share': ('clean', 'one'),
}
BOOT_KEYS = list(RATE_KEYS) + ['lap_median_s']


def _lap_matrix(rows):
    cols = ['incidents', 'collisions', 'strong', 'tk_failure', 'tk_all', 'r1_strong', 'r1_coll', 'gm_coll', 'de_coll',
            'clean', 'one', 'has_tg']
    F = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for k, c in enumerate(cols):
            if c == 'one':
                F[i, k] = 1.0
            elif c == 'has_tg':
                F[i, k] = 1.0 if r.get('tg') is not None else 0.0
            else:
                F[i, k] = float(bool(r.get(c))) if isinstance(r.get(c), bool) else float(r.get(c) or 0)
    lap = np.array([r['lap_time_s'] if r.get('lap_time_s') is not None else np.nan for r in rows], float)
    return F, {c: k for k, c in enumerate(cols)}, lap


def _weighted_median(vals, C):
    """vals(並べ替え前)と、標本ごとの重み C(n_boot × n)から、繰り返しを入れた中央値(np.median と同じ)"""
    ok = np.isfinite(vals)
    if not ok.any():
        return np.full(C.shape[0], np.nan)
    v = vals[ok]
    W = C[:, ok]
    o = np.argsort(v, kind='stable')
    v, W = v[o], W[:, o]
    cs = np.cumsum(W, axis=1)
    tot = cs[:, -1]
    out = np.full(C.shape[0], np.nan)
    good = tot > 0
    lo_pos = np.where(tot % 2 == 1, tot // 2, tot // 2 - 1)
    hi_pos = tot // 2
    i_lo = np.argmax(cs > lo_pos[:, None], axis=1)
    i_hi = np.argmax(cs > hi_pos[:, None], axis=1)
    out[good] = 0.5 * (v[i_lo[good]] + v[i_hi[good]])
    return out


def boot_samples(runs_rows, seed, n_boot=BOOT_N, chunk=BOOT_CHUNK):
    """周を並びのまま chunk 周ずつのかたまりにし、走行ごとにかたまりを復元抽出する。返り値 {指標: 標本の配列}"""
    runs_rows = [[r for r in rows if not r.get('reboot')] for rows in runs_rows]
    rows_all = [r for rows in runs_rows for r in rows]
    if not rows_all:
        return {}
    F, ci, lap = _lap_matrix(rows_all)
    rng = np.random.default_rng(seed)
    C = np.zeros((n_boot, len(rows_all)))
    off = 0
    for rows in runs_rows:
        m = len(rows)
        if not m:
            continue
        chunk_of = np.arange(m) // chunk
        nc = int(chunk_of[-1]) + 1
        draws = rng.integers(0, nc, size=(n_boot, nc))
        Cc = np.zeros((n_boot, nc))
        np.add.at(Cc, (np.repeat(np.arange(n_boot), nc), draws.ravel()), 1.0)
        C[:, off:off + m] = Cc[:, chunk_of]
        off += m
    S = C @ F
    out = {}
    for name, (num, den) in RATE_KEYS.items():
        d = S[:, ci[den]]
        out[name] = np.where(d > 0, S[:, ci[num]] / np.where(d > 0, d, 1), np.nan)
    out['lap_median_s'] = _weighted_median(lap, C)
    return out


def _pct(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def reboot_basis(windows):
    """基板の再起動/4 分の元(protocol: 基板の再起動の数 ÷ (AI の時間 / 240 秒))。windows = ログごとの score_log の window。
    ブロックの時間(AI の走り始め → 数える区間の終わり。慣らし・最後の半端な周も入れる)の再起動の全部と、その中の AI の時間を足す。
    数える周が 0 本でも出す。元が無い(古い window・AI の時間 0)なら None(判定できない)"""
    ws = [w for w in windows if w and w.get('ai_s_block') is not None]
    if not ws:
        return None
    return {'reboots': int(sum(int(w.get('reboots_block') or 0) for w in ws)),
            'ai_s': float(sum(float(w.get('ai_s_block') or 0.0) for w in ws)),
            'in_window': int(sum(int(w.get('reboots_in_window') or 0) for w in ws)),
            'all_log': int(sum(int(w.get('reboots_all_log') or 0) for w in ws)), 'logs': len(ws)}


def _reboot_metrics(out, rows_all, ai_s, basis):
    rb_laps = int(sum(r.get('n_reboots', int(bool(r.get('reboot')))) for r in rows_all))
    if basis is not None:
        rb, den, kind = basis['reboots'], basis['ai_s'], 'ai_all'
        out['reboots_in_window'] = basis.get('in_window')
        out['reboots_all_log'] = basis.get('all_log')
    else:                               # 古い呼び方(周の行だけ): 数えた周の中の再起動 ÷ 区間の AI の時間
        rb, den, kind = rb_laps, ai_s, 'laps'
    out['reboots'] = rb
    out['reboots_in_laps'] = rb_laps
    out['reboots_ai_s'] = _r(den, 1) if den else den
    out['reboots_basis'] = kind
    out['reboots_per_4min'] = _r(rb / (den / 240.0), 3) if den else None


def protocol_metrics(runs_rows, ai_s=None, seed=20261023, n_boot=BOOT_N, with_samples=False, reboot_basis=None):
    """周の行(走行ごとのリスト)から、手順の指標と 90% 区間(かたまりのブートストラップ)。
    reboot_basis(reboot_basis() の返り値)を渡すと、再起動/4 分はブロックの AI の時間の全部で数える(渡さなければ周の行だけ)"""
    rows_all = [r for rows in runs_rows for r in rows]
    counted = [r for r in rows_all if not r.get('reboot')]
    n = len(counted)
    out = {'laps_counted': n, 'reboot_laps': sum(1 for r in rows_all if r.get('reboot')), 'enough_laps': n >= 12}
    _reboot_metrics(out, rows_all, ai_s, reboot_basis)       # 周が 0 本でも出す
    if not n:
        return (out, {}) if with_samples else out
    F, ci, lap = _lap_matrix(counted)
    tot = F.sum(axis=0)
    for name, (num, den) in RATE_KEYS.items():
        k, d = tot[ci[num]], tot[ci[den]]
        out[name] = {'value': _r(k / d, 4) if d else None, 'k': int(k), 'n': int(d)}
        if d and (name.endswith('per_pass') or name == 'clean_lap_share'):
            # 正確な二項(Clopper-Pearson)の 90% 区間も並べる(かたまりのブートストラップは、かたまりが少なく出来事が周に
            # 散っていると狭く出る。V-tools 4 章 10。順位・判定は protocol どおりブートストラップの区間を使う)
            lo, hi = cp_ci(int(k), int(d))
            out[name]['ci90_exact'] = [_r(lo, 4), _r(hi, 4)]
        if k == 0 and d:
            # 周あたりの率はポアソンの正確な片側 95% の上限 3.0 ÷ 周(protocol)。通過・周の割合は二項の正確な上限 1 − 0.05^(1/n)
            out[name]['upper95_zero'] = _r(1 - 0.05 ** (1.0 / d) if (name.endswith('per_pass') or name == 'clean_lap_share') else 3.0 / d, 4)
    lt = lap[np.isfinite(lap)]
    out['lap_median_s'] = {'value': _r(np.median(lt), 3) if len(lt) else None, 'n': int(len(lt))}
    out['lap_best_s'] = _r(lt.min(), 3) if len(lt) else None
    clean_t = [r['lap_time_s'] for r in counted if r['clean'] and r.get('lap_time_s') is not None]
    out['clean_laps'] = sum(1 for r in counted if r['clean'])
    out['clean_lap_median_s'] = _r(np.median(clean_t), 3) if len(clean_t) >= 5 else None
    out['t_r1gm_median_s'] = _med([r['t_r1gm_s'] for r in counted])
    out['t_rest_median_s'] = _med([r['t_rest_s'] for r in counted])
    ent = [r['entry_erpm'] for r in counted if r.get('entry_erpm') is not None]
    out['r1_entry_erpm_median'] = _med(ent)
    out['r1_entry_ge6200_share'] = _r(np.mean(np.array(ent) >= 6200), 3) if ent else None
    out['holds'] = sum(r.get('holds', 0) for r in rows_all)
    im = [x for r in rows_all for x in r.get('infer_ms', [])]
    out['infer_ms_p99'] = _r(np.percentile(im, 99), 2) if im else None
    out['collisions_by_loc'] = {k: sum(r['collision_locs'].count(k) for r in counted) for k in LOCS}
    samples = boot_samples(runs_rows, seed, n_boot)
    for name in BOOT_KEYS:
        x = samples.get(name)
        if x is not None and isinstance(out.get(name), dict):
            out[name]['ci90'] = [_r(_pct(x, 5), 4), _r(_pct(x, 95), 4)]
    out['boot'] = {'n': n_boot, 'chunk_laps': BOOT_CHUNK, 'seed': seed}
    if with_samples:
        return out, samples
    return out


def sum_caps(caps):
    """ログごとの cap_counts を足す(欄の無い古い結果は 0 として足す)。無ければ None"""
    caps = [c for c in caps if c]
    if not caps:
        return None
    keys = []
    for c in caps:
        keys += [k for k in c if k not in keys]
    return {k: sum(int(c.get(k) or 0) for c in caps) for k in keys}


def diff_ci(cand_runs, ref_runs, seed_c, seed_r, n_boot=BOOT_N, lam=LAMBDA_S):
    """候補 − 基準 の差(別々に抽出して差を取る)と、点数 S = Δ失敗/周 + λ × Δラップの中央値 の 90% 区間"""
    pc, sc_ = protocol_metrics(cand_runs, seed=seed_c, n_boot=n_boot, with_samples=True)
    pr, sr = protocol_metrics(ref_runs, seed=seed_r, n_boot=n_boot, with_samples=True)
    out = {}
    if not pc.get('laps_counted') or not pr.get('laps_counted'):
        return out
    for name in BOOT_KEYS:
        a, b = pc.get(name, {}).get('value'), pr.get(name, {}).get('value')
        d = sc_[name] - sr[name] if name in sc_ and name in sr else None
        out['d_' + name] = {'value': _r(a - b, 4) if a is not None and b is not None else None,
                            'ci90': [_r(_pct(d, 5), 4), _r(_pct(d, 95), 4)] if d is not None else [None, None]}
    di, dl = out['d_incidents_per_lap'], out['d_lap_median_s']
    if di['value'] is not None and dl['value'] is not None:
        S = (sc_['incidents_per_lap'] - sr['incidents_per_lap']) + lam * (sc_['lap_median_s'] - sr['lap_median_s'])
        out['S'] = {'value': _r(di['value'] + lam * dl['value'], 4), 'lambda': lam, 'ci90': [_r(_pct(S, 5), 4), _r(_pct(S, 95), 4)]}
    return out


# ------------------------------------------------------------------ まとめ
def day_seed(scores):
    """ブートストラップの種 = その日(YYYYMMDD の整数。protocol.json の metrics.ci.seed と同じ)"""
    for s in scores:
        d = (s.get('desc') or {}).get('day')
        if d:
            try:
                return int(str(d).replace('-', ''))
            except ValueError:
                pass
    try:
        return int(scores[0]['log'][:8])
    except (IndexError, KeyError, ValueError):
        return 20261023


def aggregate(scores, rule='label', seed=None, n_boot=BOOT_N):
    """ログの成績を足し合わせて率を出す(分母と件数を足してから割る)。protocol = 手順の数える区間の指標"""
    S = [s for s in scores if 'error' not in s and s.get('ai_s')]
    out = {'logs': [s['log'] for s in S], 'n_logs': len(S)}
    if not S:
        return out
    seed = day_seed(S) if seed is None else seed
    out['protocol'] = protocol_metrics([s.get('laps_protocol', []) for s in S],
                                       ai_s=sum((s.get('window') or {}).get('ai_s') or 0 for s in S), seed=seed, n_boot=n_boot,
                                       reboot_basis=reboot_basis([s.get('window') for s in S]))
    out['windows'] = {s['log']: {k: v for k, v in (s.get('window') or {}).items()} for s in S}
    cap = sum_caps([(s.get('window') or {}).get('cap') for s in S])
    if cap:
        out['protocol']['cap'] = cap
        out['protocol']['cap_brake_in_r1_turn'] = cap['on_brake_frames_in_r1_turn']
    ai_s = sum(s['ai_s'] for s in S)
    lap_eq = sum(s['lap_equiv'] or 0 for s in S)
    out['ai_min'] = _r(ai_s / 60, 2)
    out['lap_equiv'] = _r(lap_eq, 2)
    col = {k: sum(s['collisions'][rule][k] for s in S) for k in ('main', 'restart', 'recovery')}
    loc = {k: sum(s['collisions'][rule]['by_loc'][k] for s in S) for k in LOCS}
    out['collision_rule'] = rule
    out['collisions_per_lap'] = rate(col['main'], lap_eq)
    out['collisions_by_loc'] = loc
    out['collisions_by_loc_per_lap'] = {k: _r(v / lap_eq, 3) if lap_eq else None for k, v in loc.items()}
    out['collisions_restart'] = col['restart']
    out['collisions_recovery'] = col['recovery']
    out['collisions_cmap_by_loc'] = {k: sum(s['collisions']['cmap']['by_loc'][k] for s in S) for k in LOCS}
    out['collisions_label_by_loc'] = {k: sum(s['collisions']['label']['by_loc'][k] for s in S) for k in LOCS}
    if rule == 'label':
        out['strong_collisions_per_lap'] = rate(sum(s['collisions']['label']['strong'] for s in S), lap_eq)
    P = [p for s in S for p in s['passes'] if p['ai']]
    Pg = [p for p in P if p['gm_exit']]
    out['passes_ai'] = len(P)
    out['r1_strong_per_pass'] = prop(sum(p['r1_strong'] for p in P), len(P))
    out['r1_collision_per_pass'] = prop(sum(p['r1_collision'] for p in P), len(P))
    out['gm_collision_per_pass'] = prop(sum(p['gm_collision'] for p in Pg), len(Pg))
    out['de_collision_per_pass'] = prop(sum(p['de_collision'] for p in Pg), len(Pg))
    ent = [p['entry_erpm'] for p in P if p['entry_erpm'] is not None]
    out['r1_entry_erpm'] = {'median': _med(ent), 'p90': _r(float(np.percentile(ent, 90)), 0) if ent else None,
                            'share_ge_6200': _r(np.mean(np.array(ent) >= 6200), 3) if ent else None}
    out['r1_entry_bins'] = {}
    for lo, hi, name in ENTRY_BINS:
        q = [p for p in P if p['entry_erpm'] is not None and lo <= p['entry_erpm'] < hi]
        out['r1_entry_bins'][name] = prop(sum(p['r1_strong'] for p in q), len(q))
    laps = [lp for s in S for lp in s['laps']]
    clean = [lp for lp in laps if lp['clean']]
    out['laps'] = {'ai_laps': len(laps), 'clean_laps': len(clean),
                   'clean_median_s': _med([lp['lap_s'] for lp in clean]), 'clean_best_s': _min([lp['lap_s'] for lp in clean]),
                   'all_median_s': _med([lp['lap_s'] for lp in laps])}
    sect = {}
    for name in le.LOC_NAMES[1:]:
        sect[name] = _med([lp['sections_s'].get(name) for lp in clean])
    for g, members in SECTION_GROUPS.items():
        sect[g] = _med([sum(lp['sections_s'].get(m) or 0 for m in members) for lp in clean])
    out['section_s_clean_median'] = sect
    lct = [x for s in S for x in s['lapcounter']['clean_times_s']]
    out['lapcounter'] = {'ai_laps': sum(s['lapcounter']['ai_laps'] for s in S), 'clean_laps': len(lct),
                         'clean_median_s': _med(lct), 'clean_best_s': _min(lct)}
    out['takeovers_failure_per_lap'] = rate(sum(s['takeovers']['failure'] for s in S), lap_eq)
    out['takeovers_all_per_lap'] = rate(sum(s['takeovers']['all'] for s in S), lap_eq)
    out['takeovers_failure_by_loc'] = {k: sum(s['takeovers']['by_loc'][k] for s in S) for k in LOCS}
    rb = sum(s['board_reboots']['all'] for s in S)
    out['board_reboots'] = {'all': rb, 'ai': sum(s['board_reboots']['ai'] for s in S),
                            'per_ai_min': _r(rb / (ai_s / 60), 3) if ai_s else None}
    out['recoveries'] = sum(s['recoveries'] or 0 for s in S)
    out['holds_ai'] = sum(s['holds_ai'] or 0 for s in S)
    out['guard_warnings_ai_per_min'] = _r(sum(s['guard_warnings_ai'] or 0 for s in S) / (ai_s / 60), 2) if ai_s else None
    return out


def group_scores(scores, rule='label', n_boot=BOOT_N):
    def grp(keyf):
        d = {}
        for s in scores:
            k = keyf(s)
            if k is None:
                continue
            d.setdefault(str(k), []).append(s)
        return {k: aggregate(v, rule, n_boot=n_boot) for k, v in sorted(d.items())}
    return {'all': aggregate(scores, rule, n_boot=n_boot),
            'by_cand': grp(lambda s: s.get('cand')),
            'by_battery': grp(lambda s: s['desc'].get('battery')),
            'by_block': grp(lambda s: ('%s#%s' % (s['desc'].get('day') or s['log'][:8], s['desc'].get('block')))
                            if s['desc'].get('block') is not None else None),
            'by_log': {s['log']: aggregate([s], rule, n_boot=n_boot) for s in scores}}


def build(logs, labels_dir=None, raw_dir=None, assign=None, bench=None, rule='label', windows=None, n_boot=BOOT_N,
          warmup=None, counted_s=None):
    """windows: {ログ: (K, 数える AI の秒)}。無いログは warmup・counted_s(既定 K=2・240 秒)"""
    assign = assign or {}
    windows = windows or {}
    scores = []
    for lg in logs:
        K, budget = windows.get(lg, (warmup, counted_s))
        try:
            scores.append(score_log(lg, labels_dir, raw_dir, assign.get(lg), bench, K, budget))
        except Exception as e:  # 壊れたログでも、ほかのログは出す
            scores.append({'log': lg, 'error': repr(e), 'desc': {}, 'cand': None})
    return {'schema': SCHEMA, 'created_at': datetime.now().isoformat(timespec='seconds'), 'tool': 'tools/section_scorecard.py',
            'collision_rule': rule, 'restart_s': RESTART_S, 'conf': CONF,
            'definitions': __doc__.split('出すものは 2 つ:')[1].strip() if '出すものは 2 つ:' in __doc__ else '',
            'per_log': {s['log']: s for s in scores}, 'groups': group_scores(scores, rule, n_boot)}


# ------------------------------------------------------------------ Markdown
def _ci(x):
    if not x or x.get('per') is None:
        return '—'
    return '%.2f [%.2f–%.2f]' % (x['per'], x['ci90'][0], x['ci90'][1])


def _pp(x):
    if not x or not x.get('n'):
        return '—'
    return '%d/%d(%.0f%%)' % (x['k'], x['n'], 100 * x['p'])


def _f(x, fmt='%.2f'):
    return '—' if x is None else fmt % x


def _pm(x, pct=False, fmt='%.2f'):
    """手順の指標 {'value', 'ci90'} → 文字"""
    if not isinstance(x, dict) or x.get('value') is None:
        return '—'
    v = x['value'] * (100 if pct else 1)
    s = (fmt % v) + ('%' if pct else '')
    c = x.get('ci90')
    if c and c[0] is not None:
        s += ' [%s–%s]' % (fmt % (c[0] * (100 if pct else 1)), fmt % (c[1] * (100 if pct else 1)))
    if 'k' in x and pct:
        s += '(%d/%d)' % (x['k'], x['n'])
    ce = x.get('ci90_exact')
    if ce and ce[0] is not None:
        s += '・正確な二項 [%s–%s]' % (fmt % (ce[0] * (100 if pct else 1)), fmt % (ce[1] * (100 if pct else 1)))
    if x.get('upper95_zero') is not None:
        s += '(0 件。片側 95%% の上限 %s)' % (fmt % (x['upper95_zero'] * (100 if pct else 1)) + ('%' if pct else ''))
    return s


def protocol_table(groups, title_key):
    rows = ['| %s | 数えた周 | 失敗/周 [90%%] | 衝突/周 [90%%] | R1 強い衝突/通過 | GM 衝突/通過 | DE 衝突/通過 | '
            'ラップ中央値 秒 [90%%] | きれいな周 | R1 進入 中央値 eRPM | 再起動/4 分 |' % title_key,
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for k, g in groups.items():
        P = g.get('protocol') or {}
        if not P.get('laps_counted'):
            rows.append('| %s | 0 | — | — | — | — | — | — | — | — | %s |' % (k, _rb(P)))
            continue
        rows.append('| %s | %d%s | %s | %s | %s | %s | %s | %s | %d | %s | %s |' % (
            k, P['laps_counted'], '' if P['enough_laps'] else '(足りない)', _pm(P['incidents_per_lap']),
            _pm(P['collisions_per_lap']), _pm(P['r1_strong_per_pass'], True, '%.0f'), _pm(P['gm_coll_per_pass'], True, '%.0f'),
            _pm(P['de_coll_per_pass'], True, '%.0f'), _pm(P['lap_median_s']), P['clean_laps'],
            _f(P['r1_entry_erpm_median'], '%.0f'), _rb(P)))
    return '\n'.join(rows)


def _rb(P):
    """再起動/4 分(回数・元の AI の時間)。元が無ければ「判定できない」"""
    if P.get('reboots_per_4min') is None:
        return '判定できない' if P.get('reboots') is not None or P else '—'
    return '%.2f(%d 回 / AI %.0f 秒)' % (P['reboots_per_4min'], P.get('reboots') or 0, P.get('reboots_ai_s') or 0)


def group_table(groups, title_key):
    rows = ['| %s | AI 分 | 周相当 | 衝突/周 [90%%] | R1 強い衝突/通過 | GM 衝突/通過 | DE 衝突/通過 | R1 進入 中央値 eRPM | '
            'きれいな周 | ラップ中央値 秒 | 失敗の交代/周 [90%%] | 基板の再起動 |' % title_key,
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, g in groups.items():
        if not g.get('n_logs'):
            continue
        rows.append('| %s | %s | %s | %s | %s | %s | %s | %s | %d | %s | %s | %d |' % (
            k, _f(g['ai_min'], '%.1f'), _f(g['lap_equiv'], '%.1f'), _ci(g['collisions_per_lap']), _pp(g['r1_strong_per_pass']),
            _pp(g['gm_collision_per_pass']), _pp(g['de_collision_per_pass']), _f(g['r1_entry_erpm']['median'], '%.0f'),
            g['laps']['clean_laps'], _f(g['laps']['clean_median_s']), _ci(g['takeovers_failure_per_lap']),
            g['board_reboots']['all']))
    return '\n'.join(rows)


def to_markdown(sc, title='区間の成績表'):
    g = sc['groups']
    A = g['all']
    L = ['# %s' % title, '',
         '- 作成: %s(tools/section_scorecard.py、読み取りだけ)' % sc['created_at'],
         '- 1〜4 章の衝突の規則: **%s**(label = 水平 2.5G・0.6 秒でまとめる。cmap = collision_map.md と同じ |‖a‖−1G| 2.5G)。'
         '1〜4 章では立て直し中(失敗の交代 → 戻して %.1f 秒以内)を数えない。0 章(手順の指標)は全部数える' % (sc['collision_rule'], sc['restart_s']),
         '- 1〜4 章の率の [ ] はポアソンの正確な 90% 区間、通過あたりは Clopper-Pearson の 90% 区間。分母の周相当 = AI が走っている間のジャイロの向き ÷ 360°',
         '- ログ: %s' % ', '.join(A.get('logs', [])), '']
    if not A.get('n_logs'):
        L.append('AI の走行がありません。')
        return '\n'.join(L) + '\n'
    P = A.get('protocol') or {}
    L += ['## 0. 手順の指標(数える区間の周。plan/protocol.json の metrics)', '',
          '- 数える区間: K 番目の AI の R1 の起点の 0.5 秒前から、AI の時間 %s 秒まで。周 = 区間の中の AI の R1 の起点。'
          '基板の再起動のあった周は除く。[ ] は 3 周ずつのかたまりのブートストラップ %s 回の 90%% 区間(種 %s)' % (
              ', '.join(sorted({'%s' % (w.get('counted_ai_s')) for w in A.get('windows', {}).values()})),
              (P.get('boot') or {}).get('n'), (P.get('boot') or {}).get('seed')), '']
    if P.get('laps_counted'):
        L += ['| 指標 | 値 |', '|---|---|',
              '| 数えた周 | %d%s(再起動の周 %d) |' % (P['laps_counted'], '' if P['enough_laps'] else '(12 周に足りない)', P['reboot_laps']),
              '| **失敗/周**(衝突 + 衝突のあとでない失敗の交代) | %s |' % _pm(P['incidents_per_lap']),
              '| 衝突/周 | %s |' % _pm(P['collisions_per_lap']),
              '| 強い衝突/周 | %s |' % _pm(P['strong_collisions_per_lap']),
              '| R1 の強い衝突/通過 | %s |' % _pm(P['r1_strong_per_pass'], True, '%.0f'),
              '| R1 の衝突/通過 | %s |' % _pm(P['r1_coll_per_pass'], True, '%.0f'),
              '| GM の衝突/通過 | %s |' % _pm(P['gm_coll_per_pass'], True, '%.0f'),
              '| DE の衝突/通過 | %s |' % _pm(P['de_coll_per_pass'], True, '%.0f'),
              '| 失敗の交代/周 | %s |' % _pm(P['failure_takeovers_per_lap']),
              '| 交代(全部)/周 | %s |' % _pm(P['takeovers_per_lap']),
              '| ラップの中央値 | %s 秒(%d 周)・ベスト %s 秒 |' % (_pm(P['lap_median_s']), P['lap_median_s']['n'], _f(P['lap_best_s'])),
              '| きれいな周 | %d(%s)・中央値 %s 秒 |' % (P['clean_laps'], _pm(P['clean_lap_share'], True, '%.0f'), _f(P['clean_lap_median_s'])),
              '| 区間の時間 R1→GM 出口・残り(中央値) | %s 秒・%s 秒 |' % (_f(P['t_r1gm_median_s']), _f(P['t_rest_median_s'])),
              '| R1 の進入 eRPM(中央値・6,200 以上) | %s・%s |' % (_f(P['r1_entry_erpm_median'], '%.0f'),
                                                       _f((P['r1_entry_ge6200_share'] or 0) * 100, '%.0f%%')),
              '| 基板の再起動/4 分(AI の走り始め → 数える区間の終わり。慣らし・半端な周も入れる) | %s(数える区間の中 %s 回・数えた周の中 %s 回・ログ全体 %s 回) |' % (
                  _rb(P), P.get('reboots_in_window', '—'), P.get('reboots_in_laps', '—'), P.get('reboots_all_log', '—')),
              '| HOLD | %d |' % P['holds'],
              '| 推論の時間 p99 | %s ms |' % _f(P['infer_ms_p99']), '']
    else:
        L += ['数える周がありません(慣らしで終わった、または AI の R1 の起点が無い)。', '',
              '- 基板の再起動/4 分(ブロックの AI の時間の全部): %s' % _rb(P), '']
    L += ['## 1. ログ全体(参考。10/10 の数え方: 周相当で割る)', '',
          '| 項目 | 値 |', '|---|---|',
          '| AI の時間 | %.1f 分 |' % A['ai_min'],
          '| 周相当 | %.2f 周 |' % A['lap_equiv'],
          '| 衝突/周 | %s(%d 件) |' % (_ci(A['collisions_per_lap']), A['collisions_per_lap']['n']),
          '| 立て直し中の衝突(数えない) | %d 件 |' % A['collisions_restart'],
          '| R1 の強い衝突/通過 | %s |' % _pp(A['r1_strong_per_pass']),
          '| GM の衝突/通過 | %s |' % _pp(A['gm_collision_per_pass']),
          '| DE の衝突/通過 | %s |' % _pp(A['de_collision_per_pass']),
          '| R1 の進入(中央値・p90) | %s・%s eRPM(6,200 以上 %s) |' % (
              _f(A['r1_entry_erpm']['median'], '%.0f'), _f(A['r1_entry_erpm']['p90'], '%.0f'),
              _f((A['r1_entry_erpm']['share_ge_6200'] or 0) * 100, '%.0f%%')),
          '| きれいな周(R1 起点) | %d / %d 周、中央値 %s 秒・ベスト %s 秒 |' % (
              A['laps']['clean_laps'], A['laps']['ai_laps'], _f(A['laps']['clean_median_s']), _f(A['laps']['clean_best_s'])),
          '| LapCounter のきれいな周 | %d 周、中央値 %s 秒・ベスト %s 秒 |' % (
              A['lapcounter']['clean_laps'], _f(A['lapcounter']['clean_median_s']), _f(A['lapcounter']['clean_best_s'])),
          '| 失敗による交代/周 | %s(%d 回) |' % (_ci(A['takeovers_failure_per_lap']), A['takeovers_failure_per_lap']['n']),
          '| 交代(全部)/周 | %s |' % _ci(A['takeovers_all_per_lap']),
          '| 基板の再起動 | %d 回(AI 側 %d、%s 回/AI 分) |' % (A['board_reboots']['all'], A['board_reboots']['ai'],
                                                     _f(A['board_reboots']['per_ai_min'], '%.2f')),
          '', '## 2. 場所ごとの衝突', '',
          '| 場所 | %s(数えたもの) | 1 周あたり | label の規則 | cmap の規則 |' % sc['collision_rule'], '|---|---|---|---|---|']
    for k in LOCS:
        L.append('| %s | %d | %s | %d | %d |' % (k, A['collisions_by_loc'][k], _f(A['collisions_by_loc_per_lap'][k], '%.3f'),
                                               A['collisions_label_by_loc'][k], A['collisions_cmap_by_loc'][k]))
    L += ['', '## 3. R1 の進入の速さと強い衝突(AI の通過)', '', '| 進入 eRPM | 強い衝突/通過 | 90% 区間 |', '|---|---|---|']
    for name, x in A['r1_entry_bins'].items():
        L.append('| %s | %s | %s |' % (name, x['text'], '—' if x['ci90'][0] is None else '%.2f–%.2f' % tuple(x['ci90'])))
    L += ['', '## 4. 区間の時間(きれいな周の中央値、秒)', '', '| 区間 | 秒 |', '|---|---|']
    for k, v in A['section_s_clean_median'].items():
        L.append('| %s | %s |' % (k, _f(v)))
    L += per_log_ctrl_table(sc['per_log'])
    for key, title_ in (('by_cand', '候補'), ('by_block', 'ブロック'), ('by_battery', '電池'), ('by_log', 'ログ')):
        if g.get(key):
            L += ['', '## %s ごと(手順の指標)' % title_, '', protocol_table(g[key], title_),
                  '', '<details><summary>ログ全体の数え方(参考)</summary>', '', group_table(g[key], title_), '', '</details>']
    errs = [s for s in sc['per_log'].values() if 'error' in s]
    if errs:
        L += ['', '## 読めなかったログ', ''] + ['- %s: %s' % (s['log'], s['error']) for s in errs]
    probs = [s for s in sc['per_log'].values() if s.get('problems')]
    comp = [s['log'] for s in sc['per_log'].values() if s.get('labels') == 'computed']
    if probs or comp:
        L += ['', '## 飛ばしたもの', '']
        L += ['- %s: 壊れた・無いファイルを飛ばした %d 件(%s)' % (s['log'], len(s['problems']), '・'.join(s['problems'][:5])
                                                     + (' ほか' if len(s['problems']) > 5 else '')) for s in probs]
        if comp:
            L.append('- ラベルのファイルが無く、その場で作った: %s' % '・'.join(comp))
    return '\n'.join(L) + '\n'


def _onoff(x):
    return '—' if x is None else ('ON' if int(x) else 'OFF')


def ctrl_cells(s):
    """走行ごとの表の「天井」「自動合わせ」「電圧」「印」の文(scorecard・bench_report で使う)"""
    sp = s.get('speed_ctrl') or {}
    c, a = sp.get('ceil') or {}, sp.get('auto') or {}
    if sp.get('ceil_recorded'):
        ceil = '%s・変えた %s・影 弱め %s/中立 %s・左 %s' % (
            _onoff(sp.get('speed_ceiling')), _f((c.get('applied_frac') or 0) * 100, '%.1f%%'),
            _f((c.get('weak_frac') or 0) * 100, '%.1f%%'), _f((c.get('neutral_frac') or 0) * 100, '%.1f%%'),
            _f((c.get('left_frac') or 0) * 100, '%.0f%%'))
    else:
        ceil = '%s(記録なし)' % _onoff(sp.get('speed_ceiling'))
    if sp.get('auto_recorded'):
        auto = '%s・m_fb %s・m %s・周 %d(外した %d)' % (_onoff(sp.get('speed_auto')), _f(a.get('m_fb_end'), '%.3f'),
                                                   _f(a.get('m_end'), '%.3f'), a.get('laps_used', 0), a.get('laps_skipped', 0))
    else:
        auto = '%s(記録なし)' % _onoff(sp.get('speed_auto'))
    vb = s.get('vbat') or {}
    volt = '—' if not vb.get('source') else '%s / %s / %s(%s)' % (
        _f(vb.get('start_v')), _f(vb.get('end_v')), _f(vb.get('min_v')),
        {'board': '基板', 'scene': 'シーン', 'description': '説明欄'}.get(vb['source'], vb['source']))
    flags = '・'.join(sp.get('flags') or []) or '—'
    return ceil, auto, volt, flags


def per_log_ctrl_table(per_log):
    """5 章: 外した R1 の起点・速さの天井・自動合わせ・電圧(走行ごと。どれかがある走行だけ)"""
    rows = [s for s in per_log.values() if 'error' not in s and (
        s.get('r1_anchors_rejected') or (s.get('speed_ctrl') or {}).get('ceil_recorded')
        or (s.get('speed_ctrl') or {}).get('auto_recorded') or (s.get('vbat') or {}).get('source'))]
    if not rows:
        return ['', '## 5. 外した R1 の起点・速さの制御・電圧', '', '- 外した起点・天井と自動合わせの記録・電圧のある走行はありません。']
    L = ['', '## 5. 外した R1 の起点・速さの制御・電圧(走行ごと)', '',
         '- 外した起点: 最後に使った起点からの向きの変化 < 270°(16 秒以内・ずっと AI。impact = 起点の −0.05〜+0.25 秒に水平 4G 以上の衝突も'
         'あった。heading_next = 次の起点とつじつまが合わない前の起点。label_events.reject_false_anchors)。通過・周・場所は外したあとの起点で数えた。',
         '- 天井: 設定・実際に出力を変えたフレームの割合・影の判定で弱める/中立にする割合・左旋回の割合(AI のフレーム)。'
         '自動合わせ: 設定・最後の m_fb と m・使った周(外した周)。電圧: 始め / 終わり / 最小 [V](元)。',
         '', '| ログ | 外した起点(時刻・理由・向き・g) | 天井 | 自動合わせ | 電圧 | 印 |', '|---|---|---|---|---|---|']
    for s in sorted(rows, key=lambda x: x['log']):
        rj = '・'.join('%.2f 秒 %s %s° %s' % (r['t_s'], r['reason'], _f(r.get('dpsi_deg'), '%.0f'),
                                            '%.1fG' % r['g'] if r.get('g') is not None else '')
                      for r in s.get('r1_anchors_rejected') or []) or '—'
        L.append('| %s | %s | %s | %s | %s | %s |' % ((s['log'], rj) + ctrl_cells(s)))
    return L


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write(sc, out_dir, title='区間の成績表'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / 'scorecard.json.tmp'
    tmp.write_text(json.dumps(sc, ensure_ascii=False, indent=1, default=_json_default), encoding='utf-8')
    os.replace(tmp, out_dir / 'scorecard.json')
    (out_dir / 'scorecard.md').write_text(to_markdown(sc, title), encoding='utf-8')
    return out_dir / 'scorecard.json', out_dir / 'scorecard.md'


def day_logs(day, raw_dir=None, ai_only=True, labels_dir=None):
    """その日(YYYYMMDD)のログ。ai_only なら AI_CMD のあるものだけ"""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    day = day.replace('-', '')
    logs = sorted(d.name for d in raw_dir.glob(day + '-*') if (d / 'log.json').exists())
    if not ai_only:
        return logs
    keep = []
    for lg in logs:
        if any((s / 'AI_CMD.npz').exists() for s in (raw_dir / lg).iterdir() if s.is_dir()):
            keep.append(lg)
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='*', help='ログ名')
    ap.add_argument('--day', help='この日(YYYYMMDD か YYYY-MM-DD)の AI のログ全部')
    ap.add_argument('--raw', default=str(config.RAW_DIR))
    ap.add_argument('--labels', default=str(config.DATA_DIR / 'labels'), help='ラベルの場所(無いログはその場で作る)')
    ap.add_argument('--bench', default=None, help='data/bench_candidates.json(説明欄に cand= が無いとき run で照合)')
    ap.add_argument('--assign', default=None, help='{ログ: 説明欄の文} の JSON。説明欄を上書きする')
    ap.add_argument('--rule', default='label', choices=['label', 'cmap'], help='衝突の規則')
    ap.add_argument('--out', default=None, help='書き出す場所(既定 data/reports/scorecard_<名前>)')
    ap.add_argument('--title', default='区間の成績表')
    ap.add_argument('--warmup', type=int, default=WARMUP_DEFAULT, help='手順の慣らしの R1 の起点の数 K(既定 2)')
    ap.add_argument('--counted-s', type=float, default=COUNTED_AI_S, help='手順の数える AI の時間 [秒](既定 240)')
    ap.add_argument('--boot', type=int, default=BOOT_N, help='ブートストラップの回数')
    args = ap.parse_args(argv)
    logs = list(args.logs)
    if args.day:
        logs += day_logs(args.day, args.raw)
    logs = sorted(set(logs))
    if not logs:
        ap.error('ログがありません')
    assign = json.loads(Path(args.assign).read_text(encoding='utf-8')) if args.assign else None
    sc = build(logs, args.labels, args.raw, assign, load_bench(args.bench), args.rule, None, args.boot, args.warmup, args.counted_s)
    out = Path(args.out) if args.out else config.DATA_DIR / 'reports' / ('scorecard_' + (args.day or logs[0]).replace('-', ''))
    pj, pm = write(sc, out, args.title)
    A = sc['groups']['all']
    if A.get('n_logs'):
        print('AI %.1f 分・周相当 %.2f・衝突/周 %s・R1 強/通過 %s・DE/通過 %s' % (
            A['ai_min'], A['lap_equiv'], _ci(A['collisions_per_lap']), _pp(A['r1_strong_per_pass']),
            _pp(A['de_collision_per_pass'])))
    print('書き出し:', pj, pm)
    return sc


if __name__ == '__main__':
    main()
