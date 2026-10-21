#!/usr/bin/env python3
"""実車のベンチマークのレポート(real_bench_20261023、2026-10-21): 記録 → 候補ごとの成績 → 順位 → README.md・report.json・図

    python tools/bench_report.py --day 20261023                                   # 1 日目の夜
    python tools/bench_report.py --day 20261023 --day 20261024                    # 2 日目の夜(判定は 10/24 の周だけ)
    python tools/bench_report.py --day 20261010 --assign mock_assign.json --bench mock_bench.json --out DIR   # 模擬

形は bench_prep_20261021/plan/bench_spec.md、比べ方は plan/protocol.json(metrics・ranking・reference_and_drift)。
読み取りと軽い計算だけ(numpy・matplotlib。GPU なし)。data/raw・data/labels は変えない。

記録 → 候補: 走行開始の「説明」欄の bench=… day=… block=… cand=…(data/bench_candidates.json の description_template)。
  --assign の JSON({ログ: 説明欄の文})で上書きできる(書き忘れの直し・模擬)。割り当てられないログは「未割り当て」に出す。
  説明欄は記録・計画と照らし合わせ、食い違いを deviations.md に書く: 候補に無い id(順位に入れない)・日が記録の日付と違う
  (記録の日付で数える。--assign の説明欄はそのまま)・計画の days[].blocks と違う・log.json の drive.run が候補の run と違う。
門の計算し直し: protocol.json の gates + 事前登録の直し A1(家族 label_r1 は門 5 を l1_steer_best_us、brake_lowspeed_frac ≤ 0.01)。
数える区間: 候補ごとにベンチで初めてのブロックは K = 3、2 回目からは K = 2(protocol run_settings.block.warmup_anchors)。
  同じブロックの走り直しの記録は K = 2 で、AI の時間を足し合わせて 240 秒(ブロックに counted_ai_s があればその秒)まで。
指標・90% 区間: tools/section_scorecard.py の protocol_metrics(protocol の metrics と同じ名前)。
基準: その日の B の全ブロック。B の最初と最後で「時間のずれあり」なら、候補のブロックの前後の B を番号で内挿する。
  ラップだけ、電圧(V_before・V_after)がそろっていて B の周の傾きがはっきり負なら、電池の直しを先に入れる。
順位: S = Δ失敗/周 + λ × Δラップの中央値(λ = 1.0 は既定の仮定。0.5・2.0 も並べる)。1 日目は上位 2 本(T1・T2)、
  2 日目は 10/24 の周だけで「B より良い」の 5 条件を判定する(protocol ranking)。
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from jetformer import config  # noqa: E402
import section_scorecard as ss  # noqa: E402

SCHEMA = 'jf-bench-report/v1'
PREP = Path('/home/jetson/jetformer_reports/bench_prep_20261021')
PROTOCOL = PREP / 'plan' / 'protocol.json'
DESK = PREP / 'desk' / 'desk_bench.json'
BENCH = config.DATA_DIR / 'bench_candidates.json'
POOL_ORDER = ['B', 'MSF', 'TCP', 'ACT', 'SIM', 'ENS', 'SLOW', 'WGT', 'AUG']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAY, INK, INK2, SURF = '#8a8984', '#0b0b0b', '#52514e', '#fcfcfb'
CAP_IDS = ('B+cap', 'BCAP', 'B_CAP', 'Bcap')
# 候補ごとの読み方の注(確認 V-models の minor、10/21 Fix-all)。run の名前で引く(ほかのベンチでは出ない)
READING_NOTES = {
    'bench_20261021-slow-s0': 'SLOW は 13 エポック目の重み(best。人のラベルの検証で選んだので学びの途中。last はもっと緩める)。'
                              'R1 の手前だけでなく全体に遅い(保留で GM +48・DE +29・全体 +30 µs)。R1 の衝突が減っても「手前で自分で'
                              '緩めた」効き目だけとは言えない。速く(≥ 2.75 m/s)近づくときは B より +112 µs 緩める。区間ごとの通過時間と '
                              'R1 の進入 eRPM を並べて読む。',
    'bench_20261021-aug-s0': 'AUG の risk「low」は境の近く(r1_app_thr_diff_us −9.72、境 −10)。C3〜C5 の順は境の近くで決まった。',
    'bench_20261021-wgt-s0': 'WGT は止まった場面(1 m/s 未満)の 6.0% で後退側の出力(B 0.9%)。立ち往生の交代・後退の補助が B と違って'
                             '見えることがある(後退の assist・pwm を見る)。',
    'bench_20261021-ens-p2': 'ENS のエンジンは読み込むと TensorRT が「別の型の機器で作ったプラン」と警告する(精度と速さは 10/21 に確かめ済み)。'
                             '推論はほかより遅い(p99 約 4.3 ms)。',
    'clean_driving_20261011-act-s0-nocmd-stride3-e30': 'ACT のエンジンも同じ警告が出る(精度と速さは 10/21 に確かめ済み)。',
}
DESK_FIELDS = ['l1_best_us', 'l1_last_us', 'l1_interv_us', 'r1_app_thr_diff_us', 'steer_amp_ratio', 'trt_median_ms',
               'trt_p99_ms', 'engine_ok', 'sanity_ok', 'gates_pass', 'gate_fail_reasons',
               'l1_steer_best_us', 'l1_thr_best_us', 'brake_lowspeed_frac']          # 下の 3 つは A1(protocol_amendments.md)
# 事前登録の直し A1(plan/protocol_amendments.md、2026-10-21 11:30): 家族 label_r1 だけ、門 5 を l1_steer_best_us で比べ、
# brake_lowspeed_frac ≤ 0.01 を足す。ほかの家族の門は protocol.json の gates のまま
A1_FAMILIES = ('label_r1',)
A1_BRAKE_LOWSPEED_MAX = 0.01
L1_GATE_US = 10.0
# protocol.json pool.family_rule.by_run_name(run の名前で家族を決める。id が変わっても同じ)
RUN_FAMILY_RULE = (('-slow-', 'label_r1'), ('-wgt-', 'weight'), ('-aug-', 'robust'), ('-ens-', 'ens'))
FLAG_WORDS = {'safety_stop': ('安全で中止', 'stop_safety', 'safety_stop'), 'cannot_lap': ('回れない', 'cannot_lap')}


def _r(x, n=3):
    return ss._r(x, n)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return default


def color_of(cid):
    if cid == 'B' or cid in CAP_IDS:
        return GRAY
    i = POOL_ORDER.index(cid) - 1 if cid in POOL_ORDER else (zlib.crc32(cid.encode()) % len(COLORS))
    return COLORS[i % len(COLORS)]


def num(x):
    try:
        v = float(str(x).replace('V', ''))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ 机上の表
def desk_rows(desk):
    """desk_bench.json → {id: 欄}(リスト・{'candidates': …}・{id: …} のどれでも読む)"""
    if not desk:
        return {}
    c = desk.get('candidates', desk) if isinstance(desk, dict) else desk
    out = {}
    if isinstance(c, list):
        for r in c:
            if isinstance(r, dict) and r.get('id'):
                out[r['id']] = r
    elif isinstance(c, dict):
        for k, r in c.items():
            if isinstance(r, dict):
                out[k] = dict(r, id=r.get('id', k))
    return out


def family_of(cid, run=None, bench=None, proto=None, desk_row=None):
    """候補の家族(protocol.json pool.family_rule): run の名前の規則(-slow- → label_r1 など)が先、無ければ表の family
    (bench_candidates.json・desk_bench.json・protocol.json の pool の順)。決まらなければ None"""
    bc = next((c for c in (bench or {}).get('candidates', []) if c.get('id') == cid), {}) or {}
    pool = ((proto or {}).get('pool') or {}).get('candidates', [])
    pc = next((c for c in pool if c.get('id') == cid), {}) or {}
    for r in (run, bc.get('run'), (desk_row or {}).get('run'), pc.get('run')):
        for key, fam in RUN_FAMILY_RULE:
            if r and key in str(r):
                return fam
    for src in (bc, desk_row or {}, pc):
        if src.get('family'):
            return src['family']
    if run or bc.get('run'):
        rr = run or bc.get('run')
        pr = next((c for c in pool if c.get('run') == rr), None)
        if pr and pr.get('family'):
            return pr['family']
    return None


def recompute_gates(rows, families=None):
    """protocol.gates の式で門を計算し直す(+ 事前登録の直し A1: 家族 label_r1 は門 5 を l1_steer_best_us で比べ、
    brake_lowspeed_frac ≤ 0.01 を足す)。families = {id: 家族}(無い id は A1 を当てない)"""
    families = families or {}
    B = rows.get('B', {})
    res = {}
    for cid, r in rows.items():
        if cid == 'B':
            continue
        fail = []
        fam = families.get(cid) or r.get('family')
        a1 = fam in A1_FAMILIES

        def val(k):
            v = r.get(k)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                fail.append('missing:' + k)
                return None
            return v
        if val('engine_ok') is not None and r['engine_ok'] is not True:
            fail.append('engine_ok')
        if val('sanity_ok') is not None and r['sanity_ok'] is not True:
            fail.append('sanity_ok')
        p99 = val('trt_p99_ms')
        if p99 is not None and not p99 <= 10.0:
            fail.append('trt_p99_ms')
        sa = val('steer_amp_ratio')
        if sa is not None and not 0.80 <= sa <= 1.25:
            fail.append('steer_amp_ratio')
        l1_key = 'l1_steer_best_us' if a1 else 'l1_best_us'
        b = B.get(l1_key)
        if b is None or (isinstance(b, float) and not math.isfinite(b)):
            b = None
        l1 = val(l1_key)
        if l1 is not None and b is not None and not (l1 - b < L1_GATE_US):
            fail.append(l1_key)
        if b is None:
            fail.append('missing:B.' + l1_key)
        if a1:
            bl = val('brake_lowspeed_frac')
            if bl is not None and not bl <= A1_BRAKE_LOWSPEED_MAX:
                fail.append('brake_lowspeed_frac')
        res[cid] = {'gates_pass': not fail, 'gate_fail_reasons': fail, 'family': fam, 'amendment': 'A1' if a1 else None,
                    'agrees_with_desk': (r.get('gates_pass') is None) or (bool(r.get('gates_pass')) == (not fail))}
    return res


# ------------------------------------------------------------------ 記録の割り当て
def _log_day(lg):
    """記録の名前(YYYYMMDD-HHMMSS)の日付 → 'YYYY-MM-DD'。読めなければ None"""
    try:
        datetime.strptime(str(lg)[:8], '%Y%m%d')
    except ValueError:
        return None
    return '%s-%s-%s' % (lg[:4], lg[4:6], lg[6:8])


def planned_block(bench, day, block):
    """bench_candidates.json の days[].blocks で、その日・ブロックの id。返り値 (日が計画にあるか, ブロックの一覧があるか, id か None)"""
    for dd in (bench or {}).get('days', []):
        if dd.get('date') == day:
            bl = dd.get('blocks') or []
            return True, bool(bl), next((b.get('id') for b in bl if b.get('block') == block), None)
    return False, False, None


def _drive_run(lg, labels_dir, raw_dir):
    """実際に走ったモデル(log.json の drive.run。無ければラベルの run)"""
    info = load_json(Path(raw_dir) / lg / 'log.json', {}) or {}
    run = (info.get('drive') or {}).get('run') if isinstance(info.get('drive'), dict) else None
    if not run:
        lab = load_json(Path(labels_dir) / (lg + '.json'), {}) or {}
        run = lab.get('run')
    return run


def collect(logs, bench, assign, labels_dir, raw_dir):
    """ログ → {log, desc, day, block, cand, checks}。説明欄の bench が違うものは外す。

    説明欄を記録と計画に照らし合わせ、食い違いを checks([(種類, 文)])に入れる(V-tools 4 章 4):
      cand が bench_candidates.json の候補に無い(→ 順位に入れない)・説明欄の day が記録の日付と違う(→ 記録の日付で数える。
      --assign で渡した説明欄は人の直しなので、その day を使う)・計画の days[].blocks の id と違う・計画に無いブロック・
      log.json の drive.run(実際に走ったモデル)が候補の run と違う"""
    name = (bench or {}).get('bench')
    known = {c.get('id'): c for c in (bench or {}).get('candidates', []) if c.get('id')}
    rows, unassigned = [], []
    for lg in logs:
        assigned = bool(assign) and lg in assign
        if assigned:
            desc_raw = assign[lg]
        else:
            p = Path(labels_dir) / (lg + '.json')
            info = load_json(p, {}) if p.exists() else load_json(Path(raw_dir) / lg / 'log.json', {})
            desc_raw = (info or {}).get('description', '')
            # 走ったあとの V_after・「安全で中止」・「回れない」はシーンの説明(scene.json)に書かれる → 記録の説明に入れる
            desc_raw, scene_fix = ss.effective_description(lg, desc_raw, raw_dir)
        d = ss.parse_description(desc_raw)
        cand = d.get('cand')
        if not cand and bench:
            cand, _ = ss.assign_candidate(d, None, bench)
        if not cand or d.get('block') is None or (name and d.get('bench') and d['bench'] != name):
            unassigned.append({'log': lg, 'description': desc_raw})
            continue
        checks = []
        if not assigned:
            for c in scene_fix['conflicts']:
                checks.append(('scene', 'シーンの説明の直しが食い違う: %s' % c))
        log_day, desc_day = _log_day(lg), d.get('day')
        if assigned or not log_day:
            day = desc_day or log_day
        else:
            day = log_day
            if desc_day and desc_day != log_day:
                checks.append(('day', '説明欄の day=%s が記録の日付 %s と違う → 記録の日付 %s で数えた(前の日の説明欄を使い回した?)' % (
                    desc_day, log_day, log_day)))
        unknown = bool(known) and cand not in known and cand not in CAP_IDS
        if unknown:
            near = [k for k in list(known) + list(CAP_IDS) if k.lower() == cand.lower() or
                    (len(k) == len(cand) and sorted(k.lower()) == sorted(cand.lower()))]
            checks.append(('cand', 'cand=%s は bench_candidates.json の候補に無い%s → 順位に入れない(--assign で直す)' % (
                cand, '(%s の書き違い?)' % '・'.join(near) if near else '')))
        in_plan, has_blocks, planned = planned_block(bench, day, d['block'])
        if bench and bench.get('days') and not in_plan:
            checks.append(('plan', '日 %s は bench_candidates.json の days に無い' % day))
        elif has_blocks and planned is None:
            checks.append(('plan', '%s のブロック %s は計画(days[].blocks)に無い' % (day, d['block'])))
        elif planned is not None and planned != cand:
            checks.append(('plan', '%s のブロック %s は計画では %s、説明欄は cand=%s' % (day, d['block'], planned, cand)))
        exp_run = (known.get(cand) or {}).get('run') or ((known.get('B') or {}).get('run') if cand in CAP_IDS else None)
        run = _drive_run(lg, labels_dir, raw_dir)
        if exp_run and run and run != exp_run:
            checks.append(('run', '実際に走ったモデル(log.json の drive.run)%s が、候補 %s の run %s と違う' % (run, cand, exp_run)))
        elif exp_run and not run:
            checks.append(('run', '記録に drive.run が無い(モデルを照らし合わせられない)'))
        rows.append({'log': lg, 'description': desc_raw, 'desc': d, 'day': day, 'block': d['block'], 'cand': cand,
                     'drive_run': run, 'expected_run': exp_run, 'unknown_cand': unknown, 'assigned': assigned, 'checks': checks,
                     'desc_scene_edits': [] if assigned else scene_fix['edited_scenes']})
    rows.sort(key=lambda r: (r['day'], r['block'], r['log']))
    return rows, unassigned


def check_deviations(entries):
    """collect の checks → deviations.md の行。モデルの食い違いは (候補, 実際の run) ごとにまとめる"""
    out, runs = [], {}
    for e in entries:
        for kind, text in e.get('checks', []):
            if kind == 'run' and e.get('drive_run'):
                runs.setdefault((e['cand'], e['drive_run'], e.get('expected_run')), []).append(e['log'])
            else:
                out.append('説明欄の照らし合わせ: %s(%s ブロック %s・cand=%s): %s' % (e['log'], e['day'], e['block'], e['cand'], text))
    for (cand, run, exp), logs in sorted(runs.items()):
        out.append('説明欄の照らし合わせ: cand=%s の記録 %d 本(%s)は、実際に走ったモデル(log.json の drive.run)が %s。候補の run は %s' % (
            cand, len(logs), '・'.join(logs), run, exp))
    n_as = sum(1 for e in entries if e.get('assigned'))
    if n_as:
        out.append('--assign で説明欄を上書きした記録が %d 本(その日・ブロック・候補は --assign のとおり)' % n_as)
    return out


def block_budget(bench, day, block, default=ss.COUNTED_AI_S):
    for dd in (bench or {}).get('days', []):
        if dd.get('date') == day:
            for b in dd.get('blocks', []):
                if b.get('block') == block and b.get('counted_ai_s'):
                    return float(b['counted_ai_s'])
    return default


def score_entries(entries, bench, labels_dir, raw_dir, k_first=3, k_later=2):
    """K と数える秒を決めながら、ログを順に採点する(K: ベンチで初めての候補 k_first、2 回目から・走り直し k_later)"""
    seen = set()
    used = {}
    for e in entries:
        key = (e['day'], e['block'])
        budget = block_budget(bench, e['day'], e['block'])
        if key in used:                                   # 走り直し
            K, left = k_later, max(0.0, budget - used[key])
            e['rerun'] = True
        else:
            K = k_first if e['cand'] not in seen else k_later
            left = budget
            e['rerun'] = False
        e['K'] = K
        e['score'] = ss.score_log(e['log'], labels_dir, raw_dir, e['description'], bench, K, left)
        w = e['score'].get('window') or {}
        # 車輪を浮かせた dry run を収集オンで録った記録(AI の時間 = 0、dry run は数えない)は、走り直しにも
        # 「この候補は走った」にも数えない(本番の記録が K = 2 にならないように。10/21 Fix-all、V-protocol minor 11)
        e['dry_only'] = not e['score'].get('error') and not (e['score'].get('ai_s') or 0.0)
        if e['dry_only']:
            e['rerun'] = False
        else:
            seen.add(e['cand'])
            used[key] = used.get(key, 0.0) + (w.get('ai_s') or 0.0)
        txt = ' '.join([e['description'] or ''])
        e['flags'] = sorted(k for k, words in FLAG_WORDS.items() if any(wd in txt for wd in words))
    return entries


# ------------------------------------------------------------------ 電池・時間のずれ・比べ
def t_quantile(p, df):
    """t 分布の分位点の近似(Cornish-Fisher。df ≥ 3 で十分)"""
    z = math.sqrt(2) * _erfinv(2 * p - 1)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def _erfinv(y):
    a = 0.147
    ln = math.log(1 - y * y)
    t1 = 2 / (math.pi * a) + ln / 2
    x = math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)
    for _ in range(3):                      # ニュートン法で仕上げる
        x -= (math.erf(x) - y) / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def lap_voltages(e):
    """周の電圧 V = V_before + (V_after − V_before) × (周の真ん中までの AI の時間 ÷ 走行の AI の時間)"""
    vb, va = num(e['desc'].get('V_before')), num(e['desc'].get('V_after'))
    rows = e['score'].get('laps_protocol') or []
    ai = (e['score'].get('window') or {}).get('ai_s') or 0
    if not rows:
        return []
    if vb is None or va is None or not ai:
        return None
    return [vb + (va - vb) * min(1.0, (r.get('ai_s_at_mid') or 0) / ai) for r in rows]


def battery_fit(day_entries):
    """B の周で lap = a + b·V。b < 0 で 90% 区間が 0 を含まないときだけ使う。電圧が 1 つでも欠ければ使わない"""
    missing = [e['log'] for e in day_entries if lap_voltages(e) is None]
    out = {'used': False, 'missing_voltage_logs': missing}
    if missing:
        out['note'] = '電圧の書き忘れがあるので直さない'
        return out
    V, T = [], []
    for e in day_entries:
        if e['cand'] != 'B':
            continue
        for v, r in zip(lap_voltages(e), e['score']['laps_protocol']):
            if r.get('lap_time_s') is not None and not r.get('reboot'):
                V.append(v)
                T.append(r['lap_time_s'])
    if len(V) < 5 or np.std(V) < 1e-6:
        out['note'] = 'B の周か電圧の幅が足りない'
        return out
    V, T = np.array(V), np.array(T)
    X = np.c_[np.ones(len(V)), V]
    beta, res, *_ = np.linalg.lstsq(X, T, rcond=None)
    dof = len(V) - 2
    s2 = float(np.sum((T - X @ beta) ** 2) / dof)
    se_b = math.sqrt(s2 / np.sum((V - V.mean()) ** 2))
    tq = t_quantile(0.95, dof)
    b = float(beta[1])
    ci = [b - tq * se_b, b + tq * se_b]
    out.update(b_s_per_V=_r(b, 4), ci90=[_r(ci[0], 4), _r(ci[1], 4)], a=_r(beta[0], 3), n=len(V), V_mean_B=_r(V.mean(), 3))
    out['used'] = bool(b < 0 and ci[1] < 0)
    if not out['used']:
        out['note'] = '傾きがはっきり負でないので直さない'
    return out


def apply_battery(day_entries, fit):
    """直しを使うときは、周の行の写しの lap_time_s を lap − b·(V − B の平均)にする"""
    for e in day_entries:
        rows = [dict(r) for r in (e['score'].get('laps_protocol') or [])]
        if fit.get('used'):
            for v, r in zip(lap_voltages(e), rows):
                if r.get('lap_time_s') is not None:
                    r['lap_time_raw_s'] = r['lap_time_s']
                    r['lap_time_s'] = _r(r['lap_time_s'] - fit['b_s_per_V'] * (v - fit['V_mean_B']), 3)
        e['rows'] = rows


def runs_of(entries):
    return [e['rows'] for e in entries]


def seed_of(day, tag):
    base = int(str(day).replace('-', ''))
    return base + (zlib.crc32(tag.encode()) % 100000) * 1000 if tag else base


def metrics_and_samples(entries, day, tag, n_boot):
    ai_s = sum((e['score'].get('window') or {}).get('ai_s') or 0 for e in entries)
    # 再起動/4 分はブロックの記録の AI の時間の全部(慣らし・半端な周・区間の後も)で数える。周が 0 本でも出す
    basis = ss.reboot_basis([e['score'].get('window') for e in entries])
    m, smp = ss.protocol_metrics(runs_of(entries), ai_s=ai_s, seed=seed_of(day, tag), n_boot=n_boot, with_samples=True,
                                 reboot_basis=basis)
    cap = ss.sum_caps([(e['score'].get('window') or {}).get('cap') for e in entries])
    if cap:
        m['cap'] = cap
        # 「R1 の旋回中のブレーキ」= ON のブレーキのフレームのうち、後から見た起点 ta → ta + 0.5 秒のもの(層の区間では数えない)
        m['cap_brake_in_r1_turn'] = cap.get('on_brake_frames_in_r1_turn', 0)
    return m, smp


def block_groups(day_entries):
    g = {}
    for e in day_entries:
        g.setdefault(e['block'], []).append(e)
    return dict(sorted(g.items()))


def compare_day(day, day_entries, n_boot, lam=ss.LAMBDA_S, use_drift=True):
    """その日の基準(B)と候補の差。返り値 {'blocks', 'reference', 'cands'}"""
    blocks = block_groups(day_entries)
    bm = {}
    for b, es in blocks.items():
        m, smp = metrics_and_samples(es, day, 'block%s' % b, n_boot)
        bm[b] = {'cand': es[0]['cand'], 'logs': [e['log'] for e in es], 'metrics': m, 'samples': smp,
                 'battery': es[0]['desc'].get('battery'), 'V_before': num(es[0]['desc'].get('V_before')),
                 'V_after': num(es[-1]['desc'].get('V_after')), 'K': es[0]['K'],
                 'ai_s_counted': _r(sum((e['score'].get('window') or {}).get('ai_s') or 0 for e in es), 1),
                 'flags': sorted({f for e in es for f in e['flags']}), 'reruns': sum(e['rerun'] for e in es)}
    b_blocks = [b for b, x in bm.items() if x['cand'] == 'B']
    B_entries = [e for e in day_entries if e['cand'] == 'B']
    ref = {'b_blocks': b_blocks}
    if B_entries:
        pm, ps = metrics_and_samples(B_entries, day, 'pooledB', n_boot)
        ref['pooled_B'] = pm
        ref['_samples'] = ps
    drift = {'flag': False}
    # protocol 8.4: B の最初と最後のブロックでくらべる。どちらかの周が 0 なら「判定できない」(「ずれ なし」とは言わない)とし、
    # 内挿はせず B のプールを基準にする(V-tools 4 章 5)
    b_lap = [b for b in b_blocks if bm[b]['metrics'].get('laps_counted')]
    if len(b_blocks) >= 2 and (b_blocks[0] not in b_lap or b_blocks[-1] not in b_lap):
        drift.update(undecided=True, first_block=b_blocks[0], last_block=b_blocks[-1],
                     note='時間のずれは判定できない: 最初か最後の B のブロックの周が 0(%s)。内挿せず、B のプールを基準にした' % '・'.join(
                         'ブロック %s %d 周' % (b, bm[b]['metrics'].get('laps_counted') or 0) for b in b_blocks))
        b_lap = []
    if len(b_lap) >= 2:
        f, l_ = bm[b_lap[0]], bm[b_lap[-1]]
        lm = [f['metrics'].get('lap_median_s', {}).get('value'), l_['metrics'].get('lap_median_s', {}).get('value')]
        dl = (lm[1] - lm[0]) if None not in lm else None
        d = None
        if 'incidents_per_lap' in f['samples'] and 'incidents_per_lap' in l_['samples']:
            d = l_['samples']['incidents_per_lap'] - f['samples']['incidents_per_lap']
            d = [ss._pct(d, 5), ss._pct(d, 95)]
        dlc = None
        if 'lap_median_s' in f['samples'] and 'lap_median_s' in l_['samples']:
            dlc = l_['samples']['lap_median_s'] - f['samples']['lap_median_s']
            dlc = [ss._pct(dlc, 5), ss._pct(dlc, 95)]
        drift.update(first_block=b_lap[0], last_block=b_lap[-1], d_lap_median_s=_r(dl, 3),
                     d_lap_median_ci90=[_r(x, 4) for x in dlc] if dlc else None,
                     d_incidents_ci90=[_r(x, 4) for x in d] if d else None)
        excl0 = lambda c: bool(c and c[0] is not None and c[1] is not None and (c[0] > 0 or c[1] < 0))   # noqa: E731
        # 直し A7(plan/protocol_amendments.md、10/21 Fix-all): ラップも「差の 90% 区間が 0 を含まない」にそろえる。
        # protocol.json のままの判定(ラップの差 > 0.3 秒)は flag_protocol_v1 に残す(ずれが無い日でも 25% で働くため)
        drift['flag_protocol_v1'] = bool((dl is not None and abs(dl) > 0.3) or excl0(d))
        drift['rule'] = 'A7'
        drift['flag'] = bool(excl0(dlc) or excl0(d))
        if drift['flag'] != drift['flag_protocol_v1']:
            drift['note'] = ('時間のずれの判定: 直し A7(ラップの差の 90%% 区間 %s・失敗/周の差の 90%% 区間 %s)では「%s」、'
                             'protocol.json のまま(ラップの差 %s 秒 > 0.3)なら「%s」。A7 を使った' % (
                                 [_r(x, 3) for x in dlc] if dlc else '—', [_r(x, 3) for x in d] if d else '—',
                                 'ずれあり' if drift['flag'] else 'ずれなし', _r(dl, 3),
                                 'ずれあり' if drift['flag_protocol_v1'] else 'ずれなし'))
        if not use_drift:
            drift['flag_ignored'], drift['flag'] = drift['flag'], False
    ref['drift'] = drift
    cands = {}
    for cid in sorted({x['cand'] for x in bm.values()} - {'B'}):
        es = [e for e in day_entries if e['cand'] == cid]
        cm, cs = metrics_and_samples(es, day, 'cand' + cid, n_boot)
        mine = [b for b, x in bm.items() if x['cand'] == cid]
        if not B_entries or not cm.get('laps_counted'):
            cands[cid] = {'metrics': cm, 'blocks': mine, 'diff': {}, 'ref': None,
                          'flags': sorted({f for b in mine for f in bm[b]['flags']})}
            continue
        if drift['flag']:
            # 前後の B を番号で内挿(候補の周の数で重み)
            w_tot = 0.0
            r_pt, r_s = {}, {}
            for b in mine:
                p = max([x for x in b_lap if x < b], default=None)
                q = min([x for x in b_lap if x > b], default=None)
                if p is None or q is None:
                    p = q = p if p is not None else q
                w = 0.0 if p == q else (b - p) / (q - p)
                lw = bm[b]['metrics'].get('laps_counted') or 0
                w_tot += lw
                for k in ss.BOOT_KEYS:
                    vp = bm[p]['metrics'].get(k, {}).get('value')
                    vq = bm[q]['metrics'].get(k, {}).get('value')
                    if vp is None or vq is None or k not in bm[p]['samples'] or k not in bm[q]['samples']:
                        continue
                    r_pt[k] = r_pt.get(k, 0.0) + lw * ((1 - w) * vp + w * vq)
                    r_s[k] = r_s.get(k, 0.0) + lw * ((1 - w) * bm[p]['samples'][k] + w * bm[q]['samples'][k])
            ref_pt = {k: v / w_tot for k, v in r_pt.items()} if w_tot else {}
            ref_s = {k: v / w_tot for k, v in r_s.items()} if w_tot else {}
            ref_kind = 'interpolated'
        else:
            ref_pt = {k: ref['pooled_B'].get(k, {}).get('value') for k in ss.BOOT_KEYS}
            ref_s = ref['_samples']
            ref_kind = 'pooled_B'
        diff = {}
        for k in ss.BOOT_KEYS:
            a, b_ = cm.get(k, {}).get('value'), ref_pt.get(k)
            d = cs[k] - ref_s[k] if k in cs and k in ref_s else None
            diff['d_' + k] = {'value': _r(a - b_, 4) if a is not None and b_ is not None else None,
                              'ci90': [_r(ss._pct(d, 5), 4), _r(ss._pct(d, 95), 4)] if d is not None else [None, None]}
        S = {}
        for lm_ in (0.5, 1.0, 2.0):
            di, dl = diff['d_incidents_per_lap'], diff['d_lap_median_s']
            if di['value'] is None or dl['value'] is None:
                continue
            sm = (cs['incidents_per_lap'] - ref_s['incidents_per_lap']) + lm_ * (cs['lap_median_s'] - ref_s['lap_median_s'])
            S[str(lm_)] = {'value': _r(di['value'] + lm_ * dl['value'], 4), 'ci90': [_r(ss._pct(sm, 5), 4), _r(ss._pct(sm, 95), 4)]}
        cands[cid] = {'metrics': cm, 'blocks': mine, 'diff': diff, 'S': S, 'ref': ref_kind, 'ref_point': {k: _r(v, 4) for k, v in ref_pt.items()},
                      'flags': sorted({f for b in mine for f in bm[b]['flags']})}
    for cid in cands:
        cands[cid]['unknown_id'] = any(e.get('unknown_cand') for e in day_entries if e['cand'] == cid)
    ref.pop('_samples', None)
    for x in bm.values():
        x.pop('samples', None)
    return {'blocks': bm, 'reference': ref, 'cands': cands}


def uncorrected_compare(day, day_entries, n_boot):
    """直す前の比べ: 電池の直しを外したラップ(lap_time_raw_s)で、内挿せず B のプールと比べる(bench_spec 5.1)"""
    raw = [dict(e, rows=[dict(r, lap_time_s=r.get('lap_time_raw_s', r.get('lap_time_s'))) for r in e['rows']]) for e in day_entries]
    return compare_day(day, raw, n_boot, use_drift=False)


def uncorrected_S(day, day_entries, n_boot, c0=None):
    """直す前の S(λ = 1.0) の点の値"""
    c0 = c0 or uncorrected_compare(day, day_entries, n_boot)
    return {cid: (c.get('S') or {}).get('1.0', {}).get('value') for cid, c in c0['cands'].items()}


def _round05(x):
    return round(round(x / 0.05) * 0.05, 2)


def rank_day1(cmp_, lam='1.0', min_laps=12):
    rows = []
    B = cmp_['reference'].get('pooled_B') or {}
    b_r1 = (B.get('r1_strong_per_pass') or {})
    for cid, c in cmp_['cands'].items():
        m = c['metrics']
        marks = []
        if not m.get('laps_counted') or m['laps_counted'] < min_laps:
            marks.append('足りない')
        if 'safety_stop' in c.get('flags', []):
            marks.append('安全で中止')
        if 'cannot_lap' in c.get('flags', []):
            marks.append('回れない')
        if c.get('unknown_id'):
            marks.append('計画に無い id')
        di = c['diff'].get('d_incidents_per_lap', {})
        if di.get('ci90') and di['ci90'][0] is not None and di['ci90'][0] > 0:
            marks.append('はっきり悪い')
        S = (c.get('S') or {}).get(lam, {})
        rows.append({'id': cid, 'S': S.get('value'), 'S_ci90': S.get('ci90'), 'marks': marks, 'eligible': not marks and S.get('value') is not None,
                     'first_block': min(c['blocks']) if c['blocks'] else 99,
                     'r1_strong': (m.get('r1_strong_per_pass') or {}).get('value'),
                     'fail_tk': (m.get('failure_takeovers_per_lap') or {}).get('value')})
    key = lambda r: (_round05(r['S']), r['r1_strong'] if r['r1_strong'] is not None else 9, r['fail_tk'] if r['fail_tk'] is not None else 9,  # noqa: E731
                     r['first_block'])
    el = sorted([r for r in rows if r['eligible']], key=key)
    rest = [r for r in rows if not r['eligible']]
    top = [r['id'] for r in el[:2]]
    swap = None
    b_val, b_k = b_r1.get('value'), b_r1.get('k', 0)
    if len(top) == 2 and b_val is not None and b_k >= 3:
        half = b_val / 2
        if all((next(r for r in el if r['id'] == t)['r1_strong'] or 0) > half for t in top):
            alt = [r for r in el[2:] if r['r1_strong'] is not None and r['r1_strong'] <= half]
            if alt:
                swap = {'out': top[1], 'in': alt[0]['id']}
                top[1] = alt[0]['id']
    for i, r in enumerate(el):
        r['rank'] = i + 1
    for r in rows:
        if r['id'] in top:
            r['marks'].append('T%d' % (top.index(r['id']) + 1))
    if swap:
        next(r for r in rows if r['id'] == swap['in'])['marks'].append('R1 の枠で入れ替え')
    return {'lambda': float(lam), 'rows': el + rest, 'T1': top[0] if top else None, 'T2': top[1] if len(top) > 1 else None,
            'r1_slot_swap': swap}


def _reboot_cond(cand, base):
    """条件 5: 再起動/4 分が B より 1 回以上多くない。どちらかが None(元の AI の時間が無い)なら None = 判定できない
    (0 として通さない。再起動で走行が切れた候補ほど通りやすくならないように)"""
    if cand is None or base is None:
        return None
    return not (cand - base >= 1.0)


def judge_day2(cmp_, bench, min_laps=12):
    B = cmp_['reference'].get('pooled_B') or {}
    out = []
    for cid, c in cmp_['cands'].items():
        if cid in CAP_IDS:
            continue
        m, d = c['metrics'], c['diff']
        S = (c.get('S') or {}).get('1.0', {})
        cond = {
            '1_S_upper_lt_0': bool(S.get('ci90') and S['ci90'][1] is not None and S['ci90'][1] < 0),
            '2_d_incidents_upper_le_0.10': bool(d.get('d_incidents_per_lap', {}).get('ci90', [None, None])[1] is not None
                                                and d['d_incidents_per_lap']['ci90'][1] <= 0.10),
            '3_r1_strong_le_B': bool((m.get('r1_strong_per_pass') or {}).get('value') is not None and B.get('r1_strong_per_pass', {}).get('value') is not None
                                     and m['r1_strong_per_pass']['value'] <= B['r1_strong_per_pass']['value']),
            '4_no_safety_stop': 'safety_stop' not in c.get('flags', []),
            '5_reboots_not_1_more': _reboot_cond(m.get('reboots_per_4min'), B.get('reboots_per_4min')),
        }
        risk = next((x.get('risk') for x in (bench or {}).get('candidates', []) if x.get('id') == cid), None)
        out.append({'id': cid, 'conditions': cond,
                    'better_than_B': all(v is True for v in cond.values()) and (m.get('laps_counted') or 0) >= min_laps and not c.get('unknown_id'),
                    'unknown_id': bool(c.get('unknown_id')), 'reboots_per_4min': m.get('reboots_per_4min'),
                    'S': S.get('value'), 'S_ci90': S.get('ci90'), 'incidents': (m.get('incidents_per_lap') or {}).get('value'), 'risk': risk})
    good = [r for r in out if r['better_than_B']]
    rk = {'low': 0, 'mid': 1, 'high': 2}
    if good:
        good.sort(key=lambda r: (_round05(r['S']), r['incidents'] if r['incidents'] is not None else 9, rk.get(r['risk'], 3)))
        verdict = {'result': 'better', 'best': good[0]['id'], 'text': '%s が B より良い' % good[0]['id']}
    else:
        nxt = min([r for r in out if r['S'] is not None], key=lambda r: r['S'], default=None)
        verdict = {'result': 'keep_B', 'best': None, 'next_to_try': nxt['id'] if nxt else None,
                   'text': 'B のまま' + ('(次に試す候補: %s。良いとは言わない)' % nxt['id'] if nxt else '')}
    if not out or all((cmp_['cands'][r['id']]['metrics'].get('laps_counted') or 0) < min_laps for r in out):
        verdict = {'result': 'insufficient', 'best': None, 'text': '足りない(周が足りない・中止が多い)'}
    return {'rows': out, 'verdict': verdict}


# ------------------------------------------------------------------ 図
def _setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def _dot_whisker(ax, labels, vals, cis, colors, zero=True, xlabel=''):
    y = np.arange(len(labels))[::-1]
    for yi, v, c, col in zip(y, vals, cis, colors):
        if v is None:
            continue
        if c and c[0] is not None:
            ax.plot([c[0], c[1]], [yi, yi], color=col, lw=2, solid_capstyle='round')
        ax.plot([v], [yi], 'o', ms=8, color=col, mec=SURF, mew=2)
        ax.annotate('%.2f' % v, (v, yi), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9, color=INK2)
    if zero:
        ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel(xlabel)


def make_figs(rep, fig_dir):
    plt = _setup_mpl()
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = {}
    days = rep['days']
    d1 = days[0] if days else None
    if d1 and d1['compare']['cands']:
        C = d1['compare']['cands']
        ids = [c for c in POOL_ORDER if c in C] + [c for c in C if c not in POOL_ORDER]
        # f1 Δ失敗/周
        fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(ids) + 1.6))
        _dot_whisker(ax, ids, [C[c]['diff'].get('d_incidents_per_lap', {}).get('value') for c in ids],
                     [C[c]['diff'].get('d_incidents_per_lap', {}).get('ci90') for c in ids], [color_of(c) for c in ids],
                     xlabel='Δ失敗/周(候補 − B、90% 区間)。右ほど悪い')
        ax.set_title('%s 候補ごとの Δ失敗/周' % d1['day'], loc='left', color=INK)
        fig.tight_layout()
        fig.savefig(fig_dir / 'f1_rank_day1.png', dpi=130)
        plt.close(fig)
        made['f1_rank_day1.png'] = '1 日目: 候補ごとの Δ失敗/周 と 90% 区間'
        # f2 交換
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for c in ids:
            di, dl = C[c]['diff'].get('d_incidents_per_lap', {}), C[c]['diff'].get('d_lap_median_s', {})
            if di.get('value') is None or dl.get('value') is None:
                continue
            ax.plot([dl['value']], [di['value']], 'o', ms=9, color=color_of(c), mec=SURF, mew=2)
            ax.annotate(c, (dl['value'], di['value']), xytext=(6, 4), textcoords='offset points', fontsize=10, color=INK)
        lim = ax.get_xlim()
        xs = np.linspace(min(lim[0], -1), max(lim[1], 1), 10)
        for lm_, ls in ((1.0, '-'), (0.5, ':'), (2.0, '--')):
            ax.plot(xs, -lm_ * xs, ls, color=INK2, lw=1)
            ax.annotate('S=0(λ=%.1f)' % lm_, (xs[-1], -lm_ * xs[-1]), fontsize=8, color=INK2, ha='right')
        ax.axhline(0, color='#c9c8c2', lw=0.8)
        ax.axvline(0, color='#c9c8c2', lw=0.8)
        ax.set_xlabel('Δラップの中央値 [秒](右ほど遅い)')
        ax.set_ylabel('Δ失敗/周(上ほど悪い)')
        ax.set_title('速さと安全の交換(線より左下が B より良い)', loc='left', color=INK)
        fig.tight_layout()
        fig.savefig(fig_dir / 'f2_tradeoff.png', dpi=130)
        plt.close(fig)
        made['f2_tradeoff.png'] = 'Δラップと Δ失敗/周。λ = 1.0・0.5・2.0 の S = 0 の線'
    # f3 区間ごと(B と候補、1 日目)
    if d1 and d1['compare']['blocks']:
        C = d1['compare']['cands']
        allm = {'B': d1['compare']['reference'].get('pooled_B') or {}}
        allm.update({c: C[c]['metrics'] for c in C})
        ids = [c for c in POOL_ORDER if c in allm] + [c for c in allm if c not in POOL_ORDER]
        fig, axs = plt.subplots(1, 3, figsize=(11, 0.55 * len(ids) + 1.8), sharey=True)
        for ax, (k, t) in zip(axs, (('r1_strong_per_pass', 'R1 の強い衝突/通過'), ('gm_coll_per_pass', 'GM の衝突/通過'),
                                     ('de_coll_per_pass', 'DE の衝突/通過'))):
            _dot_whisker(ax, ids, [(allm[c].get(k) or {}).get('value') for c in ids], [(allm[c].get(k) or {}).get('ci90') for c in ids],
                         [color_of(c) for c in ids], zero=False, xlabel=t)
            ax.set_xlim(-0.05, 1.05)
        fig.suptitle('%s 区間ごとの通過あたりの衝突(90%% 区間)' % d1['day'], x=0.01, ha='left', color=INK)
        fig.tight_layout()
        fig.savefig(fig_dir / 'f3_sections.png', dpi=130)
        plt.close(fig)
        made['f3_sections.png'] = '区間ごと(R1 強・GM・DE)の通過あたりの衝突'
    # f4 R1 の進入
    if d1:
        ent = {}
        for e in d1['entries']:
            ent.setdefault(e['cand'], []).extend([r['entry_erpm'] for r in e['rows'] if r.get('entry_erpm') is not None])
        if ent:
            ids = [c for c in POOL_ORDER if c in ent] + [c for c in ent if c not in POOL_ORDER]
            fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(ids) + 1.6))
            y = np.arange(len(ids))[::-1]
            for yi, c in zip(y, ids):
                v = np.array(ent[c])
                if not len(v):
                    continue
                jit = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.3
                ax.plot(v, yi + jit, 'o', ms=4, alpha=0.6, color=color_of(c))
                ax.plot([np.median(v)], [yi], '|', ms=18, mew=2.5, color=INK)
            ax.axvline(6200, color=INK2, lw=1, ls='--')
            ax.annotate('6,200 eRPM', (6200, y[0] + 0.4), fontsize=8, color=INK2)
            ax.set_yticks(y)
            ax.set_yticklabels(ids)
            ax.set_xlabel('R1 の進入 eRPM(起点の手前 0.4 秒の最大。縦線 = 中央値)')
            ax.set_title('%s R1 の進入の速さ' % d1['day'], loc='left', color=INK)
            fig.tight_layout()
            fig.savefig(fig_dir / 'f4_r1_entry.png', dpi=130)
            plt.close(fig)
            made['f4_r1_entry.png'] = 'R1 の進入 eRPM の分布(B と候補)'
    # f5 B のずれ(ブロックの番号が横)
    for d in days:
        bb = [(b, x) for b, x in d['compare']['blocks'].items() if x['cand'] == 'B']
        if len(bb) < 2:
            continue
        fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
        for ax, (k, t) in zip(axs, (('lap_median_s', 'ラップの中央値 [秒]'), ('incidents_per_lap', '失敗/周'))):
            xs = [b for b, _ in bb]
            vs = [(x['metrics'].get(k) or {}).get('value') for _, x in bb]
            cis = [(x['metrics'].get(k) or {}).get('ci90') for _, x in bb]
            for xx, v, c in zip(xs, vs, cis):
                if v is None:
                    continue
                if c and c[0] is not None:
                    ax.plot([xx, xx], c, color=GRAY, lw=2)
                ax.plot([xx], [v], 'o', ms=8, color=GRAY, mec=SURF, mew=2)
            ax.plot([x for x, v in zip(xs, vs) if v is not None], [v for v in vs if v is not None], color=GRAY, lw=1)
            ax.set_ylabel(t)
        axs[-1].set_xlabel('ブロックの番号')
        axs[0].set_title('%s B の 3 回(時間のずれ: %s)' % (d['day'], 'あり' if d['compare']['reference']['drift']['flag'] else 'なし'), loc='left', color=INK)
        fig.tight_layout()
        name = 'f5_b_drift_%s.png' % d['day'].replace('-', '')
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made[name] = '%s の B のブロックごとのラップと失敗/周' % d['day']
    # f6 電池
    for d in days:
        fit = d['battery_fit']
        pts = []
        for e in d['entries']:
            if e['cand'] != 'B':
                continue
            vs = lap_voltages(e)
            if vs is None:
                continue
            pts += [(v, r.get('lap_time_raw_s', r.get('lap_time_s'))) for v, r in zip(vs, e['rows']) if r.get('lap_time_s') is not None]
        if len(pts) < 3:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        V, T = np.array(pts).T
        ax.plot(V, T, 'o', ms=6, color=GRAY, mec=SURF, mew=1.5)
        if fit.get('b_s_per_V') is not None:
            xs = np.linspace(V.min(), V.max(), 10)
            ax.plot(xs, fit['a'] + fit['b_s_per_V'] * xs, color=INK, lw=1.5)
        ax.set_xlabel('周の電圧(V_before → V_after を AI の時間で内挿)[V]')
        ax.set_ylabel('ラップ [秒]')
        ax.set_title('%s B の周のラップと電圧(b = %s 秒/V、直し %s)' % (d['day'], fit.get('b_s_per_V'), '使う' if fit.get('used') else '使わない'),
                     loc='left', color=INK, fontsize=10)
        fig.tight_layout()
        name = 'f6_battery_%s.png' % d['day'].replace('-', '')
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made[name] = '%s の B の周のラップと電圧、当てた直線' % d['day']
    # f7 1 日の流れ
    for d in days:
        es = d['entries']
        if not es:
            continue
        fig, ax = plt.subplots(figsize=(10, 2.6))
        x0 = 0.0
        wmin = max(30.0, 0.04 * sum((e['score'].get('window') or {}).get('ai_s') or 0 for e in es))   # 0 秒のブロックも名前が読める幅
        for e in es:
            w0 = (e['score'].get('window') or {}).get('ai_s') or 0
            w = max(w0, wmin)
            ax.barh(0, w, left=x0, height=0.5, color=color_of(e['cand']), edgecolor=SURF, linewidth=2, alpha=1.0 if w0 >= wmin else 0.45)
            ax.annotate('%s\n#%s%s' % (e['cand'], e['block'], '' if w0 >= wmin else '\n(%.0f 秒)' % w0), (x0 + w / 2, 0), ha='center',
                        va='center', fontsize=8, color=INK)
            for r in e['rows']:
                t = x0 + (r.get('ai_s_at_mid') or 0)
                if r.get('collisions'):
                    ax.plot([t], [0.42], 'v', ms=5, color=INK)
                if r.get('tk_failure'):
                    ax.plot([t], [-0.42], '^', ms=5, color=INK2)
                if r.get('reboot'):
                    ax.plot([t], [0.0], 'x', ms=7, color='#e34948')
            x0 += w + 8
        ax.set_yticks([])
        ax.set_xlabel('数えた AI の時間を並べたもの [秒](▼ 衝突のある周・▲ 失敗の交代・× 基板の再起動)')
        ax.set_title('%s 1 日の流れ(ブロックの順)' % d['day'], loc='left', color=INK)
        ax.grid(False)
        fig.tight_layout()
        name = 'f7_timeline_%s.png' % d['day'].replace('-', '')
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made[name] = '%s の 1 日の流れ' % d['day']
    # f8 机上と実車
    desk = rep.get('desk_rows') or {}
    if d1 and desk:
        C = d1['compare']['cands']
        pts = [(c, desk[c].get('l1_interv_us'), C[c]['metrics'].get('incidents_per_lap', {}).get('value')) for c in C if c in desk]
        pts = [p for p in pts if p[1] is not None and p[2] is not None]
        if pts:
            fig, ax = plt.subplots(figsize=(6.0, 4.4))
            for c, x, y in pts:
                ax.plot([x], [y], 'o', ms=9, color=color_of(c), mec=SURF, mew=2)
                ax.annotate(c, (x, y), xytext=(6, 4), textcoords='offset points', fontsize=10, color=INK)
            ax.set_xlabel('机上: 介入の集合の L1 [µs](l1_interv_us)')
            ax.set_ylabel('実車: 失敗/周')
            ax.set_title('机上の指標と実車の失敗/周(向きの傾向だけ)', loc='left', color=INK)
            fig.tight_layout()
            fig.savefig(fig_dir / 'f8_desk_vs_real.png', dpi=130)
            plt.close(fig)
            made['f8_desk_vs_real.png'] = '机上の l1_interv_us と実車の失敗/周'
    # f9 2 日目
    if len(days) > 1 and days[1]['compare']['cands']:
        C = {k: v for k, v in days[1]['compare']['cands'].items() if k not in CAP_IDS}
        ids = list(C)
        if ids:
            fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(ids) + 1.6))
            _dot_whisker(ax, ids, [(C[c].get('S') or {}).get('1.0', {}).get('value') for c in ids],
                         [(C[c].get('S') or {}).get('1.0', {}).get('ci90') for c in ids], [color_of(c) for c in ids],
                         xlabel='S = Δ失敗/周 + 1.0 × Δラップ(候補 − B、90% 区間)。左ほど良い')
            ax.set_title('%s 2 日目の S' % days[1]['day'], loc='left', color=INK)
            fig.tight_layout()
            fig.savefig(fig_dir / 'f9_rank_day2.png', dpi=130)
            plt.close(fig)
            made['f9_rank_day2.png'] = '2 日目: T1・T2 の S と 90% 区間'
    return made


# ------------------------------------------------------------------ 本文
def _pm(x, pct=False, fmt='%.2f'):
    return ss._pm(x, pct, fmt)


def _dv(d, k, fmt='%+.2f'):
    x = d.get(k) or {}
    if x.get('value') is None:
        return '—'
    c = x.get('ci90') or [None, None]
    return (fmt % x['value']) + (' [%s–%s]' % (fmt % c[0], fmt % c[1]) if c[0] is not None else '')


def _rb(m):
    """再起動/4 分(元が無ければ「判定できない」)"""
    if not m:
        return '—'
    v = m.get('reboots_per_4min')
    if v is None:
        return '判定できない'
    return '%.2f(%d 回)' % (v, m.get('reboots') or 0)


def _v(m, k, pct=False, fmt='%.2f'):
    x = (m or {}).get(k)
    if isinstance(x, dict):
        if x.get('value') is None:
            return '—'
        return ('%.0f%%' % (100 * x['value'])) if pct else (fmt % x['value'])
    return '—' if x is None else (fmt % x)


def to_markdown(rep):
    L = []
    B = rep['bench'] or {}
    L += ['# 実車のベンチマーク %s' % (B.get('bench') or rep.get('bench_name') or ''), '']
    if rep.get('mock'):
        L += ['> **模擬のベンチマーク**です。10/10 の first_trial の記録を、仮の候補に割り当てて道具を端から端まで動かしたもの。',
              '> 数はモデルの比べではありません(全部 first_trial)。机上の表も試験用の偽物です。', '']
    L += ['- 作成: %s(tools/bench_report.py)。形は plan/bench_spec.md、比べ方は plan/protocol.json(sha256 `%s`、%s)' % (
        rep['created_at'], (rep.get('protocol_sha256') or '')[:16] + '…', '固定の確かめ OK' if rep.get('protocol_sha_ok') else '固定の確かめ できず'),
          '- 日: %s' % '・'.join(d['day'] for d in rep['days']), '']
    # 0 結論
    L += ['## 0. 結論', '']
    for d in rep['days']:
        if d.get('ranking'):
            rk = d['ranking']
            L.append('- %s(1 日目の上位 2 本の選び): T1 = **%s**、T2 = **%s**(λ = %.1f)。1 日目の上位 2 本は、B より良いとは言わない。' % (
                d['day'], rk.get('T1') or 'なし', rk.get('T2') or 'なし', rk['lambda']))
        if d.get('day2'):
            L.append('- %s(2 日目の判定、この日の周だけ): **%s**。推奨のモデルを変えるかは、ユーザーが決める。' % (d['day'], d['day2']['verdict']['text']))
    nchk = sum(len(e.get('checks', [])) for d in rep['days'] for e in d['entries'])
    if nchk:
        L.append('- **説明欄と記録・計画の食い違いが %d 件**ある(下の「外れたこと」)。数を読む前に確かめる。' % nchk)
    L += ['', '## 1. 目的', '',
          '- 候補の学習済みモデルを実車で走らせ、モデル B(推奨)より良いモデルがあるかを決める。',
          '- 今の課題(R1 に速く入るとぶつかる。10/10 は 6,200 eRPM 以上で強い衝突 6/9)への効きを見る。', '']
    # 2 条件
    L += ['## 2. 条件', '', '| 日 | ブロック | AI の時間(数えた) | AI の時間(全部) | 記録の長さ | 記録 | 電池 | 説明欄の check |',
          '|---|---|---|---|---|---|---|---|']
    for d in rep['days']:
        tot = sum(x['ai_s_counted'] or 0 for x in d['compare']['blocks'].values())
        ai_all = sum(e['score'].get('ai_s') or 0 for e in d['entries'])
        dur = sum(e['score'].get('duration_s') or 0 for e in d['entries'])
        L.append('| %s | %d | %.1f 分 | %.1f 分 | %.1f 分 | %d 本 | %s | %s |' % (
            d['day'], len(d['compare']['blocks']), tot / 60, ai_all / 60, dur / 60, len(d['entries']),
            '・'.join(sorted({'#%s' % (e['desc'].get('battery') or '?') for e in d['entries']})),
            '・'.join(sorted({str(e['desc'].get('check')) for e in d['entries'] if e['desc'].get('check')})) or '—'))
    bp = (B.get('baseline') or {}).get('params')
    L += ['', '- 基準の設定(bench_candidates.json の baseline): %s' % (json.dumps(bp, ensure_ascii=False) if bp else '—'),
          '- 記録の開始時の設定と違うもの: %s' % (', '.join(rep['param_mismatch']) or 'なし'),
          '- ソフト: jetformer の git commit `%s`' % rep.get('git_commit'),
          '- 床の基準(data/tof_floor_ref.npy)の sha256: `%s`' % ((rep.get('software') or {}).get('tof_floor_ref_sha256') or '—'),
          '- コース・照明・取り付け・床の基準・運用(ip_status・Jetson の電源)はユーザーのメモから書き足す。', '']
    sw = (rep.get('software') or {}).get('engines') or {}
    if sw:
        L += ['| id | エンジン | sha256(先頭 16) | engine_check.json | TensorRT と FP32 の差 最大 µs | 確かめたあとにエンジンが変わっていない |',
              '|---|---|---|---|---|---|']
        for cid, x in sw.items():
            L.append('| %s | %s | %s | %s | %s | %s |' % (
                cid, x.get('engine') or '—', (x.get('sha256') or '—')[:16], {True: '合格', False: '**不合格**', None: '無い'}[x.get('check_ok')],
                '—' if x.get('actions_max_us') is None else '%.2f' % x['actions_max_us'],
                {True: 'はい', False: '**いいえ(確かめ直す)**', None: '—'}[x.get('check_matches_engine')]))
        L.append('')
    sg = [m for m in rep['param_mismatch'] if 'steer_gain' in m]
    if sg:
        memos = sorted({(e['desc'].get('memo') or '').strip() for d in rep['days'] for e in d['entries']
                        if any(e['log'] in m for m in sg) and (e['desc'].get('memo') or '').strip()})
        L += ['- ステアを基準から変えた記録: %s。理由(説明欄の memo): %s' % (', '.join(sg), ' / '.join(memos) or '書かれていない(ユーザーのメモから書く)'), '']
    # 3 候補の表
    desk = rep.get('desk_rows') or {}
    gates = rep.get('gates_recomputed') or {}
    fams = rep.get('families') or {}
    d1_blocks = {}
    if rep['days'] and rep['days'][0].get('ranking') is not None:
        for b, x in rep['days'][0]['compare']['blocks'].items():
            d1_blocks.setdefault(x['cand'], []).append(str(b))
    L += ['## 3. 候補の表(机上の指標)', '', '- 元: %s%s' % (rep.get('desk_path'), '(試験用の偽物)' if rep.get('desk_fake') else ''),
          '- 門は protocol.json の gates の式で計算し直した。家族 label_r1 は事前登録の直し A1(plan/protocol_amendments.md)で、'
          '門 5 を l1_steer_best_us で比べ、brake_lowspeed_frac ≤ 0.01 を足した(A1 の印)。', '',
          '| id | 表示名 | 家族 | 学習結果 | 仮説 | 1 日目のブロック | risk |', '|---|---|---|---|---|---|---|']
    for c in B.get('candidates', []):
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            c['id'], c.get('label', ''), fams.get(c['id']) or c.get('family') or '—', c.get('run', ''), c.get('hypothesis', ''),
            '・'.join(d1_blocks.get(c['id'], [])) or '—', c.get('risk', '—')))
    L += ['', '| id | l1_best_us | l1_last_us | l1_steer_best_us | l1_thr_best_us | l1_interv_us | r1_app_thr_diff_us | steer_amp_ratio | '
          'trt_median_ms | trt_p99_ms | engine_ok | sanity_ok | brake_lowspeed_frac | desk の門 | 門(計算し直し) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in B.get('candidates', []):
        r = desk.get(c['id'], {})
        g = gates.get(c['id'])
        dg = r.get('gates_pass')
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            c['id'], _v(r, 'l1_best_us'), _v(r, 'l1_last_us'), _v(r, 'l1_steer_best_us'), _v(r, 'l1_thr_best_us'), _v(r, 'l1_interv_us'),
            _v(r, 'r1_app_thr_diff_us'), _v(r, 'steer_amp_ratio', fmt='%.3f'), _v(r, 'trt_median_ms'), _v(r, 'trt_p99_ms'),
            r.get('engine_ok', '—'), r.get('sanity_ok', '—'), _v(r, 'brake_lowspeed_frac', fmt='%.4f'),
            '—' if dg is None else (('合格' if dg else '不合格') + (' ' + ','.join(r.get('gate_fail_reasons') or []) if r.get('gate_fail_reasons') else '')),
            '(掛けない)' if c['id'] == 'B' else ('—' if not g else ('合格' if g['gates_pass'] else '不合格 ' + ','.join(g['gate_fail_reasons']))
                                                 + ('(A1)' if g.get('amendment') else '')
                                                 + ('' if g['agrees_with_desk'] else '(desk と食い違い)'))))
    L += ['', '- 机上の見方: 保留の L1 は人の揺れの床(約 31 µs)より大きく、候補の差は実用の目安(10 µs)より小さい。机上の順位で実車の順位は決めない。']
    # 選んだ手順と入らなかった候補(bench_candidates.json の slot・selected_by・selection.not_selected。bench_spec 3 章、10/21 Fix-all)
    sel = [c for c in B.get('candidates', []) if c.get('slot') or c.get('selected_by')]
    if sel:
        L += ['- 選んだ手順(bench_candidates.json): ' + '・'.join('%s = %s(%s)' % (c.get('slot') or '—', c['id'], c.get('selected_by') or '—')
                                                       for c in sel)]
    ns = ((B.get('selection') or {}).get('not_selected') or []) if isinstance(B.get('selection'), dict) else []
    if ns:
        L += ['- 入らなかった候補: ' + '・'.join('%s(%s)' % (x.get('id'), x.get('reason') or '—') for x in ns)]
    notes = [(c['id'], READING_NOTES[c.get('run')]) for c in B.get('candidates', []) if c.get('run') in READING_NOTES]
    if notes:
        L += ['- 読み方の注(確認 V-models、10/21):'] + ['  - %s: %s' % (cid, n) for cid, n in notes]
        L += ['  - engine_check の「TensorRT と FP32 の差 ≤ 2.5 µs」は 64 フレームでの確かめ。保留 6,240 フレームでは最大 3.3〜5.9 µs'
              '(p99 は 1 µs 未満。人の揺れの床 約 31 µs よりずっと小さい)。']
    L += ['']
    # 4 比べ方
    L += ['## 4. 比べ方(protocol の要約)', '',
          '- 1 ブロック = 1 モデル・AI 側で数える 240 秒。慣らしは数えない(ベンチで初めての候補は R1 の起点 3 つ目から、2 回目以降は 2 つ目から)。',
          '- 主の指標は失敗/周(衝突 + 衝突のあとでない失敗の交代)。点数 S = Δ失敗/周 + λ × Δラップの中央値(λ = 1.0、走る前に固定。直し A4)。',
          '- 90%% 区間は 3 周のかたまりのブートストラップ(%d 回)。差は候補と基準を別々に抽出して取る。' % rep['n_boot'],
          '- 基準はその日の B の全ブロック。時間のずれ(直し A7: B の最初と最後で、ラップか失敗/周の差の 90% 区間が 0 を含まない)があれば'
          '前後の B を内挿し、直す前の値でも上位 2 本を出す。ラップは電池の直しを先に入れる(使えるときだけ)。',
          '- 事前登録の直し(plan/protocol_amendments.md): A1(SLOW の門)・A2(止める条件)・A3(時計で削る)・A4(λ を固定)・'
          'A5(床の基準)・A6(上限の確かめ)・A7(ずれの判定)・A8(走ったあとの説明)・A9(画面の AI の時間)。', '']
    # 5 順位表
    L += ['## 5. 結果の順位表', '']
    for d in rep['days']:
        cm = d['compare']
        P = cm['reference'].get('pooled_B') or {}
        title = '5.1 1 日目(%s)— 上位 2 本を選ぶ' % d['day'] if d.get('ranking') else '5.2 2 日目(%s)— ベストモデル' % d['day']
        L += ['### ' + title, '',
              '| 順位 | ブロック | id | 周 | 失敗/周 [90%] | Δ失敗/周 [90%] | 衝突/周 | 強い衝突/周 | R1 強/通過 | GM/通過 | DE/通過 | 失敗の交代/周 | 交代/周 | ラップ中央値 | ベスト | Δラップ [90%] | きれいな周 | S [90%] | 再起動/4分 | 印 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        bb = '・'.join(str(b) for b in cm['reference']['b_blocks'])
        L.append('| (B) | %s | B | %s | %s | 0 | %s | %s | %s | %s | %s | %s | %s | %s | %s | 0 | %s | 0 | %s | 基準 |' % (
            bb, P.get('laps_counted', 0), _pm(P.get('incidents_per_lap')), _v(P, 'collisions_per_lap'), _v(P, 'strong_collisions_per_lap'),
            _v(P, 'r1_strong_per_pass', True), _v(P, 'gm_coll_per_pass', True), _v(P, 'de_coll_per_pass', True),
            _v(P, 'failure_takeovers_per_lap'), _v(P, 'takeovers_per_lap'), _v(P, 'lap_median_s'), _v(P, 'lap_best_s'), P.get('clean_laps', 0),
            _rb(P)))
        order = [r['id'] for r in (d.get('ranking') or {}).get('rows', [])] or [r['id'] for r in (d.get('day2') or {}).get('rows', [])]
        order += [c for c in cm['cands'] if c not in order]
        rank_of = {r['id']: r for r in (d.get('ranking') or {}).get('rows', [])}
        for cid in order:
            c = cm['cands'][cid]
            m, df = c['metrics'], c['diff']
            rr = rank_of.get(cid, {})
            marks = list(rr.get('marks', []))
            if d.get('day2'):
                d1r = (rep['days'][0].get('ranking') or {}) if rep['days'] and rep['days'][0] is not d else {}
                for tn in ('T1', 'T2'):
                    if d1r.get(tn) == cid:
                        marks.append('1 日目の ' + tn)
                j = next((x for x in d['day2']['rows'] if x['id'] == cid), None)
                if j:
                    marks.append('B より良い' if j['better_than_B'] else '条件 ' + ''.join(
                        '○' if v is True else ('?' if v is None else '×') for v in j['conditions'].values()))
            if cid in CAP_IDS:
                marks.append('初見(順位の外)')
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                rr.get('rank', '—'), '・'.join(str(b) for b in c['blocks']), cid, m.get('laps_counted', 0), _pm(m.get('incidents_per_lap')),
                _dv(df, 'd_incidents_per_lap'), _v(m, 'collisions_per_lap'), _v(m, 'strong_collisions_per_lap'), _v(m, 'r1_strong_per_pass', True),
                _v(m, 'gm_coll_per_pass', True), _v(m, 'de_coll_per_pass', True), _v(m, 'failure_takeovers_per_lap'), _v(m, 'takeovers_per_lap'),
                _v(m, 'lap_median_s'), _v(m, 'lap_best_s'), _dv(df, 'd_lap_median_s'), m.get('clean_laps', 0),
                _dv(c.get('S') or {}, '1.0'), _rb(m), '・'.join(marks) or '—'))
        if d.get('ranking'):
            rk = d['ranking']
            sens = d.get('sensitivity', {})
            L += ['', '- λ = 0.5 の順: %s / λ = 2.0 の順: %s(1.0 の順: %s)' % (
                ' > '.join(sens.get('0.5', [])) or '—', ' > '.join(sens.get('2.0', [])) or '—',
                ' > '.join(r['id'] for r in rk['rows'] if r.get('rank')) or '—'),
                  '- R1 の枠の入れ替え: %s' % (('%s → %s' % (rk['r1_slot_swap']['out'], rk['r1_slot_swap']['in'])) if rk.get('r1_slot_swap') else 'なし'),
                  '- 電池の直し: %s。時間のずれの直し: %s' % ('使った' if d['battery_fit'].get('used') else '使わない(%s)' % d['battery_fit'].get('note', ''),
                                                 '使った(内挿)' if cm['reference']['drift']['flag'] else
                                                 ('判定できない(使わない)' if cm['reference']['drift'].get('undecided') else '使わない')),
                  '- 直す前の S(λ = 1.0、電池の直しも内挿もしない B のプールとの差): %s' % (
                      ' / '.join('%s %s' % (c, '—' if v is None else '%+.2f' % v) for c, v in d['S_uncorrected'].items())
                      if d.get('S_uncorrected') else '直しを使っていないので同じ'),
                  '- 1 日目の上位 2 本は、1 日目の周で選んだので、B より良いとは言わない。']
        if d.get('day2'):
            d1 = rep['days'][0] if rep['days'] and rep['days'][0] is not d else None
            if d1:
                same = ['%s: S %s・失敗/周 %s・周 %s' % (c, _dv(d1['compare']['cands'][c].get('S') or {}, '1.0'),
                                                     _pm(d1['compare']['cands'][c]['metrics'].get('incidents_per_lap')),
                                                     d1['compare']['cands'][c]['metrics'].get('laps_counted', 0))
                        for c in cm['cands'] if c in d1['compare']['cands']]
                if same:
                    L += ['', '- 同じ候補の %s の値(選んだ日なので良い方に偏る): %s' % (d1['day'], ' / '.join(same))]
            L += ['', '- 条件の印は 1〜5 の順(○ 満たす・× 満たさない・? 判定できない。? は満たさないとして扱う)。'
                  '条件 5 = 再起動/4 分(ブロックの AI の時間の全部で数える)が B より 1 回以上多くない。',
                  '- 結論: **%s**。推奨のモデルを変えるかは、ユーザーが決める。' % d['day2']['verdict']['text']]
        L.append('')
    # 6 区間ごと
    L += ['## 6. 区間ごと', '', '| id | 日 | R1 進入 eRPM 中央値 | 進入 ≥ 6,200 の割合 | R1 強/通過 | R1 衝突/通過 | GM 衝突/通過 | DE 衝突/通過 | R1→GM 出口の時間 | DE→次の R1 の時間 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for d in rep['days']:
        cm = d['compare']
        allm = [('B', cm['reference'].get('pooled_B') or {})] + [(c, cm['cands'][c]['metrics']) for c in cm['cands']]
        for cid, m in allm:
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                cid, d['day'], _v(m, 'r1_entry_erpm_median', fmt='%.0f'), _v(m, 'r1_entry_ge6200_share', True) if isinstance(m.get('r1_entry_ge6200_share'), dict)
                else ('—' if m.get('r1_entry_ge6200_share') is None else '%.0f%%' % (100 * m['r1_entry_ge6200_share'])),
                _pm(m.get('r1_strong_per_pass'), True, '%.0f'), _v(m, 'r1_coll_per_pass', True), _v(m, 'gm_coll_per_pass', True),
                _v(m, 'de_coll_per_pass', True), _v(m, 't_r1gm_median_s'), _v(m, 't_rest_median_s')))
    L += ['', '- R1 の進入の帯ごとの強い衝突/通過(10/10 の 006/007 は 1/22・2/7・6/9):', '',
          '| id | 日 | 5,500 未満 | 5,500〜6,200 | 6,200 以上 |', '|---|---|---|---|---|']
    for d in rep['days']:
        for cid, bins in d['entry_bins'].items():
            L.append('| %s | %s | %s | %s | %s |' % (cid, d['day'], bins['<5500'], bins['5500-6200'], bins['>=6200']))
    L += ['', '- SLOW は「R1 の手前で緩めたか」(進入の eRPM が B より下がったか)と、「R1 の強い衝突が減ったか」を分けて読む。', '']
    # 7 電池ごと
    L += ['## 7. 電池ごと', '']
    for d in rep['days']:
        L += ['| 電池 | 日 | ブロック | V_before | V_after | 減り | ラップ中央値 | 失敗/周 | 再起動 |', '|---|---|---|---|---|---|---|---|---|']
        for b, x in d['compare']['blocks'].items():
            dv = (x['V_before'] - x['V_after']) if x['V_before'] is not None and x['V_after'] is not None else None
            L.append('| #%s | %s | %s(%s) | %s | %s | %s | %s | %s | %s |' % (
                x['battery'] or '?', d['day'], b, x['cand'], '—' if x['V_before'] is None else '%.2f' % x['V_before'],
                '—' if x['V_after'] is None else '%.2f' % x['V_after'], '—' if dv is None else '%.2f' % dv,
                _v(x['metrics'], 'lap_median_s'), _v(x['metrics'], 'incidents_per_lap'), x['metrics'].get('reboots', 0)))
        f = d['battery_fit']
        L.append('')
        L.append('- %s の B の周の lap = a + b·V: b = %s 秒/V(90%% 区間 %s、%s 周)。直し: %s' % (
            d['day'], f.get('b_s_per_V', '—'), f.get('ci90', '—'), f.get('n', 0), '使う' if f.get('used') else '使わない(%s)' % f.get('note', '')))
        L.append('')
    # 8 時間のずれ
    L += ['## 8. 時間のずれ(B の 3 回)', '']
    for d in rep['days']:
        L += ['| 日 | B のブロック | 周 | ラップ中央値 | 失敗/周 | R1 強/通過 | 電圧(前→後) |', '|---|---|---|---|---|---|---|']
        for b, x in d['compare']['blocks'].items():
            if x['cand'] != 'B':
                continue
            m = x['metrics']
            L.append('| %s | %s | %s | %s | %s | %s | %s → %s |' % (d['day'], b, m.get('laps_counted', 0), _pm(m.get('lap_median_s')),
                                                               _pm(m.get('incidents_per_lap')), _pm(m.get('r1_strong_per_pass'), True, '%.0f'),
                                                               x['V_before'], x['V_after']))
        dr = d['compare']['reference']['drift']
        L.append('')
        if dr.get('note'):
            L.append('- %s: %s' % (d['day'], dr['note']))
        L.append('- %s: 時間のずれ **%s**(最初と最後の B のラップの差 %s 秒、失敗/周の差の 90%% 区間 %s)' % (
            d['day'], 'あり → 内挿を使う' if dr['flag'] else ('判定できない(B のプールでくらべた)' if dr.get('undecided') else 'なし'),
            '—' if dr.get('d_lap_median_s') is None else '%+.2f' % dr['d_lap_median_s'],
            '—' if not dr.get('d_incidents_ci90') or dr['d_incidents_ci90'][0] is None else '[%+.2f, %+.2f]' % tuple(dr['d_incidents_ci90'])))
        L.append('')
    if len(rep['days']) > 1:
        P1 = rep['days'][0]['compare']['reference'].get('pooled_B') or {}
        P2 = rep['days'][1]['compare']['reference'].get('pooled_B') or {}
        if P1 and P2:
            L += ['- 1 日目と 2 日目の B の違い: ラップ中央値 %s → %s 秒・失敗/周 %s → %s・R1 強/通過 %s → %s(周 %s → %s)。'
                  '日をまたいで比べるのは B 同士だけ(候補は同じ日の B と比べる)' % (
                      _v(P1, 'lap_median_s'), _v(P2, 'lap_median_s'), _v(P1, 'incidents_per_lap'), _v(P2, 'incidents_per_lap'),
                      _v(P1, 'r1_strong_per_pass', True), _v(P2, 'r1_strong_per_pass', True), P1.get('laps_counted', 0), P2.get('laps_counted', 0)), '']
    # 9 上限
    L += ['## 9. 速度上限(初見・G1)', '']
    capd = [(d['day'], c, d['compare']['cands'][c]) for d in rep['days'] for c in d['compare']['cands'] if c in CAP_IDS]
    if capd:
        L += ['| 日 | | 周 | R1 強/通過 | DE 衝突/通過 | 失敗/周 | 衝突/周 | ラップ中央値 |', '|---|---|---|---|---|---|---|---|']
        for day, c, x in capd:
            m = x['metrics']
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (day, c, m.get('laps_counted', 0), _v(m, 'r1_strong_per_pass', True),
                                                              _v(m, 'de_coll_per_pass', True), _v(m, 'incidents_per_lap'), _v(m, 'collisions_per_lap'), _v(m, 'lap_median_s')))
        L.append('- 目安(action_plan 注 4): 衝突/周の点推定 ≤ 1.1・ラップの悪化 ≤ 0.3 秒・旋回中のブレーキ 0。約 20 周なので「初見」。')
        for day, c, x in capd:
            cp = x['metrics'].get('cap') or {}
            L.append('- %s %s: ON で R1 の旋回中(後から見た起点 ta → ta + 0.5 秒)に出たブレーキ **%s フレーム**(始まり %s 回。0 であること。'
                     'ta → GM の出口 − 0.3 秒では %s フレーム)・ON のフレーム %s・実際に変えたフレーム %s' % (
                day, c, cp.get('on_brake_frames_in_r1_turn', '—(記録に cap_* が無い)'), cp.get('on_brake_starts_in_r1_turn', '—'),
                cp.get('on_brake_frames_in_r1_turn_to_tg', '—'), cp.get('on_frames', '—'), cp.get('applied_frames', '—')))
        L.append('- 旋回中のブレーキは時刻で数える(層の区間 R1_turn では数えない。層はブレーキを R1_app の中でしか出さないので、'
                 '位相が遅れて実際の旋回中に出ても区間では 0 になる)。')
    else:
        L.append('- B+上限のブロックは無い(A 案の日だけ)。')
    rows_cap = []
    for d in rep['days']:
        cm = d['compare']
        for cid, m in [('B', cm['reference'].get('pooled_B') or {})] + [(c, cm['cands'][c]['metrics']) for c in cm['cands']]:
            cp = m.get('cap')
            if cp and cp.get('ai_frames'):
                rows_cap.append('| %s | %s | %d | %.1f%% | %d | %s | %d | %d |' % (
                    cid, d['day'], cp['ai_frames'], 100.0 * cp['shadow_cap_frames'] / cp['ai_frames'], cp['shadow_brake_starts'],
                    '%.2f' % (cp['shadow_brake_starts'] / m['laps_counted']) if m.get('laps_counted') else '—',
                    cp.get('shadow_brake_frames_in_r1_turn', 0), cp.get('on_brake_frames_in_r1_turn', 0)))
    if rows_cap:
        L += ['', '- 影の判定(AI_CMD の cap_*。zone_caps 0 の走行でも毎フレーム計算): 上限が掛かったはずの場面', '',
              '| id | 日 | AI のフレーム | 上限・中立・ブレーキのはずの割合 | ブレーキの始まり(影) | 1 周あたり | 影のブレーキのフレーム: R1 の旋回中(ta → +0.5 秒) | ON のブレーキ: R1 の旋回中 |',
              '|---|---|---|---|---|---|---|---|'] + rows_cap + ['']
    else:
        L.append('- 影の判定の欄(AI_CMD の cap_*)は記録に無い(10/21 より前の記録)。')
    g1 = rep.get('cap_replay')
    if g1:
        g = g1['summary']['gate']
        if 'G1_protocol_pass_' + (g1.get('variants') or [''])[0] in g:
            L.append('- **G1(protocol の 4 項目: R1 の合図・誤った合図・旋回中のブレーキ・ラップの損。A 案・B 案はこちら)**(tools/cap_replay.py、%s): %s' % (
                rep.get('cap_replay_path'), '・'.join(
                    '%s %s' % (v, '合格' if g.get('G1_protocol_pass_' + v) else '不合格・未定(%s)' % ', '.join(
                        g.get('G1_protocol_failed_items_' + v, []) + ['判定できない:' + x for x in g.get('G1_protocol_undecided_items_' + v, [])]))
                    for v in g1.get('variants', []))))
        L.append('- G1(6 項目 = g1_replay v1.1。強い衝突の合図・ジャイロの検出も入れた参考): %s' % '・'.join(
            '%s %s' % (v, '合格' if g.get('G1_pass_' + v) else '不合格・未定(%s)' % ', '.join(
                g.get('G1_failed_items_' + v, []) + ['判定できない:' + x for x in g.get('G1_undecided_items_' + v, [])]))
            for v in g1.get('variants', [])))
        if g1.get('errors') or not (g1.get('summary') or {}).get('g1_logs'):
            L.append('- G1 の注意: G1 のログ %d 本・読めなかったログ %d 本(%s)' % (
                len((g1.get('summary') or {}).get('g1_logs') or []), len(g1.get('errors') or {}), '・'.join(g1.get('errors') or {}) or '—'))
    br = rep.get('brake_table')
    if br:
        L.append('- ブレーキの表(tools/brake_table.py): きれいな出来事 %d 回。%s' % (br['table']['n_clean'], '・'.join(
            '%s: +200 µs で %s m/s²%s' % (s, f.get('a_b_at_200us', '—'), '(**仮**: 150 µs 以上 %s 回)' % f.get('n_ge150us') if f.get('provisional') else '')
            for s, f in br['table']['fits'].items())))
    L.append('')
    # 10 図
    L += ['## 10. 図', '']
    for name, cap in rep['figs'].items():
        L += ['![%s](fig/%s)' % (cap, name), '', '- %s(`fig/%s`)' % (cap, name), '']
    if not rep['figs']:
        L += ['- 図はありません(比べる候補が無い)。', '']
    # 11 再現
    L += ['## 11. 再現のコマンド', '', '```bash', 'cd ~/jetformer', 'PY=/home/jetson/miniconda/envs/jetracer/bin/python', rep['command'], '```', '']
    # 12 限界
    L += ['## 12. 限界', '',
          '- 周が少ない: 1 日目は候補 1 本 約 20 周。失敗/周は ±0.34 回/周、ラップは ±0.44 秒、R1 の強い衝突/通過は ±0.18 の幅(10/10 からの見込み)。3 割より小さい差は見えない。',
          '- 1 日目の上位 2 本は 1 日目の周で選んだので、1 日目の値は良い方に偏る。判定は 2 日目の周だけで行う。',
          '- 1 回の走行の中の周は互いに似る(3 周のかたまりで抽出しても、走行ごとの違いは 1 日目には測れない)。',
          '- B の 3 回で直せる時間のずれは直線まで。コース・照明・取り付けが 10/10 と違えば、10/10 の数とは直接比べられない。',
          '- λ = 1.0 は走る前に固定した値(直し A4。結果を見たあとには変えない)。「差が見えない」は「差が無い」ではない。',
          '- 見つける力(10/10 の 006・007 の周から作った模擬の日 400 回、V-protocol 4.2): どの候補も B と同じ日でも、'
          '2 日目に「B より良い」と出るのが 10.5%(ブロックごとの揺れを入れると 17%)。失敗が 3 割減る候補は、1 日目の上位 2 本に 80% 入るが、'
          '2 日目に「良い」と言えるのは 41%。半分になる候補なら 82%。安全が 3 割良くラップが 0.3 秒遅い候補は、λ = 1.0 で 7%。',
          '- 走る順と電池: 候補ごとに、電池の前半か後半か(day1.batteries)が違う。失敗/周は電池で直さない(ラップだけ、傾きがはっきりしたときに直す)。',
          '- 周の定義の細部: 周 j の終わりは「次の AI の R1 の起点」の 0.5 秒前(人が運転した起点は飛ばす)。',
          '- 机上と実車の対応は向きの傾向まで(机上の差はどれも実用の目安 10 µs より小さい。机上の順位で実車の順位を言わない)。',
          '- B+上限は約 20 周の初見で、カード 2(上限を ON で走る判断)の代わりにならない。',
          '- 通過あたりの率の 90% 区間は、protocol どおり 3 周のかたまりのブートストラップ。かたまりが少ないと狭く出るので、'
          '正確な二項の区間も並べた(順位・判定にはブートストラップの区間を使う)。']
    if rep.get('deviations'):
        L += ['', '## 外れたこと(自動で見つけたもの。deviations.md にも書く)', ''] + ['- ' + x for x in rep['deviations']]
    if rep.get('unassigned'):
        L += ['', '## 未割り当ての記録', ''] + ['- %s: %s' % (u['log'], u['description'] or '(説明なし)') for u in rep['unassigned']]
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------ 本体
def software_state(bench, runs_dir=None):
    """候補ごとのエンジンの sha256 と engine_check.json の結果、床の基準の sha256(bench_spec 2 章)。読み取りだけ"""
    runs_dir = Path(runs_dir or config.DATA_DIR / 'runs')
    out = {'engines': {}}
    fl = config.DATA_DIR / 'tof_floor_ref.npy'
    out['tof_floor_ref_sha256'] = sha256(fl) if fl.exists() else None
    for c in (bench or {}).get('candidates', []):
        run = c.get('run')
        if not c.get('id') or not run:
            continue
        rd = runs_dir / run
        engs = sorted(rd.glob('*.engine')) if rd.is_dir() else []
        x = {'run': run, 'engine': engs[0].name if engs else None, 'sha256': None, 'check_ok': None, 'actions_max_us': None,
             'check_matches_engine': None}
        if engs:
            try:
                x['sha256'] = sha256(engs[0])
            except OSError:
                pass
        ck = load_json(rd / 'engine_check.json', None)
        if isinstance(ck, dict):
            x['check_ok'] = bool(ck.get('ok')) if ck.get('ok') is not None else None
            x['actions_max_us'] = ck.get('actions_max_us')
            ce = ck.get('engine') if isinstance(ck.get('engine'), dict) else {}
            if engs and ce.get('mtime_ns') is not None:
                st = engs[0].stat()
                x['check_matches_engine'] = bool(int(ce['mtime_ns']) == st.st_mtime_ns and (ce.get('size') in (None, st.st_size)))
        out['engines'][c['id']] = x
    return out


def git_commit():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def build_report(days, bench_path=BENCH, desk_path=DESK, protocol_path=PROTOCOL, labels_dir=None, raw_dir=None, assign=None,
                 cap_replay_path=None, brake_path=None, n_boot=ss.BOOT_N, min_laps=12, mock=False, logs=None, k_first=3, k_later=2):
    labels_dir = Path(labels_dir or config.DATA_DIR / 'labels')
    raw_dir = Path(raw_dir or config.RAW_DIR)
    bench = load_json(bench_path, None)
    proto = load_json(protocol_path, {})
    sha = sha256(protocol_path) if Path(protocol_path).exists() else None
    sha_file = Path(str(protocol_path).replace('.json', '.sha256'))
    sha_ok = bool(sha and sha_file.exists() and sha in sha_file.read_text())
    if logs is None:
        logs = []
        for dd in days:
            logs += sorted(d.name for d in raw_dir.glob(dd.replace('-', '') + '-*') if (d / 'log.json').exists())
    entries, unassigned = collect(logs, bench, assign, labels_dir, raw_dir)
    entries = score_entries(entries, bench, labels_dir, raw_dir, k_first, k_later)
    devs = []
    if (k_first, k_later) != (3, 2):
        devs.append('慣らしの R1 の起点の数を K = %d(初めて)・%d(2 回目から)にした(protocol は 3・2)' % (k_first, k_later))
    if min_laps != 12:
        devs.append('順位に入れる周の下限を %d にした(protocol は 12)' % min_laps)
    if bench and bench.get('protocol_sha256') and sha and bench['protocol_sha256'] != sha:
        devs.append('bench_candidates.json の protocol_sha256 が今の protocol.json と違う')
    base = ((bench or {}).get('baseline') or {}).get('params') or {}
    mism = set()
    for e in entries:
        p = e['score'].get('params_at_start') or {}
        for k, v in base.items():
            if k in p and p[k] is not None and v is not None and abs(float(p[k]) - float(v)) > 1e-6:
                mism.add('%s(%s: %s ≠ 基準 %s)' % (e['log'], k, p[k], v))
        if e['rerun']:
            devs.append('%s はブロック %s の走り直し(K = 2、残りの秒で数えた)' % (e['log'], e['block']))
        if (e['score'].get('window') or {}).get('ai_s') is not None and not (e['score'].get('window') or {}).get('budget_reached'):
            devs.append('%s(ブロック %s・%s)は数える AI の時間に届かずに終わった(%.0f 秒)' % (
                e['log'], e['block'], e['cand'], (e['score'].get('window') or {}).get('ai_s') or 0))
            lt = (e['score'].get('window') or {}).get('last_takeover')
            if lt and lt.get('counted_as_incident') and not lt.get('handback_after') and lt.get('in_counted_lap'):
                devs.append('%s(ブロック %s・%s): 最後の交代(%.2f 秒、%s)のあと戻しが無い。走行の終わりの止め方なら失敗ではないが、'
                            '失敗/周に 1 件入っている(数え方は protocol のまま。動画・メモで確かめる)' % (
                                e['log'], e['block'], e['cand'], lt['t_s'], lt.get('class')))
    if unassigned:
        devs.append('説明欄から候補に割り当てられない記録が %d 本' % len(unassigned))
    for e in entries:
        sc_ = e['score']
        if sc_.get('error'):
            devs.append('%s(ブロック %s・%s)は読めなかった: %s' % (e['log'], e['block'], e['cand'], sc_['error']))
        if sc_.get('problems'):
            pr = sc_['problems']
            devs.append('%s(ブロック %s・%s)は壊れた・無いファイルを飛ばした %d 件(%s)' % (
                e['log'], e['block'], e['cand'], len(pr), '・'.join(pr[:3]) + (' ほか' if len(pr) > 3 else '')))
        if sc_.get('labels') == 'computed':
            devs.append('%s のラベルのファイルが無く、その場で作った' % e['log'])
    chk = check_deviations(entries)
    for x in chk:
        print('警告: ' + x, file=sys.stderr)
    devs += chk
    day_names = sorted({e['day'] for e in entries})
    out_days = []
    for i, day in enumerate(day_names):
        de = [e for e in entries if e['day'] == day]
        fit = battery_fit(de)
        apply_battery(de, fit)
        cmp_ = compare_day(day, de, n_boot)
        d = {'day': day, 'entries': de, 'battery_fit': fit, 'compare': cmp_}
        c0 = None
        if fit.get('used') or cmp_['reference']['drift'].get('flag'):
            c0 = uncorrected_compare(day, de, n_boot)
            d['S_uncorrected'] = uncorrected_S(day, de, n_boot, c0)
        is_day1 = (i == 0) and not any(x.get('date') == day and i > 0 for x in (bench or {}).get('days', []))
        bdays = [x.get('date') for x in (bench or {}).get('days', [])]
        if bdays and day in bdays:
            is_day1 = bdays.index(day) == 0
        if is_day1:
            d['ranking'] = rank_day1(cmp_, '1.0', min_laps)
            if c0 is not None:      # 直し A7: 直す前の値でも上位 2 本を出し、違えば外れたことに書く(順位は直した値のまま)
                r0 = rank_day1(c0, '1.0', min_laps)
                d['ranking_uncorrected'] = {'T1': r0.get('T1'), 'T2': r0.get('T2')}
                if (r0.get('T1'), r0.get('T2')) != (d['ranking'].get('T1'), d['ranking'].get('T2')):
                    devs.append('%s: 上位 2 本が、直した値(T1 %s・T2 %s)と直す前の値(T1 %s・T2 %s)で違う。'
                                '選びは直した値(protocol の order_of_corrections)のまま' % (
                                    day, d['ranking'].get('T1'), d['ranking'].get('T2'), r0.get('T1'), r0.get('T2')))
            d['sensitivity'] = {lm: [r['id'] for r in rank_day1(cmp_, lm, min_laps)['rows'] if r.get('rank')] for lm in ('0.5', '2.0')}
        else:
            d['day2'] = judge_day2(cmp_, bench, min_laps)
        bins = {}
        for cid in ['B'] + list(cmp_['cands']):
            rows = [r for e in de if e['cand'] == cid for r in e['rows'] if not r.get('reboot')]
            bins[cid] = {}
            for lo, hi, nm in ss.ENTRY_BINS:
                q = [r for r in rows if r.get('entry_erpm') is not None and lo <= r['entry_erpm'] < hi]
                bins[cid][nm] = '%d/%d' % (sum(r['r1_strong'] for r in q), len(q))
        d['entry_bins'] = bins
        out_days.append(d)
        dr = cmp_['reference']['drift']
        if dr.get('undecided'):
            devs.append('%s: %s' % (day, dr.get('note')))
        elif dr.get('note'):
            devs.append('%s: %s' % (day, dr['note']))
        if d.get('day2'):
            und = [r['id'] for r in d['day2']['rows'] if r['conditions'].get('5_reboots_not_1_more') is None]
            if und:
                devs.append('%s: 条件 5(再起動/4 分)が判定できない候補: %s(満たさないとして扱った)' % (day, '・'.join(und)))
        # 基板の再起動が 4 分に 1 回を超えたブロック(protocol safety: 2 回続いたら比べる走りを止める)
        over = [b for b, x in cmp_['blocks'].items() if (x['metrics'].get('reboots_per_4min') or 0) > 1.0]
        if over:
            bl = list(cmp_['blocks'])
            two = any(bl.index(b) + 1 < len(bl) and bl[bl.index(b) + 1] in over for b in over)
            devs.append('%s: 基板の再起動が 4 分に 1 回を超えたブロック %s%s' % (
                day, '・'.join('%s(%s、%.2f 回/4 分)' % (b, cmp_['blocks'][b]['cand'], cmp_['blocks'][b]['metrics']['reboots_per_4min'])
                              for b in over), '。**2 回続いた**(protocol: 比べる走りを止め、ブレーキの測定だけにする)' if two else ''))
        for cid, c in cmp_['cands'].items():
            if cid in CAP_IDS and (c['metrics'].get('cap_brake_in_r1_turn') or 0) > 0:
                devs.append('%s: %s で R1 の旋回中(ta → ta + 0.5 秒)に ON のブレーキが %d フレーム出た(0 であること)' % (
                    day, cid, c['metrics']['cap_brake_in_r1_turn']))
    desk = load_json(desk_path, None)
    drows = desk_rows(desk)
    software = software_state(bench)
    families = {cid: family_of(cid, None, bench, proto, drows.get(cid)) for cid in
                set(drows) | {c.get('id') for c in (bench or {}).get('candidates', []) if c.get('id')}}
    rep = {'schema': SCHEMA, 'created_at': datetime.now().isoformat(timespec='seconds'), 'tool': 'tools/bench_report.py',
           'bench': bench, 'bench_name': (bench or {}).get('bench'), 'bench_path': str(bench_path), 'protocol_path': str(protocol_path),
           'protocol_sha256': sha, 'protocol_sha_ok': sha_ok, 'protocol_version': proto.get('version'),
           'desk_path': str(desk_path), 'desk_fake': bool((desk or {}).get('fake')) if isinstance(desk, dict) else False,
           'desk_rows': drows, 'gates_recomputed': recompute_gates(drows, families) if drows else {}, 'families': families,
           'software': software,
           'days': out_days, 'unassigned': unassigned, 'deviations': devs, 'param_mismatch': sorted(mism), 'n_boot': n_boot,
           'min_laps': min_laps, 'mock': mock, 'git_commit': git_commit(),
           'cap_replay_path': str(cap_replay_path) if cap_replay_path else None,
           'cap_replay': load_json(cap_replay_path, None) if cap_replay_path else None,
           'brake_table': load_json(brake_path, None) if brake_path else None}
    if rep['gates_recomputed']:
        for cid, g in rep['gates_recomputed'].items():
            if not g['agrees_with_desk']:
                devs.append('門の計算し直しが desk_bench.json の gates_pass と食い違う: %s' % cid)
    return rep


def report_json(rep):
    """report.json(bench_spec 13 章の形 + 周の行)"""
    def block_rows(d):
        out = []
        for b, x in d['compare']['blocks'].items():
            out.append({'block': b, 'id': x['cand'], 'logs': x['logs'], 'battery': x['battery'], 'V_before': x['V_before'],
                        'V_after': x['V_after'], 'K': x['K'], 'ai_s_counted': x['ai_s_counted'],
                        'laps_counted': x['metrics'].get('laps_counted', 0), 'metrics': x['metrics'], 'flags': x['flags'],
                        'reruns': x['reruns']})
        return out
    days = []
    for d in rep['days']:
        days.append({'day': d['day'], 'blocks': block_rows(d),
                     'reference': {'pooled_B': d['compare']['reference'].get('pooled_B'), 'drift': d['compare']['reference']['drift'],
                                   'battery_fit': d['battery_fit']},
                     'candidates': {c: {k: v for k, v in x.items()} for c, x in d['compare']['cands'].items()},
                     'ranking': d.get('ranking'), 'sensitivity': d.get('sensitivity'), 'day2': d.get('day2'),
                     'S_uncorrected': d.get('S_uncorrected'),
                     'entry_bins': d['entry_bins'],
                     'laps': [{'log': e['log'], 'block': e['block'], 'cand': e['cand'], 'K': e['K'], 'window': e['score'].get('window'),
                               'drive_run': e.get('drive_run'), 'checks': [t for _, t in e.get('checks', [])],
                               'rows': [{k: v for k, v in r.items() if k != 'infer_ms'} for r in e['rows']]} for e in d['entries']]})
    out = {k: rep[k] for k in ('schema', 'created_at', 'tool', 'bench_name', 'bench_path', 'protocol_path', 'protocol_sha256',
                               'protocol_sha_ok', 'desk_path', 'desk_fake', 'gates_recomputed', 'families', 'software', 'unassigned', 'deviations',
                               'param_mismatch', 'n_boot', 'min_laps', 'mock', 'git_commit', 'figs', 'command')}
    out['bench'] = rep['bench_name']
    out['days'] = days
    return out


def write_report(rep, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep['figs'] = make_figs(rep, out_dir / 'fig')
    (out_dir / 'README.md').write_text(to_markdown(rep), encoding='utf-8')
    tmp = out_dir / 'report.json.tmp'
    tmp.write_text(json.dumps(report_json(rep), ensure_ascii=False, indent=1, default=ss._json_default), encoding='utf-8')
    os.replace(tmp, out_dir / 'report.json')
    dv = out_dir / 'deviations.md'
    head = '# 外れたこと(protocol.json から)\n\n- 自動で見つけたもの(tools/bench_report.py、%s)。人が見つけたことは下に足す。\n\n' % rep['created_at']
    body = '\n'.join('- ' + x for x in rep['deviations']) or '- なし'
    if dv.exists():
        old = dv.read_text(encoding='utf-8')
        marker = '\n## 人が書いたこと\n'
        manual = old[old.index(marker):] if marker in old else marker + '\n'
    else:
        manual = '\n## 人が書いたこと\n\n'
    dv.write_text(head + '## 自動\n\n' + body + '\n' + manual, encoding='utf-8')
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--day', action='append', default=[], help='日(YYYYMMDD か YYYY-MM-DD)。2 日目は 2 つ並べる')
    ap.add_argument('--logs', nargs='*', default=None, help='ログを直接指定(--day の代わり)')
    ap.add_argument('--bench', default=str(BENCH))
    ap.add_argument('--desk', default=str(DESK))
    ap.add_argument('--protocol', default=str(PROTOCOL))
    ap.add_argument('--labels', default=str(config.DATA_DIR / 'labels'))
    ap.add_argument('--raw', default=str(config.RAW_DIR))
    ap.add_argument('--assign', default=None, help='{ログ: 説明欄の文} の JSON')
    ap.add_argument('--cap-replay', default=None, help='tools/cap_replay.py の cap_replay.json')
    ap.add_argument('--brake', default=None, help='tools/brake_table.py の brake_table.json')
    ap.add_argument('--boot', type=int, default=ss.BOOT_N)
    ap.add_argument('--min-laps', type=int, default=12, help='順位に入れる周の下限(protocol は 12)')
    ap.add_argument('--mock', action='store_true', help='模擬のベンチマーク(README に印を付ける)')
    ap.add_argument('--k-first', type=int, default=3, help='ベンチで初めての候補の慣らしの R1 の起点の数(protocol は 3)')
    ap.add_argument('--k-later', type=int, default=2, help='2 回目から・走り直しの K(protocol は 2)')
    ap.add_argument('--out', default=None, help='既定 ~/jetformer_reports/<bench>/')
    args = ap.parse_args(argv)
    assign = load_json(args.assign, None) if args.assign else None
    rep = build_report(args.day, args.bench, args.desk, args.protocol, args.labels, args.raw, assign, args.cap_replay, args.brake,
                       args.boot, args.min_laps, args.mock, args.logs, args.k_first, args.k_later)
    rep['command'] = '$PY tools/bench_report.py ' + ' '.join(a if ' ' not in a else repr(a) for a in (argv if argv is not None else sys.argv[1:]))
    out = Path(args.out) if args.out else Path('/home/jetson/jetformer_reports') / (rep['bench_name'] or 'real_bench')
    write_report(rep, out)
    for d in rep['days']:
        if d.get('ranking'):
            print(d['day'], 'T1', d['ranking']['T1'], 'T2', d['ranking']['T2'])
        if d.get('day2'):
            print(d['day'], d['day2']['verdict']['text'])
    print('書き出し:', out / 'README.md', out / 'report.json', '図 %d 枚' % len(rep['figs']))
    return rep


if __name__ == '__main__':
    main()
