#!/usr/bin/env python3
"""DE の揺れの表(2026-10-24、10/23 の課題「DE の衝突は速さでなく GM の出口のステアの揺れ」)

    python tools/de_swing.py --day 20261024                         # その日の AI の記録 → 表を出す
    python tools/de_swing.py --day 20261024 --assign A.json --out DIR   # DIR/de_swing.json・de_swing.md

10/23 の公式 8 本では、AI は GM の左の切りを戻すのが人より 1〜2 フレーム遅く、DE に入ると右へ行き過ぎ、0.6 秒から左へ
切り返したところで当たった(experiments/20261024_day1_review)。2 日目の試し(先読み 4・予測の重ね合わせなど)をこの数で比べる。

DE の通過 = ラベル(data/labels/<ログ>.npz)の loc_code が DE(3)の続く区間で、30 フレーム(1 秒)以上・ずっと AI が操作していたもの。
  overshoot_us  入って 0.5 秒の右への行き過ぎ(車に出したステア out − 中央 の最小。− = 右)
  late_left_us  0.6〜1.0 秒(衝突があればその前まで)の左への切り(最大。+ = 左)
  hit           区間の中か、終わりから 0.3 秒以内に AI の衝突
設定ごと(モデル・ステアと速度のゲイン・先読み・重ね合わせ)にまとめ、通過あたりの衝突と、行き過ぎ・切り返しの中央値を出す。
読むだけ(numpy だけ)。説明欄は section_scorecard と同じ読み方(シーンの説明の直し・--assign の上書き)。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import section_scorecard as ss  # noqa: E402

DE_CODE = 3
FPS = 30
MIN_FRAMES = 30
CARRY = 9                     # 区間の終わりから 0.3 秒
EARLY_S, LATE_S = (0.0, 0.5), (0.6, 1.0)
GROUP_KEYS = ('steer_gain', 'throttle_gain', 'lookahead', 'ensemble_j', 'tof_guard')


def passes(lab, center):
    """1 本のログの DE の通過 [{t_s, frames, hit, overshoot_us, late_left_us}]"""
    de = lab['loc_code'] == DE_CODE
    edges = np.flatnonzero(np.diff(np.r_[0, de.astype(np.int8), 0]))
    t, ai, coll, steer = lab['t_s'], lab['ai'], lab['collision'], lab['out'][:, 0] - center
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < MIN_FRAMES or not ai[a:b].all():
            continue
        b2 = min(len(de), b + CARRY)
        h = np.flatnonzero(coll[a:b2] & ai[a:b2])
        end = a + int(h[0]) if len(h) else b
        rel = t[a:end] - t[a]
        s = steer[a:end]
        e = s[(rel >= EARLY_S[0]) & (rel <= EARLY_S[1])]
        late = s[(rel >= LATE_S[0]) & (rel <= LATE_S[1])]
        out.append({'t_s': round(float(t[a]), 2), 'frames': int(b - a), 'hit': bool(len(h)),
                    'overshoot_us': round(float(e.min()), 1) if len(e) else None,
                    'late_left_us': round(float(late.max()), 1) if len(late) else None})
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None


def summarize(ps):
    n = len(ps)
    k = sum(p['hit'] for p in ps)
    return {'passes': n, 'hits': k, 'hit_rate': round(k / n, 3) if n else None,
            'overshoot_med_us': _med([p['overshoot_us'] for p in ps]),
            'late_left_med_us': _med([p['late_left_us'] for p in ps]),
            'overshoot_lt_120': sum(1 for p in ps if p['overshoot_us'] is not None and p['overshoot_us'] < -120)}


def setting_key(run, params):
    p = params or {}
    return '%s | %s' % (run or '?', ' '.join('%s=%s' % (k, p.get(k)) for k in GROUP_KEYS))


def build(logs, labels_dir, raw_dir=None, assign=None):
    rows, groups = [], {}
    for lg in logs:
        pj, pn = Path(labels_dir) / (lg + '.json'), Path(labels_dir) / (lg + '.npz')
        if not (pj.exists() and pn.exists()):
            continue
        info = json.loads(pj.read_text(encoding='utf-8'))
        if not info.get('ai_log'):
            continue
        if assign and lg in assign:
            desc_raw = assign[lg]
        else:
            desc_raw, _ = ss.effective_description(lg, info.get('description', ''), raw_dir)
        desc = ss.parse_description(desc_raw)
        center = float((info.get('ref_us') or {}).get('steer_center', 1500.0))
        with np.load(pn) as z:
            lab = {k: z[k] for k in ('loc_code', 't_s', 'ai', 'collision', 'out')}
        ps = passes(lab, center)
        key = setting_key(info.get('run'), info.get('params_at_start'))
        row = {'log': lg, 'ai_no': desc.get('ai_no'), 'cand': desc.get('cand'), 'block': desc.get('block'),
               'run': info.get('run'), 'setting': key, 'pass_list': ps, **summarize(ps)}
        rows.append(row)
        groups.setdefault(key, []).extend(ps)
    return {'rows': rows, 'groups': [{'setting': k, **summarize(v)} for k, v in sorted(groups.items())]}


def to_md(res, day):
    L = ['# DE の揺れの表 %s' % day, '',
         '行き過ぎ = DE に入って 0.5 秒の右への最大(− = 右)。切り返し = 0.6〜1.0 秒の左への最大(+ = 左)。'
         '10/23 の公式 8 本: 行き過ぎの中央値 ぶつからない −73・ぶつかった −141 µs、切り返し 103・125 µs、通過あたりの衝突 10/30。', '',
         '## 設定ごと', '', '| 設定(モデル・ゲイン・先読み・重ね合わせ・ToF ガード) | 通過 | 衝突 | 率 | 行き過ぎ | 切り返し | −120 µs 超 |',
         '|---|---|---|---|---|---|---|']
    for g in res['groups']:
        L.append('| %s | %d | %d | %s | %s | %s | %d |' % (g['setting'], g['passes'], g['hits'], g['hit_rate'],
                                                      g['overshoot_med_us'], g['late_left_med_us'], g['overshoot_lt_120']))
    L += ['', '## 記録ごと', '', '| 記録 | ai_no | cand | 通過 | 衝突 | 行き過ぎ | 切り返し |', '|---|---|---|---|---|---|---|']
    for r in res['rows']:
        L.append('| %s | %s | %s | %d | %d | %s | %s |' % (r['log'], r['ai_no'], r['cand'], r['passes'], r['hits'],
                                                       r['overshoot_med_us'], r['late_left_med_us']))
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--day', required=True, help='YYYYMMDD')
    ap.add_argument('--labels', default=str(ROOT / 'data' / 'labels'))
    ap.add_argument('--raw', default=None)
    ap.add_argument('--assign', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    day = a.day.replace('-', '')
    logs = sorted(p.stem for p in Path(a.labels).glob('%s-*.json' % day))
    assign = json.loads(Path(a.assign).read_text(encoding='utf-8')) if a.assign else None
    res = build(logs, a.labels, a.raw, assign)
    md = to_md(res, day)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'de_swing.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        (out / 'de_swing.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
