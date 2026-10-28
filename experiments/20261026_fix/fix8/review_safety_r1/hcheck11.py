"""人が CH3 を人側にして AI 側へ戻したとき、ON の中止(ASSIST_GIVEUP)・リカバリーの手順が戻したあとに残らないか(fz・hm の場面で、
人側の区間の前に ON の中止・リカバリーがあった場面)。戻したあとの最初の 0.5 秒の AI 側の行の補助の種類と、人側の区間の中に
処理したフレーム(select False の行)があったかを見る。
    python -I -B hcheck11.py → out/hcheck11.json・標準出力
"""
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'hcheck')
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402

A = h11.A


def main():
    h11.check_now()
    rows = []
    for kind in ('fz', 'hm'):
        for f in sorted(glob.glob(str(HERE / 'out' / ('exp11_%s_*.json' % kind)))):
            rows += json.loads(Path(f).read_text())['rows']
    cand = [r for r in rows if r['sd'].get('human') and (r['on']['giveup_t'] is not None or r['on']['recoveries'] > 0)]
    out = {'n_cand': len(cand), 'checked': 0, 'spans': 0, 'kept_after_return': [], 'no_human_frame': 0}
    try:
        for r in cand:
            sd = r['sd']
            res = h11.run('now', h11.ON(exp11.params_of(sd)), exp11.build(sd), wrap=False)
            recs = [((t - h11.T0) / 1e9, x) for t, x in res[1].rows]
            out['checked'] += 1
            for a, b, thr in sd['human']:
                pre = [x for t, x in recs if t < a and not x['hold'] and x['select']]
                if not pre:
                    continue
                last_kind = int(pre[-1]['assist'])
                if last_kind not in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP):
                    continue        # 人側にしたとき、ON は手順の途中でも中止でもなかった
                out['spans'] += 1
                hum = [x for t, x in recs if a <= t < b and not x['select']]
                if not hum:
                    out['no_human_frame'] += 1
                after = [(round(t, 3), int(x['assist']), bool(x['hold'])) for t, x in recs if b <= t < b + 0.5 and x['select']]
                first_ai = next((k for t, k, h in after if not h), None)
                if first_ai in (A.ASSIST_RECOVERY, A.ASSIST_GIVEUP):
                    out['kept_after_return'].append({'i': sd.get('i'), 'kind': sd['kind'], 'span': (a, b, thr),
                                                     'before': last_kind, 'after': after[:6]})
    finally:
        h11.cleanup()
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in out.items()}), out['kept_after_return'][:5])
    (HERE / 'out' / 'hcheck11.json').write_text(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
