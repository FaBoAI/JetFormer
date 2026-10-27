"""r3-r3-rec-3 の確かめ(読むだけ・作り物の記録は引数の tmp だけ。out には書かない)。
probe_memo.py と同じ回 3 に、次を足して ⑤ collect.py を流す。
  A 試しなし --ignore-gain / B 試しあり --ignore-gain / C 試しあり(ゲインで分ける)/ D 試しあり --ignore-gain --skip 試しの本
  E 発進の前進の取り違え(3-6 を OFF で走ってしまい memo=試し。走り直しは ON)--ignore-gain
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import test_fix3_r1_2 as T  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=%s'


class Day2(T.Day):
    def add2(self, block, params, memo='x', **kw):
        log = '20261103-13%02d00' % (self.n * 5)
        self.n += 1
        kw.setdefault('dur', 30.0)
        write_standing_log(self.raw, log, run='runB', description=DESC % (self.n, block, memo), ramp_s=0.5, params=params, **kw)
        return log


def on_kw(k30):
    t = k30 / 30.0
    return dict(t_ai=t, t_move=t + 0.1, launch=(t, 0.5, 1))


def case(tmp, name, trial=None, args=('--ignore-gain',), skip_trial=False):
    d = tmp / name
    d.mkdir(parents=True)
    D = Day2(d)
    G = dict(T.PRM, throttle_gain=0.92)
    GON = dict(T.ON, throttle_gain=0.92)
    D.add2(12, G)
    D.add2(13, GON, **on_kw(30))
    D.add2(14, G)
    D.add2(15, GON, **on_kw(30))
    D.add2(16, G)
    tl = None
    if trial == 'gain':
        tl = D.add2(17, dict(GON, throttle_gain=1.02), memo='試し(ゲイン1.02で走った。走り直す)', **on_kw(48))
    elif trial == 'launch':
        tl = D.add2(17, G, memo='試し(発進の前進を入れ忘れた。走り直す)', t_move=2.0)
    D.add2(17, GON, **on_kw(36))
    D.add2(18, G)
    D.add2(19, GON, **on_kw(39))
    a = list(args) + (['--skip', tl] if skip_trial and tl else [])
    j, out, _, _ = D.run(name='v_' + name, args=a)
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or '外した' in x or '試し' in x
             or x.strip().startswith(('on ', 'off ', '11/03'))]
    trial_rows = [{k: r.get(k) for k in ('log', 'launch_state', 'to_r1_s', 'midrun', 'session', 'gain')}
                  for r in j['launches'] if tl and r['log'] == tl]
    trial_row_keys_with_memo = [k for r in j['launches'] if tl and r['log'] == tl for k, v in r.items()
                                if 'memo' in str(k) or '試し' in json.dumps(v, ensure_ascii=False)]
    return {'trial_log': tl, 'args': a, 'lines': lines, 'trial_rows': trial_rows,
            'trial_row_memo_fields': trial_row_keys_with_memo,
            'json_has_試し': '試し' in json.dumps(j, ensure_ascii=False),
            'groups': [{'key': (g.get('session'), g.get('gain'), g.get('gain_range')),
                        'on': g['by_launch'].get('on', {}).get('logs'), 'off': g['by_launch'].get('off', {}).get('logs'),
                        'all_ok': (g.get('compare') or {}).get('all_ok'),
                        'd_to_r1_s': (g.get('compare') or {}).get('d_to_r1_s')} for g in j['summary']]}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        res = {'A_no_trial': case(tmp, 'a'),
               'B_trial_gain_ig': case(tmp, 'b', 'gain'),
               'C_trial_gain_nog': case(tmp, 'c', 'gain', args=()),
               'D_trial_gain_ig_skip': case(tmp, 'd', 'gain', skip_trial=True),
               'E_trial_launch_ig': case(tmp, 'e', 'launch')}
        print(json.dumps(res, ensure_ascii=False, indent=1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
