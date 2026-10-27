"""反証 3 回目・係 record: 手順書 7 章「走った値が計画と違った → その本は数えない(memo に「試し」)」を ⑤ collect.py が読むか。
読むだけ・tmp だけ。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/refute3_record/probe_memo.py <tmp の場所>

回 3: OFF 4 本(3-1・3-3・3-5・3-7)と ON 4 本(3-2・3-4・3-6・3-8)。3-6 の 1 本目は速度ゲインを 1.02 で走ってしまい(表は 0.92)、
memo=試し と書いて 3-6 を走り直した(手順書 7 章)。試しの本は速く R1 に着く。
  A: 試しの本が無いとき(本物の 4 本対 4 本)
  B: 試しの本がある(memo=試し・ゲイン 1.02)。⑤ --ignore-gain(手順書 6 章の決める出力)
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

OUT = Path(__file__).resolve().parent / 'out'
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


def case(tmp, name, with_trial):
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
    if with_trial:
        D.add2(17, dict(GON, throttle_gain=1.02), memo='試し(ゲイン1.02で走った。走り直す)', **on_kw(48))
    D.add2(17, GON, **on_kw(36))
    D.add2(18, G)
    D.add2(19, GON, **on_kw(39))
    j, out, _, _ = D.run(name='m_' + name, args=['--ignore-gain'])
    lines = [x for x in out.splitlines() if 'ON − OFF' in x or x.strip().startswith(('on ', 'off ', '11/03'))]
    return {'lines': lines, 'on_logs': [g['by_launch'].get('on', {}).get('logs') for g in j['summary']]}


def main():
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    res = {'A_no_trial': case(tmp, 'a', False), 'B_with_trial_memo': case(tmp, 'b', True)}
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_memo.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in res.items():
        print('==', k)
        for x in v['lines']:
            print('  ', x)
        print('   ON の本:', v['on_logs'])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
