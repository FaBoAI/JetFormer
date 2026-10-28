"""反証 8-1 回目・係 tools・台本 6(読むだけ): 本物の 10/10・10/23・10/24 で、⑤ collect.py(--day・--ignore-gain・
--ignore-gain --first-only)・③ standing_3lap(--day)・④ quick_eval_by_params(--day・--ignore gain)・② runs_1103(--day)を、直す前
(tests/fixtures/fix8_tools_decided_before/*.orig。⑤ の中で読む ③ も同じ版)と今で流し、判定に効く値(⑤ の組・by_launch・compare・
not_reached、③ の数える発進・最速・25 秒以下、④ の組の止まってから 3 周・動く → A1)を比べる。data/raw・labels は読むだけ。出力は
../tmp_review_tools_r1/real/ に書き(終わって消す)、比べの結果だけ out/real_cmp.txt に。

  jf-heavy --lock none --mem 1500M -- python -B real_cmp.py
"""
import contextlib
import io
import json
import shutil
import sys

import common as C
from common import t8
from test_fix3_r1_2 import load_mod  # noqa: E402

DAYS = ('20261010', '20261023', '20261024')
ARGS = {'plain': [], 'ig': ['--ignore-gain'], 'igf': ['--ignore-gain', '--first-only']}


def run_all(ver, T):
    s3m, qpm = t8.mods(ver)
    res = {}
    saved = sys.modules.get('standing_3lap')
    sys.modules['standing_3lap'] = s3m
    try:
        for d in DAYS:
            for an, a in ARGS.items():
                m = load_mod('c_real_%s_%s_%s' % (ver, d, an), t8.VER[ver][0])
                out = T / ('c5_%s_%s_%s.json' % (ver, d, an))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    m.main(['--day', d, '--out', str(out)] + a)
                res[('c5', d, an)] = (json.loads(out.read_text(encoding='utf-8')), buf.getvalue())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                s3m.main(['--day', d, '--json', str(T / ('s3_%s_%s.json' % (ver, d))), '--md', str(T / ('s3_%s_%s.md' % (ver, d)))])
                qpm.main(['--day', d, '--out', str(T / ('qp_%s_%s' % (ver, d)))])
                qpm.main(['--day', d, '--ignore', 'gain', '--out', str(T / ('qpig_%s_%s' % (ver, d)))])
            res[('s3', d)] = json.loads((T / ('s3_%s_%s.json' % (ver, d))).read_text(encoding='utf-8'))
            res[('qp', d)] = json.loads((T / ('qp_%s_%s' % (ver, d)) / 'by_params.json').read_text(encoding='utf-8'))
            res[('qpig', d)] = json.loads((T / ('qpig_%s_%s' % (ver, d)) / 'by_params.json').read_text(encoding='utf-8'))
            res[('s3md', d)] = (T / ('s3_%s_%s.md' % (ver, d))).read_text(encoding='utf-8')
            res[('qpmd', d)] = (T / ('qp_%s_%s' % (ver, d)) / 'by_params.md').read_text(encoding='utf-8')
            rm = load_mod('runs_real_%s_%s' % (ver, d), t8.FIX / 'runs_1103.py.orig' if ver == 'old' else t8.NEW_RUNS)
            sys.argv = ['runs_1103.py', '--day', d, '--json']
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rm.main()
            res[('runs', d)] = buf.getvalue()
    finally:
        if saved is not None:
            sys.modules['standing_3lap'] = saved
        else:
            sys.modules.pop('standing_3lap', None)
    return res


def c5_key(j):
    out = []
    for g in j['summary']:
        out.append({'key': [g.get(k) for k in ('session', 'model', 'gain', 'speed_ceiling', 'ceil_left_mps')],
                    'by_launch': {k: {kk: v.get(kk) for kk in ('n', 'logs', 'to_r1_median_s', 'loss_median_s', 'slip_max',
                                                                 'v_lh_median_mps', 'collisions_first_lap', 'restarts')}
                                  for k, v in (g.get('by_launch') or {}).items()},
                    'compare': g.get('compare'),
                    'not_reached': {k: {'n': v['n'], 'logs': v['logs'], 'collisions': v['collisions']}
                                    for k, v in (g.get('not_reached') or {}).items()},
                    'excluded': {k: g.get(k) for k in ('midrun_excluded', 'trial_excluded', 'midrun_slip', 'trial_slip',
                                                       'midrun_slip_unmeasured', 'trial_slip_unmeasured', 'first_lap_gap_on')
                                 if g.get(k)}})
    return out


def s3_key(R):
    out = []
    for r in R if isinstance(R, list) else R.get('results', []):
        out.append([r['log'], [(s['t_ai'], s.get('counted'), s.get('from_move_s'), s.get('valid')) for s in r.get('standing') or []]])
    return out


def qp_key(Q):
    gs = Q.get('groups') or []
    return [{k: g.get(k) for k in ('text', 'standing_n', 'standing_best_s', 'standing_list_s', 'standing_excluded_n',
                                    'standing_le25', 'standing_clean_le25', 'standing_clean_best_s', 'launch_to_r1_median_s',
                                    'launch_n')} for g in gs]


def main():
    T = C.fresh('real')
    (T / 'o').mkdir()
    (T / 'n').mkdir()
    old = run_all('old', T / 'o')
    new = run_all('new', T / 'n')
    lines = []
    for k in sorted(old, key=str):
        o, n = old[k], new[k]
        if k[0] == 'c5':
            jo, so = o
            jn, sn = n
            same_json = json.dumps(jo, sort_keys=True) == json.dumps(jn, sort_keys=True)
            same_out = C.t7.no_path(so) == C.t7.no_path(sn)
            same_key = c5_key(jo) == c5_key(jn)
            ab_new = [(r['log'], r['t_ai'], r['after_abort']) for r in jn['launches'] if r.get('after_abort')]
            ab_old = [(r['log'], r['t_ai'], r['after_abort']) for r in jo['launches'] if r.get('after_abort')]
            lines.append('%s: JSON 同じ %s・標準出力 同じ %s・判定に効く値(組・by_launch・compare・not_reached・外した本)同じ %s' % (
                k, same_json, same_out, same_key))
            if ab_new != ab_old:
                lines.append('   印 前 %s' % ab_old)
                lines.append('   印 今 %s' % ab_new)
            for g in jn['summary']:
                if g.get('compare'):
                    lines.append('   今の組 %s: %s・%s' % (g.get('session'), C.verdict(g), g['compare'].get('d_to_r1_s')))
            if not same_key:
                for a, b in zip(c5_key(jo), c5_key(jn)):
                    if a != b:
                        lines.append('   違う組 前 %s' % json.dumps(a, ensure_ascii=False)[:600])
                        lines.append('   違う組 今 %s' % json.dumps(b, ensure_ascii=False)[:600])
        elif k[0] == 's3':
            lines.append('%s: 発進と数える・3 周 同じ %s' % (k, s3_key(o) == s3_key(n)))
        elif k[0] in ('qp', 'qpig'):
            lines.append('%s: 組の止まってから 3 周・動く → A1 同じ %s' % (k, qp_key(o) == qp_key(n)))
            if qp_key(o) != qp_key(n):
                for a, b in zip(qp_key(o), qp_key(n)):
                    if a != b:
                        lines.append('   前 %s' % json.dumps(a, ensure_ascii=False))
                        lines.append('   今 %s' % json.dumps(b, ensure_ascii=False))
        elif k[0] in ('s3md', 'qpmd'):
            lines.append('%s: md 同じ %s' % (k, o == n))
        else:
            same = o.replace(t8.MARK_NEW, t8.MARK_OLD) == n.replace(t8.MARK_NEW, t8.MARK_OLD)
            lines.append('%s: 印の文のほか同じ %s・バイト単位で同じ %s' % (k, same, o == n))
    C.write('real_cmp.txt', lines)
    shutil.rmtree(T)
    C.cleanup()


if __name__ == '__main__':
    main()
