#!/usr/bin/env python3
"""直す 1 回目(fix_round1)の試験が、決まり・止めの仕組みを壊したときに落ちるか(わざと壊す試し)。

c3/desk_v3.py・c3/eval_v3.py・fix2/test_desk_v3.py の写しを tmp に置き、1 か所ずつ壊して、写しの試験を回す。
本物の台本・試験は読むだけ(sha256 を前後で照らす)。書くのは tmp_fix1/mut の下だけで、終わったら消す。torch・GPU を使わない。

    cd /home/jetson/jetformer/experiments/20261026_fix/fix2
    /home/jetson/miniconda/envs/jetracer/bin/python -I mutate_round1.py [名前 ...]    # 中で jf-heavy --lock none --mem 1500M
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

FIX = Path('/home/jetson/jetformer/experiments/20261026_fix')
C3 = FIX / 'c3'
TEST = FIX / 'fix2' / 'test_desk_v3.py'
TMP = FIX / 'fix2' / 'tmp_fix1' / 'mut'
OUT = Path('/home/jetson/jetformer_reports/fix_20261026/fix2/mutate_round1.json')
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
HEAVY = ['/home/jetson/bin/jf-heavy', '--lock', 'none', '--mem', '1500M', '--need-mb', '600', '--']

# (名前, 指摘, ファイル, 元の文, 壊した文)。元の文は写しの中にちょうど 1 回。
# EQUIVALENT: 壊しても動きが変わらない(試験では見分けられない)もの → 理由
EQUIVALENT = {'e1_final_no_complete': '種のそろわない腕は keep = None なので、all(keep is not None) がもう偽になる(complete は念のための重ね)'}
MUT = [
    # F1
    ('e1_incomplete_decides', 'F1', 'eval_v3.py', "        if a.get('complete') is not True:\n", "        if False:\n"),
    ('e1_final_no_complete', 'F1', 'eval_v3.py', "'final': complete and all(out[a]['keep']", "'final': all(out[a]['keep']"),
    # F3
    ('E1_ignore_bad', 'F3', 'eval_v3.py', "        elif v == '悪い':\n            keep, why = False,", "        elif False:\n            keep, why = False,"),
    ('vs5_bad_never', 'F3', 'eval_v3.py', "    bad_b = db >= tb and all(v > fb for v in sb)\n", "    bad_b = False\n"),
    ('vs5_bad_last_never', 'F3', 'eval_v3.py', "        bad_l = dl >= tl and all(v > fl for v in sl)\n", "        bad_l = False\n"),
    ('vs5_bad_drop_all', 'F3', 'eval_v3.py', "    bad_b = db >= tb and all(v > fb for v in sb)\n", "    bad_b = db >= tb\n"),
    ('vs5_bad_last_drop_all', 'F3', 'eval_v3.py', "        bad_l = dl >= tl and all(v > fl for v in sl)\n", "        bad_l = dl >= tl\n"),
    ('complete_no_last', 'F3', 'eval_v3.py', "'complete': all(s in ok_rows and ok_rows[s].get('l1_last_us') is not None for s in exp)",
     "'complete': all(s in ok_rows for s in exp)"),
    ('exp_uses_queue', 'F3', 'eval_v3.py', "    exp = [0, 1] if f['protocol'] is True else [0, 1, 2, 3]\n",
     "    exp = [0, 1] if f['queue'] is True else [0, 1, 2, 3]\n"),
    # F4
    ('desk_cross_always_pass', 'F4', 'desk_v3.py', "'pass': bool(same_key and mx <= CROSS_LIMIT_US)", "'pass': True"),
    ('h1_cross_always_pass', 'F4', 'eval_v3.py', "'pass': bool(same_key and same_genre and mx <= D.CROSS_LIMIT_US)", "'pass': True"),
    ('check_names_no_lock2', 'F4', 'eval_v3.py', "    if not D.LOCK2.exists() and not allow_before_lock2:\n", "    if False:\n"),
    ('desk_predict_no_lock2', 'F4', 'desk_v3.py',
     "        raise SystemExit('protocol_v3_lock2.json がまだ無いので、v3 の予測はしません(protocol_v3 lock.second_lock)')\n",
     "        pass\n"),
    ('desk_trt_no_lock2', 'F4', 'desk_v3.py', "        raise SystemExit('protocol_v3_lock2.json がまだ無いので、v3 の TensorRT の仕事はしません')\n",
     "        pass\n"),
    ('tab_stamp_ignored', 'F4', 'desk_v3.py',
     "if 'weights_size' in tab and not stamp_same(tab['weights_size'], tab['weights_mtime_ns'], rd / 'best.pth'):", "if False:"),
    ('trt_engine_current_ignored', 'F4', 'desk_v3.py', "lst = [(j, st) for j, st in T.get(name, []) if engine_current(run_dir, st)]",
     "lst = [(j, st) for j, st in T.get(name, [])]"),
    ('pick_last_no_compare', 'F4', 'desk_v3.py', "if isinstance(queue_last, dict) and not probe_same(rl, queue_last):", "if False:"),
    ('probe_tol_loose', 'F4', 'desk_v3.py',
     "PROBE_TOL = {'st10_thr_vs_b_us': 0.01, 'st23_thr_vs_b_us': 0.01, 'stand_brake_side_frac': 0.0001,\n"
     "             'low10_brake_frac': 0.0001, 'stand_go_frac': 0.0001}",
     "PROBE_TOL = {'st10_thr_vs_b_us': 0.1, 'st23_thr_vs_b_us': 0.1, 'stand_brake_side_frac': 0.001,\n"
     "             'low10_brake_frac': 0.001, 'stand_go_frac': 0.001}"),
    ('moving_vmin_ignored', 'F4', 'desk_v3.py', "np.isfinite(thr0), vmin=MOVING_MPS)", "np.isfinite(thr0), vmin=None)"),
    ('fix_all_removed', 'F4', 'eval_v3.py', "'ok': bool(mean(dd) <= -t and all(x < 0 for x in dd))", "'ok': bool(mean(dd) <= -t)"),
    ('veto_iv_ge', 'F4', 'eval_v3.py', "'veto': bool(diff > thr_by_n[key] and", "'veto': bool(diff >= thr_by_n[key] and"),
    ('veto_bias_ge', 'F4', 'eval_v3.py', "'veto': bool(abs(d) > t)", "'veto': bool(abs(d) >= t)"),
    ('futility_ge', 'F4', 'eval_v3.py', "vals.append(num(p['st23_thr_vs_b_us']) > t23 or num(p['stand_brake_side_frac']) > tst)",
     "vals.append(num(p['st23_thr_vs_b_us']) >= t23 or num(p['stand_brake_side_frac']) >= tst)"),
    ('probe_gate_uses_last', 'F4', 'eval_v3.py', "    g.update(probe_gates(r.get('probe_best'), T))\n",
     "    g.update(probe_gates(r.get('probe_last') or r.get('probe_best'), T))\n"),
    # F5
    ('judge_counts_excluded', 'F5', 'eval_v3.py', "            if r.get('exclude_why'):", "            if False:"),
    ('exclusion_state_ignored', 'F5', 'eval_v3.py', "    if res.get('state') != 'done':\n", "    if False:\n"),
    ('exclusion_epochs_ignored', 'F5', 'eval_v3.py', "    if want is None or res.get('epochs_done') != want:\n", "    if False:\n"),
    ('exclusion_h1_ignored', 'F5', 'eval_v3.py', "    if not h1_current(name, 'best', proto):\n        why.append",
     "    if False:\n        why.append"),
    ('exclusion_not_wired', 'F5', 'eval_v3.py', "            r['exclude_why'] = v3_exclusion(name, proto)\n", "            r['exclude_why'] = []\n"),
    # F6
    ('self_test_all_rows', 'F6', 'eval_v3.py', "    st_new = self_test_new_gates(desk_runs, T, st_names)\n",
     "    st_new = self_test_new_gates(desk_runs, T)\n"),
    ('check_reads_v3_trt', 'F6', 'eval_v3.py', "    rows, top, dnotes = D.build_rows(proto, trt_dirs=(D.TRT2,))\n",
     "    rows, top, dnotes = D.build_rows(proto)\n"),
    # F7
    ('lock2_stop_off', 'F7', 'desk_v3.py', "    if lk['lock2'] and not lk['same']:\n", "    if False:\n"),
    ('check_names_no_sha', 'F7', 'eval_v3.py', "    D.lock2_stop('v3 の予測', allow_deviation)", "    pass"),
    ('lock2_gate_keeps_final', 'F7', 'eval_v3.py', "        for k in ('E1', 'E2', 'E3'):\n            E[k]['final'] = False\n", ""),
    ('lock2_gate_not_called', 'F7', 'eval_v3.py', "    lock2_gate(J, lk, getattr(args, 'allow_deviation', False), args.dry_run_b)\n",
     "    pass\n"),
    ('lock2_report_note_off', 'F7', 'desk_v3.py', "    if ln:\n        notes.insert(0, ln", "    if False:\n        notes.insert(0, ln"),
    # F9
    ('check_md_no_selftest_column', 'F9', 'eval_v3.py', "'| 学習 | 自己試験(low10 なし) | 新しい門(low10 も) |", "'| 学習 | 自己 | 新しい門(low10 も) |"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(name, fname=None, old=None, new=None):
    d = TMP / name
    if d.exists():
        shutil.rmtree(d)
    (d / 'c3').mkdir(parents=True)
    for f in ('desk_v3.py', 'eval_v3.py'):
        s = (C3 / f).read_text(encoding='utf-8')
        if f == 'desk_v3.py':
            a = 'FIX = HERE.parent '
            assert s.count(a) == 1
            s = s.replace(a, "FIX = Path('%s') " % FIX)
        if f == fname:
            n = s.count(old)
            if n != 1:
                raise SystemExit('%s: 元の文が %d 回(1 回のはず)' % (name, n))
            s = s.replace(old, new)
        (d / 'c3' / f).write_text(s, encoding='utf-8')
    t = TEST.read_text(encoding='utf-8')
    a = "C3 = Path('/home/jetson/jetformer/experiments/20261026_fix/c3')\n"
    assert t.count(a) == 1
    t = t.replace(a, "C3 = Path('%s')\n" % (d / 'c3'))
    b = "import eval_v3 as E  # noqa: E402\n"
    t = t.replace(b, b + "assert str(D.__file__).startswith(%r) and str(E.__file__).startswith(%r)\n" % (str(d), str(d)))
    (d / 'test_desk_v3.py').write_text(t, encoding='utf-8')
    return d


def run(d):
    t0 = time.time()
    r = subprocess.run(HEAVY + [PY, '-I', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=%s' % (d / 'pt'),
                                '--rootdir=%s' % d, str(d / 'test_desk_v3.py')], capture_output=True, text=True, cwd=str(d))
    tail = [x for x in (r.stdout + r.stderr).strip().splitlines() if x.strip()][-1:]
    failed = [x.split('::')[-1].split(' ')[0] for x in r.stdout.splitlines() if x.startswith('FAILED')]
    return {'rc': r.returncode, 'tail': tail[0] if tail else '', 'failed': failed, 'seconds': round(time.time() - t0, 1)}


def main(argv):
    before = {str(p): sha(p) for p in (C3 / 'desk_v3.py', C3 / 'eval_v3.py', TEST)}
    want = set(argv)
    out = {'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'scripts_sha256': before, 'control': None, 'mutants': []}
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        d = prepare('_control')
        out['control'] = run(d)
        shutil.rmtree(d)
        print('control: %s' % out['control']['tail'], flush=True)
        for name, fid, fname, old, new in MUT:
            if want and name not in want:
                continue
            d = prepare(name, fname, old, new)
            r = run(d)
            shutil.rmtree(d)
            r.update({'name': name, 'finding': fid, 'file': fname, 'caught': r['rc'] != 0,
                      'equivalent': EQUIVALENT.get(name)})
            out['mutants'].append(r)
            print('%-30s %s caught=%s %s %s' % (name, fid, r['caught'], r['tail'], ' '.join(r['failed'][:4])), flush=True)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
        after = {str(p): sha(p) for p in (C3 / 'desk_v3.py', C3 / 'eval_v3.py', TEST)}
        out['scripts_unchanged'] = after == before
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    n = len(out['mutants'])
    k = sum(1 for m in out['mutants'] if m['caught'])
    eq = [m['name'] for m in out['mutants'] if m['equivalent'] and not m['caught']]
    miss = [m['name'] for m in out['mutants'] if not m['caught'] and not m['equivalent']]
    out['summary'] = {'mutants': n, 'caught': k, 'equivalent_not_caught': eq, 'missed': miss}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('壊した %d 通り・試験が落ちた %d 通り・同じ動きで見分けられない %d(%s)・見逃し %d %s・本物の台本は %s' % (
        n, k, len(eq), ' '.join(eq), len(miss), ' '.join(miss), '変わっていない' if out['scripts_unchanged'] else '**変わった**'))
    return 0 if (out['control']['rc'] == 0 and not miss and out['scripts_unchanged']) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
