"""tools:F6 の確かめ(2026-10-26): setup_sandbox.sh で作った砂場の写しに 1 つずつ壊しを入れて、対象の試験だけを回す。
本物の tools・tests は触らない。CONTROL = 落ちるはずの壊し(砂場の写しを本当に読んでいるかの確かめ)"""
import os, subprocess, sys
S = '/home/jetson/jetformer/experiments/20261026_fix/fix2/tmp_refute_F6'
T3, TQ, TC = 'tests/test_standing_3lap.py', 'tests/test_quick_eval_by_params.py', 'tests/test_tools_collect.py'
ALL = [T3, TQ, TC]
SP, QP, CO = 'tools/standing_3lap.py', 'tools/quick_eval_by_params.py', 'diag_launch/collect.py'
M = [
 ('s3-frame-gap', SP, "if _max_gap(t, t_ai, e3) > le.GAP_S:", "if False:", ALL),
 ('s3-veh-gap', SP, "if _max_gap(vt, t_move, e3) > le.GAP_S:", "if False:", ALL),
 ('s3-line-lap-5-16', SP, "if not all(LAP_MIN_S <= x <= LAP_MAX_S for x in row['laps_line_s']):", "if False:", ALL),
 ('s3-anchor-chain-5-16', SP, "if len(A_) == 4 or not (LAP_MIN_S <= x - A_[-1] <= LAP_MAX_S):", "if len(A_) == 4:", ALL),
 ('s3-to-r1-15s', SP, "if not len(nxt) or nxt[0] > t1 or nxt[0] - t_ai > MAX_TO_R1_S:", "if not len(nxt) or nxt[0] > t1:", ALL),
 ('s3-cont-s', SP, "if not all(abs(w[j]['b'] - w[j + 1]['a']) <= CONT_S for j in range(2)):", "if False:", ALL),
 ('s3-dryrun-mask', SP, "ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)", "ai = lab['ai'].astype(bool)", ALL),
 ('s3-lapdist-dpsi', SP, "if DIST_DPSI[0] <= dpsi <= DIST_DPSI[1] and b - a < LAP_MAX_S:", "if b - a < LAP_MAX_S:", ALL),
 ('s3-best-min', SP, "return min(v, key=lambda s: s['from_move_s']) if v else None", "return max(v, key=lambda s: s['from_move_s']) if v else None", ALL),
 ('s3-launch-on-s-+1/30', SP, "out['on_s'] = _r(float(tm[on][-1] - t_a) + 1.0 / FPS, 3)", "out['on_s'] = _r(float(tm[on][-1] - t_a), 3)", ALL),
 ('qp-key-gain', QP, "key.append(('gain', g if g is not None else -1))", "key.append(('gain', 0))", [TQ]),
 ('qp-key-steer', QP, "key.append(('steer', sg))", "key.append(('steer', 1.0))", [TQ]),
 ('qp-key-zone', QP, "key.append(('zone', z))", "key.append(('zone', 0))", [TQ]),
 ('qp-key-mode', QP, "key.append(('mode', m if isinstance(m, int) else -1))", "key.append(('mode', -1))", [TQ]),
 ('qp-key-auto-target', QP, "key.append(('auto', (1, _num(S.get('auto_target_mps')) or 0)))", "key.append(('auto', (1, 0)))", [TQ]),
 ('qp-key-launch-us', QP, "key.append(('launch', (2, _num(S.get('launch_us')) or 0,", "key.append(('launch', (2, 0,", [TQ]),
 ('qp-standing-valid', QP, "stand = [s for r in rs for s in r.get('standing') or [] if s.get('valid')]",
  "stand = [s for r in rs for s in r.get('standing') or [] if s.get('from_move_s') is not None]", [TQ]),
 ('qp-fast-9s', QP, "'fast_laps': sum(1 for x in clean if x < FAST_LAP_S),", "'fast_laps': sum(1 for x in clean if x < 99.0),", [TQ]),
 ('col-launch-none', CO, "        return int(float(x)) == 1\n    except (TypeError, ValueError):\n        return False",
  "        return int(float(x)) == 1\n    except (TypeError, ValueError):\n        return True", [TC]),
 ('col-launch-on-s-+1/30', CO, "row['launch_on_s'] = round(float(ts[on][-1] - t0) + 1 / 30.0, 3)", "row['launch_on_s'] = round(float(ts[on][-1] - t0), 3)", [TC]),
 ('col-day-key-ceiling', CO, "return (d['model_name'], round(g, 2), d['session'], _on(S.get('speed_ceiling')), S.get('ceil_left_mps'),\n"
  "                    _on(S.get('speed_auto')), S.get('auto_target_mps'))", "return (d['model_name'], round(g, 2), d['session'])", [TC]),
 ('CONTROL-s3-human', SP, "if e3 > t1 + le.GAP_S:", "if False:", ALL),
 ('CONTROL-qp-ceiling', QP, "key.append(('ceiling', (1, _num(S.get('ceil_left_mps')) or 0, _num(S.get('ceil_all_mps')) or 0)))",
  "key.append(('ceiling', (1, 0, 0)))", [TQ]),
 ('CONTROL-col-state', CO, "row['launch_state'] = 'off' if not _on(S.get('launch_boost')) else ('on' if started else 'on_not_started')",
  "row['launch_state'] = 'off'", [TC]),
]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
for name, f, old, new, tests in M:
    p = os.path.join(S, f)
    src = open(p, encoding='utf-8').read()
    if src.count(old) != 1:
        print('%-24s BAD-PATTERN' % name)
        continue
    open(p, 'w', encoding='utf-8').write(src.replace(old, new))
    try:
        bt = os.path.join(S, 'pt_' + name.replace('/', '_'))
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', bt] + tests,
                           cwd=S, env=env, capture_output=True, text=True, timeout=600)
        last = (r.stdout.strip().splitlines() or [''])[-1]
        print('%-24s %-10s %s' % (name, 'SURVIVED' if r.returncode == 0 else 'KILLED', last))
        subprocess.run(['rm', '-rf', bt])
    finally:
        open(p, 'w', encoding='utf-8').write(src)
