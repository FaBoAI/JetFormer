"""直す 6-2 回目・係 tools: わざと壊した collect.py・standing_3lap.py・quick_eval_by_params.py で tests/test_fix6_tools_r2.py が落ちるかを
見る。本物のファイルは変えない(写しを fix6/tmp_tools_r2/mut/ に作り、環境変数 JF_FIX62_COLLECT・JF_FIX62_S3・JF_FIX62_QP で試験に
渡す)。写しと basetemp は最後に消す。⑤ の壊しは ⑤ の試験だけ、③④ の壊しは ③④ の試験だけを流す(-k。速さのため)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B mutate_fix62.py
        → out/mutate_fix62_out.txt
"""
import os
import shutil
import subprocess
from pathlib import Path

JF = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
SRC = {'collect': JF / 'experiments/20261026_fix/diag_launch/collect.py', 's3': JF / 'tools/standing_3lap.py',
       'qp': JF / 'tools/quick_eval_by_params.py'}
ENV = {'collect': 'JF_FIX62_COLLECT', 's3': 'JF_FIX62_S3', 'qp': 'JF_FIX62_QP'}
KSEL = {'collect': 'not after_ai and not runbook and not fixture', 's3': 'after_ai', 'qp': 'after_ai'}
BEFORE = {'collect': HERE / 'before' / 'collect.py.orig', 's3': HERE / 'before' / 'standing_3lap.py.orig',
          'qp': HERE / 'before' / 'quick_eval_by_params.py.orig'}
WORK = JF / 'experiments/20261026_fix/fix6/tmp_tools_r2/mut'
OUT = HERE / 'out' / 'mutate_fix62_out.txt'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'

MUT = [
    # ⑤ collect.py
    ('collect', 'lost_tail がいつも空', '    return [(s, lo) for s, lo, hi in ls if hi is None]', '    return []'),
    ('collect', 'lost_tail に真ん中の欠けも入れる', '    return [(s, lo) for s, lo, hi in ls if hi is None]',
     '    return [(s, lo) for s, lo, hi in ls]'),
    ('collect', 'ON の手がかりに log.json を見ない', "    if _on(u.get('launch_param')):\n        out.append('log.json')",
     "    if False:\n        out.append('log.json')"),
    ('collect', 'ON の手がかりに計画を見ない', "    if _on(u.get('plan_launch')):\n        out.append('計画')", "    if False:\n        out.append('計画')"),
    ('collect', 'ON の手がかりに発進の前進の出来事を見ない', "    if ev:\n        out.append('events.jsonl", "    if False:\n        out.append('events.jsonl"),
    ('collect', 'ON の手がかりに走行中の launch_boost 1 を見ない', "    if mid:\n        out.append('走行中に", "    if False:\n        out.append('走行中に"),
    ('collect', '出来事は start だけ数える', "        if isinstance(e, dict) and e.get('kind') == 'launch_boost':",
     "        if isinstance(e, dict) and e.get('kind') == 'launch_boost' and e.get('event') == 'start':"),
    ('collect', 'OFF の本も入れる', '            if not src:                # 発進の前進 OFF の本', '            if False:                # 発進の前進 OFF の本'),
    ('collect', '--first-only を見ない', 'later = bool(first_only and t_read and (t_last is None or min(t_read) <= t_last))', 'later = False'),
    ('collect', 'いつも 2 回目以降にする', 'later = bool(first_only and t_read and (t_last is None or min(t_read) <= t_last))', 'later = True'),
    ('collect', 'start で入れた本にも入れる(二重)', "if not d.get('data_missing') or log in lost_ev:", "if not d.get('data_missing'):"),
    ('collect', '読めない記録を名前の順に並べ直さない', 'or any(not v[2] for v in lost_tl.values()))', 'or False)'),
    ('collect', 'lost_after_s を出さない', 'lost_after_s=round(t_last, 3) if t_last is not None else None, ', ''),
    ('collect', 'AI のフレームが無い本に手がかりを使わない', "                    elif u['launch_state'] != 'on':", '                    elif False:'),
    ('collect', 'AI のフレームが無い本の ON にも手がかりを足す', "                    elif u['launch_state'] != 'on':", '                    else:'),
    ('collect', '「記録が欠けている本」の行の後ろの 1 文を出さない', '                elif lg in lost_tl:    # 2026-10-27 fix6-2',
     '                elif False:    # 2026-10-27 fix6-2'),
    ('collect', '「記録が欠けている本」の頭を ON にしない', '                    on_ = True\n', '                    pass\n'),
    ('collect', '読めない記録の行に手がかりを出さない', "'・%s' % '・'.join(other(src)) if other(src) else ''", "''"),
    ('collect', '組に launch_on_from を出さない', "            if any(u.get('launch_on_from') for u in us_):", '            if False:'),
    ('collect', '対照: CRIT の起点の差 0.15 → 0.50', "CRIT = {'to_r1_gain_s': 0.15,", "CRIT = {'to_r1_gain_s': 0.50,"),
    # ③ standing_3lap.py
    ('s3', '人側にした直後の衝突を足さない', "    after = cc.after_ai_collisions(ex.get('ev') or info.get('events') or {}, lab)", '    after = []'),
    ('s3', '発進(前に衝突・3 周の衝突)は AI 側だけのまま', "anchors, coll_b,\n                                      L.get('AI_CMD')",
     "anchors, coll,\n                                      L.get('AI_CMD')"),
    ('s3', '走りながらの窓にも足す', "wins = flying_windows(sc.get('laps') or [], coll)", "wins = flying_windows(sc.get('laps') or [], coll_b)"),
    ('s3', '本の衝突の数は AI 側だけ', "    cs = ss.collision_summary(coll_b)\n    res['collisions'] = {'all': len(coll_b), 'strong': sum(1 for c in coll_b",
     "    cs = ss.collision_summary(coll)\n    res['collisions'] = {'all': len(coll), 'strong': sum(1 for c in coll"),
    ('s3', 'after_ai の一覧を出さない', "        res['collisions']['after_ai'] = [", "        res['collisions']['_x'] = ["),
    ('s3', 'ai_all・ai_strong を出さない', "res['collisions'].update(ai_all=len(coll), ai_strong=", "res['collisions'].update(_a=len(coll), _b="),
    ('s3', 'text に足さない', "        res['collisions']['text'] += ", "        res['collisions']['_t'] = "),
    ('s3', '表の注を出さない', "    if any((r.get('collisions') or {}).get('after_ai') for r in results):     # 2026-10-27 fix6-2(あるときだけ)",
     '    if False:'),
    ('s3', '表の注をいつも出す', "    if any((r.get('collisions') or {}).get('after_ai') for r in results):     # 2026-10-27 fix6-2(あるときだけ)",
     '    if True:'),
    ('s3', '表の印を出さない', "    aa = (C or {}).get('after_ai')", '    aa = None'),
    # ④ quick_eval_by_params.py
    ('qp', '本に coll_after_ai を出さない', "            out[-1]['coll_after_ai'] = C['after_ai']", '            pass'),
    ('qp', '1 章の欄に印を出さない', " + s3.after_ai_mark({'after_ai': x.get('coll_after_ai')})", ''),
    ('qp', '組に collisions_after_ai を出さない', "            row['collisions_after_ai'] = aa", '            pass'),
    ('qp', '2 章の下の行を出さない', "    if any(g.get('collisions_after_ai') for g in groups):     # 2026-10-27 fix6-2(あるときだけ)", '    if False:'),
    ('qp', '2 章の下の行をいつも出す', "    if any(g.get('collisions_after_ai') for g in groups):     # 2026-10-27 fix6-2(あるときだけ)", '    if True:'),
]


def run(env_extra, tag, ksel=None, stop=True):
    bt = WORK / ('bt_' + tag)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **env_extra)
    cmd = [PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-W', 'ignore::DeprecationWarning'] + (['-x'] if stop else []) + \
        (['-k', ksel] if ksel else []) + ['--tb=line', '--basetemp=%s' % bt, 'tests/test_fix6_tools_r2.py']
    p = subprocess.run(cmd, cwd=str(JF), env=env, capture_output=True, text=True)
    shutil.rmtree(bt, ignore_errors=True)
    tail = [x for x in p.stdout.splitlines() if x.strip()][-3:]
    return p.returncode, tail


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    src = {k: p.read_text(encoding='utf-8') for k, p in SRC.items()}
    lines = []
    try:
        rc, tail = run({}, 'now', stop=False)
        lines.append('今の版: rc=%d %s' % (rc, tail[-1]))
        rc, tail = run({ENV[k]: str(BEFORE[k]) for k in SRC}, 'before', stop=False)
        lines.append('直す前の版(3 本とも): rc=%d %s' % (rc, ' / '.join(tail)))
        for i, (which, name, old, new) in enumerate(MUT):
            n = src[which].count(old)
            if n != 1:
                lines.append('[%02d] %s %s: 当たらない(%d 回)' % (i, which, name, n))
                continue
            mp = WORK / ('%s_m%02d.py' % (which, i))
            mp.write_text(src[which].replace(old, new), encoding='utf-8')
            rc, tail = run({ENV[which]: str(mp)}, 'm%02d' % i, KSEL[which])
            lines.append('[%02d] %-7s %s: %s(rc=%d)%s' % (i, which, name, '落ちる' if rc else '**通ってしまう**', rc,
                                                     '' if rc else ' ' + tail[-1]))
            if rc:
                fl = [x for x in tail if 'FAILED' in x or 'Error' in x or 'assert' in x]
                if fl:
                    lines[-1] += '  ' + fl[0][:200]
            print(lines[-1], flush=True)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
        if WORK.parent.exists() and not any(WORK.parent.iterdir()):
            WORK.parent.rmdir()
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
