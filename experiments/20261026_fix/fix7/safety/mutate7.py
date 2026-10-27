"""直す 7-1 回目・係 safety: 「わざと壊すと落ちる」の確かめ。jetformer/・tests/・pytest.ini を一時の場所
(fix7/tmp_safety/mut_<名前>)に写し、写しの assist.py・drive.py の直した所を 1 か所ずつ戻して(または決まりを壊して)、写しの上で
新しい試験 tests/test_rec_fix7_safety.py を流す。本物の jetformer/ は変えない(ほかの係が同時に試験を流していても影響しない)。
終わったら写しは消す。装置・GPU・API には触れない(fix5/safety/mutate.py と同じ形)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety/mutate7.py [名前 ...]
出力: fix7/safety/out/mutate7.json(名前ごとに 落ちた試験の数・通った数・落ちた試験の名前)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments' / '20261026_fix' / 'fix7' / 'tmp_safety'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TESTS = ['tests/test_rec_fix7_safety.py']
AS, DR = 'jetformer/assist.py', 'jetformer/drive.py'

# (名前, [(ファイル, 前の文, あとの文)], 何を壊すか)
MUTANTS = [
    ('s4a_guard_revert', [(AS, "        if backing and esc_rev is False and not self.esc_ran and self.phase in ('brake', 'wait', 'gap'):",
                           "        if False:")],
     "s4-a を _guard で戻す('brake'・'wait'・'gap' の跳ねも見張りに使う = 指摘の形)"),
    ('s4a_guard_gap_only_brake', [(AS, "        if backing and esc_rev is False and not self.esc_ran and self.phase in ('brake', 'wait', 'gap'):",
                                   "        if backing and esc_rev is False and not self.esc_ran and self.phase in ('brake', 'wait'):")],
     "s4-a を 'gap' で効かせない('gap' の跳ねを惰性に数える)"),
    ('s4a_hold_revert', [(AS, "        if esc_rev is False and not self.esc_ran and p in ('brake', 'wait', 'gap') and hb > 0.0:",
                          "        if False:")],
     "s4-a を中立止めのあとで戻す(中立止めの間の跳ねで back_hold)"),
    ('s4a_no_latch', [(AS, "        if esc_rev is not False:\n            self.esc_ran = True ",
                       "        if esc_rev is True and False:\n            self.esc_ran = True ")],
     "esc_ran を立てない(一度後退で走らせたあとの 'gap' の惰性も数えない)"),
    ('s4a_none_as_false', [(AS, "        if backing and esc_rev is False and not self.esc_ran and self.phase in ('brake', 'wait', 'gap'):",
                            "        if backing and esc_rev is not True and not self.esc_ran and self.phase in ('brake', 'wait', 'gap'):"),
                           (AS, "        if esc_rev is not False:\n            self.esc_ran = True ",
                            "        if esc_rev is True:\n            self.esc_ran = True ")],
     "esc_rev = None(分からない)も False と同じにする(今までと同じにならない)"),
    ('s4a_drive_none', [(DR, "                    esc_rev = esc_gate.rev_s(esc_q[0], q_ns) > 0.0",
                         "                    esc_rev = None")],
     "drive が esc_rev を渡さない(EscTrack を使わない)"),
    ('s4a_drive_feed_keep_only', [(DR, "            if written and guard_on():      # 同じ(fix7 safety s4-a)",
                                   "            if written and keep_on():      # 同じ(fix7 safety s4-a)")],
     "EscMayReverse に書けた値を入れるのを rec_hold_keep = 1 の日だけにする(見張りだけの日は空のまま = 下がれるとみる区間が無い)"),
    ('s4a_gate_is_esctrack', [(DR, "        esc_gate = assist_mod.EscMayReverse(ref[1])",
                               "        esc_gate = assist_mod.EscTrack(ref[1])")],
     "esc_rev に EscTrack(中立 0.08 秒・遅れ 0.06 秒)を使う(多めにみない)"),
    ('s4b_check_revert', [(AS, "            if ((self.seen_rev_s == 0.0) if back_guard else (self.backed_s - self.est_s == 0.0)) and now - self.t_check > CHECK_S:",
                           "            if self.backed_s - self.est_s == 0.0 and now - self.t_check > CHECK_S:")],
     "s4-b を戻す(やり直しを backed_s − est_s で決める)"),
    ('s4b_entry_counts', [(AS, "                if phase0 == 'reverse':         # fix7 s4-b",
                           "                if True:         # fix7 s4-b")],
     "'reverse' に入ったフレームの後ろへの動きも、やり直しの決めの見た後退に数える"),
    ('s4b_undo_as_retry', [(AS, "                if back_guard and self.skip_gap:\n",
                            "                if False:\n")],
     "'gap' を飛ばした 'reverse' で下がらないとき、ふつうのやり直し(+50 µs・RETRY_GAP_S)にする"),
    ('s4b_no_skip_flag_early', [(AS, "            self.skip_gap = True                       # fix7 s4-b: 'gap' を飛ばした",
                                 "            pass                       # fix7 s4-b: 'gap' を飛ばした")],
     "back_early で skip_gap を立てない"),
    ('off_break_check', [(AS, "            if ((self.seen_rev_s == 0.0) if back_guard else (self.backed_s - self.est_s == 0.0)) and now - self.t_check > CHECK_S:",
                          "            if self.seen_rev_s == 0.0 and now - self.t_check > CHECK_S:")],
     "OFF を壊す(見張り OFF でも s4-b のやり直しの決めを使う)= OFF の写しとの比べが落ちるか"),
    ('off_break_drive', [(DR, "        def guard_on():\n            return int(p.get('rec_back_guard', 0) or 0) == 1",
                          "        def guard_on():\n            return True")],
     "OFF を壊す(見張り OFF でも EscTrack を動かし esc_rev を渡す)= 渡す値の試験が落ちるか"),
]


def run_one(name, edits):
    d = TMP / ('mut_' + name)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copytree(JF / 'jetformer', d / 'jetformer', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(JF / 'tests', d / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(JF / 'pytest.ini', d / 'pytest.ini')
    for rel, old, new in edits:
        f = d / rel
        s = f.read_text(encoding='utf-8')
        assert s.count(old) == 1, (name, rel, s.count(old))
        f.write_text(s.replace(old, new), encoding='utf-8')
    bt = d / 'bt'
    try:
        p = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=%s' % bt] + TESTS,
                           cwd=str(d), capture_output=True, text=True, timeout=3000)
        out = p.stdout + p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)
    failed = sorted(set(re.findall(r'FAILED (\S+)', out)))
    m = re.search(r'(\d+) failed', out)
    m2 = re.search(r'(\d+) passed', out)
    return {'failed': int(m.group(1)) if m else 0, 'passed': int(m2.group(1)) if m2 else 0,
            'names': failed[:40], 'tail': out.strip().splitlines()[-1:]}


def main(names):
    res = {}
    outp = HERE / 'out' / 'mutate7.json'
    if outp.exists():
        res = json.loads(outp.read_text())
    for name, edits, what in MUTANTS:
        if names and name not in names:
            continue
        r = run_one(name, edits)
        r['what'] = what
        res[name] = r
        print(name, r['failed'], 'failed', r['passed'], 'passed', r['tail'], flush=True)
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    try:
        TMP.rmdir()
    except OSError:
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
