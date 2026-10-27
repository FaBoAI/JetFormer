"""直す 7-2 回目・係 safety: 「わざと壊すと落ちる」の確かめ。jetformer/・tests/・pytest.ini を一時の場所
(fix7/tmp_safety_r2/mut_<名前>)に写し、写しの assist.py の直した所を 1 か所ずつ戻して(または決まりを壊して)、写しの上で
新しい試験 tests/test_rec_fix7_safety_r2.py を流す。本物の jetformer/ は変えない(ほかの係が同時に試験を流していても影響しない)。
終わったら写しは消す。装置・GPU・API には触れない(7-1 回目の safety/mutate7.py と同じ形)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix7/safety_r2/mutate9.py [名前 ...]
出力: fix7/safety_r2/out/mutate9.json(名前ごとに 落ちた試験の数・通った数・落ちた試験の名前)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments' / '20261026_fix' / 'fix7' / 'tmp_safety_r2'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TESTS = ['tests/test_rec_fix7_safety_r2.py']
AS = 'jetformer/assist.py'
NONE_YET = "                none_yet = not self.back_sure and erpm <= BACKING_ERPM\n"
BRANCH = "            if back_guard and self.skip_gap:\n                # fix7 r2 s4-c"

# (名前, [(ファイル, 前の文, あとの文)], 何を壊すか)
MUTANTS = [
    ('s4c_revert', [(AS, NONE_YET, "                none_yet = self.seen_rev_s == 0.0\n")],
     "s4-c を戻す('gap' を飛ばした 'reverse' も 'reverse' の中の跳ね 1 つで early_none を止める = 指摘の形)"),
    ('s4c_two_frames', [(AS, "SURE_BACK_S = 0.2 ", "SURE_BACK_S = 0.0 ")],
     "本当に下がっているを「2 フレーム続いた」にする(反証の文の案 (b))"),
    ('s4c_sure_short', [(AS, "SURE_BACK_S = 0.2 ", "SURE_BACK_S = 0.1 ")],
     "SURE_BACK_S を 0.1 秒にする(本物の跳ねの最長 0.11 秒より短い)"),
    ('s4c_no_defer', [(AS, NONE_YET, "                none_yet = not self.back_sure\n")],
     "このフレームで後ろへ動いていても決める(下がり始めた車をやり直す)"),
    ('s4c_no_run_reset', [(AS, "                else:\n                    self.run_t0 = None\n",
                           "                else:\n                    pass\n")],
     "途切れても数え直さない(とびとびの跳ねを続けたとみる)"),
    ('s4c_no_hold_reset', [(AS, "            self.run_t0 = None          # fix7 r2 s4-c", "            pass          # fix7 r2 s4-c")],
     "中立止めの前と後の後ろへの動きをつなげる"),
    ('s4c_entry_counts', [(AS, "                if phase0 == 'reverse' and erpm > BACKING_ERPM:\n                    if self.run_t0 is None:",
                           "                if erpm > BACKING_ERPM:\n                    if self.run_t0 is None:")],
     "'reverse' に入ったフレーム(back_early の跳ね)も続けた後ろへの動きに数える"),
    ('s4c_never_sure', [(AS, "                        self.back_sure = True\n", "                        self.back_sure = False\n")],
     "本当に下がっているを一度見ても覚えない(下がっている途中の 1 フレームの途切れでやり直す)"),
    ('s4c_normal_reverse_too', [(AS, BRANCH, "            if back_guard:\n                # fix7 r2 s4-c")],
     "'gap' を飛ばさずに入った 'reverse' のやり直しにも s4-c を使う(今まで通りにならない)"),
    ('off_break', [(AS, BRANCH, "            if self.skip_gap or not back_guard:\n                # fix7 r2 s4-c")],
     "OFF を壊す(見張り OFF でもやり直しを s4-c の決めで)= OFF の写しとの比べが落ちるか"),
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
            'names': failed[:60], 'tail': out.strip().splitlines()[-1:]}


def main(names):
    res = {}
    outp = HERE / 'out' / 'mutate9.json'
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


if __name__ == '__main__':
    main(sys.argv[1:])
