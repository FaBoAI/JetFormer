"""fix6 係 safety 直す 6-1 回目: 「わざと壊すと落ちる」の確かめ。jetformer/・tests/・pytest.ini を一時の場所
(fix6/tmp_safety_r1_mut/mut_<名前>)に写し、写しの assist.py・drive.py の直した所を 1 か所ずつ戻して(または決まりを壊して)、写しの
上で試験を流す。本物の jetformer/ は変えない。終わったら写しは消す。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix6/safety_r1/mutate.py [名前 ...]
出力: fix6/safety_r1/out/mutate.json(名前ごとに 落ちた試験の数・通った数・落ちた試験の名前)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_safety_r1_mut'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TESTS = ['tests/test_rec_fix6_safety.py', 'tests/test_rec_fix5_safety.py::test_s3_loop_vehicle_gap_no_escape_into_wall']

# (名前, ファイル, 前の文, あとの文, 何を壊すか)
MUTANTS = [
    ('est_off_drive', 'jetformer/drive.py',
     "            hold_rev_s = hold_veh[4] if held else 0.0          # 値が来なかった間の後退の見積もり(fix6 safety r1)",
     "            hold_rev_s = 0.0",
     's3-a を drive で戻す(見積もりを渡さない = fix5 の形。指摘が戻る)'),
    ('est_off_assist', 'jetformer/assist.py',
     "                self.backed_s += min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))",
     "                pass",
     's3-a を assist で戻す(見積もりを下がった時間に足さない)'),
    ('gap_full_count', 'jetformer/drive.py',
     "                        if hold_veh[1] > HOLD_VEH_FAST_ERPM:",
     "                        if True:",
     '値が来なかった間をいつも全部数える(fix3 r2 の数え方に近い。前の壁へ escape)'),
    ('double_count', 'jetformer/drive.py',
     "                        if hold_veh[1] > assist_mod.BACKING_ERPM and hold_veh[3] is not None:",
     "                        if False:",
     '古い値で Recovery が数えたフレームの分を除かない(二重に数える)'),
    ('fast_off', 'jetformer/drive.py',
     "HOLD_VEH_FAST_ERPM = 1000.0 ",
     "HOLD_VEH_FAST_ERPM = 1e9 ",
     '速く下がっていても見積もりだけ数える(惰性の分を数えない。後ろへ多く)'),
    ('track_no_ready', 'jetformer/assist.py',
     "                if self.ready:\n                    self.runs.append([t_ns + int(self.DELAY_S * 1e9), None])",
     "                if True:\n                    self.runs.append([t_ns + int(self.DELAY_S * 1e9), None])",
     'EscTrack が「下がれる」を見ない(前進のあとのブレーキも後退として数える)'),
    ('track_delay_long', 'jetformer/assist.py',
     "    DELAY_S = 0.06\n",
     "    DELAY_S = 0.12\n",
     'EscTrack の下がり始めを遅くする(少なく数える側)'),
    ('cap_off', 'jetformer/assist.py',
     "                self.backed_s += min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))",
     "                self.backed_s += est",
     '見積もりで後退を 0.2 秒残す決まりをやめる(短い rev_s で前の壁へ)'),
    ('coast_shift_back', 'jetformer/assist.py',
     "        if p != 'coast':                                # fix6 s3-b: 'coast' は中立止めの間も惰性(段の時間をずらさない)",
     "        if True:",
     "s3-b を一部戻す('coast' の途中の中立止めで段の時間をずらす)"),
    ('coast_from_now', 'jetformer/assist.py',
     "            if self.backed_s >= rev_s:                  # fix6 s3-b: 後退は中立止めの始めで出し終えていた。惰性はそこから",
     "            if False:",
     "s3-b を一部戻す(中立止めのあいだに後退を出し終えても、惰性を明けてから始める)"),
    ('off_break_guard', 'jetformer/drive.py',
     "            if (int(p.get('rec_hold_keep', 0) or 0) == 1 and int(p.get('rec_back_guard', 0) or 0) == 1\n"
     "                    and veh_select and not dry_run):",
     "            if (int(p.get('rec_hold_keep', 0) or 0) == 1\n"
     "                    and veh_select and not dry_run):",
     'OFF を壊す(見張りが OFF でも中立止めでリカバリーを残す)= 直す前の写しとの比べが落ちるか'),
]


def run_one(name, rel, old, new):
    d = TMP / ('mut_' + name)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copytree(JF / 'jetformer', d / 'jetformer', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(JF / 'tests', d / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(JF / 'pytest.ini', d / 'pytest.ini')
    f = d / rel
    s = f.read_text(encoding='utf-8')
    assert s.count(old) == 1, (name, s.count(old))
    f.write_text(s.replace(old, new), encoding='utf-8')
    bt = d / 'bt'
    try:
        p = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=%s' % bt]
                           + TESTS, cwd=str(d), capture_output=True, text=True, timeout=3000)
        out = p.stdout + p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)
    failed = sorted(set(re.findall(r'^FAILED (\S+)', out, re.M)))
    m = re.search(r'(\d+) failed', out)
    m2 = re.search(r'(\d+) passed', out)
    return {'failed': int(m.group(1)) if m else 0, 'passed': int(m2.group(1)) if m2 else 0,
            'failed_tests': failed, 'tail': out.strip().splitlines()[-1:]}


def main(names):
    res = {}
    outp = HERE / 'out' / 'mutate.json'
    if outp.exists():
        res = json.loads(outp.read_text(encoding='utf-8'))
    for name, rel, old, new, what in MUTANTS:
        if names and name not in names:
            continue
        r = run_one(name, rel, old, new)
        r['what'] = what
        res[name] = r
        print(name, r['failed'], 'failed', r['passed'], 'passed', r['failed_tests'][:6], flush=True)
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main(sys.argv[1:])
