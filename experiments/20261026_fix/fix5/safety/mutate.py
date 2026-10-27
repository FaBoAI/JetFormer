"""fix5 係 safety: 「わざと壊すと落ちる」の確かめ。jetformer/・tests/・pytest.ini を一時の場所(fix5/tmp_safety/mut_<名前>)に
写し、写しの assist.py・drive.py の直した所を 1 か所ずつ戻して(または決まりを壊して)、写しの上で試験を流す。本物の
jetformer/ は変えない(ほかの係が同時に試験を流していても影響しない)。終わったら写しは消す。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix5/safety/mutate.py [名前 ...]
出力: fix5/safety/out/mutate.json(名前ごとに 落ちた試験の数・通った数・落ちた試験の名前)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments' / '20261026_fix' / 'fix5' / 'tmp_safety'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TESTS = ['tests/test_rec_fix5_safety.py']

# (名前, ファイル, 前の文, あとの文, 何を壊すか)
MUTANTS = [
    ('s1_revert', 'jetformer/assist.py',
     "                or (hold_g >= self.IMPACT_G and erpm > -self.HOLD_IMPACT_FWD_ERPM)):",
     "                or hold_g >= self.IMPACT_G):",
     's1 を戻す(衝撃だけで、走っている車も中止)'),
    ('s2_revert_assist', 'jetformer/assist.py',
     "            if rec_keep:                # このフレームは見た(fix5 s2: 次の中立止めの「見ていない時間」をここから数える)\n"
     "                self.rec.t_last = now\n",
     "",
     's2 を戻す(発進の前進のフレームで t_last を進めない)'),
    ('s2_revert_drive', 'jetformer/drive.py',
     "                                       rec_keep=(int(p.get('rec_hold_keep', 0) or 0) == 1\n"
     "                                                 and int(p.get('rec_back_guard', 0) or 0) == 1))",
     "                                       rec_keep=False)",
     's2 を drive で戻す(rec_keep を渡さない)'),
    ('s3_revert_assist', 'jetformer/assist.py',
     "            self.backed_s += hb if count_unseen else seen       # hold_unseen_s = 0 なら どちらも hold_back_s(今までと同じ)",
     "            self.backed_s += hb",
     's3 を assist で一部戻す(値が来なかった間も、下がる前の段で下がった時間に数える)'),
    ('s3_revert_assist_phase', 'jetformer/assist.py',
     "        if seen > 0.0:                                  # 実際に後ろへ動いていたサンプルがあった(fix5 s3。前は hold_back_s > 0)",
     "        if hb > 0.0:",
     's3 を assist で一部戻す(値が来なかっただけでも段を進める)'),
    ('s3_revert_drive', 'jetformer/drive.py',
     "            hold_unseen_s = hold_veh[3] if held else 0.0       # そのうち値が来なかっただけの秒(fix5 s3)",
     "            hold_unseen_s = 0.0",
     's3 を drive で戻す(値が来なかっただけの秒を分けない)'),
    ('off_break_guard', 'jetformer/drive.py',
     "            if (int(p.get('rec_hold_keep', 0) or 0) == 1 and int(p.get('rec_back_guard', 0) or 0) == 1\n"
     "                    and veh_select and not dry_run):",
     "            if (int(p.get('rec_hold_keep', 0) or 0) == 1\n"
     "                    and veh_select and not dry_run):",
     'OFF を壊す(見張りが OFF でも中立止めでリカバリーを残す)= OFF の写しとの比べが落ちるか'),
    ('rule_break_s1_thresh', 'jetformer/assist.py',
     "    HOLD_IMPACT_FWD_ERPM = 1000 ",
     "    HOLD_IMPACT_FWD_ERPM = 2500 ",
     's1 のしきい値を緩める(1.25 m/s まで衝撃で中止)'),
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


if __name__ == '__main__':
    main(sys.argv[1:])
