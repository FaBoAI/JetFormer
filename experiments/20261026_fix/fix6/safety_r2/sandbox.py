"""fix6 係 safety 直す 6-2 回目: jetformer/・tests/・pytest.ini を一時の場所(fix6/tmp_safety_r2_sb/<名前>)に写し、写しの
jetformer/assist.py を final/assist.py(または指定のファイル)にして、写しの上で pytest を流す。わざと壊す(mutate)ときは、さらに
写しの assist.py の 1 か所を書き換える。本物の jetformer/・tests/ は変えない。終わったら写しは消す。装置・GPU・API には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix6/safety_r2/sandbox.py final <試験 ...>
    ... sandbox.py mutate [名前 ...]          → out/mutate2.json
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
TMP = JF / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_safety_r2_sb'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
MUT_TESTS = ['tests/test_rec_fix6_safety_r2.py']

# (名前, 前の文, あとの文, 何を壊すか)。どれも final/assist.py の文
MUTANTS = [
    ('back_hold_est', "        if seen > 0.0:\n            if p in ('brake', 'wait'):",
     "        if seen > 0.0 or est > 0.0:\n            if p in ('brake', 'wait'):",
     "見積もりだけで 'brake'・'wait' から 'reverse' へ進める(指摘 1 の形に戻す)"),
    ('check_backed', "            if self.backed_s - self.est_s == 0.0 and now - self.t_check > CHECK_S:",
     "            if self.backed_s == 0.0 and now - self.t_check > CHECK_S:",
     'やり直しを backed_s(見積もりを含む)で決める(指摘 2 の形に戻す)'),
    ('retry_keeps_est', "                    self.backed_s = self.est_s = 0.0       # 後退側を出しても下がらない: 見積もりの後退も無かった(fix6 r2)",
     "                    pass",
     'やり直すとき見積もりを捨てない(やり直したあと見積もりの分だけ短い)'),
    ('est_s_not_kept', "                self.est_s += add       # fix6 r2 s3-c: 見積もりで足した分('reverse' のやり直しは見た後退だけで決める)",
     "                pass",
     '見積もりの秒を est_s に持たない(見た後退と分けない = やり直しが止まる)'),
    ('est_not_counted', "                self.backed_s += add\n",
     "                pass\n",
     '見積もりを下がった時間に足さない(fix6 r1 の指摘の形 = 多く下がるに戻す)'),
    ('drop_est_at_gap', "                return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY\n            self._go('reverse', now)",
     "                return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY\n"
     "            self.backed_s = max(0.0, self.backed_s - self.est_s)\n            self.est_s = 0.0\n            self._go('reverse', now)",
     "'gap' → 'reverse' で見積もりを捨てる(選ばなかった形。当たっていた見積もりも捨てて多く下がる)"),
    ('off_break', "        self.est_s = 0.0             # backed_s のうち、見ていない間の後退の見積もり(hold_rev_s)で足した秒(2026-10-27 fix6 r2 s3-c)",
     "        self.est_s = 1e-9",
     'OFF を壊す(est_s の始めを 0 でなくする = 中立止めの無い走行のやり直しが変わる)'),
    ('off_break_start', "        self.backed_s = 0.0\n        self.est_s = 0.0\n        self.back_t = None",
     "        self.backed_s = 0.0\n        self.est_s = 1e-9\n        self.back_t = None",
     'OFF を壊す(手順の始めの est_s を 0 でなくする = OFF でも CHECK_S のやり直しが起きない)'),
]


def make(name, assist_src):
    d = TMP / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copytree(JF / 'jetformer', d / 'jetformer', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(JF / 'tests', d / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(JF / 'pytest.ini', d / 'pytest.ini')
    shutil.copyfile(assist_src, d / 'jetformer' / 'assist.py')
    return d


def pytest_in(d, tests, timeout=6000):
    bt = d / 'bt'
    try:
        p = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rf', '--basetemp=%s' % bt] + tests,
                           cwd=str(d), capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
    finally:
        shutil.rmtree(d, ignore_errors=True)
    failed = sorted(set(re.findall(r'^FAILED (\S+)', out, re.M)))
    m = re.search(r'(\d+) failed', out)
    m2 = re.search(r'(\d+) passed', out)
    return {'failed': int(m.group(1)) if m else 0, 'passed': int(m2.group(1)) if m2 else 0,
            'failed_tests': failed, 'tail': out.strip().splitlines()[-3:], 'out': out}


def main(args):
    try:
        if args[0] == 'final':
            r = pytest_in(make('final', HERE / 'final' / 'assist.py'), args[1:])
            print(r['out'][-6000:])
            return
        names = args[1:]
        res = {}
        outp = HERE / 'out' / 'mutate2.json'
        if outp.exists():
            res = json.loads(outp.read_text(encoding='utf-8'))
        src = (HERE / 'final' / 'assist.py').read_text(encoding='utf-8')
        for name, old, new, what in MUTANTS:
            if names and name not in names:
                continue
            assert src.count(old) == 1, (name, src.count(old))
            mp = HERE / 'out' / ('mut_%s_assist.py' % name)
            mp.write_text(src.replace(old, new), encoding='utf-8')
            try:
                r = pytest_in(make('mut_' + name, mp), MUT_TESTS)
            finally:
                mp.unlink()
            r.pop('out')
            r['what'] = what
            res[name] = r
            print(name, r['failed'], 'failed', r['passed'], 'passed', r['failed_tests'][:8], flush=True)
            outp.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main(sys.argv[1:])
