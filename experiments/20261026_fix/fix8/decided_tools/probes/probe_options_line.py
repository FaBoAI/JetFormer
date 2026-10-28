"""fix8 tools-decided(読むだけ・作り物の log.json だけ): ② runs_1103 の表の下の試験の選択肢の行(options_line)と印(round_options)を、
手順書の 4 つの形で今と直す前の写しに通す。本「—」(run_id の無い本)を区切りに並べるのを直すか決める材料。
作り物は ../../tmp_tools_decided/opt/ に作って消す。出力は ../out/probe_options_line.txt"""
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
from test_fix3_r1_2 import load_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_decided/opt'
BASE = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
        'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}
G = {'rec_back_guard': 1, 'rec_hold_keep': 1}
OFF = {}
# (記録の名前, 説明欄, 設定)。block は bench_candidates.json の 11/3 のブロック(1 = 1-1 …)
CASES = {
    'A 手順書どおり(朝の確かめを収集したまま・memo に「試し」→ 試験の選択肢 ON → 1-1・1-2)': [
        ('20261103-081500', 'ai_no=1 memo=試し', {'launch_boost': 1}), ('20261103-083000', 'block=1 ai_no=2', G),
        ('20261103-084000', 'block=2 ai_no=3', G)],
    'B わざと OFF にした日(1-3 から OFF・紙に「本 1-3 から」)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G), ('20261103-085000', 'block=3 ai_no=4', OFF),
        ('20261103-090000', 'block=4 ai_no=5', OFF)],
    'C 起こし直し(1-3 の前に再起動して入れ直し忘れ。紙に何も無い)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G), ('20261103-085000', 'block=3 ai_no=4', OFF)],
    'D 昼の本「—」(回の本の間に block の無い走行。そのあと起こし直しのまま 3-1)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-120000', 'ai_no=9 memo=試し', OFF), ('20261103-121000', 'block=12 ai_no=10', OFF)],
}


def run(path, raw, tag):
    m = load_mod('runs_probe_%s' % tag, path)
    m.RAW = raw
    sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main()
    lines = buf.getvalue().strip().splitlines()
    rows = json.loads(lines[-1])
    opt = [x for x in lines if x.startswith('試験の選択肢(')]
    return opt, [(r['log'], r['run_id'] or '—', [x for x in r['diffs'] if '試験の選択肢' in x]) for r in rows]


def main():
    out = []
    for i, (name, recs) in enumerate(CASES.items()):
        raw = TMP / ('c%d' % i) / 'raw'
        for log, desc, kw in recs:
            d = raw / log
            d.mkdir(parents=True)
            dd = desc if 'block=' not in desc else 'day=2026-11-03 official=1 battery=#A ' + desc
            (d / 'log.json').write_text(json.dumps({'description': dd, 'drive': {
                'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': dict(BASE, **kw), 'dry_run': False}}),
                encoding='utf-8')
        out.append('## ' + name)
        for tag, path in (('今', JF / 'experiments/20261026_fix/plan/runs_1103.py'),
                          ('直す前', JF / 'tests/fixtures/fix8_tools_decided_before/runs_1103.py.orig')):
            opt, rows = run(path, raw, '%d_%s' % (i, 'n' if tag == '今' else 'o'))
            out.append('- %s: %s' % (tag, opt[0] if opt else '(行なし)'))
            for lg, rid, mk in rows:
                if mk:
                    out.append('    印 %s(本 %s): %s' % (lg, rid, mk[0]))
    txt = '\n'.join(out) + '\n'
    (HERE.parent / 'out' / 'probe_options_line.txt').write_text(txt, encoding='utf-8')
    print(txt)
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
