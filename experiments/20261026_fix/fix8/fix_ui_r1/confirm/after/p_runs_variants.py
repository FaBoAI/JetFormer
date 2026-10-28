"""反証 8-1 回目・係 ui の指摘 runs-opt-mark-on-the-on-runs を、11/3 の手順書どおりの「気づいたあと」の形で流し直す(打ち消しの票 B)。
読むだけ: 本物の runs_1103.py を、この場所の tmp/ の作り物の log.json に通す(終わって消す)。data/raw・bench_candidates.json は読むだけ。
形:
  E0 指摘のまま(回 2 の前に起き直し、2-1 を OFF のまま走り、2-2 の前に気づいて入れ直した)
  E1 E0 に手順書 7 章の「気づいたら、入れ直す前に走った本の走り終わりのメモに「試し」」を足した(2-1 の memo=試し)
  E2 回の途中で起き直し(2-1 ON・2-2 OFF のまま・2-3 の前に入れ直した)= 回の最初の本が ON
  K0 指摘のまま(朝の確かめのあと選択肢を入れる前に 1-1 を走り、1-2 から ON)
  K1 K0 に 1-1 の memo=試し
出力 out/runs_variants.txt"""
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
BASE = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
        'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}
G = {'rec_back_guard': 1, 'rec_hold_keep': 1}
OFF = {}
CASES = {
    'E0 回 2 の前に起き直し・2-1 OFF・2-2 の前に入れ直し(指摘のまま)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G),
        ('20261103-100000', 'block=6 ai_no=7', OFF), ('20261103-101000', 'block=7 ai_no=8', G), ('20261103-102000', 'block=8 ai_no=9', G)],
    'E1 E0 + 手順書 7 章どおり 2-1 の走り終わりのメモに「試し」': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G),
        ('20261103-100000', 'block=6 ai_no=7 memo=試し(起き直しのまま)', OFF), ('20261103-101000', 'block=7 ai_no=8', G),
        ('20261103-102000', 'block=8 ai_no=9', G)],
    'E2 回の途中で起き直し(2-1 ON・2-2 OFF・2-3 の前に入れ直し)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G),
        ('20261103-100000', 'block=6 ai_no=7', G), ('20261103-101000', 'block=7 ai_no=8', OFF), ('20261103-102000', 'block=8 ai_no=9', G)],
    'K0 朝に選択肢を入れ忘れて 1-1 を OFF・1-2 から ON(指摘のまま)': [
        ('20261103-081500', 'ai_no=1 memo=試し', OFF), ('20261103-083000', 'block=1 ai_no=2', OFF), ('20261103-084000', 'block=2 ai_no=3', G),
        ('20261103-085000', 'block=3 ai_no=4', G)],
    'K1 K0 + 1-1 の走り終わりのメモに「試し」': [
        ('20261103-081500', 'ai_no=1 memo=試し', OFF), ('20261103-083000', 'block=1 ai_no=2 memo=試し(選択肢の入れ忘れ)', OFF),
        ('20261103-084000', 'block=2 ai_no=3', G), ('20261103-085000', 'block=3 ai_no=4', G)],
}
RAWB = json.loads(Path('/home/jetson/jetformer/data/bench_candidates.json').read_text(encoding='utf-8'))
EXPECT = {int(b['block']): {k: v for k, v in (b.get('expect') or {}).items() if v is not None}
          for d in RAWB['days'] if d.get('date') == '2026-11-03' for b in d.get('blocks') or []}


def plan_params(desc):
    m = re.search(r'block=(\d+)', desc)
    return dict(EXPECT.get(int(m.group(1)), {})) if m else {}


def load(tag):
    spec = importlib.util.spec_from_file_location('runs_%s' % tag, RUNS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    out = ['runs_1103.py sha256 %s' % hashlib.sha256(RUNS.read_bytes()).hexdigest()[:12]]
    tmp = HERE / 'tmp' / 'raw'
    shutil.rmtree(HERE / 'tmp', ignore_errors=True)
    try:
        for i, (name, recs) in enumerate(CASES.items()):
            raw = tmp / ('c%d' % i)
            for log, desc, kw in recs:
                d = raw / log
                d.mkdir(parents=True)
                dd = desc if 'block=' not in desc else 'bench=real_bench_20261023 day=2026-11-03 official=1 battery=#A ' + desc
                (d / 'log.json').write_text(json.dumps({'description': dd, 'drive': {
                    'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': dict(BASE, **plan_params(desc), **kw),
                    'dry_run': False}}), encoding='utf-8')
            m = load(i)
            m.RAW = raw
            sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m.main()
            lines = buf.getvalue().strip().splitlines()
            rows = json.loads(lines[-1])
            out.append('## ' + name)
            out += ['  ' + x for x in lines[:-1] if x.startswith('試験の選択肢(') or ' 本・計画と違う ' in x]
            for r in rows:
                opt = '・'.join(k for k in m.OPT_KEYS if r['options'].get(k)) or 'なし'
                mk = [x for x in r['diffs'] if '試験の選択肢' in x]
                out.append('  %-16s 本 %-4s 選択肢 %-30s 印: %s' % (r['log'], r['run_id'] or '—', opt, mk[0][:60] if mk else '—'))
    finally:
        shutil.rmtree(HERE / 'tmp', ignore_errors=True)
    txt = '\n'.join(out) + '\n'
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'runs_variants_after.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
