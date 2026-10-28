"""② runs_1103.py(fix8 tools-decided の版 33bc7fca)を作り物の log.json(この場所の tmp/raw_*。終わって消す)に通す。読むだけ。
回の最初の本が OFF(起き直しのあと入れ直す前に 1 本走った)で、2 本目から入れ直して ON の日に、印と「計画と違う N 本」がどの本に付くか。
比べ: 手順書どおり(全部 ON)・回の途中で起き直した日(1-3 から OFF)・3 章で止めて OFF にした日。出力 out/runs_first_off.txt"""
import contextlib
import hashlib
import importlib.util
import io
import json
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
    'E 回 2 の前に起き直し、2-1 を OFF のまま走り、2-2 の前に気づいて入れ直した(紙: 2-1 は起き直しのまま)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G),
        ('20261103-100000', 'block=6 ai_no=7', OFF), ('20261103-101000', 'block=7 ai_no=8', G), ('20261103-102000', 'block=8 ai_no=9', G)],
    'K 朝の確かめ(本「—」)のあと選択肢を入れる前に 1-1 を走り、1-2 から ON': [
        ('20261103-081500', 'ai_no=1 memo=試し', OFF), ('20261103-083000', 'block=1 ai_no=2', OFF), ('20261103-084000', 'block=2 ai_no=3', G),
        ('20261103-085000', 'block=3 ai_no=4', G)],
    '比べ C 回の途中で起き直し(1-3 から OFF)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G), ('20261103-085000', 'block=3 ai_no=4', OFF)],
    '比べ 手順書どおり(全部 ON)': [
        ('20261103-083000', 'block=1 ai_no=2', G), ('20261103-084000', 'block=2 ai_no=3', G), ('20261103-085000', 'block=3 ai_no=4', G)],
}


RAWB = json.loads(Path('/home/jetson/jetformer/data/bench_candidates.json').read_text(encoding='utf-8'))
EXPECT = {int(b['block']): {k: v for k, v in (b.get('expect') or {}).items() if v is not None}
          for d in RAWB['days'] if d.get('date') == '2026-11-03' for b in d.get('blocks') or []}


def plan_params(desc):
    """説明欄の block の計画(expect の null でない値)どおりの設定(試験の選択肢のほかは計画と同じにする = 印は試験の選択肢だけ)"""
    import re
    m = re.search(r'block=(\d+)', desc)
    return dict(EXPECT.get(int(m.group(1)), {})) if m else {}


def load(tag):
    spec = importlib.util.spec_from_file_location('runs_%s' % tag, RUNS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    out = ['runs_1103.py sha256 %s' % hashlib.sha256(RUNS.read_bytes()).hexdigest()[:12]]
    tmp = HERE / 'tmp' / 'raw_first_off'
    shutil.rmtree(tmp, ignore_errors=True)
    for i, (name, recs) in enumerate(CASES.items()):
        raw = tmp / ('c%d' % i)
        for log, desc, kw in recs:
            d = raw / log
            d.mkdir(parents=True)
            dd = desc if 'block=' not in desc else 'bench=real_bench_20261023 day=2026-11-03 official=1 battery=#A ' + desc
            (d / 'log.json').write_text(json.dumps({'description': dd, 'drive': {
                'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': dict(BASE, **plan_params(desc), **kw), 'dry_run': False}}), encoding='utf-8')
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
            out.append('  %-16s 本 %-4s 選択肢 %-30s 印: %s' % (r['log'], r['run_id'] or '—', opt, mk[0] if mk else '—'))
    shutil.rmtree(tmp)
    txt = '\n'.join(out) + '\n'
    (HERE / 'out' / 'runs_first_off.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
