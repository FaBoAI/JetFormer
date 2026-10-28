"""反証 8-2 回目・係 ui: ② runs_1103.py(8-1 の直しの版 c2d616c2)の試験の選択肢の印(round_options = その回で ON がいちばん多い本と
比べる)を、作り物の log.json のいろいろな形(回の全部が OFF・同じ数の ON・回の途中で 2 回起き直し など。両方 ON の日 = 計画は
見張り・残す ON)に通す。比べ: 8-1 の前の写し(tests/fixtures/fix8_ui_r1_before/runs_1103.py)。
作り物の記録はこの場所の tmp/raw_*(終わって消す)。data/raw・bench_candidates.json は読むだけ。出力 out/p5_runs.txt"""
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
NOW = Path('/home/jetson/jetformer/experiments/20261026_fix/plan/runs_1103.py')
BEFORE = Path('/home/jetson/jetformer/tests/fixtures/fix8_ui_r1_before/runs_1103.py')
BASE = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
        'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}
GK = {'rec_back_guard': 1, 'rec_hold_keep': 1}            # 11/3 の計画(両方 ON)
G = {'rec_back_guard': 1}                                 # 残すを入れ忘れ
GN = {'rec_back_guard': 1, 'launch_no_kick': 1}           # 残すのかわりに補助なしを押した
OFF = {}
RAWB = json.loads(Path('/home/jetson/jetformer/data/bench_candidates.json').read_text(encoding='utf-8'))
DAY = next(d for d in RAWB['days'] if d.get('date') == '2026-11-03')
EXPECT = {int(b['block']): {k: v for k, v in (b.get('expect') or {}).items() if v is not None} for b in DAY.get('blocks') or []}
RUNID = {int(b['block']): b.get('run_id') for b in DAY.get('blocks') or []}
BLK = {v: k for k, v in RUNID.items() if v}               # 'R-N' → ブロックの番号


def recs(spec):
    """spec = [(run_id か None, 選択肢, memo)] → [(記録名, 説明欄, 設定)]"""
    out = []
    for i, (rid, opt, memo) in enumerate(spec):
        log = '20261103-%02d%02d00' % (8 + i // 6, (i % 6) * 10)
        if rid is None:
            desc = 'ai_no=%d%s' % (i + 1, ' memo=' + memo if memo else '')
            p = dict(BASE, **opt)
        else:
            b = BLK[rid]
            desc = 'bench=real_bench_20261023 day=2026-11-03 official=1 battery=#A block=%d ai_no=%d%s' % (b, i + 1, ' memo=' + memo if memo else '')
            p = dict(BASE, **EXPECT.get(b, {}), **opt)
        out.append((log, desc, p))
    return out


R1 = [('1-1', GK, ''), ('1-2', GK, ''), ('1-3', GK, '')]
CASES = {
    'S1 回 2 の全部が OFF(回 2 の前に起き直し・入れ直さないまま回 2 を走った)': R1 + [('2-1', OFF, ''), ('2-2', OFF, ''), ('2-3', OFF, '')] + [('3-1', GK, ''), ('3-2', GK, '')],
    'S2a 同じ数の ON(2-2 だけ残すのかわりに補助なし)': R1 + [('2-1', GK, ''), ('2-2', GN, ''), ('2-3', GK, '')],
    'S2b 同じ数の ON(回の最初の 2-1 が残すのかわりに補助なし、2-2 から直した)': R1 + [('2-1', GN, ''), ('2-2', GK, ''), ('2-3', GK, '')],
    'S2c 残すの入れ忘れ(2-1・2-3 は見張りだけ)': R1 + [('2-1', G, ''), ('2-2', GK, ''), ('2-3', G, '')],
    'S3a 回の途中で 2 回起き直し(ON・OFF・入れ直し ON・OFF・入れ直し ON)': R1 + [('2-1', GK, ''), ('2-2', OFF, ''), ('2-3', GK, ''), ('2-4', OFF, ''), ('2-5', GK, '')],
    'S3b 回の途中で 2 回起き直し(1 回目のあと見張りだけ入れ直し)': R1 + [('2-1', GK, ''), ('2-2', OFF, ''), ('2-3', G, ''), ('2-4', OFF, ''), ('2-5', GK, '')],
    'S3c 回の最初が OFF + 途中でもう 1 回起き直し(OFF・ON・OFF・ON)': R1 + [('2-1', OFF, ''), ('2-2', GK, ''), ('2-3', OFF, ''), ('2-4', GK, '')],
    'S3d 回の頭から 2 本 OFF(起き直しを 2 回)、3 本目から入れ直し': R1 + [('2-1', OFF, ''), ('2-2', OFF, ''), ('2-3', GK, '')],
    'S4 起き直し(2-2)+ 3 章で止めて OFF(紙: 本 2-4 から)': R1 + [('2-1', GK, ''), ('2-2', OFF, ''), ('2-3', GK, ''), ('2-4', OFF, ''), ('2-5', OFF, '')],
    'S5 起き直しのまま 2-1 を走り「試し」と書き、入れ直して 2-1 を走り直す': R1 + [('2-1', OFF, '試し(起き直しのまま)'), ('2-1', GK, ''), ('2-2', GK, '')],
    'S6 朝の確かめ(本「—」)→ 1-1 を入れ忘れ → 1-2 から ON → 回 2 の前に起き直し → 2-1 OFF → 2-2 から ON': [(None, OFF, '試し(朝の確かめ)'), ('1-1', OFF, ''), ('1-2', GK, ''), ('1-3', GK, ''), ('2-1', OFF, ''), ('2-2', GK, ''), ('2-3', GK, '')],
}
PLAN = set(GK)


def load(path, tag):
    spec = importlib.util.spec_from_file_location('runs_%s' % tag, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_one(path, tag, raw):
    m = load(path, tag)
    m.RAW = raw
    sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main()
    lines = buf.getvalue().strip().splitlines()
    return m, lines[:-1], json.loads(lines[-1])


def main():
    out = ['runs_1103.py 今 %s・前 %s' % (hashlib.sha256(NOW.read_bytes()).hexdigest()[:12], hashlib.sha256(BEFORE.read_bytes()).hexdigest()[:12]),
           '印 = 「回 N の試験の選択肢が … と違う」。正しい = 計画(見張り・残す ON・補助なし OFF)と違う本の全部に印、計画どおりの本には印なし']
    tmp = HERE / 'tmp' / 'raw_p5'
    shutil.rmtree(tmp, ignore_errors=True)
    score = {'今': [0, 0], '前': [0, 0]}
    for ci, (name, spec) in enumerate(CASES.items()):
        raw = tmp / ('c%d' % ci)
        for log, desc, p in recs(spec):
            d = raw / log
            d.mkdir(parents=True)
            (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
                'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': p, 'dry_run': False}}), encoding='utf-8')
        out.append('## ' + name)
        for tag, path in (('今', NOW), ('前', BEFORE)):
            m, txt, rows = run_one(path, '%s%d' % (tag, ci), raw)
            ol = [x for x in txt if x.startswith('試験の選択肢(')]
            tot = [x for x in txt if ' 本・計画と違う ' in x]
            marked, should = [], []
            for r in rows:
                on = {k for k in m.OPT_KEYS if r['options'].get(k)}
                mk = [x for x in r['diffs'] if '試験の選択肢' in x]
                if mk:
                    marked.append(r['run_id'] or '—')
                if r['run_id'] and on != PLAN:
                    should.append(r['run_id'])
            ok = sorted(marked) == sorted(should)
            score[tag][0] += ok
            score[tag][1] += 1
            out.append('  [%s] 印: %s / 計画と違う本: %s → %s・%s' % (tag, '・'.join(marked) or 'なし', '・'.join(should) or 'なし', '合う' if ok else '!! 合わない', tot[0] if tot else ''))
            if tag == '今':
                out.append('       表の下: %s' % (re.sub(r'^試験の選択肢\([^)]*\): ', '', ol[0]) if ol else '(行なし)'))
                for r in rows:
                    mk = [x for x in r['diffs'] if '試験の選択肢' in x]
                    if mk:
                        out.append('       %s(%s): %s' % (r['run_id'] or '—', r['log'], mk[0].split('(再起動で')[0]))
    shutil.rmtree(tmp)
    out.append('')
    out.append('合う形: 今 %d / %d・前 %d / %d' % (score['今'][0], score['今'][1], score['前'][0], score['前'][1]))
    txt = '\n'.join(out) + '\n'
    (HERE / 'out' / 'p5_runs.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
