#!/usr/bin/env python3
"""DAgger の記録の準備(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 3・4)

    python3 -I prepare_record.py --select <select_teacher.mjs の結果.json> --arm <SDIV-s0 など>

  1) 状態集めの走行(closed_loop_dagger/<腕>/<腕>-Dn-single.json)と、先生の選び・k(select_teacher.mjs)から
     web/dagger_plan.mjs(開始ごとの k と、k までの JetFormer の操作の列)を書く。完走した走りは k = 300 だが、単周の記録は
     周が終わったフレームまでしか無く、最後のフレームは単周の sim が止めるので、k = min(300, 記録のフレーム数 − 1)とする
  2) 記録の連続試験の計画(12 シーン。REC60 の 1 条件目の物理・センサー、連続周回 20 秒、ジャンル 8、全部 train、種 1・baseline、
     開始は D0〜D11)を dagger/record_plan.json に書く
  3) 今日の写しの dist がユーザーの dist とバイト単位で同じことを確かめ(足したもの dist/dagger/・dagger.html は除く)、
     web/ を dist/dagger/ に写し、dagger.html(index.html と同じで、import map だけ ./rc-speed-controller.js → ./dagger/entry-DAG.js)を作る
  4) web/ の teacher_core.mjs・rt/・gate_miss.mjs が 10/14 の写しと sha256 で同じことを確かめる
結果は /home/jetson/jetformer_reports/sim_diverse_train_20261015/dagger/prepare_record.json。ユーザーのファイルは変えない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261015_sim_diverse_train')
PREV_WEB = Path('/home/jetson/jetformer/experiments/20261014_sim_dart_margin/web')
WEB = EXP / 'web'
REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
DIST = REPORT / 'sim' / 'dist'
USER_DIST = Path('/home/jetson/sim/extracted/course-simulator/dist')
REC60 = Path('/home/jetson/sim/extracted/course-simulator/qa/rc-collection-plan-60.json')
OUT = REPORT / 'dagger'
OLD_MAP = '<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/addons/"}}</script>'
DSTARTS = {'D0': (0, 0.04, 1.0), 'D1': (0, -0.04, -1.0), 'D2': (0, 0.01, 2.0), 'D3': (0, -0.01, -2.0), 'D4': (0, 0.06, -3.0), 'D5': (0, -0.06, 3.0),
           'D6': (0, 0.0, 1.0), 'D7': (0, 0.0, -1.0), 'D8': (5, 0.0, 0.0), 'D9': (15, 0.0, 0.0), 'D10': (24, 0.0, 0.0), 'D11': (31, 0.0, 0.0)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(g, o, y):
    return f'{g}|{o:.4f}|{y:.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', required=True)
    ap.add_argument('--arm', required=True, help='状態集めの腕のラベル(SDIV-s0 など)')
    ap.add_argument('--onnx-sha256', default='')
    a = ap.parse_args()
    sel = json.loads(Path(a.select).read_text())
    teacher = sel['teacher']
    scenes, plan, ks = {}, [], {}
    rows = {Path(r['file']).stem: r for r in sel['rows']}
    rec60 = json.loads(REC60.read_text(encoding='utf-8'))[0]
    for d, (g, o, y) in DSTARTS.items():
        name = f'{a.arm}-{d}-single'
        r = rows.get(name)
        if r is None:
            sys.exit(f'先生の選びの結果に {name} がありません')
        rep = json.loads(Path(r['file']).read_text())
        sv = rep['context']['start_variation']
        if (sv['gate'], round(sv['left_offset_m'], 4), round(sv['left_yaw_deg'], 4)) != (g, round(o, 4), round(y, 4)):
            sys.exit(f'{name} の開始が計画と違います: {sv}')
        n = len(rep['actions'])
        k = r['k']
        if k is None:
            sys.exit(f'{name}: 先生に任せて完走できるフレームがありません(k なし)。この開始は使えません')
        # 完走した走り: k = 300 だが単周の記録は周が終わったフレームまで。最後のフレーム(route_complete で単周の sim が止まるフレーム)は
        # 連続周回の sim と位置が違う(G1 の確かめで、最後の 1 フレームだけ 1〜14 cm 違った)ので、その 1 つ前で先生に替える
        completed = rep['outcome']['reason'] == 'route_complete' and rep['metrics']['collisions'] == 0
        k_eff = min(k, n - 1) if completed else min(k, n)
        ks[d] = {'k_protocol': k, 'k': k_eff, 'frames_single': n, 'reason': rep['outcome']['reason']}
        scenes[key(g, o, y)] = {'name': f'dagger-{a.arm}-{d}', 'start': [g, o, y], 'k': k_eff,
                                'actions': [[x['steering'], x['speed']] for x in rep['actions'][:k_eff]], 'source_run': name}
        c = dict(rec60)
        c.update({'name': f'dagger-{a.arm.lower()}-mode8-train-{d.lower()}', 'controller': 'demo', 'teacher-profile': 'speed-rc',
                  'episode-mode': 'continuous-laps-v1', 'limit': 20, 'dataset-mode': 8, 'dataset-split': 'train', 'scenario': 'baseline', 'seed': 1,
                  'start-gate': g, 'spawn-offset': o, 'spawn-yaw': y})
        plan.append(c)
    body = json.dumps(scenes, sort_keys=True, separators=(',', ':'))
    psha = sha(REPORT / 'protocol.json')
    wkey = 'weights_fast.mjs' if teacher == 'fast' else 'weights_margin.mjs'
    dplan = {'schema': 'jf-sim-diverse-dagger-plan/v1', 'teacher': teacher,
             'teacher_id': f'jf-sim-dagger-{a.arm}-{"T_fast" if teacher == "fast" else "T_margin"}-v1',
             'weights_sha256': sha(WEB / 'rt' / wkey), 'source_arm': a.arm, 'source_onnx_sha256': a.onnx_sha256 or None,
             'protocol_sha256': psha, 'planned_single_lap_s': 8.932 if teacher == 'margin' else 8.092,
             'plan_sha256': hashlib.sha256(body.encode()).hexdigest(), 'scenes': scenes}
    (WEB / 'dagger_plan.mjs').write_text('// dagger/prepare_record.py が書いたもの(手で変えない)。開始ごとの k と、k までの JetFormer の操作(状態集めの走行の記録)\n'
                                         'export const DAGGER_PLAN = Object.freeze(' + json.dumps(dplan, ensure_ascii=False) + ');\n', encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'record_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
    # 4) web/ の写しが 10/14 と同じか
    same = {}
    for p in sorted(PREV_WEB.rglob('*')):
        rel = p.relative_to(PREV_WEB)
        if p.is_file() and (str(rel).startswith('rt/') or rel.name in ('teacher_core.mjs', 'gate_miss.mjs')):
            q = WEB / rel
            same[str(rel)] = q.is_file() and sha(p) == sha(q)
    if not all(same.values()):
        sys.exit('web/ の写しが 10/14 と違います: ' + str([k for k, v in same.items() if not v]))
    # 3) dist の確かめと、dist/dagger/・dagger.html
    diff = []
    for p in sorted(USER_DIST.rglob('*')):
        if p.is_file():
            q = DIST / p.relative_to(USER_DIST)
            if not q.is_file() or sha(p) != sha(q):
                diff.append(str(p.relative_to(USER_DIST)))
    extra = sorted(str(p.relative_to(DIST)) for p in DIST.rglob('*') if p.is_file() and not (USER_DIST / p.relative_to(DIST)).is_file()
                   and not str(p.relative_to(DIST)).startswith(('dagger/', 'dart/')) and p.name not in ('dagger.html', 'd1.html', 'd2.html', 'd3.html'))
    # dart/・d1〜d3.html は 10/14 の写しから来た(DART の記録のページ。今日は使わない)
    if diff or extra:
        sys.exit(f'写しの dist が元と違います: {diff[:5]} 余分: {extra[:5]}')
    target = DIST / 'dagger'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(WEB, target, ignore=shutil.ignore_patterns('__pycache__'))
    for p in WEB.rglob('*'):
        if p.is_file() and sha(p) != sha(target / p.relative_to(WEB)):
            sys.exit('dist/dagger/ への写しが違います: ' + str(p))
    html = (DIST / 'index.html').read_text(encoding='utf-8')
    if html.count(OLD_MAP) != 1:
        sys.exit('index.html の import map が想定と違います')
    imap = {'imports': {'three': './vendor/three.module.js', 'three/addons/': './vendor/addons/', './rc-speed-controller.js': './dagger/entry-DAG.js'},
            'scopes': {'./dagger/': {'./rc-speed-controller.js': './rc-speed-controller.js'}}}
    new = html.replace(OLD_MAP, '<script type="importmap">' + json.dumps(imap, separators=(',', ':')) + '</script>')
    new = new.replace('<title>Course Lab — AI走行評価</title>', '<title>Course Lab — AI走行評価(DAgger の先生の差し替え)</title>')
    (DIST / 'dagger.html').write_text(new, encoding='utf-8')
    out = {'teacher': teacher, 'arm': a.arm, 'k': ks, 'plan': str(OUT / 'record_plan.json'), 'dagger_plan_sha256': sha(WEB / 'dagger_plan.mjs'),
           'scenes_body_sha256': dplan['plan_sha256'], 'web_same_as_1014': same, 'dist_identical_files': sum(1 for p in USER_DIST.rglob('*') if p.is_file()),
           'page': {'file': str(DIST / 'dagger.html'), 'sha256': sha(DIST / 'dagger.html'), 'import_map': imap},
           'dist_dagger_files': {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}}
    (OUT / 'prepare_record.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('teacher', 'arm', 'k', 'dagger_plan_sha256')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
