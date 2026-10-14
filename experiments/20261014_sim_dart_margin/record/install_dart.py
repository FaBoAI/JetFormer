#!/usr/bin/env python3
"""準備の段: コピーした dist に、DART・余裕の線の先生で走る差し替えページ d1.html・d2.html・d3.html を置く(2026-10-14、sim_dart_20261014)

    python -I ~/jetformer/experiments/20261014_sim_dart_margin/record/install_dart.py

  1) コピーの dist(/home/jetson/jetformer_reports/sim_dart_20261014/sim/dist)がユーザーの dist とバイト単位で同じかを確かめる
     (足したもの dist/dart/ と d1〜d3.html は除く)
  2) web/rt/ の写しが元(modes の rl/runtime・ベストラップの dist/rl/policy_weights.mjs)と sha256 で同じか、
     weights_margin.mjs の中身が line r2 の policy.json と同じか、weights_fast.mjs の中身が T_fast の policy.json と同じかを確かめる
  3) web/dart_manifest.mjs(データセットごとの先生・乱し・塩・方策の sha256、web/ のファイルの sha256)を書く
  4) web/ を dist/dart/ に写し、d1〜d3.html を作る(index.html と同じ内容で、import map だけに
       imports: "./rc-speed-controller.js" → "./dart/entry-Dn.js"
       scopes:  "./dart/" の中からの "./rc-speed-controller.js" は元のファイルのまま
     を足したもの)
  結果は /home/jetson/jetformer_reports/sim_dart_20261014/record_prep/install.json にも書く。ユーザーのファイルは変えない。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261014_sim_dart_margin')
WEB = EXP / 'web'
REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
DIST = REPORT / 'sim' / 'dist'
USER_DIST = Path('/home/jetson/sim/extracted/course-simulator/dist')
MODES = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
BEST = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
OLD_MAP = '<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/addons/"}}</script>'
POLICY_FAST = BEST / 'rl' / 'best' / 'main' / 'policy.json'
POLICY_MARGIN = MODES / 'candidates' / 'line_r2' / 'policy.json'
PAGES = {'D1': 'd1.html', 'D2': 'd2.html', 'D3': 'd3.html'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def esm_body(p):
    text = Path(p).read_text(encoding='utf-8')
    return json.loads(text[text.index('export default ') + len('export default '):].strip().rstrip(';'))


def main():
    protocol = json.loads((REPORT / 'protocol.json').read_text())
    psha = sha(REPORT / 'protocol.json')
    if psha != (REPORT / 'protocol.sha256').read_text().split()[0]:
        sys.exit('protocol.json の sha256 が protocol.sha256 と違います')
    # 1) コピーの dist が元と同じか
    diff = []
    for p in sorted(USER_DIST.rglob('*')):
        if p.is_file():
            q = DIST / p.relative_to(USER_DIST)
            if not q.is_file() or sha(p) != sha(q):
                diff.append(str(p.relative_to(USER_DIST)))
    extra = sorted(str(p.relative_to(DIST)) for p in DIST.rglob('*') if p.is_file() and not (USER_DIST / p.relative_to(DIST)).is_file()
                   and not str(p.relative_to(DIST)).startswith('dart/') and p.name not in PAGES.values())
    if diff or extra:
        sys.exit(f'コピーの dist が元と違います: {diff[:5]} 余分: {extra[:5]}')
    # 2) 写しの確かめ
    checks = {}
    for p in sorted((MODES / 'rl' / 'runtime').iterdir()):
        if p.suffix in ('.mjs', '.json'):
            q = WEB / 'rt' / p.name
            if sha(p) != sha(q):
                sys.exit('rt/ の写しが元と違います: ' + p.name)
            checks[f'rt/{p.name}'] = sha(q)
    if sha(BEST / 'sim' / 'dist' / 'rl' / 'policy_weights.mjs') != sha(WEB / 'rt' / 'weights_fast.mjs'):
        sys.exit('weights_fast.mjs が ベストラップの policy_weights.mjs と違います')
    if sha(POLICY_FAST) != protocol['teachers']['T_fast']['policy_sha256']:
        sys.exit('T_fast の policy.json の sha256 が protocol と違います')
    if esm_body(WEB / 'rt' / 'weights_fast.mjs') != json.loads(POLICY_FAST.read_text()):
        sys.exit('weights_fast.mjs の中身が T_fast の policy.json と違います')
    if sha(POLICY_MARGIN) != protocol['teachers']['T_margin']['policy_sha256']:
        sys.exit('T_margin の policy.json の sha256 が protocol と違います')
    if esm_body(WEB / 'rt' / 'weights_margin.mjs') != json.loads(POLICY_MARGIN.read_text()):
        sys.exit('weights_margin.mjs の中身が line r2 の policy.json と違います')
    gm_src = (MODES / 'rl' / 'gate_miss.mjs').read_text(encoding='utf-8').replace("'./runtime/features.mjs'", "'./rt/features.mjs'")
    if gm_src != (WEB / 'gate_miss.mjs').read_text(encoding='utf-8'):
        sys.exit('gate_miss.mjs の写しが元(import の道だけ変えたもの)と違います')
    # 3) manifest
    rec = protocol['recording']['datasets']
    ids = protocol['recording']['teacher_metadata']['ids']
    files = {}
    for p in sorted(WEB.rglob('*')):
        if p.is_file() and p.name != 'dart_manifest.mjs':
            files[str(p.relative_to(WEB))] = sha(p)
    t_fast, t_margin = protocol['teachers']['T_fast'], protocol['teachers']['T_margin']
    manifest = {
        'schema': 'jf-sim-dart-browser-manifest/v1',
        'protocol_sha256': psha,
        'datasets': {
            'D1': {'kind': 'fast', 'noise': 'N_fast', 'salt': 0xD1A7, 'teacher_id': ids['D1'], 'collection_name': rec['D1']['collection_name'],
                   'policy_sha256': t_fast['policy_sha256'], 'weights_sha256': files['rt/weights_fast.mjs'],
                   'planned_single_lap_s': t_fast['measured_now']['E0_lap_s'], 'note': 'T_fast(ベストラップの RL 方策)+ DART の乱し N_fast'},
            'D2': {'kind': 'margin', 'noise': None, 'salt': None, 'teacher_id': ids['D2'], 'collection_name': rec['D2']['collection_name'],
                   'policy_sha256': t_margin['policy_sha256'], 'weights_sha256': files['rt/weights_margin.mjs'],
                   'planned_single_lap_s': t_margin['measured_now']['E0_lap_s'], 'note': 'T_margin(経路いろいろ line r2、c0 = +0.3・c_v = 1.0・線 v2)乱しなし'},
            'D3': {'kind': 'margin', 'noise': 'N_margin', 'salt': 0xD3A7, 'teacher_id': ids['D3'], 'collection_name': rec['D3']['collection_name'],
                   'policy_sha256': t_margin['policy_sha256'], 'weights_sha256': files['rt/weights_margin.mjs'],
                   'planned_single_lap_s': t_margin['measured_now']['E0_lap_s'], 'note': 'T_margin + DART の乱し N_margin'},
        },
        'files': files,
    }
    (WEB / 'dart_manifest.mjs').write_text(
        '// 準備の段の record/install_dart.py が書いたもの(手で変えない)。データセットごとの先生・乱し・塩と、web/ のファイルの sha256\n'
        'export const DART_MANIFEST = Object.freeze(' + json.dumps(manifest, ensure_ascii=False, indent=1) + ');\n', encoding='utf-8')
    # 4) dist/dart/ と d1〜d3.html
    target = DIST / 'dart'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(WEB, target, ignore=shutil.ignore_patterns('__pycache__'))
    for p in WEB.rglob('*'):
        if p.is_file() and sha(p) != sha(target / p.relative_to(WEB)):
            sys.exit('dist/dart/ への写しが違います: ' + str(p))
    html = (DIST / 'index.html').read_text(encoding='utf-8')
    if html.count(OLD_MAP) != 1:
        sys.exit('index.html の import map が想定と違います')
    pages = {}
    for d, page in PAGES.items():
        imap = {'imports': {'three': './vendor/three.module.js', 'three/addons/': './vendor/addons/', './rc-speed-controller.js': f'./dart/entry-{d}.js'},
                'scopes': {'./dart/': {'./rc-speed-controller.js': './rc-speed-controller.js'}}}
        new = html.replace(OLD_MAP, '<script type="importmap">' + json.dumps(imap, separators=(',', ':')) + '</script>')
        new = new.replace('<title>Course Lab — AI走行評価</title>', f'<title>Course Lab — AI走行評価(DART {d} の先生の差し替え)</title>')
        (DIST / page).write_text(new, encoding='utf-8')
        pages[d] = {'page': page, 'sha256': sha(DIST / page), 'import_map': imap}
    out = {'dist': str(DIST), 'dist_identical_files': sum(1 for p in USER_DIST.rglob('*') if p.is_file()), 'copies_checked': checks,
           'manifest_sha256': sha(WEB / 'dart_manifest.mjs'), 'pages': pages, 'manifest': manifest}
    (REPORT / 'record_prep').mkdir(exist_ok=True)
    (REPORT / 'record_prep' / 'install.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ('dist', 'dist_identical_files', 'manifest_sha256')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
