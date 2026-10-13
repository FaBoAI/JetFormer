#!/usr/bin/env python3
"""コピーした dist に、教師モジュールだけを差し替えるページ rl.html と rl/ の JS を置く(元のファイルは変えない)

    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/install_wrapper.py [teacher_module.js]

  - rl.html = dist/index.html と同じ内容で、import map だけに次を足したもの:
      imports: "./rc-speed-controller.js" → "./rl/<teacher_module>"  (app.js の import をこのページでだけ差し替える)
      scopes:  "./rl/" の中からの "./rc-speed-controller.js" は元のファイルのまま(差し替えモジュールが元を読める)
  - app.js・sim-core.js・dataset_recorder.py などは一切変えないので、収集のZIPの形式はアプリ本来のものと同じ。
  - dist の元のファイルが、ユーザーのシミュレーターの dist とバイト単位で同じかを毎回確かめる。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER_DIST = Path('/home/jetson/sim/extracted/course-simulator/dist')
DIST = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/sim/dist')
OLD_MAP = '<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/addons/"}}</script>'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    module = sys.argv[1] if len(sys.argv) > 1 else 'rc-teacher-probe.js'
    # 1) コピーの dist が元と同じか(rl.html と rl/ は除く)
    diff = []
    for p in sorted(USER_DIST.rglob('*')):
        if p.is_file():
            q = DIST / p.relative_to(USER_DIST)
            if not q.is_file() or sha(p) != sha(q):
                diff.append(str(p.relative_to(USER_DIST)))
    if diff:
        sys.exit(f'コピーの dist が元と違います: {diff[:5]}')
    # 2) rl/ に JS を置く
    (DIST / 'rl').mkdir(exist_ok=True)
    for js in sorted((HERE / 'web').glob('*.js')):
        shutil.copy2(js, DIST / 'rl' / js.name)
    if not (DIST / 'rl' / module).is_file():
        sys.exit(f'web/{module} がありません')
    # 3) rl.html を作る
    html = (DIST / 'index.html').read_text(encoding='utf-8')
    if html.count(OLD_MAP) != 1:
        sys.exit('index.html の import map が想定と違います')
    imap = {'imports': {'three': './vendor/three.module.js', 'three/addons/': './vendor/addons/',
                        './rc-speed-controller.js': f'./rl/{module}'},
            'scopes': {'./rl/': {'./rc-speed-controller.js': './rc-speed-controller.js'}}}
    new = html.replace(OLD_MAP, '<script type="importmap">' + json.dumps(imap, separators=(',', ':')) + '</script>')
    new = new.replace('<title>Course Lab — AI走行評価</title>', '<title>Course Lab — AI走行評価(RL教師の差し替え)</title>')
    (DIST / 'rl.html').write_text(new, encoding='utf-8')
    print(json.dumps({'rl_html': str(DIST / 'rl.html'), 'teacher_module': module, 'import_map': imap,
                      'dist_identical_files': sum(1 for p in USER_DIST.rglob('*') if p.is_file())}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
