#!/usr/bin/env python3
"""記録の段: コピーした dist に、RL のベストラップの方策で走る差し替えページ rl.html を置く(2026-10-13)

    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/install_bestlap.py [--policy-dir rl/best/main]

  1) install_wrapper.py rl-teacher-bestlap.js を呼ぶ(コピーの dist がユーザーの dist とバイト単位で同じかを確かめ、
     web/*.js を dist/rl/ に置き、rl.html の import map で ./rc-speed-controller.js を ./rl/rl-teacher-bestlap.js に差し替える)
  2) 方策の計算に要る ESM を dist/rl/ に写す(RL の学習・評価で使ったファイルそのもの):
       rl/features.mjs・rl/reference_loop.mjs・rl/policy_forward.mjs・<policy-dir>/policy_weights.mjs
  3) policy_weights.mjs の中身が policy.json(selection.json の sha256)と同じかを確かめる
  4) dist/rl/bestlap_manifest.mjs(写したファイルの sha256・方策の出どころ)を書く。教師の metadata はここから作る
  結果は /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/install.json にも書く。
  ユーザーのシミュレーター・アプリのファイルは変えない(dist/rl/ と dist/rl.html だけを作る)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
DIST = REPORT / 'sim' / 'dist'
MODULE = 'rl-teacher-bestlap.js'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy-dir', default=str(REPORT / 'rl' / 'best' / 'main'))
    a = ap.parse_args()
    pdir = Path(a.policy_dir)
    selection = json.loads((REPORT / 'rl' / 'selection.json').read_text())
    sel = selection['selected_main']
    prov = json.loads((pdir / 'provenance.json').read_text())
    # 方策の JSON が選んだもの(selection.json の sha256)と同じか
    if pdir.resolve() == Path(sel['policy_json']).parent.resolve() and sha(pdir / 'policy.json') != sel['policy_json_sha256']:
        sys.exit('policy.json の sha256 が selection.json と違います')
    # policy_weights.mjs の中身 = policy.json(先頭のコメント行と "export default " を除いた JSON を比べる)
    text = (pdir / 'policy_weights.mjs').read_text(encoding='utf-8')
    body = text[text.index('export default ') + len('export default '):].strip().rstrip(';')
    if json.loads(body) != json.loads((pdir / 'policy.json').read_text()):
        sys.exit('policy_weights.mjs と policy.json の中身が違います')
    # 1) rl.html と web/*.js
    r = subprocess.run([sys.executable, '-I', str(EXP / 'install_wrapper.py'), MODULE], capture_output=True, text=True)
    if r.returncode:
        sys.exit('install_wrapper.py が失敗しました: ' + r.stderr + r.stdout)
    wrapper = json.loads(r.stdout)
    # 2) 方策の計算に要る ESM
    sources = {'features.mjs': EXP / 'rl' / 'features.mjs', 'reference_loop.mjs': EXP / 'rl' / 'reference_loop.mjs',
               'policy_forward.mjs': EXP / 'rl' / 'policy_forward.mjs', 'policy_weights.mjs': pdir / 'policy_weights.mjs'}
    for name, src in sources.items():
        shutil.copy2(src, DIST / 'rl' / name)
        if sha(src) != sha(DIST / 'rl' / name):
            sys.exit('写したファイルが元と違います: ' + name)
    files = {name: sha(DIST / 'rl' / name) for name in [*sources, MODULE]}
    # 標準の開始の周回時間(表示用の planned_time_s): 選んだ方策は selection.json、ほか(r3 など)は同じフォルダーの final_eval.json
    if pdir.resolve() == Path(sel['policy_json']).parent.resolve() and prov['run'] in selection['runs']:
        best = selection['runs'][prov['run']]['best']
        std = {'std_single': best['std_single'], 'std_flying': best['std_flying']}
    elif (pdir / 'final_eval.json').is_file():
        ss = json.loads((pdir / 'final_eval.json').read_text())['selection_sets']
        std = {'std_single': ss['STD_single_lap_from_stop'], 'std_flying': ss['STD_flying_lap']}
    else:
        std = {}
    meta = json.loads((pdir / 'policy.json').read_text()).get('meta', {})
    steps = prov.get('step') or prov.get('step_finetune') or meta.get('steps')
    manifest = {
        'schema': 'rl-bestlap-browser-manifest/v1',
        'run': prov['run'], 'variant': prov['variant'], 'seed': prov['seed'], 'checkpoint_steps': steps,
        'policy_json_sha256': sha(pdir / 'policy.json'),
        'reward_version': prov['reward_version'], 'features_version': prov['features_version'],
        'reference_loop_sha256': prov['reference_loop_sha256'],
        'protocol_sha256': (REPORT / 'protocol.sha256').read_text().split()[0],
        'std_single_lap_s': std.get('std_single'), 'std_flying_lap_s': std.get('std_flying'),
        'files': files,
    }
    (DIST / 'rl' / 'bestlap_manifest.mjs').write_text(
        '// 記録の段の install_bestlap.py が書いたもの(手で変えない)。RL のベストラップの方策の出どころと、dist/rl/ に写したファイルの sha256\n'
        'export const BESTLAP_MANIFEST = Object.freeze(' + json.dumps(manifest, ensure_ascii=False, indent=1) + ');\n', encoding='utf-8')
    out = {'wrapper': wrapper, 'policy_dir': str(pdir), 'manifest': manifest,
           'bestlap_manifest_sha256': sha(DIST / 'rl' / 'bestlap_manifest.mjs'), 'rl_html_sha256': sha(DIST / 'rl.html')}
    (REPORT / 'recording').mkdir(exist_ok=True)
    (REPORT / 'recording' / 'install.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
