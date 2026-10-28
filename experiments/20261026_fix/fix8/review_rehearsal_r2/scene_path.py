"""反証 8-2 回目・係 rehearsal: 手順書 3 章 5・4 章 9・7 章「起こし直した」の行の終わりの道(走り終わりのメモの欄が出ない本を
「試し」にする: 画面の「シーン」のタブ → 名前がその本の記録で始まるシーンを 1 つ開く → 「走行の説明」の memo= のすぐ後ろに
`試し(理由)。`(memo= が無ければ終わりに ` memo=試し(理由)`)→「保存」)を、本物の app.js・index.html(node の vm・sim_r2.js)と
本物の jetformer/dataset.py・web/bench.py(作り物の家の data/raw に向けて関数を呼ぶ。サーバーは起動しない)でたどる。

  1. 走り終わりのメモ: 1-4・3-8 は「記録に書く」(本物の bench.api_bench_after を関数で呼ぶ = 全部のシーンに書く)。3-6 は「書かずに閉じる」
  2. 「シーン」のタブ: GET /api/scenes(dataset.list_scenes)・/api/scenes/<名前>(load_scene_meta + scene_timeseries)・/api/modes(modes.load)
     の答えを作り、画面でタブ → 絞り込みに記録の名前 → カードを開く(openScene)→ 説明欄を手順書のとおりに直す → 「保存」
  3. 画面が送った PATCH の体を、本物の dataset.update_scene(server.py の api_scene_update と同じ呼び方)で作り物の scene.json に書く
  4. このあと night_rehearsal.sh scene_memo 8.1 で ③④⑤ を流す

    jf-heavy --lock none --mem 1500M -- python -I -B scene_path.py
出力: out/scene_path.txt・out/ui_scene_path.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
FH = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal_r2/scene_memo/home'
RAW = FH / 'jetformer' / 'data' / 'raw'
sys.path.insert(0, str(JF))
from jetformer import config  # noqa: E402
config.RAW_DIR = RAW                       # 作り物の家の data/raw(本物の data/raw は読まない・書かない)
from jetformer import dataset, modes, drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

STATIC = JF / 'jetformer' / 'web' / 'static'
SIM = HERE / 'sim_r2.js'
OUT = HERE / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal_r2/ui'
LIVE = json.loads((OUT / 'live_status.json').read_text(encoding='utf-8'))
MAN = json.loads((FH.parent / 'manifest.json').read_text(encoding='utf-8'))
BY_RUN = {b['run_id']: b['log'] for b in MAN['books'] if b['run_id']}
MORNING = next(b['log'] for b in MAN['books'] if b['block'] is None)

lines = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    lines.append(s)
    print(s)


def scene_json(name):
    return json.loads((dataset.scene_path(name) / 'scene.json').read_text(encoding='utf-8'))


def after_memo(lg, reason, memo, battery='A', va='7.95'):
    r = bench.api_bench_after({'log': lg, 'battery': battery, 'V_before': '8.31', 'V_after': va, 'reason': reason, 'memo': memo})
    log('走り終わりのメモ「記録に書く」(本物の bench.api_bench_after):', lg, 'シーン', r['scenes'], '→', r['description'][-120:])


# 手順書の言う直し(画面の説明欄の文に、人が手で足す)
EDIT_JS = ("{ const e = document.querySelector('#m-desc'); const v = e.value; const i = v.indexOf('memo='); "
           "e.value = i >= 0 ? v.slice(0, i + 5) + %s + v.slice(i + 5) : v + ' memo=' + %s; e.value }")


def run_ui(targets):
    scenes = []
    for s in dataset.list_scenes():
        s.pop('path', None)
        scenes.append(s)
    routes = {'/api/scenes': scenes,
              '/api/modes': [dict(m, scenes=0, minutes=0.0) for m in modes.load()]}
    for lg, sc, why in targets:
        meta = dataset.load_scene_meta(sc)
        meta.pop('path', None)
        meta['series'] = dataset.scene_timeseries(sc)
        routes['/api/scenes/' + sc] = meta
    c = bench.load_candidates(None)
    c = dict(c, today='2026-11-03')
    steps = [['tab', 'drive'], ['tick'], ['tab', 'scenes'], ['el', '#scene-summary', 'text']]
    for lg, sc, why in targets:
        steps += [['set', '#scene-filter', lg], ['eval', 'renderScenes(); 1'], ['el', '#scene-grid', 'html'], ['el', '#scene-summary', 'text'],
                  ['aeval', 'openScene(%s)' % json.dumps(sc)], ['el', '#modal', 'hidden'], ['el', '#m-name', 'text'],
                  ['el', '#m-desc', 'value'], ['el', '#m-mode', 'value'], ['el', '#m-mode', 'html'], ['el', '#m-tags', 'value'],
                  ['eval', EDIT_JS % (json.dumps('試し(%s)。' % why), json.dumps('試し(%s)' % why))],
                  ['click', '#m-save'], ['patched'], ['toasts'], ['click', '#m-close'], ['el', '#modal', 'hidden']]
    steps += [['tab', 'drive'], ['start'], ['toasts'], ['tick'], ['tick']]
    inp = {'bench': c, 'server_params': dict(LIVE['params']), 'steps': steps, 'server_client_version': 9, 'confirm': True,
           'boot_at': LIVE.get('boot_at'), 'models': [c['baseline']['run']],
           'script_src': 'http://x/static/app.js?v=%d' % int((STATIC / 'app.js').stat().st_mtime), 'server_boot_check': True, 'routes': routes}
    TMP.mkdir(parents=True, exist_ok=True)
    pin = TMP / 'scene_path.json'
    pin.write_text(json.dumps(inp, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run(['node', str(SIM), str(STATIC / 'app.js'), str(STATIC / 'index.html'), str(pin)], capture_output=True, text=True,
                       timeout=300, env=dict(os.environ, TZ='Asia/Tokyo'))
    if r.returncode != 0:
        raise SystemExit(r.stderr[-3000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    (OUT / 'ui_scene_path.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return out


def main():
    b14, b36, b38 = BY_RUN['1-4'], BY_RUN['3-6'], BY_RUN['3-8']
    log('# 作り物の家', FH)
    log('## 1. 走り終わりのメモ(欄が出ていた本)')
    after_memo(b14, '240秒走った', '手 —')
    after_memo(b38, 'B止め', '')
    log('3-6(%s)は「書かずに閉じる」(欄の afterDone だけ。シーンは書かない)' % b36)
    n14 = bench.log_scenes(b14)
    n36 = bench.log_scenes(b36)
    n38 = bench.log_scenes(b38)
    nm = bench.log_scenes(MORNING)
    log('シーン: 1-4 %s / 3-6 %s / 3-8 %s / 朝の確かめ %s' % (n14, n36, n38, nm))
    targets = [(MORNING, nm[0], '朝の確かめ'), (b14, n14[len(n14) // 2], 'ゲインを間違えた'), (b36, n36[-1], '左 4.2 のまま'),
               (b38, n38[0], '天井を入れ忘れた')]
    for lg, sc, why in targets:
        log('  直す前の scene.json の説明(%s): %s' % (sc, scene_json(sc)['description']))
    log('## 2. 画面の「シーン」のタブ(本物の app.js)')
    out = run_ui(targets)
    pats = []
    for i, st in enumerate(out['steps']):
        op = st['op']
        if op == 'el' and st['sel'] in ('#scene-grid',):
            names = [x.split('"')[0] for x in st['value'].split('data-name="')[1:]]
            log('  [%d] 絞り込みのあとのカード %d 枚: %s' % (i, len(names), names))
        elif op == 'el' and st['sel'] in ('#scene-summary', '#m-name', '#m-desc', '#m-mode', '#m-tags', '#modal'):
            log('  [%d] %s %s = %s' % (i, st['sel'], st['prop'], json.dumps(st['value'], ensure_ascii=False)[:400]))
        elif op == 'eval' and isinstance(st['value'], str):
            log('  [%d] 手で直した説明: %s' % (i, st['value']))
        elif op == 'aeval':
            log('  [%d] openScene → %s' % (i, st['value']))
        elif op == 'patched':
            for nm_, body in st['value']:
                log('  [%d] 画面が送った PATCH /api/scenes/%s: %s' % (i, nm_, json.dumps(body, ensure_ascii=False)))
                pats.append((nm_, body))
        elif op == 'toasts':
            log('  [%d] トースト: %s' % (i, st['value'][-3:]))
        elif op == 'start':
            log('  [%d] シーンのタブから走行タブに戻ってすぐ ▶: status=%s confirms=%s' % (i, st['status'], [c[:80] for c in st['confirms']]))
    log('## 3. PATCH の体を本物の dataset.update_scene(server.py api_scene_update と同じ呼び方)で書く')
    for nm_, body in pats:
        r = dataset.update_scene(nm_, body.get('description'), body.get('tags'), body.get('mode_id'))
        log('  %s → mode_id %s・tags %s・説明 %s' % (nm_, r.get('mode_id'), r.get('tags'), r.get('description')))
    log('## 4. 夜の道具の読み方(本物の section_scorecard.effective_description → standing_3lap.trial_memo)')
    sys.path.insert(0, str(JF / 'tools'))
    import section_scorecard as ss  # noqa: E402
    import standing_3lap as s3  # noqa: E402
    for lg, sc, why in targets:
        base = json.loads((RAW / lg / 'log.json').read_text(encoding='utf-8'))['description']
        d, info = ss.effective_description(lg, base, RAW)
        log('  %s: trial_memo = %r(直したシーン %s・conflicts %s)' % (lg, s3.trial_memo(d), info['edited_scenes'], info['conflicts']))
    (OUT / 'scene_path.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
