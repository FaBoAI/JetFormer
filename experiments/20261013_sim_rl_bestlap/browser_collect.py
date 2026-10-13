#!/usr/bin/env python3
"""コピーしたシミュレーターを Firefox(DISPLAY=:0、WebGL は Mesa llvmpipe)で開き、アプリ自身の「連続試験」と収集器で
データセットを作る(2026-10-13、強化学習のベストラップ)

    # 1) ブリッジ(コピーの dist を配信。PWM 換算は元の20分データと同じ 1442/1498 µs)
    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I run_bridge.py --port 8791 --no-browser --policy calibration_only_policy.py &
    # 2) 差し替え教師のページを用意(元のファイルは変えない)
    python -I install_wrapper.py rl-teacher-selftest.js
    # 3) 収集(重い処理なので jf-heavy で包む。Firefox・geckodriver はこのプロセスの子として同じ上限に入る)
    jf-heavy --mem 2500M --lock big -- python -I browser_collect.py --plan plan.json --tag selftest --collection-name rl-selftest

  - 計画(--plan)はアプリの「連続試験」の JSON 配列(qa/rc-collection-plan-60.json と同じ形)。
    省略すると自己テストの 2 条件(train 1・val 1、開始ずれ ±0.1 m・±6° 前後、速度×0.9 の差し替え教師)。
  - 「20分・60シーンの収集条件を用意」を押してジャンル 8・自動ラベルの設定にしてから、計画を差し替えて「連続試験を開始」を押す。
  - 終わったら、この実行で作ったシーンだけで「1つのZIPにまとめる」(/api/datasets/collection)を行う(--collection-name)。
  - 他の Firefox が動いていないこと・空きメモリ 2 GiB 以上を確かめてから始める(webgl_probe.py と同じ)。
  - snap の Firefox は jf-heavy のメモリ上限の外(自分の systemd スコープ)に移るので、起動後にそのスコープにも上限を付ける。
  - 結果: /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/webgl_probe/<tag>/result.json(1 シーンごとの時間、
    フレーム送信の往復時間、PNG 化の時間、収集 ZIP の情報)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # 同じフォルダーの webgl_probe.py(自分で書いたもの)だけを読む
from webgl_probe import (BRIDGE, GECKO, GECKO_PORT, PROFILE, REPORT, WEBGL_INFO, WebDriver,  # noqa: E402
                         mem_available_mb, other_firefox, save_canvas, wait_preflight)

XAUTH = '/run/user/1000/gdm/Xauthority'

TIMING = r"""
const e=performance.getEntriesByType('resource').filter(x=>x.name.includes('/api/datasets/frame'));
const d=e.map(x=>x.duration).sort((a,b)=>a-b);const q=p=>d.length?d[Math.min(d.length-1,Math.floor(p*d.length))]:null;
const c=document.getElementById('camera');let t0=performance.now();for(let i=0;i<20;i++)c.toDataURL('image/png');const png=(performance.now()-t0)/20;
performance.clearResourceTimings();
return {frame_posts:d.length,post_ms_median:q(.5),post_ms_p90:q(.9),post_ms_max:d.length?d[d.length-1]:null,png_encode_ms:png,
  probe:globalThis.__rlTeacherProbe?JSON.parse(JSON.stringify(globalThis.__rlTeacherProbe)):null};"""


def cap_snap_scope(pid, mem):
    """snap の Firefox は起動時に自分の systemd スコープ(snap.firefox.geckodriver-*.scope)へ移るため、jf-heavy の
    メモリ上限の外に出る(2026-10-13 に確認)。そのスコープにも同じ上限を付ける(超えたら Firefox だけが止まる)"""
    try:
        scope = Path(f'/proc/{pid}/cgroup').read_text().strip().splitlines()[-1].rsplit('/', 1)[-1]
    except OSError as e:
        return {'error': str(e)}
    if not scope.startswith('snap.'):
        return {'scope': scope, 'note': 'snap のスコープではない(jf-heavy の上限の中)'}
    r = subprocess.run(['systemctl', '--user', 'set-property', '--runtime', scope, f'MemoryMax={mem}', 'MemorySwapMax=0'],
                       capture_output=True, text=True)
    shown = subprocess.run(['systemctl', '--user', 'show', scope, '-p', 'MemoryMax', '-p', 'MemoryCurrent'],
                           capture_output=True, text=True).stdout.split()
    return {'scope': scope, 'returncode': r.returncode, 'stderr': r.stderr.strip(), 'show': shown}


def default_plan(page_js):
    """自己テストの 2 条件(元の計画の 0 番・49 番を元に、開始ずれだけを広げる)"""
    plan = json.loads(page_js)
    a, b = dict(plan[0]), dict(plan[49])
    a.update(name='selftest-train-00', seed=2026101300, **{'spawn-offset': 0.10, 'spawn-yaw': 6.0, 'dataset-split': 'train'})
    b.update(name='selftest-val-01', seed=2026101301, **{'spawn-offset': -0.08, 'spawn-yaw': -5.0, 'dataset-split': 'val'})
    return [a, b]


def wait_text(d, script, done, timeout, every=5.0, log=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = d.js(script)
        if log:
            log(v)
        if done(v):
            return v
        time.sleep(every)
    raise TimeoutError(f'待ち時間切れ: {script[:60]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default='rl.html')
    ap.add_argument('--tag', default='selftest')
    ap.add_argument('--plan', help='連続試験の JSON 配列(省略で自己テストの 2 条件)')
    ap.add_argument('--display', default=':0')
    ap.add_argument('--collection-name', default='', help='この実行のシーンを1つのZIPにまとめるときの名前')
    ap.add_argument('--scene-timeout', type=float, default=600, help='1 シーンあたりの待ち時間の上限 [秒]')
    ap.add_argument('--firefox-mem', default='2500M', help='snap の Firefox のスコープに付けるメモリ上限')
    a = ap.parse_args()
    out = REPORT / 'webgl_probe' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    result = {'tag': a.tag, 'page': a.page, 'display': a.display, 'preflight': wait_preflight()}
    log = open(out / 'geckodriver.log', 'w')
    env = {'DISPLAY': a.display, 'XAUTHORITY': XAUTH} if a.display else {}
    import os
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = WebDriver(GECKO_PORT)
    progress = []

    def note(v):
        v = dict(v, wall_s=round(time.time() - t_batch, 1), mem_available_mb=mem_available_mb())
        progress.append(v)
        print('[collect]', json.dumps(v, ensure_ascii=False), flush=True)

    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001  起動待ち
                time.sleep(0.5)
        args = ['-profile', str(PROFILE)] + ([] if a.display else ['-headless'])
        caps = d.start(args, {}, {'MOZ_HEADLESS_WIDTH': '1400', 'MOZ_HEADLESS_HEIGHT': '900', **env})
        result['memory_cap'] = cap_snap_scope(gd.pid, a.firefox_mem)
        result['browser'] = {k: caps.get(k) for k in ('browserName', 'browserVersion', 'platformName', 'moz:headless')}
        d.get(f'{BRIDGE}/{a.page}')
        info = wait_text(d, WEBGL_INFO, lambda v: v.get('ready') or not v.get('fatal_hidden', True), 120, 1)
        result['webgl'] = {k: info.get(k) for k in ('webgl2', 'fatal_text', 'ready', 'probe', 'inner')}
        if not info.get('ready'):
            raise RuntimeError('ページの準備ができません: ' + str(info.get('fatal_text')))
        # ブリッジの PWM 換算(1442/1498 µs)を読み込む
        d.js("document.getElementById('connect').click();")
        result['connection_status'] = wait_text(d, "return document.getElementById('connection-status').textContent;",
                                                lambda v: '接続済み' in v, 30, 1)
        # ジャンル8・自動ラベルの設定(「20分・60シーンの収集条件を用意」)→ 計画を差し替える
        d.js("document.getElementById('collection-plan').click();")
        page_plan = d.js("return document.getElementById('batch-plan').value;")
        plan = json.loads(Path(a.plan).read_text(encoding='utf-8')) if a.plan else default_plan(page_plan)
        result['plan'] = plan
        d.js("""document.getElementById('batch-plan').value=arguments[0];document.getElementById('batch-record').checked=true;
                document.getElementById('collection-ids').value='';""", json.dumps(plan, ensure_ascii=False))
        result['ui_before_run'] = d.js("""const $=id=>document.getElementById(id);return {dataset_mode:$('dataset-mode').value,
            automated_labels:$('automated-labels').checked,batch_record:$('batch-record').checked};""")
        d.js("performance.clearResourceTimings();performance.setResourceTimingBufferSize(100000);")
        t_batch = time.time()
        d.js("document.getElementById('batch-run').click();")
        status_js = """const s=window.courseSimulator.snapshot();return {frame:s.state.frame,laps:s.metrics.completed_laps??null,
            coll:s.metrics.collisions,batch:document.getElementById('batch-status').textContent,
            record:document.getElementById('record-status').textContent};"""
        final = wait_text(d, status_js, lambda v: '連続試験終了' in v['batch'] or '試験停止' in v['batch'] or v['batch'].startswith('条件'),
                          a.scene_timeout * len(plan), 10, note)
        result['batch_wall_s'] = round(time.time() - t_batch, 1)
        result['batch_final'] = final
        result['timing'] = d.js(TIMING)
        batch = json.loads(d.js("return document.getElementById('batch-results').value;") or '[]')
        result['batch_results'] = batch
        result['scenes'] = [{'name': r['name'], 'status': r['batch_status'], 'reason': r['outcome']['reason'], 'frames': r['frames'],
                             'collisions': r['metrics']['collisions'],
                             'laps': [round(x['elapsed'], 4) for x in r['metrics'].get('lap_events', [])],
                             'dataset_session_id': (r.get('dataset') or {}).get('session_id'), 'error': r.get('error')} for r in batch]
        d.screenshot(out / 'after_batch.png')
        save_canvas(d, 'camera', out / 'camera_rgb_after.png')
        ids = [s['dataset_session_id'] for s in result['scenes'] if s['dataset_session_id'] and s['frames'] == 600 and s['collisions'] == 0]
        if a.collection_name and ids:
            t_col = time.time()
            d.js("""const $=id=>document.getElementById(id);$('collection-ids').value=arguments[0];$('collection-name').value=arguments[1];
                    $('collection-status').textContent='';$('collection-build').click();""", '\n'.join(ids), a.collection_name)
            result['collection_status'] = wait_text(
                d, "return document.getElementById('collection-status').textContent;",
                lambda v: 'シーン /' in v or '集約できません' in v, 1800, 5)
            result['collection_wall_s'] = round(time.time() - t_col, 1)
            result['collection'] = json.loads(d.js("try{return localStorage.getItem('course-20s-collection');}catch(e){return null;}") or 'null')
    except Exception as e:  # noqa: BLE001  失敗も結果に残す
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        result['progress'] = progress
        d.quit()
        gd.terminate()
        try:
            gd.wait(20)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
        time.sleep(2)
        result['firefox_left'] = other_firefox()
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps({k: result.get(k) for k in ('error', 'batch_wall_s', 'scenes', 'timing', 'collection_status', 'firefox_left')},
                         ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
