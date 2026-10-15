#!/usr/bin/env python3
"""DAgger の記録(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.procedure 4): コピーしたシミュレーターを Firefox(DISPLAY=:0)で
開き、アプリ自身の「連続試験」と収集器で、差し込みモジュール dagger-teacher.js(フレーム k まで学習した JetFormer の操作を再生 → k から先生)で
20 秒のシーンを記録する。10/14 の browser/browser_collect.py の写しを作り替えたもの:
  - ページは dagger.html(dagger/install_dagger.py が置く)。乱しの k の取り直しは無い(1 シーン 1 回。だめなら 1 回だけ同じ計画で取り直す)
  - 連続試験のあと、ページの globalThis.__daggerLog(差し込みモジュールが開始ごとに残す記録)を読み、本当のゲートの外しを
    同じ gate_miss.mjs でページの中で数え、シーンごとの横の記録(sidecar)に書く
  - 合格の判定: 600 フレーム・衝突 0・ゲートの外し 0・全部の周が有効(33 ゲート・黒シートの横断 1 回・衝突 0)・データセットに保存・
    再生のフレーム数 = k
  - まとめ(ZIP)は合格したシーンだけを計画の順に並べて「1つのZIPにまとめる」(--collection-name)

    jf-heavy --mem 2500M --lock big --need-mb 2048 -- python -I dagger_collect.py --page dagger.html --plan <計画.json> \\
        --collection-name <名前> --out-dir <結果>
  ブリッジ(口 8822、calibration_only_policy.py)は run_dagger_record.sh が立てる。実車・I2C には触れない。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'browser'))       # 今日の browser/webgl_probe.py(写したもの)だけを読む
from webgl_probe import (GECKO, GECKO_PORT, PROFILE, REPORT, WEBGL_INFO, WebDriver,  # noqa: E402
                         mem_available_mb, other_firefox, save_canvas, wait_preflight)

XAUTH = '/run/user/1000/gdm/Xauthority'
BRIDGE = 'http://127.0.0.1:8822'   # 記録のブリッジ(評価の 8821 と分ける)

TIMING = r"""
const e=performance.getEntriesByType('resource').filter(x=>x.name.includes('/api/datasets/frame'));
const d=e.map(x=>x.duration).sort((a,b)=>a-b);const q=p=>d.length?d[Math.min(d.length-1,Math.floor(p*d.length))]:null;
performance.clearResourceTimings();
return {frame_posts:d.length,post_ms_median:q(.5),post_ms_p90:q(.9),post_ms_max:d.length?d[d.length-1]:null,
  probe:globalThis.__rlTeacherProbe?JSON.parse(JSON.stringify(globalThis.__rlTeacherProbe)):null};"""

# ページの中で、差し込みモジュールが残した開始ごとの記録をまとめ、本当のゲートの外しを数える(同じ gate_miss.mjs)
DAGGER_LOG = r"""
const done=arguments[arguments.length-1];
import(location.origin+'/dagger/gate_miss.mjs').then(gm=>{
  const out={};
  for(const [key,e] of Object.entries(globalThis.__daggerLog||{})){
    const log=e.log||[];
    const r=gm.realGateMisses(log.map(q=>q.s),log.map(q=>q.g));
    const atk=log.find(q=>q.f===e.k)||null;
    out[key]={key:e.key,name:e.name,k:e.k,created:e.created,frames:e.frames,replay_frames:e.replay_frames,gate_misses:r.misses,
      gate_stuck_max_m:r.stuck_max_m,gate_miss_frames:r.frames,min_dmin:log.length?Math.min(...log.map(q=>q.dmin)):null,
      last_frame:log.length?log[log.length-1].f:null,pose_at_k:atk?{x:atk.x,z:atk.z,yaw:atk.yaw,speed:atk.speed}:null,
      first:log.length?{x:log[0].x,z:log[0].z,yaw:log[0].yaw}:null};
  }
  done(out);
}).catch(err=>done({__error:String(err)}));"""


def cap_snap_scope(pid, mem):
    """snap の Firefox は起動時に自分の systemd スコープへ移るため、jf-heavy のメモリ上限の外に出る。そのスコープにも同じ上限を付ける"""
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


def accept(scene, side, cond):
    """1 シーンの合格の判定(ブラウザの結果と、ページで数えたゲートの外し)"""
    why = []
    if scene['frames'] != 600:
        why.append(f"フレーム {scene['frames']}")
    if scene['collisions']:
        why.append(f"衝突 {scene['collisions']}")
    if not scene['dataset_session_id']:
        why.append('データセットに保存されていない')
    if not side or side.get('gate_misses') is None:
        why.append('差し込みの記録が無い')
    else:
        if side['gate_misses']:
            why.append(f"ゲートの外し {side['gate_misses']}")
        if side.get('replay_frames') != min(side.get('k') or 0, 600):
            why.append(f"再生のフレーム {side.get('replay_frames')} ≠ k {side.get('k')}")
    laps = scene['lap_detail']
    if not laps:
        why.append('有効な周が無い')
    for i, l in enumerate(laps):
        if l['gates'] != 33 or l['crossings'] != 1 or l['collisions']:
            why.append(f'{i + 1} 周目が有効でない')
    return why


def skey(c):
    return f"{c['start-gate']}|{float(c['spawn-offset']):.4f}|{float(c['spawn-yaw']):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default='dagger.html')
    ap.add_argument('--plan', required=True, help='連続試験の JSON 配列(dagger/make_dagger_plan.py)')
    ap.add_argument('--max-retries', type=int, default=1)
    ap.add_argument('--max-scenes', type=int, default=0)
    ap.add_argument('--display', default=':0')
    ap.add_argument('--collection-name', default='')
    ap.add_argument('--scene-timeout', type=float, default=600)
    ap.add_argument('--firefox-mem', default='2500M')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--screens-dir', default='')
    ap.add_argument('--screens-every-s', type=float, default=900)
    ap.add_argument('--screens-max', type=int, default=6)
    ap.add_argument('--progress-file', default='')
    ap.add_argument('--chunk', type=int, default=0, help='連続試験を n シーンずつに分ける(途中で止まっても失うのは 1 回分だけ。0 = 分けない)')
    ap.add_argument('--resume', action='store_true',
                    help='<out-dir>/sidecars の結果を読み、合格済みのシーンは飛ばし、試した回数の続きから取り直す(記録の段で追加)')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sidecars').mkdir(exist_ok=True)
    screens = Path(a.screens_dir) if a.screens_dir else None
    if screens:
        screens.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
    if a.max_scenes:
        plan = plan[:a.max_scenes]
    shots, progress = [], []
    attempts = {c['name']: [] for c in plan}
    accepted = {}
    result = {'page': a.page, 'plan': a.plan, 'scenes_planned': len(plan), 'display': a.display, 'preflight': wait_preflight(),
              'chunk': a.chunk, 'batches': []}
    log = open(out / 'geckodriver.log', 'w')
    env = {'DISPLAY': a.display, 'XAUTHORITY': XAUTH} if a.display else {}
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = WebDriver(GECKO_PORT)
    t_batch = time.time()

    def shoot(v, label):
        try:
            stem = f'{len(shots):02d}_{label}'
            d.screenshot(screens / f'{stem}_window.png')
            save_canvas(d, 'camera', screens / f'{stem}_camera.png')
            save_canvas(d, 'ranges', screens / f'{stem}_tof.png')
            shots.append({'stem': stem, 'wall_s': v.get('wall_s'), 'frame': v.get('frame'), 'batch': v.get('batch')})
        except Exception as e:  # noqa: BLE001
            shots.append({'error': f'{type(e).__name__}: {e}'})

    def note(v):
        v = dict(v, wall_s=round(time.time() - t_batch, 1), mem_available_mb=mem_available_mb())
        progress.append(v)
        print('[collect]', json.dumps(v, ensure_ascii=False), flush=True)
        if a.progress_file:
            with open(a.progress_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(v, ensure_ascii=False) + '\n')
        if screens and len(shots) < a.screens_max:
            first_scene = not shots and 150 <= (v.get('frame') or 0) <= 450
            later = shots and v['wall_s'] - (shots[-1].get('wall_s') or 0) >= a.screens_every_s and 60 <= (v.get('frame') or 0) <= 540
            if first_scene or later:
                shoot(v, f"t{int(v['wall_s']):05d}_f{v.get('frame')}")

    def run_batch(sub_plan, override, tag):
        """連続試験を 1 回。override: {種: {k, scale}}(取り直し)。戻り値 シーンの結果と乱しの記録"""
        # k の上書きはページの側の関数に JSON の文字列で渡す(WebDriver のスクリプトが作ったオブジェクトはページのコードから読めないことがある)。
        # 連続試験の表示(batch-status)は前の回の「連続試験終了」が残るので、押す前に消す(準備の段の試しで、2 回目の待ちがすぐ終わった)
        d.js("""document.getElementById('batch-plan').value=arguments[0];document.getElementById('batch-record').checked=true;
                document.getElementById('collection-ids').value='';document.getElementById('batch-results').value='';
                document.getElementById('batch-status').textContent='';return true;""", json.dumps(sub_plan, ensure_ascii=False))
        d.js("performance.clearResourceTimings();performance.setResourceTimingBufferSize(100000);")
        t0 = time.time()
        d.js("document.getElementById('batch-run').click();")
        status_js = """const s=window.courseSimulator.snapshot();return {frame:s.state.frame,laps:s.metrics.completed_laps??null,
            coll:s.metrics.collisions,batch:document.getElementById('batch-status').textContent,
            record:document.getElementById('record-status').textContent};"""
        final = wait_text(d, status_js, lambda v: '連続試験終了' in v['batch'] or '試験停止' in v['batch'],
                          a.scene_timeout * len(sub_plan), 10, note)
        batch = json.loads(d.js("return document.getElementById('batch-results').value;") or '[]')
        side = d.js_async(DAGGER_LOG)
        if isinstance(side, dict) and side.get('__error'):
            raise RuntimeError('差し込みの記録を読めません: ' + side['__error'])
        scenes = []
        for r in batch:
            scenes.append({'name': r['name'], 'seed': r.get('config', {}).get('seed'), 'status': r['batch_status'], 'reason': r['outcome']['reason'],
                           'frames': r['frames'], 'collisions': r['metrics']['collisions'],
                           'laps': [round(x['elapsed'], 4) for x in r['metrics'].get('lap_events', [])],
                           'lap_detail': [{'elapsed': x['elapsed'], 'gates': x['passed_gates'], 'crossings': len(x['required_crossing_events']),
                                           'collisions': x['collisions']} for x in r['metrics'].get('lap_events', [])],
                           'dataset_session_id': (r.get('dataset') or {}).get('session_id'), 'error': r.get('error')})
        return {'tag': tag, 'final': final, 'wall_s': round(time.time() - t0, 1), 'scenes': scenes, 'side': side, 'timing': d.js(TIMING)}

    def process_batch(part, override, attempt, b):
        """1 回の連続試験の結果を判定し、シーンごとの sidecar を書く(途中で止まっても、それまでの結果が残る)"""
        result['batches'].append({k: v for k, v in b.items() if k != 'side'} | {'override': override})
        byname = {s['name']: s for s in b['scenes']}
        for c in part:
            s = byname.get(c['name'])
            side = (b['side'] or {}).get(skey(c))
            why = accept(s, side, c) if s else ['結果が無い']
            rec = {'attempt': attempt, 'override': None, 'scene': s, 'why': why}
            attempts[c['name']].append(rec)
            sc = {'name': c['name'], 'seed': c['seed'], 'split': c['dataset-split'], 'scenario': c['scenario'], 'attempt': attempt,
                  'override': None, 'start_key': skey(c),
                  'accepted': not why, 'why': why, 'session_id': (s or {}).get('dataset_session_id'), 'browser': s, 'dagger': side,
                  'page': a.page, 'made': time.strftime('%Y-%m-%dT%H:%M:%S')}
            (out / 'sidecars' / f"{c['name']}.attempt{attempt}.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1))
            if not why:
                accepted[c['name']] = sc
                (out / 'sidecars' / f"{c['name']}.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1))

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
        if not (info.get('probe') or {}).get('module') == 'dagger-teacher.js':
            raise RuntimeError('差し込みモジュール dagger-teacher.js が読み込まれていません: ' + json.dumps(info.get('probe')))
        d.js("document.getElementById('connect').click();")
        result['connection_status'] = wait_text(d, "return document.getElementById('connection-status').textContent;",
                                                lambda v: '接続済み' in v, 30, 1)
        d.js("document.getElementById('collection-plan').click();")     # ジャンル 8・自動ラベル
        if a.collection_name:
            d.js("document.getElementById('collection-name').value=arguments[0];", a.collection_name)
        result['ui_before_run'] = d.js("""const $=id=>document.getElementById(id);return {dataset_mode:$('dataset-mode').value,
            automated_labels:$('automated-labels').checked,batch_record:$('batch-record').checked,teacher_profile:$('teacher-profile').value};""")
        # 1 回目: 計画のすべて(k は差し込みモジュールの表 = Node の予想)
        result['resumed'] = {}
        if a.resume:
            # 前の実行の結果(シーンごとの sidecar)を読み直す。合格済みは飛ばし、だめだった回の数から続ける
            for c in plan:
                for f in sorted((out / 'sidecars').glob(f"{c['name']}.attempt*.json"), key=lambda q: int(q.stem.rsplit('attempt', 1)[1])):
                    sc = json.loads(f.read_text())
                    attempts[c['name']].append({'attempt': sc['attempt'], 'override': None, 'scene': sc.get('browser'), 'why': sc['why'],
                                                'resumed_from': f.name})
                    if sc.get('accepted') and sc.get('session_id'):
                        accepted[c['name']] = sc
            result['resumed'] = {'accepted': [n for n in accepted], 'attempts': {n: len(v) for n, v in attempts.items() if v}}
            prev = out / 'result.json'
            if prev.exists():
                prev.rename(out / f"result.before_resume_{time.strftime('%H%M%S')}.json")
        for attempt in range(a.max_retries + 1):
            # この回で試すシーン = まだ合格していない、かつ試した回数がちょうど attempt のもの(取り直しは同じ計画のまま)
            todo = [c for c in plan if c['name'] not in accepted and len(attempts[c['name']]) == attempt]
            if not todo:
                continue
            n = a.chunk if a.chunk > 0 else len(todo)
            for i0 in range(0, len(todo), n):
                part = todo[i0:i0 + n]
                process_batch(part, {}, attempt, run_batch(part, {}, f'attempt{attempt}_part{i0 // n}'))

        result['attempts'] = attempts
        result['accepted'] = [c['name'] for c in plan if c['name'] in accepted]
        result['rejected'] = [c['name'] for c in plan if c['name'] not in accepted]
        result['retried'] = {n: len(v) - 1 for n, v in attempts.items() if len(v) > 1}
        ids = [accepted[c['name']]['session_id'] for c in plan if c['name'] in accepted]
        result['session_ids_for_collection'] = ids
        d.screenshot(out / 'after_batch.png')
        save_canvas(d, 'camera', out / 'camera_rgb_after.png')
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
        result['screens'] = shots
        d.quit()
        gd.terminate()
        try:
            gd.wait(20)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
        time.sleep(2)
        result['firefox_left'] = other_firefox()
        result['wall_s'] = round(time.time() - t_batch, 1)
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps({k: result.get(k) for k in ('error', 'accepted', 'rejected', 'retried', 'collection_status', 'firefox_left', 'wall_s')},
                         ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
