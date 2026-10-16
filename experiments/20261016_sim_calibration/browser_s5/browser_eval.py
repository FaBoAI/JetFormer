#!/usr/bin/env python3
"""(sim_calibration_20261016 の写し、2026-10-16: --page で開くページ(index.html・calib.html・calib_cap.html)を選べるようにし、
結果をページごとのフォルダー・画面の名前にページの名前を入れた。ブリッジ 8831・geckodriver 4451・プロファイル sim_calibration_20261016/ff_profile)
(元: sim_diverse_train_20261015 の写し、2026-10-15: ブリッジ 8821・geckodriver 4450・プロファイル sim_diverse_train_20261015/ff_profile。
足したところ: (1) 1 つの段の連続試験が「試験停止」(アプリの 5 秒の打ち切りなど)で終わったら、止まった走行の JSON を stopped/ に移して、
同じセッションでページを開き直して残りをもう 1 回走らせる(--retries。protocol eval_Q1.sessions_and_order 1 の「もう 1 回」)、
(2) 画面を撮るフレームを複数にした(--shots。最初の走行の途中の窓・車載カメラ・ToF)。走らせ方・結果の受け取り方は 10/14 と同じ)
閉ループ評価: コピーしたシミュレーターのアプリ(index.html)を Firefox(DISPLAY=:0)で開き、
モデル ZIP を画面の「モデル ZIP を読み込む」で読み込んで、アプリ自身の「連続試験」(controller external)で走らせる

    # 1) ブリッジ(10/14 の専用の venv の onnxruntime、CPU。ポート 8821)
    jf-heavy --mem 1500M --lock none -- <10/14>/.venv-bridge/bin/python -I browser/run_bridge.py --port 8821 --no-browser
    # 2) 評価(Firefox・geckodriver はこのプロセスの子として jf-heavy の上限に入る。snap のスコープにも上限を付ける)
    jf-heavy --mem 2500M --lock big --need-mb 2048 -- python -I eval/browser_eval.py --jobs EXT:e0,A1-s0:e0,SDIV-s0:core,...

  - 計画は eval_common.phase_plan(段 e0 = E0 だけ、core = E0〜E5、ext = E6〜E11、flying = 連続周回)。
  - 1 モデルごとにページを開き直し、「接続を確認」→ ZIP をファイル入力に渡して「ZIP からモデルを読み込む」→ 計画を入れて
    「連続試験を開始」。終わったら「結果 JSON を保存」の中身(sim.report() + world + context + inference の全部)を、
    ダウンロードの代わりにページの中で受け取り、1 走行ずつ <out>/<label>/<name>.json に書く。
  - すでに全部の走行の JSON があるモデルは飛ばす(電源断のあとの続き)。0 バイト・壊れた JSON は作り直す。
  - 他の Firefox が動いていないこと・空きメモリ 2 GiB 以上を確かめてから始める(webgl_probe.wait_preflight)。
  - 実車・I2C には触れない。ユーザーのファイルは変更しない。
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'browser'))     # この作業の webgl_probe.py・browser_collect.py の写しだけを読む
from eval_common import BRIDGE_PORT, CL, EVAL, MODEL_BY_LABEL, phase_plan  # noqa: E402
from webgl_probe import GECKO, GECKO_PORT, PROFILE, WEBGL_INFO, WebDriver, mem_available_mb, other_firefox, save_canvas, wait_preflight  # noqa: E402
from browser_collect import XAUTH, cap_snap_scope, wait_text  # noqa: E402

ELEMENT_KEY = 'element-6066-11e4-a52e-4f735466cecf'
PAGE_STEM = 'index'

STATUS_JS = """const s=window.courseSimulator.snapshot();const $=id=>document.getElementById(id);
return {frame:s.state.frame,t:s.state.sim_time,gates:s.metrics.passed_gates,coll:s.metrics.collisions,terminal:s.state.terminal,
  reason:s.state.reason??null,batch:$('batch-status').textContent,status:$('status')?.textContent??null};"""

# 「結果 JSON を保存」(batch-save)は Blob の URL を a.download でダウンロードさせる。ダウンロードの代わりに中身をページに残す
CAPTURE_JS = """window.__evalSaved=null;
if(!window.__evalPatched){const orig=HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click=function(){if(this.download&&String(this.href).startsWith('blob:')){const name=this.download;
    fetch(this.href).then(r=>r.text()).then(t=>{window.__evalSaved={name,text:t};}).catch(e=>{window.__evalSaved={name,error:String(e)};});return;}
    return orig.call(this);};window.__evalPatched=true;}
document.getElementById('batch-save').click();return true;"""
SAVED_LEN_JS = "return window.__evalSaved?(window.__evalSaved.error?{error:window.__evalSaved.error}:{len:window.__evalSaved.text.length}):null;"
SAVED_CHUNK_JS = "return window.__evalSaved.text.slice(arguments[0],arguments[0]+arguments[1]);"


def find(d, css):
    v = d.s('POST', '/element', {'using': 'css selector', 'value': css})
    return v[ELEMENT_KEY]


def done_runs(out, plan):
    """すでに書いた走行(読める JSON で、batch_status が completed)"""
    ok = set()
    for c in plan:
        p = out / f"{c['name']}.json"
        try:
            r = json.loads(p.read_text(encoding='utf-8'))
            if r.get('batch_status') == 'completed' and r.get('name') == c['name']:
                ok.add(c['name'])
        except (OSError, ValueError):
            pass
    return ok


def read_saved(d):
    t0 = time.time()
    while time.time() - t0 < 120:
        v = d.js(SAVED_LEN_JS)
        if v:
            break
        time.sleep(0.5)
    else:
        raise TimeoutError('結果 JSON を受け取れません')
    if 'error' in v:
        raise RuntimeError('結果 JSON の受け取りに失敗: ' + v['error'])
    n, parts, step = v['len'], [], 4_000_000
    for i in range(0, n, step):
        parts.append(d.js(SAVED_CHUNK_JS, i, step))
    return json.loads(''.join(parts))


def run_job(d, a, m, phase, plan, out, bridge, screens, attempt):
    """1 つの段を 1 回走らせる。書いた走行の名前と、ジョブの記録を返す"""
    have = done_runs(out, plan)
    todo = [c for c in plan if c['name'] not in have]
    job = {'label': m['label'], 'phase': phase, 'attempt': attempt, 'planned': [c['name'] for c in plan], 'skipped_done': sorted(have),
           'ran': [c['name'] for c in todo]}
    if not todo:
        print(f"[eval] {m['label']} {phase}: 全部の走行がすでにあります", flush=True)
        return job
    (out / f'plan_{phase}.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    t_job = time.time()
    d.get(f'{bridge}/{a.page}')
    info = wait_text(d, WEBGL_INFO, lambda v: v.get('ready') or not v.get('fatal_hidden', True), 180, 1)
    job['webgl'] = {k: info.get(k) for k in ('webgl2', 'ready', 'inner', 'fatal_text')}
    if not info.get('ready'):
        raise RuntimeError('ページの準備ができません: ' + str(info.get('fatal_text')))
    if a.view:
        # 画面の視点(俯瞰・平面・追従)。表示だけで、車載カメラ・物理・推論には関係しない(画面を撮るための走行だけで使う)
        d.js("document.getElementById(arguments[0]).click();", a.view)
    d.js("document.getElementById('connect').click();")
    job['connect'] = wait_text(d, "return document.getElementById('connection-status').textContent;",
                               lambda v: '接続済み' in v, 60, 1)
    # 画面の「モデル ZIP を読み込む」: ファイル入力に ZIP を渡して「ZIP からモデルを読み込む」を押す
    d.s('POST', f"/element/{find(d, '#model-bundle-file')}/value", {'text': m['zip']})
    d.js("document.getElementById('upload-status').textContent='';document.getElementById('load-bundle').click();")
    job['load'] = wait_text(d, "return document.getElementById('upload-status').textContent;",
                            lambda v: '読み込み完了' in v or '読み込み失敗' in v, 300, 1)
    if '読み込み完了' not in job['load']:
        raise RuntimeError(f"{m['label']}: モデルを読み込めません: {job['load']}")
    job['model_summary'] = d.js("return document.getElementById('model-summary').textContent;")
    job['connection'] = d.js("return document.getElementById('connection-status').textContent;")
    print(f"[eval] {m['label']} {phase}(試み {attempt}): 読み込み完了 {job['model_summary']!r}", flush=True)
    d.js("""const $=id=>document.getElementById(id);$('batch-plan').value=arguments[0];$('batch-record').checked=false;
            document.querySelectorAll('details').forEach(x=>{if(x.contains($('batch-plan')))x.open=true;});""",
         json.dumps(todo, ensure_ascii=False))
    d.js("performance.clearResourceTimings();performance.setResourceTimingBufferSize(100000);")
    d.js("document.getElementById('batch-run').click();")
    t_batch, progress, shot_done = time.time(), [], set()
    shots = sorted(int(x) for x in a.shots.split(',') if x.strip())

    def note(v):
        v = dict(v, wall_s=round(time.time() - t_batch, 1), mem_available_mb=mem_available_mb())
        progress.append(v)
        if len(progress) % 6 == 1:
            print('[eval]', m['label'], json.dumps(v, ensure_ascii=False), flush=True)
        # 画面: 最初の走行の途中で、--shots のフレームを越えた最初の見回りごとに 1 組(窓・車載カメラ・ToF)
        idx = v['batch'].split(' / ')[0].strip()
        fr = v.get('frame') or 0
        if idx == '1' and not v.get('terminal'):
            for s in shots:
                if fr >= s and s not in shot_done and (s == shots[-1] or fr < [x for x in shots if x > s][0]):
                    shot_done.add(s)
                    stem = f"eval_{PAGE_STEM}_{todo[0]['name']}_f{fr:03d}"
                    try:
                        d.screenshot(screens / f'{stem}_window.png')
                        save_canvas(d, 'camera', screens / f'{stem}_camera.png')
                        save_canvas(d, 'ranges', screens / f'{stem}_tof.png')
                        job.setdefault('screens', []).append(stem)
                    except Exception as e:  # noqa: BLE001  画面の保存に失敗しても評価は続ける
                        job.setdefault('screens_error', []).append(f'{type(e).__name__}: {e}')
                    break

    final = wait_text(d, STATUS_JS, lambda v: '連続試験終了' in v['batch'] or '試験停止' in v['batch'],
                      a.run_timeout * len(todo), 3, note)
    job['batch_wall_s'] = round(time.time() - t_batch, 1)
    job['batch_final'] = final
    job['progress'] = progress
    d.screenshot(screens / f"eval_{PAGE_STEM}_{m['label']}_{phase}_a{attempt}_after_batch_window.png")
    d.js(CAPTURE_JS)
    results = read_saved(d)
    frames = 0
    stopped_dir = out / 'stopped'
    for r in results:
        p = out / f"{r['name']}.json"
        if r.get('batch_status') != 'completed':
            # 止まった走行(打ち切りなど)は数えない。記録として stopped/ に残し、次の試みで走らせ直す
            stopped_dir.mkdir(exist_ok=True)
            p = stopped_dir / f"{r['name']}.attempt{attempt}.json"
            job.setdefault('stopped', []).append({'name': r['name'], 'batch_status': r.get('batch_status'), 'frames': len(r.get('actions') or [])})
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
        tmp.replace(p)
        frames += len(r.get('actions') or [])
    job['written'] = [r['name'] for r in results]
    job['frames'] = frames
    job['fps'] = round(frames / max(job['batch_wall_s'], 1e-9), 2)
    job['job_wall_s'] = round(time.time() - t_job, 1)
    job['timing'] = d.js("""const e=performance.getEntriesByType('resource').filter(x=>x.name.includes('/api/predict'));
        const v=e.map(x=>x.duration).sort((a,b)=>a-b);const q=p=>v.length?v[Math.min(v.length-1,Math.floor(p*v.length))]:null;
        return {predict_posts:v.length,predict_ms_median:q(.5),predict_ms_p90:q(.9),predict_ms_max:v.length?v[v.length-1]:null};""")
    print(f"[eval] {m['label']} {phase}: {len(results)} 走行・{frames} フレーム・{job['fps']} フレーム/秒・{final['batch']}", flush=True)
    if '試験停止' in final['batch']:
        job['error'] = final['batch']
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='EXT:e0,A1-s0:e0,SDIV-s0:core,SDIV-s0:ext,SDIV-s0:flying',
                    help='「モデル:段」の並び(段は eval_common.phase_plan)。この順に走らせる')
    ap.add_argument('--only', default='', help='計画の中のこの名前(カンマ区切り)だけを走らせる(試し)')
    ap.add_argument('--port', type=int, default=BRIDGE_PORT)
    ap.add_argument('--page', default='index.html', choices=['index.html', 'calib.html', 'calib_cap.html'], help='開くページ(物理: 元・較正・較正 + 上限)')
    ap.add_argument('--display', default=':0')
    ap.add_argument('--out', default=str(CL))
    ap.add_argument('--screens', default=str(EVAL.parent / 'screens'))
    ap.add_argument('--firefox-mem', default='2500M')
    ap.add_argument('--shots', default='8,60,120,200', help='最初の走行の途中の画面を撮るフレーム(この値を越えた最初の見回りで撮る)')
    ap.add_argument('--window', default='', help='窓の大きさ 例 1400x900(画面を撮るための走行だけで使う。既定は変えない)')
    ap.add_argument('--view', default='', choices=['', 'overview', 'top', 'follow'], help='画面の視点(画面を撮るための走行だけで使う)')
    ap.add_argument('--retries', type=int, default=2, help='「試験停止」で終わった段を、同じセッションで走らせ直す回数')
    ap.add_argument('--run-timeout', type=float, default=900, help='1 走行あたりの待ち時間の上限 [秒]')
    a = ap.parse_args()
    global PAGE_STEM
    PAGE_STEM = a.page.replace('.html', '')
    out_root, screens = Path(a.out) / PAGE_STEM, Path(a.screens)
    out_root.mkdir(parents=True, exist_ok=True)
    screens.mkdir(parents=True, exist_ok=True)
    bridge = f'http://127.0.0.1:{a.port}'
    jobs = []
    for spec in a.jobs.split(','):
        label, phase = spec.split(':')
        m = MODEL_BY_LABEL[label]
        if not m.get('available'):
            print(f'[eval] {label}: 一式がありません(train/runs.json)。飛ばします', flush=True)
            continue
        plan = phase_plan(m, phase)
        if a.only:
            keep = set(a.only.split(','))
            plan = [c for c in plan if c['name'] in keep]
        if plan:
            jobs.append((m, phase, plan))
    PROFILE.mkdir(parents=True, exist_ok=True)   # 空のフォルダーでよい(geckodriver が user.js を書く)
    session = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'args': vars(a), 'preflight': wait_preflight(), 'jobs': []}
    log = open(out_root / f'geckodriver_{time.strftime("%H%M%S")}.log', 'w')
    env = {'DISPLAY': a.display, 'XAUTHORITY': XAUTH}
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = WebDriver(GECKO_PORT)
    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001  起動待ち
                time.sleep(0.5)
        caps = d.start(['-profile', str(PROFILE)], {}, env)
        session['memory_cap'] = cap_snap_scope(gd.pid, a.firefox_mem)
        if a.window:
            w, h = (int(x) for x in a.window.split('x'))
            session['window_rect'] = d.s('POST', '/window/rect', {'width': w, 'height': h})
        session['browser'] = {k: caps.get(k) for k in ('browserName', 'browserVersion', 'platformName', 'moz:headless')}
        for m, phase, plan in jobs:
            out = out_root / m['label']
            out.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, a.retries + 2):
                job = run_job(d, a, m, phase, plan, out, bridge, screens, attempt)
                session['jobs'].append(job)
                left = [c['name'] for c in plan if c['name'] not in done_runs(out, plan)]
                if not left:
                    break
                print(f"[eval] {m['label']} {phase}: 残り {left} を走らせ直します(試み {attempt + 1})", flush=True)
            # 1 つの段が済むたびにセッションの記録を書く(電源断に備える)
            (out_root / 'session_current.json').write_text(json.dumps(session, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    except Exception as e:  # noqa: BLE001  失敗も結果に残す
        session['error'] = f'{type(e).__name__}: {e}'
        print('[eval] エラー:', session['error'], flush=True)
    finally:
        d.quit()
        gd.terminate()
        try:
            gd.wait(20)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
        time.sleep(2)
        session['firefox_left'] = other_firefox()
        session['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        name = f"session_{time.strftime('%Y%m%d-%H%M%S')}.json"
        (out_root / name).write_text(json.dumps(session, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        try:
            (out_root / 'session_current.json').unlink()
        except OSError:
            pass
        print('[eval] 記録:', out_root / name, flush=True)


if __name__ == '__main__':
    main()
