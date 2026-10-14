"""記録の段: 強化学習モードの見本(モードごとに 5 分 = 15 シーン)を、ユーザーと同じ道で記録する(2026-10-14、sim_rl_modes)

ユーザーが Mac から使うときと同じ非ループバックのアドレス http://192.168.55.1:8080 を DISPLAY :0 の snap Firefox で開き、
JetFormer の「シミュレーター」タブ → カード「強化学習モード」→ モード × 分を入れて「プランを作って開く」→ 新しいタブで「開始」
だけで記録する(統合の段の integrate/browser_check.py の plan_check と同じ操作。WebDriver などはそのファイルから読む)。
ページの実行係がシーンを 1 つずつ記録し、外れたシーン(衝突など)は替えのシーンで記録し直し、モードごとに 1 つのデータセットに取り込む。

  jf-heavy --mem 2500M --lock big --need-mb 1800 -- python -u record/record_plan.py --minutes fastest=5,no_wall=5,recover=5,speed=5 --keep-sources
  (途中で止まったら)  ... record/record_plan.py --resume <プランの ID>

--keep-sources: カードの「取り込んだら、元のシーンを消す」を外す(記録の確かめで決まりを満たさないシーンを外してデータセットを作り直せるように、
元の ZIP を残す。確かめが終わったら record/finish_sources.py で消す)。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('browser_check', HERE.parent / 'integrate' / 'browser_check.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

REPORTS = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
OUT = REPORTS / 'record'
SCREENS = REPORTS / 'screens'
PROFILE = OUT / 'ff_profile'


def log(*a):
    print(time.strftime('%H:%M:%S'), '[record]', *a, flush=True)


def plan_done(p):
    imps = p.get('imports') or {}
    return p['state'] in ('finished', 'stopped', 'error') and all((imps.get(k) or {}).get('state') in ('done', 'error') for k in p['modes'])


def left_scenes(p):
    fin = {'done', 'failed', 'skipped'}
    return [sid for sid in p['order'] if (p['progress'].get(sid) or {}).get('status') not in fin]


def open_card_and_create(d, minutes, keep_sources, res):
    d.get(f'{bc.BASE}/')
    d.js("window.jetformer.showTab('sim');")
    bc.wait(d, "return document.querySelectorAll('#sim-rl-modes .sim-rl-mode').length;", lambda v: v and v >= 4, 60, 1)
    time.sleep(1.5)
    d.js("document.getElementById('sim-rl-card').scrollIntoView();")
    d.screenshot(SCREENS / 'record_tab_card_before.png')
    res['card'] = d.js("""return [...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].map(e=>({name:e.querySelector('b')?.textContent,
        badge:e.querySelector('.badge')?.textContent, headline:e.querySelector('.headline')?.textContent, nums:e.querySelector('.nums')?.textContent}));""")
    d.js("""const want=arguments[0];for(const inp of document.querySelectorAll('[data-sim-rl-min]')){inp.value=String(want[inp.dataset.simRlMin]||0);
        inp.dispatchEvent(new Event('input',{bubbles:true}));}""", minutes)
    if keep_sources:
        d.js("const c=document.getElementById('sim-rl-delete');c.checked=false;c.dispatchEvent(new Event('change',{bubbles:true}));")
    res['form'] = d.js("""return {minutes:Object.fromEntries([...document.querySelectorAll('[data-sim-rl-min]')].map(i=>[i.dataset.simRlMin,i.value])),
        scenes:Object.fromEntries([...document.querySelectorAll('[data-sim-rl-scenes]')].map(i=>[i.dataset.simRlScenes,i.textContent])),
        val:document.getElementById('sim-rl-val').value, seed:document.getElementById('sim-rl-seed').value, delete:document.getElementById('sim-rl-delete').checked};""")
    log('フォーム', json.dumps(res['form'], ensure_ascii=False))
    d.js("document.getElementById('sim-rl-plan-form').scrollIntoView({block:'start'});")
    time.sleep(0.5)
    d.screenshot(SCREENS / 'record_tab_plan_form.png')
    before = set(d.handles())
    d.js("document.getElementById('btn-sim-rl-plan').scrollIntoView({block:'center'});")
    d.click('#btn-sim-rl-plan')
    t0 = time.time()
    new = []
    while time.time() - t0 < 30 and not new:
        time.sleep(0.5)
        new = [h for h in d.handles() if h not in before]
    if not new:
        raise RuntimeError('プランのタブが開きません: ' + str(d.js("return document.getElementById('toast')?.textContent")))
    d.switch(new[0])
    return list(before)[0]


def wait_plan_page(d):
    v = bc.wait(d, "return {url:location.href, panel:document.querySelector('#jf-rl-plan')?.textContent||null, "
                   "ready:!!(window.courseSimulator&&window.courseSimulator.sensorSummary().ready), secure:window.isSecureContext, origin:location.origin};",
                lambda v: v and v.get('ready') and v.get('panel') and '読み込み中' not in v['panel'], 180, 1)
    return v


def press_start(d):
    d.js("document.getElementById('jf-rl-plan').scrollIntoView({block:'center'});")
    d.click('#jf-rl-plan .jf-rl-btn.primary')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', default='fastest=5,no_wall=5,recover=5,speed=5')
    ap.add_argument('--keep-sources', action='store_true')
    ap.add_argument('--resume', default=None, help='このプランの ID のページを開いて「開始」を押す(作り直さない)')
    ap.add_argument('--max-hours', type=float, default=4.5)
    ap.add_argument('--max-restarts', type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SCREENS.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    res = {'base': bc.BASE, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'args': vars(a), 'restarts': [], 'polls': []}
    t0 = time.time()
    while bc.other_firefox() or bc.mem_available_mb() < 1800:
        if time.time() - t0 > 3600:
            sys.exit('待ち時間切れ(ほかの Firefox かメモリ)')
        log(f'待機: 他の firefox {len(bc.other_firefox())} / 空き {bc.mem_available_mb()} MB')
        time.sleep(30)
    glog = open(OUT / 'geckodriver.log', 'a')
    env = {'DISPLAY': ':0', 'XAUTHORITY': bc.XAUTH}
    gd = subprocess.Popen([bc.GECKO, '--port', str(bc.GECKO_PORT)], stdout=glog, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = bc.WebDriver(bc.GECKO_PORT)
    pid = a.resume
    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        caps = d.start(['-profile', str(PROFILE)], {}, env)
        res['browser'] = {k: caps.get(k) for k in ('browserName', 'browserVersion', 'platformName')}
        res['memory_cap'] = bc.cap_snap_scope(gd.pid, '2500M')
        d.s('POST', '/window/rect', {'width': 1500, 'height': 1000})
        if pid is None:
            minutes = {k: int(v) for k, v in (x.split('=') for x in a.minutes.split(',') if x)}
            res['main_handle'] = open_card_and_create(d, minutes, a.keep_sources, res)
        else:
            d.get(f'{bc.BASE}/sim/c/default/?rlplan={pid}')
        v = wait_plan_page(d)
        pid = v['url'].split('rlplan=', 1)[1].split('&')[0]
        res['plan_id'] = pid
        res['page'] = v
        log('プラン', pid, 'secure', v.get('secure'), v.get('origin'))
        (OUT / 'plan_id.txt').write_text(pid + '\n')
        time.sleep(2)
        d.screenshot(SCREENS / 'record_plan_before_start.png')
        press_start(d)
        res['pressed_start_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        t0 = time.time()
        last_shot = 0
        restarts = 0
        while time.time() - t0 < a.max_hours * 3600:
            time.sleep(20)
            try:
                p = bc.api('GET', f'/api/sim/rl/plans/{pid}')
            except Exception as e:  # noqa: BLE001
                log('API の読み取りに失敗', e)
                continue
            prog = p['progress']
            done = sum(1 for x in prog.values() if x.get('status') == 'done')
            failed = sum(1 for x in prog.values() if x.get('status') == 'failed')
            cur = next((sid for sid in p['order'] if (prog.get(sid) or {}).get('status') == 'recording'), None)
            imps = {k: (v or {}).get('state') for k, v in (p.get('imports') or {}).items()}
            log(f"状態 {p['state']} 済み {done} 外れ {failed} / {len(p['order'])} 記録中 {cur} 取り込み {imps} {p.get('message') or ''}")
            res['polls'].append({'t': round(time.time() - t0), 'state': p['state'], 'done': done, 'failed': failed, 'order': len(p['order']), 'imports': imps})
            if time.time() - last_shot > 900 and cur:
                try:
                    d.screenshot(SCREENS / 'record_plan_recording.png')
                except Exception as e:  # noqa: BLE001
                    log('画面の保存に失敗', e)
                last_shot = time.time()
            if plan_done(p) and not left_scenes(p):
                break
            if p['state'] in ('error', 'stopped') and left_scenes(p):
                # 実行係が止まった(2 シーン続けて記録できない・ブリッジの換算が変わったなど)。理由を残して、数回まで開き直して続ける
                restarts += 1
                res['restarts'].append({'at': time.strftime('%H:%M:%S'), 'state': p['state'], 'message': p.get('message'), 'left': len(left_scenes(p))})
                log('実行係が止まりました:', p.get('message'), '→ 開き直し', restarts)
                d.screenshot(SCREENS / f'record_plan_stopped_{restarts}.png')
                if restarts > a.max_restarts:
                    break
                d.get(f'{bc.BASE}/sim/c/default/?rlplan={pid}')
                wait_plan_page(d)
                time.sleep(3)
                press_start(d)
        res['plan'] = {k: p.get(k) for k in ('state', 'message', 'progress', 'imports', 'calibration', 'order', 'minutes', 'seed', 'val_ratio',
                                              'delete_sources')}
        res['plan']['modes'] = {k: {kk: v.get(kk) for kk in ('dataset_name', 'scenes', 'train', 'val', 'spares_used', 'policy_sha256', 'state')}
                                for k, v in p['modes'].items()}
        try:
            d.js("window.scrollTo(0,0);")
            d.screenshot(SCREENS / 'record_plan_after.png')
            d.js("document.getElementById('jf-rl-plan').scrollIntoView({block:'start'});")
            time.sleep(0.5)
            d.screenshot(SCREENS / 'record_plan_panel_after.png')
            res['plan_log'] = d.js("return document.querySelector('#jf-rl-plan pre, #jf-rl-plan .jf-rl-plan-log')?.textContent||null;")
        except Exception as e:  # noqa: BLE001
            res['screenshot_error'] = str(e)
        if res.get('main_handle'):
            try:
                d.switch(res['main_handle'])
                d.js("window.jetformer.loadSim();")
                time.sleep(3)
                d.js("document.getElementById('sim-rl-card').scrollIntoView();")
                d.screenshot(SCREENS / 'record_tab_card_after.png')
                d.js("document.getElementById('sim-rl-plans').scrollIntoView();")
                time.sleep(0.5)
                d.screenshot(SCREENS / 'record_tab_plans_after.png')
                res['card_after'] = d.js("""return [...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].map(e=>({name:e.querySelector('b')?.textContent,
                    badge:e.querySelector('.badge')?.textContent, nums:e.querySelector('.nums')?.textContent}));""")
            except Exception as e:  # noqa: BLE001
                res['tab_after_error'] = str(e)
    finally:
        res['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        (OUT / ('record_result_%s.json' % (pid or 'none'))).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        d.quit()
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
        glog.close()
    log('終わり', pid)


if __name__ == '__main__':
    main()
