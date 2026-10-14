"""昇格の段のブラウザの確認(2026-10-15、r3/promote)。統合の段の integrate/browser_check.py の道具(WebDriver・走行・待ち)を使う。

必ず Mac から使うときと同じ非ループバックのアドレス http://192.168.55.1:8080 で開く。DISPLAY :0 の snap Firefox を geckodriver(ポート 4448)で動かす。
jf-heavy --mem 2500M --lock big の中で実行すること。プロファイルは jetformer_reports/sim_rl_modes_20261013/r3/promote/ff_profile(終わったら消す)。

  A. F-ID(U5): 昇格したモードの保存した snapshot の列(no_wall 5・recover 4・line 2)と、変わっていないモード(fastest 1・speed 2)を、
     ページの teacher.js(import map で差し替えた教師。登録簿の mode.json の runtime を渡す)に 1 フレームずつ与え、Node の操作と比べる。
     1 フレームの計算の時間(performance.now)も測る(no_wall の抜け出しの計画・line のシールド・recover のすくみをほどく)。
     比べのため、同じことを Node でも teacher.js を通して行う(本物の rc-speed-controller.js)。
  B. 各モードで sim の約 10 秒走らせる(U1): /sim/c/default/?rl=<key> → 「▶ 評価を開始」(実行速度「高速」)→ 画面を r3/screens/ に。
  C. モードを選ばない(U2): RC加速教師のまま。
  D. JetFormer の「シミュレーター」タブのカード(U3)の文字と画面。

    python -u r3/promote/browser_promote.py [--skip-fid] [--skip-drive] [--drive-modes no_wall,recover,line,fastest,speed,rc]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes')
spec = importlib.util.spec_from_file_location('browser_check', EXP / 'integrate' / 'browser_check.py')
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

REPORTS = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
R3 = REPORTS / 'r3'
OUT = R3 / 'promote' / 'browser'
SCREENS = R3 / 'screens'
PROFILE = R3 / 'promote' / 'ff_profile'
JF = Path('/home/jetson/jetformer')
REG = JF / 'data/sim_rl/modes'
B = EXP / 'bundle'
FIDS = [('no_wall', B / 'no_wall/fid_std.json'), ('no_wall', B / 'no_wall/fid_slow.json'), ('no_wall', B / 'no_wall/fid_corner.json'),
        ('no_wall', B / 'no_wall/fid_escape.json'), ('no_wall', B / 'no_wall/fid_escape_rev.json'),
        ('recover', B / 'recover/fid/fid_FP_P11.json'), ('recover', B / 'recover/fid/fid_FP_P24.json'), ('recover', B / 'recover/fid/fid_RP_P10.json'),
        ('recover', B / 'recover/fid/fid_RP_P27.json'),
        ('line', B / 'line/fid_std_left08.json'), ('line', B / 'line/fid_ln_g30r120_r4b_g8_shield.json'),
        ('fastest', REPORTS / 'eval/fastest/fid_std.json'), ('speed', REPORTS / 'eval/speed/r2/fid_std_speed_r2_cv1.json'),
        ('speed', REPORTS / 'eval/speed/r2/fid_std_speed_r2_cv0.5.json')]

# ページの teacher.js で F-ID(フレームの時間も測る)。recover の F-ID ファイルには mode が無いので、引数のモードを使う
FID_JS = r"""
const done=arguments[arguments.length-1];const key=arguments[0], fid=JSON.parse(arguments[1]);
(async()=>{
  const T=await import('/sim/c/default/_jf/rl/teacher.js');
  const RL=window.JETFORMER_RL;const prev={mode:RL.mode,sp:RL.sceneParams};
  try{
    await RL.ui.loadWeights(key);
    RL.mode=key;RL.sceneParams={mode:key,params:fid.params||{}};
    const t=T.createRcTrainingTeacher(fid.world,fid.config,fid.variation);
    let maxS=0,maxV=0,info=0,first=null;const ms=[];let sh=0,unstick=0,esc=0;
    fid.frames.forEach((fr,i)=>{const t0=performance.now();const a=t.action(fr.snapshot);ms.push(performance.now()-t0);const b=fr.action;
      const ds=Math.abs(a.steering-b.steering),dv=Math.abs(a.speed-b.speed);
      maxS=Math.max(maxS,ds);maxV=Math.max(maxV,dv);if(JSON.stringify(a.info)!==JSON.stringify(b.info))info++;if(first===null&&(ds>1e-6||dv>1e-6))first=i;
      if(a.info?.jf?.sh)sh++;if(a.speed<0)esc++;});
    const s=[...ms].sort((x,y)=>x-y);
    const strip=m=>{const x={...m};delete x.episode_mode;delete x.teacher_slot;return JSON.stringify(x);};
    done({mode:key,frames:fid.frames.length,max_abs_steering:maxS,max_abs_speed:maxV,info_mismatch:info,first_frame_over:first,
      metadata_same:fid.metadata?strip(t.metadata)===strip(fid.metadata):null,policy_sha256_registry:RL.modes[key].policy_sha256,policy_sha256_fid:fid.policy_sha256,
      runtime_passed:RL.modes[key].runtime?{shield:RL.modes[key].runtime.shield,select:RL.modes[key].runtime.shield_params?.select??null}:null,
      shield_version:t.metadata.shield?.version??null,shield_frames:sh,negative_speed_frames:esc,
      frame_ms:{mean:ms.reduce((a,b)=>a+b,0)/ms.length,p50:s[Math.floor(.5*(s.length-1))],p95:s[Math.floor(.95*(s.length-1))],p99:s[Math.floor(.99*(s.length-1))],
        max:s[s.length-1],over33:ms.filter(x=>x>1000/30).length},
      pass:maxS<=1e-6&&maxV<=1e-6&&info===0&&RL.modes[key].policy_sha256===fid.policy_sha256});
  }catch(e){done({mode:key,error:String(e&&e.stack||e)});}
  finally{RL.mode=prev.mode;RL.sceneParams=prev.sp;}
})();"""

NODE_FID = r"""
import fs from 'node:fs';
const RL = globalThis.JETFORMER_RL = {mode: 'rc', weights: {}, modes: {}};
const T = await import('./_jf/rl/teacher.js');
const cases = JSON.parse(process.argv[2]);
const out = [];
for (const [key, f] of cases) {
  const fid = JSON.parse(fs.readFileSync(f, 'utf8'));
  const dir = '/home/jetson/jetformer/data/sim_rl/modes/' + key;
  const mode = JSON.parse(fs.readFileSync(dir + '/mode.json', 'utf8'));
  RL.weights[key] = JSON.parse(fs.readFileSync(dir + '/policy.json', 'utf8'));
  // sim-rl.js の loadWeights と同じ: 登録簿の mode.json の runtime を渡す
  RL.modes[key] = {name: key, policy_sha256: mode.policy_sha256, planned_time_s: null, state: 'accepted', runtime: mode.runtime ?? null};
  RL.mode = key; RL.sceneParams = {mode: key, params: fid.params || {}};
  const t = T.createRcTrainingTeacher(fid.world, fid.config, fid.variation);
  let maxS = 0, maxV = 0, info = 0;
  for (const fr of fid.frames) { const a = t.action(fr.snapshot), b = fr.action; maxS = Math.max(maxS, Math.abs(a.steering - b.steering));
    maxV = Math.max(maxV, Math.abs(a.speed - b.speed)); if (JSON.stringify(a.info) !== JSON.stringify(b.info)) info++; }
  const strip = m => { const x = {...m}; delete x.episode_mode; delete x.teacher_slot; return JSON.stringify(x); };
  out.push({mode: key, file: f, frames: fid.frames.length, max_abs_steering: maxS, max_abs_speed: maxV, info_mismatch: info,
    metadata_same: fid.metadata ? strip(t.metadata) === strip(fid.metadata) : null, policy_ok: mode.policy_sha256 === fid.policy_sha256,
    pass: maxS <= 1e-6 && maxV <= 1e-6 && info === 0 && mode.policy_sha256 === fid.policy_sha256});
}
console.log(JSON.stringify(out));
"""


def node_fid(out_dir):
    page = out_dir / 'node_page'
    if page.exists():
        shutil.rmtree(page)
    (page / '_jf').mkdir(parents=True)
    rel = json.loads((JF / 'data/sim/active.json').read_text())['release']
    dist = JF / 'data/sim/releases' / rel / 'dist'
    for n in ('rc-speed-controller.js', 'sim-core.js', 'speed-controller.js'):
        shutil.copy(dist / n, page / n)
    shutil.copytree(JF / 'jetformer/sim/static/rl', page / '_jf' / 'rl')
    (page / 'package.json').write_text('{"type": "module"}')
    (page / 'fid.mjs').write_text(NODE_FID)
    r = subprocess.run(['nice', '-n', '10', 'node', 'fid.mjs', json.dumps([[k, str(f)] for k, f in FIDS])], cwd=page, capture_output=True,
                       text=True, timeout=1800)
    shutil.rmtree(page, ignore_errors=True)
    if r.returncode:
        return {'error': r.stderr[-2000:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def card_check(d, res, suffix=''):
    d.get(f'{bc.BASE}/')
    d.js("window.jetformer.showTab('sim');")
    bc.wait(d, "return document.querySelectorAll('#sim-rl-modes .sim-rl-mode').length;", lambda v: v and v >= 5, 90, 1)
    time.sleep(2)
    d.js("document.getElementById('sim-rl-card').scrollIntoView();")
    d.screenshot(SCREENS / f'promote_tab_card{suffix}.png')
    res['card'] = d.js("""return [...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].map(e=>({name:e.querySelector('b')?.textContent,
        badge:e.querySelector('.badge')?.textContent, headline:e.querySelector('.headline')?.textContent, nums:e.querySelector('.nums')?.textContent,
        open:!!e.querySelector('[data-sim-rl-open]'), details:e.querySelector('details')?.textContent?.slice(0,1500)}));""")
    # 「試験の結果と注意」を開いた画面(no_wall・recover・line)
    for i, key in enumerate(('no_wall', 'recover', 'line')):
        try:
            d.js("""const k=arguments[0];const e=[...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].find(x=>x.querySelector('[data-sim-rl-open]')?.dataset.simRlOpen===k)
                    ||[...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].find(x=>(x.textContent||'').includes(arguments[1]));
                    if(e){const dd=e.querySelector('details');if(dd)dd.open=true;e.scrollIntoView({block:'start'});}""",
                 key, {'no_wall': '壁にぶつからない', 'recover': '離脱復活', 'line': '経路いろいろ'}[key])
            time.sleep(0.8)
            d.screenshot(SCREENS / f'promote_tab_card_{key}{suffix}.png')
        except Exception as e:  # noqa: BLE001
            res.setdefault('card_errors', []).append(f'{key}: {e}')
    d.js("document.getElementById('sim-rl-plan-form')?.scrollIntoView({block:'start'});")
    time.sleep(0.5)
    d.screenshot(SCREENS / f'promote_tab_plan_form{suffix}.png')
    res['plan_form'] = d.js("""return Object.fromEntries([...document.querySelectorAll('[data-sim-rl-min]')].map(i=>[i.dataset.simRlMin,{disabled:i.disabled,max:i.max}]));""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-fid', action='store_true')
    ap.add_argument('--skip-drive', action='store_true')
    ap.add_argument('--skip-card', action='store_true')
    ap.add_argument('--drive-modes', default='no_wall,recover,line,fastest,speed,rc')
    ap.add_argument('--drive-sim-seconds', type=float, default=10.0)
    ap.add_argument('--suffix', default='', help='画面のファイル名・結果の名前の後ろに付ける(2 回目の確認)')
    ap.add_argument('--no-fid-file', action='store_true', help='browser_fid.json を書かない')
    a = ap.parse_args()
    bc.SCREENS, bc.PROFILE, bc.GECKO_PORT, bc.OUT = SCREENS, PROFILE, 4448, OUT
    for p in (OUT, SCREENS, PROFILE):
        p.mkdir(parents=True, exist_ok=True)
    res = {'base': bc.BASE, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'fids': [[k, str(f)] for k, f in FIDS]}
    if not a.skip_fid:
        print('[promote] Node の F-ID(teacher.js 経由)', flush=True)
        res['node_fid'] = node_fid(OUT)
        print(json.dumps(res['node_fid'], ensure_ascii=False)[:1500], flush=True)
    t0 = time.time()
    while bc.other_firefox() or bc.mem_available_mb() < 1800:
        if time.time() - t0 > 3600:
            sys.exit('待ち時間切れ(ほかの Firefox かメモリ)')
        print(f'[promote] 待機: 他の firefox {len(bc.other_firefox())} / 空き {bc.mem_available_mb()} MB', flush=True)
        time.sleep(30)
    log = open(OUT / 'geckodriver.log', 'w')
    env = {'DISPLAY': ':0', 'XAUTHORITY': bc.XAUTH}
    gd = subprocess.Popen([bc.GECKO, '--port', '4448'], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = bc.WebDriver(4448)
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
        d.get(f'{bc.BASE}/sim/c/default/')
        res['page_ready'] = bc.wait(d, bc.READY, lambda v: v and v.get('ready') and v.get('registry'), 120, 1)
        res['browser_env'] = d.js("return {crypto_uuid: typeof crypto.randomUUID, subtle: typeof crypto.subtle, secure: window.isSecureContext, origin: location.origin, ua: navigator.userAgent};")
        fids = {}
        for key, f in ([] if a.skip_fid else FIDS):
            r = d.js_async(FID_JS, key, f.read_text(), timeout=900)
            r['file'] = str(f)
            fids[str(f)] = r
            print('[promote] F-ID', key, f.name, json.dumps({k: r.get(k) for k in ('max_abs_steering', 'max_abs_speed', 'info_mismatch', 'metadata_same',
                                                                                  'shield_frames', 'frame_ms', 'pass', 'error')}), flush=True)
            (OUT / f'result{a.suffix}.json').write_text(json.dumps({**res, 'browser_fid': fids}, ensure_ascii=False, indent=1))
        res['browser_fid'] = fids
        res['drive'] = {}
        if not a.skip_drive:
            for key in a.drive_modes.split(','):
                try:
                    bc.drive_mode(d, key, res['drive'], 10.0, a.drive_sim_seconds or None)
                    # 走行のあとの欄(注意の文・シールドのフレーム・線の形)
                    res['drive'][key]['panel'] = d.js("""const b=document.getElementById('jf-rl');return b?{info:b.querySelector('.jf-rl-info')?.textContent,
                        opts:b.querySelector('.jf-rl-opts')?.textContent, live:b.querySelector('.jf-rl-live')?.textContent, status:b.querySelector('.jf-rl-status')?.textContent}:null;""")
                except Exception as e:  # noqa: BLE001
                    res['drive'][key] = {'error': f'{type(e).__name__}: {e}', 'pass': False}
                    print('[promote] 走行の失敗', key, e, flush=True)
                    try:
                        d.screenshot(SCREENS / f'drive_{key}_error.png')
                    except Exception:  # noqa: BLE001
                        pass
                (OUT / f'result{a.suffix}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
        if not a.skip_card:
            try:
                card_check(d, res, a.suffix)
            except Exception as e:  # noqa: BLE001
                res['card_error'] = f'{type(e).__name__}: {e}'
    finally:
        res['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        (OUT / f'result{a.suffix}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
        d.quit()
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
    modes = {}
    for name, r in res.get('browser_fid', {}).items():
        m = r.get('mode')
        e = modes.setdefault(m, {'pass': True, 'files': []})
        e['files'].append({k: r.get(k) for k in ('file', 'frames', 'max_abs_steering', 'max_abs_speed', 'info_mismatch', 'metadata_same', 'frame_ms',
                                                 'shield_frames', 'error')})
        e['pass'] = e['pass'] and bool(r.get('pass'))
    for m, e in modes.items():
        e['where'] = bc.BASE + '(Firefox %s・DISPLAY :0)' % (res.get('browser') or {}).get('browserVersion')
        e['checked_at'] = res['finished_at']
    if modes and not a.no_fid_file:
        (R3 / 'promote' / 'browser_fid.json').write_text(json.dumps({'schema': 'jf-sim-rl-browser-fid/v1', 'modes': modes,
                                                                     'written_by': 'r3/promote/browser_promote.py'}, ensure_ascii=False, indent=1))
    print('[promote] 終わり', json.dumps({m: e['pass'] for m, e in modes.items()}),
          json.dumps({k: v.get('pass') for k, v in res.get('drive', {}).items()}), flush=True)


if __name__ == '__main__':
    main()
