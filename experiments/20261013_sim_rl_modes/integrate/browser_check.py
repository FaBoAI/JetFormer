"""強化学習モードの統合の確認(ブラウザ。2026-10-14、sim_rl_modes の統合の段)

必ず Mac から使うときと同じ非ループバックのアドレス http://192.168.55.1:8080 で開く(127.0.0.1 では crypto.randomUUID などの違いが隠れる)。
DISPLAY :0 の snap Firefox を geckodriver(ポート 4447)で動かす。jf-heavy --mem 2500M --lock big の中で実行すること。

  A. F-ID(U5): 保存した 600 フレームの snapshot の列(Node の試験の段の fid_*.json)を、ページの teacher.js(import map で差し替えた教師)に
     1 フレームずつ与え、Node の操作と比べる(|Δ| ≤ 1e-6・info.jf も同じ)。方策は JetFormer の API から読む(ページと同じ道)。
     比べのため、同じことを Node でも teacher.js を通して行う(元の rc-speed-controller.js を使う)。
  B. 各モードで ~10 秒走らせる(U1): /sim/c/default/?rl=<key> → 「▶ 評価を開始」→ 10 秒 → 画面を保存。
  C. モードを選ばない(U2): RC加速教師のまま走る(teacher.js が元の教師を返す)。
  D. 収集プラン(U3・U4): JetFormer の「シミュレーター」タブのカード → 分を入れて「プランを作って開く」→ 新しいタブで「開始」→
     記録 → モードごとの取り込み(学習除外・タグ)を API で確かめる。

    python integrate/browser_check.py [--skip-plan] [--plan-minutes fastest=1,recover=1] [--out <dir>]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
OUT = REPORTS / 'integrate' / 'browser'
SCREENS = REPORTS / 'screens'
PROFILE = REPORTS / 'integrate' / 'ff_profile'        # snap の Firefox が読めるよう、ホームの隠しでないフォルダー
GECKO, GECKO_PORT = '/snap/bin/geckodriver', 4447
BASE = 'http://192.168.55.1:8080'                     # 非ループバック(Mac から開くときと同じ)
API = 'http://127.0.0.1:8080'                         # Python 側の API の読み書き(ページはいつも BASE)
XAUTH = '/run/user/1000/gdm/Xauthority'
JF = Path('/home/jetson/jetformer')
FIDS = [('fastest', REPORTS / 'eval/fastest/fid_std.json'),
        ('no_wall', REPORTS / 'eval/no_wall/fid_std_no_wall_r2.json'),
        ('no_wall', REPORTS / 'eval/no_wall/fid_slow_no_wall_r2.json'),
        ('recover', REPORTS / 'train/recover/fid_scene_P5.json'),
        ('recover', REPORTS / 'train/recover/fid_std.json'),
        ('speed', REPORTS / 'eval/speed/r2/fid_std_speed_r2_cv1.json'),
        ('speed', REPORTS / 'eval/speed/r2/fid_std_speed_r2_cv0.5.json')]


# ---------------------------------------------------------------- WebDriver(最小。ベストラップの作業の webgl_probe.py の WebDriver の写し)
class WebDriver:
    def __init__(self, port):
        self.base = f'http://127.0.0.1:{port}'
        self.sid = None

    def _req(self, method, path, body=None, timeout=300):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b'{}').get('value')
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{method} {path}: {e.code} {e.read()[:800]!r}') from None

    def start(self, args, prefs, env):
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {'args': args, 'prefs': prefs, 'env': env}}}}
        v = self._req('POST', '/session', caps, timeout=180)
        self.sid = v['sessionId']
        return v.get('capabilities', {})

    def s(self, method, path, body=None, timeout=300):
        return self._req(method, f'/session/{self.sid}{path}', body, timeout)

    def get(self, url):
        return self.s('POST', '/url', {'url': url})

    def js(self, script, *args, timeout=300):
        return self.s('POST', '/execute/sync', {'script': script, 'args': list(args)}, timeout)

    def js_async(self, script, *args, timeout=300):
        return self.s('POST', '/execute/async', {'script': script, 'args': list(args)}, timeout)

    def screenshot(self, path):
        Path(path).write_bytes(base64.b64decode(self.s('GET', '/screenshot')))

    def click(self, css):
        """WebDriver の Element Click(本当の操作として扱われるので、window.open がポップアップとして止められない)"""
        el = self.s('POST', '/element', {'using': 'css selector', 'value': css})
        eid = next(iter(el.values()))
        return self.s('POST', f'/element/{eid}/click', {})

    def handles(self):
        return self.s('GET', '/window/handles')

    def switch(self, h):
        return self.s('POST', '/window', {'handle': h})

    def quit(self):
        if self.sid:
            try:
                self._req('DELETE', f'/session/{self.sid}', timeout=60)
            except Exception as e:  # noqa: BLE001
                print('[check] セッション終了の失敗:', e)
            self.sid = None


def mem_available_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def other_firefox():
    out = subprocess.run(['pgrep', '-af', 'firefox'], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if 'pgrep' not in l and 'browser_check.py' not in l and ('/firefox' in l.split(' ', 1)[1].split(' ')[0])]


def cap_snap_scope(pid, mem):
    try:
        scope = Path(f'/proc/{pid}/cgroup').read_text().strip().splitlines()[-1].rsplit('/', 1)[-1]
    except OSError as e:
        return {'error': str(e)}
    if not scope.startswith('snap.'):
        return {'scope': scope}
    r = subprocess.run(['systemctl', '--user', 'set-property', '--runtime', scope, f'MemoryMax={mem}', 'MemorySwapMax=0'], capture_output=True, text=True)
    return {'scope': scope, 'returncode': r.returncode}


def api(method, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b'null')


READY = r"""
const s=window.courseSimulator;const RL=window.JETFORMER_RL||{};
return {ready:!!(s&&s.sensorSummary().ready), fatal:document.getElementById('fatal')?.hidden===false?document.getElementById('fatal').textContent:null,
  rl_mode:RL.mode||null, teacher_loaded:!!RL.teacherModule, registry:!!RL.registry, sel:document.getElementById('jf-rl-mode')?.value??null,
  rl_status:document.getElementById('jf-rl')?.querySelector('.jf-rl-status')?.textContent??null,
  status:document.getElementById('status')?.textContent??null, frame:s?s.snapshot().state.frame:null,
  teacher:document.getElementById('teacher-profile')?.value, controller:document.getElementById('controller')?.value,
  teacher_label:[...(document.getElementById('teacher-profile')?.options||[])].find(o=>o.value==='speed-rc')?.textContent??null,
  bar:document.getElementById('jf-simbar')?.textContent??null};"""

LIVE = r"""
const s=window.courseSimulator.snapshot();const RL=window.JETFORMER_RL||{};
return {frame:s.state.frame, sim_time:s.state.sim_time, speed:s.state.speed, terminal:s.state.terminal, reason:s.state.reason,
  collisions:s.metrics.collisions, laps:s.metrics.completed_laps??null, gates:s.metrics.passed_gates??null,
  live:RL.live?JSON.parse(JSON.stringify(RL.live)):null, calls:RL.calls?{...RL.calls}:null,
  live_text:document.querySelector('#jf-rl .jf-rl-live')?.textContent??null, status:document.getElementById('status')?.textContent??null,
  crypto_uuid:typeof crypto.randomUUID, secure:window.isSecureContext, origin:location.origin};"""

FID_JS = r"""
const done=arguments[arguments.length-1];const key=arguments[0], fid=JSON.parse(arguments[1]);
(async()=>{
  const T=await import('/sim/c/default/_jf/rl/teacher.js');
  const RL=window.JETFORMER_RL;const prev={mode:RL.mode,sp:RL.sceneParams};
  try{
    await RL.ui.loadWeights(key);
    RL.mode=key;RL.sceneParams={mode:key,params:fid.params||{}};
    const t=T.createRcTrainingTeacher(fid.world,fid.config,fid.variation);
    let maxS=0,maxV=0,info=0,first=null;
    fid.frames.forEach((fr,i)=>{const a=t.action(fr.snapshot),b=fr.action;const ds=Math.abs(a.steering-b.steering),dv=Math.abs(a.speed-b.speed);
      maxS=Math.max(maxS,ds);maxV=Math.max(maxV,dv);if(JSON.stringify(a.info)!==JSON.stringify(b.info))info++;if(first===null&&(ds>1e-6||dv>1e-6))first=i;});
    const strip=m=>{const x={...m};delete x.episode_mode;delete x.teacher_slot;return JSON.stringify(x);};
    done({mode:key,frames:fid.frames.length,max_abs_steering:maxS,max_abs_speed:maxV,info_mismatch:info,first_frame_over:first,
      metadata_same:strip(t.metadata)===strip(fid.metadata),policy_sha256_registry:RL.modes[key].policy_sha256,policy_sha256_fid:fid.policy_sha256,
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
  const p = '/home/jetson/jetformer/data/sim_rl/modes/' + key + '/policy.json';
  RL.weights[key] = JSON.parse(fs.readFileSync(p, 'utf8'));
  RL.modes[key] = {name: key, policy_sha256: fid.policy_sha256};
  RL.mode = key; RL.sceneParams = {mode: key, params: fid.params || {}};
  const t = T.createRcTrainingTeacher(fid.world, fid.config, fid.variation);
  let maxS = 0, maxV = 0, info = 0;
  for (const fr of fid.frames) { const a = t.action(fr.snapshot), b = fr.action; maxS = Math.max(maxS, Math.abs(a.steering - b.steering));
    maxV = Math.max(maxV, Math.abs(a.speed - b.speed)); if (JSON.stringify(a.info) !== JSON.stringify(b.info)) info++; }
  out.push({mode: key, file: f, frames: fid.frames.length, max_abs_steering: maxS, max_abs_speed: maxV, info_mismatch: info,
    pass: maxS <= 1e-6 && maxV <= 1e-6 && info === 0});
}
console.log(JSON.stringify(out));
"""


def node_fid(out_dir):
    """Node で teacher.js(本物の rc-speed-controller.js)を通して F-ID を確かめる(ブラウザと同じ比べ方)"""
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
                       text=True, timeout=900)
    if r.returncode:
        return {'error': r.stderr[-2000:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def wait(d, script, ok, timeout, every=1.0):
    t0 = time.time()
    v = None
    while time.time() - t0 < timeout:
        v = d.js(script)
        if ok(v):
            return v
        time.sleep(every)
    raise TimeoutError(f'待ち時間切れ: {json.dumps(v, ensure_ascii=False)[:400]}')


def save_canvas(d, cid, path):
    url = d.js("return document.getElementById(arguments[0]).toDataURL('image/png');", cid)
    Path(path).write_bytes(base64.b64decode(url.split(',', 1)[1]))


def drive_mode(d, key, res, seconds=10.0, sim_seconds=None):
    """/sim/c/default/?rl=<key> で開き、▶ で seconds 秒(sim_seconds があれば sim の時間でその秒数・実行速度は「高速」)走らせる"""
    d.get(f'{BASE}/sim/c/default/?rl={key}')
    v = wait(d, READY, lambda v: v and v.get('ready') and (v.get('rl_mode') == key or key == 'rc') and v.get('teacher') == ('speed-rc' if key != 'rc' else v.get('teacher')),
             120, 1)
    r = {'ready': v}
    if key == 'rc':
        # モードを選ばない: RC加速教師(ゲート 0・ずれ 0)をそのまま使う。物理・センサーは記録の条件(RC_COLLECTION_PLAN[0])
        d.js("""const $=id=>document.getElementById(id);const b=window.JETFORMER_RL.basePlan;
                const keep=new Set(['name','dataset-split','seed','scenario','execution-speed','dataset-mode']);
                for(const [k,v] of Object.entries(b)){if(keep.has(k))continue;const e=$(k);if(!e)continue;if(e.type==='checkbox')e.checked=!!v;else e.value=String(v);}
                $('controller').value='demo';$('teacher-profile').value='speed-rc';
                $('episode-mode').value='continuous-laps-v1';$('limit').value='20';$('start-gate').value='0';$('spawn-offset').value='0';$('spawn-yaw').value='0';
                $('teacher-profile').dispatchEvent(new Event('change'));""")
        time.sleep(1.5)
    if sim_seconds:
        d.js("const e=document.getElementById('execution-speed');e.value='fast';")
    r['before'] = d.js(LIVE)
    d.js("document.getElementById('run').click();")
    t0 = time.time()
    samples = []
    while (time.time() - t0 < seconds) if not sim_seconds else (time.time() - t0 < 180 and (not samples or (samples[-1]['sim_time'] or 0) < sim_seconds
                                                                                          and not samples[-1]['terminal'])):
        time.sleep(2.5)
        samples.append(d.js(LIVE))
    if sim_seconds:
        d.js("const b=document.getElementById('run');if(/一時停止/.test(b.textContent))b.click();")
    stem = f'drive_{key}'
    d.js("window.scrollTo(0,0);")
    d.screenshot(SCREENS / f'{stem}_window.png')
    d.js("const x=document.getElementById('experiments');if(x)x.open=true;const p=document.getElementById('jf-rl')||document.getElementById('teacher-profile');if(p)p.scrollIntoView({block:'center'});")
    time.sleep(0.5)
    d.screenshot(SCREENS / f'{stem}_panel.png')
    d.js("window.scrollTo(0,0);")
    try:
        save_canvas(d, 'camera', SCREENS / f'{stem}_camera.png')
    except Exception as e:  # noqa: BLE001
        r['camera_error'] = str(e)
    after = d.js(LIVE)
    d.js("const b=document.getElementById('run');if(/一時停止/.test(b.textContent))b.click();")
    r.update(samples=samples, after=after, wall_s=round(time.time() - t0, 1), screenshot=f'screens/{stem}_window.png')
    r['pass'] = bool(after['frame'] and after['frame'] > 30 and after['collisions'] == 0 and (key == 'rc' or (after['live'] or {}).get('mode') == key)
                     and not (after['live'] or {}).get('error'))
    if key == 'rc':
        r['pass'] = r['pass'] and after['calls'] and after['calls']['rl_teachers'] == 0 and after['calls']['createRcTrainingTeacher'] >= 1
    res[key] = r
    print('[check] 走行', key, json.dumps({k: after.get(k) for k in ('frame', 'collisions', 'laps', 'speed')}, ensure_ascii=False),
          (after.get('live_text') or '')[:160], 'pass', r['pass'], flush=True)


def plan_check(d, minutes, res):
    """JetFormer のタブ → カード → プランを作って開く → 新しいタブで開始 → 記録・取り込みを待つ"""
    d.get(f'{BASE}/')
    d.js("window.jetformer.showTab('sim');")
    wait(d, "return document.querySelectorAll('#sim-rl-modes .sim-rl-mode').length;", lambda v: v and v >= 4, 60, 1)
    time.sleep(1)
    d.js("document.getElementById('sim-rl-card').scrollIntoView();")
    d.screenshot(SCREENS / 'tab_rl_card.png')
    card = d.js("""return [...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].map(e=>({name:e.querySelector('b')?.textContent,
        badge:e.querySelector('.badge')?.textContent, headline:e.querySelector('.headline')?.textContent, nums:e.querySelector('.nums')?.textContent,
        open:!!e.querySelector('[data-sim-rl-open]')}));""")
    res['card'] = card
    # 分を入れる(ほかは 0)
    d.js("""const want=arguments[0];for(const inp of document.querySelectorAll('[data-sim-rl-min]')){inp.value=String(want[inp.dataset.simRlMin]||0);
        inp.dispatchEvent(new Event('input',{bubbles:true}));}""", minutes)
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
    main = list(before)[0]
    d.switch(new[0])
    v = wait(d, "return {url:location.href, panel:document.querySelector('#jf-rl-plan')?.textContent||null, ready:!!(window.courseSimulator&&window.courseSimulator.sensorSummary().ready)};",
             lambda v: v and v.get('ready') and v.get('panel') and '読み込み中' not in v['panel'], 120, 1)
    pid = v['url'].split('rlplan=', 1)[1].split('&')[0]
    res['plan_id'] = pid
    d.screenshot(SCREENS / 'plan_before_start.png')
    d.click('#jf-rl-plan .jf-rl-btn.primary')
    t0 = time.time()
    shot = False
    while time.time() - t0 < 45 * 60:
        time.sleep(10)
        p = api('GET', f'/api/sim/rl/plans/{pid}')
        done = sum(1 for x in p['progress'].values() if x.get('status') in ('done', 'failed', 'skipped'))
        print('[check] プラン', p['state'], f'{done}/{len(p["order"])}', json.dumps(p.get('imports'), ensure_ascii=False)[:300], flush=True)
        if not shot and any(x.get('status') == 'recording' for x in p['progress'].values()) and time.time() - t0 > 40:
            d.screenshot(SCREENS / 'plan_recording.png')
            shot = True
        imps = p.get('imports') or {}
        if p['state'] in ('finished', 'stopped', 'error') and all((imps.get(k) or {}).get('state') in ('done', 'error') for k in p['modes']):
            break
    res['plan'] = {k: p.get(k) for k in ('state', 'message', 'progress', 'imports', 'calibration', 'order')}
    d.screenshot(SCREENS / 'plan_after.png')
    d.switch(main)
    d.js("window.jetformer.loadSim();")
    time.sleep(3)
    d.js("document.getElementById('sim-rl-plans').scrollIntoView();")
    d.screenshot(SCREENS / 'tab_rl_after_plan.png')
    # 取り込んだデータセットを確かめる(ジャンル 8・タグ・学習除外)
    ds = {}
    for key, imp in (p.get('imports') or {}).items():
        if imp.get('state') != 'done':
            continue
        name = imp['dataset']
        root = JF / 'data/datasets' / name
        cardj = json.loads((root / 'dataset_card.json').read_text())
        import numpy as np
        sc = json.loads((root / 'splits.json').read_text())
        masked, frames, modes = 0, 0, set()
        for s in sc['train'] + sc['val']:
            with np.load(root / 'aligned' / f'{s}.npz') as z:
                frames += len(z['mode'])
                masked += int((z['mode'] == 0).sum())
                modes |= set(int(x) for x in np.unique(z['mode']))
        ds[key] = {'name': name, 'tags': cardj.get('tags'), 'scenes': cardj.get('scenes'), 'frames': frames, 'modes_in_npz': sorted(modes),
                   'mode0_frames': masked, 'label_mask': {k: (cardj.get('label_mask') or {}).get(k) for k in ('masked_frames', 'rule_id', 'scenes')},
                   'rl': cardj.get('rl'), 'training_frames': cardj.get('training_frames')}
    res['datasets'] = ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--skip-plan', action='store_true')
    ap.add_argument('--skip-drive', action='store_true')
    ap.add_argument('--plan-minutes', default='fastest=1,recover=1')
    ap.add_argument('--drive-seconds', type=float, default=10.0)
    ap.add_argument('--drive-sim-seconds', type=float, default=0.0, help='0 でなければ、sim の時間でこの秒数走らせる(実行速度「高速」)')
    ap.add_argument('--skip-fid', action='store_true')
    ap.add_argument('--drive-modes', default='fastest,no_wall,recover,speed,rc')
    ap.add_argument('--firefox-mem', default='2500M')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    SCREENS.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    res = {'base': BASE, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    print('[check] Node の F-ID(teacher.js 経由)', flush=True)
    res['node_fid'] = node_fid(a.out) if not a.skip_fid else None
    print(json.dumps(res['node_fid'], ensure_ascii=False)[:600], flush=True)
    t0 = time.time()
    while other_firefox() or mem_available_mb() < 1800:
        if time.time() - t0 > 3600:
            sys.exit('待ち時間切れ(ほかの Firefox かメモリ)')
        print(f'[check] 待機: 他の firefox {len(other_firefox())} / 空き {mem_available_mb()} MB', flush=True)
        time.sleep(30)
    log = open(a.out / 'geckodriver.log', 'w')
    env = {'DISPLAY': ':0', 'XAUTHORITY': XAUTH}
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = WebDriver(GECKO_PORT)
    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        caps = d.start(['-profile', str(PROFILE)], {}, env)
        res['browser'] = {k: caps.get(k) for k in ('browserName', 'browserVersion', 'platformName')}
        res['memory_cap'] = cap_snap_scope(gd.pid, a.firefox_mem)
        d.s('POST', '/window/rect', {'width': 1500, 'height': 1000})
        # A. F-ID(ブラウザ)
        d.get(f'{BASE}/sim/c/default/')
        res['page_ready'] = wait(d, READY, lambda v: v and v.get('ready') and v.get('registry'), 120, 1)
        fids = {}
        for key, f in ([] if a.skip_fid else FIDS):
            r = d.js_async(FID_JS, key, f.read_text(), timeout=600)
            r['file'] = str(f.relative_to(REPORTS))
            fids[r['file']] = r                      # 名前だけだと eval/fastest/fid_std.json と train/recover/fid_std.json が重なる
            print('[check] F-ID', key, f.name, json.dumps({k: r.get(k) for k in ('max_abs_steering', 'max_abs_speed', 'info_mismatch', 'pass', 'error')}), flush=True)
        res['browser_fid'] = fids
        res['browser_env'] = d.js("return {crypto_uuid: typeof crypto.randomUUID, subtle: typeof crypto.subtle, secure: window.isSecureContext, origin: location.origin, ua: navigator.userAgent};")
        # B・C. 各モードで走らせる / モードを選ばない
        res['drive'] = {}
        if not a.skip_drive:
            for key in a.drive_modes.split(','):
                try:
                    drive_mode(d, key, res['drive'], a.drive_seconds, a.drive_sim_seconds or None)
                except Exception as e:  # noqa: BLE001
                    res['drive'][key] = {'error': f'{type(e).__name__}: {e}', 'pass': False}
                    print('[check] 走行の失敗', key, e, flush=True)
                    try:
                        d.screenshot(SCREENS / f'drive_{key}_error.png')
                    except Exception:  # noqa: BLE001
                        pass
        # D'. プランを作らないときは、最後のプランの枠とカードの様子だけを画面に残す
        if a.skip_plan:
            try:
                plans = api('GET', '/api/sim/rl/plans')['plans']
                if plans:
                    d.get(BASE + plans[0]['url'])
                    wait(d, "return document.querySelector('#jf-rl-plan table tr:nth-child(2)')?.textContent||null;", lambda v: v, 60, 1)
                    time.sleep(2)
                    d.screenshot(SCREENS / 'plan_panel_after.png')
                d.get(f'{BASE}/')
                d.js("window.jetformer.showTab('sim');")
                wait(d, "return document.querySelectorAll('#sim-rl-modes .sim-rl-mode').length;", lambda v: v and v >= 4, 60, 1)
                time.sleep(2)
                d.js("document.getElementById('sim-rl-card').scrollIntoView();")
                d.screenshot(SCREENS / 'tab_rl_card_after.png')
                d.js("document.getElementById('sim-rl-plan-form').scrollIntoView({block:'start'});")
                time.sleep(0.5)
                d.screenshot(SCREENS / 'tab_rl_plan_form.png')
                res['card_after'] = d.js("""return [...document.querySelectorAll('#sim-rl-modes .sim-rl-mode')].map(e=>({name:e.querySelector('b')?.textContent,
                    badge:e.querySelector('.badge')?.textContent, nums:e.querySelector('.nums')?.textContent}));""")
            except Exception as e:  # noqa: BLE001
                res['show_plan_error'] = f'{type(e).__name__}: {e}'
        # D. 収集プラン
        if not a.skip_plan:
            minutes = {k: int(v) for k, v in (x.split('=') for x in a.plan_minutes.split(',') if x)}
            try:
                plan_check(d, minutes, res)
            except Exception as e:  # noqa: BLE001
                res['plan_error'] = f'{type(e).__name__}: {e}'
                print('[check] プランの失敗', e, flush=True)
                try:
                    d.screenshot(SCREENS / 'plan_error.png')
                except Exception:  # noqa: BLE001
                    pass
    finally:
        res['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        (a.out / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
        d.quit()
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
    # 登録簿の判定に使う形(最速の F4・各モードの U5)
    modes = {}
    for name, r in res.get('browser_fid', {}).items():
        m = r.get('mode')
        e = modes.setdefault(m, {'pass': True, 'files': []})
        e['files'].append({k: r.get(k) for k in ('file', 'frames', 'max_abs_steering', 'max_abs_speed', 'info_mismatch', 'metadata_same', 'error')})
        e['pass'] = e['pass'] and bool(r.get('pass'))
    for m, e in modes.items():
        e['where'] = BASE + '(Firefox %s・DISPLAY :0)' % (res.get('browser') or {}).get('browserVersion')
        e['checked_at'] = res['finished_at']
    if not modes:
        print('[check] F-ID をしていないので browser_fid.json は書きません', flush=True)
        return
    (REPORTS / 'integrate' / 'browser_fid.json').write_text(json.dumps({'schema': 'jf-sim-rl-browser-fid/v1', 'modes': modes}, ensure_ascii=False, indent=1))
    print('[check] 終わり', json.dumps({m: e['pass'] for m, e in modes.items()}), flush=True)


if __name__ == '__main__':
    main()
