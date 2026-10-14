"""離脱復活 r3 の確認: 候補の走行中の仕組み(r3/candidates/recover_r3/runtime/)が Firefox と Node で同じ操作を出すか(F-ID、2026-10-14 21:20)

候補はまだ JetFormer(8080)に入っていない(昇格の段で web/rl に入れる)ので、候補の runtime・方策・F-ID の材料(ck_fid_make.mjs)を一時フォルダーに写し、
**Mac から開くときと同じ非ループバックの http://192.168.55.1:8093/**(安全でない生まれ)で配る。DISPLAY :0 の snap Firefox を geckodriver(ポート 4448)で動かし、
ページで import('./runtime/mode_runtime.mjs') → 各フレームの {state, metrics, config} を教師に与え、Node の出力と比べる(|Δ| ≤ 1e-6・info.jf も同じ)。
jf-heavy --mem 2500M --lock big の中で実行する。終わったら Firefox・geckodriver・配るサーバーを止め、プロファイルを消す。

    python ck_browser_fid.py <F-ID の材料のフォルダー> <出力.json>
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FID_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CAND = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/recover_r3')
WORK = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3/browser')   # snap の Firefox が読めるホームの下
PROFILE = WORK / 'ff_profile'
SERVE = WORK / 'serve'
HOST, PORT, GECKO_PORT = '192.168.55.1', 8093, 4448
GECKO = '/snap/bin/geckodriver'
ENV = dict(os.environ, DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority')


class WebDriver:
    def __init__(self, port):
        self.base = 'http://127.0.0.1:%d' % port
        self.sid = None

    def _req(self, method, path, body=None, timeout=300):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b'{}').get('value')
        except urllib.error.HTTPError as e:
            raise RuntimeError('%s %s: %s %r' % (method, path, e.code, e.read()[:800])) from None

    def start(self):
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {
            'args': ['-profile', str(PROFILE), '-no-remote'], 'prefs': {'browser.shell.checkDefaultBrowser': False}}}}}
        v = self._req('POST', '/session', caps, timeout=180)
        self.sid = v['sessionId']
        return v.get('capabilities', {})

    def s(self, method, path, body=None, timeout=300):
        return self._req(method, '/session/%s%s' % (self.sid, path), body, timeout)

    def quit(self):
        if self.sid:
            try:
                self._req('DELETE', '/session/%s' % self.sid, timeout=60)
            except Exception as e:  # noqa: BLE001
                print('[fid] セッション終了の失敗:', e)
            self.sid = None


FID_JS = r"""
const done=arguments[arguments.length-1];const files=arguments[0];
(async()=>{
  const out=[];
  try{
    const RT=await import('./runtime/mode_runtime.mjs');
    const weights=await (await fetch('./policy.json')).json();
    for(const f of files){
      const fid=await (await fetch('./'+f)).json();
      const t=RT.createModeTeacher(fid.world,fid.config,fid.variation,{mode:'recover',weights,policySha256:fid.policy_sha256,params:fid.params||{}});
      let maxS=0,maxV=0,info=0,first=null;
      fid.frames.forEach((fr,i)=>{const a=t.action(fr.snapshot),b=fr.action;const ds=Math.abs(a.steering-b.steering),dv=Math.abs(a.speed-b.speed);
        maxS=Math.max(maxS,ds);maxV=Math.max(maxV,dv);if(JSON.stringify(a.info)!==JSON.stringify(b.info))info++;
        if(first===null&&(ds>1e-6||dv>1e-6||JSON.stringify(a.info)!==JSON.stringify(b.info)))first=i;});
      // 閉じた回(ページで元の sim-core.js を走らせる): Node の記録と操作・出来事・終わり方を比べる(記録の前の予想がブラウザの記録と同じになるか)
      const C=await import('./sim-core.js');
      const sim=C.createSimulation(fid.world,{...fid.config});let snap=sim.reset({seed:fid.config.seed,scenario:'baseline'});
      const t2=RT.createModeTeacher(fid.world,snap.config,fid.variation,{mode:'recover',weights,policySha256:fid.policy_sha256,params:fid.params||{}});
      let cl={frames:0,maxS:0,maxV:0,firstOver:null,infoMismatch:0};
      while(!snap.state.terminal){const i=snap.state.frame;const a=t2.action({state:snap.state,metrics:snap.metrics,config:snap.config});
        const b=fid.frames[i]?.action;if(b){const ds=Math.abs(a.steering-b.steering),dv=Math.abs(a.speed-b.speed);cl.maxS=Math.max(cl.maxS,ds);cl.maxV=Math.max(cl.maxV,dv);
          if(JSON.stringify(a.info)!==JSON.stringify(b.info))cl.infoMismatch++;if(cl.firstOver===null&&(ds>1e-6||dv>1e-6))cl.firstOver=i;}
        cl.frames++;snap=sim.step({steering:a.steering,speed:a.speed});}
      cl.reason=snap.state.reason;cl.events=t2.events.length;cl.shield=t2.shieldFrames;cl.final={x:snap.state.x,z:snap.state.z};
      out.push({closed_loop:cl,closed_loop_pass:cl.frames===fid.frames.length&&cl.maxS<=1e-6&&cl.maxV<=1e-6&&cl.infoMismatch===0&&cl.reason===fid.reason&&cl.events===fid.events,
        file:f,frames:fid.frames.length,events_node:fid.events,events_browser:t.events.length,shield_node:fid.shield_frames,shield_browser:t.shieldFrames,
        max_abs_steering:maxS,max_abs_speed:maxV,info_mismatch:info,first_mismatch:first,metadata_same:JSON.stringify(t.metadata)===JSON.stringify(fid.metadata),
        pass:maxS<=1e-6&&maxV<=1e-6&&info===0&&t.events.length===fid.events&&t.shieldFrames===fid.shield_frames});
    }
    done({ok:true,origin:location.origin,secure:window.isSecureContext,ua:navigator.userAgent,results:out});
  }catch(e){done({ok:false,error:String(e&&e.stack||e),results:out});}
})();"""


CL_JS = r"""
const done=arguments[arguments.length-1];
(async()=>{
  try{
    const RT=await import('./runtime/mode_runtime.mjs');const C=await import('./sim-core.js');
    const weights=await (await fetch('./policy.json')).json();const specs=await (await fetch('./cl_specs.json')).json();
    const out=[];
    for(const sp of specs){
      const sim=C.createSimulation(sp.world,{...sp.config});let snap=sim.reset({seed:sp.config.seed,scenario:'baseline'});
      if(snap.state.terminal){out.push({id:sp.id,invalid_spawn:true});continue;}
      const t=RT.createModeTeacher(sp.world,snap.config,sp.variation,{mode:'recover',weights,params:sp.params||{}});
      const v=[],ss=[],g=[];
      while(!snap.state.terminal){const a=t.action({state:snap.state,metrics:snap.metrics,config:snap.config});
        v.push(Math.round(snap.state.speed*1000)/1000);ss.push(Math.round(t.log.at(-1).s*1e4)/1e4);g.push(snap.metrics.passed_gates);
        snap=sim.step({steering:a.steering,speed:a.speed});}
      out.push({id:sp.id,reason:snap.state.reason,frames:snap.state.frame,collisions:snap.metrics.collisions,
        events:t.events.map(e=>[e.ps,e.pe,e.type,e.kind??null,e.cls]),shield:t.shieldFrames,v,s:ss,g});
    }
    done({ok:true,results:out});
  }catch(e){done({ok:false,error:String(e&&e.stack||e)});}
})();"""


def main():
    if subprocess.run(['pgrep', '-x', 'firefox'], capture_output=True).returncode == 0:
        print('[fid] ほかの Firefox が動いています(-no-remote の別のプロファイルで続けます)')
    shutil.rmtree(WORK, ignore_errors=True)
    (SERVE / 'runtime').mkdir(parents=True)
    PROFILE.mkdir(parents=True)
    for f in (CAND / 'runtime').iterdir():
        shutil.copy(f, SERVE / 'runtime' / f.name)
    shutil.copy(CAND / 'policy.json', SERVE / 'policy.json')
    shutil.copy('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js', SERVE / 'sim-core.js')   # ユーザーの元の物理
    files = sorted(p.name for p in FID_DIR.glob('fid_*.json'))
    if (FID_DIR / 'cl_specs.json').exists():
        shutil.copy(FID_DIR / 'cl_specs.json', SERVE / 'cl_specs.json')
    for f in files:
        shutil.copy(FID_DIR / f, SERVE / f)
    (SERVE / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>recover r3 F-ID</title><p>F-ID</p>', encoding='utf-8')
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', HOST, '--directory', str(SERVE)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gk = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wd = WebDriver(GECKO_PORT)
    res = {'url': 'http://%s:%d/' % (HOST, PORT), 'files': files}
    try:
        for _ in range(120):                       # snap の geckodriver は立ち上がりが遅いので /status が返るまで待つ
            try:
                wd._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        caps = wd.start()
        res['browser'] = {'name': caps.get('browserName'), 'version': caps.get('browserVersion')}
        wd.s('POST', '/timeouts', {'script': 900000, 'pageLoad': 120000})   # 閉じた回は 30 秒(既定)より長くかかる
        wd.s('POST', '/url', {'url': res['url']})
        if os.environ.get('CK_SKIP_FID') != '1':
            res['fid'] = wd.s('POST', '/execute/async', {'script': FID_JS, 'args': [files]}, timeout=900)
        if (FID_DIR / 'cl_specs.json').exists():
            res['closed_loop_batch'] = wd.s('POST', '/execute/async', {'script': CL_JS, 'args': []}, timeout=900)
    except Exception as e:  # noqa: BLE001
        res['error'] = repr(e)
    finally:
        wd.quit()
        for p in (gk, srv):
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(WORK, ignore_errors=True)    # プロファイル・写しを消す
    OUT.write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != 'closed_loop_batch'}, ensure_ascii=False, indent=1)[:3000])


if __name__ == '__main__':
    main()
