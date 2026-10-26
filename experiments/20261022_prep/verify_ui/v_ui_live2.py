"""段 V-ui(2026-10-22 の作業・10/23 早朝): 本物の jetformer-web(:8080)の走行タブを、LAN の住所 http://192.168.55.1:8080 から
ヘッドレス Firefox で確かめる。10/21 の experiments/20261021_bench_prep/verify_ui/v_ui_live.py の写し。変えたのは:
  - 出力の場所(jetformer_reports/prep_20261022/verify_ui/)
  - 10/23 の並び B・SLOW-SEC1・ENS・B・AUG・ACT・WGT・B(段 Decide が C1 を SLOW → SLOW-SEC1 に替えた)。期待はファイルから作る
  - 0: API がファイルを読み直していること(サーバーの起動はファイルの更新より前)・API とファイルの中身が同じ
  - 5・7: ブロック 2 のモデル名(bench_20261022-slow-sec-s1)を名指しで確かめる
  - 最後に、動画用の写真を別のセッション(画素 2 倍)で撮り、1 枚にまとめる(ui_candidates_20261023.png)

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python -I v_ui_live2.py

- 走行開始・保存のボタンは押さない。区間の速度上限は ON にしない(確認はキャンセルで閉じる)。
- 念のため、ページを開いた直後に fetch と XMLHttpRequest を包み、GET/HEAD 以外の要求は送らずに記録する(サーバーへは届かない)。
  confirm は記録してキャンセル(false)を返す。開いた直後(包む前)に何も送らないことは段 V-drive の偽のサーバーで確かめ済み。
- 終わったら、サーバーの値(/api/drive/status の params・last)と data/drive_settings.json の sha256 が始める前と同じかを見る。
- 実車・PCA9685・I2C・カメラ・ToF には触れない(センサーの開始の API も呼ばない)。GPU は使わない。
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'http://192.168.55.1:8080'
LOCAL = 'http://127.0.0.1:8080'
GECKO = '/snap/bin/geckodriver'
WD_PORT = 4447
REPORT = Path('/home/jetson/jetformer_reports/prep_20261022')
OUTDIR = REPORT / 'verify_ui'
PHOTO = REPORT / 'ui_candidates_20261023.png'
WORK = OUTDIR / 'V-ui_out'
PROFILE = WORK / 'ff_profile'
SETTINGS = Path('/home/jetson/jetformer/data/drive_settings.json')
RECOMMENDED = Path('/home/jetson/jetformer/data/drive_recommended.json')
CANDS = Path('/home/jetson/jetformer/data/bench_candidates.json')
ELEM = 'element-6066-11e4-a52e-4f735466cecf'
B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
C1 = 'bench_20261022-slow-sec-s1'          # 段 Decide(A10)が選んだ C1 = SLOW-SEC1

res = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'base': BASE, 'checks': [], 'shots': []}


def log(*x):
    print(*x, flush=True)


def check(name, ok, **info):
    res['checks'].append(dict(name=name, ok=bool(ok), **info))
    log(('OK  ' if ok else 'NG  ') + name, json.dumps(info, ensure_ascii=False, default=str)[:600])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


class WD(object):
    def __init__(self):
        self.base, self.sid = 'http://127.0.0.1:%d' % WD_PORT, None

    def req(self, method, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read() or b'{}').get('value')
        except urllib.error.HTTPError as e:
            raise RuntimeError('%s %s: %s %r' % (method, path, e.code, e.read()[:400]))

    def s(self, method, path, body=None, timeout=120):
        return self.req(method, '/session/%s%s' % (self.sid, path), body, timeout)

    def js(self, script, *args):
        return self.s('POST', '/execute/sync', {'script': script, 'args': list(args)})

    def find(self, css):
        return self.s('POST', '/element', {'using': 'css selector', 'value': css})[ELEM]

    def click(self, css):
        """本物のクリック(WebDriver のマウス操作)。押す前に画面の真ん中へ寄せる"""
        eid = self.find(css)
        self.s('POST', '/execute/sync', {'script': "arguments[0].scrollIntoView({block:'center'})", 'args': [{ELEM: eid}]})
        time.sleep(0.2)
        self.s('POST', '/element/%s/click' % eid, {})

    def shot(self, name, full=False):
        b64 = self.s('GET', '/moz/screenshot/full' if full else '/screenshot')
        (OUTDIR / name).write_bytes(base64.b64decode(b64))
        res['shots'].append(name)

    def elem_shot(self, css, name):
        """要素の写真: ページ全体の写真を撮り、要素の範囲で切り出す(窓より高い要素も全部写る)。
        上に貼り付く見出し(header は sticky)は、写す間だけ static にする(見た目だけ。サーバーへは何も送らない)"""
        import io
        from PIL import Image
        eid = self.find(css)
        r = self.s('POST', '/execute/sync', {'script': "const h = document.querySelector('body > header'); if (h) h.style.position = 'static';"
                                                       " const b = arguments[0].getBoundingClientRect();"
                                                       " return [b.left + scrollX, b.top + scrollY, b.width, b.height, devicePixelRatio];",
                                             'args': [{ELEM: eid}]})
        time.sleep(0.3)
        img = Image.open(io.BytesIO(base64.b64decode(self.s('GET', '/moz/screenshot/full'))))
        x, y, w, h, dpr = r
        box = tuple(int(round(v * dpr)) for v in (x - 4, y - 4, x + w + 4, y + h + 4))
        box = (max(box[0], 0), max(box[1], 0), min(box[2], img.width), min(box[3], img.height))
        img.crop(box).save(OUTDIR / name)
        self.s('POST', '/execute/sync', {'script': "const h = document.querySelector('body > header'); if (h) h.style.position = '';", 'args': []})
        res['shots'].append(name)


def cap_snap_scope(pid, mem):
    """snap の Firefox は snap.firefox.geckodriver-*.scope に移り jf-heavy の上限の外に出るので、そのスコープにも上限を付ける"""
    try:
        scope = Path('/proc/%d/cgroup' % pid).read_text().strip().splitlines()[-1].rsplit('/', 1)[-1]
    except OSError as e:
        return {'error': str(e)}
    if not scope.startswith('snap.'):
        return {'scope': scope, 'note': 'snap のスコープではない(jf-heavy の上限の中)'}
    r = subprocess.run(['systemctl', '--user', 'set-property', '--runtime', scope, 'MemoryMax=%s' % mem, 'MemorySwapMax=0'],
                       capture_output=True, text=True)
    return {'scope': scope, 'returncode': r.returncode, 'stderr': r.stderr.strip()}


# ページを開いた直後に入れる見張り: GET/HEAD 以外は送らない・confirm はキャンセル・エラーを集める
GUARD = r"""
if (!window._vui) {
  window._vui = { writes: [], confirms: [], alerts: [], errs: [], cerr: [] };
  const of = window.fetch.bind(window);
  window.fetch = (url, opt) => {
    const m = String((opt && opt.method) || 'GET').toUpperCase();
    if (m !== 'GET' && m !== 'HEAD') {
      window._vui.writes.push({ m, url: String(url), body: opt && opt.body ? String(opt.body) : null });
      return Promise.reject(new Error('V-ui: 書き込みの要求を止めました'));
    }
    return of(url, opt);
  };
  const xo = XMLHttpRequest.prototype.open, xs = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (m, u) { this._vm = String(m).toUpperCase(); this._vu = String(u); return xo.apply(this, arguments); };
  XMLHttpRequest.prototype.send = function (b) {
    if (this._vm !== 'GET' && this._vm !== 'HEAD') { window._vui.writes.push({ m: this._vm, url: this._vu, xhr: true }); return; }
    return xs.apply(this, arguments);
  };
  window.confirm = (msg) => { window._vui.confirms.push(String(msg)); return false; };
  window.alert = (msg) => { window._vui.alerts.push(String(msg)); };
  window.addEventListener('error', (e) => window._vui.errs.push(String(e.message || e)));
  window.addEventListener('unhandledrejection', (e) => window._vui.errs.push('unhandledrejection: ' + String(e.reason && e.reason.message || e.reason)));
  const ce = console.error.bind(console);
  console.error = (...a) => { window._vui.cerr.push(a.map(String).join(' ')); ce(...a); };
}
return true;
"""

STATE = r"""
const q = (s) => document.querySelector(s);
const val = (s) => q(s) ? q(s).value : null;
return {
  tab: q('#tab-drive').classList.contains('active'),
  model: val('#dr-model'), mode: val('#dr-mode'),
  steer: +val('#dr-steer-gain'), thr: +val('#dr-throttle-gain'), trim: +val('#dr-trim'), look: +val('#dr-look'), osc: +val('#dr-osc'),
  rev_us: +val('#dr-rev-us'), rev_s: +val('#dr-rev-s'), speed_cap: +val('#dr-speed-cap'), speed_scale: +val('#dr-speed-scale'),
  ens_j: +val('#dr-ens-j'), ens_m: +val('#dr-ens-m'), tof: val('#dr-tof-guard'), tcp: val('#dr-tcp-mode'),
  recovery: q('#dr-recovery').checked, esc_reverse: q('#dr-esc-reverse').checked,
  zone: q('#dr-zone-caps').checked, zoneDis: q('#dr-zone-caps').disabled,
  zoneSpec: q('#dr-zone-spec').textContent, zoneUnknown: q('#dr-zone-unknown').hidden ? '' : q('#dr-zone-unknown').textContent,
  zoneNote: q('#dr-zone-note').textContent, zonePanelText: q('.zone-panel').innerText,
  record: q('#dr-record').checked, desc: val('#dr-desc'), dry: q('#dr-dry') ? q('#dr-dry').checked : null,
  checkCount: q('#dr-check-count').textContent, checkBase: q('#dr-check-base').textContent,
  checkDisk: q('#dr-check-disk').textContent, checkHeavy: q('#dr-check-heavy').textContent,
  checkItems: [...document.querySelectorAll('.dr-check')].map((c) => c.closest('label').innerText.trim()),
  baseInfo: q('#dr-base-info').textContent,
  baseNote: q('#dr-base-note').classList.contains('hidden') ? '' : q('#dr-base-note-text').textContent,
  pending: Object.keys(window.jetformer.state.drive.pending || {}).sort(),
  card: !q('#bench-card').classList.contains('hidden'), benchHead: q('#bench-head').textContent, benchErr: q('#bench-error').textContent,
  days: [...document.querySelectorAll('#bench-days .bench-day')].map((d) => ({
    text: d.innerText,
    blocks: [...d.querySelectorAll('button')].map((b) => ({ t: b.textContent, dis: b.disabled, title: b.title, active: b.classList.contains('active') })) })),
  why: [...document.querySelectorAll('#bench-days .bench-why')].map((x) => x.textContent),
  candRows: [...document.querySelectorAll('#bench-cands tr')].slice(1).map((r) => [...r.querySelectorAll('td')].map((td) => td.textContent.trim())),
  modelInfo: q('#dr-model-info').innerText,
  modelOpts: [...q('#dr-model').options].map((o) => ({ v: o.value, t: o.textContent, dis: o.disabled })),
  modeOpts: [...q('#dr-mode').options].map((o) => ({ v: o.value, dis: o.disabled })),
  startBtn: q('#btn-dr-start') ? { dis: q('#btn-dr-start').disabled, t: q('#btn-dr-start').textContent } : null,
  baselineDis: q('#btn-dr-baseline').disabled,
  toast: q('#toast') ? q('#toast').textContent : '',
  vui: window._vui,
  scroll: [document.documentElement.scrollWidth, window.innerWidth],
};
"""



# 写真のための読み取り: 要素の範囲(ページの中の CSS px)。sels の外接の四角。cut_to があれば、その要素の下端で切る
RECT = r"""
const els = arguments[0].map((s) => document.querySelector(s)).filter(Boolean);
const rs = els.map((e) => e.getBoundingClientRect());
let l = Math.min(...rs.map((r) => r.left)), t = Math.min(...rs.map((r) => r.top));
let r = Math.max(...rs.map((r) => r.right)), b = Math.max(...rs.map((r) => r.bottom));
if (arguments[1]) { const c = document.querySelector(arguments[1]); if (c) b = c.getBoundingClientRect().bottom; }
return [l, t, r, b, devicePixelRatio, innerHeight];
"""


def photo():
    """動画用の写真(ui_candidates_20261023.png): 別のセッションで画素 2 倍にし、ブロック 2 を押したあとの
    (1) 10/23 の並び (2) 候補の一覧 (3) モデルの欄と区間の速度上限 を 1 枚に並べる。書き込みは 10/21 と同じ見張りで止める"""
    import io
    from PIL import Image, ImageDraw, ImageFont
    gd = wd = None
    glog = WORK / 'geckodriver_photo.log'
    prof = WORK / 'ff_profile_photo'
    parts = {}
    info = {}
    try:
        gd = subprocess.Popen([GECKO, '--port', str(WD_PORT)], stdout=open(glog, 'w'), stderr=subprocess.STDOUT)
        wd = WD()
        for _ in range(60):
            try:
                wd.req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        if prof.exists():
            shutil.rmtree(prof)
        prof.mkdir(parents=True)
        caps = {'capabilities': {'alwaysMatch': {
            'browserName': 'firefox', 'pageLoadStrategy': 'normal', 'unhandledPromptBehavior': 'dismiss',
            'moz:firefoxOptions': {'args': ['-headless', '-profile', str(prof)],
                                   'env': {'MOZ_HEADLESS_WIDTH': '1440', 'MOZ_HEADLESS_HEIGHT': '1800'},
                                   'prefs': {'browser.shell.checkDefaultBrowser': False, 'intl.accept_languages': 'ja',
                                             'layout.css.devPixelsPerPx': '2.0'}}}}}
        v = wd.req('POST', '/session', caps, timeout=180)
        wd.sid = v['sessionId']
        res['photo_memory_cap'] = cap_snap_scope(gd.pid, '2500M')
        wd.s('POST', '/window/rect', {'width': 1440, 'height': 1800})
        wd.s('POST', '/url', {'url': BASE + '/'})
        wd.js(GUARD)
        for _ in range(40):
            if wd.js("return !!(window.jetformer && document.querySelector('button.tab[data-tab=drive]'))"):
                break
            time.sleep(0.25)
        wd.click('button.tab[data-tab="drive"]')
        for _ in range(60):
            if wd.js("return !!(window.jetformer.state.bench && window.jetformer.state.bench.data && document.querySelector('#bench-days button'))"):
                break
            time.sleep(0.25)
        time.sleep(2.5)
        wd.click('#bench-days button[data-bench-day="0"][data-bench-i="1"]')      # ブロック 2(走行は始めない)
        time.sleep(1.5)
        wd.js("document.querySelector('#bench-card details').open = true;"
              " const h = document.querySelector('body > header'); if (h) h.style.position = 'static'; return true;")
        time.sleep(0.5)
        s = wd.js(STATE)
        info = {'model': s['model'], 'zone': s['zone'], 'blocks': [b['t'] for b in s['days'][0]['blocks']],
                'active': [b['t'] for b in s['days'][0]['blocks'] if b['active']],
                'drivable_rows': sum('走れる' in r[3] and '走れない' not in r[3] for r in s['candRows']), 'rows': len(s['candRows']),
                'dpr': wd.js('return devicePixelRatio'), 'vui': s['vui']}
        model_label = "document.querySelector('#dr-model').closest('label')"
        wd.js("%s.id = '_vui_model_label'; return true;" % model_label)       # 写真の範囲を取るための目印(見た目は変わらない)
        crops = (('blocks', ['#bench-card'], '#bench-applied'), ('cands', ['#bench-card details'], None),
                 ('model', ['#_vui_model_label', '#dr-model-info'], None), ('zone', ['.zone-panel'], None))
        for name, sels, cut in crops:
            wd.js("document.querySelector(arguments[0]).scrollIntoView({block:'start'}); window.scrollBy(0, -12); return true;", sels[0])
            time.sleep(0.4)
            l, t, r, b, dpr, ih = wd.js(RECT, sels, cut)
            img = Image.open(io.BytesIO(base64.b64decode(wd.s('GET', '/screenshot'))))
            box = tuple(int(round(v * dpr)) for v in (l - 6, t - 6, r + 6, b + 6))
            box = (max(box[0], 0), max(box[1], 0), min(box[2], img.width), min(box[3], img.height))
            parts[name] = img.crop(box).convert('RGB')
            info['crop_' + name] = [round(x, 1) for x in (l, t, r, b)] + [box[3] - box[1] < (b - t + 12) * dpr - 2]
        s2 = wd.js(STATE)
        info['vui_end'] = s2['vui']
    finally:
        if wd and wd.sid:
            try:
                wd.req('DELETE', '/session/%s' % wd.sid, timeout=60)
            except Exception as e:  # noqa: BLE001
                log('session end:', e)
        if gd:
            gd.terminate()
            try:
                gd.wait(10)
            except subprocess.TimeoutExpired:
                gd.kill()
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)
    res['photo'] = info
    ok_state = (info.get('model') == C1 and info.get('zone') is False and info.get('active') == ['2. SLOW-SEC1']
                and info.get('drivable_rows') == info.get('rows') == 6 and info['vui_end']['writes'] == []
                and info['vui_end']['confirms'] == [] and info['vui_end']['errs'] == [])
    clipped = [k for k in ('blocks', 'cands', 'model', 'zone') if info.get('crop_' + k, [0] * 5)[4]]
    check('13 写真のセッション(画素 2 倍): ブロック 2 を押す → モデル %s・上限 OFF・6 本とも走れる・書き込み 0・エラー 0・切れた写真なし' % C1,
          ok_state and not clipped, clipped=clipped, **{k: v for k, v in info.items() if k != 'vui'})
    # 1 枚にまとめる(左: 並び・中: 候補の一覧・右: モデルの欄と上限)
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    f_title = ImageFont.truetype(font_path, 44, index=0)
    f_sub = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 30, index=0)
    f_lab = ImageFont.truetype(font_path, 32, index=0)
    pad, gap, head = 40, 36, 150
    col1 = [parts['blocks']]
    col2 = [parts['cands']]
    col3 = [parts['model'], parts['zone']]
    labels = ['10/23 の並び(ブロック 2 を押したところ)', '候補の一覧(%d 本とも「走れる」)' % info['drivable_rows'],
              'モデルの欄と区間の速度上限(%s)' % ('OFF' if info['zone'] is False else 'ON')]
    lab_h = 52
    def col_h(c):
        return sum(im.height for im in c) + gap * (len(c) - 1)
    widths = [max(im.width for im in c) for c in (col1, col2, col3)]
    H = head + lab_h + max(col_h(c) for c in (col1, col2, col3)) + pad
    W = pad * 2 + sum(widths) + gap * 2
    canvas = Image.new('RGB', (W, H), (236, 239, 243))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 26), '走行タブ「ベンチマークの候補」(本物の画面)', font=f_title, fill=(20, 24, 31))
    d.text((pad, 88), '%s 撮影・%s・走行開始と保存は押していない' % (time.strftime('%Y-%m-%d %H:%M'), BASE), font=f_sub, fill=(75, 85, 99))
    x = pad
    for c, w, lab in zip((col1, col2, col3), widths, labels):
        d.text((x, head), lab, font=f_lab, fill=(17, 94, 89))
        y = head + lab_h
        for im in c:
            canvas.paste(im, (x, y))
            y += im.height + gap
        x += w + gap
    canvas.save(PHOTO, optimize=True)
    for k, im in parts.items():
        im.save(OUTDIR / ('photo_part_%s.png' % k))
    res['photo_file'] = {'path': str(PHOTO), 'size': [W, H]}
    log('photo', PHOTO, W, H)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    cands = json.loads(CANDS.read_text(encoding='utf-8'))
    runs = {c['id']: c['run'] for c in cands['candidates']}
    labels = {c['id']: c['label'] for c in cands['candidates']}
    day1 = cands['days'][0]
    tpl = cands['description_template']
    base_p = cands['baseline']['params']

    sha0 = {p.name: sha(p) for p in (SETTINGS, RECOMMENDED, CANDS)}
    st0 = get(LOCAL, '/api/drive/status')
    s0 = get(LOCAL, '/api/status')
    res['before'] = {'sha256': sha0, 'params': st0['params'], 'last': st0.get('last'), 'running': st0['running'],
                     'recording': st0['recording'], 'sensors_running': s0['sensors_running'], 'driving': s0['driving']}
    check('0 前提: 走行・収集していない・センサーは止まったまま', not st0['running'] and not st0['recording'] and not s0['driving']
          and not s0['sensors_running'], **res['before'])
    api = get(LOCAL, '/api/bench/candidates')
    st_mtime = CANDS.stat().st_mtime
    started = subprocess.run(['systemctl', 'show', 'jetformer-web', '-p', 'ActiveEnterTimestampMonotonic', '-p', 'ActiveEnterTimestamp',
                              '-p', 'MainPID'], capture_output=True, text=True).stdout.strip().splitlines()
    res['service'] = started
    want_cands = [(c['id'], c['run']) for c in cands['candidates']]
    got_cands = [(c['id'], c['run']) for c in api['candidates']]
    want_days = [[(b['block'], b['id']) for b in d['blocks']] for d in cands['days']]
    got_days = [[(b['block'], b['id']) for b in d['blocks']] for d in api['days']]
    check('0 GET /api/bench/candidates は今のファイルと同じ中身(候補・並び)・ok・警告 0・8 ブロックとも drivable',
          api['ok'] and not api['warnings'] and not api['error'] and got_cands == want_cands and got_days == want_days
          and all(b['drivable'] for b in api['days'][0]['blocks']) and all(c['drivable'] for c in api['candidates']),
          got_days=got_days, got_cands=got_cands, warnings=api['warnings'], error=api['error'])
    check('0 API は呼ばれるたびにファイルを読み直す(サーバーの起動 10/21 20:18 < ファイルの更新 %s)'
          % time.strftime('%m/%d %H:%M:%S', time.localtime(st_mtime)),
          got_days == want_days and api['days'][0]['blocks'][1]['run'] == C1, service=started, file_mtime=st_mtime)
    lan = get(BASE, '/api/drive/status')
    check('0 LAN の住所 192.168.55.1:8080 が同じサーバーを返す', lan['params'] == st0['params'] and 'zone_caps_spec' in lan,
          params=lan['params'])

    gd = wd = None
    glog = WORK / 'geckodriver.log'
    try:
        gd = subprocess.Popen([GECKO, '--port', str(WD_PORT)], stdout=open(glog, 'w'), stderr=subprocess.STDOUT)
        wd = WD()
        for _ in range(60):
            try:
                wd.req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        if PROFILE.exists():
            shutil.rmtree(PROFILE)
        PROFILE.mkdir(parents=True)
        caps = {'capabilities': {'alwaysMatch': {
            'browserName': 'firefox', 'pageLoadStrategy': 'normal', 'unhandledPromptBehavior': 'dismiss',
            'moz:firefoxOptions': {'args': ['-headless', '-profile', str(PROFILE)],
                                   'env': {'MOZ_HEADLESS_WIDTH': '1440', 'MOZ_HEADLESS_HEIGHT': '1000'},
                                   'prefs': {'devtools.console.stdout.content': True, 'browser.shell.checkDefaultBrowser': False,
                                             'intl.accept_languages': 'ja'}}}}}
        v = wd.req('POST', '/session', caps, timeout=180)
        wd.sid = v['sessionId']
        res['browser'] = {k: v['capabilities'].get(k) for k in ('browserVersion', 'moz:headless', 'moz:profile')}
        res['memory_cap'] = cap_snap_scope(gd.pid, '2500M')
        wd.s('POST', '/window/rect', {'width': 1440, 'height': 1000})

        # ---------------------------------------------------------------- 1. 開く → 走行タブ
        wd.s('POST', '/url', {'url': BASE + '/'})
        wd.js(GUARD)
        for _ in range(40):
            if wd.js("return !!(window.jetformer && document.querySelector('button.tab[data-tab=drive]'))"):
                break
            time.sleep(0.25)
        wd.click('button.tab[data-tab="drive"]')
        for _ in range(60):
            if wd.js("return !!(window.jetformer.state.bench && window.jetformer.state.bench.data && document.querySelector('#bench-days button'))"):
                break
            time.sleep(0.25)
        time.sleep(2.5)                     # 状態の同期(1 秒ごと)を 2 回待つ
        s = wd.js(STATE)
        res['initial'] = {k: s[k] for k in ('model', 'mode', 'steer', 'zone', 'zoneDis', 'checkCount', 'checkBase', 'benchHead')}
        check('1 走行タブが開く・最初は推奨のモデル B・保存値のステア %s(画面に出すだけ)' % st0['params']['steer_gain'],
              s['tab'] and s['model'] == B and s['steer'] == st0['params']['steer_gain'], model=s['model'], steer=s['steer'], info=s['modelInfo'][:200])
        check('1 開いた直後: 書き込みの要求 0・確認 0', s['vui']['writes'] == [] and s['vui']['confirms'] == [], vui=s['vui'])
        wd.shot('ui_drive_initial.png')
        wd.shot('ui_drive_initial_full.png', full=True)

        # ---------------------------------------------------------------- 2. 区間の速度上限のスイッチ(既定 OFF・注意書き)
        check('2 上限のスイッチ: OFF・押せる(再起動後のサーバーは対応)', s['zone'] is False and s['zoneDis'] is False,
              zone=s['zone'], dis=s['zoneDis'], note=s['zoneNote'])
        check('2 上限: 3 区間の値(v1.1 R1 2.5・GM 2.2・DE 3.0)と 4 つめを出す',
              'R1 2.5' in s['zoneSpec'] and 'GM 2.2' in s['zoneSpec'] and 'DE 3.0' in s['zoneSpec'] and '4 つめ' in s['zoneSpec'],
              spec=s['zoneSpec'])
        check('2 上限: 注意書き(G1 合格・ユーザーの了承のあと・既定 OFF・起動のたびに OFF)',
              'G1 に合格し、ユーザーが了承したあと' in s['zonePanelText'] and '既定は OFF' in s['zonePanelText']
              and '起動するたびに OFF' in s['zonePanelText'], text=s['zonePanelText'][:400])
        check('2 上限: 4 つめの上限(unknown 3.0 m/s)の赤い注意を出す', '4 つめの上限' in s['zoneUnknown'] and '3.0 m/s' in s['zoneUnknown'],
              unknown=s['zoneUnknown'][:200])
        wd.elem_shot('.zone-panel', 'ui_zone_switch_off.png')
        # ON を押して、確認でキャンセル(サーバーへ送らない・OFF のまま)
        wd.click('#dr-zone-caps')
        time.sleep(1.5)
        s = wd.js(STATE)
        msg = s['vui']['confirms'][-1] if s['vui']['confirms'] else ''
        check('2 上限 ON を押すと確認が出る(G1・了承・4 つめ)', 'G1 に合格し、ユーザーが了承したあと' in msg and '4 つめ' in msg and 'R1 2.5' in msg,
              confirm=msg[:400])
        check('2 確認をキャンセル → OFF のまま・何も送らない', s['zone'] is False and s['vui']['writes'] == [], zone=s['zone'], writes=s['vui']['writes'])
        st = get(LOCAL, '/api/drive/status')
        check('2 サーバーの zone_caps は 0 のまま', st['params'].get('zone_caps') == 0, zone_caps=st['params'].get('zone_caps'))

        # ---------------------------------------------------------------- 3. 走る前のチェック
        check('3 チェックリスト: 6 項目・0 / 6', len(s['checkItems']) == 6 and s['checkCount'] == '0 / 6', items=s['checkItems'],
              count=s['checkCount'])
        check('3 チェックの手がかり: 空き容量・学習の有無・基準との違い(ステア 1.50、基準 1.00)',
              'GB' in s['checkDisk'] and '学習は動いていません' in s['checkHeavy'] and '基準と違う' in s['checkBase']
              and 'ステア 1.50(基準 1.00)' in s['checkBase'], disk=s['checkDisk'], heavy=s['checkHeavy'], base=s['checkBase'])
        wd.elem_shot('#dr-prep', 'ui_prep_before.png')
        for i in range(6):                  # 6 つのチェックを本物のクリックで押す
            eid = wd.js("const c = document.querySelectorAll('.dr-check')[%d]; c.scrollIntoView({block:'center'}); return c;" % i)
            wd.s('POST', '/element/%s/click' % eid[ELEM], {})
        time.sleep(0.3)
        s = wd.js(STATE)
        check('3 6 つ押すと 6 / 6・送らない', s['checkCount'] == '6 / 6' and s['vui']['writes'] == [], count=s['checkCount'])
        wd.elem_shot('#dr-prep', 'ui_prep_checked.png')
        wd.click('#btn-dr-check-clear')
        time.sleep(0.3)
        s = wd.js(STATE)
        check('3 全部外す → 0 / 6', s['checkCount'] == '0 / 6', count=s['checkCount'])

        # ---------------------------------------------------------------- 4. 基準の設定(画面に入れるだけ・保存しない)
        wd.click('#btn-dr-baseline')
        time.sleep(3.0)                     # 同期で戻らないか(1 秒ごと)を 3 回待つ
        s = wd.js(STATE)
        got = {'model': s['model'], 'mode': s['mode'], 'steer_gain': s['steer'], 'throttle_gain': s['thr'], 'steer_trim_us': s['trim'],
               'lookahead': s['look'], 'osc_percent': s['osc'], 'rev_us': s['rev_us'], 'rev_s': s['rev_s'], 'speed_cap': s['speed_cap'],
               'speed_scale': s['speed_scale'], 'ensemble_j': s['ens_j'], 'ensemble_m': s['ens_m'], 'tof_guard': int(s['tof']),
               'tcp_mode': int(s['tcp']), 'recovery': int(s['recovery']), 'esc_reverse': int(s['esc_reverse']), 'zone_caps': int(s['zone'])}
        want = dict(base_p)
        want.update(model=runs['B'], mode=str(cands['baseline']['mode_id']))
        diff = {k: (got.get(k), want[k]) for k in want if (got.get(k) != want[k] if isinstance(want[k], str)
                                                             else abs(float(got.get(k, 'nan')) - float(want[k])) > 1e-6)}
        res['baseline_fields'] = got
        check('4 基準の設定: 3.4 のとおり(B・ジャンル 5・ステア 1.0・速度 1.0・先読み 3・ToF 1・重ね合わせ 0・speed_cap 0・上限 OFF、ほかは DEFAULT_PARAMS)',
              not diff, diff=diff, got=got)
        check('4 基準の設定: 送らない(書き込み 0・確認 0)・pending はステアだけ・同期で戻らない',
              s['vui']['writes'] == [] and s['vui']['confirms'][1:] == [] and s['pending'] == ['steer_gain'] and s['steer'] == 1.0,
              writes=s['vui']['writes'], pending=s['pending'])
        check('4 基準の設定: 変えた所とまだ送っていないことを出す', 'ステア 1.50 → 1.00' in s['baseNote'] and 'まだサーバーへは送っていません' in s['baseNote'],
              note=s['baseNote'])
        check('4 チェックの「基準」の行が「同じ」になる', '画面の値は基準と同じ' in s['checkBase'], base=s['checkBase'])
        check('4 基準の説明(基準の出どころ data/bench_candidates.json)', 'data/bench_candidates.json' in s['baseInfo'] and 'ジャンル 5' in s['baseInfo'],
              info=s['baseInfo'])
        st = get(LOCAL, '/api/drive/status')
        check('4 サーバーの値は変わらない(ステア 1.5 のまま)・保存値の sha256 も同じ',
              st['params'] == st0['params'] and sha(SETTINGS) == sha0[SETTINGS.name], params=st['params'])
        wd.elem_shot('#dr-prep', 'ui_baseline.png')
        wd.shot('ui_baseline_view.png')

        # ---------------------------------------------------------------- 5. 候補のパネル
        s = wd.js(STATE)
        check('5 候補のパネルが出る・bench 名・手順の sha256・作成時刻', s['card'] and 'real_bench_20261023' in s['benchHead']
              and cands['protocol_sha256'][:12] in s['benchHead'] and cands['created_at'] in s['benchHead'] and not s['benchErr'].strip(),
              head=s['benchHead'], err=s['benchErr'])
        want_blocks = ['%d. %s' % (b['block'], b['id']) for b in day1['blocks']]
        d1 = s['days'][0] if s['days'] else {'blocks': [], 'text': ''}
        check('5 10/23 のブロックが data/bench_candidates.json の順(B・SLOW-SEC1・ENS・B・AUG・ACT・WGT・B)',
              [b['t'] for b in d1['blocks']] == want_blocks, got=[b['t'] for b in d1['blocks']], want=want_blocks)
        b2 = d1['blocks'][1] if len(d1['blocks']) > 1 else {'t': '', 'title': ''}
        check('5 ブロック 2 = SLOW-SEC1(遅く入る型)・モデル名 %s がボタンの説明に出る' % C1,
              b2['t'] == '2. SLOW-SEC1' and C1 in b2['title'] and runs.get('SLOW-SEC1') == C1, t=b2['t'], title=b2['title'])
        check('5 8 ブロックとも押せる(走れないブロックの理由 0)', all(not b['dis'] for b in d1['blocks']) and s['why'] == [],
              dis=[b['t'] for b in d1['blocks'] if b['dis']], why=s['why'])
        check('5 ブロックの説明(title)に名前とモデル', all(labels[b['id']] in x['title'] and runs[b['id']] in x['title']
                                                          for b, x in zip(day1['blocks'], d1['blocks'])),
              titles=[x['title'] for x in d1['blocks']])
        d2 = s['days'][1] if len(s['days']) > 1 else {'blocks': [], 'text': ''}
        check('5 10/24 はブロック無し・「10/23 の夜に決める」', d2['blocks'] == [] and '10/23 の夜に決める' in d2['text'], text=d2['text'][:200])
        rows = {r[0]: r for r in s['candRows']}
        check('5 候補の一覧: 6 本(B・SLOW-SEC1・ENS・AUG・ACT・WGT)すべて「走れる」・危なさ(ENS は mid)・古い SLOW は無い',
              sorted(rows) == sorted(runs) and all('走れる' in r[3] and '走れない' not in r[3] for r in rows.values())
              and rows.get('ENS', ['', '', ''])[2] == 'mid' and 'SLOW' not in rows and C1 in rows.get('SLOW-SEC1', ['', ''])[1],
              rows=s['candRows'])
        wd.js("document.querySelector('#bench-card details').open = true")
        time.sleep(0.4)
        wd.elem_shot('#bench-card', 'ui_candidates.png')

        # ---------------------------------------------------------------- 6. 走行の一覧(モデルの選択肢)で候補が走れる
        opts = {o['v']: o for o in s['modelOpts']}
        check('6 モデルの一覧に候補 6 本があり、どれも選べる(走れません無し)',
              all(r in opts and not opts[r]['dis'] and '走れません' not in opts[r]['t'] for r in runs.values()),
              opts=[opts.get(r, {'v': r, 'missing': True}) for r in runs.values()])
        check('6 推奨は B のまま(★推奨は B だけ)', [o['v'] for o in s['modelOpts'] if '★推奨' in o['t']] == [B],
              rec=[o['v'] for o in s['modelOpts'] if '★推奨' in o['t']])

        # ---------------------------------------------------------------- 7. ブロックを押す → モデル・説明欄の雛形・収集 ON(走らない)
        blocks_res = []
        for i, b in enumerate(day1['blocks']):
            wd.click('#bench-days button[data-bench-day="0"][data-bench-i="%d"]' % i)
            time.sleep(1.2)
            s = wd.js(STATE)
            desc_want = tpl.replace('{date}', day1['date']).replace('{block}', str(b['block'])).replace('{id}', b['id']).replace('{run}', runs[b['id']])
            mode_ok = any(o['v'] == '5' and not o['dis'] for o in s['modeOpts'])
            ok = (s['model'] == runs[b['id']] and s['desc'] == desc_want and s['record'] and s['steer'] == 1.0 and s['mode'] == '5'
                  and mode_ok and s['zone'] is False and '走行できません' not in s['modelInfo']
                  and [x['t'] for x in s['days'][0]['blocks'] if x['active']] == [want_blocks[i]])
            blocks_res.append({'block': b['block'], 'id': b['id'], 'ok': ok, 'model': s['model'], 'desc': s['desc'], 'record': s['record'],
                               'steer': s['steer'], 'mode': s['mode'], 'policy': s['modelInfo'].split('\n')[0][:120]})
            if i == 1:
                res['block2'] = {'model': s['model'], 'modelInfo': s['modelInfo'], 'desc': s['desc'],
                                 'opt': [o for o in s['modelOpts'] if o['v'] == s['model']]}
                check('7 ブロック 2 を押す → モデルの欄が %s・説明欄に cand=SLOW-SEC1 model=%s・走行できませんは出ない' % (C1, C1),
                      s['model'] == C1 and 'cand=SLOW-SEC1' in s['desc'] and ('model=%s' % C1) in s['desc']
                      and '走行できません' not in s['modelInfo'], **res['block2'])
                wd.elem_shot('#bench-card', 'ui_block2_slowsec1.png')
                wd.js("document.querySelector('#dr-model').scrollIntoView({block:'start'})")
                time.sleep(0.3)
                wd.shot('ui_block2_slowsec1_fields.png')
            if i == 2:
                wd.js("document.querySelector('#dr-model').scrollIntoView({block:'start'})")
                time.sleep(0.3)
                wd.shot('ui_block3_ens_fields.png')
        res['blocks'] = blocks_res
        check('7 8 ブロックとも: そのモデル・ジャンル 5・ステア 1.0・説明欄の雛形・収集 ON・上限 OFF・押したブロックを示す',
              all(x['ok'] for x in blocks_res), bad=[x for x in blocks_res if not x['ok']])
        s = wd.js(STATE)
        check('7 ブロックを押しても書き込み 0・確認 0(走行は始まらない)', s['vui']['writes'] == [] and len(s['vui']['confirms']) == 1,
              writes=s['vui']['writes'], confirms=len(s['vui']['confirms']))
        st = get(LOCAL, '/api/drive/status')
        check('7 サーバーは走っていない・値は始める前と同じ', not st['running'] and st['params'] == st0['params'] and st.get('last') == st0.get('last'),
              running=st['running'], params=st['params'])

        # ---------------------------------------------------------------- 8. この日のステア 1.3・取り消す
        wd.js("const e = document.querySelector('#dr-base-steer'); e.value = '1.3'; e.dispatchEvent(new Event('change'))")
        time.sleep(0.5)
        wd.click('#btn-dr-baseline')
        time.sleep(1.5)
        s = wd.js(STATE)
        check('8 この日のステア 1.30 で基準の設定 → ステア 1.3・モデル B', s['steer'] == 1.3 and s['model'] == B and '1.30(この日)' in s['baseInfo'],
              steer=s['steer'], model=s['model'], info=s['baseInfo'][:160])
        wd.js("const e = document.querySelector('#dr-base-steer'); e.value = ''; e.dispatchEvent(new Event('change'))")
        wd.click('#btn-dr-base-undo')
        time.sleep(2.0)
        s = wd.js(STATE)
        check('8 取り消す → サーバーの値(ステア %s)・モデル B・説明欄は空・pending 0' % st0['params']['steer_gain'],
              s['steer'] == st0['params']['steer_gain'] and s['model'] == B
              and s['desc'] == '' and s['pending'] == [] and s['record'] is False, steer=s['steer'], model=s['model'], desc=s['desc'],
              pending=s['pending'], record=s['record'])

        # ---------------------------------------------------------------- 9. エラー(ふつうの幅のセッション)
        s = wd.js(STATE)
        res['vui_final'] = s['vui']
        check('9 JavaScript のエラー・console.error なし', s['vui']['errs'] == [] and s['vui']['cerr'] == [],
              errs=s['vui']['errs'], cerr=s['vui']['cerr'])
        check('9 全体で書き込みの要求 0(見張りが止めたものも 0)・alert 0', s['vui']['writes'] == [] and s['vui']['alerts'] == [],
              writes=s['vui']['writes'], alerts=s['vui']['alerts'])

        # ---------------------------------------------------------------- 10. スマホの幅(同じ住所の iframe 390 px)
        # ヘッドレスの Firefox は窓を 500 px より狭くできないので、同じ住所(192.168.55.1:8080)の軽いファイルを開き、
        # その中に幅 390 px の iframe で管理画面を開く(同じ住所なので中を読める)
        wd.s('POST', '/url', {'url': BASE + '/static/style.css'})
        wd.js("document.documentElement.innerHTML = '<body style=\"margin:0;background:#fff\"><iframe id=\"ph\" src=\"/\" '"
              " + 'style=\"width:390px;height:844px;border:0;display:block\"></iframe></body>'; return true;")
        for _ in range(60):
            if wd.js("const w = document.querySelector('#ph').contentWindow; return !!(w && w.jetformer && w.document.querySelector('button.tab[data-tab=drive]'))"):
                break
            time.sleep(0.25)
        wd.js("const w = document.querySelector('#ph').contentWindow; return (new w.Function(arguments[0]))()", GUARD)
        wd.js("document.querySelector('#ph').contentWindow.document.querySelector('button.tab[data-tab=drive]').click()")
        for _ in range(60):
            if wd.js("const w = document.querySelector('#ph').contentWindow; return !!(w.jetformer.state.bench && w.jetformer.state.bench.data && w.document.querySelector('#bench-days button'))"):
                break
            time.sleep(0.25)
        time.sleep(2.5)
        PH = "const w = document.querySelector('#ph').contentWindow; return (new w.Function(arguments[0]))()"
        s = wd.js(PH, STATE)
        over = wd.js(PH, "const W = document.documentElement.clientWidth; return [...document.querySelectorAll('body *')]"
                         ".filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.right > W + 0.5; })"
                         ".map((e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (typeof e.className === 'string' && e.className ? '.' + e.className.trim().split(/\\s+/).join('.') : ''))")
        res['phone_page_overflow'] = {'scroll_inner': s['scroll'], 'elements': over,
                                      'header_height': wd.js(PH, "return Math.round(document.querySelector('body > header').getBoundingClientRect().height)"),
                                      'viewport_height': wd.js(PH, "return window.innerHeight")}
        check('10 スマホの幅(390 px): ページの横のはみ出しは見出しのタブの列(nav)だけ(前からのもの。走行タブの変更ではない)',
              s['scroll'][1] <= 400 and set(over) <= {'nav', 'button.tab', 'button.tab.active'}, scroll=s['scroll'], over=over)
        check('10 スマホの幅: 候補のパネル・8 ブロック・上限 OFF が出る', s['card'] and len(s['days'][0]['blocks']) == 8 and s['zone'] is False,
              card=s['card'], zone=s['zone'])
        wide = wd.js(PH, "return [...document.querySelectorAll('#tab-drive *')].filter((e) => e.getBoundingClientRect().right > window.innerWidth + 1)"
                         ".slice(0, 8).map((e) => e.tagName + (e.id ? '#' + e.id : '') + '.' + e.className)")
        check('10 スマホの幅: 走行タブの中に幅からはみ出す要素なし', wide == [], wide=wide)
        for name, css in (('ui_phone_prep.png', '#dr-prep'), ('ui_phone_candidates.png', '#bench-card'), ('ui_phone_zone.png', '.zone-panel')):
            wd.js(PH, "document.querySelector('%s').scrollIntoView({block:'start'}); window.scrollBy(0, -(document.querySelector('body > header').getBoundingClientRect().height + 8)); return true;" % css)
            time.sleep(0.5)
            wd.elem_shot('#ph', name)
        s = wd.js(PH, STATE)
        check('10 スマホの幅: 書き込み 0・エラー 0', s['vui']['writes'] == [] and s['vui']['errs'] == [] and s['vui']['cerr'] == [],
              vui=s['vui'])
    except Exception as e:  # noqa: BLE001
        res['exception'] = repr(e)
        log('EXCEPTION', repr(e))
        try:
            wd.shot('ui_exception.png')
        except Exception:  # noqa: BLE001
            pass
    finally:
        if wd and wd.sid:
            try:
                wd.req('DELETE', '/session/%s' % wd.sid, timeout=60)
            except Exception as e:  # noqa: BLE001
                log('session end:', e)
        if gd:
            gd.terminate()
            try:
                gd.wait(10)
            except subprocess.TimeoutExpired:
                gd.kill()
        time.sleep(1)
        shutil.rmtree(PROFILE, ignore_errors=True)
    # Firefox の標準出力(geckodriver.log)に出た、このページの JavaScript のエラー(読み込みの途中のものも含む)
    try:
        lines = glog.read_text(errors='replace').splitlines()
    except OSError:
        lines = []
    jserr = [x for x in lines if ('JavaScript error' in x or 'console.error' in x or 'TypeError' in x or 'ReferenceError' in x)
             and ('192.168.55.1' in x or 'app.js' in x)]
    res['firefox_js_errors'] = jserr[:20]
    check('11 Firefox のログ(読み込みの途中を含む)に、このページの JavaScript のエラーなし', not jserr, lines=jserr[:5])
    try:
        photo()
    except Exception as e:  # noqa: BLE001
        res['photo_exception'] = repr(e)
        log('PHOTO EXCEPTION', repr(e))
        check('13 写真(動画用)', False, exception=repr(e))
    st1 = get(LOCAL, '/api/drive/status')
    s1 = get(LOCAL, '/api/status')
    sha1 = {p.name: sha(p) for p in (SETTINGS, RECOMMENDED, CANDS)}
    res['after'] = {'sha256': sha1, 'params': st1['params'], 'last': st1.get('last'), 'running': st1['running'],
                    'sensors_running': s1['sensors_running']}
    check('12 終わり: サーバーの値・前回のモデル・保存値/推奨/候補の sha256 が始める前と同じ・センサーは止まったまま',
          st1['params'] == st0['params'] and st1.get('last') == st0.get('last') and sha1 == sha0 and not st1['running']
          and not s1['sensors_running'], **res['after'])
    left = subprocess.run(['pgrep', '-af', 'geckodriver --port %d' % WD_PORT], capture_output=True, text=True).stdout.strip()
    res['leftover'] = left
    res['ok'] = bool(res['checks']) and all(c['ok'] for c in res['checks']) and 'exception' not in res
    res['passed'] = sum(c['ok'] for c in res['checks'])
    res['total'] = len(res['checks'])
    res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (WORK / 'v_ui_live.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    log('%s %d/%d' % ('ALL OK' if res['ok'] else 'SOME NG', res['passed'], res['total']))
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
