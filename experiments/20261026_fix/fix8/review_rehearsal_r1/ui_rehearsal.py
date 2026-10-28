"""反証 8-1 回目・係 rehearsal(通し稽古): 手順書 runbook_20261103.md の画面の手順を、本物の app.js と index.html を
node の台本(tests/fixtures/app_sim_fix7_r2.js の写し sim_rehearsal.js。偽のサーバー)で頭から流す。両方 ON の日の形。
車・本物のサーバーには触れない(偽のサーバーの params は本物の GET /api/drive/status の値 out/live_status.json から始める)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B ui_rehearsal.py

出力: out/ui_<場面>.json(操作ごとの結果)と out/ui_rehearsal.txt(読み上げに使う文の抜き書き)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

STATIC = JF / 'jetformer' / 'web' / 'static'
SIM = HERE / 'sim_rehearsal.js'
OUT = HERE / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal/ui'
LIVE = json.loads((OUT / 'live_status.json').read_text(encoding='utf-8'))
SERVER = dict(LIVE['params'])                     # 本物のサーバーの今の値(10/28 00:53:23 の起動のあと。全部既定)
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}


def cands(today='2026-11-03'):
    c = bench.load_candidates(os.environ.get('REH_BENCH') or None)       # REH_BENCH: 作り物の家の bench_candidates.json(24〜27 を足した形)
    di = [i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03'][0]
    return dict(c, today=today), di


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def run(name, steps, today='2026-11-03', server=None, **extra):
    c, di = cands(today)
    TMP.mkdir(parents=True, exist_ok=True)
    inp = TMP / ('%s.json' % name)
    d = {'bench': c, 'server_params': server if server is not None else SERVER, 'steps': steps(c, di), 'server_client_version': 9,
         'confirm': True, 'boot_at': 1000.0, 'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=%d' % int((STATIC / 'app.js').stat().st_mtime), 'server_boot_check': True}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run(['node', str(SIM), str(STATIC / 'app.js'), str(STATIC / 'index.html'), str(inp)], capture_output=True,
                       text=True, timeout=300, env=dict(os.environ, TZ='Asia/Tokyo'))
    if r.returncode != 0:
        raise SystemExit('%s: %s' % (name, r.stderr[-3000:]))
    out = json.loads(r.stdout.strip().splitlines()[-1])
    (OUT / ('ui_%s.json' % name)).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return out


LOOK = [['read', 'client_ver'], ['el', '#dr-ceil-spec', 'text'], ['read', 'ceil'], ['read', 'auto'], ['read', 'launch'],
        ['el', '#dr-launch-boost', 'disabled'], ['el', '#dr-launch-note', 'html'], ['el', '#dr-launch-spec', 'text'],
        ['el', '#dr-rec-back-guard', 'checked'], ['el', '#dr-rec-back-guard', 'disabled'],
        ['el', '#dr-rec-hold-keep', 'checked'], ['el', '#dr-rec-hold-keep', 'disabled'], ['el', '#dr-rec-hold-keep-note', 'html'],
        ['el', '#dr-launch-no-kick', 'checked'], ['el', '#dr-opt-rec-missing', 'text'], ['el', '#dr-opt-launch-missing', 'text'],
        ['read', 'gain'], ['read', 'gain_text'], ['read', 'trim'], ['read', 'record'], ['read', 'dry'], ['read', 'desc']]


def stop():
    return [['running', False], ['poll']]


def book(c, di, no, gain_clicks=0, switches=(), desc_add=' battery=#A V_before=8.31 check=合格', api=None, baseline=True):
    """4 章の 1 本: 基準の設定 → ベンチマーク N → ゲイン(±0.01 を押す回数)→ 押したあとに入れるもの → 説明欄 → ▶"""
    s = []
    if baseline:
        s += [['baseline'], ['read', 'note']]
    s += [['pick', di, blk(c, di, no)], ['read', 'note'], ['read', 'desc'], ['read', 'record'], ['read', 'dry'],
          ['el', '#dr-launch-note', 'html'], ['el', '#dr-ceil-note', 'html'], ['read', 'gain']]
    btn = '#btn-dr-tg-up' if gain_clicks > 0 else '#btn-dr-tg-down'
    for _ in range(abs(gain_clicks)):
        s.append(['click', btn])
    s.append(['read', 'gain_text'])
    for a in (api or []):
        s += [['api', a], ['poll']]
    for sw in switches:
        s.append(['check', sw, True])
    s += [['poll'], ['read', 'desc']]
    js = ("{ const e = document.querySelector('#dr-desc'); e.value = e.value.replace('battery=#__', 'battery=#A')"
          ".replace('V_before=__', 'V_before=8.31').replace('check=__', 'check=合格'); e.value }")
    s += [['eval', js]]
    s += [['start'], ['read', 'server']]
    return s


def scenes():
    S = {}
    # ---- 3 章のはじめ: Jetson の日付が 11/02 のまま(NTP 前)
    S['date_wrong'] = (lambda c, di: [['el', '#dr-bench-pick', 'html'], ['read', 'bench_error'], ['tapday', di, blk(c, di, 1)],
                                      ['read', 'desc']], {'today': '2026-11-02'})
    # ---- 3 章: 朝(読み直した画面)・基準の設定・dry run・発進の前進の確かめ・試験の選択肢
    def morning(c, di):
        s = [['el', '#dr-bench-pick', 'html']] + LOOK
        s += [['baseline'], ['read', 'check_base'], ['read', 'note']] + LOOK
        # dry run 30 秒(「モーターに出力しない」)→ 止める → 外す
        s += [['check', '#dr-dry', True], ['start'], ['read', 'toasts']] + stop()
        s += [['check', '#dr-dry', False]]
        # 発進の前進の確かめ(3 章の 1〜5): 基準の設定 → 収集を外す → 説明欄 → 発進の前進 ON → dry を外す → ▶
        s += [['baseline'], ['check', '#dr-record', False], ['read', 'desc'], ['check', '#dr-launch-boost', True], ['check', '#dr-dry', False],
              ['start']] + stop() + LOOK
        # 試験の選択肢: 残すを先に押してみる → 見張り ON → 残す ON
        s += [['check', '#dr-rec-hold-keep', True], ['read', 'toasts'], ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True],
              ['poll']] + LOOK + [['read', 'check_base'], ['read', 'server']]
        return s
    S['morning'] = (morning, {})
    # ---- 3 章: dry run を外し忘れてブロックを選ぶ / ブロックの本で dry のまま ▶
    def dry_forgot(c, di):
        s = [['check', '#dr-dry', True], ['start']] + stop()
        s += [['baseline'], ['pick', di, blk(c, di, 1)], ['read', 'note'], ['read', 'dry'],
              ['check', '#dr-dry', True], ['confirm', False], ['start'], ['confirm', True], ['read', 'dry']]
        return s
    S['dry_forgot'] = (dry_forgot, {})
    # ---- 試験の選択肢の API(残すだけ)
    S['api_keep_only'] = (lambda c, di: [['api', {'rec_hold_keep': 1}], ['api', {'rec_back_guard': 1, 'rec_hold_keep': 1}],
                                         ['api', {'rec_back_guard': 0}], ['read', 'server']], {})
    # ---- 回 1〜4 の通し(両方 ON の日。朝に見張り → 残す ON・発進の前進の確かめは収集を外した)
    def day(c, di):
        s = [['baseline'], ['check', '#dr-record', False], ['check', '#dr-launch-boost', True], ['start']] + stop()
        s += [['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['poll']]
        # 回 1: 1-1(0.92)・1-2(0.92)・1-3・1-4(天井 ON)・1-5
        s += book(c, di, 1, gain_clicks=2) + stop()
        s += book(c, di, 2, gain_clicks=2) + stop()
        s += book(c, di, 3, gain_clicks=2) + stop()
        s += book(c, di, 4, gain_clicks=2, switches=['#dr-speed-ceiling']) + stop()
        s += book(c, di, 5, gain_clicks=2) + stop()
        # 回 2: 2-1(天井)・2-2・2-3・2-4(天井 + 自動)・2-5(API 4.2 → 天井)・2-6(API 4.1 → 天井)
        s += book(c, di, 6, gain_clicks=5, switches=['#dr-speed-ceiling']) + stop()
        s += book(c, di, 7, gain_clicks=2) + stop()
        s += book(c, di, 8, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-speed-auto']) + stop()
        s += book(c, di, 9, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-speed-auto']) + stop()
        s += book(c, di, 10, gain_clicks=5, api=[{'ceil_left_mps': 4.2}], switches=['#dr-speed-ceiling']) + stop()
        s += book(c, di, 11, gain_clicks=5, api=[{'ceil_left_mps': 4.1}], switches=['#dr-speed-ceiling']) + stop()
        # 回 3: 3-1 OFF・3-2 ON・(car_check の比 > 1.3 → API launch_us 200)・3-3 OFF・3-4 ON(200)
        s += book(c, di, 12, gain_clicks=5, switches=['#dr-speed-ceiling']) + stop()
        s += book(c, di, 13, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-launch-boost']) + stop()
        s += [['api', {'launch_us': 200}], ['poll'], ['el', '#dr-launch-spec', 'text']]
        s += book(c, di, 14, gain_clicks=5, switches=['#dr-speed-ceiling']) + stop()
        s += book(c, di, 15, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-launch-boost']) + stop()
        # 起こし直し(3-4 のあと。起動のたびに既定へ)→ 7 章の「起こし直した」: 見張り → 残す・launch_us 200 を入れ直す
        s += [['reboot', 2000.0, RESET], ['poll'], ['tick'], ['realpoll'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html']] + LOOK
        s += [['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['api', {'launch_us': 200}], ['poll'],
              ['realpoll'], ['el', '#dr-message', 'html']] + LOOK
        s += book(c, di, 16, gain_clicks=5, switches=['#dr-speed-ceiling']) + stop()
        s += book(c, di, 17, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-launch-boost']) + stop()
        # 回 4: 4-1(本番の形 = 天井 + 発進の前進 ON・200)
        s += book(c, di, 20, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-launch-boost']) + stop()
        return s
    S['day'] = (day, {})
    # ---- 7 章: 最初のリカバリーがおかしい → 見張りを OFF(残すもサーバーが OFF)→ そのあと起こし直し
    def guard_off(c, di):
        s = [['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['poll']]
        s += book(c, di, 1, gain_clicks=2) + stop()
        s += [['check', '#dr-rec-back-guard', False], ['read', 'toasts'], ['poll']] + LOOK
        s += book(c, di, 2, gain_clicks=2) + stop()
        s += [['reboot', 2000.0, RESET], ['poll'], ['tick'], ['realpoll'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html']]
        s += book(c, di, 3, gain_clicks=2) + stop()
        return s
    S['guard_off'] = (guard_off, {})
    # ---- 2 章・5 章: 左 4.2 を API で入れてからブロックを押す(2-5 を「API で左 4.2 → 天井 ON」の前に API を先に入れた形)
    def left_before_pick(c, di):
        s = [['api', {'ceil_left_mps': 4.2}], ['poll']]
        s += book(c, di, 10, gain_clicks=0, switches=['#dr-speed-ceiling']) + stop()
        s += [['el', '#dr-ceil-spec', 'text']]
        s += book(c, di, 12, gain_clicks=0, switches=['#dr-speed-ceiling']) + stop()
        return s
    S['left_before_pick'] = (left_before_pick, {})
    # ---- 5 章の注意: 前の本が ON のあと API で launch_boost 1 → ブロック → ▶(黙って OFF)
    def launch_api(c, di):
        s = book(c, di, 13, switches=['#dr-launch-boost']) + stop()
        s += [['baseline'], ['pick', di, blk(c, di, 15)], ['el', '#dr-launch-note', 'html'], ['api', {'launch_boost': 1}], ['poll'],
              ['read', 'launch'], ['start']]
        return s
    S['launch_api'] = (launch_api, {})
    # ---- 4 章 10 の表「直す」: トリム +15(スライダー)→ 基準の設定・ブロックでも残る・確認の文に「トリム: +15 µs」
    def trim(c, di):
        s = [['slide', '#dr-trim', 15], ['read', 'trim'], ['baseline'], ['read', 'trim'], ['pick', di, blk(c, di, 2)], ['read', 'trim']]
        s += book(c, di, 3, gain_clicks=0, baseline=True) + stop() + [['read', 'trim']]
        return s
    S['trim'] = (trim, {})
    # ---- 2-3(天井 + 自動合わせ)を 4 章の順(3 ゲイン → 4 スイッチ → 5 説明欄 → 6 ▶)で。説明欄に V_before を書かないまま ▶ した形も
    def auto_vb(c, di):
        s = book(c, di, 8, gain_clicks=5, switches=['#dr-speed-ceiling', '#dr-speed-auto']) + stop()
        s += [['baseline'], ['pick', di, blk(c, di, 9)], ['click', '#btn-dr-tg-up'], ['check', '#dr-speed-ceiling', True],
              ['check', '#dr-speed-auto', True], ['start']]
        return s
    S['auto_vb'] = (auto_vb, {})
    # ---- 2 章: 同じベンチマークを選び直す(やり直しの本)。選んだあと選ぶ欄は「— 今: …」に戻り、もう一度選ぶと入れ直す
    S['repick'] = (lambda c, di: [['pick', di, blk(c, di, 13)], ['read', 'pick'], ['el', '#dr-bench-pick', 'html'], ['check', '#dr-launch-boost', True],
                                  ['pick', di, blk(c, di, 13)], ['read', 'note'], ['read', 'launch'], ['el', '#dr-launch-note', 'html']], {})
    # ---- 3 章 5・4 章 9・7 章: 走り終わりのメモが出るか(止まったあとのサーバーの status = record_log・description を足した形を本物の
    #      renderDrive に渡す)。① 朝の発進の前進の確かめを収集したまま走った本(説明欄 = 空 → 'AI走行 …')② ブロックの本 ③ ② のメモを
    #      書いたあと次の本が止まった → ② の本のメモは出るか ④ 起こし直したあと(status に record_log が無い)
    def after(c, di):
        def st(log, desc, running=False):
            return ("{ const s = Object.assign({}, __T.state.drive.st, { running: %s, phase: 'idle', record_log: %s, description: %s, "
                    "dry_run: false, stats: { ai_s: 30 } }); __T.renderDrive(s); 'ok' }" % (
                        'true' if running else 'false', json.dumps(log), json.dumps(desc)))
        d1 = 'ai_no=1 AI走行 clean_driving_20261011-nocmd-s3-20261011-024626'
        d2 = ('ai_no=2 bench=real_bench_20261023 day=2026-11-03 block=1 cand=EXP model=clean_driving_20261011-nocmd-s3-20261011-024626 '
              'battery=#A V_before=8.31 V_after=__ check=合格 tune=1-1 throttle_gain=0.9 official=0 memo=計画と違うので試し(元 cand=B): x')
        d3 = d2.replace('ai_no=2', 'ai_no=3').replace('block=1', 'block=2')
        return [['eval', st('20261103-081500', d1)], ['el', '#dr-after', 'hidden'],
                ['eval', st('20261103-083000', d2)], ['el', '#dr-after', 'hidden'], ['el', '#dr-after-log', 'text'],
                ['eval', "localStorage.setItem('jetformer.bench.afterDone', '20261103-083000'); 'saved'"],
                ['eval', st('20261103-083000', d2)], ['el', '#dr-after', 'hidden'],
                ['eval', st('20261103-083400', d3)], ['el', '#dr-after', 'hidden'], ['el', '#dr-after-log', 'text'],
                ['eval', "{ const s = Object.assign({}, __T.state.drive.st, { running: false, phase: 'idle', dry_run: false }); "
                         "delete s.record_log; delete s.description; __T.renderDrive(s); 'rebooted' }"], ['el', '#dr-after', 'hidden']]
    S['after'] = (after, {})
    # ---- 11/1 に 24〜27 を足した日(REH_BENCH)の朝: 選ぶ欄に何本並ぶか・24 を押したときの速度
    S['v1'] = (lambda c, di: [['eval', "document.querySelector('#dr-bench-pick').innerHTML.split('<option').length - 2"],
                              ['el', '#dr-bench-pick', 'html']] + book(c, di, 24, gain_clicks=0) + stop(), {})
    return S


def lines_of(out):
    rows = []
    for i, x in enumerate(out['steps']):
        if x['op'] == 'start':
            rows.append('  [%d] start sent=%s refused=%s status=%s dry=%s record=%s desc=%r' % (
                i, {k: x['sent'].get(k) for k in ('throttle_gain', 'speed_ceiling', 'speed_auto', 'launch_boost', 'steer_trim_us')}
                if x['sent'] else None, x['refused'], x['status'], x['dry'], x['record'], x['desc']))
            for cf in x['confirms']:
                rows.append('      confirm: ' + cf.replace('\n', ' / '))
        elif x['op'] in ('read', 'el', 'eval', 'check', 'api', 'tapday'):
            v = x.get('value', x.get('confirm', x.get('body')))
            if x['op'] == 'check':
                v = {'checked': x['checked'], 'confirm': (x.get('confirm') or '')[:160]}
            if x['op'] == 'api':
                v = {'status': x['status'], 'body': x['body'] if x['status'] != 200 else {k: x['body'].get(k) for k in (
                    'rec_back_guard', 'rec_hold_keep', 'ceil_left_mps', 'launch_us', 'launch_boost')}}
            rows.append('  [%d] %s %s %s = %s' % (i, x['op'], x.get('what', x.get('sel', '')), x.get('prop', ''),
                                                   json.dumps(v, ensure_ascii=False)[:600]))
    rows.append('  toasts: ' + ' | '.join(out['toasts']))
    return rows


def main():
    txt = []
    only = sys.argv[1:]
    for name, (fn, kw) in scenes().items():
        if only and name not in only:
            continue
        out = run(name, fn, **kw)
        txt.append('## ' + name)
        txt += lines_of(out)
    p = OUT / ('ui_rehearsal%s.txt' % ('_' + '_'.join(only) if only else ''))
    p.write_text('\n'.join(txt) + '\n', encoding='utf-8')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()
