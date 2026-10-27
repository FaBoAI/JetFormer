"""fix5 t1 の確かめ(反証 4 回目 S1 の写し + 直し (b) の照らし)。本物の app.js(今の版と直す前の写し)を tests/fixtures/app_sim_fix5.js
で動かし、偽のサーバーは server.py の _boot_refusal と同じに走行開始を照らす(server_boot_check)。走行開始の POST は本物の
server._boot_refusal(boot = 起き直したあとの本物の drive.Driver の boot_at)に通し、断られなければ本物の Driver.start に入れる。

場面: no_sync(起き直したあと同期せずに ▶)・dialog_spans_restart(落ちている間に問い合わせが失敗 → ▶。(a) が断る)・
dialog_only(問い合わせは失敗していない。確認の文を読んでいる間に起き直した。(b) だけが断る)・synced(1 回同期してから ▶)・
retry(断られたあと、画面が同期してから、もう一度 ▶)。比べ: 直す前の app.js(tests/fixtures/fix5_ui_before/app.js。版 8 で boot_at を
送らない)と直す前のサーバー(照らさない)の組で、反証 4 回目と同じ形が起きることも見る)。
"""
import json

import common5 as C
from jetformer.web import server as srv

c = C.bench_today()
di = C.day_index(c)
b15 = C.blk(c, di, 15)
pre = C.server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, launch_no_kick=1, ceil_left_mps=4.2, launch_us=200.0)
RESET = {k: C.drive.DEFAULT_PARAMS[k] for k in C.drive.RESET_ON_BOOT}
setup = [['pick', di, b15], ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'],
         ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True], ['poll'],
         ['confirm', False], ['start'], ['confirm', True], ['posts']]
SCEN = {
    'no_sync': [['reboot', 2000.0, RESET], ['start'], ['posts'], ['read', 'toasts']],
    'dialog_spans_restart': [['down', True], ['realpoll'], ['realpoll'], ['el', '#btn-dr-start', 'disabled'],
                             ['reboot_on_confirm', 2000.0, RESET], ['start'], ['posts'], ['read', 'toasts']],
    'dialog_only': [['reboot_on_confirm', 2000.0, RESET], ['start'], ['posts'], ['read', 'toasts']],
    'synced': [['reboot', 2000.0, RESET], ['poll'], ['start'], ['posts'], ['read', 'toasts']],
    'retry': [['reboot', 2000.0, RESET], ['start'], ['posts'], ['start'], ['posts'], ['read', 'toasts']],
}
KEYS = ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'ceil_left_mps', 'launch_us', 'speed_ceiling', 'launch_boost', 'throttle_gain')
LINES = ('見張り', '残す', '補助も', '天井', '発進の前進:')


def mismatch(conf, real):
    out = []
    for k, w in (('rec_back_guard', '後ろの見張り: ON'), ('rec_hold_keep', '残す: ON'), ('launch_no_kick', '補助も出さない: ON')):
        if (+real[k] == 1) != any(w in x for x in conf):
            out.append(k)
    if (+real['speed_ceiling'] == 1) != any('速さの天井: ON' in x for x in conf):
        out.append('speed_ceiling')
    if real['speed_ceiling'] and not any('左旋回中 %.1f' % real['ceil_left_mps'] in x for x in conf):
        out.append('ceil_left_mps')
    if (+real['launch_boost'] == 1) != any(x.startswith('発進の前進: ON') for x in conf):
        out.append('launch_boost')
    if real['launch_boost'] and not any('前進 %.0f µs' % real['launch_us'] in x for x in conf):
        out.append('launch_us')
    return out


out = {}
for app_name, app in (('now', None), ('before', C.JF / 'tests' / 'fixtures' / 'fix5_ui_before' / 'app.js')):
    for name, tail in SCEN.items():
        res, toasts = C.sim(c, setup + tail, server=pre, app=app, server_boot_check=(app is None))
        d1 = C.driver(params=pre, settings={'params': pre, 'run': C.BRUN, 'mode_id': 5})
        d1._save_settings()
        d2 = C.driver()                                  # 起き直したサーバー(同じ設定ファイル。RESET_ON_BOOT)
        # 偽のサーバーの boot_at → 本物の時刻(前の起動 = 今の起動の 60 秒前。d1 と d2 は同じミリ秒に作られることがあるので d1 の値は使わない)
        boots = {1000.0: d2.boot_at - 60.0, 2000.0: d2.boot_at}
        rows = []
        for s, ps in zip(C.starts(res)[1:], C.posts(res)[1:]):
            st_posts = [p for p in ps if p[0] == '/api/drive/start']
            if not st_posts:
                rows.append({'no_start_post': True, 'confirm_shown': bool(s['confirm'])})
                continue
            for path, body, status in st_posts:
                b = dict(body, boot_at=boots.get(body.get('boot_at'), body.get('boot_at')))
                # 今の版 = 直したあとの server.py の照らし。直す前の写し = 直す前のサーバー(照らさない。版 8 で始まる)
                why = srv._boot_refusal(b, d2.boot_at) if app is None else None
                real = None
                if why is None:
                    d2.start(body['run'], body['mode_id'], params=body['params'], description=body.get('description', ''),
                             dry_run=body.get('dry_run', False))
                    d2.thread.join(1)
                    real = {k: d2.params[k] for k in KEYS}
                conf = [x for x in s['confirm'].split('\n') if any(w in x for w in LINES)]
                rows.append({'sim_status': status, 'real_server_refused': why is not None, 'why': (why or '')[:70],
                             'confirm_lines': conf, 'real_driver': real,
                             'confirm_vs_real_mismatch': mismatch(conf, real) if real else None})
        out['%s/%s' % (app_name, name)] = {'starts': rows, 'toasts_tail': toasts[-2:],
                                           'start_disabled_while_down': [x['value'] for x in res if x['op'] == 'el']}
print(json.dumps(out, ensure_ascii=False, indent=1))
(C.HERE / 'out').mkdir(exist_ok=True)
(C.HERE / 'out' / 's1b_reboot_boot_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
C.cleanup()
