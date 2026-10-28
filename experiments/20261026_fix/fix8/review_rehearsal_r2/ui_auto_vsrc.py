"""反証 8-2 回目・係 rehearsal: 手順書 4 章 6 の新しい行「自動合わせ ON の本(2-3・2-4)は「電圧の源」も読む」(10/28 fix8 docs r1)を、
本物の app.js・index.html(node の vm・sim_r2.js = tests/fixtures/app_sim_fix8.js の写し)で 4 章の順にたどる。偽のサーバーの params は
本物の GET /api/drive/status(out/live_status.json。ファームなし = vbat.supported false)。車・本物のサーバーには触れない。

  2-3: 1 基準 → 2 ベンチマーク 8 → 3 +0.01 × 5 → 4 天井・自動合わせのスイッチ(確認の文の「電圧の源」)→ 5 説明欄(V_before=8.31)→ 6 ▶
  2-4: 同じ順で 5 の V_before を書き忘れ → ▶ の確認の文の「電圧の源 なし」→ 手順書のとおりキャンセル → 説明欄に V_before= → もう一度 ▶
  2-4':V_before を「V_before=8,31」(読点)・「V_before=８.３１」(全角)と書いた形も(画面の決め /(^|\\s)V_before=\\d/ と、
        サーバー jetformer/speed_auto.v_before_of の読み方が同じか)
出力: out/ui_auto_vsrc.txt・out/ui_auto_vsrc.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer.web import bench  # noqa: E402
from jetformer import speed_auto  # noqa: E402

STATIC = JF / 'jetformer' / 'web' / 'static'
SIM = HERE / 'sim_r2.js'
OUT = HERE / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal_r2/ui'
LIVE = json.loads((OUT / 'live_status.json').read_text(encoding='utf-8'))

c = dict(bench.load_candidates(None), today='2026-11-03')
DI = next(i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03')


def blk(no):
    return next(i for i, b in enumerate(c['days'][DI]['blocks']) if int(b['block']) == no)


def fill(vb):
    return ("{ const e = document.querySelector('#dr-desc'); e.value = e.value.replace('battery=#__', 'battery=#A')"
            ".replace('V_before=__', %s).replace('check=__', 'check=合格'); e.value }" % json.dumps(vb))


def book(no, vb, extra=()):
    s = [['baseline'], ['pick', DI, blk(no)], ['el', '#dr-throttle-gain', 'value']]
    s += [['click', '#btn-dr-tg-up']] * 5 + [['el', '#dr-throttle-gain', 'value']]
    s += [['check', '#dr-speed-ceiling', True], ['check', '#dr-speed-auto', True], ['eval', fill(vb)]]
    s += list(extra)
    return s


def stop():
    return [['set_running', False], ['tick'], ['tick']]


steps = [['tab', 'drive'], ['tick'], ['tick']]
steps += book(8, 'V_before=8.31') + [['start']] + stop()
steps += book(9, 'V_before=__') + [['confirm', False], ['start'], ['confirm', True],
                                   ['eval', fill('V_before=8.31').replace("'V_before=__'", "'V_before=__'")],
                                   ['eval', "{ const e = document.querySelector('#dr-desc'); e.value = e.value.replace('V_before=__', 'V_before=8.31'); e.value }"],
                                   ['start']] + stop()
for vb in ('V_before=8,31', 'V_before=８.３１', 'V_before=0.83', 'V_before=831'):
    steps += book(9, vb) + [['confirm', False], ['start'], ['confirm', True]]
inp = {'bench': c, 'server_params': dict(LIVE['params']), 'steps': steps, 'server_client_version': 9, 'confirm': True,
       'boot_at': LIVE.get('boot_at'), 'models': [c['baseline']['run']],
       'script_src': 'http://x/static/app.js?v=%d' % int((STATIC / 'app.js').stat().st_mtime), 'server_boot_check': True}
TMP.mkdir(parents=True, exist_ok=True)
pin = TMP / 'auto_vsrc.json'
pin.write_text(json.dumps(inp, ensure_ascii=False, default=str), encoding='utf-8')
r = subprocess.run(['node', str(SIM), str(STATIC / 'app.js'), str(STATIC / 'index.html'), str(pin)], capture_output=True, text=True,
                   timeout=300, env=dict(os.environ, TZ='Asia/Tokyo'))
if r.returncode != 0:
    raise SystemExit(r.stderr[-3000:])
out = json.loads(r.stdout.strip().splitlines()[-1])
(OUT / 'ui_auto_vsrc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
lines = []
for i, st in enumerate(out['steps']):
    op = st['op']
    if op == 'check' and st.get('confirm'):
        cf = st['confirm']
        lines.append('[%d] スイッチ %s → 確認の文の電圧の源: %s' % (i, st['sel'], [x for x in cf.split('\n') if '電圧の源' in x] or '(行なし)'))
    elif op == 'eval' and isinstance(st['value'], str):
        lines.append('[%d] 説明欄: %s' % (i, st['value'][-140:]))
    elif op == 'el':
        lines.append('[%d] %s %s = %s' % (i, st['sel'], st['prop'], st['value']))
    elif op == 'start':
        cfs = [ln for cf in st['confirms'] for ln in cf.split('\n') if '自動合わせ' in ln or '電圧' in ln]
        lines.append('[%d] ▶ status=%s 送った description の V_before 欄=%s → 確認の文: %s' % (
            i, st['status'], [t for t in (st['desc'] or '').split() if t.startswith('V_before')], cfs))
        if st['desc']:
            v = speed_auto.v_before_of(st['desc'])
            lines.append('      サーバー(speed_auto.v_before_of)が読む V_before: %s' % v)
lines.append('toasts: %s' % out['toasts'])
# 画面に出なかった(キャンセルした)本の説明欄も、サーバーの読み方で照らす
for vb in ('V_before=8.31', 'V_before=8,31', 'V_before=８.３１', 'V_before=0.83', 'V_before=831'):
    lines.append('サーバーの読み方 v_before_of(%r) = %s' % (vb, speed_auto.v_before_of('battery=#A %s check=合格' % vb)))
txt = '\n'.join(lines) + '\n'
(OUT / 'ui_auto_vsrc.txt').write_text(txt, encoding='utf-8')
print(txt)
