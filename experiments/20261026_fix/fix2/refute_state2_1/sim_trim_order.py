"""反証(state2-1): トリムを「ブロックを選んだあと」に入れた正しい順でも、次の本で 0 に戻るかを本物の app.js で確かめる。
読むだけ(サーバー・車・ブラウザは使わない)。refute2_state/sim_state2.py の部品を使う。"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute2_state'))
import sim_state2 as s  # noqa: E402

HARNESS = r"""
function poll() { const p = JSON.parse(JSON.stringify(server)); state.drive.st = { running: false, params: p, stats: {} }; setDriveParams(p); }
async function startClick() {
  const c0 = confirms.length, p0 = posts.length;
  for (const fn of $('#btn-dr-start').__ls.click) await fn();
  const sent = posts.slice(p0).find((x) => x[1] === '/api/drive/start');
  return { confirm: confirms.slice(c0).join('\n---\n'), sent: sent ? sent[2].params : null };
}
async function pick(value) { const sel = $('#dr-bench-pick'); const fired = sel.value !== value; sel.value = value;
  if (fired) for (const fn of sel.__ls.change) await fn({ target: sel }); return fired; }
(async () => {
  const out = {};
  const B = BENCH.baseline.run;
  state.drive.models = [{ name: B, drivable: true }];
  state.bench.data = BENCH;
  $('#dr-model').value = B; $('#dr-mode').value = '5';
  const day = BENCH.days.findIndex((d) => d.date === '2026-11-03');
  server = JSON.parse(JSON.stringify(SERVER0)); poll();
  // 1-1 を走った → car_check「直す」。手順書 4 章の順: 2 で 1-2 を選ぶ → 3 速度 → 4 のあとでトリム +15
  await pick(day + ':0'); await startClick(); poll();
  await pick(day + ':1'); poll();
  $('#dr-trim').value = '15'; delete state.drive.pending.steer_trim_us; server.steer_trim_us = 15; poll();   // スライダー(すぐ送る)
  const r1 = await startClick(); poll();
  // 「1 本走り、もう一度」→ 合格。次の 1-3 を選ぶ(4 章の 2)。トリムは手順書のどこにも「入れ直す」と無い
  await pick(day + ':2');
  const note = $('#dr-base-note-text').innerHTML; poll();
  const r2 = await startClick();
  out.C = { run_1_2_sent_trim: r1.sent.steer_trim_us, run_1_3_sent_trim: r2.sent.steer_trim_us, server_after: server.steer_trim_us,
            note_1_3: note, confirm_1_3_has_trim: /トリム|trim/.test(r2.confirm), toasts_last: toasts.slice(-2) };
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def main():
    js = s.APP_JS.read_text(encoding='utf-8')
    data = s.bench.load_candidates()
    data['today'] = '2026-11-03'
    saved = json.loads((s.JF / 'data' / 'drive_settings.json').read_text(encoding='utf-8'))
    server0 = dict(s.drive.DEFAULT_PARAMS)
    server0.update({k: v for k, v in saved['params'].items() if k in s.drive.DEFAULT_PARAMS})
    for k in s.drive.RESET_ON_BOOT:
        server0[k] = s.drive.DEFAULT_PARAMS[k]
    code = '\n'.join([s.STUBS] + [s._extract(js, n) for n in s.NAMES]
                     + [s._handler(js, '#dr-bench-pick', 'change'), s._handler(js, '#btn-dr-start', 'click'),
                        'const BENCH = %s;' % json.dumps(data, ensure_ascii=False),
                        'const SERVER0 = %s;' % json.dumps(server0), HARNESS])
    r = subprocess.run([shutil.which('node'), '-e', code], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise SystemExit(r.stderr)
    print(json.dumps({'server0_trim': server0['steer_trim_us'], **json.loads(r.stdout)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
