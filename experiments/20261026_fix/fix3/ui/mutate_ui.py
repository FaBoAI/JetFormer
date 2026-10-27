"""係 ui(2026-10-26 fix3): わざと壊した app.js・index.html の写しで tests/test_ui_fix3.py が落ちるかを見る(本物のファイルは変えない)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/ui/mutate_ui.py <作業場所>
結果は <作業場所>/mutate_ui.json と標準出力。
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
APP, INDEX = JF / 'jetformer/web/static/app.js', JF / 'jetformer/web/static/index.html'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'

# (名前, ファイル, 前, 後, 落ちるはずの試験)
MUTS = [
    ('state1_revert', 'app', "on ? { speed_ceiling: 1, speed_auto: 0 } :", "on ? { speed_ceiling: 1 } :", 'test_state1'),
    ('step_005', 'index', 'max="2" step="0.01" value="1">\n            <div class="gain-scale"><span>0(停止)',
     'max="2" step="0.05" value="1">\n            <div class="gain-scale"><span>0(停止)', 'test_state2'),
    # 0.01 の段への丸めを外す変異は toFixed(2) と同じ答えになる(見分けられない)ので、範囲の外へ出る変異にした
    ('gainstep_no_clamp', 'app', '  return +Math.min(GAIN_MAX, Math.max(GAIN_MIN, x)).toFixed(2);', '  return +x.toFixed(2);',
     'test_state2_gain_step'),
    ('button_no_send', 'app', "    el.dispatchEvent(new Event('input'));\n", '', 'test_state2_buttons'),
    ('client_6', 'app', 'const body = { client: 7,', 'const body = { client: 6,', 'test_state5'),
    ('opt_running_on', 'app', "      if (st.running) { el.checked = false; toast(`走行中は${o.name}を ON にできません",
     "      if (false) { el.checked = false; toast(`走行中は${o.name}を ON にできません", 'test_opt_cancel_and_running'),
    ('opt_disabled_running', 'app', '    el.disabled = (running && !srvOn) || (!needOk && !srvOn);',
     '    el.disabled = (!needOk && !srvOn);', 'test_opt_cancel_and_running'),
    ('opt_no_confirm', 'app', "      if (!confirm(optConfirmText(o))) { el.checked = false; return; }", '', 'test_opt'),
    ('opt_sent_at_start', 'app', "  p.launch_boost = $('#dr-launch-boost').checked ? 1 : 0;                                  // 発進の前進(2026-10-26)\n",
     "  p.launch_boost = $('#dr-launch-boost').checked ? 1 : 0;\n  p.rec_back_guard = $('#dr-rec-back-guard').checked ? 1 : 0;\n",
     'test_opt'),
    ('opt_block_turns_off', 'app', "    delete f.params[o.key];\n",
     "    delete f.params[o.key];\n    if (block) { $(o.sel).checked = false; }\n", 'test_opt_switch_on_off'),
    ('opt_show_old_server', 'app', "    if (line) line.classList.toggle('hidden', !ok);", "    if (line) line.classList.toggle('hidden', false);",
     'test_opt_old_server'),
    ('opt_no_mirror', 'app', '    if (!optBusy[o.key]) el.checked = srvOn;', '', 'test_opt_block_and_baseline'),
    ('trim_reset', 'app', "  for (const k of ['steer_trim_us', 'rec_back_guard']) delete p[k];", "  for (const k of ['rec_back_guard']) delete p[k];",
     'test_trim'),
    ('html_switch_checked', 'index', '<input type="checkbox" id="dr-rec-hold-keep" autocomplete="off">',
     '<input type="checkbox" id="dr-rec-hold-keep" autocomplete="off" checked>', 'test_opt_switches_static'),
    ('button_unguarded', 'app', "  if (!$(id)) continue;                                           // 古い index.html(キャッシュ)でも画面全体を止めない\n",
     '', 'test_new_app_with_old_index'),
    ('no_first_sync_guard', 'app', "  if (!state.drive.paramsLoaded) { toast(", "  if (false) { toast(", 'test_start_waits_for_first_sync'),
    ('hold_keep_no_needs_check', 'app',
     "      if (o.needs && +before[o.needs] !== 1) { el.checked = false;", "      if (false) { el.checked = false;",
     'test_opt_hold_keep_needs_back_guard'),
    ('hold_keep_no_needs_disable', 'app', "    el.disabled = (running && !srvOn) || (!needOk && !srvOn);",
     "    el.disabled = (running && !srvOn);", 'test_opt_hold_keep_needs_back_guard'),
    ('server_sends_ceiling_only_mirror', 'app', "    if (!optBusy[o.key]) el.checked = srvOn;", "    if (!optBusy[o.key] && srvOn) el.checked = srvOn;",
     'test_opt'),
]


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    src = {'app': APP.read_text(encoding='utf-8'), 'index': INDEX.read_text(encoding='utf-8')}
    out = []
    for name, which, a, b, want in MUTS:
        assert src[which].count(a) == 1, (name, src[which].count(a))
        d = work / name
        d.mkdir(exist_ok=True)
        files = {'app': d / 'app.js', 'index': d / 'index.html'}
        for k, p in files.items():
            p.write_text(src[k].replace(a, b) if k == which else src[k], encoding='utf-8')
        env = dict(os.environ, JF_UI_FIX3_APP=str(files['app']), JF_UI_FIX3_INDEX=str(files['index']))
        r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp=%s' % (d / 'pt'), '-rf',
                            str(JF / 'tests/test_ui_fix3.py')], capture_output=True, text=True, env=env, cwd=str(JF), timeout=900)
        failed = sorted(set(re.findall(r'FAILED tests/test_ui_fix3.py::([A-Za-z0-9_]+)', r.stdout)))
        row = {'mutation': name, 'failed': failed, 'caught': bool(failed), 'want_prefix': want,
               'want_caught': any(f.startswith(want) for f in failed), 'tail': r.stdout.strip().splitlines()[-1:]}
        out.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (work / 'mutate_ui.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落ちた(見つけた)', sum(r['caught'] for r in out), '/', len(out), ' ねらいの試験で', sum(r['want_caught'] for r in out))


if __name__ == '__main__':
    main()
