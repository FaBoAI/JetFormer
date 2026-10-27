"""fix7 画面の係 直す 7-1 回目: わざと壊すと tests/test_ui_fix7.py が落ちるか(本物の app.js・台本は変えない。写しを壊して
JF_UI7_APP・JF_UI7_SIM で渡す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix7/ui_r1/mut/run_mut.py > experiments/20261026_fix/fix7/ui_r1/mut/out_mut.txt

壊し方ごとに、落ちた試験の名前を出す。「差分の形」の試験(test_before_copy_and_only_the_boot_note_changed)と、表を読むだけの試験
(test_boot_defaults_are_drive_defaults・test_refusal_pattern_does_not_match_the_new_toasts)のほかに、node で画面を動かす試験が
1 本以上落ちなければ「見逃し」。一時の写しは experiments/20261026_fix/fix7/tmp_ui/mut/ に置き、終わったら消す。
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

JF = Path('/home/jetson/jetformer')
APP = JF / 'jetformer' / 'web' / 'static' / 'app.js'
SIM = JF / 'tests' / 'fixtures' / 'app_sim_fix7.js'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
TMP = JF / 'experiments' / '20261026_fix' / 'fix7' / 'tmp_ui' / 'mut'
STATIC_TESTS = ('test_before_copy_and_only_the_boot_note_changed', 'test_boot_defaults_are_drive_defaults',
                'test_refusal_pattern_does_not_match_the_new_toasts')

ONOFF = ("  const on = has.filter((o) => +p[o.key] === 1).map((o) => o.name), off = has.filter((o) => +p[o.key] === 0).map((o) => o.name);\n"
         "  const odd = has.filter((o) => +p[o.key] !== 0 && +p[o.key] !== 1).map((o) => `${o.name}は ${p[o.key]}`);")
APP_MUTS = {
    'トーストをいつも前の文(値を見ない)': ('  if (!diff.length) return `', '  if (true) return `'),
    '知らせのときの値でトーストを作らない(空の値)': ('  const msg = bootToastText(hm, p);', '  const msg = bootToastText(hm, {});'),
    '走行タブの文を知らせのときの値で固定(作り直さない)': [
        ('  state.drive.bootHm = hm;', '  state.drive.bootHm = hm; state.drive.bootP = p;'),
        ('esc(bootNoteText(state.drive.bootHm, st.params || {}))', 'esc(bootNoteText(state.drive.bootHm, state.drive.bootP || {}))')],
    '試験の選択肢を値を見ずに全部 OFF(前の形)': (ONOFF, "  const on = [], off = has.map((o) => o.name);\n  const odd = [];"),
    'ON の印「(起動のあとに入れ直し済み)」なし': ('は ON(起動のあとに入れ直し済み)', 'は ON'),
    '天井の左・強さに「今 X(既定 Y)」を出さない(前の形)': ('  const val = (k, name, unit) => (bootDiffers(p, k) ? ', '  const val = (k, name, unit) => (false ? '),
    'ほかの既定と違う項目を出さない': ("    + (other.length ? `(ほかに", "    + (false ? `(ほかに"),
    'トーストに試験の選択肢を挙げない': ('  const diff = Object.keys(BOOT_DEFAULTS).filter((k) => bootDiffers(p, k))',
                                       '  const diff = Object.keys(BOOT_DEFAULTS).filter((k) => bootDiffers(p, k) && !OPT_SWITCHES.some((o) => o.key === k))'),
    '既定と違うかの判定が反対': ('!(Math.abs(+p[k] - BOOT_DEFAULTS[k]) < 1e-9)', '(Math.abs(+p[k] - BOOT_DEFAULTS[k]) < 1e-9)'),
    '既定の左を 4.2 に': ('ceil_left_mps: 4.1, launch_us: 250.0,', 'ceil_left_mps: 4.2, launch_us: 250.0,'),
    '既定の強さを 200 に': ('ceil_left_mps: 4.1, launch_us: 250.0,', 'ceil_left_mps: 4.1, launch_us: 200.0,'),
    '▶ の出し直しを前の固定の文に': ('  state.drive.bootToast = msg;', '  state.drive.bootToast = bootToastText(hm, {});'),
    '走行タブの文を出さない': ('    + (state.drive.bootHm ? `<div', '    + (false ? `<div'),
    'TCP の出し方の名前を出さない': ("  if (k === 'tcp_mode') return `${name}「${TCP_MODES[v] || v}」`;\n", ''),
    '単位を出さない': ("const BOOT_UNITS = { ceil_left_mps: ' m/s', launch_us: ' µs', ceil_all_mps: ' m/s', auto_target_mps: ' m/s', launch_mps: ' m/s', launch_max_s: ' 秒' };",
                     'const BOOT_UNITS = {};'),
    'ON/OFF の項目を数で出す': ("  if (BOOT_FLAGS.includes(k) && (+v === 0 || +v === 1))", "  if (false)"),
}
FIRE = "const fireTimers = async (pick, wait) => { const fired = []; for (const x of timers.filter(pick)) { const i = timers.indexOf(x); if (i < 0) continue; timers.splice(i, 1); fired.push(x); const r = x[0](); if (wait) await r; } return fired; };"
SIM_MUTS = {
    '台本: click・slide のあと長いタイマーを捨てる(もとの形)': (
        "const runShortTimers = async () => { await fireTimers((x) => x[1] <= 200, true); await flush(); };",
        "const runShortTimers = async () => { const t = timers.splice(0); for (const [fn, ms] of t) if (ms <= 200) await fn(); await flush(); };"),
    '台本: 消されたタイマーも鳴らす(先に全部取り出す)': (
        FIRE, "const fireTimers = async (pick, wait) => { const t = timers.filter(pick); for (const x of t) timers.splice(timers.indexOf(x), 1); "
              "for (const x of t) { const r = x[0](); if (wait) await r; } return t; };"),
    "台本: ['timers'] がもとの形": (
        "    else if (op === 'timers') { await fireTimers((x) => x[1] <= 200, false); await flush();",
        "    else if (op === 'timers') { const t = timers.splice(0); const rest = []; for (const x of t) { if (x[1] <= 200) x[0](); else rest.push(x); } timers.push(...rest); await flush();"),
    "台本: ['tick'] がもとの形": (
        "    else if (op === 'tick') { const t = await fireTimers(() => true, false); await flush();",
        "    else if (op === 'tick') { const t = timers.splice(0); for (const x of t) x[0](); await flush();"),
}


def run(name, env, d):
    r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rf', '--basetemp', str(d / 'bt'), 'tests/test_ui_fix7.py'],
                       cwd=str(JF), capture_output=True, text=True, env=dict(os.environ, **env), timeout=900)
    failed = sorted(set(re.findall(r'^FAILED tests/test_ui_fix7\.py::(.+?)(?: - .*)?$', r.stdout, re.M)))
    behave = [f for f in failed if not f.startswith(STATIC_TESTS)]
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    print('== %s' % name)
    print('   落ちた動きの試験 %d 本: %s' % (len(behave), ', '.join(behave) or '-'))
    print('   落ちた表・形の試験: %s' % (', '.join(f for f in failed if f.startswith(STATIC_TESTS)) or '-'))
    print('   ' + tail)
    return bool(behave)


def main():
    src, sim = APP.read_text(encoding='utf-8'), SIM.read_text(encoding='utf-8')
    TMP.mkdir(parents=True, exist_ok=True)
    missed, n = 0, 0
    try:
        for i, (name, reps) in enumerate(APP_MUTS.items()):
            m = src
            for old, new in (reps if isinstance(reps, list) else [reps]):
                assert m.count(old) == 1, (name, old)
                m = m.replace(old, new)
            d = TMP / ('a%02d' % i)
            d.mkdir(exist_ok=True)
            (d / 'app.js').write_text(m, encoding='utf-8')
            n += 1
            missed += not run(name, {'JF_UI7_APP': str(d / 'app.js')}, d)
        for i, (name, (old, new)) in enumerate(SIM_MUTS.items()):
            assert sim.count(old) == 1, name
            d = TMP / ('s%02d' % i)
            d.mkdir(exist_ok=True)
            (d / 'sim.js').write_text(sim.replace(old, new), encoding='utf-8')
            n += 1
            missed += not run(name, {'JF_UI7_SIM': str(d / 'sim.js')}, d)
        d = TMP / 'none'
        d.mkdir(exist_ok=True)
        print('== 壊さない(今の app.js・台本)')
        r = subprocess.run([PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', str(d / 'bt'), 'tests/test_ui_fix7.py'],
                           cwd=str(JF), capture_output=True, text=True, timeout=900)
        print('   ' + r.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print('見逃し %d / %d' % (missed, n))


if __name__ == '__main__':
    main()
