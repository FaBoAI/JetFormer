#!/usr/bin/env python3
"""記録の段: 取り込んだ RL のデータセットが JetFormer の画面(データセット・シーン・ジャンルのタブ)に出ていることを、
ヘッドレス Firefox で撮る(2026-10-14)。見るだけで、ボタンは押さない(タブの切り替えだけ)。

    jf-heavy --mem 2500M --lock big -- python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/ui_screens.py

  - 他の Firefox が無いこと・空きメモリ 2 GiB 以上を確かめてから始める(webgl_probe.wait_preflight)
  - geckodriver 4445・専用プロファイル(sim/ff_profile_rl)・snap のスコープにもメモリ上限 2500M(browser_collect.cap_snap_scope)
  - 最初の「ライブ・収集」タブは開いた直後にデータセットのタブへ切り替える(ライブ映像の表示を止める)
  - 出力: /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/screens/jetformer_*.png
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from webgl_probe import GECKO, GECKO_PORT, PROFILE, REPORT, WebDriver, other_firefox, wait_preflight  # noqa: E402
from browser_collect import cap_snap_scope  # noqa: E402

UI = 'http://127.0.0.1:8080/'
NAME = json.loads((REPORT / 'rl_dataset.json').read_text())['name']
OUT = REPORT / 'screens'


def main():
    res = {'preflight': wait_preflight(), 'dataset': NAME, 'shots': []}
    log = open(REPORT / 'recording' / 'ui_screens_geckodriver.log', 'w')
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT)
    d = WebDriver(GECKO_PORT)
    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(.5)
        d.start(['-headless', '-profile', str(PROFILE)], {}, {'MOZ_HEADLESS_WIDTH': '1500', 'MOZ_HEADLESS_HEIGHT': '1100'})
        res['memory_cap'] = cap_snap_scope(gd.pid, '2500M')
        d.get(UI)
        time.sleep(2)
        d.js("document.querySelector('.tab[data-tab=\"datasets\"]').click();")
        time.sleep(4)
        found = d.js("""const n=arguments[0];const el=[...document.querySelectorAll('#tab-datasets *')].find(e=>e.children.length===0&&e.textContent.includes(n));
            if(el){el.scrollIntoView({block:'center'});return el.textContent.slice(0,200);}return null;""", NAME)
        res['datasets_tab_found'] = found
        time.sleep(1)
        d.screenshot(OUT / 'jetformer_datasets_tab.png'); res['shots'].append('jetformer_datasets_tab.png')
        d.js("document.querySelector('.tab[data-tab=\"scenes\"]').click();")
        time.sleep(4)
        log_name = 'ds-' + NAME
        res['scenes_tab_found'] = d.js("""const n=arguments[0];const el=[...document.querySelectorAll('#tab-scenes *')].find(e=>e.children.length===0&&e.textContent.includes(n));
            if(el){el.scrollIntoView({block:'center'});return el.textContent.slice(0,200);}return null;""", log_name)
        time.sleep(2)
        d.screenshot(OUT / 'jetformer_scenes_tab.png'); res['shots'].append('jetformer_scenes_tab.png')
        d.js("document.querySelector('.tab[data-tab=\"modes\"]').click();")
        time.sleep(3)
        d.screenshot(OUT / 'jetformer_modes_tab.png'); res['shots'].append('jetformer_modes_tab.png')
    except Exception as e:  # noqa: BLE001
        res['error'] = f'{type(e).__name__}: {e}'
    finally:
        d.quit()
        gd.terminate()
        try:
            gd.wait(20)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
        time.sleep(2)
        res['firefox_left'] = other_firefox()
        (REPORT / 'recording' / 'ui_screens.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
