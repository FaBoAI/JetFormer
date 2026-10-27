"""反証 2 回目・state の確かめ C(読むだけ。車・装置・サーバーは使わない)。

走行中に OFF にした天井・自動合わせ・発進の前進(画面も API も「走行中は OFF にはできる」)が、記録(log.json の drive.params・
events.jsonl)に残るかを見る。本物の drive.Driver.set_params を、走行スレッドの代わりの生きたスレッドで「走行中」にして呼ぶ。
設定の保存先は tmp に向ける(data/drive_settings.json は書かない)。そのあと、11/3 の夜の道具(runs_1103.diffs・
standing_3lap.settings_of・quick_eval_by_params)が、その記録をどう読むかを見る。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix2/refute2_state/check_midrun_off.py
"""
import importlib.util
import json
import sys
import tempfile
import threading
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
from jetformer import drive  # noqa: E402


class FakeRecorder(object):
    recording = True

    def __init__(self):
        self.events = []
        self.log_json = None

    def start(self, description='', tags=None, mode_id=None, drive=None, **kw):     # 走行ループが 1 回だけ呼ぶ形
        self.log_json = {'description': description, 'drive': json.loads(json.dumps(drive))}
        return '20261103-101500'

    def add_event(self, kind, data):
        self.events.append((kind, data))
        return True

    def set_mode(self, mode_id):
        pass


def main():
    tmp = Path(tempfile.mkdtemp(prefix='refute2_state_'))
    drive.SETTINGS_PATH = tmp / 'drive_settings.json'          # 本物の設定ファイルは書かない
    rec = FakeRecorder()
    d = drive.Driver(hub=None, recorder=rec)
    # 2-3(天井 4.1 + 自動合わせ)を始めた形: 走行開始で送った値 → 走行ループが log.json に drive.params を書く(drive.py 1315 行)
    d.params = d._launch_rules(d._speed_ctrl_rules(d._clean_params(
        {'throttle_gain': 0.95, 'speed_ceiling': 1, 'speed_auto': 1, 'launch_boost': 1}), False), False)
    stop = threading.Event()
    d.thread = threading.Thread(target=stop.wait, daemon=True)  # 走行中(running = スレッドが生きている)
    d.thread.start()
    rec.start(description='bench=real_bench_20261023 day=2026-11-03 block=8 cand=EXP', drive={'run': 'B', 'dry_run': False,
                                                                                         'params': dict(d.params)})
    # 走っている途中で「揺れています」が出たので、手順書 7 章のとおり天井・自動合わせを OFF に戻す(画面のスイッチと同じ API)
    out = {'running': d.running}
    p1 = d.set_params({'speed_ceiling': 0, 'speed_auto': 0})
    p2 = d.set_params({'launch_boost': 0})
    out['server_after'] = {k: p2[k] for k in ('speed_ceiling', 'speed_auto', 'launch_boost')}
    out['log_json_params'] = {k: rec.log_json['drive']['params'][k] for k in ('speed_ceiling', 'speed_auto', 'launch_boost')}
    out['events_written'] = rec.events
    try:
        d.set_params({'speed_ceiling': 1})
        out['on_again'] = 'できた'
    except ValueError as e:
        out['on_again'] = '断った: %s' % e
    stop.set()
    d.thread.join(1)

    # 夜の道具の読み方
    spec = importlib.util.spec_from_file_location('runs_1103', JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
    r1103 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r1103)
    bc = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    day = next(x for x in bc['days'] if x['date'] == '2026-11-03')
    b8 = next(b for b in day['blocks'] if b['block'] == 8)
    expect = dict(b8['expect'])
    expect.pop('launch_boost', None)                            # 発進の前進は 2-3 の計画では 0。ここでは天井・自動だけを見る
    out['runs_1103_diffs'] = r1103.diffs(expect, rec.log_json['drive']['params'])
    import standing_3lap as s3                                  # noqa: E402
    s = s3.settings_of({'drive': rec.log_json['drive'], 'mode_id': 5})
    out['standing_3lap_settings'] = s3.settings_text(s)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


if __name__ == '__main__':
    main()
