"""指摘 state2-2 の打ち消しの確かめ(読むだけ・車・装置・サーバーは使わない)。

本物の drive.Driver と本物の recorder.Recorder(偽のハブ・記録の置き場は tmp)で、走行中(スレッドが生きている)に
天井・自動合わせ・発進の前進を OFF にする。ディスクの log.json・events.jsonl・drive_settings.json を読み、
そのあと 11/3 の夜の道具(runs_1103.main・standing_3lap.settings_of/settings_text・quick_eval_by_params.key_of)に渡す。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix2/refute_state2_2/check_midrun_record.py
"""
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
from jetformer import config, drive, recorder  # noqa: E402


class FakeCh(object):
    connected = True
    count = 0

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


class FakeHub(object):
    channels = ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE')

    def __init__(self):
        self.ch = {n: FakeCh() for n in self.channels}

    def __getitem__(self, k):
        return self.ch[k]

    def start(self):
        pass


def main():
    tmp = Path(tempfile.mkdtemp(prefix='state2_2_', dir=str(Path(__file__).parent)))
    raw = tmp / 'raw'
    raw.mkdir()
    config.RAW_DIR = raw                                     # 本物の data/raw は書かない
    drive.SETTINGS_PATH = tmp / 'drive_settings.json'        # 本物の設定ファイルは書かない
    out = {}
    try:
        rec = recorder.Recorder(FakeHub())
        rec._warm_up = lambda timeout=8.0: None
        d = drive.Driver(hub=FakeHub(), recorder=rec)
        # 走行開始(start と同じ順: _clean_params → _speed_ctrl_rules → _launch_rules。running=False)。2-3 の形(計画どおり: 天井 4.1・自動 4.27・発進 0)
        d.params = d._launch_rules(d._speed_ctrl_rules(d._clean_params(
            {'throttle_gain': 0.95, 'speed_ceiling': 1, 'ceil_left_mps': 4.1, 'speed_auto': 1, 'auto_target_mps': 4.27,
             'launch_boost': 0}), False), False)
        stop = threading.Event()
        d.thread = threading.Thread(target=stop.wait, daemon=True)
        d.thread.start()
        # 走行ループの記録の始め(drive.py 1313〜1323 と同じ形: drive_info の params は dict(self.params) の写し)
        drive_info = {'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'dataset': 'x', 'dry_run': False, 'params': dict(d.params)}
        name = rec.start(description='bench=real_bench_20261023 day=2026-11-03 block=8 cand=EXP', tags=[], mode_id=5,
                         drive=drive_info)
        out['running'] = d.running
        # 走っている途中: 「揺れています」→ 7 章どおり OFF(画面のスイッチと同じ API の形)
        d.set_params({'speed_ceiling': 0, 'speed_auto': 0})
        out['server_params_after'] = {k: d.params[k] for k in ('speed_ceiling', 'speed_auto', 'launch_boost')}
        try:
            d.set_params({'speed_ceiling': 1})
            out['reon'] = 'できた'
        except ValueError as e:
            out['reon'] = '断った'
        rec.stop(only_log=name)
        stop.set()
        d.thread.join(1)
        log_dir = raw / name
        lj = json.loads((log_dir / 'log.json').read_text(encoding='utf-8'))
        out['disk_log_json_drive_params'] = {k: lj['drive']['params'][k] for k in ('speed_ceiling', 'speed_auto',
                                                                                    'launch_boost')}
        out['disk_log_json_has_params_end'] = 'params_end' in lj['drive']
        ev = log_dir / 'events.jsonl'
        out['disk_events_jsonl'] = ev.read_text(encoding='utf-8').splitlines() if ev.exists() else '(ファイル無し)'
        sj = json.loads(drive.SETTINGS_PATH.read_text(encoding='utf-8'))
        out['drive_settings_json_after'] = {k: sj['params'][k] for k in ('speed_ceiling', 'speed_auto', 'launch_boost')}

        # 夜の道具 1: runs_1103.main(RAW を tmp に向ける。記録の名前を 11/3 にする)
        dst = raw / '20261103-101500'
        log_dir.rename(dst)
        spec = importlib.util.spec_from_file_location('runs_1103', JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
        r1103 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r1103)
        r1103.RAW = raw
        argv = sys.argv
        sys.argv = ['runs_1103.py', '--day', '20261103']
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r1103.main()
        sys.argv = argv
        out['runs_1103_output'] = buf.getvalue().strip().splitlines()
        # 夜の道具 2: standing_3lap
        import standing_3lap as s3  # noqa: E402
        meta = json.loads((dst / 'log.json').read_text(encoding='utf-8'))
        S = s3.settings_of(meta)
        out['standing_3lap_settings_text'] = s3.settings_text(S)
        # 夜の道具 3: quick_eval_by_params.key_of(発進の前進は走行の始めに 1 回出たとする)
        import quick_eval_by_params as q  # noqa: E402
        key, txt = q.key_of({'settings': S, 'standing': [{'launch': {'on_frames': 5}}]})
        out['quick_eval_by_params_key'] = [list(x) if isinstance(x, tuple) else x for x in key]
        out['quick_eval_by_params_text'] = txt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
