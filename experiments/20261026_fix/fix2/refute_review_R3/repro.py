"""R3 の確かめ: 本物の Recorder・clock で、drive.launch_note と同じ書き方の events.jsonl と AI_CMD を作り、
standing_3lap.launch_events / launch_record・diag_launch/collect.launch_events で読む(装置なし・偽の hub)"""
import json, sys, time, tempfile, shutil
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'experiments/20261026_fix/diag_launch'))
from jetformer import config, clock, recorder
from jetformer.sensors.base import Sample
import standing_3lap as s3
import collect

class Ch:
    count, connected = 0, True
    def __init__(self): self.subs = []
    def subscribe(self, f): self.subs.append(f)
    def unsubscribe(self, f): self.subs.remove(f)
class Hub(dict):
    channels = ('CAM_FRONT',)
    def start(self): pass

tmp = Path(tempfile.mkdtemp(prefix='r3_', dir=str(ROOT / 'experiments/20261026_fix/fix2/refute_review_R3')))
config.RAW_DIR = tmp
hub = Hub(CAM_FRONT=Ch())
rec = recorder.Recorder(hub)
rec._warm_up = lambda timeout=8.0: None
name = rec.start(description='r3')
img = np.zeros((8, 8, 3), np.uint8)
t_cam = []
for i in range(60):                     # 2 秒・30 fps。AI の区間
    t_ns = clock.now_ns()
    rec._on_sample('CAM_FRONT', Sample(t_ns, {'image': img, 'seq': i}))
    on = 10 <= i < 30
    rec.add_ai(t_ns, {'pred': np.zeros((10, 2)), 'raw': [0, 0], 'out': [1500, 1600], 'gains': [1, 1], 'trim': [0, 0],
                      'lookahead': 0, 'mode_id': 5, 'hold': False, 'select': True, 'latency_ms': 1, 'infer_ms': 1,
                      'dry_run': False, 'assist': 5 if on else 0, 'launch_on': on, 'launch_end': 1 if i == 30 else 0})
    # drive.launch_note と同じ: info['cam_t_us'] = int(cam_t_ns // 1000) → recorder.add_event('launch_boost', info)
    if i == 10:
        rec.add_event('launch_boost', dict(event='start', us=250.0, run='x', cam_t_us=int(t_ns // 1000)))
    if i == 30:
        rec.add_event('launch_boost', dict(event='end', code=1, reason='速さに届いた', run='x', cam_t_us=int(t_ns // 1000)))
    if i == 45:   # 別の区間なら skip(ここでは同じ区間の中に置いて理由が拾えるかを見る)
        rec.add_event('launch_boost', dict(event='skip', why='near', reason='前が近い(TTC の中央値 < 3.3 秒)', run='x',
                                           cam_t_us=int(t_ns // 1000)))
    time.sleep(1 / 30)
rec.stop()
time.sleep(0.3)
ev_lines = (tmp / name / 'events.jsonl').read_text(encoding='utf-8').splitlines()
print('events.jsonl[0]:', ev_lines[0])
sc = sorted(p for p in (tmp / name).iterdir() if p.is_dir())
A = dict(np.load(sc[0] / 'AI_CMD.npz'))
t0 = int(A['timestamps_us'][0])
print('AI_CMD.timestamps_us[0] =', t0, ' events cam_t_us =', json.loads(ev_lines[0])['cam_t_us'],
      ' t_us =', json.loads(ev_lines[0])['t_us'])
print('uptime_s=%.0f  diff_s=%.1f' % (time.monotonic(), (json.loads(ev_lines[0])['cam_t_us'] - t0) / 1e6))
evs = s3.launch_events(name, tmp)
print('standing_3lap.launch_events:', evs)
lr = s3.launch_record(A, t0, 0.0, 2.0, evs)
print('launch_record.events:', lr['events'], ' on_frames', lr['on_frames'], ' end', lr['end_reason'])
# 正しく epoch に直したら
evs2 = [(clock.to_epoch_us(tu * 1000), e, w) for tu, e, w in evs]
print('epoch に直せば:', s3.launch_record(A, t0, 0.0, 2.0, evs2)['events'])
# 出なかった本: launch_on が全部 False で skip の出来事あり
A2 = dict(A); A2['launch_on'] = np.zeros_like(A['launch_on']); A2['launch_end'] = np.zeros_like(A['launch_end'])
A2['assist'] = np.zeros_like(A['assist'])
print('出なかった本 launch_text:', s3.launch_text(s3.launch_record(A2, t0, 0.0, 2.0, evs)),
      '/ epoch なら:', s3.launch_text(s3.launch_record(A2, t0, 0.0, 2.0, evs2)))
collect.RAW = tmp
ce = collect.launch_events(name, t0)
print('collect.launch_events:', ce)
print('collect の窓(t0-0.2..t1=2)に入るもの:', [e for e in ce if -0.2 <= e[0] <= 2.0])
shutil.rmtree(tmp)
