"""recorder.add_event と同じ書き方(open 'a' → write 1 行 → close)の時間を SD カードの上で測る(学習が動いている今)"""
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1]) / 'events_probe.jsonl'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
ts = []
for i in range(n):
    line = json.dumps({'what': 'rear_impact', 'phase': 'reverse', 'backed_s': 0.123, 'run': 'x' * 40, 'cam_epoch_us': 1792757506521747, 'kind': 'recovery_guard', 't_us': i})
    t0 = time.perf_counter()
    with open(p, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    ts.append((time.perf_counter() - t0) * 1000)
    time.sleep(0.01)
ts.sort()
print(json.dumps({'n': n, 'p50_ms': round(ts[n // 2], 3), 'p99_ms': round(ts[int(n * 0.99)], 3), 'p999_ms': round(ts[int(n * 0.999)], 3), 'max_ms': round(ts[-1], 3), 'over10ms': sum(1 for x in ts if x > 10), 'over33ms': sum(1 for x in ts if x > 33)}))
p.unlink()
