"""作り直した dinov2 のエンジン(engine.precision = fp16_norm_fp32)の確認を、各学習結果のフォルダーの engine_check.json に写す
(CPU・GPU なし。train/write_engine_check.py と同じ形)。走行タブ(drive.model_info → engine_check_of)が読む。

    cd ~/jetformer && python /home/jetson/jetformer_reports/policy_zoo_20261012/eval/write_engine_check_dinov2.py

確認(data/zoo/policy_zoo_20261012/engine/dinov2_s<種>.json)の後にエンジンが変わっていたら写さない(大きさ・更新時刻で確かめる)。
"""
import json
import os
from pathlib import Path

RUNS = Path('/home/jetson/jetformer/data/runs')
ZOO = Path('/home/jetson/jetformer/data/zoo/policy_zoo_20261012')
ENGINE_FILE = 'jetformer_fp16.engine'
GATE_US = 2.5
NOTE = ('FP16 + 正規化・Softmax だけ FP32(fp16_norm_fp32)で作り直したエンジン(2026-10-12 評価の段)。確認に合格し走行タブで選べるが、'
        '方策の比較の判定では、先に決めた FP16 のエンジンの確認(種 1 が 2.85 µs で不合格)により候補に入れていない(zoo_rules.json D7)')


def main():
    for s in (0, 1):
        src = ZOO / 'engine' / ('dinov2_s%d.json' % s)
        c = json.load(open(src, encoding='utf-8'))
        run_dir = RUNS / c['run']
        st = (run_dir / ENGINE_FILE).stat()
        eng = c.get('engine') or {}
        if eng.get('size') != st.st_size or eng.get('mtime_ns') != st.st_mtime_ns:
            print('飛ばす(確認の後にエンジンが変わった):', c['run'])
            continue
        acc, lat = c['accuracy'], c['latency_ms']
        out = {'ok': bool(c['all_gates']), 'accuracy_ok': bool(acc['gate_actions_2p5us']), 'speed_ok': bool(lat['gate_run_numpy']),
               'actions_max_us': acc['actions_max_us'], 'actions_p99_us': acc['actions_p99_us'], 'gate_us': GATE_US,
               'run_numpy_median_ms': lat['run_numpy_median'], 'run_numpy_p99_ms': lat['run_numpy_p99'],
               'checked_at': c['checked_at'], 'note': NOTE,
               'engine': {'file': ENGINE_FILE, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
               'engine_precision': c.get('engine_precision'), 'source': str(src),
               'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。'
                         '速さは 100 回の空回しの後 600 回(tools/policy_zoo.py engine-check)'}
        dst = run_dir / 'engine_check.json'
        tmp = str(dst) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, dst)
        print('書いた:', c['run'], 'ok' if out['ok'] else '確認に落ちた', out['actions_max_us'])


if __name__ == '__main__':
    main()
