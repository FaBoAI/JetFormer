"""エンジンの確認(train/engine_check/<名前>.json)を、各学習結果のフォルダーの engine_check.json に写す(CPU・GPU なし)。
走行タブ(drive.model_info → engine_check_of)が読み、確認に落ちたエンジンに警告を出す(走行できるかどうかは変えない)。
エンジンの大きさ・更新時刻も書くので、エンジンを作り直すと古い確認は表示されない。

    cd ~/jetformer && python /home/jetson/jetformer_reports/policy_zoo_20261012/train/write_engine_check.py

確認の後にエンジンが変わっていたら(エンジンの更新時刻 > 確認の時刻)写さない。出来ているものは中身が同じなら書かない。
"""
import datetime
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/jetson/jetformer/data/runs')
ENGINE_FILE = 'jetformer_fp16.engine'
GATE_US = 2.5
# 方策として(2つの種の両方で)ゲートに落ちたときの注記(spec §5.1。2026-10-12 のレビューで、保留データの数字を出す前に決めた)
POLICY_NOTE = {
    'dinov2': 'dinov2 は種 1 が FP16 の精度の確認(2.5 µs)に落ちたため、方策の比較では走行の候補にしない'
              '(spec §5.1。先に決めた代わりの bf16・fp32 もだめ。FP16 + 正規化・Softmax だけ FP32 は事後の参考)',
}


def main():
    plan = json.load(open(HERE / 'plan.json'))
    runs = plan['runs']                                   # {'<方策>/s<種>': 学習結果の名前}
    names = list(runs.values()) if isinstance(runs, dict) else list(runs)
    for name in names:
        src = HERE / 'engine_check' / (name + '.json')
        run_dir = RUNS / name
        if not src.exists() or not (run_dir / ENGINE_FILE).exists():
            print('飛ばす(確認かエンジンが無い):', name)
            continue
        c = json.load(open(src))
        st = (run_dir / ENGINE_FILE).stat()
        checked = datetime.datetime.strptime(c['checked_at'], '%Y-%m-%d %H:%M:%S').timestamp()
        if st.st_mtime > checked:
            print('飛ばす(確認の後にエンジンが変わった):', name)
            continue
        acc, lat = c.get('accuracy') or {}, c.get('latency_ms') or {}
        policy = c.get('policy')
        out = {
            'ok': bool(c.get('all_gates')), 'accuracy_ok': bool(acc.get('gate_actions_2p5us')),
            'speed_ok': bool(lat.get('gate_run_numpy')), 'actions_max_us': acc.get('actions_max_us'),
            'actions_p99_us': acc.get('actions_p99_us'), 'gate_us': GATE_US,
            'run_numpy_median_ms': lat.get('run_numpy_median'), 'run_numpy_p99_ms': lat.get('run_numpy_p99'),
            'checked_at': c['checked_at'], 'note': POLICY_NOTE.get(policy, ''),
            'engine': {'file': ENGINE_FILE, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
            'source': str(src), 'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。'
                                          '速さは 100 回の空回しの後 600 回(train/engine_load_check.py)',
        }
        dst = run_dir / 'engine_check.json'
        if dst.exists() and json.load(open(dst)) == out:
            print('同じ:', name)
            continue
        tmp = str(dst) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, dst)
        print('書いた:', name, 'ok' if out['ok'] else '確認に落ちた', out['actions_max_us'])


if __name__ == '__main__':
    main()
