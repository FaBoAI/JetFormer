"""学習済みモデルを、Jetson の外(Mac のエミュレーターなど)で使うための書き出し

1つの run を zip(エミュレーター用の一式)にして、ブラウザや HTTP でそのまま取れるようにする。

    <run 名>/
      jetformer.onnx          推論用(TensorRT のエンジンは Jetson 専用なので入れない)
      model_config.json       学習の設定(ジャンル・正規化の定数・操作値を入れるかなど)
      result.json             学習の結果(検証誤差など。あれば)
      io_spec.json            入力の作り方と出力の読み方(下の io_spec()。ONNX の入出力の形はファイルから読む)
      README.md               使い方(日本語)
      run_onnx_example.py     onnxruntime で動かす例(tools/emulator_run_onnx.py と同じもの)
      best.pth                PyTorch の重み(pth=True のときだけ)

zip はその場で作りながら送る(Jetson にファイルを作らない)。
"""

import datetime
import json
import zipfile
from pathlib import Path

from .training import RUNS_DIR

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent / 'tools' / 'emulator_run_onnx.py'
ONNX_NAME = 'jetformer.onnx'
ENGINE_NAME = 'jetformer_fp16.engine'
GRAVITY = 9.80665


def _read_json(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _run_dir(name):
    if not name or '..' in name or '/' in name or name.startswith('.'):
        raise KeyError(name)
    d = RUNS_DIR / name
    if not (d / 'model_config.json').exists() or not (d / ONNX_NAME).exists():
        raise KeyError(name)
    return d


def list_models():
    """ONNX と model_config.json がある学習済みモデルの一覧(新しい順)"""
    out = []
    if not RUNS_DIR.exists():
        return out
    for d in RUNS_DIR.iterdir():
        if d.name.startswith(('.', '_')) or not (d / 'model_config.json').exists() or not (d / ONNX_NAME).exists():
            continue
        cfg = _read_json(d / 'model_config.json') or {}
        res = _read_json(d / 'result.json') or {}
        run = _read_json(d / 'run.json') or {}
        onnx = d / ONNX_NAME
        out.append({
            'name': d.name,
            'policy': cfg.get('policy', 'jetformer_v0'),
            'dataset': cfg.get('dataset') or run.get('dataset'),
            'command_input': cfg.get('command_input', True),
            'trained_modes': cfg.get('trained_modes', []),
            'best_val_us': res.get('best_val_us'),
            'best_epoch': res.get('best_epoch'),
            'created_at': run.get('created_at') or datetime.datetime.fromtimestamp(onnx.stat().st_mtime).isoformat(timespec='seconds'),
            'onnx_bytes': onnx.stat().st_size,
            'has_pth': (d / 'best.pth').exists(),
            'pth_bytes': (d / 'best.pth').stat().st_size if (d / 'best.pth').exists() else 0,
            # Jetson で走らせられるか(オフライン強化学習の実験のように、エンジンの名前を変えて走行から外したものは False)
            'drivable': (d / ENGINE_NAME).exists(),
        })
    return sorted(out, key=lambda m: (m['created_at'] or '', m['name']), reverse=True)


def _onnx_io(path):
    """ONNX の入出力の名前・形・型(重みは読まない)"""
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    types = {1: 'float32', 10: 'float16', 7: 'int64', 6: 'int32', 9: 'bool'}

    def desc(v):
        t = v.type.tensor_type
        return {'name': v.name, 'shape': [d.dim_value if d.dim_value else (d.dim_param or None) for d in t.shape.dim],
                'dtype': types.get(t.elem_type, str(t.elem_type))}
    inits = {i.name for i in m.graph.initializer}
    return [desc(v) for v in m.graph.input if v.name not in inits], [desc(v) for v in m.graph.output]


def io_spec(name):
    """入力の作り方・出力の読み方(エミュレーターがセンサーの値からモデルの入力を作るための仕様)"""
    from . import model as jm
    from . import tof_format
    d = _run_dir(name)
    cfg = _read_json(d / 'model_config.json') or {}
    inputs, outputs = _onnx_io(d / ONNX_NAME)
    center = float(cfg.get('pwm_center', jm.PWM_CENTER))
    scale = float(cfg.get('pwm_scale', jm.PWM_SCALE))
    num_modes = int(cfg.get('num_modes', 8))
    modes = {str(m['id']): m['name'] for m in cfg.get('modes', [])}
    use_command = bool(cfg.get('command_input', True))
    notes = {
        'image': {'from': 'カメラ IMX219 の 1920x1080(sensor_mode 2)全体を、切り取らずに 224x224 へ縮小した BGR 画像(縦横比は変わる)',
                  'steps': ['BGR -> RGB', '/ 255', '(x - mean) / std(ImageNet)', 'HWC -> CHW', 'バッチの次元を足す'],
                  'mean': list(jm.IMAGENET_MEAN), 'std': list(jm.IMAGENET_STD), 'size': jm.IMAGE_SIZE},
        'tof': {'from': 'VL53L9 の 42x54 ゾーン(binning 2)の生の値 uint16。下位13bit = 距離 mm(0x1FFF = 無効 -> 0)、上位3bit = 信頼度 0-7',
                'channels': ['距離: clip(距離mm / %g, 0, 1)。無効は 0' % float(cfg.get('tof_max_mm', jm.TOF_MAX_MM)),
                             '信頼度: (生の値 >> %d) / %d' % (tof_format.CONF_SHIFT, tof_format.CONF_MAX)],
                'shape': list(jm.TOF_SHAPE), 'max_mm': float(cfg.get('tof_max_mm', jm.TOF_MAX_MM))},
        'imu': {'from': 'ICM-20948 の直近 %d サンプル(100Hz、古い順)。各行 [ax, ay, az, gx, gy, gz]' % jm.IMU_STEPS,
                'units': '加速度 m/s^2 を %.5f で割る(G 単位)。ジャイロは rad/s のまま' % GRAVITY,
                'axes': 'センサーの取り付けの向きのまま: +x = 右、+y = 前、z = 上。gz が正 = 左回り(左旋回)'},
        'state': {'from': '[今のステアリング µs, 今のスロットル µs, モーターの eRPM]',
                  'formula': ['(steering_us - %g) / %g' % (center, scale), '(throttle_us - %g) / %g' % (center, scale),
                              'clip(erpm / %g, -3, 3)' % float(cfg.get('erpm_scale', jm.ERPM_SCALE))],
                  'command_input': use_command,
                  'note': ('今の操作値を入れるモデル' if use_command else
                           'このモデルは今の操作値を入れない: 先頭2つは常に 0(回転数だけを入れる)'),
                  'erpm_sign': 'eRPM は負 = 前進。速さの目安 m/s = -eRPM x 0.5e-3(タイヤ 65mm)'},
        'mode': {'from': '走行ジャンルの one-hot(長さ %d)' % num_modes, 'trained_modes': cfg.get('trained_modes', []),
                 'names': modes, 'note': '学習したジャンル(trained_modes)のどれか1つを 1 にする。例: 5 = 低速走行'},
    }
    out_notes = {'actions': {
        'shape': '[1, %d, 2] = 1〜%d フレーム先(30fps、約 %.2f 秒先まで)の [ステアリング, スロットル]' % (
            jm.FUTURE_STEPS, jm.FUTURE_STEPS, jm.FUTURE_STEPS / 30.0),
        'to_us': 'µs = 値 x %g + %g' % (scale, center),
        'lookahead': 'Jetson の走行では k フレーム先(既定 3)の値を使う: actions[0, k-1]',
        'gains': ('走行時のゲイン: ステア = 中心 + トリム + ゲイン x (予測 - 中心)。スロットルのゲインは前進側(中立より小さい側)だけ'),
        'conventions': {'steer_center_us': cfg.get('steer_center_us'), 'throttle_neutral_us': cfg.get('throttle_neutral_us'),
                        'steering': 'µs が中心より大きい = 左', 'throttle': 'µs が中立より小さい = 前進(大きい = ブレーキ・後退)'}}}
    # TCP / TransFuser++ 型の追加の出力(走行の既定 tcp_mode 0 は actions だけを使う)
    pp = cfg.get('postprocess') if isinstance(cfg.get('postprocess'), dict) else {}
    bins = pp.get('speed_bins') or [round(0.5 * k, 1) for k in range(10)]
    out_notes['waypoints'] = {
        'shape': '[1, 10, 2] = 撮像時刻の車の位置を原点にした、0.1 秒刻みで 1 秒先までの通過点 [x, y](m)',
        'axes': '自車の座標: x = 前、y = 左(m)', 'dt_s': pp.get('wp_dt', 0.1)}
    out_notes['speed_logits'] = {
        'shape': '[1, %d] = 目標速度のクラス(%s m/s)の logits' % (len(bins), '・'.join('%g' % b for b in bins)),
        'to_speed': '目標速度 m/s = Σ softmax(logits) × クラスの速度(期待値)', 'speed_bins': bins}
    post_note = None
    if pp.get('kind') == 'tcp_controller':
        post_note = ('制御器は jetformer/policies/controller.py(numpy、TCPPost)。model_config.json の postprocess に較正の値がある。'
                     '走行の既定(tcp_mode 0)は actions をそのまま使い、速度の上限(speed_cap)を掛けたときだけ前進を弱める')
    return {'name': name, 'policy': cfg.get('policy', 'jetformer_v0'), 'onnx': ONNX_NAME, 'opset_note': 'バッチは 1 固定',
            'inputs': [dict(i, **{'how': notes.get(i['name'], {})}) for i in inputs],
            'outputs': [dict(o, **{'how': out_notes.get(o['name'], {})}) for o in outputs],
            'postprocess_note': post_note, 'model_config': cfg}


def readme(name, spec):
    ins = '\n'.join('- `%s` %s %s' % (i['name'], i['shape'], i['dtype']) for i in spec['inputs'])
    outs = '\n'.join('- `%s` %s %s' % (o['name'], o['shape'], o['dtype']) for o in spec['outputs'])
    st = next((i['how'] for i in spec['inputs'] if i['name'] == 'state'), {})
    return '''# %s(JetFormer の学習済みモデル・エミュレーター用)

- ポリシー: `%s`  ・  学習データ: `%s`  ・  学習したジャンル: %s
- %s

## ファイル
- `jetformer.onnx` — 推論用(バッチ 1 固定)。Mac では onnxruntime(CPU / CoreML)で動く
- `model_config.json` — 学習の設定と正規化の定数
- `io_spec.json` — 入力の作り方・出力の読み方(センサーの値 → 入力)
- `run_onnx_example.py` — onnxruntime で動かす例(`pip install onnxruntime numpy opencv-python`)
- `best.pth` — PyTorch の重み(付けたときだけ。JetFormer のリポジトリの jetformer/ と一緒に使う)

## 入力
%s

## 出力
%s

## 動かし方(Mac)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install onnxruntime numpy opencv-python
python run_onnx_example.py .                     # ダミーの入力で1回推論して、µs に直した値を表示
python run_onnx_example.py . --image frame.jpg --erpm -4000 --mode 5
```

## 注意
- カメラ画像は 1920x1080 の全体を切り取らずに 224x224 へ縮小したもの(縦横比が変わる)。エミュレーターの画像も同じ作り方にする
- ステアリングは中心より大きい µs が左。スロットルは中立より小さい µs が前進。eRPM は負が前進
- IMU はセンサーの取り付けの向きのまま(+x 右、+y 前、gz 正 = 左回り)
- 実車ではこの出力に、ゲイン・トリム・ToF ガード・自動リカバリー・人の交代などの安全の仕組みがかかる(エミュレーター側で再現する)
- どのモデルの種類(ポリシー)も `actions` を出す。TCP 型の `waypoints`・`speed_logits` は追加の出力(走行の既定は `actions` だけを使う)%s

書き出し: %s
''' % (name, spec['policy'], spec['model_config'].get('dataset'), spec['model_config'].get('trained_modes'),
       st.get('note', ''), ins, outs,
       ('\n- ' + spec['postprocess_note']) if spec.get('postprocess_note') else '',
       datetime.datetime.now().isoformat(timespec='seconds'))


class _Sink(object):
    """zipfile の書き出し先(シークできない)。書かれたバイトを溜め、iter_bundle_zip が少しずつ取り出す"""

    def __init__(self):
        self.parts, self.size, self.pos = [], 0, 0

    def write(self, b):
        b = bytes(b)
        self.parts.append(b)
        self.size += len(b)
        self.pos += len(b)
        return len(b)

    def tell(self):
        return self.pos

    def flush(self):
        pass

    def take(self):
        out = b''.join(self.parts)
        self.parts, self.size = [], 0
        return out


def bundle_files(name, pth=False):
    """zip に入れるもの [(zip の中の名前, ローカルのパス or バイト列)]"""
    d = _run_dir(name)
    spec = io_spec(name)
    items = [('%s/%s' % (name, ONNX_NAME), d / ONNX_NAME), ('%s/model_config.json' % name, d / 'model_config.json')]
    if (d / 'result.json').exists():
        items.append(('%s/result.json' % name, d / 'result.json'))
    items.append(('%s/io_spec.json' % name, json.dumps(spec, ensure_ascii=False, indent=1).encode('utf-8')))
    items.append(('%s/README.md' % name, readme(name, spec).encode('utf-8')))
    if EXAMPLE.exists():
        items.append(('%s/run_onnx_example.py' % name, EXAMPLE))
    if pth and (d / 'best.pth').exists():
        items.append(('%s/best.pth' % name, d / 'best.pth'))
    return items


def iter_bundle_zip(name, pth=False, chunk=4 << 20):
    """エミュレーター用の一式を zip にしながら、少しずつバイト列を返す"""
    items = bundle_files(name, pth)            # 名前の確認はここで(ジェネレーターを回す前に KeyError を出す)

    def gen():
        sink = _Sink()
        with zipfile.ZipFile(sink, 'w', allowZip64=True) as z:
            for arc, src in items:
                big = not isinstance(src, bytes)
                info = zipfile.ZipInfo(arc, date_time=datetime.datetime.now().timetuple()[:6])
                info.compress_type = zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with z.open(info, 'w', force_zip64=big) as w:
                    if big:
                        with open(src, 'rb') as f:
                            for block in iter(lambda: f.read(1 << 20), b''):
                                w.write(block)
                                if sink.size >= chunk:
                                    yield sink.take()
                    else:
                        w.write(src)
                if sink.size >= chunk:
                    yield sink.take()
        rest = sink.take()
        if rest:
            yield rest
    return gen()


def bundle_bytes_estimate(name, pth=False):
    """zip のおおよその大きさ(画面の表示用)"""
    d = _run_dir(name)
    n = (d / ONNX_NAME).stat().st_size + (64 << 10)
    if pth and (d / 'best.pth').exists():
        n += (d / 'best.pth').stat().st_size
    return n

