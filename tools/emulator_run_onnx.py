#!/usr/bin/env python3
"""JetFormer の学習済みモデル(ONNX)を、Jetson の外(Mac のエミュレーターなど)で動かす例

管理画面の「学習」タブ →「エミュレーター用(zip)」で取ったフォルダの中で使う(このファイルも zip に入っている)。
JetFormer のリポジトリは要らない(numpy・onnxruntime と、画像を読むときだけ opencv-python か Pillow)。

    pip install onnxruntime numpy opencv-python
    python run_onnx_example.py <展開したフォルダ>                      # ダミーの入力で1回推論
    python run_onnx_example.py <フォルダ> --image frame.jpg --erpm -4000 --mode 5 --lookahead 3

エミュレーターから使うときは JetFormerOnnx(フォルダ).predict(image_bgr, tof_raw, imu, vehicle, mode_id) を呼ぶ。
入力の作り方は Jetson の jetformer/train_data.py normalize_inputs と同じ(io_spec.json にも書いてある)。
"""

import argparse
import json
import os
import sys

import numpy as np

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), np.float32)
GRAVITY = 9.80665
DIST_MASK, INVALID, CONF_SHIFT, CONF_MAX = 0x1FFF, 0x1FFF, 13, 7


def _resize_bgr(image, size):
    if image.shape[0] == size and image.shape[1] == size:
        return image
    try:
        import cv2
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(image[:, :, ::-1]).resize((size, size), Image.BILINEAR))[:, :, ::-1]


class JetFormerOnnx(object):
    def __init__(self, folder, providers=None):
        import onnxruntime as ort
        with open(os.path.join(folder, 'model_config.json'), encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.sess = ort.InferenceSession(os.path.join(folder, 'jetformer.onnx'),
                                         providers=providers or ['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.center = float(self.cfg.get('pwm_center', 1500.0))
        self.scale = float(self.cfg.get('pwm_scale', 500.0))
        self.tof_max = float(self.cfg.get('tof_max_mm', 4000.0))
        self.erpm_scale = float(self.cfg.get('erpm_scale', 20000.0))
        self.num_modes = int(self.cfg.get('num_modes', 8))
        self.use_command = bool(self.cfg.get('command_input', True))
        self.size = int(self.sess.get_inputs()[0].shape[2] or 224)

    def inputs(self, image_bgr, tof_raw, imu, vehicle, mode_id):
        """センサーの値 → モデルの入力 {名前: 配列}

        image_bgr: uint8 [H,W,3](BGR。1920x1080 の全体を切り取らずに縮小したもの)
        tof_raw:   uint16 [42,54](下位13bit = 距離 mm、0x1FFF = 無効、上位3bit = 信頼度)
        imu:       [10,6] 直近 10 サンプル(100Hz、古い順)[ax, ay, az (m/s^2), gx, gy, gz (rad/s)]、+x 右・+y 前・gz 正 = 左回り
        vehicle:   [steering_us, throttle_us, mode_us, erpm](erpm は負 = 前進)
        mode_id:   走行ジャンル(model_config.json の trained_modes のどれか。例 5 = 低速走行)"""
        img = _resize_bgr(np.asarray(image_bgr)[:, :, :3], self.size)[:, :, ::-1]          # BGR -> RGB
        img = ((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        raw = np.asarray(tof_raw, np.uint16)
        dist = (raw & DIST_MASK).astype(np.float32)
        dist[dist == INVALID] = 0.0
        tof = np.stack([np.clip(dist / self.tof_max, 0.0, 1.0), (raw >> CONF_SHIFT).astype(np.float32) / CONF_MAX])
        imu_n = np.asarray(imu, np.float32).copy()
        imu_n[:, :3] /= GRAVITY
        v = np.asarray(vehicle, np.float32)
        state = np.array([(v[0] - self.center) / self.scale, (v[1] - self.center) / self.scale,
                          np.clip(v[3] / self.erpm_scale, -3.0, 3.0)], np.float32)
        if not self.use_command:
            state[:2] = 0.0
        mode = np.zeros(self.num_modes, np.float32)
        mode[int(mode_id)] = 1.0
        feed = {'image': img[None], 'tof': tof[None].astype(np.float32), 'imu': imu_n[None], 'state': state[None],
                'mode': mode[None]}
        return {k: feed[k] for k in self.input_names}

    def predict(self, image_bgr, tof_raw, imu, vehicle, mode_id):
        """戻り値: [10,2] の µs(1〜10 フレーム先の [ステアリング, スロットル])と、ONNX の全出力 {名前: 配列}"""
        outs = self.sess.run(None, self.inputs(image_bgr, tof_raw, imu, vehicle, mode_id))
        named = dict(zip([o.name for o in self.sess.get_outputs()], outs))
        actions = named.get('actions', outs[0])[0] * self.scale + self.center
        return actions, named


def apply_gains(raw_us, cfg, steer_gain=1.3, throttle_gain=1.0, trim_us=0.0):
    """Jetson の走行と同じゲインの掛け方(スロットルのゲインは前進側 = 中立より小さい側だけ)"""
    sc = float(cfg.get('steer_center_us') or 1443.0)
    tn = float(cfg.get('throttle_neutral_us') or 1500.0)
    steer = sc + trim_us + steer_gain * (raw_us[0] - sc)
    dt = raw_us[1] - tn
    throttle = tn + (throttle_gain * dt if dt < 0 else dt)
    return float(np.clip(steer, 1000, 2000)), float(np.clip(throttle, 1000, 2000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', help='zip を展開したフォルダ(jetformer.onnx と model_config.json がある所)')
    ap.add_argument('--image', help='カメラ画像(無ければ灰色の画像)')
    ap.add_argument('--erpm', type=float, default=-4000.0, help='モーターの eRPM(負 = 前進)')
    ap.add_argument('--mode', type=int, default=None, help='走行ジャンル(既定: 学習したジャンルの最後)')
    ap.add_argument('--lookahead', type=int, default=3)
    args = ap.parse_args()
    m = JetFormerOnnx(args.folder)
    mode = args.mode if args.mode is not None else (m.cfg.get('trained_modes') or [5])[-1]
    if args.image:
        try:
            import cv2
            image = cv2.imread(args.image)
        except ImportError:
            from PIL import Image
            image = np.asarray(Image.open(args.image).convert('RGB'))[:, :, ::-1]
        if image is None:
            sys.exit('画像を読めません: %s' % args.image)
    else:
        image = np.full((224, 224, 3), 128, np.uint8)
    tof = np.full((42, 54), (4 << CONF_SHIFT) | 1500, np.uint16)       # 1.5m 先の壁・信頼度 4
    imu = np.zeros((10, 6), np.float32)
    imu[:, 2] = GRAVITY
    vehicle = [1500.0, 1500.0, 2000.0, args.erpm]
    actions, named = m.predict(image, tof, imu, vehicle, mode)
    k = min(max(args.lookahead, 1), len(actions)) - 1
    print('入力:', {k_: list(v.shape) for k_, v in m.inputs(image, tof, imu, vehicle, mode).items()})
    print('出力:', {k_: list(v.shape) for k_, v in named.items()})
    print('予測 µs(1〜10 フレーム先の [ステア, スロットル]):')
    for i, (s, t) in enumerate(actions):
        print('  %2d: %7.1f %7.1f%s' % (i + 1, s, t, '  ← 先読み %d' % args.lookahead if i == k else ''))
    print('ゲイン(ステア 1.3・速度 1.0)をかけた出力 µs:', apply_gains(actions[k], m.cfg))


if __name__ == '__main__':
    main()
