# -*- coding: utf-8 -*-
"""計測の共通の道具: 結果の JSON(途中から再開できるよう段ごとに書く)・GPU を使っている別のプロセスの確認・実データの入力"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
WORK = os.path.join(HERE, 'work')
REPO = '/home/jetson/jetformer'
DATASET = os.path.join(REPO, 'data', 'datasets', 'clean_driving_20261011')
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {} if default is None else default


def save_json(path, obj):
    """書きかけで電源が落ちても壊れないよう、一時ファイルに書いてから置き換える"""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def gpu_users(exclude=()):
    """GPU で計算している別のプロセス [(pid, コマンド)]。CUDA の文脈を作ると /dev/nvgpu/igpu0/ctrl を開き、
    libcuda.so を読み込む(Orin で確認)。両方そろったものだけを数える"""
    me = {os.getpid(), os.getppid()} | set(exclude)
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) in me:
            continue
        try:
            fds = os.listdir('/proc/%s/fd' % pid)
        except OSError:
            continue
        busy = False
        for fd in fds:
            try:
                if os.readlink('/proc/%s/fd/%s' % (pid, fd)).startswith('/dev/nvgpu'):
                    busy = True
                    break
            except OSError:
                pass
        if busy:        # 画面(gnome-shell など)も /dev/nvgpu を開くので、CUDA(libcuda.so)を読み込んでいるものだけ数える
            try:
                with open('/proc/%s/maps' % pid) as f:
                    busy = 'libcuda.so' in f.read()
            except OSError:
                busy = False
        if busy:
            try:
                cmd = open('/proc/%s/cmdline' % pid, 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                cmd = '?'
            out.append((int(pid), cmd[:160]))
    return out


def wait_gpu_free(poll_s=30, max_wait_s=6 * 3600, log=print):
    """ほかのプロセスが GPU を使っている間は待つ(GPU は1プロセスずつの決まり)"""
    t0 = time.time()
    while True:
        users = gpu_users()
        if not users:
            return True
        if time.time() - t0 > max_wait_s:
            return False
        log('GPU を使っているプロセスがあるので待ちます: %s' % users)
        time.sleep(poll_s)


def gpu_freq_mhz():
    try:
        with open('/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq') as f:
            return int(f.read()) / 1e6
    except (OSError, ValueError):
        return None


def real_inputs(n=16, num_modes=6, seed=0):
    """検証の分割の実際のフレーム n 個を、走行と同じ正規化(今の操作値なし)で → (image, tof, imu, state, mode) の numpy"""
    import cv2
    from jetformer.train_data import normalize_inputs
    splits = load_json(os.path.join(DATASET, 'splits.json'))
    rng = np.random.default_rng(seed)
    scenes = list(splits['val'])
    out = [[], [], [], [], []]
    for k in range(n):
        name = scenes[k % len(scenes)]
        with np.load(os.path.join(DATASET, 'aligned', name + '.npz')) as z:
            i = int(rng.integers(0, len(z['timestamps_us'])))
            img = cv2.imread(os.path.join(DATASET, str(z['cam_files'][i])))
            a = normalize_inputs(img, z['tof'][i], z['imu'][i], z['vehicle'][i], use_command=False)
            m = np.zeros(num_modes, np.float32)
            m[int(z['mode'][i]) if int(z['mode'][i]) < num_modes else 0] = 1.0
        for j, v in enumerate(list(a) + [m]):
            out[j].append(v)
    return tuple(np.stack(v).astype(np.float32) for v in out)
