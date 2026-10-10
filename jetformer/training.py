"""Jetson上での学習の実行管理(管理画面から使う)

tools/train.py を別プロセスで起動し、「@@{JSON}」の進捗行を読んで状態を保持する。
終了後は(指定があれば)TensorRTエンジンを作る。結果は data/runs/<run>/ に残る:
  best.pth / jetformer.onnx / model_config.json / result.json / train.log / jetformer_fp16.engine
"""

import collections
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from . import config

RUNS_DIR = config.DATA_DIR / 'runs'
TRAIN_SCRIPT = config.ROOT / 'tools' / 'train.py'
ENGINE_SCRIPT = config.ROOT / 'tools' / 'build_engine.py'
# 学習の最大使用量は約3GB(Orin Nano 8GB・スワップなし)。足りないまま始めるとメモリが尽きて本体ごと止まる
MIN_FREE_MB = 3500


def mem_available_mb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    return 0
ARTIFACTS = ('best.pth', 'jetformer.onnx', 'jetformer_fp16.engine', 'model_config.json', 'result.json', 'train.log')


class TrainingManager(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.state = None          # 実行中/直近の学習の状態(dict)
        self.listeners = []        # 出来事の通知先 cb(event, **info)(音声案内など)

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None or bool(self.state and self.state['phase'] == 'engine')

    def start(self, dataset, stride=3, epochs=30, batch=64, lr=3e-4, pretrained=True, build_engine=True, command_input=True,
              mask_impacts=True):
        with self.lock:
            if self.running:
                raise RuntimeError('すでに学習中です')
            ds_root = config.DATASET_DIR / dataset
            if '..' in dataset or not (ds_root / 'dataset_card.json').exists():
                raise ValueError('データセットがありません: %s' % dataset)
            stride, epochs, batch = int(stride), int(epochs), int(batch)
            if not (1 <= stride <= 30 and 1 <= epochs <= 500 and 1 <= batch <= 256):
                raise ValueError('設定値が範囲外です')
            free = mem_available_mb()
            if free < MIN_FREE_MB:
                raise RuntimeError('空きメモリが足りません(%.1f GB、学習には %.1f GB 必要)。ほかの学習・推論のプログラムを止めるか、'
                                   '管理画面のサービスを再起動してください' % (free / 1024, MIN_FREE_MB / 1024))
            name = '%s-s%d-%s' % (dataset, stride, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
            out = RUNS_DIR / name
            out.mkdir(parents=True)
            params = {'dataset': dataset, 'stride': stride, 'epochs': epochs, 'batch': batch, 'lr': float(lr),
                      'pretrained': bool(pretrained), 'build_engine': bool(build_engine), 'command_input': bool(command_input),
                      'mask_impacts': bool(mask_impacts)}
            with open(out / 'run.json', 'w', encoding='utf-8') as f:
                json.dump(dict(params, name=name, created_at=datetime.datetime.now().isoformat(timespec='seconds')),
                          f, ensure_ascii=False, indent=1)
            cmd = [sys.executable, str(TRAIN_SCRIPT), '--data', str(ds_root), '--out', str(out),
                   '--stride', str(stride), '--epochs', str(epochs), '--batch', str(batch), '--lr', str(lr),
                   '--workers', '2', '--progress-json']
            if not pretrained:
                cmd.append('--no-pretrained')
            if not command_input:
                cmd.append('--no-command-input')
            if not mask_impacts:
                cmd.append('--keep-impacts')
            self.state = {'name': name, 'params': params, 'phase': 'starting', 'started': time.time(),
                          'epoch': 0, 'epochs': epochs, 'step': 0, 'steps': 0, 'step_phase': '',
                          'history': [], 'info': {}, 'eta_s': None, 'message': '起動中…', 'error': None,
                          'log': collections.deque(maxlen=200), 'finished': None}
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            self.proc = subprocess.Popen(cmd, cwd=str(config.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, env=env, start_new_session=True)
            threading.Thread(target=self._watch, args=(self.proc, out), name='train-watch', daemon=True).start()
            return name

    def stop(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError('学習は実行されていません')
            self.state['message'] = '停止中…(最良のモデルを書き出しています)'
            os.killpg(self.proc.pid, signal.SIGINT)

    # ------------------------------------------------------------ 進捗の監視
    def _watch(self, proc, out):
        st = self.state
        t_steps = collections.deque(maxlen=50)
        with open(out / 'train.log', 'w', encoding='utf-8') as logf:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                line = line.rstrip('\n')
                if not line.startswith('@@'):
                    if line.strip():
                        st['log'].append(line)
                    continue
                try:
                    ev = json.loads(line[2:])
                except ValueError:
                    continue
                kind = ev.get('type')
                if kind == 'start':
                    st['info'] = ev
                    st['phase'] = 'training'
                    st['message'] = '学習中'
                elif kind == 'step':
                    st['epoch'], st['step'], st['steps'], st['step_phase'] = ev['epoch'], ev['step'], ev['steps'], ev['phase']
                    t_steps.append((time.time(), self._global_step(st)))
                    st['eta_s'] = self._eta(st, t_steps)
                elif kind == 'epoch':
                    st['history'].append(ev)
                elif kind == 'error':
                    st['error'] = ev.get('message')
                elif kind == 'done':
                    st['result'] = ev
        code = proc.wait()
        result = self._read_json(out / 'result.json')
        if code != 0 or result is None:
            st['phase'] = 'error'
            st['error'] = st['error'] or '学習が異常終了しました(終了コード %s)。ログを確認してください' % code
        elif result.get('state') == 'stopped_no_model':
            st['phase'] = 'stopped'
            st['message'] = '停止しました(1エポックも終わらなかったためモデルはありません)'
        else:
            if st['params']['build_engine'] and (out / 'jetformer.onnx').exists():
                st['phase'] = 'engine'
                st['message'] = 'TensorRTエンジンを作成中…(約1分)'
                # 別プロセスで作る(サーバーの中で作ると、TensorRTが確保した約2GBがサーバーに残り続ける)
                try:
                    r = subprocess.run([sys.executable, str(ENGINE_SCRIPT), str(out / 'jetformer.onnx'),
                                        str(out / 'jetformer_fp16.engine')], cwd=str(config.ROOT),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1200)
                    with open(out / 'train.log', 'a', encoding='utf-8') as logf:
                        logf.write(r.stdout)
                    for line in r.stdout.splitlines()[-5:]:
                        st['log'].append(line)
                    if r.returncode != 0:
                        st['error'] = 'TensorRTエンジンの作成に失敗しました(終了コード %d)。ログを確認してください' % r.returncode
                except subprocess.TimeoutExpired:
                    st['error'] = 'TensorRTエンジンの作成が20分で終わりませんでした'
            st['phase'] = 'stopped' if result.get('state') == 'stopped' else 'done'
            st['message'] = '停止しました(途中までの最良モデルを保存)' if st['phase'] == 'stopped' else '完了しました'
        st['eta_s'] = 0
        st['finished'] = time.time()
        with self.lock:
            self.proc = None
        for cb in list(self.listeners):
            try:
                cb('train_done', phase=st['phase'], name=st['name'])
            except Exception:
                pass

    @staticmethod
    def _global_step(st):
        """学習開始からの通算ステップ(学習+検証)"""
        info = st['info']
        steps_epoch = info.get('steps_train', 0) + info.get('steps_val', 0)
        done_in_epoch = st['step'] + (info.get('steps_train', 0) if st['step_phase'] == 'val' else 0)
        return (st['epoch'] - 1) * steps_epoch + done_in_epoch

    @staticmethod
    def _eta(st, t_steps):
        """直近の進み具合(秒/ステップ)から残り時間を見積もる。進捗の通知は10ステップごととは限らないので、
        通算ステップの差で割る"""
        info = st['info']
        if len(t_steps) < 3 or not info:
            return None
        (t0, g0), (t1, g1) = t_steps[0], t_steps[-1]
        if g1 <= g0:
            return None
        per_step = (t1 - t0) / (g1 - g0)
        steps_epoch = info.get('steps_train', 0) + info.get('steps_val', 0)
        return int((st['epochs'] * steps_epoch - g1) * per_step)

    @staticmethod
    def _read_json(path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    # ------------------------------------------------------------ 状態・一覧
    def status(self):
        st = self.state
        if st is None:
            return {'running': False}
        out = {k: v for k, v in st.items() if k != 'log'}
        out['log'] = list(st['log'])[-30:]
        out['running'] = self.running
        out['elapsed_s'] = int((st['finished'] or time.time()) - st['started'])
        return out

    def list_runs(self):
        runs = []
        if not RUNS_DIR.exists():
            return runs
        for d in RUNS_DIR.iterdir():
            meta = self._read_json(d / 'run.json')
            if meta is None:
                continue
            res = self._read_json(d / 'result.json') or {}
            active = self.state is not None and self.state['name'] == d.name and self.running
            runs.append({
                'name': d.name, 'params': {k: meta.get(k, True if k == 'command_input' else None)
                                           for k in ('dataset', 'stride', 'epochs', 'batch', 'lr', 'pretrained', 'command_input',
                                                     'mask_impacts')},
                'created_at': meta.get('created_at'), 'state': 'running' if active else res.get('state', 'error'),
                'epochs_done': res.get('epochs_done'), 'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
                'best_val_per_mode_us': res.get('best_val_per_mode_us', {}), 'modes': res.get('modes', []),
                'train_samples': res.get('train_samples'), 'duration_s': res.get('duration_s'),
                'files': [f for f in ARTIFACTS if (d / f).exists()],
            })
        return sorted(runs, key=lambda r: (r['created_at'] or '', r['name']), reverse=True)     # 新しい順

    def delete_run(self, name):
        d = RUNS_DIR / name
        if '..' in name or '/' in name or not (d / 'run.json').exists():
            raise KeyError(name)
        if self.state is not None and self.state['name'] == name and self.running:
            raise RuntimeError('実行中の学習は削除できません')
        shutil.rmtree(d)

    def artifact_path(self, name, filename):
        if filename not in ARTIFACTS or '..' in name or '/' in name:
            raise KeyError(filename)
        p = RUNS_DIR / name / filename
        if not p.exists():
            raise KeyError(filename)
        return p
