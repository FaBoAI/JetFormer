#!/usr/bin/env python3
"""段 Train(2026-10-23、A10): gpu_queue2.sh の手伝い。experiments/20261021_bench_prep/gpu_queue.py の写し
(変えたのは置き場所・表・状態のファイルの名前・run.json の note・arm・experiment だけ)。もとの説明:

段 G2(2026-10-21): gpu_queue.sh の手伝い(torch を読まない・GPU を使わない・車と装置に触れない)

    python -I gpu_queue2.py status                 # 各学習の状態(学習・エンジン・確認・走行の一覧)
    python -I gpu_queue2.py trained <名前>          # 学習が済んでいれば終了コード 0
    python -I gpu_queue2.py need-engine <名前>      # エンジンが無い・ONNX より古ければ 0
    python -I gpu_queue2.py need-check <名前>       # 今のエンジンの engine_check.json が無ければ 0
    python -I gpu_queue2.py begin <名前> <段>       # 試みの始まりを記録(段 = train・engine・check)
    python -I gpu_queue2.py end <名前> <段> <rc>    # 試みの終わりを記録
    python -I gpu_queue2.py fails <名前> <段>       # 終わったのに済まなかった試みの数を出す(電源断で途切れたものは数えない)
    python -I gpu_queue2.py last-rc <名前> <段>     # 最後に終わった試みの終了コード
    python -I gpu_queue2.py runjson <名前> <コマンド> # run.json を書く(無いときだけ。管理画面の学習タブ・走行タブの一覧用)
    python -I gpu_queue2.py enginecheck <名前>      # logs/engine_check_<名前>.json → data/runs/<名前>/engine_check.json
    python -I gpu_queue2.py drivable <名前>         # drive.model_info で drivable・engine_check を確かめる(合格で 0)
    python -I gpu_queue2.py webbusy                 # 管理画面が学習・走行・収集中なら 0(GET /api/status だけ)
    python -I gpu_queue2.py summary                 # 一覧(名前・L1・エンジンの確認・時間)を logs/gpu_queue2_summary.json と表で出す

状態は logs/gpu_queue2_state.json(tmp → 置き換え。電源断でも半端に残らない)。
"""

import datetime
import hashlib
import json
import os
import shlex
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
HERE = REPO / 'experiments' / '20261022_prep'
LOGS = HERE / 'logs'
QUEUE = HERE / 'gpu_queue2.tsv'
STATE = LOGS / 'gpu_queue2_state.json'
SUMMARY = LOGS / 'gpu_queue2_summary.json'
RUNS = REPO / 'data' / 'runs'
ENGINE = 'jetformer_fp16.engine'
GATE_US = 2.5
LOAD_CHECK = REPO / 'experiments' / '20261013_sim_cotrain' / 'train' / 'engine_load_check.py'
NOTE = {
    'slow-sec': '実車ベンチ(10/23)の候補 SLOW-SEC(遅く入る型・区間だけ、A10): 区間の速度上限 v1.1 を学習の記録に当て、'
                'R1_app・R1_turn・GM・DE の行だけスロットルの正解を上限の出力に付け替えて学習'
                '(data/derived/bench_20261021/slow_labels_sections.npz)。机上の門とユーザーの了承の前は走らせない',
    'slow': '実車ベンチ(10/23)の候補 SLOW(遅く入る型): 区間の速度上限 v1.1 を学習の記録に当て、上限がかかる行のスロットルの'
            '正解を上限の出力に付け替えて学習(data/derived/bench_20261021/slow_labels.npz)。机上の門とユーザーの了承の前は走らせない',
    'wgt': '実車ベンチ(10/23)の候補 WGT(苦手区間重み型): R1・GM・DE の区間のフレームの損失の重みを 3 倍'
           '(data/derived/bench_20261021/wgt_weights.npz)。机上の門とユーザーの了承の前は走らせない',
    'aug': '実車ベンチ(10/23)の候補 AUG(ずれに強い型): カメラの明るさ 0.15・コントラスト 0.2・ずらし 4 画素・ToF の雑音 σ 0.02 の'
           '拡張で学習。机上の門とユーザーの了承の前は走らせない',
}
ARM = {'slow-sec': 'SLOW-SEC', 'slow': 'SLOW', 'wgt': 'WGT', 'aug': 'AUG'}
ARM_NAME = {'bench_20261022-slow-sec-s0': 'SLOW-SEC0', 'bench_20261022-slow-sec-s1': 'SLOW-SEC1',
            'bench_20261021-slow-s1': 'SLOW1'}     # A10 の名前
TABLE_ARGS = ('--label-override-npz', '--sample-weights-npz', '--drop-mask', '--mode-remap-npz')
CODE_FILES = ('tools/train.py', 'jetformer/train_data.py', 'jetformer/dataset.py', 'tools/build_engine.py',
              'jetformer/trt_runtime.py', 'jetformer/policies/jetformer_v0.py')


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def queue():
    """gpu_queue2.tsv → [(名前, コマンド)]"""
    out = []
    for line in QUEUE.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, cmd = line.split('\t', 1)
        out.append((name.strip(), cmd.strip()))
    return out


def arg_value(args, key):
    if key in args:
        i = args.index(key)
        return args[i + 1] if i + 1 < len(args) else None
    return None


def train_args(cmd):
    """キューのコマンド → tools/train.py の引数(train.py より後ろ)"""
    a = shlex.split(cmd)
    return a[a.index('tools/train.py') + 1:]


def epochs_of(name):
    for n, cmd in queue():
        if n == name:
            return int(arg_value(train_args(cmd), '--epochs') or 30)
    return 30


def kind_of(name):
    for k in NOTE:
        if '-%s-' % k in name:
            return k
    return None


def is_trained(name):
    run = RUNS / name
    res = load(run / 'result.json')
    return bool(res and res.get('state') == 'done' and res.get('epochs_done') == epochs_of(name)
                and all((run / f).exists() for f in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json')))


def need_engine(name):
    run = RUNS / name
    eng, onnx = run / ENGINE, run / 'jetformer.onnx'
    return (not eng.exists()) or eng.stat().st_mtime_ns < onnx.stat().st_mtime_ns


def check_current(run):
    """engine_check.json が今のエンジン(大きさ・更新時刻)のものなら dict、違えば None(drive.engine_check_of と同じ規則)"""
    c = load(run / 'engine_check.json')
    eng = run / ENGINE
    if not isinstance(c, dict) or not eng.exists():
        return None
    st = eng.stat()
    e = c.get('engine') or {}
    if int(e.get('size', -1)) != st.st_size or int(e.get('mtime_ns', -1)) != st.st_mtime_ns:
        return None
    return c


# ---------------------------------------------------------------- 試みの記録

def state():
    return load(STATE, {'runs': {}})


def cmd_begin(name, step):
    s = state()
    r = s['runs'].setdefault(name, {})
    r.setdefault(step, []).append({'started': now(), 't0': time.time()})
    s['updated'] = now()
    write_json(STATE, s)


def cmd_end(name, step, rc):
    s = state()
    atts = s['runs'].setdefault(name, {}).setdefault(step, [])
    if not atts or 'ended' in atts[-1]:
        atts.append({'started': None, 't0': None})
    a = atts[-1]
    a.update({'ended': now(), 'rc': int(rc), 'wall_s': round(time.time() - a['t0'], 1) if a.get('t0') else None,
              'ok': step_ok(name, step, int(rc))})
    s['updated'] = now()
    write_json(STATE, s)
    return a


def step_ok(name, step, rc):
    run = RUNS / name
    if step == 'train':
        return is_trained(name)
    if step == 'engine':
        return rc == 0 and not need_engine(name)
    if step == 'check':
        c = load(LOGS / ('engine_check_%s.json' % name))
        return rc == 0 and bool(c) and bool(c.get('all_gates'))
    return rc == 0 and run.exists()


def fails(name, step):
    atts = state()['runs'].get(name, {}).get(step, [])
    return sum(1 for a in atts if 'ended' in a and not a.get('ok'))


def last_rc(name, step):
    atts = [a for a in state()['runs'].get(name, {}).get(step, []) if 'ended' in a]
    return atts[-1]['rc'] if atts else ''


def wall(name, step):
    atts = state()['runs'].get(name, {}).get(step, [])
    w = [a.get('wall_s') for a in atts if a.get('wall_s')]
    return round(sum(w), 1) if w else None


# ---------------------------------------------------------------- run.json・engine_check.json

def cmd_runjson(name, cmd):
    run = RUNS / name
    if (run / 'run.json').exists():
        print('run.json はあります: %s' % (run / 'run.json'))
        return 0
    from jetformer import policies      # torch を読まない
    args = train_args(cmd)
    k = kind_of(name)
    atts = state()['runs'].get(name, {}).get('train', [])
    created = next((a['started'] for a in atts if a.get('started')), None) or now()
    meta = {'dataset': Path(arg_value(args, '--data')).name, 'stride': int(arg_value(args, '--stride')),
            'epochs': int(arg_value(args, '--epochs')), 'batch': int(arg_value(args, '--batch')),
            'lr': float(arg_value(args, '--lr')), 'pretrained': '--no-pretrained' not in args, 'build_engine': True,
            'command_input': '--no-command-input' not in args, 'mask_impacts': '--keep-impacts' not in args,
            'policy': arg_value(args, '--policy'), 'policy_cfg': policies.get(arg_value(args, '--policy')).resolve({}),
            'impact_g': float(arg_value(args, '--impact-g')), 'seed': int(arg_value(args, '--seed')),
            'name': name, 'created_at': created, 'note': NOTE.get(k, '実車ベンチ(10/23)の候補'),
            'experiment': 'prep_20261022', 'amendment': 'A10', 'arm': ARM_NAME.get(name, ARM.get(k, name))}
    tables = {}
    for key in TABLE_ARGS:
        v = arg_value(args, key)
        if v:
            meta[key.lstrip('-').replace('-', '_')] = v
            tables[key.lstrip('-').replace('-', '_')] = sha256(REPO / v)
    if tables:
        meta['tables_sha256'] = tables
    aug = {a: float(arg_value(args, '--aug-' + a.replace('_', '-'))) for a in ('brightness', 'contrast', 'shift_px', 'tof_noise')
           if arg_value(args, '--aug-' + a.replace('_', '-'))}
    if aug:
        meta['aug'] = aug
    meta['train_env'] = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'MALLOC_ARENA_MAX': '2'}
    meta['code_sha256'] = {f: sha256(REPO / f) for f in CODE_FILES if (REPO / f).exists()}
    meta['queue_cmd'] = cmd
    write_json(run / 'run.json', meta)
    print('run.json を書きました: %s' % (run / 'run.json'))
    return 0


def cmd_enginecheck(name):
    """確認の json → <学習結果>/engine_check.json(10/13 の cotrain_train.write_engine_check と同じ形)。
    確認の後にエンジンが変わっていたら写さない"""
    run = RUNS / name
    src = LOGS / ('engine_check_%s.json' % name)
    c = load(src)
    eng = run / ENGINE
    if not c or not eng.exists():
        print('確認の json かエンジンがありません: %s' % src)
        return 1
    st = eng.stat()
    checked = datetime.datetime.strptime(c['checked_at'], '%Y-%m-%d %H:%M:%S').timestamp()
    if st.st_mtime > checked:
        print('確認の後にエンジンが変わっています(写しません)')
        return 1
    acc, lat = c.get('accuracy') or {}, c.get('latency_ms') or {}
    out = {'ok': bool(c.get('all_gates')), 'accuracy_ok': bool(acc.get('gate_actions_2p5us')) and bool(acc.get('finite')),
           'speed_ok': bool(lat.get('gate_run_numpy')), 'actions_max_us': acc.get('actions_max_us'),
           'actions_p99_us': acc.get('actions_p99_us'), 'gate_us': GATE_US,
           'run_numpy_median_ms': lat.get('run_numpy_median'), 'run_numpy_p99_ms': lat.get('run_numpy_p99'),
           'checked_at': c['checked_at'], 'note': '',
           'engine': {'file': eng.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns},
           'source': str(src),
           'method': '実フレーム 64 個で TensorRT(run_numpy)と PyTorch FP32(CPU・best.pth)の actions の差。速さは 100 回の'
                     '空回しの後 600 回(experiments/20261013_sim_cotrain/train/engine_load_check.py、10/23 段 Train)'}
    write_json(run / 'engine_check.json', out)
    print(json.dumps({k: out[k] for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'run_numpy_median_ms',
                                          'run_numpy_p99_ms')}, ensure_ascii=False))
    return 0 if out['accuracy_ok'] else 1


def cmd_drivable(name):
    from jetformer import drive     # torch を読まない(cv2・numpy だけ)
    info = drive.model_info(name)
    listed = [m['name'] for m in drive.list_models()]
    rec = drive.recommended()
    out = {'checked_at': now(), 'drivable': bool(info and info.get('drivable')),
           'why_not': info and info.get('why_not'), 'engine_check': info and info.get('engine_check'),
           'listed': name in listed, 'recommended_run': rec and rec.get('run'),
           'policy': info and info.get('policy'), 'trained_modes': info and info.get('trained_modes'),
           'steer_center_us': info and info.get('steer_center_us'),
           'throttle_neutral_us': info and info.get('throttle_neutral_us')}
    s = state()
    s['runs'].setdefault(name, {})['drivable'] = out
    s['updated'] = now()
    write_json(STATE, s)
    print(json.dumps(out, ensure_ascii=False))
    ok = out['drivable'] and out['listed'] and bool(out['engine_check']) and out['engine_check'].get('accuracy_ok') is True
    return 0 if ok else 1


def cmd_webbusy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8080/api/status', timeout=3) as r:
            st = json.load(r)
    except Exception:          # noqa: BLE001  管理画面が止まっていれば確かめない
        return 1
    why = [k for k, v in (('training', st.get('training')), ('driving', st.get('driving')),
                          ('recording', (st.get('recorder') or {}).get('recording')),
                          ('session', (st.get('session') or {}).get('active'))) if v]
    if why:
        print('管理画面が使用中: %s' % '・'.join(why))
        return 0
    return 1


# ---------------------------------------------------------------- 状態・一覧

def row(name):
    run = RUNS / name
    res = load(run / 'result.json') or {}
    hist = res.get('history') or []
    c = check_current(run)
    lc = load(LOGS / ('engine_check_%s.json' % name)) or {}
    s = state()['runs'].get(name, {})
    d = s.get('drivable') or {}
    if is_trained(name):
        tstate = 'done'
    elif (run / 'last.pth').exists():
        tstate = 'resume'
    else:
        tstate = 'fresh'
    return {
        'name': name, 'arm': ARM_NAME.get(name, ARM.get(kind_of(name), name)), 'train': tstate,
        'epochs_done': res.get('epochs_done'), 'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
        'last_val_us': hist[-1]['val_us'] if hist else None,
        'best_val_per_mode_us': res.get('best_val_per_mode_us'),
        'last_val_per_mode_us': hist[-1].get('val_per_mode_us') if hist else None,
        'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
        'bench_options': res.get('bench_options'), 'peak_mem_mb': res.get('peak_mem_mb'),
        'train_duration_s': res.get('duration_s'),
        'engine': (not need_engine(name)) if (run / 'jetformer.onnx').exists() else False,
        'engine_mb': round((run / ENGINE).stat().st_size / 2 ** 20, 1) if (run / ENGINE).exists() else None,
        'engine_check': c and {k: c.get(k) for k in ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'actions_p99_us',
                                                      'run_numpy_median_ms', 'run_numpy_p99_ms', 'checked_at')},
        'engine_only_ms': lc.get('latency_ms') and {k: lc['latency_ms'].get(k) for k in ('engine_median', 'engine_p99')},
        'drivable': d.get('drivable'), 'listed': d.get('listed'),
        'wall_s': {st: wall(name, st) for st in ('train', 'engine', 'check')},
        'attempts': {st: len(s.get(st, [])) for st in ('train', 'engine', 'check')},
        'fails': {st: fails(name, st) for st in ('train', 'engine', 'check')},
    }


def cmd_status():
    for name, _ in queue():
        r = row(name)
        ec = r['engine_check'] or {}
        print('%-26s 学習 %-6s ep %-4s best %-6s last %-6s エンジン %-5s 確認 %-5s 精度 %-6s 走行 %s' % (
            name, r['train'], r['epochs_done'], r['best_val_us'], r['last_val_us'], r['engine'],
            ec.get('ok'), ec.get('actions_max_us'), r['drivable']))
    return 0


def cmd_summary():
    rows = [row(n) for n, _ in queue()]
    write_json(SUMMARY, {'made_at': now(), 'runs': rows,
                         'note': 'best_val_us・last_val_us は tools/train.py の検証(clean_driving_20261011 の val、stride 3・'
                                 '衝突の前後を除く)。desk_bench.json の l1_best_us・l1_last_us(保留 4,670 フレーム・realacc_lib)は Desk の段で測る'})
    print('| 名前 | 学習 | L1 best [µs](エポック) | L1 last [µs] | エンジン | 精度 最大差 [µs] | run_numpy 中央/p99 [ms] | 走行の一覧 | 学習/エンジン/確認 [秒] |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        ec = r['engine_check'] or {}
        w = r['wall_s']
        print('| %s | %s | %s(%s) | %s | %s | %s(%s) | %s / %s | %s | %s / %s / %s |' % (
            r['name'], r['train'], r['best_val_us'], r['best_epoch'], r['last_val_us'],
            '%s MB' % r['engine_mb'] if r['engine'] else 'なし', ec.get('actions_max_us'),
            '合格' if ec.get('accuracy_ok') else ('不合格' if ec else '未'), ec.get('run_numpy_median_ms'),
            ec.get('run_numpy_p99_ms'), 'drivable' if r['drivable'] else r['drivable'], w['train'], w['engine'], w['check']))
    return 0


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    c, a = argv[1], argv[2:]
    if c == 'status':
        return cmd_status()
    if c == 'trained':
        return 0 if is_trained(a[0]) else 1
    if c == 'need-engine':
        return 0 if need_engine(a[0]) else 1
    if c == 'need-check':
        return 0 if check_current(RUNS / a[0]) is None else 1
    if c == 'begin':
        cmd_begin(a[0], a[1])
        return 0
    if c == 'end':
        r = cmd_end(a[0], a[1], a[2])
        print('%s %s: rc=%s ok=%s %s 秒' % (a[0], a[1], r['rc'], r['ok'], r['wall_s']))
        return 0
    if c == 'fails':
        print(fails(a[0], a[1]))
        return 0
    if c == 'last-rc':
        print(last_rc(a[0], a[1]))
        return 0
    if c == 'runjson':
        return cmd_runjson(a[0], a[1])
    if c == 'enginecheck':
        return cmd_enginecheck(a[0])
    if c == 'drivable':
        return cmd_drivable(a[0])
    if c == 'webbusy':
        return cmd_webbusy()
    if c == 'summary':
        return cmd_summary()
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
