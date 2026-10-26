"""verify: 生の記録(data/raw)だけを読む小さな読み込み。labels・scorecard は使わない。"""
import glob, json, os
import numpy as np

RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665


def _ok(d):
    """AI_CMD・VEHICLE・IMU の 3 つが読めるチャンクだけ使う(No.10 は 5 番目から VEHICLE が無い・壊れている)。"""
    for n in ('AI_CMD', 'VEHICLE', 'IMU'):
        f = os.path.join(d, n + '.npz')
        try:
            with np.load(f) as z:
                if 'timestamps_us' not in z.files or len(z['timestamps_us']) == 0:
                    return False
        except Exception:
            return False
    return True


def chunks(log):
    return sorted(d for d in glob.glob(os.path.join(RAW, log, log + '-*')) if os.path.isdir(d) and _ok(d))


def _cat(log, name, keys):
    out = {k: [] for k in keys}
    for d in chunks(log):
        f = os.path.join(d, name + '.npz')
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            for k in keys:
                if k in z.files:
                    out[k].append(z[k])
    return {k: (np.concatenate(v) if v else None) for k, v in out.items()}


def load(log):
    meta = json.load(open(os.path.join(RAW, log, 'log.json')))
    a = _cat(log, 'AI_CMD', ['timestamps_us', 'out', 'model_out', 'raw', 'gains', 'dry_run', 'select', 'assist'])
    v = _cat(log, 'VEHICLE', ['timestamps_us', 'erpm', 'select', 'board_t_us', 'pwm'])
    i = _cat(log, 'IMU', ['timestamps_us', 'accel', 'gyro'])
    c = _cat(log, 'CAM_FRONT', ['timestamps_us', 'files'])
    t0 = int(a['timestamps_us'][0])
    r = dict(log=log, meta=meta, t0=t0)
    r['a_t'] = (a['timestamps_us'] - t0) / 1e6
    r['a_out'] = a['out'].astype(float)
    r['a_model'] = a['model_out'].astype(float)
    r['a_gain'] = a['gains'].astype(float)
    r['a_dry'] = a['dry_run'].astype(bool)
    r['a_sel'] = a['select'].astype(bool)
    r['a_assist'] = a['assist']
    r['v_t'] = (v['timestamps_us'] - t0) / 1e6
    r['v_erpm'] = v['erpm'].astype(float)
    r['v_sel'] = v['select'].astype(bool)
    r['v_board'] = v['board_t_us']
    r['i_t'] = (i['timestamps_us'] - t0) / 1e6
    r['i_acc'] = i['accel'].astype(float)
    r['i_gz'] = i['gyro'][:, 2].astype(float)
    r['c_t'] = (c['timestamps_us'] - t0) / 1e6 if c['timestamps_us'] is not None else None
    # 画像のパス(チャンクのフォルダを前に付ける)
    files = []
    for d in chunks(log):
        f = os.path.join(d, 'CAM_FRONT.npz')
        if os.path.exists(f):
            with np.load(f) as z:
                files += [os.path.join(d, p) for p in z['files']]
    r['c_files'] = files
    return r


def desc_field(meta, key):
    for tok in meta.get('description', '').split():
        if tok.startswith(key + '='):
            return tok.split('=', 1)[1]
    return None
