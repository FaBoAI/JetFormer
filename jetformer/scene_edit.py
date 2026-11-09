"""シーンの分割と、範囲の削除(2026-11-07 のユーザーのご依頼: 「いらない部分を範囲指定で削除したい」「分割と削除の機能があると
いい。区間で切れたデータができるリスクを減らせる」。消し方は「本当に消す(戻せない)」、範囲はシーンの詳細のつまみで決める)

- 分割: フレーム i で 2 つのシーンに分ける(0〜i-1 は元の名前のまま、i〜最後は新しい名前)。絵もセンサーの記録も消さない
- 範囲の削除: フレーム a〜b(両端を含む)を消す。戻せない。
  範囲がシーンの途中なら、前と後ろの 2 つのシーンに分けて間を消す(1 つのシーンの中に時刻の穴を作らない。穴があると、
  そこをまたぐ正解の操作や IMU の履歴がずれるため)。始め・終わりにかかる範囲は、残りを元の名前のシーンにする。
  シーンの全部は消さない(画面の「このシーンを削除」を使う)
- センサーの記録は、フレームの時刻で分ける。部分 [a, b] には、t が cam_t[a] 以上(a = 0 なら下限なし)で cam_t[b+1] 未満
  (b が最後なら上限なし)のサンプルを入れる。分割では 1 つも落とさない(前と後ろで重ならない)。削除では
  [cam_t[a], cam_t[b+1]) のサンプルが消える
- 新しいシーンの名前: 元のシーン(最初に分けた・消したときのシーン)の名前 + '.' + その最初のフレームからのミリ秒(6 桁)。
  例 20261106-180940-0002.005230。並べると時刻の順になり、'-' の前(ログの名前)は変わらない(dataset.scene_path と同じ分け方)
- 書き方: data/.edit/<作業>/ に新しいシーンを作り(残す絵は raw の絵への hardlink。ディスクは増えない)、読み直して確かめてから、
  元のフォルダと入れ替える(rename)。最後に元のフォルダ(消した絵と前の npz)を消す。
  ジャンルの区間(mode_segments・mode_segments_before・label_mask の unset_segments)は、各部分の最初のフレームからの秒に直す
- そのログのラベル(data/labels/<ログ>.json・.npz。フレームの番号とシーンの順で作った物)は古くなるので、名前に
  .stale-<時刻> を付けて外す(使うときは tools/label_events.py で作り直す)
- 収集中・走行中は断る(web/server.py)。マイクの音声(AUDIO.npz)のあるシーンと、知らないファイルのあるシーンは断る
- 確かめの係の指摘で足した物(11/7): 入れ替えの前と後で os.sync()(電源断で 0 バイトの npz を残さない)・同時に 1 つだけ(_LOCK)・
  画面の版の照らし(expect_token・expect_frames。違えば StaleError)・分けた部分は全部に新しい token(画面の絵の URL の版)・
  1 秒(MIN_PART_FRAMES)より短い部分は作らない・サンプルの無いセンサーの npz は書かない・入れ替えと戻しが両方失敗したら
  作業のフォルダを消さずに残す・起動のときに data/.edit に残った作業を直す(recover_pending)
"""
import datetime
import os
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np

from . import config

ALLOWED = {'CAM_FRONT', 'scene.json', 'thumb.jpg'}
REFUSE_NPZ = {'AUDIO.npz'}
THUMB_FRAME = 29                 # recorder.py は 30 枚目の絵でサムネイルを作る(buf.cam が 30 枚になったとき)
MIN_PART_FRAMES = 30             # 分けたあと・消したあとに残すシーンの最小の枚数(約 1 秒)
_LOCK = threading.Lock()          # 分割・削除は同時に 1 つだけ


class EditError(ValueError):
    pass


class StaleError(EditError):
    """画面が見ているシーンと今のシーンが違う(ほかの画面で分けた・消したなど)"""


def _now():
    return datetime.datetime.now()


def _read_json(p):
    import json
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _write_json(p, obj):
    from .dataset import _write_json as w
    w(p, obj, indent=1)


def _scene_dir(name, raw_dir):
    log = name.rsplit('-', 1)[0]
    p = Path(raw_dir) / log / name
    if '..' in name or '/' in name or not (p / 'scene.json').exists():
        raise KeyError(name)
    return p


def _compressed(path):
    with zipfile.ZipFile(path) as z:
        return any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist())


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _check_files(p):
    bad = []
    for e in sorted(os.listdir(p)):
        if e in ALLOWED:
            continue
        if e.endswith('.npz') and e not in REFUSE_NPZ:
            continue
        bad.append(e)
    if bad:
        raise EditError('このシーンには、分割・削除で扱えないファイルがあります: %s' % ', '.join(bad))


def make_thumb(img_path, depth, out_path):
    """サムネイル = カメラの絵 + ToF の色の絵(recorder.py と同じ並べ方)。絵が読めなければ絵をそのまま写す"""
    import cv2
    from .recorder import depth_colormap
    img = cv2.imread(str(img_path))
    tmp = str(out_path) + '.tmp.jpg'
    if img is None:
        shutil.copyfile(img_path, tmp)
    else:
        if depth is not None:
            img = np.hstack([img, depth_colormap(depth, img.shape[1], img.shape[0])])
        if not cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise EditError('サムネイルを書けません: %s' % out_path)
    os.replace(tmp, out_path)


def _nearest_depth(tof, t_us):
    """ToF の記録(dict)から、時刻 t_us にいちばん近い距離の絵(無ければ None)"""
    if not tof or 'depth' not in tof or not len(tof['timestamps_us']):
        return None
    j = int(np.argmin(np.abs(tof['timestamps_us'].astype(np.int64) - int(t_us))))
    return tof['depth'][j]


def refresh_thumb(name, raw_dir=None, bump_token=True):
    """シーンのサムネイルを作り直す(カメラの 30 枚目 + その時刻に近い ToF)。bump_token なら scene.json の token を新しくして、
    画面のサムネイルの URL の版を変える(ブラウザが前の絵を出さない)。11/7 の最初の版の分割・削除で ToF の無いサムネイルになった物の直し"""
    p = _scene_dir(name, Path(raw_dir or config.RAW_DIR))
    cam = _load_npz(p / 'CAM_FRONT.npz')
    tof = _load_npz(p / 'TOF_FRONT.npz') if (p / 'TOF_FRONT.npz').exists() else None
    i = min(THUMB_FRAME, len(cam['timestamps_us']) - 1)
    make_thumb(p / str(cam['files'][i]), _nearest_depth(tof, cam['timestamps_us'][i]), p / 'thumb.jpg')
    if bump_token:
        sj = _read_json(p / 'scene.json')
        sj['token'] = uuid.uuid4().hex
        _write_json(p / 'scene.json', sj)
    return p / 'thumb.jpg'


def _shift_segments(segs, dt_s, dur_s):
    """区間 [{start_s, end_s, ...}] を dt_s 秒前へずらし、0〜dur_s に切る(はみ出した区間は落とす)"""
    out = []
    for sg in segs or []:
        if not isinstance(sg, dict) or not isinstance(sg.get('start_s'), (int, float)) or not isinstance(sg.get('end_s'), (int, float)):
            continue
        a, b = sg['start_s'] - dt_s, sg['end_s'] - dt_s
        if b < -1e-3 or a > dur_s + 1e-3:
            continue
        out.append(dict(sg, start_s=round(max(a, 0.0), 3), end_s=round(min(b, dur_s), 3)))
    return out


def _mask(t, lo, hi):
    m = np.ones(len(t), bool)
    if lo is not None:
        m &= t >= lo
    if hi is not None:
        m &= t < hi
    return m


def _part_arrays(arr, m):
    """npz の中身を、サンプルの印 m で切る(長さがサンプル数と同じ配列だけ。0 次元などはそのまま)"""
    n = len(arr['timestamps_us'])
    return {k: (v[m] if (v.ndim >= 1 and len(v) == n) else v) for k, v in arr.items()}


def plan_parts(n, op, a=None, b=None, index=None, min_part=MIN_PART_FRAMES):
    """フレーム数 n と操作 → 残すフレームの範囲 [(a, b), ...](両端を含む・時刻の順)と、消すフレームの範囲。
    残す部分はどれも min_part 枚以上(短すぎるシーンは開けない・学習にも使えないので作らない)"""
    if n < 2:
        raise EditError('フレームが少なすぎます(%d 枚)' % n)
    if op == 'split':
        if index is None or not (1 <= int(index) <= n - 1):
            raise EditError('分ける位置は 2〜%d 枚目にしてください(今 %s)' % (n, None if index is None else int(index) + 1))
        i = int(index)
        if i < min_part or n - i < min_part:
            raise EditError('分けたあとのシーンが %d 枚(約 1 秒)より短くなります(前 %d 枚・後ろ %d 枚)' % (min_part, i, n - i))
        return [(0, i - 1), (i, n - 1)], None
    if op == 'delete':
        if a is None or b is None:
            raise EditError('消す範囲の始めと終わりを決めてください')
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        if a < 0 or b > n - 1:
            raise EditError('消す範囲がシーンの外です(1〜%d 枚目)' % n)
        if a == 0 and b == n - 1:
            raise EditError('シーンの全部は消せません(「このシーンを削除」を使ってください)')
        keep = []
        if a > 0:
            keep.append((0, a - 1))
        if b < n - 1:
            keep.append((b + 1, n - 1))
        short = [pb - pa + 1 for pa, pb in keep if pb - pa + 1 < min_part]
        if short:
            raise EditError('消したあとに残るシーンが %d 枚(約 1 秒)より短くなります(%s 枚)。短い所も消すか、シーンごと削除してください'
                            % (min_part, '・'.join(map(str, short))))
        return keep, (a, b)
    raise EditError('操作は split か delete です: %r' % op)


def _part_meta(sj, name, a, b, cam_t, counts, origin, edit, keep_start):
    dur = round((int(cam_t[b]) - int(cam_t[a])) / 1e6, 2)
    dt = (int(cam_t[a]) - int(cam_t[0])) / 1e6
    m = dict(sj)
    m['name'] = name
    m['token'] = uuid.uuid4().hex          # 分けた・削った部分は別の物(画面の絵の URL の版・画面の版の照らしにも使う)
    m['start_us'] = int(sj['start_us']) if (keep_start and 'start_us' in sj) else int(cam_t[a])
    m['end_us'] = int(cam_t[b])
    m['duration_s'] = dur
    m['counts'] = counts
    m['rates'] = {k: (round(v / dur, 1) if dur > 0 else 0.0) for k, v in counts.items()}
    segs = _shift_segments(sj.get('mode_segments'), dt, dur)
    main = sj.get('mode_id', 0)
    if not segs:
        segs = [{'start_s': 0.0, 'end_s': dur, 'mode_id': main}]
    m['mode_segments'] = [{k: v for k, v in sg.items() if k != 'mode_name'} for sg in segs]
    if main not in {sg['mode_id'] for sg in m['mode_segments']}:
        main = m['mode_segments'][0]['mode_id']
    m['mode_id'] = main
    m.pop('mode_name', None)
    m.pop('mode_mixed', None)
    if 'mode_segments_before' in sj:
        m['mode_segments_before'] = _shift_segments(sj['mode_segments_before'], dt, dur)
    lm = sj.get('label_mask')
    if isinstance(lm, dict) and lm.get('unset_segments'):
        m['label_mask'] = dict(lm, unset_segments=_shift_segments(lm['unset_segments'], dt, dur))
    m['complete'] = False                 # 分けた・削った物は「完全なシーン(20 秒・完走)」ではない
    m['edit_origin'] = origin
    m['edit'] = dict(edit, frames_in_source=[int(a), int(b)],
                     note='stats は分ける・削る前のシーン全体の値。counts・rates・時刻・ジャンルの区間はこのシーンの値')
    m['edit_history'] = list(sj.get('edit_history') or []) + [m['edit']]
    return m


def edit_scene(name, op, a=None, b=None, index=None, raw_dir=None, data_dir=None, now=None, expect_token=None, expect_frames=None):
    """分割(op='split', index)か範囲の削除(op='delete', a, b)。返り値 {'scenes': [...], 'deleted_frames': n, ...}。
    expect_token・expect_frames は画面が見ているシーンの token と枚数(違えば StaleError。古い画面・2 度押しで別の所を消さない)"""
    if not _LOCK.acquire(timeout=0.2):
        raise EditError('ほかの分割・削除をしています。終わってからもう一度')
    try:
        return _edit_scene(name, op, a, b, index, raw_dir, data_dir, now, expect_token, expect_frames)
    finally:
        _LOCK.release()


def _edit_scene(name, op, a, b, index, raw_dir, data_dir, now, expect_token, expect_frames):
    raw_dir = Path(raw_dir or config.RAW_DIR)
    data_dir = Path(data_dir or config.DATA_DIR)
    now = now or _now()
    p = _scene_dir(name, raw_dir)
    _check_files(p)
    sj = _read_json(p / 'scene.json')
    npz = sorted(f for f in os.listdir(p) if f.endswith('.npz'))
    if 'CAM_FRONT.npz' not in npz:
        raise EditError('カメラの記録(CAM_FRONT.npz)がありません')
    arrays = {f[:-4]: _load_npz(p / f) for f in npz}
    for k, v in arrays.items():
        if 'timestamps_us' not in v:
            raise EditError('%s に時刻がありません' % k)
    cam = arrays['CAM_FRONT']
    cam_t = cam['timestamps_us'].astype(np.int64)
    n = len(cam_t)
    if 'files' not in cam or len(cam['files']) != n:
        raise EditError('カメラの絵の一覧が時刻と合いません')
    rel = [Path(str(f)) for f in cam['files']]      # シーンのフォルダからの場所(本物は 'CAM_FRONT/<時刻>.jpg'。dataset.scene_frame と同じ)
    if any(r.is_absolute() or '..' in r.parts or len(r.parts) != 2 or r.parts[0] != 'CAM_FRONT' for r in rel):
        raise EditError('カメラの絵の場所が CAM_FRONT/ の下にありません')
    missing = [str(r) for r in rel if not (p / r).exists()]
    if missing:
        raise EditError('カメラの絵が %d 枚ありません(例 %s)' % (len(missing), missing[0]))
    if expect_token is not None and str(expect_token) != str(sj.get('token')):
        raise StaleError('画面が古いままです(ほかで分けた・消したなど)。シーンを開き直してください')
    if expect_frames is not None and int(expect_frames) != n:
        raise StaleError('画面の枚数(%d 枚)と今のシーン(%d 枚)が違います。シーンを開き直してください' % (int(expect_frames), n))
    if np.any(np.diff(cam_t) <= 0):
        raise EditError('カメラの時刻が順になっていません')
    keep, deleted = plan_parts(n, op, a, b, index)
    origin = sj.get('edit_origin') or {'name': sj['name'], 'cam0_us': int(cam_t[0])}
    stamp = now.strftime('%Y%m%d-%H%M%S')
    names = []
    for k, (pa, pb) in enumerate(keep):
        if k == 0:
            names.append(sj['name'])               # 時刻の早い部分が元の名前(評価の記録などが名前で指しているため)
        else:
            off_ms = int(round((int(cam_t[pa]) - int(origin['cam0_us'])) / 1000.0))
            if off_ms < 0 or off_ms > 999999:
                raise EditError('新しいシーンの名前を決められません(元の始めから %d ms)' % off_ms)
            nm = '%s.%06d' % (origin['name'], off_ms)
            if (p.parent / nm).exists() or nm in names:
                raise EditError('同じ名前のシーンがあります: %s' % nm)
            names.append(nm)
    edit = {'op': op, 'from': sj['name'], 'at': now.isoformat(timespec='seconds'), 'n_frames_source': int(n),
            'deleted_frames': list(deleted) if deleted else None}
    stage = data_dir / '.edit' / ('%s-%s-%s' % (stamp, op, sj['name']))
    if stage.exists():
        raise EditError('作業のフォルダがすでにあります: %s' % stage)
    stage.mkdir(parents=True)
    try:
        parts = []
        for k, ((pa, pb), nm) in enumerate(zip(keep, names)):
            d = stage / nm
            (d / 'CAM_FRONT').mkdir(parents=True)
            lo = int(cam_t[pa]) if pa > 0 else None
            hi = int(cam_t[pb + 1]) if pb + 1 < n else None
            counts = {}
            part_tof = None
            for sensor, arr in arrays.items():
                if sensor == 'CAM_FRONT':
                    m = np.zeros(n, bool)
                    m[pa:pb + 1] = True
                else:
                    m = _mask(arr['timestamps_us'].astype(np.int64), lo, hi)
                if not m.any():
                    continue                         # サンプルの無いセンサーの npz は書かない(recorder.py と同じ)
                out = _part_arrays(arr, m)
                if sensor == 'TOF_FRONT':
                    part_tof = out
                save = np.savez_compressed if _compressed(p / ('%s.npz' % sensor)) else np.savez
                save(d / ('%s.npz' % sensor), **out)
                counts[sensor] = int(len(out['timestamps_us']))
            for r in rel[pa:pb + 1]:
                os.link(p / r, d / r)                                  # 残す絵は hardlink(ディスクは増えない)
            if k == 0 and pa == 0 and pb >= THUMB_FRAME and (p / 'thumb.jpg').exists():
                os.link(p / 'thumb.jpg', d / 'thumb.jpg')            # 元のサムネイルの絵(30 枚目)が残る
            else:                                                     # 部分の 30 枚目のカメラの絵 + その時刻に近い ToF(recorder.py と同じ並べ方)
                ti = min(pa + THUMB_FRAME, pb)
                make_thumb(p / rel[ti], _nearest_depth(part_tof, cam_t[ti]), d / 'thumb.jpg')
            meta = _part_meta(sj, nm, pa, pb, cam_t, counts, origin, edit, keep_start=(k == 0 and pa == 0))
            _write_json(d / 'scene.json', meta)
            parts.append((nm, d, pa, pb, counts))
        # 読み直して確かめる(絵の数・時刻の範囲・センサーの数)
        for nm, d, pa, pb, counts in parts:
            c = _load_npz(d / 'CAM_FRONT.npz')
            if len(c['timestamps_us']) != pb - pa + 1 or not np.array_equal(c['timestamps_us'], cam_t[pa:pb + 1]):
                raise EditError('確かめに落ちました(カメラの時刻): %s' % nm)
            if sorted(os.listdir(d / 'CAM_FRONT')) != sorted(Path(str(f)).name for f in c['files']) or \
                    not all((d / str(f)).exists() for f in c['files']):
                raise EditError('確かめに落ちました(絵の数): %s' % nm)
            for sensor in counts:
                if len(_load_npz(d / ('%s.npz' % sensor))['timestamps_us']) != counts[sensor]:
                    raise EditError('確かめに落ちました(%s): %s' % (sensor, nm))
        if op == 'split':                          # 分割では、どのセンサーのサンプルも 1 つも落とさない
            for sensor, arr in arrays.items():
                if sum(c.get(sensor, 0) for _, _, _, _, c in parts) != len(arr['timestamps_us']):
                    raise EditError('確かめに落ちました(分けたあとのサンプルの数 %s)' % sensor)
        os.sync()                                  # 新しい npz・絵の hardlink・サムネイルをディスクに書き切ってから入れ替える
        # 入れ替える(元のフォルダを作業のフォルダへ移し、新しいシーンを raw へ)
        old = stage / '__before__'
        os.rename(p, old)
        moved = []
        try:
            for nm, d, *_ in parts:
                dst = p.parent / nm
                if dst.exists():
                    raise EditError('同じ名前のシーンがあります: %s' % nm)
                os.rename(d, dst)
                moved.append((dst, d))
        except Exception:
            for dst, d in reversed(moved):         # 途中で失敗したら、元に戻す
                os.rename(dst, d)
            os.rename(old, p)
            raise
    except Exception as e:
        if (stage / '__before__').exists():
            if not p.exists():
                os.rename(stage / '__before__', p)
            else:                                  # 戻しも失敗した: 元は作業のフォルダにしか無いので、消さずに残す(起動のときに直す)
                raise EditError('入れ替えに失敗しました。元のシーンは %s に残っています(%s)' % (stage / '__before__', e)) from e
        shutil.rmtree(stage, ignore_errors=True)
        raise
    stale = []
    lab = data_dir / 'labels'
    log = sj['name'].rsplit('-', 1)[0]
    for ext in ('.json', '.npz'):
        f = lab / (log + ext)
        if f.exists():
            g = lab / ('%s%s.stale-%s' % (log, ext, stamp))
            try:
                os.rename(f, g)
                stale.append(str(g))
            except OSError as e:                   # シーンはもう入れ替えたので、ここで止めない(知らせるだけ)
                stale.append('外せなかった %s: %s' % (f, e))
    deleted_files = 0
    if deleted:
        deleted_files = deleted[1] - deleted[0] + 1
    os.sync()                                      # 入れ替え(rename)をディスクに書き切ってから、元のフォルダを消す
    shutil.rmtree(stage)                           # 元のフォルダ(消した絵・前の npz)と作業のフォルダを消す
    return {'ok': True, 'op': op, 'scenes': [nm for nm, *_ in parts], 'deleted_frames': deleted_files,
            'deleted_s': round((int(cam_t[deleted[1]]) - int(cam_t[deleted[0]])) / 1e6 + 1 / 30.0, 2) if deleted else 0.0,
            'frames': [pb - pa + 1 for _, _, pa, pb, _ in parts],
            'durations_s': [round((int(cam_t[pb]) - int(cam_t[pa])) / 1e6, 2) for _, _, pa, pb, _ in parts],
            'labels_stale': stale}


def split_scene(name, index, **kw):
    return edit_scene(name, 'split', index=index, **kw)


def delete_range(name, a, b, **kw):
    return edit_scene(name, 'delete', a=a, b=b, **kw)


def recover_pending(raw_dir=None, data_dir=None):
    """data/.edit に残った作業を直す(入れ替えの途中で電源が切れたときなど。管理画面の起動のときに呼ぶ)。返り値は直した物の一覧。
    - __before__(元のシーン)があり、元の名前が raw に無い → 元を戻す(入れ替えの前か、戻しの途中で止まった)
    - __before__ があり、元の名前が raw にある → 入れ替えは始まっていた。残りの部分を raw へ移して、元を消す(新しい部分は
      入れ替えの前に os.sync() で書き切ってある)
    - __before__ が無い → 作りかけ(元は raw にそのまま)。作業のフォルダを消す"""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    root = Path(data_dir or config.DATA_DIR) / '.edit'
    done = []
    if not root.exists():
        return done
    with _LOCK:
        for stage in sorted(x for x in root.iterdir() if x.is_dir()):
            before = stage / '__before__'
            if before.exists() and (before / 'scene.json').exists():
                name = _read_json(before / 'scene.json')['name']
                target = raw_dir / name.rsplit('-', 1)[0] / name
                if not target.exists():
                    os.rename(before, target)
                    done.append('元に戻した: %s' % name)
                else:
                    for d in sorted(x for x in stage.iterdir() if x.is_dir() and x.name != '__before__'):
                        dst = raw_dir / d.name.rsplit('-', 1)[0] / d.name
                        if (d / 'scene.json').exists() and not dst.exists():
                            os.rename(d, dst)
                            done.append('入れ替えを終えた: %s' % d.name)
                    done.append('元を消した: %s' % name)
            else:
                done.append('作りかけを消した: %s' % stage.name)
            os.sync()
            shutil.rmtree(stage, ignore_errors=True)
    return done
