#!/usr/bin/env python3
"""正解の付け替え(データセット 1 つ)— DAgger 用の写し(2026-10-15、sim_diverse_train_20261015。protocol.json dagger.procedure 5)。
10/14 の relabel/relabel_dataset.py の写しで、変えたのは: 1 シーンの道具を dagger/relabel_scene_dagger.mjs にした・--noise に dagger と
dart-recorded(記録の info.dart をそのまま読む)を足した・途中のファイルと写しの置き場所を今日のレポートの dagger/ にした・
アプリの写しを今日のもの(sim_diverse_train_20261015/sim)にした。先生の方策の sha256 は 10/14 の protocol.json から読む(読むだけ)。
以下は 10/14 の元の説明(R1 は --noise none のとき):

    python -I relabel_dataset.py --dataset-dir <data/datasets/<名前>> --teacher fast|margin --noise N_fast|N_margin|none
           --mode check|write [--jobs 3] [--scenes id1,id2] [--reference-scenes 0] [--reference-stride 37] [--states recorded]
           [--report <報告の JSON>] [--work <途中のファイルの置き場所>] [--backup <元の配列の写しの置き場所>]

  1) シーンごとに relabel_scene.mjs(Node・元の sim-core.js・ブラウザと同じ先生のファイル)で、s_{t+1} から先生だけで運転した
     10 フレームの操作(SI 単位)と「先生が 2 秒以内に立て直せるか」を計算する(--jobs 並列)。
  2) SI → PWM は収集器そのもの(コピーした sim/dataset_recorder.py の action_to_pwm、session.json の calibration・model_config)で行い、
     float32 にする(収集器が NPZ に書くのと同じ)。
  3) --mode check: NPZ は変えない。新しい正解と元の future_actions の差(R1: 乱しなしのデータで ≤ 1 µs)・外すフレームを数える。
     --mode write: 元の future_actions・future_valid を --backup に写してから(元の NPZ の sha256 も)、aligned/<シーン>.npz の
     future_actions(t < N − 10)と future_valid(= 元の future_valid かつ 先生が立て直せる)だけを書き換える(ほかの配列は同じ)。
     書いたあと読み直して確かめ、dataset_card.json に relabel の記録を足す。
  4) 確かめ(どちらのモードでも): R2(再生の差 ≤ 1e-9、記録の info.dart.clean と再計算の差)、R3(乱していないフレーム t+1 では
     future_actions[t, 0] = 記録の vehicle[t+1](≤ 1 µs)、乱したフレームでは違う)、R3b(10 フレームの窓 t+1..t+1+k に乱しが無ければ
     future_actions[t, k] = vehicle[t+1+k]、乱しがあれば違ってよい)、R4(外すフレームの割合)、記録の操作の PWM の作り直し(vehicle と差 0)。
  ユーザーのファイルは変えない。JetFormer のデータセットは --mode write のときだけ変える(付け替えの対象のデータセットだけ)。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path('/home/jetson/jetformer/experiments/20261015_sim_diverse_train')
REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
SIM = REPORT / 'sim'
SCENE_TOOL = EXP / 'dagger' / 'relabel_scene_dagger.mjs'
PROTOCOL = Path('/home/jetson/jetformer_reports/sim_dart_20261014/protocol.json')   # 先生の方策の sha256 だけを読む
H = 10
TOL_US = 1.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_recorder():
    spec = importlib.util.spec_from_file_location('dart_dataset_recorder_copy', SIM / 'dataset_recorder.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reusable(out_json):
    """--reuse-work: 前に同じ設定で計算して合格した結果があれば使う(途中で止まった付け替えの続き)"""
    try:
        r = json.loads(Path(out_json).read_text())
    except (OSError, ValueError):
        return False
    return bool(r.get('pass')) and r.get('labels') is not None


def run_scene(scene_dir, out_json, a, reference):
    if a.reuse_work and reusable(out_json):
        return 0.0
    cmd = ['nice', '-n', '10', 'node', str(SCENE_TOOL), '--scene', str(scene_dir), '--teacher', a.teacher, '--noise', a.noise,
           '--states', a.states, '--horizon', str(H), '--validity', str(a.validity), '--out', str(out_json)]
    if reference:
        cmd += ['--reference-stride', str(a.reference_stride)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{scene_dir.name}: relabel_scene.mjs が失敗しました: {r.stderr[-2000:]}')
    return time.time() - t0


def to_pwm(rec, labels_si, calibration, model):
    """labels_si: (T, 2H) の SI(δ, v を交互)→ (T, H, 2) の float32 PWM(収集器の action_to_pwm と同じ式・同じ丸め)"""
    T = len(labels_si)
    out = np.zeros((T, H, 2), np.float64)
    for t in range(T):
        row = labels_si[t]
        for k in range(H):
            out[t, k] = rec.action_to_pwm({'steering': row[2 * k], 'speed': row[2 * k + 1]}, calibration, model)
    return out.astype(np.float32)


def check_scene(rec, z, res, session, noisy):
    """付け替えた正解の確かめ。z: 元の NPZ の配列(dict)。戻り値 (new_fa, new_fv, stats)"""
    N = len(z['timestamps_us'])
    assert res['frames'] == N, f"フレーム数が違います({res['frames']} と {N})"
    cal, model = session['calibration'], session['model_config']
    T = N - H
    new = to_pwm(rec, res['labels'], cal, model)
    fa0, fv0, veh = z['future_actions'], z['future_valid'].astype(bool), z['vehicle']
    assert fa0.shape == (N, H, 2) and fv0.shape == (N,)
    tv = np.asarray(res['valid_teacher'], bool)
    assert len(tv) == T and new.shape == (T, H, 2)
    new_fa = fa0.copy()
    new_fa[:T] = new
    new_fv = fv0.copy()
    new_fv[:T] = fv0[:T] & tv
    # 記録の操作(乱し入り)の PWM の作り直し = vehicle(収集器と同じ式か)
    applied = np.array([rec.action_to_pwm({'steering': c[0], 'speed': c[1]}, cal, model) for c in res['applied_si']], np.float32)
    pwm_recon_max = float(np.abs(applied - veh[:, :2]).max())
    # 元の正解との差(学習に使うフレーム)
    used = fv0[:T]
    diff = np.abs(new.astype(np.float64) - fa0[:T].astype(np.float64)).max(axis=(1, 2))
    # R3: 乱していないフレーム t+1 では new[t, 0] = vehicle[t+1]
    nx = noisy[1:T + 1]
    d0 = np.abs(new[:, 0, :].astype(np.float64) - veh[1:T + 1, :2].astype(np.float64)).max(axis=1)
    # R3b: 窓 t+1..t+1+k に乱しが無いなら new[t, k] = vehicle[t+1+k]
    nz_cum = np.zeros((T, H), bool)
    for t in range(T):
        nz_cum[t] = np.logical_or.accumulate(noisy[t + 1:t + 1 + H])
    dk = np.abs(new.astype(np.float64) - np.stack([veh[t + 1:t + 1 + H, :2] for t in range(T)]).astype(np.float64)).max(axis=2)
    stats = {
        'frames': int(N), 'labelled_frames': int(T), 'noisy_frames': int(noisy.sum()),
        'pwm_recon_vs_vehicle_max_us': pwm_recon_max,
        'diff_vs_original_max_us_used': float(diff[used].max()) if used.any() else 0.0,
        'frames_diff_gt_1us_used': int((diff[used] > TOL_US).sum()),
        'changed_frames': int((diff > 0).sum()),
        'invalid_teacher_frames': int((~tv).sum()), 'invalid_teacher_frames_used': int((~tv & used).sum()),
        'future_valid_before': int(fv0.sum()), 'future_valid_after': int(new_fv.sum()),
        'r3_unnoisy_next_frames': int((~nx).sum()), 'r3_unnoisy_next_max_us': float(d0[~nx].max()) if (~nx).any() else 0.0,
        'r3_unnoisy_next_gt_1us': int((d0[~nx] > TOL_US).sum()),
        'r3_noisy_next_frames': int(nx.sum()), 'r3_noisy_next_mean_us': float(d0[nx].mean()) if nx.any() else None,
        'r3_noisy_next_frac_gt_1us': float((d0[nx] > TOL_US).mean()) if nx.any() else None,
        'r3b_clean_window_cells': int((~nz_cum).sum()), 'r3b_clean_window_max_us': float(dk[~nz_cum].max()) if (~nz_cum).any() else 0.0,
        'r3b_clean_window_gt_1us': int((dk[~nz_cum] > TOL_US).sum()),
        'r3b_noisy_window_cells': int(nz_cum.sum()), 'r3b_noisy_window_mean_us': float(dk[nz_cum].mean()) if nz_cum.any() else None,
        'r3b_noisy_window_frac_gt_1us': float((dk[nz_cum] > TOL_US).mean()) if nz_cum.any() else None,
        'label_minus_applied_steer_us_noisy_next_mean': float((new[:, 0, 0].astype(np.float64) - veh[1:T + 1, 0])[nx].mean()) if nx.any() else None,
    }
    return new_fa, new_fv, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--teacher', choices=['fast', 'margin'], required=True)
    ap.add_argument('--noise', choices=['dagger', 'dart-recorded', 'none'], required=True)
    ap.add_argument('--mode', choices=['check', 'write'], required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--scenes', default='')
    ap.add_argument('--states', choices=['recorded', 'replay'], default='recorded')
    ap.add_argument('--validity', type=int, default=60)
    ap.add_argument('--reference-scenes', type=int, default=0, help='最初の n シーンで基準の作り方(O(N²))とも比べる')
    ap.add_argument('--reference-stride', type=int, default=37)
    ap.add_argument('--reuse-work', action='store_true', help='--work に合格した結果があるシーンは計算し直さない(どの道具の版で作ったかは報告に書く)')
    ap.add_argument('--report', default='')
    ap.add_argument('--work', default='')
    ap.add_argument('--backup', default='')
    a = ap.parse_args()
    root = Path(a.dataset_dir).resolve()
    name = root.name
    card_path = root / 'dataset_card.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    lm = card.get('label_mask')
    if lm and int(lm.get('masked_frames', 0) or 0) > 0:
        sys.exit('このデータセットには label_mask(学習から外した印)が掛かっています。付け替えと混ざるので止めます')
    if a.mode == 'write' and card.get('relabel'):
        sys.exit('このデータセットはすでに付け替え済みです(dataset_card.json の relabel)')
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    scenes = list(splits.get('train', [])) + list(splits.get('val', []))
    if a.scenes:
        want = a.scenes.split(',')
        scenes = [s for s in scenes if s in want]
    work = Path(a.work) if a.work else REPORT / 'dagger' / 'relabel_work' / name
    work.mkdir(parents=True, exist_ok=True)
    backup = Path(a.backup) if a.backup else REPORT / 'dagger' / 'relabel_backup' / name
    protocol = json.loads(PROTOCOL.read_text())
    policy_sha = protocol['teachers']['T_fast' if a.teacher == 'fast' else 'T_margin']['policy_sha256']
    t0 = time.time()
    # 1) Node(並列)
    jobs = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for i, s in enumerate(scenes):
            sd = root / 'simulation' / 'scenes' / s
            if not (sd / 'frames.jsonl').is_file():
                sys.exit(f'{s}: simulation/scenes/{s}/frames.jsonl がありません')
            jobs[s] = ex.submit(run_scene, sd, work / f'{s}.json', a, i < a.reference_scenes)
        walls = {s: f.result() for s, f in jobs.items()}
    # 2) 3) 4)
    rec = load_recorder()
    per_scene, totals = {}, {}
    if a.mode == 'write':
        backup.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        res = json.loads((work / f'{s}.json').read_text())
        if not res.get('pass'):
            sys.exit(f'{s}: 付け替えに失敗: {res.get("error")}')
        sd = root / 'simulation' / 'scenes' / s
        session = json.loads((sd / 'session.json').read_text())
        rows = [json.loads(l) for l in (sd / 'frames.jsonl').read_text().splitlines() if l.strip()]
        noisy = np.array([bool((r.get('action_info') or {}).get('dart', {}) and (r['action_info']['dart'].get('noise') is not None)) for r in rows])
        report = json.loads((sd / 'evaluation-report.json').read_text())
        res['applied_si'] = [[x['steering'], x['speed']] for x in report['actions']]
        npz = root / 'aligned' / f'{s}.npz'
        with np.load(npz, allow_pickle=False) as zf:
            keys = list(zf.files)
            z = {k: zf[k] for k in keys}
        new_fa, new_fv, st = check_scene(rec, z, res, session, noisy)
        st.update({'replay_max_abs_error': res['replay_max_abs_error'], 'r2': res['r2'], 'fork': {k: v for k, v in res['fork'].items() if k != 'invalid_frames'},
                   'invalid_frames_list': res['fork']['invalid_frames'][:50], 'reference': res['reference'], 'node_wall_s': round(walls[s], 1),
                   'display_name': res.get('display_name'), 'noise_params': res.get('noise_params'), 'teacher_id': res.get('teacher_id'),
                   'relabel_scene_sha256': res['tool']['relabel_scene_sha256'], 'states': res.get('states'), 'reused': walls[s] == 0.0})
        if a.mode == 'write':
            orig_sha = sha(npz)
            np.savez_compressed(backup / f'{s}.npz', future_actions=z['future_actions'], future_valid=z['future_valid'],
                                source_sha256=np.array(orig_sha))
            out = dict(z)
            out['future_actions'] = new_fa.astype(z['future_actions'].dtype)
            out['future_valid'] = new_fv.astype(z['future_valid'].dtype)
            tmp = npz.with_name(npz.name + '.relabel-tmp.npz')
            np.savez_compressed(tmp, **{k: out[k] for k in keys})
            with np.load(tmp, allow_pickle=False) as chk:
                assert list(chk.files) == keys
                for k in keys:
                    want = out[k]
                    assert chk[k].dtype == z[k].dtype and chk[k].shape == z[k].shape, k
                    assert np.array_equal(chk[k], want), k
            os.replace(tmp, npz)
            st['npz_sha256_before'] = orig_sha
            st['npz_sha256_after'] = sha(npz)
        per_scene[s] = st
    # まとめ
    def tot(k, f=sum):
        vals = [v[k] for v in per_scene.values() if v.get(k) is not None]
        return f(vals) if vals else None
    frames = tot('frames'); labelled = tot('labelled_frames')
    summary = {
        'scenes': len(per_scene), 'frames': frames, 'labelled_frames': labelled, 'noisy_frames': tot('noisy_frames'),
        'noisy_frac': tot('noisy_frames') / frames if frames else None,
        'replay_max_abs_error': max(max(v['replay_max_abs_error']['pre'], v['replay_max_abs_error']['post']) for v in per_scene.values()),
        'r2_clean_vs_info_max': tot('r2', lambda xs: max(x['clean_vs_info_max'] for x in xs)),
        'r2_info_frames': tot('r2', lambda xs: sum(x['info_frames'] for x in xs)),
        'r2_clean_vs_applied_unnoisy_max': tot('r2', lambda xs: max(x['clean_vs_applied_unnoisy_max'] for x in xs)),
        'r2_noise_flag_mismatch': tot('r2', lambda xs: sum(x['noise_flag_mismatch'] for x in xs)),
        'r2_noise_value_mismatch_max': tot('r2', lambda xs: max(x['noise_value_mismatch_max'] for x in xs)),
        'pwm_recon_vs_vehicle_max_us': tot('pwm_recon_vs_vehicle_max_us', max),
        'diff_vs_original_max_us_used': tot('diff_vs_original_max_us_used', max),
        'frames_diff_gt_1us_used': tot('frames_diff_gt_1us_used'), 'changed_frames': tot('changed_frames'),
        'invalid_teacher_frames': tot('invalid_teacher_frames'), 'invalid_teacher_frames_used': tot('invalid_teacher_frames_used'),
        'invalid_frac_of_labelled': tot('invalid_teacher_frames') / labelled if labelled else None,
        'future_valid_before': tot('future_valid_before'), 'future_valid_after': tot('future_valid_after'),
        'r3_unnoisy_next_frames': tot('r3_unnoisy_next_frames'), 'r3_unnoisy_next_max_us': tot('r3_unnoisy_next_max_us', max),
        'r3_unnoisy_next_gt_1us': tot('r3_unnoisy_next_gt_1us'), 'r3_noisy_next_frames': tot('r3_noisy_next_frames'),
        'r3_noisy_next_mean_us': (sum(v['r3_noisy_next_mean_us'] * v['r3_noisy_next_frames'] for v in per_scene.values() if v['r3_noisy_next_frames'])
                                  / max(1, tot('r3_noisy_next_frames') or 0)) if tot('r3_noisy_next_frames') else None,
        'r3b_clean_window_cells': tot('r3b_clean_window_cells'), 'r3b_clean_window_max_us': tot('r3b_clean_window_max_us', max),
        'r3b_clean_window_gt_1us': tot('r3b_clean_window_gt_1us'), 'r3b_noisy_window_cells': tot('r3b_noisy_window_cells'),
        'fork_first_label_vs_main_bitwise': tot('fork', lambda xs: sum(x['first_label_vs_main_bitwise'] for x in xs)),
        'invalid_reasons': tot('fork', lambda xs: {k: sum(x[k] for x in xs) for k in ('collisions', 'gate_misses', 'low_clearance', 'early_end')}),
        'reference': [{'scene': s, **v['reference']} for s, v in per_scene.items() if v.get('reference')],
        'relabel_scene_versions': sorted({v['relabel_scene_sha256'] for v in per_scene.values()}), 'reused_scenes': sum(1 for v in per_scene.values() if v['reused']),
        'wall_s': round(time.time() - t0, 1),
    }
    checks = {
        'R1_identity': (summary['frames_diff_gt_1us_used'] == 0 and summary['invalid_teacher_frames'] == 0) if a.noise == 'none' else None,
        'R2_replay': summary['replay_max_abs_error'] <= 1e-9 and (summary['r2_clean_vs_info_max'] or 0) <= 1e-9,
        'R3_noise_frames': summary['r3_unnoisy_next_gt_1us'] == 0 and (a.noise == 'none' or (summary['r3_noisy_next_mean_us'] or 0) > 0),
        'R4_invalid_le_2pct': (summary['invalid_frac_of_labelled'] or 0) <= .02,
        'pwm_recon_0us': (summary['pwm_recon_vs_vehicle_max_us'] or 0) <= 1e-3,
        'reference_equal': all(r['max_abs_diff_si'] <= 1e-9 for r in summary['reference']) if summary['reference'] else None,
    }
    tool = {'relabel_dataset.py': sha(__file__), 'relabel_scene.mjs': sha(SCENE_TOOL)}
    first = next(iter(per_scene.values()), {})
    files_sha = json.loads((work / f'{scenes[0]}.json').read_text())['tool']['files_sha256'] if scenes else {}
    out = {'schema': 'jf-sim-diverse-dagger-relabel-report/v1', 'made': datetime.datetime.now().isoformat(timespec='seconds'), 'dataset': name,
           'mode': a.mode, 'teacher': a.teacher, 'policy_sha256': policy_sha, 'noise': a.noise, 'states': a.states, 'horizon': H,
           'validity_horizon_frames': a.validity, 'summary': summary, 'checks': checks, 'tool_sha256': tool, 'teacher_files_sha256': files_sha,
           'backup': str(backup) if a.mode == 'write' else None, 'per_scene': per_scene}
    if a.mode == 'write':
        card = json.loads(card_path.read_text(encoding='utf-8'))
        card['relabel'] = {'schema': 'jf-sim-dart-relabel/v1', 'made': out['made'], 'teacher': a.teacher, 'teacher_id': first.get('teacher_id'),
                           'policy_sha256': policy_sha, 'noise': a.noise, 'states': a.states, 'horizon': H, 'validity_horizon_frames': a.validity,
                           'label_rule': 'future_actions[t,k] = PWM(â_{t+1+k})、â は記録の状態 s_{t+1} から先生だけで運転した操作(t < N-10)',
                           'validity_rule': 'future_valid[t] = 元の future_valid[t] かつ 先生の 2 秒の分かれた走りで衝突なし・ゲートの外しなし・最小すき間 ≥ 0.02 m',
                           'frames': frames, 'changed_frames': summary['changed_frames'], 'invalid_frames': summary['invalid_teacher_frames'],
                           'future_valid_before': summary['future_valid_before'], 'future_valid_after': summary['future_valid_after'],
                           'backup': str(backup), 'tool_sha256': {**tool, **{'teacher/' + k: v for k, v in files_sha.items()}}}
        tmp = card_path.with_name('dataset_card.json.relabel-tmp')
        tmp.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, card_path)
    rp = Path(a.report) if a.report else work / 'relabel_report.json'
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({'dataset': name, 'mode': a.mode, 'summary': {k: v for k, v in summary.items() if k != 'reference'},
                      'reference': summary['reference'], 'checks': checks, 'report': str(rp)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
