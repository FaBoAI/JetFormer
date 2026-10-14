#!/usr/bin/env python3
"""記録の段の数字をまとめる(record.md の表の元。2026-10-14、sim_dart_20261014)。読むだけ。

    python -I summarize_record.py D1 D2 > <まとめ.json>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_dart_20261014')


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = {}
    for d in sys.argv[1:]:
        o = R / 'recording' / d
        res = load(o / 'result.json') or {}
        side = sorted((o / 'sidecars').glob('*.json'))
        acc = [json.loads(f.read_text()) for f in side if 'attempt' not in f.name]
        att = [json.loads(f.read_text()) for f in side if 'attempt' in f.name]
        pred = load(R / 'record_prep' / f'prediction_{d}.json') or {}
        pscenes = {s['name']: s for s in pred.get('scenes', [])}
        lap_diff = []
        for s in acc:
            p = pscenes.get(s['name'])
            if not p:
                continue
            pl = (p.get('chosen') or {}).get('laps') or p['clean']['laps']
            if pl and s['browser']['laps']:
                lap_diff.append(s['browser']['laps'][0] - pl[0]['elapsed_s'])
        prog = res.get('progress') or []
        ver = (load(o / 'verify' / 'collection_verify.json') or {}).get('summary')
        imp = load(o / 'import.json') or {}
        rc = load(o / 'relabel_check.json') or {}
        rw = load(o / 'relabel.json') or {}
        zc = load(o / 'collection_zip_check.json') or {}
        dev = (load(o / 'deviation.json') or {}).get('summary')
        col = load(o / 'collection.json') or {}
        out[d] = {
            'browser': {'accepted': len(res.get('accepted') or []), 'rejected': res.get('rejected'), 'retried': res.get('retried'),
                        'attempt_files': len(att), 'rejections': [(a['name'], a['attempt'], a['why']) for a in att if not a['accepted']],
                        'wall_s': res.get('wall_s'), 'error': res.get('error'), 'collection_status': res.get('collection_status'),
                        'lap1_mean_s': statistics.mean(s['browser']['laps'][0] for s in acc) if acc else None,
                        'lap2_mean_s': statistics.mean(s['browser']['laps'][1] for s in acc if len(s['browser']['laps']) > 1) if acc else None,
                        'lap1_minus_node_mean_s': statistics.mean(lap_diff) if lap_diff else None,
                        'lap1_minus_node_absmax_s': max(abs(x) for x in lap_diff) if lap_diff else None,
                        'noisy_frames': sum((s.get('dart') or {}).get('noisy_frames') or 0 for s in acc),
                        'events': sum(len((s.get('dart') or {}).get('events') or []) for s in acc),
                        'aborted_events': sum(sum(1 for e in (s.get('dart') or {}).get('events') or [] if e.get('aborted') is not None) for s in acc),
                        'min_dmin_m': min(((s.get('dart') or {}).get('min_dmin') or 9) for s in acc) if acc else None,
                        'k_used': {s['name']: (s.get('dart') or {}).get('noise', {}).get('k') if (s.get('dart') or {}).get('noise') else None for s in acc},
                        'fps_median': (statistics.median(
                            [(b['frame'] - a['frame']) / (b['wall_s'] - a['wall_s']) for a, b in zip(prog, prog[1:])
                             if b.get('frame') and a.get('frame') is not None and b['frame'] > a['frame'] and b['wall_s'] > a['wall_s']]) if len(prog) > 2 else None)},
            'collection': {'zip': col.get('zip'), 'reserve_gib': col.get('reserve_gib'), 'response': {k: (col.get('response') or {}).get(k) for k in ('filename', 'scenes', 'frames', 'bytes')}},
            'zip_check': {k: zc.get(k) for k in ('pass', 'frames', 'training_frames', 'pwm_recompute_max_abs_us', 'collection_contract_teacher', 'card_keys_same_as_original')},
            'verify': ver,
            'import': {k: imp.get(k) for k in ('name', 'path', 'tags', 'to_scenes', 'num_modes', 'mode_frames', 'scenes', 'frames', 'training_frames',
                                               'train_scenes', 'val_scenes', 'teacher_id', 'label_masked_frames', 'card_label_mask', 'label_mask_clear',
                                               'validate_errors', 'validate_warnings', 'real_modes_json_unchanged', 'archive_sha256')},
            'relabel_check': {'summary': {k: v for k, v in (rc.get('summary') or {}).items() if k != 'reference'}, 'reference': (rc.get('summary') or {}).get('reference'),
                              'checks': rc.get('checks')} if rc else None,
            'relabel_write': {'summary': {k: v for k, v in (rw.get('summary') or {}).items() if k != 'reference'}, 'checks': rw.get('checks'), 'backup': rw.get('backup')} if rw else None,
            'deviation': dev,
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
