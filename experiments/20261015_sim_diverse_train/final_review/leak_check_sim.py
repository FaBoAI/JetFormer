#!/usr/bin/env python3
"""最終の確かめ(10/15 のレポートの段): シミュレーターのデータの分け方と「漏れ」を、組み合わせのフォルダーの splits.json と
各データセットの splits.json・simulation/collection.json(シーンごとの開始)だけから確かめる。

    python3 -I leak_check_sim.py

  1. 各腕(S-DIV・S-MOD・A4・C-DIV)の train と val が重ならない。val のシーンは元のデータセットの val のまま(train に入っていない)
  2. C-DIV に入れたシミュレーターのシーンは 5 モードの train の 60 シーンだけ(5 モードの val 15 シーンは保留の参考に残る)
  3. 学習のシーンの開始(ゲート・横・向き)が、閉ループ評価の E0〜E11 の開始と同じものはないか(同じなら評価の始まりの状態を学習で見ている)。
     参考: 同じゲートからの開始の数(E0〜E4・E6〜E9 = ゲート 0、E5 = 20、E10 = 10、E11 = 28)
  4. DAgger の 12 シーンの開始が D0〜D11(評価の E とは別)であること
出力: /home/jetson/jetformer_reports/sim_diverse_train_20261015/review/final/leak_check_sim.json
"""
import json
import os

DS = '/home/jetson/jetformer/data/datasets'
TD = '/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train/data'
REP = '/home/jetson/jetformer_reports/sim_diverse_train_20261015'
OUT = os.path.join(REP, 'review/final/leak_check_sim.json')
P = json.load(open(os.path.join(REP, 'protocol.json'), encoding='utf-8'))
E = {k: v for k, v in P['eval_Q1']['starts'].items() if k.startswith('E')}
D = {k: v for k, v in P['dagger']['states_starts'].items() if k.startswith('D')}
MODES = P['data']['accepted_modes']['datasets']
D1, D2 = P['data']['dart_D1']['dataset'], P['data']['margin_D2']['dataset']
DAG = 'jetformer-simulator-rl-dagger-sdiv-4min_20261015-061535_b24e555a'


def js(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def starts_of(ds):
    """シーン → (split, gate, 横, 向き)"""
    out = {}
    col = os.path.join(DS, ds, 'simulation/collection.json')
    for s in js(col)['scenes']:
        ps = s.get('plan_start') or {}
        out[s['scene']] = (s.get('split'), ps.get('gate'), ps.get('left_offset_m'), ps.get('left_yaw_deg'))
    return out


def same_start(a, b):
    return a[1] == b[0] and a[2] is not None and abs(a[2] - b[1]) < 1e-3 and abs(a[3] - b[2]) < 0.05


def main():
    res = {'schema': 'sim-diverse-train-final-leak-sim/v1', 'arms': {}}
    src_split = {}
    for ds in MODES + [D1, D2, DAG]:
        sp = js(os.path.join(DS, ds, 'splits.json'))
        for k in ('train', 'val', 'test'):
            for s in sp.get(k, []):
                src_split[s] = (ds, k)
    for arm in ('S-DIV', 'S-MOD', 'A4', 'C-DIV'):
        sp = js(os.path.join(TD, arm, 'splits.json'))
        tr, va = set(sp['train']), set(sp['val'])
        sim_val_from_train = sorted(s for s in va if s in src_split and src_split[s][1] != 'val')
        sim_train_from_val = sorted(s for s in tr if s in src_split and src_split[s][1] == 'val')
        res['arms'][arm] = {'train': len(tr), 'val': len(va), 'overlap': sorted(tr & va),
                            'val_scenes_that_are_train_in_source': sim_val_from_train,
                            'train_scenes_that_are_val_in_source': sim_train_from_val,
                            'sim_train_by_source': {ds: sum(1 for s in tr if src_split.get(s, (None,))[0] == ds) for ds in MODES + [D1, D2, DAG]}}
    # C-DIV: シミュレーターは 5 モードの train だけ
    cdiv_tr = set(js(os.path.join(TD, 'C-DIV/splits.json'))['train'])
    mode_val = {s for s, (ds, k) in src_split.items() if ds in MODES and k == 'val'}
    res['C-DIV_mode_val_in_train'] = sorted(cdiv_tr & mode_val)
    res['C-DIV_mode_val_count'] = len(mode_val)
    # 開始
    st = {}
    for ds in MODES + [D1, D2, DAG]:
        try:
            st[ds] = starts_of(ds)
        except FileNotFoundError:
            st[ds] = None
    exact = []
    gate_counts = {}
    for ds, m in st.items():
        if m is None:
            continue
        for sc, v in m.items():
            for e, sv in E.items():
                if same_start(v, sv):
                    exact.append({'dataset': ds, 'scene': sc, 'split': v[0], 'start': e})
            if ds != DAG and v[0] == 'train':
                gate_counts.setdefault(str(v[1]), 0)
                gate_counts[str(v[1])] += 1
    res['train_scenes_with_exact_eval_start'] = exact
    res['train_scene_start_gate_counts_SDIV_sources'] = {g: gate_counts.get(g, 0) for g in ('0', '10', '20', '28')}
    res['train_scenes_SDIV_sources_total'] = sum(gate_counts.values())
    res['starts_missing'] = [ds for ds, m in st.items() if m is None]
    # DAgger の開始
    dag = st.get(DAG) or {}
    dag_starts = sorted({next((k for k, sv in D.items() if same_start(v, sv)), 'other') for v in dag.values()})
    res['dagger_scene_starts'] = dag_starts
    res['dagger_scenes'] = len(dag)
    res['dagger_any_eval_start'] = any(same_start(v, sv) for v in dag.values() for sv in E.values())
    res['ok'] = all(not a['overlap'] and not a['val_scenes_that_are_train_in_source'] and not a['train_scenes_that_are_val_in_source']
                    for a in res['arms'].values()) and not res['C-DIV_mode_val_in_train'] and not res['dagger_any_eval_start']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(OUT + '.tmp', OUT)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main()


# ---- 追記: 開始の姿勢(frames.jsonl の最初のフレームの pose_pre_action)で照らす(DART・余裕の線・DAgger は collection.json に開始が無いため)
def first_pose(ds, scene):
    p = os.path.join(DS, ds, 'simulation/scenes', scene, 'frames.jsonl')
    with open(p, encoding='utf-8') as f:
        r = json.loads(f.readline())
    q = r['pose_pre_action']
    return q['x'], q['z'], q['yaw']


def eval_poses(folder, prefix, names):
    out = {}
    for n in names:
        r = js(os.path.join(folder, '%s-%s-single.json' % (prefix, n)))
        t = r['trace'][0]
        out[n] = (t['x'], t['z'], t['yaw'])
    return out


def pose_check():
    import math
    ep = eval_poses(os.path.join(REP, 'eval/closed_loop/SDIV-s0'), 'SDIV-s0', list(E))
    dp = eval_poses(os.path.join(REP, 'eval/closed_loop_dagger/SDIV-s0'), 'SDIV-s0', list(D))
    res = {'eval_start_pose': ep, 'dagger_start_pose': dp, 'by_dataset': {}}
    for ds in MODES + [D1, D2, DAG]:
        sp = js(os.path.join(DS, ds, 'splits.json'))
        rows = []
        for k in ('train', 'val'):
            for sc in sp.get(k, []):
                x, z, yaw = first_pose(ds, sc)
                hitE = [e for e, (ex, ez, ey) in ep.items() if math.hypot(x - ex, z - ez) < 0.005 and abs(yaw - ey) < 0.005]
                hitD = [d for d, (ex, ez, ey) in dp.items() if math.hypot(x - ex, z - ez) < 0.005 and abs(yaw - ey) < 0.005]
                rows.append({'scene': sc, 'split': k, 'pose': [round(x, 3), round(z, 3), round(yaw, 4)], 'eval_start': hitE, 'dagger_start': hitD})
        res['by_dataset'][ds] = {'scenes': len(rows),
                                 'train_at_eval_start': sorted({e for r in rows if r['split'] == 'train' for e in r['eval_start']}),
                                 'n_train_at_eval_start': sum(1 for r in rows if r['split'] == 'train' and r['eval_start']),
                                 'n_at_dagger_start': sum(1 for r in rows if r['dagger_start']),
                                 'rows': rows}
    return res


if __name__ == '__main__':
    pc = pose_check()
    OUT2 = os.path.join(REP, 'review/final/leak_check_sim_pose.json')
    with open(OUT2 + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(pc, f, ensure_ascii=False, indent=1)
    os.replace(OUT2 + '.tmp', OUT2)
    for ds, v in pc['by_dataset'].items():
        print(ds, 'scenes', v['scenes'], 'train_at_eval_start', v['n_train_at_eval_start'], v['train_at_eval_start'], 'at_dagger_start', v['n_at_dagger_start'])
