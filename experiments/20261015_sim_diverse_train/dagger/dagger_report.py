#!/usr/bin/env python3
"""DAgger(Q1b)の記録をまとめる(sim_diverse_train_20261015、2026-10-15)

    python3 -I dagger_report.py [--cut "<削った理由>"]

dagger/ の結果(teacher_select.json・prepare_record.json・g1.json・predict.json・g2.json・rec/result.json・rec/verify.json・
rec/relabel_check.json・rec/relabel_write.json・train/a4_plan.json)と eval_sim.json の A4 の行を読み、
eval/dagger_status.json(eval_sim.md の 8 章の文と数)を書く。図 fig/dagger_states.png(状態集めの走行と切り替えのフレーム k)も作る。
"""
import argparse
import json
from pathlib import Path

R = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
D = R / 'dagger'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def figure(sel, prep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            break
    rows = sel['rows']
    first = json.loads(Path(rows[0]['file']).read_text())
    world = first['world']
    fig, ax = plt.subplots(figsize=(11, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in world['surfaces']:
        if s['id'] == 'carpet':
            continue
        col = {'shortcut': '#3b3b3b', 'turf': '#7fbf7f', 'ptfe': '#e8e2c8', 'blue-mat': '#b9d3f0'}.get(s['id'], '#cccccc')
        ax.add_patch(plt.Rectangle((s['x'] - s['w'] / 2, s['z'] - s['d'] / 2), s['w'], s['d'], color=col, alpha=.3, lw=0))
    for seg in world['segments']:
        ax.plot([seg['a'][0], seg['b'][0]], [seg['a'][1], seg['b'][1]], color='#9b4a3c', lw=1.0, alpha=.7)
    for dsk in world['disks']:
        ax.add_patch(plt.Circle((dsk['x'], dsk['z']), max(dsk.get('radius', .02), .02), color='#6f6e69', alpha=.6, lw=0))
    for r in rows:
        rep = json.loads(Path(r['file']).read_text())
        tr = rep['trace']
        d = r['name'].split('-')[2]
        k = prep['k'][d]['k']
        ok = rep['outcome']['reason'] == 'route_complete'
        xs, zs = [t['x'] for t in tr], [t['z'] for t in tr]
        ax.plot(xs[:k], zs[:k], color='#4a3aa7', lw=1.1, alpha=.85)
        ax.plot(xs[k - 1:], zs[k - 1:], color='#4a3aa7', lw=.9, alpha=.35, ls=':')
        ax.plot(xs[max(0, k - 1)], zs[max(0, k - 1)], marker='o', color='#eda100', ms=7, mec=INK, mew=.6)
        ax.text(xs[max(0, k - 1)], zs[max(0, k - 1)], f' {d}', fontsize=7.5, color=INK)
        if not ok:
            ax.plot(xs[-1], zs[-1], marker='x', color=INK, ms=8, mew=2)
    b = world['bounds']
    ax.set_xlim(b['minX'], b['maxX'])
    ax.set_ylim(b['maxZ'], b['minZ'])
    ax.set_aspect('equal')
    ax.grid(True, color=GRID, lw=.5)
    ax.set_title('DAgger の状態集め(S-DIV の D0〜D11、ブラウザの閉ループ)\n実線 = 記録で再生した JetFormer の走り(フレーム k まで)、'
                 '● = 先生に替えたフレーム k、点線 = その先の JetFormer の実際の走り、× = 衝突', fontsize=9.5, loc='left', color=INK)
    ax.set_xlabel('x [m]', color=INK2)
    ax.set_ylabel('z [m]', color=INK2)
    fig.tight_layout()
    out = R / 'fig' / 'dagger_states.png'
    fig.savefig(out, dpi=115, facecolor=SURFACE)
    plt.close(fig)
    return 'fig/dagger_states.png'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', default='')
    a = ap.parse_args()
    sel, prep, g1, pred, g2 = (load(D / f) for f in ('teacher_select.json', 'prepare_record.json', 'g1.json', 'predict.json', 'g2.json'))
    res, ver, rck, rwr = (load(D / 'rec' / f) for f in ('result.json', 'verify.json', 'relabel_check.json', 'relabel_write.json'))
    a4 = load(R / 'train' / 'a4_plan.json')
    ev = load(R / 'eval_sim.json') or {}
    runs = (load(R / 'train' / 'runs.json') or {}).get('runs', {})
    st = {'schema': 'jf-sim-diverse-dagger-status/v1'}
    L = []
    if sel:
        n_ran = len(sel['rows'])
        comp = sum(1 for r in sel['rows'] if r['reason'] == 'route_complete')
        L.append(f"- 状態集め(procedure 1): 一番良い腕 S-DIV(種 0)で D0〜D11 の単周を評価と同じ設定で走らせた。完走 {comp} / {n_ran}"
                 f"(`eval/closed_loop_dagger/SDIV-s0/`)。")
        L.append(f"- 先生の選び(procedure 2): 各フレームの次の状態から 2 秒(60 フレーム)先生に任せて立て直せるフレームの数 — T_fast {sel['totals']['fast']}・"
                 f"T_margin {sel['totals']['margin']}(各 {sel['tried']['fast']} フレーム中)→ **{('T_fast' if sel['teacher'] == 'fast' else 'T_margin')}**(`dagger/teacher_select.json`)。")
        ks = prep['k'] if prep else {}
        L.append('- 切り替えのフレーム k(procedure 3。その先生に任せて単周を完走できる最後のフレーム。完走した走りは 300 → 単周の最後のフレームの 1 つ前): ' +
                 '、'.join(f"{d} {v['k']}/{v['frames_single']}" for d, v in ks.items()) + '(k / 状態集めの走行のフレーム数)。')
        st.update({'teacher': sel['teacher'], 'totals': sel['totals'], 'k': ks, 'states_completed': comp, 'states_ran': n_ran})
    if g1:
        L.append(f"- G1(Node で単周の操作を連続周回で再生し、k まで ≤ 1e-6 m): {'合格' if g1['all_pass'] else '不合格'}"
                 f"(12 本、差の最大 {max(x['max_abs_xz_diff_m'] for x in g1['rows']):.1e} m)。完走した走りは最後の 1 フレームだけ 1〜14 cm 違った"
                 '(単周の sim が周の終わりで止めるため)ので、k をその 1 つ前にした(deviations.md)。')
        st['G1'] = g1['all_pass']
    if pred:
        L.append(f"- 記録の Node の予想(同じ差し込みモジュール・元の sim-core.js): {sum(1 for r in pred['rows'] if r['ok'])} / {len(pred['rows'])} シーンが 600 フレーム・衝突 0・"
                 '全部の周が有効・ゲートの外し 0・k までの位置が状態集めと同じ。')
    if g2:
        s = g2['scenes'][0] if g2.get('scenes') else {}
        L.append(f"- G2(ブラウザの記録 1 シーンの試し): {'合格' if g2['all_pass'] else '不合格'}(フレーム k の位置の差 {s.get('pose_diff_at_k_m')} m、"
                 f"info.dart が全フレーム・jf なし・再生した操作の差 {s.get('replay_action_max_diff')})。")
        st['G2'] = g2['all_pass']
    if res:
        L.append(f"- 記録(procedure 4): 合格 {len(res.get('accepted', []))} / {res.get('scenes_planned')} シーン(取り直し {res.get('retried') or 0})、"
                 f"エラー {res.get('error') or 'なし'}。")
        st['recorded'] = len(res.get('accepted', []))
    if ver:
        L.append(f"- 取り込んだデータセットの確かめ: {'すべて合格' if ver['all_pass'] else '不合格あり'}(フレーム k の位置の差の最大 "
                 f"{max((x.get('pose_diff_at_k_m') or 0) for x in ver['scenes']):.2e} m。画像はシーンごとに {min(x.get('images_unique', 0) for x in ver['scenes'])}〜"
                 f"{max(x.get('images_unique', 0) for x in ver['scenes'])} / 600 が違い、同じ画像の組はフレーム 0・1(遅延 1 でフレーム 0 の操作が 0、"
                 f"車が動いていない同じ位置)だけ)。")
        st['verify'] = ver['all_pass']
    if rck:
        c, s = rck['checks'], rck['summary']
        L.append(f"- 付け替え(procedure 5、check): R2 {c['R2_replay']}(再生の差 {s['replay_max_abs_error']:.1e}・先生の操作の再計算の差 {s['r2_clean_vs_info_max']:.1e})・"
                 f"R3 {c['R3_noise_frames']}(先生が運転したフレームの正解 = 記録の操作、差の最大 {s['r3_unnoisy_next_max_us']:.2f} µs)・PWM の作り直し {c['pwm_recon_0us']}・"
                 f"基準の作り方との一致 {c['reference_equal']}。JetFormer の操作を再生したフレーム {s['noisy_frames']} / {s['frames']}(正解の付け替えで平均 "
                 f"{(s['r3_noisy_next_mean_us'] or 0):.1f} µs 変わった)、先生が 2 秒で立て直せず学習から外したフレーム {s['invalid_teacher_frames']}。")
        g3 = load(D / 'g3_d1_check.json')
        g3ok = bool(g3 and g3['summary']['frames_diff_gt_1us_used'] == 0)
        L.append(f"- G3(付け替えの写し): DAgger の記録で R2・R3 {'合格' if c['R2_replay'] and c['R3_noise_frames'] else '不合格'}、10/14 の D1 の 2 シーンで 10/14 の道具と同じ正解"
                 f"(差の最大 {g3['summary']['diff_vs_original_max_us_used'] if g3 else '—'} µs)→ {'合格' if g3ok and c['R2_replay'] and c['R3_noise_frames'] else '不合格'}。")
        st['G3'] = bool(g3ok and c['R2_replay'] and c['R3_noise_frames'])
    if rwr:
        L.append(f"- 付け替え write: {rwr['dataset']}(future_valid {rwr['summary']['future_valid_before']} → {rwr['summary']['future_valid_after']})。")
    if a4:
        L.append(f"- 学習(procedure 6): `{a4['init']}` から、S-DIV のデータ + DAgger の 12 シーン(全部 train)で 10 エポック(lr 1e-4)。データ `{a4['data']}`。")
    r4 = runs.get('A4/s0')
    if r4:
        L.append(f"- A4 の学習結果: `{r4.get('run')}`、最良の検証 L1 {r4.get('best_val_us')} µs({r4.get('best_epoch')} エポック目。検証は S-DIV と同じ val)。")
    v = (ev.get('verdicts') or {}).get('Q1b_DAgger')
    s4 = (ev.get('summary') or {}).get('A4')
    if v and s4:
        L.append(f"- **Q1b の判定: 「{v['verdict']}」**(A4 の完走 {s4['n_completed']} / {s4['n_ran']}・g4 {s4['g4']}・fs {s4['fs']}・d {s4['d_median_m']:.1f} m、"
                 f"元の腕 {v['base_arm']} との差: 完走 {v['c_diff']:+d}・g4 {v['g4_diff']:+d}・fs {v['fs_diff']:+d})。")
        st['verdict'] = v['verdict']
    if a.cut:
        L.append(f"- **DAgger は途中で止めた**: {a.cut}")
        st['cut'] = a.cut
    try:
        if sel and prep:
            st['figure'] = figure(sel, prep)
            L.append(f"- 図: `{st['figure']}`(状態集めの走行と切り替えのフレーム k)")
    except Exception as e:  # noqa: BLE001
        st['figure_error'] = f'{type(e).__name__}: {e}'
    st['verdict_text'] = st.get('verdict') and f"「{st['verdict']}」" or (f"止めた: {a.cut}" if a.cut else '途中')
    st['md_lines'] = L
    (R / 'eval' / 'dagger_status.json').write_text(json.dumps(st, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
