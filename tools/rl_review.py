#!/usr/bin/env python3
"""オフライン強化学習 その3(tools/rl_evaluate.py)のレビューで足した確認。すべて事後の分析で、heldout.json の
判定の規則と、先に決めた判定(evaluation.json の verdict・candidate)は変えない。

    python tools/rl_review.py control-prepare   # (h) 人だけの微調整のフレームの表 → data/rl/<データセット>/policy/drop_human.npz
    python tools/rl_review.py control-train     # GPU: (h) を学習。エンジンは作らない(管理画面の走行の一覧に出ない)。続きから再開
    python tools/rl_evaluate.py predict --ckpt last                  # GPU: 先に決めた7通りを last.pth でも予測
    python tools/rl_evaluate.py predict --configs h [--ckpt last]    # GPU: (h) の予測
    python tools/rl_review.py analyze --report-dir <dir>   # CPU: 下の 1〜9 → eval/review.json(evaluation.json の review にも入れる)
    python tools/rl_review.py disable-engines   # 走らせない方策のエンジンの名前を変える(管理画面の走行の一覧から消える)
    python tools/rl_review.py figures --fig-dir <dir>

analyze で調べること(レビューの指摘の順):
  1 チェックポイントのばらつき: 同じ学習の best.pth と last.pth の出力の差(R1・DE)と、モデルどうしの差を
    best/best・last/last・混ぜた組で計算し、結論がチェックポイントで変わるかを見る(ゲートC の判定も4通りで計算し直す)
  2 印だけが違う (a) − (n)(フレーム・初期値・種・エポック・ジャンルの戻しが同じ)
  3 全体のずれ: モデルの差が場所に関係なく全部のフレームで出ているか(シーンの平均を引いた DE・R1 の差)
  4 DE の出口で、人の実際の操作との誤差(ゲインを掛ける前の 3フレーム先の予測 − 正解)
  5 保留した人のきれいな走行での、ステアの偏り(予測 − 人)と L1((h) を含む)
  6 学習データの目標: (c) の学習フレームの AI と人のステア、学習側の DE 出口の通過
  7 項目5(学習の印)を場所でそろえた見直しと、人の後退を「比べる集団との差」で見た場合
  8 価値 V の AUROC を速さの十分位・衝突までの時間で分けたもの
  9 保留データの使われ方(最良のエポックの選択・common-val の時刻・V の交差推定の fold)

GPU を使うのは control-train だけ(1つずつ。ほかの GPU の処理が動いていたら始めない)。電源が落ちても、同じコマンドを
もう一度実行すれば続きから(学習は train.py --resume、予測は1通りずつ npz)。
"""

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from jetformer import config  # noqa: E402
import rl_policies as rp  # noqa: E402
import rl_evaluate as ev  # noqa: E402

DATASET = ev.DATASET
CKPTS = ev.CKPTS
PRE = list(ev.CONFIGS)                     # 先に決めた7通り
ALL = PRE + list(ev.POSTHOC_CONFIGS)       # + (h)
# モデルどうしの差(x − 基準)。R1 は + = 遅い、DE は + = 左
CONTRASTS = [('c', 'B'), ('h', 'B'), ('c', 'h'), ('a_good', 'c'), ('b', 'c'), ('n_good', 'c'), ('a_good', 'n_good'),
             ('a_base', 'n_base'), ('a_good', 'B'), ('b', 'B'), ('n_good', 'B')]
PAIRINGS = (('best', 'best'), ('last', 'last'), ('best', 'last'), ('last', 'best'))   # (x の重み, 基準の重み)
# ゲートC を4通りの重みで計算し直す: 方策((a)(b)(n))の重み / 基準((c)・モデル B)の重み
GATE_VARIANTS = {'best/best': ('best', 'best'), 'last/last': ('last', 'last'),
                 'policies_last/refs_best': ('last', 'best'), 'policies_best/refs_last': ('best', 'last')}
REFS = ('B', 'c')
TRAINING_NOISE_US = 10.0                   # 別々に学習したモデルの差で、これ以下は学習のばらつきの内とみなす(1 の結果から)
DISABLE_RUNS = ('n', 'a', 'b', 'c')        # 走らせない方策(ゲートC に合格したものが無い)


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- numpy だけの規則(テストする)

def sign_test(d):
    """符号検定(両側)。0 は除く。{'pos': 正の数, 'n': 0 でない数, 'p': 両側の p 値}"""
    d = np.asarray(d, np.float64)
    k, n = int((d > 0).sum()), int((d != 0).sum())
    if n == 0:
        return {'pos': 0, 'n': 0, 'p': None}
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0 ** n
    return {'pos': k, 'n': n, 'p': round(min(1.0, 2 * tail), 4)}


def range_summary(cis):
    """区間のリスト [[平均, 下, 上], ...] → {'points': [最小, 最大], 'all_pos': 全部の下限 > 0, 'all_neg': 全部の上限 < 0,
    'any_excludes_zero': どれかの区間が 0 を含まない}"""
    cis = [c for c in cis if c and c[0] is not None]
    if not cis:
        return {'points': None, 'all_pos': False, 'all_neg': False, 'any_excludes_zero': False}
    pts = [c[0] for c in cis]
    return {'points': [min(pts), max(pts)], 'all_pos': all(c[1] > 0 for c in cis),
            'all_neg': all(c[2] < 0 for c in cis), 'any_excludes_zero': any(c[1] > 0 or c[2] < 0 for c in cis)}


def minus_group_mean(d, group, base_mask):
    """行ごとの差 d から、同じグループ(シーン)の base_mask の行の平均を引く。グループに base の行が無ければ NaN"""
    d = np.asarray(d, np.float64)
    group = np.asarray(group)
    base_mask = np.asarray(base_mask, bool)
    out = np.full(len(d), np.nan)
    for g in np.unique(group[base_mask]):
        m = group == g
        out[m] = d[m] - d[m & base_mask].mean()
    return out


def location_adjusted_share(flag, good, loc, ref):
    """flag のフレームの「良い」の割合と、同じ場所の flag でない ref のフレームから見込んだ割合(場所の構成をそろえる)。
    戻り値 {'n', 'observed', 'expected', 'excess', 'by_loc': {場所: (flag の数, flag の割合, flag でない割合)}}"""
    flag, good, ref = np.asarray(flag, bool), np.asarray(good, bool), np.asarray(ref, bool)
    loc = np.asarray(loc)
    n = int(flag.sum())
    if n == 0:
        return {'n': 0, 'observed': None, 'expected': None, 'excess': None, 'by_loc': {}}
    exp_sum, by = 0.0, {}
    for k in np.unique(loc[flag]):
        f_k = flag & (loc == k)
        r_k = ref & ~flag & (loc == k)
        sh_ref = float(good[r_k].mean()) if r_k.any() else float(good[ref & ~flag].mean())
        exp_sum += f_k.sum() * sh_ref
        by[int(k)] = (int(f_k.sum()), round(float(good[f_k].mean()), 3), round(sh_ref, 3))
    obs = float(good[flag].mean())
    return {'n': n, 'observed': round(obs, 4), 'expected': round(exp_sum / n, 4), 'excess': round(obs - exp_sum / n, 4),
            'by_loc': by}


def auroc(score, y):
    """AUROC(順位。同じ値は平均順位)。陽性か陰性が無ければ None"""
    score, y = np.asarray(score, np.float64), np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(score, kind='mergesort')
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def human_only_drop(drop_clean, source, split):
    """(h) のフレームの表: (c) の drop に、学習側(train)の AI のフレームを足す。検証は (c) と同じ(最良のエポックを
    同じ検証フレームで選ぶため)"""
    return np.asarray(drop_clean, bool) | ((np.asarray(source) == 1) & (np.asarray(split) == 'train'))


def disable_engine(run_dir, suffix=rp.DISABLED_ENGINE_SUFFIX):
    """jetformer_fp16.engine → jetformer_fp16.engine.do_not_drive(何度実行しても同じ)。戻り値 'renamed'・'already'・
    'missing'"""
    run_dir = Path(run_dir)
    src, dst = run_dir / 'jetformer_fp16.engine', run_dir / ('jetformer_fp16.engine' + suffix)
    if src.exists() and not dst.exists():
        os.replace(src, dst)
        return 'renamed'
    if dst.exists() and not src.exists():
        return 'already'
    if src.exists() and dst.exists():
        raise SystemExit('%s と %s の両方があります。どちらかを手で消してください' % (src, dst))
    return 'missing'


# ---------------------------------------------------------------- (h) 人だけの微調整(対照)

def cmd_control_prepare(args):
    pdir = rp.policy_dir(DATASET)
    out = pdir / 'drop_human.npz'
    if out.exists():
        print('作成済み: %s' % out)
        return
    with np.load(pdir / 'drop_clean.npz', allow_pickle=False) as z:
        dc = {k: z[k] for k in z.files}
    f = ev._load_frames()
    if not (np.array_equal(dc['scene'], f['scene'].astype(dc['scene'].dtype)) and np.array_equal(dc['frame'], f['frame'])):
        raise SystemExit('drop_clean.npz と frames.npz のフレームの並びが違います')
    drop = human_only_drop(dc['drop'], f['source'], f['split'])
    rp._save_npz(out, drop=drop, scene=dc['scene'], frame=dc['frame'])
    st = (f['frame'] % rp.TRAIN['stride']) == 0
    for sp in ('train', 'val'):
        m = ~drop & (f['split'] == sp)
        print('%s: 使う(間引き前)%d・人 %d・AI %d(間引き後 %d。ただし 2.5G の除外の前)' % (
            sp, int(m.sum()), int((m & (f['source'] == 0)).sum()), int((m & (f['source'] == 1)).sum()), int((m & st).sum())))
    print('書き出し:', out)


def cmd_control_train(args):
    pdir = rp.policy_dir(DATASET)
    mask = pdir / 'drop_human.npz'
    if not mask.exists():
        raise SystemExit('先に control-prepare を実行してください')
    out = rp.RUNS_DIR / ev.H_RUN
    with rp.RunLock(pdir / 'run.lock'):
        out.mkdir(parents=True, exist_ok=True)
        if not (out / 'run.json').exists():
            rp._save_json(out / 'run.json', {
                'dataset': DATASET, 'stride': rp.TRAIN['stride'], 'epochs': ev.EPOCHS, 'batch': rp.TRAIN['batch'],
                'lr': rp.TRAIN['lr'], 'pretrained': True, 'build_engine': False, 'command_input': False,
                'mask_impacts': True, 'impact_g': rp.TRAIN['impact_g'], 'name': out.name,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'), 'rl_variant': 'h', 'init': rp.BASE_RUN,
                'note': 'オフラインRLのレビューの対照 (h): (c) filtered BC と同じ設定で、学習を人のフレームだけにした微調整'
                        '(tools/rl_review.py)。「AI のデータを足した効果」と「微調整そのものの効果」を分けるため。'
                        'エンジンは作らない。走らせない'})
        if rp.training_done(out, ev.EPOCHS):
            print('学習済み: %s' % out.name)
            return
        rp._check_gpu_free()
        resume = (out / 'last.pth').exists()
        cmd = rp.train_command('c', out, ev.EPOCHS, DATASET, resume=resume)
        cmd[cmd.index('--drop-mask') + 1] = str(mask)
        print('学習%s: %s' % ('(再開)' if resume else '', ' '.join(cmd)), flush=True)
        t0 = time.time()
        started = datetime.datetime.now().isoformat(timespec='seconds')
        with open(out / 'train.log', 'a', encoding='utf-8') as log:
            log.write('\n==== %s %s\n' % (started, ' '.join(cmd)))
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(config.ROOT))
        rp._append_wall(out, {'step': 'train', 'start': started, 'seconds': round(time.time() - t0, 1), 'resume': resume,
                              'returncode': rc})
        if rc != 0 or not rp.training_done(out, ev.EPOCHS):
            raise SystemExit('学習が終わっていません(終了コード %d)。%s/train.log を見て、同じコマンドで再開してください' % (rc, out))
        print('学習 %.0f 秒: %s' % (time.time() - t0, out), flush=True)


# ---------------------------------------------------------------- 走らせない方策のエンジン

def cmd_disable_engines(args):
    rows = {}
    for v in DISABLE_RUNS:
        d = rp.RUNS_DIR / ev.run_name(v)
        state = disable_engine(d)
        rows[v] = {'run': d.name, 'state': state,
                   'restore': 'mv %s/jetformer_fp16.engine%s %s/jetformer_fp16.engine' % (d, rp.DISABLED_ENGINE_SUFFIX, d)}
        print(v, state, d.name)
    from jetformer import drive
    listed = [m['name'] for m in drive.list_models()]
    rec = {'at': datetime.datetime.now().isoformat(timespec='seconds'), 'runs': rows, 'drive_list_models_after': listed,
           'note': 'ゲートC に合格した方策が無いので、4本とも管理画面の走行の一覧から外した(エンジンの名前を変えただけ。'
                   '戻すときは restore のコマンド)。rl_policies.py run は .do_not_drive があるとエンジンを作り直さない'}
    rp._save_json(ev.eval_dir() / 'disabled_engines.json', rec)
    print('走行の一覧:', listed)


# ---------------------------------------------------------------- analyze(CPU)

def _setup():
    held = ev._heldout()
    tab = ev.pred_table()
    cols, f = ev._val_arrays(tab)
    preds = {}
    for ck in CKPTS:
        for n in ALL:
            if ev.pred_path(n, ck).exists():
                preds['%s@%s' % (n, ck)] = ev.load_preds([n], ck)[n]
    refs = {n: ev._ref_of(n) for n in ALL if (rp.RUNS_DIR / ev.config_of(n)[0] / 'model_config.json').exists()}
    s = ev.DRIVE_SETTINGS
    outs = {k: ev.drive_outputs(p, refs[k.split('@')[0]], s['steer_gain'], s['throttle_gain'], s['lookahead'],
                                s['steer_trim_us']) for k, p in preds.items()}
    m0 = tab['part'] == 0
    prow = np.full(len(f['scene']), -1, int)
    prow[cols['frow'][m0]] = np.nonzero(m0)[0]
    with open(rp.rl_dir(DATASET) / 'passes.json') as fh:
        passes = json.load(fh)
    r1, de = ev.heldout_passes(passes, f['log'], f['timestamps_us'], f['split'])
    pv = ev._pass_values(r1, prow, outs, cols)
    pd_ = ev._pass_values(de, prow, outs, cols)
    return {'held': held, 'tab': tab, 'cols': cols, 'f': f, 'preds': preds, 'outs': outs, 'refs': refs, 'm0': m0,
            'prow': prow, 'r1': r1, 'de': de, 'pv': pv, 'pd': pd_, 'passes': passes}


def _have(S, *keys):
    return all(k in S['outs'] for k in keys)


def _pass_ci(P, key, x, ref, logs=False):
    d = np.array([q['%s_%s' % (key, x)] - q['%s_%s' % (key, ref)] for q in P])
    out = {'ci': ev.bootstrap_mean(d), 'n': len(d)}
    ai = np.array([q['ai'] for q in P], bool)
    out['ci_ai'] = ev.bootstrap_mean(d[ai])
    out['ci_human'] = ev.bootstrap_mean(d[~ai])
    if logs:
        out['log_cluster_ci'] = ev.cluster_bootstrap({'d': d}, np.array([q['log'] for q in P]), lambda s, c: s['d'] / c)
        out['logs'] = len(set(q['log'] for q in P))
        out['sign'] = sign_test(d)
    return out


def ckpt_noise(S):
    """1a 同じ学習の last.pth − best.pth(R1 のスロットル・DE のステア・全フレームの平均の動き)と検証の誤差"""
    out = {}
    m0 = S['m0']
    for n in ALL:
        b, l = '%s@best' % n, '%s@last' % n
        if not _have(S, b, l):
            continue
        res = rp._read_json(rp.RUNS_DIR / ev.config_of(n)[0] / 'result.json') or {}
        hist = res.get('history') or []
        ob, ol = S['outs'][b][m0], S['outs'][l][m0]
        out[n] = {'r1_d_thr_last_minus_best': _pass_ci(S['pv'], 'thr', l, b)['ci'],
                  'de_d_steer_last_minus_best': _pass_ci(S['pd'], 'steer', l, b)['ci'],
                  'all_val_mean_d_out_us': [_r((ol[:, 0] - ob[:, 0]).mean()), _r((ol[:, 1] - ob[:, 1]).mean())],
                  'all_val_mean_abs_d_out_us': [_r(np.abs(ol[:, 0] - ob[:, 0]).mean()), _r(np.abs(ol[:, 1] - ob[:, 1]).mean())],
                  'best_epoch': res.get('best_epoch'), 'epochs': res.get('epochs_done'),
                  'val_us_best': res.get('best_val_us'), 'val_us_last': hist[-1]['val_us'] if hist else None}
    return out


def contrasts(S):
    """1b モデルどうしの差を4通りの重みの組で。2 の (a) − (n) もここに入る(符号検定・ログ単位の区間つき)"""
    out = {}
    for x, ref in CONTRASTS:
        row = {}
        for cx, cr in PAIRINGS:
            a, b = '%s@%s' % (x, cx), '%s@%s' % (ref, cr)
            if not _have(S, a, b):
                continue
            row['%s/%s' % (cx, cr)] = {'r1_d_thr': _pass_ci(S['pv'], 'thr', a, b, logs=True),
                                      'de_d_steer': _pass_ci(S['pd'], 'steer', a, b, logs=True)}
        if row:
            row['range_r1'] = range_summary([v['r1_d_thr']['ci'] for k, v in row.items() if '/' in k])
            row['range_de'] = range_summary([v['de_d_steer']['ci'] for k, v in row.items() if '/' in k])
            out['%s-%s' % (x, ref)] = row
    return out


def gate_by_ckpt(S):
    """1c ゲートC と (c) の候補の判定を、4通りの重みの組で計算し直す(判定の規則は heldout.json のまま)"""
    out = {}
    for name, (cp, cr) in GATE_VARIANTS.items():
        preds = {n: S['preds']['%s@%s' % (n, cr if n in REFS else cp)] for n in PRE}
        res = ev.evaluate_all(preds=preds)
        v = res['verdict']
        out[name] = {
            'table': v['table'], 'negative_control_rule_met': v['negative_control_rule_met'],
            'gate_discriminates': v['gate_discriminates'], 'separating_items': v['separating_items'],
            'candidate': res['candidate'],
            'r1_vs_c': {x: res['item2_r1']['vs']['%s-c' % x]['d_thr_us'] for x in ev.JUDGED + ev.EXTRA},
            'de_vs_c': {x: res['item3_de']['vs']['%s-c' % x]['d_steer_us'] for x in ev.JUDGED + ev.EXTRA},
            'c_minus_B': {'r1': res['item2_r1']['vs']['c-B']['d_thr_us'], 'de': res['item3_de']['vs']['c-B']['d_steer_us']},
            'share_lt5500': {n: res['item2_r1']['per_config'][n]['share_lt5500'] for n in PRE},
            'collapse_vs_B': {x: {k: res['item4_collapse']['vs']['%s-B' % x][k] for k in ('straight_drop', 'l1_ratio', 'pass')}
                              for x in ['c'] + ev.JUDGED},
            'human_clean_l1': {n: res['accuracy']['metrics'][n]['human_clean']['l1_us'] for n in PRE},
            'human_clean_l1_diff_vs_B': {x: res['accuracy']['diffs']['%s-B/human_clean' % x]['l1'] for x in PRE if x != 'B'},
            'iv_all_l1': {n: res['accuracy']['metrics'][n]['iv_all']['l1_us'] for n in PRE},
            'iv_all_diff_vs_B': {x: {k: res['accuracy']['diffs']['%s-B/iv_all' % x][k] for k in ('l1', 'throttle_mae')}
                                 for x in PRE if x != 'B'},
            'iv_log_heldout_l1': {n: res['accuracy']['metrics'][n]['iv_log_heldout']['l1_us'] for n in PRE},
            'iv_fwd_bias': {n: res['accuracy']['metrics'][n]['iv_all']['fwd_bias_us'] for n in PRE}}
    return out


def _sets(S):
    held, tab, cols, m0 = S['held'], S['tab'], S['cols'], S['m0']
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human & cols['fv'] & ~cols['impact']
    return {'all_val': m0, 'human_clean': clean, 'human_clean_genre4': clean & (cols['mode'] == 4),
            'human_clean_genre5': clean & (cols['mode'] == 5), 'ai_val': m0 & (cols['source'] == 1)}


def global_offset(S):
    """3 全体のずれ: 出力の差を、保留の全フレーム・人のきれいな走行(ジャンル別)・AI のフレームで平均し、DE・R1 の窓の差から
    同じシーンの平均の差を引いたもの(場所に固有の差)と並べる"""
    tab, sets = S['tab'], _sets(S)
    scene = tab['scene']
    out = {}
    for x, ref in [('c', 'B'), ('h', 'B'), ('c', 'h'), ('a_good', 'c'), ('b', 'c'), ('n_good', 'c'), ('a_good', 'n_good'),
                   ('a_good', 'B'), ('b', 'B'), ('n_good', 'B')]:
        for ck in CKPTS:
            a, b = '%s@%s' % (x, ck), '%s@%s' % (ref, ck)
            if not _have(S, a, b):
                continue
            row = {}
            for j, nm in ((0, 'steer'), (1, 'thr')):
                d = S['outs'][a][:, j] - S['outs'][b][:, j]
                row[nm] = {k: ev.cluster_bootstrap({'d': d[m]}, scene[m], lambda s, c: s['d'] / c) for k, m in sets.items()}
                dm = minus_group_mean(d, scene, S['m0'])
                P, rows_of = (S['pd'], S['de']) if nm == 'steer' else (S['pv'], S['r1'])
                w_raw, w_dm = [], []
                for p in rows_of:
                    r = S['prow'][p['rows']]
                    r = r[r >= 0]
                    w_raw.append(d[r].mean())
                    w_dm.append(np.nanmean(dm[r]))
                key = 'de_windows' if nm == 'steer' else 'r1_windows'
                row[nm][key] = ev.bootstrap_mean(w_raw)
                row[nm][key + '_minus_scene_mean'] = ev.bootstrap_mean(w_dm)
                ai = np.array([p['ai'] for p in rows_of], bool)
                row[nm][key + '_ai'] = ev.bootstrap_mean(np.array(w_raw)[ai])
            out['%s-%s@%s' % (x, ref, ck)] = row
    out['note'] = ('走行の出力(ゲイン 1.3・1.0・先読み 3)の差の平均。区間はシーン単位(窓はシーンの中の通過単位)の'
                   'ブートストラップ。minus_scene_mean = 行ごとの差から、そのシーンの保留の全フレームの差の平均を引いてから'
                   '窓で平均したもの(場所に固有の差)')
    return out


def vs_labels(S):
    """4・5 人(とAI)の実際の操作との比較。ゲインを掛ける前の 3フレーム先の予測 − 正解(先読み 3 と同じ位置)"""
    cols, preds = S['cols'], S['preds']
    out = {'de_passes': {}, 'human_clean': {}}
    names = [k for k in preds]
    # DE の通過ごと
    per = []
    for p, q in zip(S['de'], S['pd']):
        r = S['prow'][p['rows']]
        r = r[(r >= 0)]
        r = r[cols['fv'][r]]
        if not len(r):
            continue
        y = cols['y'][r, 2, 0]
        e = {k: preds[k][r, 2, 0] - y for k in names}
        per.append({'log': q['log'], 'ai': bool(q['ai']), 'frames': int(len(r)), 'label_steer': _r(y.mean(), 1),
                    'signed': {k: float(v.mean()) for k, v in e.items()},
                    'mae': {k: float(np.abs(v).mean()) for k, v in e.items()}})
    hum = [p for p in per if not p['ai']]
    for grp, P in (('human', hum), ('all', per)):
        g = {'passes': len(P), 'signed_err_us': {k: ev.bootstrap_mean([p['signed'][k] for p in P]) for k in names},
             'mae_us': {k: _r(np.mean([p['mae'][k] for p in P]), 2) for k in names}, 'mae_diff': {}}
        for x, ref in (('c', 'B'), ('h', 'B'), ('c', 'h'), ('a_good', 'B'), ('b', 'B'), ('n_good', 'B'), ('a_good', 'n_good'),
                       ('a_good', 'c')):
            for ck in CKPTS:
                a, b = '%s@%s' % (x, ck), '%s@%s' % (ref, ck)
                if a in preds and b in preds:
                    g['mae_diff']['%s-%s@%s' % (x, ref, ck)] = ev.bootstrap_mean([p['mae'][a] - p['mae'][b] for p in P])
        out['de_passes'][grp] = g
    out['de_pass_rows'] = [{k: (v if not isinstance(v, dict) else {kk: _r(vv, 1) for kk, vv in v.items()
                                                                    if kk.split('@')[0] in ('B', 'c', 'h')})
                            for k, v in p.items()} for p in per]
    # 保留した人のきれいな走行: ステアの偏り・L1
    sets = _sets(S)
    m = sets['human_clean']
    sc = S['tab']['scene'][m]
    l1 = {k: np.abs(preds[k][m] - cols['y'][m]).mean(axis=(1, 2)) for k in names}
    for k in names:
        out['human_clean'][k] = {
            'steer_bias_us': ev.cluster_bootstrap({'d': preds[k][m, 2, 0] - cols['y'][m, 2, 0]}, sc, lambda s, c: s['d'] / c),
            'throttle_bias_us': ev.cluster_bootstrap({'d': preds[k][m, 2, 1] - cols['y'][m, 2, 1]}, sc, lambda s, c: s['d'] / c),
            'l1_us': _r(l1[k].mean(), 2)}
    logs = np.array([s[:15] for s in sc])
    out['human_clean_l1_diff'] = {}
    for x, ref in (('c', 'B'), ('h', 'B'), ('c', 'h'), ('b', 'B'), ('a_good', 'B'), ('n_good', 'B')):
        for ck in CKPTS:
            a, b = '%s@%s' % (x, ck), '%s@%s' % (ref, ck)
            if a in l1 and b in l1:
                d = l1[a] - l1[b]
                out['human_clean_l1_diff']['%s-%s@%s' % (x, ref, ck)] = {
                    'scene_ci': ev.cluster_bootstrap({'d': d}, sc, lambda s, c: s['d'] / c),
                    'log_ci': ev.cluster_bootstrap({'d': d}, logs, lambda s, c: s['d'] / c),
                    'ratio': _r(l1[a].mean() / l1[b].mean(), 4)}
    out['human_clean_frames'] = int(m.sum())
    out['human_clean_logs'] = int(len(set(logs)))
    out['note'] = 'signed = 予測 − 正解(ステア、+ = 人より左)。DE の窓は緑マット出口〜0.8秒、future_valid のフレームだけ'
    return out


def training_targets(S):
    """6 学習データの目標(3フレーム先の正解のステア): (c)・(h)・(b)(a)(n) の学習フレームの AI と人、学習側の DE 出口"""
    f = S['f']
    pdir = rp.policy_dir(DATASET)
    with np.load(pdir / 'drop_clean.npz') as z:
        dc = z['drop']
    with np.load(pdir / 'drop_all.npz') as z:
        da, keep = z['drop'], z['keep_impact']
    imp, usable = rp._impact_and_stride(DATASET, f['scene'], f['frame'], rp.TRAIN['impact_g'])
    st = (f['frame'] % rp.TRAIN['stride']) == 0
    tr = (f['split'] == 'train') & usable & st
    used = {'c': tr & ~dc & ~imp, 'h': tr & ~dc & ~imp & (f['source'] == 0), 'b_a_n': tr & ~da & ~(imp & ~keep)}
    lab = f['future_actions'][:, 2, 0].astype(np.float64)
    out = {'label': '3フレーム先の正解のステア [µs](+ = 左)', 'sets': {}}
    for k, m in used.items():
        row = {}
        for src, sn in ((0, 'human'), (1, 'ai')):
            for g in (4, 5):
                mm = m & (f['source'] == src) & (f['mode'] == g)
                if mm.any():
                    row['%s_genre%d' % (sn, g)] = {'n': int(mm.sum()), 'steer_us': _r(lab[mm].mean(), 1)}
        out['sets'][k] = row
    # 学習側の DE 出口の通過(緑マット出口〜0.8秒、(c) が使うフレーム)
    by_log = {}
    for lg in np.unique(f['log']):
        mm = np.nonzero(f['log'] == lg)[0]
        by_log[str(lg)] = mm[np.argsort(f['timestamps_us'][mm], kind='stable')]
    grp = {}
    for p in S['passes']:
        if p.get('tg_s') is None or p['log'] not in by_log:
            continue
        rows = by_log[p['log']]
        t = (f['timestamps_us'][rows] - p['t0_us']) / 1e6
        w = rows[(t >= p['tg_s']) & (t <= p['tg_s'] + ev.DE_WINDOW_S)]
        w = w[(f['split'][w] == 'train') & ~dc[w] & np.isfinite(lab[w])]
        if len(w):
            key = '%s_%s' % ('ai' if p['ai'] else 'human', 'collision' if p['de_collision'] else 'no_collision')
            grp.setdefault(key, []).append(lab[w].mean())
    out['train_de_exit'] = {k: {'passes': len(v), 'steer_us': _r(np.mean(v), 1)} for k, v in sorted(grp.items())}
    return out


def label_audit_adjusted(S):
    """7 項目5 の見直し(事後): 空転を場所でそろえる、人の後退を比べる集団(人の使えるフレーム)との差で見る"""
    f = S['f']
    adv, mk = ev.label_masks(f)
    th = ev.RULES['thresholds']
    out = {}
    for lab_name in ('I', 'I_neg'):
        good = adv[lab_name] == 1
        elig = mk['elig']
        spin = location_adjusted_share(elig & mk['spin'], good, f['loc_code'], elig & mk['ref_spin'])
        spin['by_loc'] = {ev.LOC_NAMES[k]: v for k, v in spin['by_loc'].items()}
        rev = elig & mk['reverse']
        ref = elig & (f['source'] == 0)
        rev_share, ref_share = float(good[rev].mean()), float(good[ref].mean())
        # 人の後退を除いた参照(後退が参照に入っている分を除く)
        ref_ex = ref & ~mk['reverse']
        out[lab_name] = {
            'spin_location_adjusted': spin, 'spin_pass_adjusted': bool(spin['excess'] <= th['label_excess_max']),
            'human_reverse': {'n': int(rev.sum()), 'good_share': _r(rev_share, 4), 'ref_good_share': _r(ref_share, 4),
                              'ref_good_share_excl_reverse': _r(float(good[ref_ex].mean()), 4),
                              'excess': _r(rev_share - float(good[ref_ex].mean()), 4)},
            'reverse_pass_excess_rule': bool(rev_share - float(good[ref_ex].mean()) <= th['label_excess_max'])}
    out['note'] = ('事後の見直し(判定は heldout.json の規則のまま)。空転: 同じ場所の空転でないフレームの「良い」の割合で'
                   '見込んだ割合との差。人の後退: 絶対の 10% ではなく、人の後退以外の使えるフレームとの差(ほかの項目と同じ形)')
    return out


def value_diagnostics():
    """8 価値 V(位置なし)の AUROC を、速さの十分位と、衝突までの時間で分ける(value_eval_scores.npz)"""
    z = dict(np.load(rp.rl_dir(DATASET) / 'value_eval_scores.npz', allow_pickle=False))
    f = ev._load_frames()
    m = z['eval_ai']
    y, sV, sR, sp = z['y'][m], z['score_V_feat'][m], z['score_V_feat_risk'][m], z['score_speed'][m]
    nc = z['next_collision_s'][m]
    out = {'frames': int(m.sum()), 'positives': int(y.sum()), 'auroc_V': _r(auroc(sV, y), 4),
           'auroc_risk': _r(auroc(sR, y), 4), 'auroc_speed': _r(auroc(sp, y), 4)}
    edges = np.quantile(sp, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, sp, side='right') - 1, 0, 9)
    out['by_speed_decile'] = [{'decile': k + 1, 'frames': int((dec == k).sum()), 'pos': int(y[dec == k].sum()),
                               'auroc_V': _r(auroc(sV[dec == k], y[dec == k]), 3),
                               'auroc_risk': _r(auroc(sR[dec == k], y[dec == k]), 3)} for k in range(10)]
    out['by_time_to_collision'] = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0)):
        pm = y & (nc >= lo) & (nc < hi) if hi < 1.0 else y & (nc >= lo) & (nc <= hi)
        keep = pm | ~y
        out['by_time_to_collision'].append({'s': [lo, hi], 'pos': int(pm.sum()), 'auroc_V': _r(auroc(sV[keep], pm[keep]), 3),
                                            'auroc_risk': _r(auroc(sR[keep], pm[keep]), 3),
                                            'auroc_speed': _r(auroc(sp[keep], pm[keep]), 3)})
    loc = f['loc_code'][m]
    out['by_location'] = {ev.LOC_NAMES[k]: {'frames': int((loc == k).sum()), 'mean_score_V': _r(sV[loc == k].mean(), 3),
                                            'collision_rate': _r(y[loc == k].mean(), 3)}
                          for k in np.unique(loc)}
    out['note'] = ('score_V = −V(高いほど危ない)。十分位は速さ(score_speed)で切った。衝突までの時間は、その範囲の陽性と、'
                   '陰性の全部で AUROC')
    return out


def heldout_usage(S):
    """9 保留データが選択に使われたか(最良のエポック・common-val の時刻・V の交差推定の fold)"""
    out = {'heldout_created_at': S['held']['created_at']}
    cv = rp.policy_dir(DATASET) / 'common_val.json'
    if cv.exists():
        out['common_val_written_at'] = datetime.datetime.fromtimestamp(cv.stat().st_mtime).isoformat(timespec='seconds')
    out['best_epochs'] = {}
    for n in ['B', 'c', 'b', 'a_good', 'n_good', 'h']:
        run = ev.config_of(n)[0]
        res = rp._read_json(rp.RUNS_DIR / run / 'result.json') or {}
        out['best_epochs'][n] = {'run': run, 'best_epoch': res.get('best_epoch'), 'epochs': res.get('epochs_done'),
                                 'best_val_us': res.get('best_val_us'), 'dataset': res.get('dataset')}
    folds = (rp._read_json(rp.rl_dir(DATASET) / 'log_folds.json') or {}).get('folds') or []
    val_logs = sorted({s[:15] for s in S['held']['val_scenes']})
    out['value_folds'] = {'fold0': folds[0] if folds else None, 'val_logs': val_logs,
                          'val_logs_in_fold0': [lg for lg in val_logs if folds and lg in folds[0]],
                          'note': 'V は fold ごとに、ほかの fold で学習した(内側の検証 fold も、ほかの fold から選ぶ)。学習側'
                                  '(fold 1〜4)のフレームの A・印は、fold 0(val のログ 172505・204208 を含む)を学習か検証に'
                                  '使った V から作られている'}
    return out


def analyze():
    S = _setup()
    missing = [k for k in ('%s@%s' % (n, c) for n in PRE for c in CKPTS) if k not in S['preds']]
    if missing:
        raise SystemExit('予測がありません: %s(rl_evaluate.py predict --ckpt last)' % missing)
    t0 = time.time()
    rec = {'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'note': 'レビューの後の事後の分析(tools/rl_review.py)。heldout.json の判定の規則と先に決めた判定は変えていない',
           'available': sorted(S['preds']),
           'counts': {'r1_passes': len(S['pv']), 'de_passes': len(S['pd'])},
           'training_noise_us': TRAINING_NOISE_US}
    rec['ckpt_noise'] = ckpt_noise(S)
    rec['contrasts'] = contrasts(S)
    rec['global_offset'] = global_offset(S)
    rec['vs_labels'] = vs_labels(S)
    rec['training_targets'] = training_targets(S)
    rec['heldout_usage'] = heldout_usage(S)
    rec['value_diagnostics'] = value_diagnostics()
    rec['label_audit_adjusted'] = label_audit_adjusted(S)
    rec['gate_by_ckpt'] = gate_by_ckpt(S)
    # (a)(n) の、同じ方策でジャンルだけ替えたとき(last.pth)
    gs = {}
    for x, ref in (('a_good', 'a_base'), ('n_good', 'n_base')):
        for ck in CKPTS:
            a, b = '%s@%s' % (x, ck), '%s@%s' % (ref, ck)
            gs['%s-%s@%s' % (x, ref, ck)] = {'r1_d_thr': _pass_ci(S['pv'], 'thr', a, b)['ci'],
                                             'de_d_steer': _pass_ci(S['pd'], 'steer', a, b)['ci']}
    rec['genre_switch'] = gs
    dis = rp._read_json(ev.eval_dir() / 'disabled_engines.json')
    if dis:
        rec['disabled_engines'] = dis
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


def cmd_analyze(args):
    rec = analyze()
    rp._save_json(ev.eval_dir() / 'review.json', rec)
    res = rp._read_json(ev.eval_dir() / 'evaluation.json')
    if not res:
        raise SystemExit('先に rl_evaluate.py evaluate を実行してください')
    res['review'] = rec
    rp._save_json(ev.eval_dir() / 'evaluation.json', res)
    if args.report_dir:
        rp._save_json(Path(args.report_dir) / 'evaluation.json', res)
        rp._save_json(Path(args.report_dir) / 'evaluation' / 'evaluation.json', res)
        rp._save_json(Path(args.report_dir) / 'evaluation' / 'review.json', rec)
    print('書き出し: %s(%.0f秒)' % (ev.eval_dir() / 'review.json', rec['seconds']))


def cmd_figures(args):
    import figures_rl_review as fig
    rec = rp._read_json(ev.eval_dir() / 'review.json')
    if not rec:
        raise SystemExit('先に analyze を実行してください')
    fig.make_all(rec, Path(args.fig_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('control-prepare')
    sub.add_parser('control-train')
    p = sub.add_parser('analyze')
    p.add_argument('--report-dir', default='')
    sub.add_parser('disable-engines')
    p = sub.add_parser('figures')
    p.add_argument('--fig-dir', required=True)
    args = ap.parse_args()
    {'control-prepare': cmd_control_prepare, 'control-train': cmd_control_train, 'analyze': cmd_analyze,
     'disable-engines': cmd_disable_engines, 'figures': cmd_figures}[args.cmd](args)


if __name__ == '__main__':
    main()
