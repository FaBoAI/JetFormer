#!/usr/bin/env python3
"""evaluation.json → evaluation.md(数字はすべて json から入れる。表は make_tables.py)

    cd ~/jetformer && python /home/jetson/jetformer_reports/sim_cotrain_20261013/eval/make_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_tables as mt  # noqa: E402

REPORT = mt.REPORT
CLS = mt.CLS


def f(x, nd=2, sign=False):
    return mt.f(x, nd, sign)


def load_review():
    """レビューの結果(eval/review/review_check.json)。無ければ None(レビューの前の文のまま作る)"""
    p = HERE / 'review' / 'review_check.json'
    if not p.exists() or p.stat().st_size < 3:
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def pct(x):
    return '%d' % int(round(100 * x))


def main():
    r = json.loads((REPORT / 'evaluation.json').read_text(encoding='utf-8'))
    RV = load_review()
    P = r['primary']
    V = r['verdict']
    L = P['L_us']
    am = P['arm_mean_us']
    vs = P['vs_A']
    sec = P['secondary']
    M = r['accuracy']['metrics']
    R = r['robustness']
    S = r['sim']
    G = r['gates']
    LT = r['latency']
    H = r['speed_habit']
    cc = r.get('crosscheck') or {}
    rep = r['reproduce_1012']

    def gmean(arm, g):
        return float(np.mean([M[n]['human_clean_genre%d' % g]['l1_us'] for n in mt.SEEDN[arm]]))

    def worse(arm):            # + = A より大きい(悪い向き)
        return -vs[arm]['best']['gain_us']

    up = '上がりました' if V['accuracy_went_up'] else '上がりませんでした'
    dirC = {'worse': '2 つの種 × best・last の 4 通りすべてで A の 2 本より大きく、悪い向きにそろっています',
            'better': '4 通りすべてで A の 2 本より小さく、良い向きにそろっています', None: '向きはそろっていません'}
    hab_max = max(abs(H[a][g]['shift_us']) for a in H for g in H[a])
    rn = {n: LT['per_run'][n]['run_numpy_median_ms'] for n in mt.RUNLAB}
    new_rn = [rn[n] for n in ('C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'AFT_s0', 'AFT_s1')]
    acc_max = [G['per_run'][n]['actions_max_us'] for n in ('C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'AFT_s0', 'AFT_s1')]
    fg = S['forgetting']
    r1, de = r['r1'], r['de']
    gate_all = all(G['per_arm'].values())
    collapse_any = any(r['collapse']['per_arm'][a]['collapse'] for a in mt.ARMS3)

    md = []
    w = md.append
    w('# シミュレーターと実機を混ぜた学習(案 C)の評価 — 実機の精度は上がったか')
    w('')
    w('> **呼び名**: モデルは **TransFuser軽量型(JetFormer v0)**(`jetformer_v0`。今の走行のモデル B と同じ形)。データは **実機**(`clean_driving_20261011`、'
      '人のきれいな高速・低速の周回 52 シーン、ジャンル 4・5)と **シミュレーター**(`jetformer-simulator-id8-20min_20261012-150956_b55616cf`、60 シーン、ジャンル 8)。')
    w('> 腕: **A** = 実機だけ(10/12 の v0 の種 0・1)、**C** = シミュレーターと実機を混ぜて学習、**C+FT** = C のあと実機だけで微調整(finetuning)、'
      '**A+FT** = A を同じ方法で微調整(対照)。')
    w('')
    w('作成 %s ・ 対象 `/home/jetson/jetformer`(作業ツリー、commit はしていない)・ 判定の規則 `protocol.json`(%s に固定、sha256 %s…、'
      '評価のときも同じ: %s)・ 道具 `tools/sim_cotrain_eval.py`(10/12 の `tools/policy_zoo.py` の関数をそのまま使う)'
      % (r['created_at'].replace('T', ' '), r['protocol']['written_at'].replace('T', ' '), r['protocol']['sha256'][:8],
         'はい' if r['protocol']['unchanged'] else '**いいえ**'))
    w('')
    w('---')
    w('')
    w('## 0. 結論')
    w('')
    w('**実機の精度は%s。** シミュレーターを混ぜて学習しても(C)、そのあと実機のデータで微調整しても(C+FT)、保留した実機の人の走行に対する誤差は、'
      '実機だけで学習した A と「同等」でした(先に決めた規則の 3 番)。走行のモデルはモデル B のままです。' % up)
    w('')
    w('| 腕 | 主指標 L1(2 つの種の平均、best.pth) | A との差 | 判定(best・last の両方) |')
    w('|---|---|---|---|')
    w('| A(実機だけ・基準) | %s µs | — | — |' % f(am['A']['best'], 2))
    for arm in mt.ARMS3:
        w('| %s | %s µs | %s µs(%s%%) | **%s** |' % (mt.LAB[arm], f(am[arm]['best'], 2), f(worse(arm), 2, True),
                                                f(vs[arm]['diff_pct'], 1, True), CLS[vs[arm]['final_class']]))
    w('| モデル B(参考) | %s µs | %s µs | — |' % (f(L['best']['B'], 2), f(L['best']['B'] - am['A']['best'], 2, True)))
    w('')
    w('(+ は A より誤差が大きい = 悪い向き。「良い」と言うには、2 つの種の両方が A の 2 本より小さく、かつ平均で %s µs(= 2 × A のばらつき %s µs)'
      'より大きく改善する必要がありました。)' % (f(P['threshold_us']['best'], 3), f(P['s_A_us']['best'], 3)))
    w('')
    note_dir = ('' if vs['C']['consistent_direction'] is None else
                '(ただし best と last は同じ種の重みで独立ではなく、実質は 2 つの種です。本当の差が無くても、'
                '2 つの種がそろって A の 2 本より上(悪い側)に来る確率は 1/6 あるので、強い証拠ではありません)')
    w('- **混ぜるだけ(C)**: A より平均 %s µs(%s%%)大きい誤差でした。%s%s。ただし差は閾値 %s µs より小さいので「同等」です。'
      % (f(worse('C'), 2), f(vs['C']['diff_pct'], 1), dirC[vs['C']['consistent_direction']], note_dir, f(P['threshold_us']['best'], 3)))
    w('- **混ぜてから微調整(C+FT)**: A より平均 %s µs 大きく、「同等」です。微調整で C から変わった量は %s µs(best)・%s µs(last)で、'
      '微調整は C の誤差をほとんど変えませんでした(C+FT − C も「同等」)。C+FT は 2 つの種の差が %s µs と大きく(%s)、種 1 だけ見ると A の範囲に入ります。'
      % (f(worse('C+FT'), 2), f(-sec['C+FT_vs_C']['best']['gain_us'], 3, True), f(-sec['C+FT_vs_C']['last']['gain_us'], 3, True),
         f(P['seed_range_us']['C+FT']['best'], 3), 'ばらつきが大きい: 2 × s_A を超える' if P['own_spread_large']['C+FT'] else '2 × s_A の内'))
    w('- **シミュレーターの効果(C+FT と A+FT の比較)**: 同じ微調整をした A+FT と比べても C+FT は %s µs 大きく、「同等」です。'
      'A+FT − A も %s µs で「同等」なので、12 エポックの追加の微調整そのものにも効果はありませんでした。'
      % (f(-sec['C+FT_vs_A+FT']['best']['gain_us'], 2), f(worse('A+FT'), 3, True)))
    if RV is None:
        w('- **速さの癖は移りませんでした**: シミュレーターは実機より速く走りますが(スロットルの中央値 1267 µs vs 1315 µs)、ジャンル別の前進の偏りの '
          'A からのずれは最大 %s µs で、目安の 10 µs より小さいままでした。ジャンル別の出力の頭が速さの違いを分けて持てています。' % f(hab_max, 1))
    else:
        hs = RV['breakdown']['cells']['g5_HS']
        fb = hs['fwd_bias_per_run']
        w('- **速さの癖は、ジャンルの平均では移りませんでした。ただし低速の直線では少し移っています**: シミュレーターは実機より速く走りますが'
          '(スロットルの中央値 1267 µs vs 1315 µs)、ジャンル別の前進の偏りの A からのずれは最大 %s µs で、目安の 10 µs より小さいままでした'
          '(先に決めた規則では「移りなし」)。ただし、レビューで場所別に見ると(後から見た内訳で、判定には使いません)、低速(5)の直線(場所の札 HS、%d フレーム)では'
          '前進の偏りが A %s / %s µs に対して C %s / %s µs、C+FT %s / %s µs で(負 = 人より強く前進)、C は両方の種、C+FT は種 1 で、'
          'A の種どうしの差(%s µs)と 10 µs の目安を超えて強く前進を出していました。ジャンル別の頭は速さの違いの大部分を分けて持てていますが、'
          'この直線だけは分けきれていません(9 章)。'
          % (f(hab_max, 1), hs['n'], f(fb['A_s0'], 1, True), f(fb['A_s1'], 1, True), f(fb['C_s0'], 1, True), f(fb['C_s1'], 1, True),
             f(fb['C+FT_s0'], 1, True), f(fb['C+FT_s1'], 1, True), f(hs['A_seed_gap_fwd'], 1)))
    lost_txt = ('シミュレーターで得た強さは失っていません' if RV is None else
                'C で得た強さ(A との差)のうち微調整で戻ったのは平均で約 %s%%(種 1 で大きい)で、大部分は残りました'
                % pct(RV['robust_sim']['robust']['ft_lost_share_of_C_gain']))
    w('- **ずれへの強さは上がり、微調整のあとも%s残りました**: 画像を数行ずらす・ToF を 1 行ずらすなどの小さなずれで誤差が増える量(ΔL1)は、'
      'C が A の %s 倍、C+FT が %s 倍で、どちらも先に決めた「ずれに強い」の条件(0.8 倍以下・両方の種で A 以下)を満たしました。A+FT は %s 倍で変わりません。'
      '微調整で ΔL1 が変わった量は種ごとに %s / %s µs で、%s。ただし、差は 0.3 µs ほどで、主指標の 49 µs に比べると小さな量です。'
      % ('' if RV is None else '大部分が', f(R['per_arm']['C']['ratio_to_ref'], 2), f(R['per_arm']['C+FT']['ratio_to_ref'], 2),
         f(R['per_arm']['A+FT']['ratio_to_ref'], 2),
         f(R['ft_change']['CFT_minus_C_dL1_us'][0], 2, True), f(R['ft_change']['CFT_minus_C_dL1_us'][1], 2, True), lost_txt))
    w('- **シミュレーターのことは覚えていて、微調整でもほぼ忘れませんでした**: シミュレーターの 12 シーン(C では学習に入ったシーン)で、C はジャンル 8 の頭で '
      '%s µs、C+FT は %s µs でした(%s µs 増えただけ)。実機だけの A は、頭 5 で %s µs(頭 4 で %s µs)と大きく外れます。'
      'C が A より良くなった %s µs のうち、微調整で失ったのは %s%% だけです。'
      % (f(fg['C_h8_us'], 1), f(fg['CFT_h8_us'], 1), f(fg['vs_A_h5']['increase_us'], 1, True), f(fg['A_h5_us'], 1), f(fg['A_h4_us'], 1),
         f(fg['vs_A_h5']['gain_c_us'], 0), f(100 * fg['vs_A_h5']['share_lost'], 1)))
    w('- **難しい場所(R1・DE)と崩れ**: どの腕も R1・DE で「良い」にはなりませんでした。崩れ(直線で前進が弱まる・向きを間違える)は%s。'
      'DE で C+FT − A が %s µs(95%% の区間が 0 をわずかに外れて「悪い」の印)でしたが、同じ作り方の種どうしでも DE で 2.8 µs、R1 で 7.7 µs の差が出るので(10/12 の陰性対照)、'
      'それとほぼ同じ大きさです。参考の項目で、判定には使いません(3 章の表 5・6)。'
      % ('どの腕にもありません' if not collapse_any else '**ある腕があります**', f(de['per_arm']['C+FT']['d_mean'], 1, True)))
    w('- **速さ**: 6 本とも TensorRT FP16 のエンジンで走行の速さのゲート(6 ms)の内です。run_numpy の中央値は %s〜%s ms で、A(%s・%s ms、10/12 に測定)と同じです。'
      'エンジンと PyTorch の差は最大 %s〜%s µs(ゲート 2.5 µs)。' % (f(min(new_rn), 2), f(max(new_rn), 2), f(rn['A_s0'], 2), f(rn['A_s1'], 2),
                                                        f(min(acc_max), 2), f(max(acc_max), 2)))
    w('')
    w('**まとめ**: この条件では、「シミュレーターのデータで先に学び、実機のデータで仕上げる」やり方で、実機の精度は上がりませんでした。'
      '平均では 0.4 µs(1%% 弱)悪い向きでしたが、区別できない幅の内です。得られたのは、小さな入力のずれへの強さと、シミュレーターの場面への当てはまりです。'
      'どちらも微調整のあとも%s残りました。走行の推奨はモデル B のままで、実車の A/B/A で試す候補もありません(「良い」の腕が無いため)。' % (
          '' if RV is None else '大部分が'))
    if RV is not None:
        w('')
        w('**レビュー**: 保留データの漏れ・比べ方の公平さ・ドロップの表・数字の再計算を CPU だけで確かめ、判定と結論を覆すものはありませんでした。'
          '差は場面で一様ではなく、2 つのシーン(とくに低速の直線)で大きく悪く、ほかではわずかに小さい向きでした(後から見た内訳。9 章)。')
    w('')
    w('この結論は、TransFuser軽量型(JetFormer v0)・この学習の条件(A と同じレシピで調整なし。混ぜる割合はシミュレーター 2 : 実機 1)・'
      'このシミュレーターのデータ(60 シーン・合成の先生)に限ります。机上の結論で、実車の走りを測ったものではありません。')
    w('')
    w('---')
    w('')
    w('## 1. 比べたもの')
    w('')
    w('| 腕 | 学習 | 学習結果(`data/runs/`) | 予測 |')
    w('|---|---|---|---|')
    w('| A(基準) | 実機だけ・30 エポック・lr 3e-4(10/12 の policy zoo) | `%s`・`%s` | 10/12 の予測をそのまま読む |' % (r['names']['A_s0']['run'], r['names']['A_s1']['run']))
    w('| モデル B(別の行) | 実機だけ(今の走行) | `%s` | 10/12 の予測をそのまま読む |' % r['names']['B']['run'])
    w('| C | 実機 + シミュレーター(学習 17,732 サンプル、うちシミュレーター 11,820)・30 エポック・lr 3e-4・ジャンル数 9 | `%s`・`%s` | この評価 |'
      % (r['names']['C_s0']['run'], r['names']['C_s1']['run']))
    w('| C+FT | 同じ種の C の best.pth から、実機だけ(A と同じ 5,912 サンプル)・12 エポック・lr 1e-4 | `%s`・`%s` | この評価 |'
      % (r['names']['CFT_s0']['run'], r['names']['CFT_s1']['run']))
    w('| A+FT(対照) | 同じ種の A の best.pth から、C+FT と同じ微調整 | `%s`・`%s` | この評価 |' % (r['names']['AFT_s0']['run'], r['names']['AFT_s1']['run']))
    w('')
    w('学習の記録は `training.md`。検証(最良のエポックを選ぶ)はどの腕も実機の 11 シーンだけです。')
    w('')
    w('## 2. 手順(学習の前に固定した protocol.json のとおり)')
    w('')
    w('- **フレーム**: 10/12 と同じ保留の表(`rl_evaluate.pred_table()`、%s フレーム、key `%s`)。主指標は人の 11 シーンのきれいな走行(未来が有効・車両の値が有効・'
      '2.5G の衝突の前後でない)%s フレーム(ジャンル 4: %s、ジャンル 5: %s)。' % (
          f'{r["frames"]["pred_table"]:,}', r['frames']['pred_table_key'][:8], f'{r["frames"]["human_clean"]:,}',
          f'{r["frames"]["human_clean_genre4"]:,}', f'{r["frames"]["human_clean_genre5"]:,}'))
    w('- **予測**: PyTorch FP32(best.pth・last.pth)。ジャンルは aligned の mode(4・5)。10/12 と同じ `policy_zoo._Frames`・`_predict_frames`。'
      '確かめ: 10/12 の v0 種 0 をこの評価の道具で予測し直すと、10,985 フレームすべてで 10/12 の予測と差 %s µs(L1 %s / %s µs)。'
      % (f(cc.get('max_abs_us'), 1), cc.get('L_new_us'), cc.get('L_1012_us')))
    w('- **主指標 L**: |予測 − 人|(10 ステップ × [ステア, スロットル])の平均 [µs]。腕の値 = 2 つの種の平均。')
    w('- **ばらつき s_A** = {A 種 0, A 種 1, モデル B} の L の範囲 = %s µs(best)・%s µs(last)。'
      '**良い** = 腕の 2 本が A の 2 本より小さく、かつ平均の改善 > 2 × s_A(%s µs)。**悪い** = その逆。ほかは **同等**。'
      'last.pth でも同じ計算(閾値 %s µs)をして、best だけ「良い」なら「同等」に下げる。|差| ≥ 10 µs なら「実用上の差」。'
      % (f(P['s_A_us']['best'], 3), f(P['s_A_us']['last'], 3), f(P['threshold_us']['best'], 3), f(P['threshold_us']['last'], 3)))
    w('- **腕どうし(副)**: C+FT vs C、C+FT vs A+FT を同じ規則で(ばらつき = 基準の腕の種の範囲と s_A の大きい方)。')
    w('- **副指標(判定は変えない)**: ジャンル別・速さの癖・R1/DE(10/12 と同じ通過の選び直し 10,000 回)・崩れ・ずれへの強さ・交代の後の人の修正(参考)・'
      'シミュレーターの L1(参考)・エンジンの速さ。')
    w('- **10/12 と同じ値か**: A・モデル B の値を 10/12 の予測から同じ道具で計算し直し、protocol.json に写した 10/12 の値と比べた → **%s**(3 章の表 13)。'
      % ('すべて一致' if rep['all_match'] else '一致しない項目あり'))
    w('')
    w('## 3. 結果の表')
    w('')
    w('数字はすべて `evaluation.json` から `eval/make_tables.py` で作りました(手で写していません)。µs は PWM のパルス幅の差です。')
    w('')
    titles = [('表 1 主指標(保留した実機の人のきれいな走行の L1、4,670 フレーム、PyTorch FP32)', mt.t_primary),
              ('表 2 腕どうしの比較(同じ規則。基準の腕を変えたもの)', mt.t_secondary),
              ('表 3 ジャンル別(best.pth。k3 = 3 フレーム先の誤差 [µs]、前進の偏り = 3 フレーム先のスロットル − 人。負 = 人より強く前進)', mt.t_genre),
              ('表 4 速さの癖の移り(ジャンル別の前進の偏り、種の平均)', mt.t_habit),
              ('表 5 R1(進入前 0.4 秒のスロットル。+ = 遅い)、%d 通過(参考)' % r1['passes'], lambda x: mt.t_pass(x, 'r1')),
              ('表 6 DE(緑マット出口〜0.8 秒のステア。+ = 左)、%d 通過(参考)' % de['passes'], lambda x: mt.t_pass(x, 'de')),
              ('表 7 崩れ(点推定、種の平均どうし)', mt.t_collapse),
              ('表 8 ずれへの強さ(ΔL1 = ずれありの L1 − ずれなしの L1 [µs]、%s フレーム、best.pth)' % f'{R["per_run"]["A_s0"]["frames"]:,}', mt.t_robust),
              ('表 9 シミュレーターの L1(参考。元のシミュレーターのデータセットの val の 12 シーン、%s フレーム。L1 [µs])' % f'{S["frames"]:,}', mt.t_sim),
              ('表 10 交代の後の人の修正(intervention_val、L1 [µs]。参考。学習データの中のログを含む)', mt.t_iv),
              ('表 11 ゲート・速さ(新しい 6 本は学習の段のエンジンの確認、A・B は 10/12 の値)', mt.t_gate),
              ('表 12 学習(学習中の検証。主指標ではない。`training.md`)', mt.t_train),
              ('表 13 10/12 と同じ値になるか(A・モデル B)', mt.t_repro)]
    for t, fn in titles:
        w('### ' + t)
        w('')
        w(fn(r))
        w('')
    w('表 1 の注: C+FT の 2 つの種の範囲 %s µs は 2 × s_A(%s µs)を超えます(protocol の「ばらつきが大きい」)。C・A+FT は超えません。'
      % (f(P['seed_range_us']['C+FT']['best'], 3), f(P['threshold_us']['best'], 3)))
    w('')
    w('表 2 の注: 閾値は「基準の腕の種の範囲 × 2」と「2 × s_A」の大きい方です(C の範囲 %s µs → %s µs、A+FT の範囲 %s µs → %s µs)。'
      % (f(P['seed_range_us']['C']['best'], 3), f(sec['C+FT_vs_C']['best']['threshold_us'], 3), f(P['seed_range_us']['A+FT']['best'], 3),
         f(sec['C+FT_vs_A+FT']['best']['threshold_us'], 3)))
    w('')
    w('表 5・6 の注: 「良い」には、99.375%% の区間の下限 > 0・両方の種で > 0・4 組すべて ≥ 10 µs・差の差の区間の下限 > 0 の全部が要ります。'
      '「悪い」は 95%% の区間の上限 < 0。10/12 の陰性対照では、同じ作り方の種どうしでも R1 で %s µs・DE で %s µs の差が出て、R1 の陰性対照 B は「悪い」になりました。'
      'この大きさ以下の差は「ばらつきの内」と読みます。' % (f(abs(r1['nc_1012']['NC-A']['d_mean']), 1), f(abs(de['nc_1012']['NC-A']['d_mean']), 1)))
    w('')
    w('## 4. 判定(protocol.json の conclusion_rules を順に当てた結果)')
    w('')
    w('1. C+FT vs A: **%s**(%s µs、閾値 %s µs、実用上の差 %s)。C vs A: **%s**(%s µs、%s)。'
      % (CLS[V['C+FT_vs_A']], f(worse('C+FT'), 3, True), f(P['threshold_us']['best'], 3), 'あり' if vs['C+FT']['practical'] else 'なし',
         CLS[V['C_vs_A']], f(worse('C'), 3, True), dirC[vs['C']['consistent_direction']]))
    w('2. C+FT が「良い」ではないので、C+FT vs A+FT(**%s**)はシミュレーターの効果の判定に使いません。' % CLS[V['C+FT_vs_A+FT']])
    if V['conclusion']['key'] == 'no_gain':
        w('3. C+FT・C のどちらも「同等」→ **「シミュレーターを混ぜても実機の保留データの精度は上がらなかった(A と同じ)」**。')
    else:
        w('3. ' + V['conclusion']['text'])
    w('4. 走行の推奨を変える条件(「良い」かつ 10 µs 以上かつ崩れなし・ゲート合格)を満たす腕はありません → **推奨は %s**。'
      'ゲートは全腕 %s、崩れは %s。' % (V['recommended'], '合格' if gate_all else '**不合格あり**', 'なし' if not collapse_any else '**あり**'))
    w('5. 机上の結論で、実車の改善を示すものではありません。TransFuser軽量型(JetFormer v0)・この学習の条件・このシミュレーターのデータに限ります。')
    w('')
    w('## 5. 微調整で、シミュレーターで得たものを失ったか')
    w('')
    w('| 見るもの | C | C+FT | A(実機だけ) | 読み方 |')
    w('|---|---|---|---|---|')
    w('| ずれへの強さ(平均 ΔL1、種の平均)[µs] | %s(A の %s 倍) | %s(%s 倍) | %s | どちらも「強い」。微調整の前後の差は種ごとに %s / %s µs%s |' % (
        f(R['per_arm']['C']['dL1_mean_us'], 3), f(R['per_arm']['C']['ratio_to_ref'], 2), f(R['per_arm']['C+FT']['dL1_mean_us'], 3),
        f(R['per_arm']['C+FT']['ratio_to_ref'], 2), f(R['ref_dL1_us'], 3), f(R['ft_change']['CFT_minus_C_dL1_us'][0], 3, True),
        f(R['ft_change']['CFT_minus_C_dL1_us'][1], 3, True),
        '' if RV is None else '(C の得の約 %s%% が戻った)' % pct(RV['robust_sim']['robust']['ft_lost_share_of_C_gain'])))
    if RV is None:
        read_tof = '強さの多くは ToF のずれと画像の上ずれから'
    else:
        cg = RV['robust_sim']['robust']['contribution_to_gain']
        pname = {'img_down3': '画像の下ずれ', 'img_up3': '画像の上ずれ', 'bright_0.8': '明るさ ×0.8', 'bright_1.2': '明るさ ×1.2',
                 'tof_down1': 'ToF のずれ'}
        both = {a: [k for k, v in cg[a].items() if v['both_seeds_below_both_A']] for a in ('C', 'C+FT')}
        read_tof = ('A との差への寄与が最も大きいのは画像の下ずれ(C %s%%・C+FT %s%%)。ただし A 種 1 の値が大きいためで、'
                    '両方の種でそろって A の 2 本より小さいのは、C は%s、C+FT は%s'
                    % (pct(cg['C']['img_down3']['share']), pct(cg['C+FT']['img_down3']['share']),
                       '・'.join(pname[k] for k in both['C']) or 'なし', '・'.join(pname[k] for k in both['C+FT']) or 'なし'))
    w('| ToF を 1 行ずらしたときの ΔL1(種の平均)[µs] | %s | %s | %s | %s |' % (
        f(R['per_arm']['C']['per_pert_mean_us']['tof_down1'], 2, True), f(R['per_arm']['C+FT']['per_pert_mean_us']['tof_down1'], 2, True),
        f(R['per_arm']['A']['per_pert_mean_us']['tof_down1'], 2, True), read_tof))
    w('| シミュレーターの L1、ジャンル 8 の頭 [µs] | %s | %s | 頭 5: %s(頭 4: %s)| C+FT は %s µs 増えた(種ごと %s / %s)。C の得 %s µs の %s%% |' % (
        f(fg['C_h8_us'], 1), f(fg['CFT_h8_us'], 1), f(fg['A_h5_us'], 1), f(fg['A_h4_us'], 1), f(fg['vs_A_h5']['increase_us'], 1, True),
        f(fg['paired_CFT_minus_C_h8_us'][0], 1, True), f(fg['paired_CFT_minus_C_h8_us'][1], 1, True), f(fg['vs_A_h5']['gain_c_us'], 0),
        f(100 * fg['vs_A_h5']['share_lost'], 1)))
    w('| シミュレーターの L1、実機の頭(4・5 の平均)[µs] | %s | %s | %s | 実機の頭でも A よりシミュレーターに近い |' % (
        f((S['per_arm']['C']['h4']['l1_mean_us'] + S['per_arm']['C']['h5']['l1_mean_us']) / 2, 1),
        f((S['per_arm']['C+FT']['h4']['l1_mean_us'] + S['per_arm']['C+FT']['h5']['l1_mean_us']) / 2, 1),
        f((S['per_arm']['A']['h4']['l1_mean_us'] + S['per_arm']['A']['h5']['l1_mean_us']) / 2, 1)))
    w('')
    if RV is not None:
        wc = RV['weight_change']
        w('**答え: 大部分は失っていません。** ずれへの強さは C+FT でも「強い」のままで(C で得た分の約 %s%% は微調整で戻った)、'
          'シミュレーターの場面への当てはまりも少し落ちただけ(ジャンル 8 で %s → %s µs)です。'
          '微調整で、学習する重みは %s〜%s%%、BatchNorm の統計は約 %s%% 動きましたが、ジャンル 8 の埋め込みの列は動いていません(実機のフレームだけで学ぶので'
          'その列には勾配が来ない)。頭 8 で測るシミュレーターの L1 がほぼ残るのは、この仕組みからも自然です。'
          % (pct(RV['robust_sim']['robust']['ft_lost_share_of_C_gain']), f(fg['C_h8_us'], 1), f(fg['CFT_h8_us'], 1),
             f(100 * min(wc['C_s0->C+FT']['params'], wc['C_s1->C+FT']['params']), 1),
             f(100 * max(wc['C_s0->C+FT']['params'], wc['C_s1->C+FT']['params']), 1),
             pct((wc['C_s0->C+FT']['bn_stats'] + wc['C_s1->C+FT']['bn_stats']) / 2)))
    w(('' if RV is not None else '**答え: 失っていません。** ずれへの強さは C+FT でも「強い」のままで、シミュレーターの場面への当てはまりも少し落ちただけ(ジャンル 8 で %s → %s µs)です。'
       % (f(fg['C_h8_us'], 1), f(fg['CFT_h8_us'], 1))) +
      'ただし、シミュレーターの 60 シーンは C の学習にすべて入っている(保留ではない)ので、この L1 は「覚えているか」の目安で、新しいシミュレーターの場面への強さではありません。'
      'A をシミュレーターの画像で動かすと、前進の偏りが +%s µs(遅く出す向き)と大きく、画像の見た目の違い(暗い床・黒い背景・バンパーが写らない)で実機の学習だけでは当たりません。'
      % f(S['per_arm']['A']['h5']['fwd_bias_mean_us'], 0))
    w('')
    w('## 6. 図')
    w('')
    for fn, cap in (('eval_primary.png', '主指標。● best.pth・○ last.pth、縦の棒は 2 つの種の平均、帯は A の平均 ± 0.626 µs(区別できない幅)'),
                    ('eval_genre.png', 'ジャンル別の L1(上)と前進の偏り(下、帯は A ± 10 µs)'),
                    ('eval_sim_l1.png', 'シミュレーターの 12 シーンの L1(参考)。頭(ジャンル)ごと'),
                    ('eval_robust.png', 'ずれへの強さ。ずれごとの ΔL1(棒は種の平均、● は種)'),
                    ('eval_r1_de.png', 'R1・DE の差の平均と区間(参考)')):
        w('- `fig/%s` — %s' % (fn, cap))
    w('')
    w('学習の段の図(`fig/train_*.png`)は `training.md` にあります。')
    w('')
    w('## 7. 注意と限界')
    w('')
    w('- 主指標の 11 シーンは、どの腕でも最良のエポックを選んだ検証のシーンと同じです(10/12 と同じ条件)。どの腕にも同じ形で入る甘さなので、'
      'last.pth でも同じ判定をしました(結果は同じく「同等」)。')
    w('- C+FT の 2 つの種の差(%s µs)は 2 × s_A を超えました。種を増やせば平均は動きえますが、どちらの種も A の小さい方(種 0、%s µs)より大きかった'
      '(C+FT の小さい方は種 1 の %s µs)ので、「良い」になる形ではありません。' % (f(P['seed_range_us']['C+FT']['best'], 3), f(L['best']['A_s0'], 3),
                                                       f(min(L['best']['CFT_s0'], L['best']['CFT_s1']), 3)))
    w('- A+FT 種 0 は電源断のあと 2 エポック目から再開しました(`training.md` §5)。重み・最適化・スケジューラは last.pth から戻しましたが、'
      '並べ替えと色の揺らしの順番は途切れなかった場合と違います。')
    w('- シミュレーターの L1 は、C・C+FT では学習に入ったシーンで測った値です(指示のとおりシミュレーターは全部学習に使ったので保留が無い)。'
      'A・A+FT・B はジャンル数 6 で頭 8 が無いので、頭 5(低速。指示のとおり)と頭 4(高速)で予測しました。')
    w('- R1・DE・交代の後の人の修正は参考です。R1・DE の「悪い」の印(DE の C+FT − A %s µs、C+FT − A+FT %s µs、R1 の C+FT − C %s µs)は、'
      '主指標の判定にも推奨にも使いません。C+FT − A+FT の DE の差の多くは、A+FT が A より左に %s µs 動いたことによるものです。'
      % (f(de['per_arm']['C+FT']['d_mean'], 1, True), f(de['CFT_vs_AFT']['d_mean'], 1, True), f(r1['CFT_vs_C']['d_mean'], 1, True),
         f(de['per_arm']['A+FT']['d_mean'], 1)))
    w('- 速さは、新しい 6 本は今日の学習の段、A・B は 10/12 に同じ道具で測った値です。測り直すと 0.15 ms ほど揺れます。走行ループ・カメラ → PWM は見積もりで、実測ではありません。')
    w('- C・C+FT のエンジンはジャンル数 9 で、走行タブではジャンル 8(シミュレーター)も選べます。実車ではジャンル 4・5 だけを使ってください。'
      'どの新しい学習結果も机上の比較用で、走行の推奨はモデル B のままです。')
    w('- 混ぜる割合(シミュレーター 2 : 実機 1)・学習率・エポック数は調整していません(protocol のとおり A と同じレシピ)。'
      '上がらなかった理由は確かめていません。考えられるのは、実機のデータが同じコースをすでに十分に覆っていること、シミュレーターの画像の見た目が実機と大きく違うこと、'
      '合成の先生の操作が人と違うこと、です(どれも仮説)。')
    w('')
    w('## 8. ファイルと再現の仕方')
    w('')
    w('| もの | 場所 |')
    w('|---|---|')
    w('| 評価の結果(全部の数字) | `evaluation.json`(同じものが `/home/jetson/jetformer/data/zoo/sim_cotrain_20261013/evaluation.json`) |')
    w('| 表 | `eval/tables.md`(`eval/make_tables.py`)、この文書は `eval/make_report.py` |')
    w('| 予測(新しい 6 本の best・last・ずれ、全 9 本のシミュレーター) | `/home/jetson/jetformer/data/zoo/sim_cotrain_20261013/{pred,pert,sim}/`、確かめ `crosscheck.json` |')
    w('| A・モデル B の予測(読むだけ・書き換えていない) | `/home/jetson/jetformer/data/zoo/policy_zoo_20261012/{pred,pert}/` |')
    w('| 実行の記録 | `eval/logs/predict.log`(GPU: %s〜%s)、`eval/logs/evaluate.log` |' % ('14:45', '15:06'))
    w('| コード | `tools/sim_cotrain_eval.py`(predict・crosscheck・evaluate・figures)、`tools/sim_cotrain_figures.py`、'
      '`tests/test_sim_cotrain_eval.py`(CPU のテスト)、`tools/rl_policies.py` の GPU_PATTERNS に 2 つ追加 |')
    w('')
    w('```')
    w('cd ~/jetformer')
    w('nice -n 10 python -u tools/sim_cotrain_eval.py crosscheck       # GPU(1 プロセス)。済みなら飛ばす')
    w('nice -n 10 python -u tools/sim_cotrain_eval.py predict          # GPU(1 プロセス)。出来た npz は飛ばす')
    w('CUDA_VISIBLE_DEVICES= python tools/sim_cotrain_eval.py evaluate # CPU → evaluation.json')
    w('CUDA_VISIBLE_DEVICES= python tools/sim_cotrain_eval.py figures  # CPU → fig/eval_*.png')
    w('python /home/jetson/jetformer_reports/sim_cotrain_20261013/eval/make_report.py')
    w('```')
    w('')
    if RV is not None:
        md.extend(review_section(RV, r))
    text = '\n'.join(md) + '\n'
    (REPORT / 'evaluation.md').write_text(text, encoding='utf-8')
    (REPORT / 'eval' / 'tables.md').write_text('\n\n'.join('## %s\n\n%s' % (k, fn(r)) for k, fn in mt.ALL) + '\n', encoding='utf-8')
    print(REPORT / 'evaluation.md', len(text))


def review_section(RV, r):
    """9 章「レビューで確かめた点」(数字は eval/review/review_check.json から)"""
    lk, sc_, rc_, dr, rcp = RV['leak'], RV['sim_content'], RV['real_content'], RV['drop'], RV['recompute']
    rob, sim, bd, wc, cp = RV['robust_sim']['robust'], RV['robust_sim']['sim'], RV['breakdown'], RV['weight_change'], \
        RV.get('cpu_predict_max_abs_us', {})
    E = r['primary']
    am = rcp['arm_mean']
    ok_files = all(v for k, v in RV['files'].items() if k != 'protocol_sha256')
    rng = rcp['seed_range_best']
    sig = float(np.mean([rng[a] for a in ('A', 'C', 'C+FT', 'A+FT')])) / 1.128      # 2 本の範囲の期待値 = 1.128 σ
    d_cft = am['C+FT']['best'] - am['A']['best']
    ub = -d_cft + 1.96 * sig                                                          # 改善の 95% の上限の目安
    w2 = bd['two_worst_scenes_for_C']
    ps = bd['per_scene']
    hs = bd['cells']['g5_HS']
    fb = hs['fwd_bias_per_run']
    aft = [v['A+FT-A'] for v in ps.values()]
    other_max = max(max(abs(v['fwd_shift_C']), abs(v['fwd_shift_CFT'])) for k, v in bd['cells'].items() if k != 'g5_HS')
    cp_vals = [v for k, v in cp.items() if not k.startswith('control')]
    mism = rcp['mismatch_vs_evaluation_json']
    o = []
    w = o.append
    w('## 9. レビューで確かめた点')
    w('')
    w('この評価を崩せないかを、CPU だけで確かめました(学習と GPU の予測のやり直しはしていません)。道具は `eval/review/review_check.py`、'
      '結果は `eval/review/review_check.json`(段ごとに `eval/review/review_parts/`)です。')
    w('**判定(5 つの比較すべて「同等」)と結論(実機の精度は上がらなかった・推奨はモデル B のまま)を覆すものはありませんでした。** '
      '直したのは本文の言い方 4 か所です(9.7)。主指標・判定・閾値の数字は変えていません。')
    w('')
    w('### 9.1 保留データの漏れ — ありませんでした')
    w('')
    w('- 主指標の 11 シーン(`heldout.json` の human_clean_scenes)は `clean_driving_20261011` の val と同じ: %s。混ぜたデータセットの val とも順番まで同じ: %s。'
      'C の学習の %d シーン(実機 %d・シミュレーター %d)には入っていません(重なり %d シーン)。'
      % ('はい' if lk['heldout_json_human_clean_equals_clean_val'] else '**いいえ**', 'はい' if lk['mix_val_equals_A_val_in_order'] else '**いいえ**',
         lk['n_real_train'] + lk['n_sim_train'], lk['n_real_train'], lk['n_sim_train'], len(lk['mix_train_in_heldout_scenes'])))
    w('- 保留の 11 シーンの全 %s フレームと C の学習の全 %s フレームで、時刻の重なり %d・画像のファイル名の重なり %d。'
      % ('{:,}'.format(lk['held_frames_all']), '{:,}'.format(lk['C_train_frames_all']), lk['timestamp_overlap_heldout_vs_C_train'],
         lk['cam_basename_overlap_heldout_vs_C_train']))
    w('- 混ぜたデータセットの実機の %d シーンは `clean_driving_20261011` と aligned が全部同じ(違い %d 件、画像 %d 枚の md5 も同じ)。'
      'シミュレーターの %d シーンも元のデータセットと同じ(違い %d 件、画像 %d 枚)で、ジャンルは %s だけ、実機の録画の時間帯に入る時刻は %d。'
      % (rc_['real_scenes'], len(rc_['mismatches']), rc_['images_compared'], sc_['scenes'], sc_['array_or_image_mismatches'],
         sc_['images_compared'], '・'.join(str(x) for x in sc_['sim_modes']), sc_['sim_frames_inside_real_time_range']))
    w('- 評価に使う `ai_mix_20261011` の aligned と `clean_driving_20261011` の aligned の違いは、future_valid の 2 フレーム'
      '(20261010-193638-0003・20261010-202633-0007 の 0 フレーム目)だけです。どちらも vehicle_valid が偽なので、主指標にも学習中の検証にも入りません。')
    w('- 同じ録画(ログ)の別のシーンが学習と検証に分かれているのは、検証の %d ログすべてで同じです。これは A・C・C+FT・A+FT に同じ形で入るので、'
      '腕の差にはなりません。' % lk['val_recording_logs'])
    w('')
    w('### 9.2 比べ方 — 公平でした')
    w('')
    w('- 実機の学習サンプルは A・C で同じ %s(並びも同じ: %s)。C はそれにシミュレーター %s を足しただけです。30 エポックで見る実機のサンプルは'
      ' A・C とも %s で、C の更新回数(%s 回)が A(%s 回)の約 3 倍なのは、足したシミュレーターの分です。'
      % ('{:,}'.format(dr['FT_train']), 'はい' if dr['C_real_part_same_as_A'] else '**いいえ**', '{:,}'.format(dr['C_sim_samples']),
         '{:,}'.format(dr['real_samples_seen']['A']), '{:,}'.format(dr['optimizer_steps_per_run']['C']),
         '{:,}'.format(dr['optimizer_steps_per_run']['A'])))
    w('- best のエポックを選ぶ検証は、全部の腕で同じ %s サンプル(実機の 11 シーン)です: %s。シミュレーターは検証に入っていません。'
      % ('{:,}'.format(dr['val_samples']), 'はい' if dr['val_same_C_FT_A'] else '**いいえ**'))
    w('- 学習率・バッチ・間引き・種・エポック数・衝突の除外は protocol のとおりでした(各 result.json)。C+FT の初期値は同じ種の C の best.pth、'
      'A+FT は同じ種の A の best.pth です(学習の記録の sha1 と今のファイルが一致: %s)。'
      % ('はい' if all(v['sha1_match'] and list(v.values())[1] for v in dr['inits'].values()) else '**いいえ**'))
    w('- protocol.json は 11:57:51 に固定し、最初の学習は 12:00:31 でした。固定の前の版(11:52)との違いは学習結果の名前だけです。'
      'sha256 は今も %s…、protocol に書いたファイル(混ぜたデータセットの splits・ドロップの表・10/12 の zoo のファイルなど)の sha256 も%s。'
      % (RV['files']['protocol_sha256'][:8], 'すべて同じ' if ok_files else '**違うものがあります**'))
    w('- 微調整の腕は、同じ検証のシーンで best を 2 回選んでいます(A → A+FT、C → C+FT)。この甘さは微調整の腕に有利な向きですが、'
      'それでも C+FT は「良い」にならず、選び方の無い last.pth でも同じ判定でした。')
    w('')
    w('### 9.3 ドロップの表 — シミュレーターのフレームをちょうど全部外していました')
    w('')
    w('- ドロップの表の True は %s 行で、シミュレーターの %s フレーム(60 シーン全部)とちょうど同じです(%s)。実機のフレームは 1 つも外していません。'
      % ('{:,}'.format(dr['drop_true']), '{:,}'.format(dr['sim_frames_in_mix']), 'はい' if dr['drop_equals_sim_rows'] else '**いいえ**'))
    w('- C+FT の学習サンプルは %s で、A と並びまで同じ(%s)、シミュレーターは %d。学習の記録でも drop で外したのは %s(有効なシミュレーターのフレーム全部)です。'
      % ('{:,}'.format(dr['FT_train']), 'はい' if dr['FT_train_same_as_A_incl_order'] else '**いいえ**', dr['FT_sim_samples'],
         '{:,}'.format(dr['FT_dropped_before_stride'])))
    w('')
    w('### 9.4 数字の再計算 — evaluation.json と一致しました')
    w('')
    w('評価の道具(`sim_cotrain_eval.evaluate_all`)を使わず、保存された予測と aligned から計算し直しました。')
    w('')
    w('| 項目 | 再計算 | evaluation.json |')
    w('|---|---|---|')
    w('| 主指標のフレーム(ジャンル 4 / 5) | %s(%s / %s) | %s(%s / %s) |' % (
        '{:,}'.format(rcp['frames']['clean']), '{:,}'.format(rcp['frames']['g4']), '{:,}'.format(rcp['frames']['g5']),
        '{:,}'.format(r['frames']['human_clean']), '{:,}'.format(r['frames']['human_clean_genre4']), '{:,}'.format(r['frames']['human_clean_genre5'])))
    w('| L(9 本 × best・last の 18 個) | 合わない数 %d | — |' % len([m for m in mism if m[0] == 'L']))
    w('| 腕の平均 L̄(best): A / C / C+FT / A+FT [µs] | %s / %s / %s / %s | %s / %s / %s / %s |' % tuple(
        [f(am[a]['best'], 3) for a in ('A', 'C', 'C+FT', 'A+FT')] + [f(E['arm_mean_us'][a]['best'], 3) for a in ('A', 'C', 'C+FT', 'A+FT')]))
    w('| s_A(best / last)[µs] | %s / %s | %s / %s |' % (f(rcp['s_A_recomputed']['best'], 3), f(rcp['s_A_recomputed']['last'], 3),
                                                       f(E['s_A_us']['best'], 3), f(E['s_A_us']['last'], 3)))
    w('| 判定 C・C+FT・A+FT vs A、C+FT vs C、C+FT vs A+FT | %s | %s |' % (
        '・'.join(CLS[rcp['verdicts'][k]['final']] for k in ('C_vs_A', 'C+FT_vs_A', 'A+FT_vs_A', 'C+FT_vs_C', 'C+FT_vs_A+FT')),
        '・'.join(CLS[r['verdict'][k]] for k in ('C_vs_A', 'C+FT_vs_A', 'A+FT_vs_A', 'C+FT_vs_C', 'C+FT_vs_A+FT'))))
    w('| ずれへの強さ A との比 C / C+FT / A+FT | %s / %s / %s | %s / %s / %s |' % (
        f(rob['ratio_to_A']['C'], 3), f(rob['ratio_to_A']['C+FT'], 3), f(rob['ratio_to_A']['A+FT'], 3),
        f(rob['eval_json_ratio']['C'], 3), f(rob['eval_json_ratio']['C+FT'], 3), f(rob['eval_json_ratio']['A+FT'], 3)))
    w('| シミュレーターの L1: C 頭 8 / C+FT 頭 8 / A 頭 5 [µs](%s フレーム) | %s / %s / %s | %s / %s / %s |' % (
        '{:,}'.format(sim['frames']), f(sim['C_h8'], 3), f(sim['C+FT_h8'], 3), f(sim['A_h5'], 3),
        f(r['sim']['forgetting']['C_h8_us'], 3), f(r['sim']['forgetting']['CFT_h8_us'], 3), f(r['sim']['forgetting']['A_h5_us'], 3)))
    w('')
    if cp_vals:
        w('- 新しい 6 本の best・last(12 個)の重みを CPU で読み直し、24 フレームを予測し直すと、保存された予測との差は最大 %s〜%s µs でした'
          '(GPU と CPU の計算の差の大きさ)。対照として C+FT 種 0 の重みと C 種 0 の予測を比べると %s µs 違うので、保存された予測は正しい重みから出ています。'
          % (f(min(cp_vals), 2), f(max(cp_vals), 2), f(cp.get('control_CFT_s0_weights_vs_C_s0_pred', float('nan')), 1)))
    w('- 微調整で重みは動いています。学習する重みの相対変化は C → C+FT が %s%%・%s%%、A → A+FT が %s%%・%s%%。BatchNorm の統計は C → C+FT が約 %s%%'
      '(A → A+FT は約 %s%%)で、シミュレーターの画像が 2/3 の混ざったバッチで作られた C の統計が、実機だけの統計に置き換わりました。'
      'ジャンル 8 の埋め込みの列は動いていません(相対変化 %s)。'
      % (f(100 * wc['C_s0->C+FT']['params'], 1), f(100 * wc['C_s1->C+FT']['params'], 1), f(100 * wc['A_s0->A+FT']['params'], 1),
         f(100 * wc['A_s1->A+FT']['params'], 1), pct((wc['C_s0->C+FT']['bn_stats'] + wc['C_s1->C+FT']['bn_stats']) / 2),
         pct((wc['A_s0->A+FT']['bn_stats'] + wc['A_s1->A+FT']['bn_stats']) / 2),
         f(max(wc['C_s0->C+FT']['mode8_col_rel_change'] + wc['C_s1->C+FT']['mode8_col_rel_change']), 1)))
    w('')
    w('### 9.5 結論は、先に決めた規則と種のばらつきから言えるか — 言えます')
    w('')
    w('- 規則のとおり: C・C+FT とも、「良い」の条件(2 つの種の両方が A の 2 本より小さい)を best でも満たしません。'
      'C+FT の小さい方(種 1、%s µs)も A 種 0(%s µs)より大きいです。' % (f(rcp['L']['C+FT_s1_best'], 3), f(rcp['L']['A_s0_best'], 3)))
    w('- 種のばらつき: 種 2 本の範囲は A %s・C %s・C+FT %s・A+FT %s µs で、s_A(%s µs、3 本の範囲)はやや小さめの見積もりです。'
      'この 4 つの範囲から種 1 本の標準偏差を見積もると約 %s µs で、2 本の平均どうしの差の標準誤差も約 %s µs です。'
      'C+FT − A は %s µs(悪い向き)なので、改善があるとしても約 %s µs(1%% 程度)より小さいと見るのが妥当で、10 µs の実用上の差はありません。'
      'ばらつきをもっと大きく見ても、閾値が広がるだけで判定は「同等」のままです。'
      % (f(rng['A'], 3), f(rng['C'], 3), f(rng['C+FT'], 3), f(rng['A+FT'], 3), f(E['s_A_us']['best'], 3), f(sig, 2), f(sig, 2),
         f(d_cft, 2, True), f(ub, 1)))
    w('- 「C は 4 通りすべてで A より悪い向き」は、best と last が同じ種の重みなので実質 2 つの種です。本当の差が無くても 1/6 の確率で起きるので、'
      '「混ぜると悪くなった」とも言えません(0 章にもこの注を足しました)。')
    w('')
    w('### 9.6 後から見た内訳(判定には使いません。読み方の注意)')
    w('')
    w('- **差は場面で一様ではありません。** C の +%s µs は、2 つのシーンでの大きな悪化から来ています: %s(ジャンル %d)%s µs、%s(ジャンル %d)%s µs。'
      'どちらのシーンも C・C+FT の 4 本すべてが A の 2 本より悪く、全体の差への寄与はこの 2 シーンで +%s µs、残りの 9 シーンで %s µs です。'
      'ほかの 9 シーンだけで比べると C が平均 %s µs、'
      'C+FT が %s µs 小さい向きでした(C が A より小さいシーンは 11 のうち %d、C+FT は %d)。ただし、9 シーンの差は A+FT − A のシーンごとの揺れ'
      '(%s〜%s µs)と同じくらいで、後から選んだ分け方なので、改善の証拠にはなりません。'
      % (f(am['C']['best'] - am['A']['best'], 2), w2[0], ps[w2[0]]['genre'], f(ps[w2[0]]['C-A'], 1, True), w2[1], ps[w2[1]]['genre'],
         f(ps[w2[1]]['C-A'], 1, True), f(sum(bd['two_worst_contribution_to_overall_diff_us']['C'].values()), 2),
         f((am['C']['best'] - am['A']['best']) - sum(bd['two_worst_contribution_to_overall_diff_us']['C'].values()), 2, True),
         f(-bd['without_two_worst']['C'], 2), f(-bd['without_two_worst']['C+FT'], 2), bd['scenes_C_better'], bd['scenes_CFT_better'],
         f(min(aft), 1, True), f(max(aft), 1, True)))
    w('- **低速の直線で、速さの癖が少し移っています。** 先に決めた規則(ジャンルの平均)では「移りなし」ですが、ジャンル × 場所で見ると、'
      '低速(5)の直線(場所の札 HS、%d フレーム)で前進の偏りが A %s / %s µs に対して、C %s / %s µs、C+FT %s / %s µs でした(負 = 人より強く前進)。'
      'C は両方の種、C+FT は種 1 で、A の種どうしの差(%s µs)と 10 µs の目安を超えます。この直線での L1 は C が A より %s µs 大きく、'
      '%s の悪化の多くはこの直線です(%d フレームで C − A %s µs)。ほかの場所のずれは最大 %s µs でした。'
      'C・C+FT を実車で使うと、低速の直線で速めに出るおそれがあります(どちらも推奨ではないので、今は実車で試しません)。'
      % (hs['n'], f(fb['A_s0'], 1, True), f(fb['A_s1'], 1, True), f(fb['C_s0'], 1, True), f(fb['C_s1'], 1, True), f(fb['C+FT_s0'], 1, True),
         f(fb['C+FT_s1'], 1, True), f(hs['A_seed_gap_fwd'], 1), f(hs['L1_C-A'], 1), w2[0], ps[w2[0]]['HS_frames'],
         f(ps[w2[0]]['HS_C-A_L1'], 1, True), f(other_max, 1)))
    w('')
    w('### 9.7 本文で直したこと(言い方だけ。数字と判定は同じ)')
    w('')
    w('1. 0 章「速さの癖は移りませんでした … ジャンル別の出力の頭が速さの違いを分けて持てています」→ ジャンルの平均では移っていないが、'
      '低速の直線では少し移っている、と書き直しました(9.6)。')
    w('2. 5 章の表の「強さの多くは ToF のずれと画像の上ずれから」→ A との差への寄与が最も大きいのは画像の下ずれ(C %s%%・C+FT %s%%)で、'
      'ただしそれは A 種 1 の値によるもの、と直しました。' % (pct(rob['contribution_to_gain']['C']['img_down3']['share']),
                                         pct(rob['contribution_to_gain']['C+FT']['img_down3']['share'])))
    w('3. 0 章・5 章の「(ずれへの強さを)失っていません」→ C で得た分の約 %s%% が微調整で戻り、大部分が残った、と直しました。'
      % pct(rob['ft_lost_share_of_C_gain']))
    w('4. 0 章の「4 通りすべてで悪い向きにそろっています」に、実質 2 つの種で、差が無くても 1/6 の確率で起きる、という注を足しました。')
    w('')
    return o


if __name__ == '__main__':
    main()
