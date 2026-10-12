"""評価の段(段階4の評価)の手順と判定の規則を、保留データの予測の前に eval/protocol.json に書く(CPU だけ・1 回だけ)。

    python /home/jetson/jetformer_reports/policy_zoo_20261012/eval/write_protocol.py

- 判定の規則そのものは results/zoo_rules.json(19:29:27 に固定、逸脱 D1〜D13)。ここでは変えない。sha256 を写して、
  評価の道具(tools/policy_zoo.py evaluate)がその規則をどう計算するかを、予測の前に書き残す。
- 同じ名前のファイルがあって中身(written_at を除く)が違えば止まる。保留データの予測(data/zoo/policy_zoo_20261012/pred/)が
  既にあれば、書かずに止まる(結果を見た後に手順を書かない)。
"""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ZOO = Path('/home/jetson/jetformer/data/zoo/policy_zoo_20261012')
RULES = [REPORT / 'results' / 'zoo_rules.json', ZOO / 'zoo_rules.json']
OUTS = [HERE / 'protocol.json', ZOO / 'eval_protocol.json']


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def protocol():
    rules = json.loads(RULES[0].read_text(encoding='utf-8'))
    shas = [sha256(p) for p in RULES]
    if len(set(shas)) != 1:
        raise SystemExit('zoo_rules.json の2つの写しが違います: %s' % shas)
    return {
        'title': '方策の比較(policy zoo)の評価の手順。保留データの予測の前に固定(判定の規則は zoo_rules.json のまま)',
        'version': 1,
        'rules_file': {'paths': [str(p) for p in RULES], 'sha256': shas[0], 'written_at': rules['written_at'],
                       'note': '判定の規則は zoo_rules.json の verdict_rules_effective・parameters・deviations(D1〜D13)を'
                               'そのまま使う。この protocol はその計算の仕方を書くだけで、規則は変えない'},
        'runs': rules['runs'],
        'code': {
            'tool': 'tools/policy_zoo.py(predict・crosscheck・engine-check・evaluate・figures)',
            'reuse': 'rl_evaluate.py の drive_outputs・heldout_passes・_pass_values・_erpm_map・entry_estimate・straight_mask・'
                     'pred_table・_val_arrays、compare_models.metrics、rl_review.minus_group_mean、post.replay(TCPPost)',
        },
        'predictions': {
            'model': 'runtime.load_run(<学習結果>, ckpt) → runtime.forward(FP32・autocast なし・eval。GPU。rl_evaluate predict と同じ)',
            'frames': 'rl_evaluate.pred_table()(heldout.json の pred_table_key 3d339512… と一致を確かめる): part 0 = ai_mix の val の'
                      '全シーンの vehicle_valid の 9,415 フレーム、part 1 = intervention_val の採点の 1,570 フレーム。計 10,985',
            'ckpts': ['best', 'last'],
            'runs': '10 本 + モデル B(B も同じコードで予測し直し、前回の data/rl/ai_mix_20261011/eval/pred_B.npz と差を確かめる)',
            'genre': 'aligned の mode(学習していないジャンルは trained_modes[0] で代用。rl_evaluate._genres と同じ)',
            'tcp_extra': 'tcp は waypoints [10,2] m・speed_logits [10] も保存',
            'perturbation': '主指標のフレーム(4,670)のうちフレーム番号 % 3 == 0 のもので、5 つのずれ: 画像(224×224 の BGR)を 3 行下へ'
                            '(上の 3 行は元の 1 行目を複製)・3 行上へ(下の 3 行は最後の行を複製)、明るさ ×0.8・×1.2(float で掛けて'
                            '四捨五入し 0〜255 に切る)、ToF の生の値を 1 行下へ(上の 1 行 = 0 = 無効)。正規化(normalize_inputs)の前に掛ける。'
                            'best.pth だけ(10 本 + B)',
            'crosscheck': '各方策の種 0 とモデル B で、1 シーン(20261010-202633-0003)を tools/compare_models.py の predict_scene'
                          '(走行と同じ post.Stage・1 フレームずつ・教師あり/自己入力)で予測し、まとめた予測との差の最大を出す',
        },
        'primary': {
            'L': '人のきれいな走行の L1 [µs] = |予測 − 人|(10 ステップ × [ステア, スロットル] の平均)のフレーム平均。フレーム = part 0 の'
                 '人の 11 シーン & future_valid & vehicle_valid & 水平 2.5G の衝突の前後でない(4,670)。tcp は actions',
            'policy_value': 'L̄_P = 2 つの種の平均、[min_P, max_P]',
            'reference': 'R = jetformer_v0 の種 0・1(L̄_R・[min_R, max_R])。モデル B は別の行',
            'delta': 'δ = max(10, {L(B), L(v0 種0), L(v0 種1)} の範囲)。best.pth の値で計算',
            'classes': '良い: Δ ≤ −δ かつ max_P < min_R / 悪い: Δ ≥ +δ かつ min_P > max_R / ほかは同等',
            'ckpt_check': 'last.pth の予測で同じ計算(δ も last の 3 本の範囲と 10 の大きい方)。best と last で分類が違えば「良い」は「同等」に下げる',
        },
        'r1_de': {
            'outputs': '走行の出力 = drive_outputs(予測, (steer_center_us, throttle_neutral_us), ゲイン 1.3, 1.0, 先読み 3, トリム 0)',
            'passes': '保留の R1 17 通過(窓 = 起点の 0.4 秒前〜起点)、DE 14 通過(緑マット出口〜0.8 秒)。heldout.json と数を確かめる',
            'per_pass': '通過ごとに窓の出力の平均(R1 = スロットル、DE = ステア)。方策の値 = 種の平均、R = v0 の種の平均',
            'd_bar': 'd̄ = 通過で平均した (P̄ − R̄)。d_ij = 通過で平均した (P_i − R_j)(4 組)',
            'bootstrap': '通過の選び直し 10,000 回・種 0(np.random.default_rng(0))・全部の学習結果で同じ添え字',
            'good': '(a) d̄ の 99.375% 区間(0.3125%〜99.6875% 点)の下限 > 0、(b) 2 つの種の両方で d̄_i(種 i − R̄)> 0、'
                    '(c) 4 組すべてで d_ij ≥ 10 µs、(d) 差の差(行ごとの差 P̄ − R̄ から、同じシーンの part 0 の全フレームの差の平均を引いて'
                    '窓で平均)の 99.375% 区間の下限 > 0。全部満たして「良い」',
            'bad': 'd̄ の 95% 区間(同じ 10,000 回)の上限 < 0',
            'sign': 'R1: + = スロットルの µs が大きい = 進入が遅い。DE: + = 左',
            'entry_estimate': '参考: 見積もった進入 eRPM = 実際の進入 eRPM + 対応表(_erpm_map)での差(出力 − 実際の指令)。人の実際の進入と並べる',
        },
        'collapse': {
            'straight': '保留の直線 542 フレーム(straight_mask & part 0 & val)の前進の強さ 1498 − 出力のスロットル。低下 = 1 − P̄/R̄ ≤ 0.05',
            'fwd_bias': 'compare_models.metrics の fwd_bias_us(3 フレーム先の予測 − 人、主指標のフレーム)の種の平均 ≥ −30',
            'sign_acc': 'compare_models.metrics の steer_sign_acc(主指標のフレーム)の種の平均 ≥ R の種の平均 − 0.01',
            'rule': 'どれか 1 つでも外れたら「崩れ」(点推定。種ごとの値も載せる)',
        },
        'robustness': {
            'delta_l1': 'ΔL1(r) = 5 つのずれの L1 の平均 − ずれなしの L1(同じ約 1,557 フレーム、best.pth)',
            'good': 'ΔL1_P(種の平均)≤ 0.8 × ΔL1_R(R = v0 の種の平均)かつ 2 つの種の両方で ΔL1 ≤ ΔL1_R',
        },
        'negative_controls': {
            'NC-A': '方策 = v0 種 1、基準 = {v0 種 0, モデル B}',
            'NC-B': '方策 = v0 種 0、基準 = {v0 種 1, モデル B}',
            'apply': '主指標(δ は上と同じ)・R1・DE(D1〜D3、組は 1 × 2、「両方の種」は 1 本)・崩れ・ずれ。どれかで「良い」か「崩れ」なら規則 3 は無効。'
                     '「崩れ」・「悪い」なら、ほかの方策の同じ判定に「ばらつきの内の可能性」と書く',
        },
        'gates': {
            'source': 'エンジンの確認(drive.model_info → TRTRunner → check_io → 空回し、実フレーム 64 個で TensorRT と PyTorch FP32(CPU)の差、'
                      '100 回の空回しの後 600 回の run_numpy)。10 本の FP16 のエンジンは train/engine_check/(学習の段で測ったもの)、'
                      'モデル B と作り直した dinov2 は policy_zoo.py engine-check(同じ計算)',
            'items': 'エンジン(作成の終了コード 0・check_io)、精度 ≤ 2.5 µs(tcp は通過点 ≤ 0.02 m・速度の期待値 ≤ 0.05 m/s)、'
                     'run_numpy 中央 ≤ 6.0 ms・p99 ≤ 8.0 ms、peak_mem_mb ≤ 4,096、drivable、保留の 10,985 フレームで NaN・Inf なし(best・last)',
            'both_seeds': '方策は 2 つの種の両方が合格すること',
            'dinov2': 'D7 のとおり、判定のゲートは学習の段の FP16 のエンジン(種 1 が 2.85 µs で不合格)= 候補にしない。主の判断(2026-10-12 19:05)'
                      'で build_engine.py に精度 fp16_norm_fp32(FP16 + LayerNorm・Softmax だけ FP32)を足し、2 本のエンジンを作り直して同じ'
                      '確認をする。これは「走行タブで走れるか(drivable の印)」を決めるだけで、判定の候補には入れない(事後の案のため)。'
                      '作り直しでも落ちる種があれば dinov2 は「走行不可」と印を付け、机上の比較には残す。参考として「dinov2 を候補に入れたら'
                      '判定が変わるか」も出す(判定には使わない)',
        },
        'verdict': {
            'rules': rules['verdict_rules_effective'],
            'tie_speed': 'D11: run_numpy の中央値(種の平均)が 0.3 ms 以上違うときだけ速さで決める',
            'wording': 'D5: δ は「実用上の差の閾値(L1 の約 20%)」。結論は「v0 より 10 µs 以上良い方策は無い。〜はほぼ同じ。〜は約 x µs(y%)悪い」の形',
        },
        'reference_not_judged': {
            'self_fed_mae': 'compare_models.metrics の k1〜k10 のステア・スロットルの MAE、向きの正解率、前進の偏り(主指標のフレーム = clean、'
                            '衝突の前後も含む human_all、ジャンル 4・5 別)。今の操作値を入れないモデルなので教師あり = 自己入力(crosscheck で確かめる)',
            'intervention_val': 'iv_all・ログ単位で保留の 218 フレーム(3 窓)・学習のログの窓・分類別。D12: 学習データの中の参考',
            'ai_frames': 'AI の val のフレームとの一致(AI 自身の出力との一致。精度ではない)',
            'd8_boundary': '境目 3 か所の検証の側 1 秒(152405-0002 の最初と最後の 1 秒、152707-0001 の最後の 1 秒)を除いた主指標',
            'latency': 'run_numpy・エンジンだけの中央値と p99(同じ道具)、パラメータ数。走行ループの見積もり = 5.6 ms(モデル B の実車)+ '
                       '(run_numpy_P − run_numpy_B)(B も同じ道具で測る)。カメラ → PWM も 28 ms + 同じ差。見積もりで実測ではない(D11)',
            'spread': '種どうしの範囲、best − last の差(主指標・学習中の検証)',
            'tcp': '§5.5: 保存した予測を post.replay(走行と同じ Stage・TCPPost)でシーンごとに時刻の順(2 フレーム以上あいたら reset)、'
                   'v_now = aligned の vehicle[:, 3]。control(0, 上限 0)・control + 上限(0, 2.75)・hybrid(1, 0)・hybrid + 上限(1, 2.75)・'
                   'trajectory(3, 0)の L1、R1 の窓のスロットル(上限あり − なしの通過ごとのブートストラップ 95%)、見積もった進入 eRPM、'
                   '直線の前進の強さ、v_hat の誤差。通過点の ADE/FDE [m]・目標速度の誤差 [m/s](clean_driving の val のラベル、wp_valid の'
                   'フレーム)。「速度の上限が使える」と書く条件は spec §5.5 のまま。D13: tcp_mode 1・3 は今の較正ではまだ使えない精度',
        },
        'recommendation': {
            'rule': 'ベストが jetformer_v0 なら、推奨のエンジン = モデル B のエンジン(今の走行の標準のまま)。ベストが v0 以外なら、その方策の'
                    '2 つの種のうち主指標 L(best.pth)が小さい方のエンジン。どちらも、走行と同じ手順(drive.model_info → TRTRunner → check_io → '
                    '空回し → Driver._loop を偽の PCA9685 で回す)で読めることを確かめる。机上の結論は実車の改善を示さない(規則 7)',
        },
        'car_card': {
            'order': 'A(モデル B)→ B(推奨)→ A(モデル B)。推奨がモデル B のときは、カードは「基準値を取る走行」と、机上の次点を試すときの形を書く',
            'fixed': 'ステアのゲイン 1.3・速度のゲイン 1.0・先読み 3・トリム 0・ジャンル 5・ToF ガード 1(記録だけ)・リカバリー オン・'
                     '後退の前に中立 オン・重ね合わせ オフ。tcp なら tcp_mode 0・speed_cap 0',
            'amount': '各ブロック 43 周以上(目標 60 周)',
        },
    }


def main():
    found = [str(p) for p in (ZOO / 'pred', ZOO / 'pert') if p.exists() and any(p.iterdir())]
    if found:
        raise SystemExit('保留データの予測が既にあります(手順を後から書かない): %s' % found)
    rec = protocol()
    for out in OUTS:
        if out.exists():
            old = json.loads(out.read_text(encoding='utf-8'))
            if {k: v for k, v in old.items() if k != 'written_at'} != rec:
                raise SystemExit('%s は別の中身で書かれています(上書きしない)' % out)
            print('同じ中身で書かれています:', out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(out) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(dict(rec, written_at=datetime.datetime.now().isoformat(timespec='seconds')), f, ensure_ascii=False, indent=1)
        os.replace(tmp, out)
        print('書きました:', out)


if __name__ == '__main__':
    sys.exit(main())
