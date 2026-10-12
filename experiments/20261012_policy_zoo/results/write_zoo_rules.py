"""評価の規則(spec.md §5)を、保留データの予測(policy_zoo.py predict)の前に固定して zoo_rules.json に書く(CPU だけ)。

    python /home/jetson/jetformer_reports/policy_zoo_20261012/results/write_zoo_rules.py [--dry-run]

- spec.md §5 をそのまま写し(verbatim)、spec.md の sha256 と更新時刻、書いた時刻、その時点で見えていた結果・見えていなかった結果、
  §5 からの逸脱(2026-10-12 のレビュー)と §5 のあいまいな点の決め方を残す。
- 書くのは 1 回だけ: 同じ名前のファイルがあって中身(written_at を除く)が違えば止まる(heldout.json と同じ考え方)。
- 保留データの予測(data/zoo/policy_zoo_20261012/pred/・results/ の予測)が既にあれば止まる(結果を見た後に規則を書かない)。
書く先: このフォルダーの zoo_rules.json と /home/jetson/jetformer/data/zoo/policy_zoo_20261012/zoo_rules.json
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
SPEC = REPORT / 'spec.md'
REPO = Path('/home/jetson/jetformer')
ZOO = REPO / 'data' / 'zoo' / 'policy_zoo_20261012'
OUTS = [HERE / 'zoo_rules.json', ZOO / 'zoo_rules.json']
SPEC_SHA256_REVIEWED = 'd9bc9077227500a78a64ece0ef7d73f25cc48104d95559e73c45fea1451cad05'   # レビューが読んだ版
POLICIES = ['act', 'tcp', 'msfusion', 'dinov2']
BONF_N = len(POLICIES) * 2                       # 4 方策 × {R1, DE}
CI_GOOD = round(1.0 - 0.05 / BONF_N, 6)          # 0.99375


def section5(text):
    """spec.md の「## 5. 評価の手順」から「## 6.」の前まで(そのまま)"""
    a = text.index('## 5. 評価の手順')
    b = text.index('## 6. 管理画面')
    return text[a:b].rstrip().rstrip('-').rstrip()


def runs():
    plan = json.load(open(REPORT / 'train' / 'plan.json'))
    r = plan['runs']
    return {k.replace('/s', '_s'): v for k, v in r.items()}


def predictions_exist():
    """保留データの予測がもうあるか(あれば規則を書かない)"""
    found = []
    for d in (ZOO / 'pred', ZOO / 'pert', HERE / 'pred'):
        if d.exists() and any(d.iterdir()):
            found.append(str(d))
    for p in (HERE / 'zoo_eval.json', ZOO / 'zoo_eval.json'):
        if p.exists():
            found.append(str(p))
    return found


def rules(spec_text):
    sha = hashlib.sha256(spec_text.encode('utf-8')).hexdigest()
    st = SPEC.stat()
    return {
        'title': '方策の比較(policy zoo)の評価の規則。保留データの予測の前に固定',
        'version': 1,
        'spec': {'path': str(SPEC), 'sha256': sha, 'sha256_matches_review': sha == SPEC_SHA256_REVIEWED,
                 'mtime': datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
                 'section_5_verbatim': section5(spec_text)},
        'timeline': {
            'spec_last_modified': '2026-10-12T13:51:14(spec.md。学習の開始より前)',
            'training_started': '2026-10-12T14:11:19(train/plan.json)〜14:11:25(1 本目の学習)',
            'training_finished': '2026-10-12T18:24:05(10 本とエンジン)',
            'engine_checks': '2026-10-12T18:25〜18:27(train/engine_check/)、dinov2 の一部 FP32 の実験 〜18:40(train/mixed_check/)',
            'review': '2026-10-12 夜(この規則の逸脱 D1〜D13 の元)',
            'note': 'spec §5 は「学習を始める前に zoo_rules.json に書く」としていたが、書かれないまま学習が進んだ。'
                    '救いは spec.md の最後の更新(13:51:14)が学習の開始より前で、§5 の中身は学習の結果を見る前のものであること。'
                    'このファイルは §5 をそのまま写し、学習の結果を見た後の変更は全部「逸脱」として下に書く',
        },
        'seen_before_rules': {
            'visible': ['学習中の検証 L1(best.pth を選んだ検証 = 保留の人の 11 シーンの stride 3。best・last の両方)',
                        'エンジンの確認(実フレーム 64 個の TensorRT と FP32 の差・速さ。10 本)',
                        'dinov2 の一部 FP32 のエンジンの実験(2 本)'],
            'not_visible': ['主指標(stride 1 の 4,670 フレーム)・R1・DE・崩れ・ずれへの強さ・交代の後の人の修正の、どの予測も計算していない'],
            'caveat': '主指標のフレームの 1/3 は学習中の検証と同じフレームそのもので、残りも 33 ms 隣のフレームなので、'
                      '主指標は事実上もう見えている(学習中の検証 49.1〜50.8 µs)。そのため δ = 10 µs は変えない(D5)',
            'predictions_found': predictions_exist(),
        },
        'runs': {'policy': runs(), 'model_B': 'clean_driving_20261011-nocmd-s3-20261011-024626'},
        'parameters': {
            'delta_us': 10.0, 'delta_rule': 'δ = max(10 µs, {L(B), L(v0 種0), L(v0 種1)} の範囲)(spec §5.2 のまま)',
            'r1_de_good_min_pair_d_us': 10.0, 'ci_good': CI_GOOD, 'ci_bad': 0.95, 'bonferroni_comparisons': BONF_N,
            'bootstrap_n': 10000, 'bootstrap_unit': '通過(R1 17・DE 14。spec §5.3 の通過ごと)', 'bootstrap_seed': 0,
            'speed_tiebreak_min_ms': 0.3,
            'drive_settings': {'steer_gain': 1.3, 'throttle_gain': 1.0, 'lookahead': 3, 'steer_trim_us': 0.0},
        },
        'deviations': [
            {'id': 'D1', 'from': 'spec §5.3 R1・DE の「良い」(区間の下限 > 0 かつ 2 つの種の両方で d の平均 > 0)',
             'to': '「良い」には、方策の各種 i × v0 の各種 j の 4 組すべてで d_ij ≥ 10 µs も要る'
                   '(d_ij = 通過で平均した P_i − R_j。R1 はスロットル(+ = 進入が遅い)、DE はステア(+ = 左))。'
                   '3 本目の種(O1)を回したときは全部の組(3 × 3)',
             'why': '通過ごとのブートストラップの区間には学習のばらつきが入っていない。前回の実験(experiments/20261011_offline_rl)で、'
                    '同じ学習の last.pth − best.pth だけで 16 組中 12 組の区間が 0 を含まず、R1 で最大 9 µs・DE で最大 6 µs 動いた。'
                    '10 µs は前回の実験が「学習のばらつきの内」と決めた大きさ(tools/rl_review.py TRAINING_NOISE_US)',
             'source': 'review 2026-10-12 must_fix 1(a)'},
            {'id': 'D2', 'from': 'spec §5.3(差だけ)',
             'to': '「良い」には、シーンの平均を引いた差(差の差: 行ごとの差から、同じシーンの保留の全フレームの差の平均を引いてから'
                   '窓で平均。tools/rl_review.py minus_group_mean と同じ)の d̄ の区間の下限 > 0 も要る',
             'why': '前回の実験では R1・DE の差の多くが場所によらない全体のずれだった(c − B の DE は −11.1 µs、シーンの平均を引くと +6.8 µs で'
                    '向きが逆)。全体に遅い・左に寄っただけの方策を「良い」にしない',
             'source': 'review 2026-10-12 must_fix 1(b)'},
            {'id': 'D3', 'from': 'spec §5.3 の 95% 区間・2,000 回',
             'to': '「良い」の区間(D2 も)は 8 回(4 方策 × R1・DE)の Bonferroni で %.3f%%(両側、下限は %.4f%% 点)。'
                   'すそを安定させるためブートストラップは 10,000 回。「悪い」(区間の上限 < 0、候補から外す向き)は spec のまま 95%%'
                   % (CI_GOOD * 100, (1 - CI_GOOD) / 2 * 100),
             'why': '多重比較の補正が無いと、学習のばらつき σ ≈ 3〜5 µs のもとで、どれか 1 つが偶然「良い」になる確率を約 25〜60% と見積もった',
             'source': 'review 2026-10-12 must_fix 1(c)'},
            {'id': 'D4', 'from': '(spec に無い)',
             'to': '陰性対照: NC-A = v0 種 1 を「方策」、{v0 種 0, モデル B} を基準 / NC-B = v0 種 0 を「方策」、{v0 種 1, モデル B} を基準 '
                   'として、§5.2(主指標)・§5.3(R1・DE は D1〜D3 つき、組は 1 × 2)・崩れ・ずれへの強さを全部そのまま当てる。'
                   'どちらかの陰性対照が、主指標・R1・DE・ずれのどれかで「良い」、または「崩れ」になったら、§5.4 の規則 3 は無効'
                   '(規則 2 が当てはまらなければ規則 4 = jetformer_v0)。陰性対照が「崩れ」・「悪い」になったときは、ほかの方策の同じ判定に'
                   '「ばらつきの内の可能性」と書く(候補から外すのはそのまま。v0 が残る向きなので)',
             'why': '規則 3 と「崩れ」・「ずれ」の判定が、同じ方策の別の種どうしで偶然に出ないことを、データで確かめる',
             'source': 'review 2026-10-12 must_fix 1(d)・should_fix 7'},
            {'id': 'D5', 'from': 'spec §5.4-5「δ の中の順位には意味が無いと明記」、training.md の「別々に学習しただけで主指標は最大 ~10 µs 動く」',
             'to': 'δ = 10 µs は「実用上の差の閾値(L1 の約 20%)」と書き、学習のばらつきとは書かない(判定の δ は変えない)。'
                   '学習のばらつきは実測を出す: 主指標・学習中の検証の、種どうしの範囲と best・last の差。'
                   '全部の種 × {best, last} で向きがそろう差は「小さいが一貫した差」と書く(δ より小さいので判定は「同等」のまま)。'
                   '結論は「v0 より 10 µs 以上良い方策は無い。〜はほぼ同じ。〜は約 x µs(y%)悪い」の形で書き、'
                   '「v0 がベスト」とも「順位は付けられない」とも書かない',
             'why': '10 µs は R1・DE の窓の出力のばらつき(前回の実験)で、L1 のばらつきではない。L1 の種の範囲は今回 0.0〜0.8 µs、'
                    '前回の RL の各方策でも 50.85〜51.55 µs に収まっていた。act・dinov2 は best でも last でも 2 つの種で v0 より悪い',
             'source': 'review 2026-10-12 must_fix 2'},
            {'id': 'D6', 'from': 'spec §5.3・§5.4 のあいまいな点',
             'to': {'R': 'd の基準 R は v0 の 2 つの種の平均(d̄ = 通過で平均した P̄ − R̄ = 4 組の d_ij の平均)。区間はこの d̄ で計算し、'
                         '通過の選び直しは全部の学習結果で同じ添え字を使う',
                    'pairs': '種の番号は方策どうしで対応しないので、組は「どの種どうし」ではなく全部の組(2 × 2)',
                    'collapse': '「崩れ」は方策の種の平均と v0 の種の平均で判定する(spec の点推定のまま)。種ごとの値も全部載せる。'
                                'ばらつきの確かめは D4 の陰性対照(1 本どうしなので、平均どうしよりばらつきが大きい = 厳しい確かめ)',
                    'robustness': 'ずれへの強さの R も v0 の種の平均(「両方の種で R 以下」の R も同じ)'},
             'why': 'spec では決まっていなかった。予測の前に決める',
             'source': 'review 2026-10-12 should_fix 1'},
            {'id': 'D7', 'from': 'spec §5.1(精度のゲート)・§3.4(代わりの精度 bf16 → fp32)',
             'to': 'dinov2 は spec のとおり「ゲート不合格 = 走行の候補にしない」(種 1 の差の最大 2.85 µs > 2.5 µs。'
                   '先に決めた代わりの bf16 は 4.31 µs で差で不合格、fp32 は run_numpy 9.5 ms で速さで不合格)。'
                   'FP16 + 正規化・Softmax だけ FP32 のエンジン(種 0 1.72・種 1 2.26 µs で合格)は結果を見た後の案なので、'
                   'この比較では事後の参考として載せるだけ。主指標・副指標は FP32 で計算して全部載せる(順位には影響しない。影響は候補かどうかだけ)',
             'why': '保留データの数字を出す前に決める。64 フレームの差の最大はエンジンの作成ごとに揺れる(同じ設定で 1.43(段階3の試しの重み)・'
                    '2.46(種 0)・2.85 µs(種 1))ので、種 0 合格・種 1 不合格をモデルの差と読まない',
             'source': 'review 2026-10-12 should_fix 2'},
            {'id': 'D8', 'from': 'spec §5「保留」',
             'to': '主指標の 11 シーンは「同じセッションの中で保留したシーン」と書く(検証のシーンは全部、学習のシーンと同じ記録から取られ、'
                   '3 か所は境目が 0.033 秒で続いている: 152405-0001(学習)→ 0002(検証)、152405-0002(検証)→ 0003(学習)、'
                   '152707-0001(検証)→ 0002(学習))。感度の確認(参考。判定に使わない)として、この 3 つの境目の前後 1 秒(検証の側)を除いた'
                   '主指標も出す。セッションの外にあるのは AI の R1 4 通過・AI の DE 4 通過(記録 20261010-172505・204208)と'
                   '交代の後の人の修正のログ単位で保留の 3 窓だけ',
             'why': 'stride 3 の学習中の検証は stride 1 の主指標の 1/3 そのもので、best.pth もこの検証で選んでいる',
             'source': 'review 2026-10-12 should_fix 3'},
            {'id': 'D9', 'from': 'spec §5.6「best.pth は … 主指標は全方策で同じだけ楽観的」',
             'to': 'この文は使わない。エポックを選ぶことによる甘さは、終盤の検証の揺れが大きい方策ほど大きい。'
                   '学習中の検証の best − last は v0 0.1 / 0.3・act 0.3 / 0.5・tcp 0.2 / 0.1・msfusion 1.0 / 0.1・dinov2 0.4 / 0.8 µs'
                   '(種 0 / 種 1)で、どれも δ よりずっと小さく判定は変わらない。主指標でも last.pth の値を並べる(spec §5.2 のまま)',
             'why': '理屈として正しくなかった',
             'source': 'review 2026-10-12 should_fix 4'},
            {'id': 'D10', 'from': '(spec に無い)',
             'to': '結論は「v0 の学習の条件(lr 3e-4・cosine・30 エポック・AdamW wd 1e-4・勾配のクリップ 1.0・バッチ 64・AMP)で、'
                   '方策ごとの調整なし」に限ると書く',
             'why': 'この条件は v0 向けに決めたもので、v0 に有利になりうる。msfusion は 1〜10 エポックで検証が 91〜150 µs まで暴れ'
                    '学習 L1 も 32 µs(v0 14 µs)、act・tcp は収束が遅い(10 エポックまで 55〜65 µs)',
             'source': 'review 2026-10-12 should_fix 5'},
            {'id': 'D11', 'from': 'spec §5.4-2・3 の「速い方」',
             'to': '速さで決めるのは run_numpy の中央値(種の平均)が 0.3 ms 以上違うときだけ(1 回の測定は測り直すと ~0.15 ms 揺れる)。'
                   'それより小さければ L̄ の小さい方。v0・act・tcp(2.4〜2.7 ms)の速さには順位を付けない。モデル B の「TRT 1.9 ms・'
                   '走行ループ 5.6 ms」(別の道具・実車)と spec §0.1 の表(乱数の重み)は、今回の数字と並べない。tcp の数字は制御器'
                   '(TCPPost、tcp_mode ≠ 0 のとき)の時間を含まない。新しい方策の実車での走行ループの時間は測っていない',
             'why': '測り直しの揺れ(v0 1.63 → 1.48 ms、tcp 種 0 2.66・種 1 2.52 ms)より小さい差で順位を付けない',
             'source': 'review 2026-10-12 should_fix 6'},
            {'id': 'D12', 'from': 'spec §5.3「参考」の交代の後の人の修正(intervention_val)',
             'to': '数字を出すなら「学習データの中の参考」と書き、方策の比較の根拠にはしない(29 窓のうちログ単位で保留なのは 3 窓だけ)',
             'why': 'intervention_val は学習に使った記録とほとんど重なっている',
             'source': 'review 2026-10-12 should_fix 8'},
            {'id': 'D13', 'from': 'spec §5.5(tcp の追加の評価)・§5.7(実車のカード)',
             'to': '§5.5 は参考のまま行う。ただし報告には「tcp_mode 1・3 のスロットルの写像は今の較正ではまだ使えない精度。車で試すのは '
                   'tcp_mode 0 だけ」と書く(正解の目標速度を入れても人のスロットルとの相関 0.48・平均の差 71 µs・前進を出す割合 85%'
                   '(人 93%)。tcp_check/oracle_modes.json)。§5.7 の別のカード(hybrid・上限)は、写像を直すまで行わない',
             'why': '制御器の写像の精度の確かめ(CPU)',
             'source': 'review 2026-10-12 should_fix(tcp_mode 1・3 の写像)'},
        ],
        'verdict_rules_effective': [
            '1. 候補 C = v0 以外で、§5.1 のゲートを 2 つの種の両方で満たし(dinov2 は D7 で不合格)、主指標が「悪い」でなく、'
            '「崩れ」でなく、R1・DE のどちらも「悪い」(95%)でない方策',
            '2. C の中に主指標が「良い」(Δ ≤ −δ かつ max_P < min_R、best.pth と last.pth で分類が同じ)のものがあれば、'
            'ベスト = その中で L̄ が最小(L̄ の差が δ/2 未満なら D11 の速さ → L̄)',
            '3. 2 が無く、陰性対照(D4)がどれも「良い」・「崩れ」にならず、C の中に主指標が「同等」で R1・DE のどちらかが「良い」'
            '(D1〜D3 の全部を満たす)ものがあれば、「机上の候補(L1 は同等、R1/DE で良い)」= {R1 良い, DE 良い, ずれに強い} の数が多いもの'
            '(同じなら L̄ → D11 の速さ)',
            '4. どちらも無ければ、ベスト = jetformer_v0(モデル B のまま)。「机上では v0 より良いと言えない」と書く',
            '5. 全方策の L̄ と範囲・δ・副指標・陰性対照を表に出す。書き方は D5(δ = 実用上の閾値、小さいが一貫した差)',
            '6. O1(3 本目の種)を回したときは 3 本でやり直す(δ も v0 の 4 本の範囲、D1 の組は 3 × 3)',
            '7. 机上の結論は実車の改善を示さない(spec §5.4-7 のまま)',
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='表示だけ(書かない)')
    a = ap.parse_args()
    spec_text = SPEC.read_text(encoding='utf-8')
    r = rules(spec_text)
    if r['seen_before_rules']['predictions_found']:
        sys.exit('保留データの予測がもうあります。規則は予測の前にしか書きません: %s' % r['seen_before_rules']['predictions_found'])
    if not r['spec']['sha256_matches_review']:
        sys.exit('spec.md がレビューの版(%s…)から変わっています。逸脱を見直してから書いてください' % SPEC_SHA256_REVIEWED[:12])
    for p in OUTS:
        if p.exists():
            old = json.load(open(p, encoding='utf-8'))
            old.pop('written_at', None)
            old['seen_before_rules'].pop('predictions_found', None)
            cur = json.loads(json.dumps(r, ensure_ascii=False))
            cur['seen_before_rules'].pop('predictions_found', None)
            if old != cur:
                sys.exit('%s があって中身が違います。規則は 1 回だけ書きます(変えるなら逸脱として別に記録する)' % p)
    r['written_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    text = json.dumps(r, ensure_ascii=False, indent=1)
    if a.dry_run:
        print(text)
        return
    for p in OUTS:
        if p.exists():
            print('同じ中身があります(書かない):', p)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        os.replace(tmp, p)
        print('書いた:', p, r['written_at'])
    print('spec.md sha256', r['spec']['sha256'])


if __name__ == '__main__':
    main()
