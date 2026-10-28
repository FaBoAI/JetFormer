"""AI走行の補助(モデルの出力を受け取り、実際に車へ出す値を返す)

このESCの後退(2026-10-10 のログ116回分から確認。data/reports/ai_runs_20261010/human_recovery.md):
  ・前進の直後の後退側の指令は、必ずブレーキになる(何秒中立を挟んでも、前へ転がっている間は後退しない)
  ・止まってから(または後ろへ転がっている間に)後退側 → 中立(約0.12秒)→ 後退側 で下がる(89%)
  ・下がり始めるのは後退側を出してから約0.08秒後
人のリカバリー: ブレーキ約0.27秒 → 中立 → 後退(約1712µs)を約0.3秒 → 惰性で下がる → 止まる前の曲がる向きへ
ハンドルを切って前進(約1369µs・0.6秒)→ AIへ戻す。

1. 自動リカバリー(Recovery)
   検出: (a) 前へ走っていて衝撃(水平2.5G以上)を受け、0.6秒以内に止まり、そのまま STUCK_AFTER_IMPACT_S 進まない
         (b) 一度動いてから、前へも後ろへも進まない状態が STUCK_S 続いた
   動作: ブレーキ → 止まるのを待つ → 中立 → 後退(強さ rev_us・時間 rev_s。下がり始めなければ中立を長くしてやり直し)
         → 中立で惰性 → 止まる前の曲がる向きへハンドルを切って前進(ESCAPE)→ モデルへ戻す
   後退中に衝撃を受けたら(後ろにぶつかった)すぐ前進へ。GIVEUP_N 回が GIVEUP_WINDOW_S 秒以内に続いたら、
   あきらめて中立のまま人の交代を待つ(CH3を人側へ戻すと解除)。
   後ろの見張り(back_guard。drive の rec_back_guard、2026-10-26 fix2 safety2-1。既定 OFF。OFF は今までと同じ):
   直前のモデルの指令がブレーキ側だと、ESC は後退を待っている。そのとき最初の「ブレーキ」がそのまま後退になる
   (10/23 20261023-205458-0002: 'brake'・'gap' の間に下がり始め、'reverse' が 0 から数え直して 0.5 秒後退を出し、
   1.24 m 下がって後ろに 5.8G で当たった)。ON のときは、どの段でも
   ・後ろへ動いている(eRPM > BACKING_ERPM)なら、もう下がっているとして数える('brake'・'wait' → すぐ 'reverse'。
     'gap' の惰性も数える)。下がった時間が rev_s に届いたら惰性('coast')へ
   ・後ろへ動いている・直前 BACK_RECENT_S まで動いていたときの衝撃(IMPACT_G 以上)は、後ろにぶつかったとしてすぐ前進
     ('escape'。やり直しの +50 µs はしない)
   中立止めのあとも残す(Assist.reset の keep_recovery。drive の rec_hold_keep、2026-10-26 fix3 safety1-1。既定 OFF。
   OFF は今までと同じ = 中立止め 1 回で Recovery を全部戻す): ON のとき、AI 側のままの中立止め(カメラ 0.1 秒・処理の遅れ・
   センサーが古い・I2C の失敗など。1 フレームでもよい)では Recovery を戻さない。中止(giveup)は中止のまま、途中の段は続きから
   (段の時間は中立止めの間も進む)、armed・衝撃・あきらめの数も残す。戻すのは CH3 を人側にしたとき(step の select = False)と
   走行の始め(drive は走行ごとに Assist を作る)。後退の受け付け(EscReverse)は今まで通り戻す。
   drive は後ろの見張り(rec_back_guard)も ON のときだけ使う(残した後退の途中で後ろに当たったとき、見張りが無いと
   'reverse' のやり直しが後ろを押し続けるため。今までは中立止めの reset が手順を偶然切っていた)。
   中立止めの間の衝撃(2026-10-26 fix3 r1-rs1-1): 中立止めの間は step が呼ばれず、impact_g は処理したフレームの直近 0.15 秒の
   IMU しか見ないので、残した手順の途中(後退側・下がる惰性の段)に中立止めをはさんで後ろへ当たると見落とし、明けたあと
   'reverse' が後ろの物を押し続けた(最大 約 0.8 秒)。drive は残した中立止めのあとの最初のフレームで、その間の IMU の
   水平 G の最大を hold_g で渡し、IMPACT_G 以上ならすぐ前進('escape'。出来事 recovery_guard の rear_impact_hold)
   中立止めの時間(2026-10-27 fix3 直す 2 回目 r2-r2-rs-1・rs-2): 前は段の時間(t_phase)・止まっている時間(slow_since)が
   中立止めの間も進み、下がった時間(backed_s)は明けたフレームの dt(0.2 秒まで。止まっていれば 0)しか数えなかった。そのため
   'reverse' の始めの中立止めのあとは「下がらない」と見てやり直し(+50 µs)、新しく rev_s 下がった(中立止めなしより後ろへ多く)。
   2〜5 秒の中立止めのあとは、明けて 0.02〜0.35 秒で後退を出した(段がすぐ終わる・すぐ stuck と見る)。drive は残した中立止めの
   あとの最初のフレームに hold_since(残した中立止めが中立を書き始めた時刻)と、その間に後ろへ動いていた秒 hold_back_s
   (車両の値から。多めに数える側)を渡す:
   ・下がった時間: hold_back_s を足す(このフレームの dt は足さない)。'brake'・'wait' なら見張りと同じく 'reverse'(rev_s に
     届いていれば 'coast')へ、'gap' で rev_s に届いたら 'coast' へ(出来事 recovery_guard の back_hold)
   ・長い中立止め(見ていない時間 = 前に処理したフレームから HOLD_KEEP_MAX_S 秒より長い): 後退側・下がる惰性の段か、止まっている時間を数えている
     途中(段なし・slow_since あり)なら 'giveup'(中立で人を待つ。見ていない間に人が近づいたかもしれない。出来事 hold_long)。
     中立止めの間に後ろの衝撃(hold_g)があったときは rear_impact_hold が先(すぐ前へ)
   ・短い中立止め: 段の時間・止まっている時間は中立を出していた間(hold_since から)は進めない(t_phase・slow_since を後ろへ
     ずらす。中立を書くまでの前の値が出ていた間は進める)。'reverse' の
     下がり始めを待つ時間(CHECK_S)は明けたフレームから数え直す(t_check。中立で ESC の後退が切れたため)。後退を出す長さの
     上限(MAX_REV_TOTAL_S)は止めた時間を数えないまま(中立止めをくり返しても延びない)
   長い中立止めのあとの中止の広げ(2026-10-27 fix3 直す 3 回目 r3-r3-safety-1・safety-2):
   ・段なしで止まっている時間を数えていない(前に処理したフレームでは走っていた・escape のあとの cooldown の中)ときと 'escape' の
     途中も、armed で、明けたフレームで止まっている(|eRPM| < STOP_ERPM)か中立止めの間の IMU が IMPACT_G 以上なら 'giveup'
     (出来事 hold_long)。前は、前の壁に当たった直後に 1 フレームも処理できずにカメラが 0.6〜2 秒止まると、明けてモデルの前進が
     壁を約 0.8 秒押したあと後退に進み 1.1〜1.4 m 下がった(当たりのあと 1 フレーム処理できたときは中止。位相だけで分かれた)。
     明けたフレームで走っていて衝撃も無ければ今まで通り(何もしない)
   ・中止にするとき明けたフレームで後ろへ動いていれば(eRPM > HOLD_HALT_ERPM = BACKING_ERPM = 300)、先に
     前進 ESCAPE_US を後ろへ動かなくなる(eRPM ≤ BACKING_ERPM)か HOLD_HALT_MAX_S(0.3 秒)まで出す(段 'halt'。種類は
     リカバリー)。そのあと中立で人を待つ。前はすぐ中立にして惰性で下がり、rec_hold_keep = 0・中立止めなしでは当たらない後ろの
     物に当たった('reverse' の途中で −1.2〜−1.4 m/s のとき、明けて 0.47〜1.21 m 下がった)
   hold_since = None(rec_hold_keep = 0・見張り OFF・人側・dry run。今の既定)は今までとビット単位で同じ('halt' にも入らない)
   反証 4 回目の直し(2026-10-27 fix5 係 safety。どれも rec_hold_keep = 1・見張り ON の日だけ効く。OFF はビット単位で同じ):
   ・s1 長い中立止めの間の衝撃(hold_g ≥ IMPACT_G)だけで中止にするのは、明けたフレームで前へ HOLD_IMPACT_FWD_ERPM
     (1000 eRPM = 0.5 m/s)より遅いか後ろへ動いているときだけ。速く走ったままならモデルに任せる(段差・こすりの 2.5G 以上は
     走行中に普通に出る。前は約 3 m/s のまま中止になり、中立・ハンドルまっすぐの惰性で進み、ToF ガードのブレーキも上書きした)。
     明けたとき止まっている(|eRPM| < STOP_ERPM)車の中止は前と同じ
   ・s2 発進の前進・発進の補助が出したフレームでも Recovery.t_last を進める(Assist.step の rec_keep)。前は区間の始めから
     None のままで、発進の前進の途中のカメラの止まりの間に壁に当たっても、長い中立止めのあとの中止が効かなかった
   ・s3 車両の値が来なかっただけの間(hold_unseen_s。前後のサンプルはどちらも後ろへ動いていない)は、下がり始めたあと
     ('reverse'・'coast' で、前から下がっている か 中立止めの間に後ろへ動いていたサンプルがある)だけ下がった時間に数える。
     'brake'・'wait'・'gap' から 'reverse'・'coast' へ進めるのも、後ろへ動いていたサンプルがあったときだけ。前は 0.25〜0.45 秒
     値が来ないだけで 1 cm も下がっていない車が前の壁へ escape を約 0.5 秒出した
   反証 6-1 回目の直し(2026-10-27 fix6 係 safety。どれも rec_hold_keep = 1・見張り ON の日だけ効く。OFF はビット単位で同じ):
   ・s3-a 車両の値が来なかった間に、値が古いまま(max_sensor_age 0.15 秒まで)出した後退側で車が下がり始めることがある
     (ESC の後退の入りぎわ・ESC が後退を待っていた 'brake')。drive はその間を、PCA9685 に書けた値から見積もった「ESC が後退で
     走らせていたかもしれない秒」(EscTrack)として hold_rev_s で渡す(hold_unseen_s は渡さない = 0)。_after_hold はそれを
     下がった時間に足す(どの段でも。ただし後退を HOLD_EST_MIN_LEFT_S = 0.2 秒(rev_s の下限)残すところまで)。足したら
     'brake'・'wait' → 'reverse'・'coast'、'gap' → 'coast' へ進める(後ろへ動いていたサンプルと同じ)。前は、明けたサンプルが
     300 eRPM 未満だと 1 秒も数えず、明けてから rev_s をまるごと下がり直した(値の止まりなしより最大 0.33 m 多く下がり、後ろ
     1.35〜1.6 m の物に当たった)。逆に、明けたサンプルが後ろへ動いていると値が来なかった間を全部(古い値で数えたフレームと
     二重に)数えて後退を早く切り、0.2〜0.4 m しか下がらずに前の壁へ escape を出した(fix3 r2 の数え方。どちらも同じ窓)
   ・s3-b 残した中立止めの間の出力は中立 = 惰性。'coast' の途中の中立止めは惰性の時間に数え(段の時間をずらさない)、
     中立止めのあとで後退を出し終えていたら('reverse' で下がった時間が rev_s 以上)、'coast' は中立止めの始めからとする。
     前は明けてから 'coast' の 0.25 秒をやり直し、keep 0 より最大 0.37 m 多く下がった
   反証 6-2 回目の直し(2026-10-27 fix6 係 safety r2。どれも rec_hold_keep = 1・見張り ON の日だけ効く。OFF はビット単位で同じ):
   ・s3-c 見積もり(hold_rev_s)は書けた値から見た「下がれたかもしれない」で、下がったかは分からない(EscTrack は多めに数える側。
     このESCが下がれるまでの中立の秒も分からない)。そこで見積もりは下がった時間(backed_s)に今まで通り足すが(後退を 0.2 秒残す
     ところまで)、足した分を est_s に別に持ち、
     - 見積もりだけでは段を進めない('brake'・'wait' → 'reverse'・'coast'、'gap' → 'coast' は後ろへ動いていたサンプルがあったとき
       だけ。見積もりだけなら 'gap' の中立も残す)
     - 'reverse' の下がり始めを待つやり直し(CHECK_S)は、見た後退(backed_s − est_s。BACKING_ERPM を超えたサンプルで数えた秒)が
       0 かで決める。やり直すときは見積もりも捨てる(後退側を CHECK_S 出しても下がらない = 見ていない間も下がっていなかった)
     見積もりが外れていた(ESC はまだ下がれなかった)ときは、そのぶん後退が短くなる(後退を 0.2 秒残すところまで。閉じたループの
     偽物の ESC で確かめた場面では、それで escape が前の壁を押すことは無かった)。見積もりを捨てる決まり(明けたあと後ろへ動くのを
     見なければ捨てる)は入れない: 見ていない間に本当に下がって止まった車と見分けられず、捨てると fix6 r1 の指摘の形(値の止まり
     なしより 0.3〜0.45 m 多く下がる)に戻るため
     前は、ブレーキのあとの 0.10 秒の中立止め(閉じたループの偽物の ESC はまだ下がれない)を EscTrack が中立 0.08 秒で「下がれる」
     とし、続く車両の値の止まりの見積もり 0.07 秒だけで 'brake' から 'gap' の中立を飛ばして 'reverse' へ進み、backed_s > 0 なので
     やり直しも効かず、1796 µs を 1.2 秒出しても 0 m のまま escape が前の壁を約 0.5 秒押した(反証 w2-safety-r62-safety-1)。
     最初の 'gap' 0.15 秒では下がれない ESC(10/10 のログで 11%)で、後退の入りぎわに車両の値が 1 回止まるだけでも、見積もりで
     やり直しが止まり同じことが起きた(w2-safety-r62-safety-2)
   反証 7-1 回目の直し(2026-10-27 fix7 係 safety。どれも見張り ON(back_guard = True)の日だけ効く。rec_hold_keep は問わない。
   見張りが OFF ならビット単位で同じ):
   ・s4-a 本物の eRPM は止まっている車でも 1 サンプルだけ跳ねる(ホールの縁の行き来。10/10・10/23 の記録で 1053〜1687 eRPM、
     0.1 秒に 2〜4 回続くこともある)。前進のあとの 'brake'・'wait' の後退側はこのESCではブレーキで、後退では走らない
     (モジュールの始め)。drive は書けた値の EscMayReverse(EscTrack より多めにみる。ブレーキのあと中立を 1 回でも書いたら下がれる・
     後退側を書いたときから後退)から「前に処理したフレームからいままでに ESC が後退で走らせていたかもしれないか」を esc_rev で
     渡す(rec_back_guard = 1 のとき。None = 分からない = 今までと同じ)。この手順の中で esc_rev がずっと
     False の間(= ESC はまだ一度も後退で走らせていない。esc_ran が False)の 'brake'・'wait'・'gap' では、後ろへ動いたサンプルを
     見張りに使わない(back_early で 'gap' の中立を飛ばさない・'gap' の惰性に数えない・後ろの当たりの「直前に後ろへ動いていた」に
     しない・中立止めのあとの数え・back_hold もしない = OFF と同じ。出来事 recovery_guard の back_no_esc を手順ごとに 1 回)。
     ESC が後退を待っていた 10/23 の形(ブレーキ側 → 止まって中立 → 'brake')は、'brake' を書いたときから後退とみるので今まで通り
     (10/23 の記録 20261023-205458-0002 を流し直しても back_early のフレーム・出す値は同じ)。やり直しの 'gap' など、この手順で
     一度後退で走らせたかもしれないあとは今まで通り数える
   ・s4-b 'reverse' の下がり始めを待つやり直し(CHECK_S)は、'reverse' の中で見た後退(seen_rev_s: 'reverse' に入ったフレームの
     ほかのフレームで BACKING_ERPM を超えたサンプル・'reverse' の中立止めの間に後ろへ動いていたサンプル)が 0 かで決める。
     'gap' の惰性の数え・back_early で入ったフレーム・'brake'・'wait'・'gap' の中立止めの間のサンプルは、下がった時間(backed_s)
     には今まで通り数えるが、やり直しの決めには使わない(どれも後退側を出す前のサンプル)。'gap' を飛ばして入った 'reverse'
     (back_early・back_hold)で後退を見なければ、やり直し(+50 µs)にはせず、ふつうの 'gap'(GAP_S)からやり直す(出来事
     early_none。下がった時間・見積もりも捨てる)
     前は、'brake' の間の 1 サンプルの跳ね(301〜1322 eRPM)だけで back_early が 'gap' を飛ばして 'reverse' へ進み、
     下がった時間 > 0 なのでやり直しも効かず、偽物の ESC はブレーキのまま 1.23 秒 0 m、escape が前の壁を 0.5 秒押した
     (OFF は 1.13 m 下がる。反証 w1-safety-rv7-safety-r1-1)
   反証 7-2 回目の直し(2026-10-27 fix7 係 safety r2。見張り ON(back_guard = True)の日だけ効く。rec_hold_keep は問わない。
   見張りが OFF ならビット単位で同じ):
   ・s4-c 'gap' を飛ばして入った 'reverse'(back_early・back_hold = skip_gap)の early_none は、「本当に下がっている」を見たかで
     決める。本当に下がっている = 'reverse' に入ったフレームのほかのフレームで、続けて SURE_BACK_S(0.2 秒)後ろへ動いていた
     (BACKING_ERPM を超えたフレームが途切れずに続いた。残した中立止めがあればそこから数え直す)。CHECK_S のあと、まだ本当に
     下がっているを見ていなくて、このフレームで後ろへ動いていなければ early_none('gap' から。下がった時間・見積もり・見た後退を
     捨てる)。このフレームで後ろへ動いていれば決めを次のフレームへ延ばす(下がり始めた車をやり直さない)。一度本当に下がっているを
     見たら、その 'reverse' ではやり直さない。前は 'reverse' の中で後ろへ動いたサンプルが 1 つでもあれば early_none もやり直しも
     しなかった。止まっている車の eRPM の跳ねがフレームから続けて見えるのは、本物の記録 308 本で最長 0.11 秒(人側。ほかは 0.06 秒
     以下)。本当に後退で走っている車は、後退側を出している間(下がった時間が rev_s に届くまで)ずっと後ろへ動く。'gap' を飛ばさずに
     入った 'reverse' のやり直しは今まで通り(s4-b)
     前は、ESC がまだ下がれないのに 'brake' の中の約 0.03 秒の中立止め(rec_hold_keep で手順を残す)で EscMayReverse が「下がれる」
     とみて(中立を 1 回でも書いたら下がれる)この手順の s4-a が外れ、跳ね 1 つ目で back_early、'reverse' の中の跳ね 2 つ目で s4-b の
     early_none もやり直しも止まり、0 m のまま後退側を約 1.3 秒書いて escape が前の壁を 0.5 秒押した(OFF は 1.27 m 下がる。反証
     w2-safety-rv7-safety-r2-1)。EscMayReverse は変えない(本物の ESC は人の操作で中立 0.05〜0.07 秒でも後退に入った。10/10 の
     表の 23 回のうち 4 回。下がれないとみると、本当に下がっている 'brake' の後ろへの動きを見張りが数えない = OFF の形になる)

2. 後退の受け付け(EscReverse) — リカバリーをOFFにしたときだけ使う
   モデル自身の後退の指令を、止まってから ブレーキ → 中立 → 後退 の順に出す(強さ・長さに上限)。
   リカバリーがONのときは、止まっている間のモデルの後退は中立にして、下がるのはリカバリーに任せる
   (2026-10-10、AIが自分で長く下がって後ろにぶつかった例があるため、下がる量を決まった範囲にする)。

3. 発進の補助(StartKick)・発進の前進(LaunchBoost、2026-10-26。既定 OFF)
   発進の前進: CH3 を AI 側にした最初のフレームで、止まっていて前が空いていれば、前進を少なくとも launch_us(中立から)
   出す。速さ・時間・ヨー・舵・衝撃・前が近い・CH3・中立で止めた・車両の値 のどれかで終わり、同じ AI の区間ではもう
   出さない(StartKick も出さない)。仕様は jetformer_reports/fix_20261026/protocol_v3.md 14 章。
   始めなかった区間(前が近い・ToF が無い など)では、仕様どおり今まで通り StartKick が出る(protocol_v3.json launch_boost.start)。
   launch_no_kick(drive の同じ名前、2026-10-26 fix3 safety1-2。既定 OFF。OFF は今までと同じ)が ON なら、発進の前進が
   「出さない」と決めた区間では StartKick も出さない(launch_boost = 0 のときは決めないので効かない)。「出さない」になった
   区間は、途中で launch_no_kick を 0 にしても StartKick を出さない(CH3 を人側にするまで。fix3 r1-rs1-2)。
   StartKick の 2 秒の窓は、最初に StartKick を見たフレームから数える。走行中に start_kick を 0 → 1 にすると区間の途中で
   新しい窓が開き、リカバリーより先に見るので中止・後退を上書きする。そこで drive は走行中の start_kick の 0 → 1 を断る。

時刻は秒(time.monotonic())で渡す。どちらも人が操縦中(select=False)なら何もしない。
"""

import collections
import math
from typing import NamedTuple, Tuple

import numpy as np

ASSIST_NONE, ASSIST_ESC, ASSIST_RECOVERY, ASSIST_GIVEUP, ASSIST_START, ASSIST_LAUNCH = 0, 1, 2, 3, 4, 5
ASSIST_NAMES = {ASSIST_NONE: '', ASSIST_ESC: '後退の受け付け', ASSIST_RECOVERY: 'リカバリー', ASSIST_GIVEUP: 'リカバリー中止',
                ASSIST_START: '発進の補助', ASSIST_LAUNCH: '発進の前進'}

DEFAULT_REV_US = 300.0       # 後退の強さ(中立から)。人は約+210µs。2026-10-10 の +180µs は弱かった
DEFAULT_REV_S = 0.4          # 下がり始めてから後退を出し続ける時間(その後は惰性)
GAP_OFFSET_US = 10.0         # 中立を挟むときの値(中立 1478〜1538µs の帯の中で、わずかに後退側)
STOPPED_ERPM = -150          # これより遅く前へ転がっている(ほぼ止まった)ら、中立 → 後退 が効く
BACKING_ERPM = 300           # これより速く後ろへ動いていたら「下がっている」
CHECK_S = 0.35               # 後退を出してから、この時間で下がり始めなければやり直す(人は0.08秒で動く)
HOLD_EST_MIN_LEFT_S = 0.2    # 見ていない間の後退の見積もり(hold_rev_s)で後退を短くするのは、これだけ残すところまで(rev_s の下限。
                             # 2026-10-27 fix6 safety r1。見積もりの後退は中立止めで切られて遅く、そのまま差し引くと短い rev_s で
                             # 下がる距離が足りず escape が前の壁に当たる)
SURE_BACK_S = 0.2            # 'gap' を飛ばして入った 'reverse' で「本当に下がっている」とみる、フレームで続けて後ろへ動いていた秒
                             # (2026-10-27 fix7 r2 s4-c。止まっている車の eRPM の跳ねは本物の記録で続けて最長 0.11 秒)


class Recovery(object):
    STOP_ERPM = 300
    ARM_ERPM = 1000          # 一度この速さで動いてから止まったときだけ検出する(発進前・スタート直後は対象外)
    STUCK_S = 0.8
    STUCK_AFTER_IMPACT_S = 0.5
    IMPACT_G = 2.5
    BRAKE_S = 0.25
    WAIT_STOP_S = 1.0        # ブレーキで止まるのを待つ上限(止まらなければもう一度ブレーキ)
    GAP_S = 0.15
    RETRY_GAP_S = 0.3
    RETRY_EXTRA_US = 50.0
    MAX_RETRIES = 2
    MAX_REV_TOTAL_S = 1.2    # 後退を出す長さの上限(下がっていなくても)
    REV_STEER_US = 150.0     # 後退中のハンドル: 止まる前と反対へ少しだけ(人はほぼ真ん中)
    COAST_S = 0.25
    ESCAPE_US = 130.0        # 前進(中立から)
    ESCAPE_S = 0.5
    ESCAPE_STEER_FRAC = 0.85 # 前進のハンドル: 止まる前に曲がっていた向きへ
    COOLDOWN_S = 1.0
    GIVEUP_N = 3
    GIVEUP_WINDOW_S = 20.0
    MAX_REV_US = 500.0
    BACK_RECENT_S = 0.3      # 後ろの見張り: 衝撃の直前これだけの間に後ろへ動いていたら「後ろにぶつかった」
    HOLD_KEEP_MAX_S = 0.5    # 残した中立止めのうち、続きから進めてよい長さ(前に処理したフレームから。より長ければ後退側は中止。r2-r2-rs-2)
    HOLD_HALT_ERPM = BACKING_ERPM    # 長い中立止めで中止にするとき、これより速く後ろへ動いていたら先に前進で止める(r3-r3-safety-2)
    HOLD_HALT_MAX_S = 0.3    # その前進(ESCAPE_US)を出す長さの上限(後ろへ動かなくなったらそこで中立)
    HOLD_IMPACT_FWD_ERPM = 1000      # 長い中立止めの間の衝撃(hold_g)だけで中止にするのは、明けたとき前へこれより遅い(0.5 m/s)か
                                     # 後ろへ動いているときだけ(速く走ったままならモデルに任せる。2026-10-27 fix5 s1)

    def __init__(self, ref, steer_range=500.0):
        self.center, self.n = float(ref[0]), float(ref[1])
        self.range = float(steer_range)
        self.count = 0
        self.history = collections.deque(maxlen=60)      # (t, モデルのステア)
        self.starts = collections.deque(maxlen=self.GIVEUP_N)
        self.reset()

    def reset(self):
        self.phase = None            # None / brake / wait / gap / reverse / coast / escape / halt(中止の前の止め。r3-r3-safety-2) / giveup
        self.t_phase = 0.0
        self.t_check = 0.0           # 'reverse' の下がり始めを待つ時間(CHECK_S)の起点。段を変えると t_phase と同じ(r2-r2-rs-1)
        self.slow_since = None
        self.impact_t = None
        self.impact_moving = False
        self.cooldown_until = 0.0
        self.armed = False
        self.retries = 0
        self.backed_s = 0.0
        self.est_s = 0.0             # backed_s のうち、見ていない間の後退の見積もり(hold_rev_s)で足した秒(2026-10-27 fix6 r2 s3-c)
        self.seen_rev_s = 0.0        # 'reverse' の中で見た後退の秒(入ったフレームは除く。見張り ON のやり直しの決め。fix7 s4-b)
        self.skip_gap = False        # この 'reverse' は 'gap' を飛ばして入った(back_early・back_hold。fix7 s4-b)
        self.noesc_noted = False     # この手順で back_no_esc を記録した(fix7 s4-a)
        self.q_early = False         # 案 qA・qC: この手順で back_early を記録した
        self.esc_ran = False         # この手順で ESC が後退で走らせていたかもしれない(esc_rev が False でなかった)フレームがあった(fix7 s4-a)
        self.run_t0 = None           # 'gap' を飛ばして入った 'reverse' で、続けて後ろへ動いているフレームの始めの時刻(fix7 r2 s4-c)
        self.back_sure = False       # その 'reverse' で SURE_BACK_S 続けて後ろへ動いた(本当に下がっている。fix7 r2 s4-c)
        self.t_last = None
        self.turn = 0.0              # 止まる前に曲がっていた向き(+1 左 / -1 右 / 0 まっすぐ)
        self.starts.clear()          # 人が交代して直したら、あきらめの数え直し
        self.back_t = None           # 後ろの見張り: この手順の中で最後に後ろへ動いていた時刻
        self.guard_event = None      # 後ろの見張りが手順を変えた (何, 段, 下がった秒)。drive.py が記録に書いて None に戻す

    @property
    def active(self):
        return self.phase is not None

    def _go(self, phase, now):
        self.phase, self.t_phase = phase, now
        self.t_check = now

    def _start(self, now):
        self.count += 1
        self.starts.append(now)
        if len(self.starts) >= self.GIVEUP_N and now - self.starts[0] <= self.GIVEUP_WINDOW_S:
            self._go('giveup', now)
            return
        before = [s for t, s in self.history if now - t <= 1.5 and t <= (self.slow_since or now)]
        dev = (float(np.median(before)) if before else self.center) - self.center
        self.turn = 0.0 if abs(dev) < 0.2 * self.range else float(np.sign(dev))
        self.retries = 0
        self.backed_s = 0.0
        self.est_s = 0.0
        self.seen_rev_s = 0.0
        self.skip_gap = False
        self.noesc_noted = False
        self.q_early = False
        self.esc_ran = False
        self.run_t0, self.back_sure = None, False
        self.back_t = None
        self._go('brake', now)

    def _detect(self, model_us, erpm, now, impact_g):
        self.history.append((now, model_us[0]))
        if impact_g >= self.IMPACT_G:
            self.impact_t, self.impact_moving = now, erpm < -500 or self.impact_moving
        if self.impact_t is not None and now - self.impact_t > 1.2:
            self.impact_t, self.impact_moving = None, False
        if abs(erpm) >= self.ARM_ERPM:
            self.armed = True
        if not self.armed or now < self.cooldown_until:
            self.slow_since = None
            return False
        if -self.STOP_ERPM < erpm < BACKING_ERPM:          # 前へも後ろへも進んでいない
            if self.slow_since is None:
                self.slow_since = now
            need = self.STUCK_AFTER_IMPACT_S if (self.impact_t is not None and self.impact_moving) else self.STUCK_S
            return now - self.slow_since >= need
        self.slow_since = None
        return False

    def _guard(self, erpm, now, dt, impact_g, rev_s, esc_rev=None):
        """後ろの見張り(back_guard = True のときだけ呼ぶ)。手順を変えたら戻り値 (出す値, 種類)、変えなければ None
        (段を進めただけのときも None。あとの段の処理がそのまま続ける)。esc_rev = drive の EscMayReverse が、前に処理したフレームから
        いままでに ESC が後退で走らせていたかもしれないと見たか(None = 分からない = 今までと同じ。False で、この手順でまだ一度も
        後退で走らせていない(esc_ran が False)の 'brake'・'wait'・'gap' は後ろへ動いたサンプルを使わない。fix7 s4-a)"""
        backing = erpm > BACKING_ERPM
        if backing and esc_rev is False and not self.esc_ran and self.phase in ('brake', 'wait', 'gap'):
            # fix7 s4-a: この手順でまだ ESC が後退で走らせていない(前進のあとのブレーキ・そのあとの中立)間に後ろへ動いたサンプル
            # = eRPM の跳ね・たわみの戻り。見張りに使わない(前は 1 サンプルの跳ねで 'gap' の中立を飛ばし、ESC はブレーキのまま 0 m で
            # escape が前の壁を押した。'gap' の跳ねは後退を短くし、やり直しも止めた)
            backing = False
            if not self.noesc_noted:
                self.noesc_noted = True
                self.guard_event = ('back_no_esc', self.phase, round(self.backed_s, 3))
        if backing:
            self.back_t = now
        if impact_g >= self.IMPACT_G and self.back_t is not None and now - self.back_t <= self.BACK_RECENT_S:
            self.guard_event = ('rear_impact', self.phase, round(self.backed_s, 3))     # 後ろにぶつかった: すぐ前へ
            self._go('escape', now)
            return self._escape(now)
        if not backing:
            return None
        if self.phase in ('brake', 'wait'):            # 前のブレーキ側で ESC が後退を待っていた: もう下がっている
            # 案 qC(fix8 r1): 段は飛ばさず(OFF と同じ 'brake' → 'wait' → 'gap' → 'reverse')、下がった時間に数えるだけ。'gap'(中立)の
            # 惰性はこの手順では数えない(後退側を書いていた間の下がった時間で rev_s を数える = 今の ON と同じ長さの後退側)
            if not self.q_early:
                self.q_early = True
                self.guard_event = ('back_early', self.phase, round(self.backed_s, 3))
            self.backed_s += dt
            if self.backed_s >= rev_s:
                self.guard_event = ('back_enough', self.phase, round(self.backed_s, 3))
                self._go('coast', now)
        elif self.phase == 'gap' and self.q_early:      # 案 qC: back_early のあとの 'gap' の惰性は数えない
            pass
        elif self.phase == 'gap':                      # 中立の惰性で下がっている分も数える(やり直しの決めには使わない。fix7 s4-b)
            self.backed_s += dt
            if self.backed_s >= rev_s:
                self.guard_event = ('back_enough', self.phase, round(self.backed_s, 3))
                self._go('coast', now)
        return None

    HOLD_HIT_PHASES = ('brake', 'wait', 'gap', 'reverse', 'coast')     # 後ろの見張りと同じ段(後退側・下がる惰性)

    def _hold_giveup(self, p, erpm, gap, now):
        """長い中立止めのあとの中止(出来事 hold_long。4 つ目 = 見ていない秒)。明けたフレームで後ろへ動いていれば
        (erpm > HOLD_HALT_ERPM)、先に前進 ESCAPE_US で後ろへの惰性を止める('halt'。後ろへ動かなくなるか HOLD_HALT_MAX_S まで。
        そのあと中立で人を待つ)。2026-10-27 fix3 直す 3 回目 r3-r3-safety-2: 前はすぐ中立にして、ブレーキも見張りも無いまま
        惰性で下がった(rec_hold_keep = 0 ならモデルの前進が、中立止めなしなら 'coast' → 'escape' の前進が止めていた)"""
        self.guard_event = ('hold_long', p, round(self.backed_s, 3), round(gap, 3))
        if erpm > self.HOLD_HALT_ERPM:
            self._go('halt', now)
            return (self.center, self.n - self.ESCAPE_US), ASSIST_RECOVERY
        self._go('giveup', now)
        return (self.center, self.n), ASSIST_GIVEUP

    def _after_hold(self, gap, hold_s, hold_g, hold_back_s, rev_s, now, erpm=0.0, hold_unseen_s=0.0, hold_rev_s=0.0,
                    esc_rev=None):
        """手順を残した中立止めのあとの最初のフレーム(2026-10-27 fix3 r2-r2-rs-1・rs-2。モジュールの説明 1 の終わり)。
        gap = 前に処理したフレームからの秒(見ていない時間)。hold_s = 中立を出していた秒(hold_since から)。erpm = 明けたフレームの
        車両の値。hold_unseen_s = hold_back_s のうち、車両の値が来なかっただけの間(前後のサンプルはどちらも後ろへ動いていない)の秒
        (2026-10-27 fix5 s3。既定 0 = 全部を後ろへ動いていたサンプルの秒として今までと同じ)。hold_rev_s = 車両の値が来なかった間に
        ESC が後退で走らせていたかもしれない秒(drive が書いた値から見積もる。hold_back_s とは別。2026-10-27 fix6 s3-a。既定 0)。
        見積もりは下がった時間に足すが、足した分は est_s に別に持ち、見積もりだけでは段を進めない(fix6 r2 s3-c)。
        esc_rev = step の同じ名前(False で esc_ran も False の 'brake'・'wait'・'gap' は後ろへ動いていたサンプルを使わない。fix7 s4-a)。
        手順を変えて出す値があれば (出す値, 種類)、なければ None(段を進めただけのときも None。あとの段の処理がそのまま続ける)"""
        long_hold = gap > self.HOLD_KEEP_MAX_S
        p = self.phase
        if p is None and self.slow_since is not None:   # 段なし: 止まっている時間を数えている途中なら
            if long_hold:                               # 長く見ていない: 明けてすぐ stuck と見て下がらない。中立で人を待つ
                return self._hold_giveup(None, erpm, gap, now)
            self.slow_since += hold_s                   # 中立を出していた間は止まっている時間に数えない
            return None
        # 2026-10-27 fix3 直す 3 回目 r3-r3-safety-1: 段なしで止まっている時間を数えていない(前に処理したフレームでは走っていた・
        # escape のあとの cooldown の中)ときと 'escape' の途中も、長く見ていない間に止まった・当たったなら中止にする(armed で、
        # 明けたフレームで止まっている か 中立止めの間の IMU が IMPACT_G 以上)。前は、当たりのあと 1 フレームも処理できずにカメラが
        # 止まると、明けてモデルの前進が壁を押し、約 0.8 秒あとに後退へ進んだ(当たりとフレームの位相だけで中止か後退かが分かれた)
        # 2026-10-27 fix5 s1: 衝撃だけで中止にするのは、明けたフレームで前へ HOLD_IMPACT_FWD_ERPM(0.5 m/s)より遅いか後ろへ
        # 動いている(止まりかけ・跳ね返った)ときだけ。前は約 3 m/s で走ったままの車も、走行中に普通に出る 2.5G 以上(段差・こすり)
        # だけで中止になり、中立・ハンドルまっすぐの惰性で 0.4〜3 m 進み、ToF ガードのブレーキも中立で上書きした
        if long_hold and p in (None, 'escape') and self.armed and (
                -self.STOP_ERPM < erpm < BACKING_ERPM
                or (hold_g >= self.IMPACT_G and erpm > -self.HOLD_IMPACT_FWD_ERPM)):
            return self._hold_giveup(p, erpm, gap, now)
        if p is None:
            return None
        # 中立止めの間に後ろへ動いていた秒(drive が車両の値から)。2026-10-27 fix5 s3: 車両の値が来なかっただけの間(unseen)を
        # 下がった時間に数えるのは、この手順で実際に下がり始めたあと('reverse'・'coast' で、前から下がっている か この中立止めの間に
        # 後ろへ動いていたサンプルがある)だけ = 後退を短くする側。'brake'・'wait'・'gap'(下がる前)は数えず、'reverse'・'coast' へ
        # 進めるのも実際に後ろへ動いていたサンプルがあったときだけ。前は 0.25〜0.45 秒値が来ないだけで下がった時間が rev_s に届き、
        # 1 cm も下がっていない車が前の壁へ escape(前進 130 µs)を約 0.5 秒出した
        hb = max(float(hold_back_s), 0.0)
        unseen = min(max(float(hold_unseen_s), 0.0), hb)
        seen = hb - unseen
        est = max(float(hold_rev_s), 0.0)
        if esc_rev is False and not self.esc_ran and p in ('brake', 'wait', 'gap') and hb > 0.0:
            # fix7 s4-a: この手順でまだ ESC が後退で走らせていない(前進のあとのブレーキ・中立・中立止めの中立)。後ろへ動いていた
            # サンプルは eRPM の跳ね・たわみの戻りとして使わない(下がった時間に数えない・back_hold で 'gap' を飛ばさない)
            hb = unseen = seen = 0.0
            if not self.noesc_noted:
                self.noesc_noted = True
                self.guard_event = ('back_no_esc', p, round(self.backed_s, 3))
        if p in self.HOLD_HIT_PHASES:
            count_unseen = p in ('reverse', 'coast') and (self.backed_s > 0.0 or seen > 0.0)     # 下がり始めたあと
            obs = hb if count_unseen else seen      # hold_unseen_s = 0 なら どちらも hold_back_s(今までと同じ)
            self.backed_s += obs
            if p == 'reverse':
                self.seen_rev_s += obs      # fix7 s4-b: 'reverse' の中で見た後退(やり直しの決め)
            if est > 0.0:       # fix6 s3-a: 見ていない間の後退の見積もり。後退を HOLD_EST_MIN_LEFT_S 残すところまで足す
                add = min(est, max(0.0, rev_s - HOLD_EST_MIN_LEFT_S - self.backed_s))
                self.backed_s += add
                self.est_s += add       # fix6 r2 s3-c: 見積もりで足した分('reverse' のやり直しは見た後退だけで決める)
        if long_hold:
            if p in self.HOLD_HIT_PHASES and hold_g < self.IMPACT_G:     # 後退側・下がる惰性の段: 続けずに中立で人を待つ
                return self._hold_giveup(p, erpm, gap, now)
            return None                                 # 'escape'(前進)は時間で終わる。後ろの衝撃は rear_impact_hold で前へ
        if p != 'coast':                                # fix6 s3-b: 'coast' は中立止めの間も惰性(段の時間をずらさない)
            self.t_phase += hold_s                      # 段の時間は中立を出していた間は進めない
        if p == 'reverse':
            self.t_check = now                          # 下がり始めを待つ時間は明けたフレームから(中立で後退が切れた)
            if self.backed_s >= rev_s:                  # fix6 s3-b: 後退は中立止めの始めで出し終えていた。惰性はそこから
                self._go('coast', now)
                self.t_phase = now - hold_s
        # 後ろへ動いていたサンプルがあった: 段を進める(fix6 r2 s3-c: 見積もりだけでは進めない。見積もりは「下がれたかもしれない」
        # だけで、前は ESC がまだ下がれない 'brake' から 'gap' の中立を飛ばして 'reverse' へ進み、0 m のまま escape が前の壁を押した)
        if seen > 0.0:
            if p in ('brake', 'wait'):                  # 見張りと同じ: もう下がっている
                self.guard_event = ('back_hold', p, round(self.backed_s, 3))
                self._go('coast' if self.backed_s >= rev_s else 'reverse', now)
                if self.phase == 'coast':
                    self.t_phase = now - hold_s         # fix6 s3-b: 惰性は中立止めの始めから
                else:
                    self.skip_gap = True                # fix7 s4-b: 'gap' を飛ばした(後退を見なければ 'gap' からやり直す)
            elif p == 'gap' and self.backed_s >= rev_s:
                self.guard_event = ('back_hold', p, round(self.backed_s, 3))
                self._go('coast', now)
                self.t_phase = now - hold_s             # fix6 s3-b
        return None

    def step(self, model_us, erpm, now, impact_g=0.0, rev_us=DEFAULT_REV_US, rev_s=DEFAULT_REV_S, back_guard=False,
             hold_g=0.0, hold_since=None, hold_back_s=0.0, hold_unseen_s=0.0, hold_rev_s=0.0, esc_rev=None):
        """戻り値 (出す値 (steer, throttle) または None(モデルのまま), 補助の種類)。
        back_guard = 後ろの見張り(既定 False = 今までと同じ。モジュールの説明 1 の終わり)。
        hold_g = 残した手順の中立止めの間(前に処理したフレームのあと)に来た IMU の水平 G の最大(drive が rec_hold_keep で
        リカバリーを残した中立止めのあとの最初のフレームだけ渡す。既定 0 = 今までと同じ。2026-10-26 fix3 r1-rs1-1)。
        hold_since = このフレームの前に手順を残した中立止めがあったとき、その中立を書き始めた時刻(now と同じ時計。drive が
        中立止めのあとの最初のフレームだけ渡す。既定 None = 今までと同じ)。hold_back_s = その間(前に処理したフレームから
        このフレームまで)に後ろへ動いていた秒(2026-10-27 fix3 r2-r2-rs-1・rs-2)。hold_unseen_s = hold_back_s のうち、
        車両の値が来なかっただけの間の秒(2026-10-27 fix5 s3。既定 0 = 今までと同じ。_after_hold)。hold_rev_s = 車両の値が来なかった
        間に ESC が後退で走らせていたかもしれない秒(drive の見積もり。2026-10-27 fix6 s3-a。既定 0 = 今までと同じ。_after_hold)。
        esc_rev = 前に処理したフレームからいままでに、ESC が後退で走らせていたかもしれないか(drive が書けた値の EscMayReverse から。
        rec_back_guard = 1 のとき。既定 None = 分からない = 今までと同じ)。見張り ON で、この手順でまだ一度も True・None に
        なっていない間の 'brake'・'wait'・'gap' は、後ろへ動いたサンプルを見張りに使わない(2026-10-27 fix7 s4-a。モジュールの説明 1 の
        終わり)"""
        gap = None if self.t_last is None else max(now - self.t_last, 0.0)
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.2)
        self.t_last = now
        if self.phase == 'giveup':
            return (self.center, self.n), ASSIST_GIVEUP
        if self.phase == 'halt':        # 長い中立止めのあとの中止の前: 後ろへの惰性を前進で止める(r3-r3-safety-2。_hold_giveup)
            if erpm > BACKING_ERPM and now - self.t_phase < self.HOLD_HALT_MAX_S:
                return (self.center, self.n - self.ESCAPE_US), ASSIST_RECOVERY
            self._go('giveup', now)     # 後ろへ動かなくなった・上限の時間: 中立で人を待つ(CH3 を人側にすると解ける)
            return (self.center, self.n), ASSIST_GIVEUP
        phase0 = self.phase             # 中立止めの前(このフレームの前)の段
        if esc_rev is not False:
            self.esc_ran = True         # fix7 s4-a: この手順で ESC が後退で走らせていたかもしれない(None = 分からない も)
        if hold_since is not None and gap is not None:
            dt = 0.0                    # 中立止めの間の後退は hold_back_s で数えた(このフレームの dt は足さない)
            self.run_t0 = None          # fix7 r2 s4-c: 続けて後ろへ動いていたかは、中立止めのあとのフレームから数え直す
            hold_s = min(max(now - float(hold_since), 0.0), gap)
            r = self._after_hold(gap, hold_s, hold_g, hold_back_s, rev_s, now, erpm, hold_unseen_s, hold_rev_s, esc_rev)
            if r is not None:
                return r
        if self.phase is None:
            if not self._detect(model_us, erpm, now, impact_g):
                return None, ASSIST_NONE
            self._start(now)
            if self.phase == 'giveup':
                return (self.center, self.n), ASSIST_GIVEUP
        rev = min(rev_us + self.RETRY_EXTRA_US * self.retries, self.MAX_REV_US)
        rev_steer = self.center - self.turn * self.REV_STEER_US
        if back_guard and self.phase in ('brake', 'wait', 'gap', 'reverse', 'coast'):
            g = self._guard(erpm, now, dt, impact_g, rev_s, esc_rev)
            if g is not None:
                return g
        if hold_g >= self.IMPACT_G and phase0 in self.HOLD_HIT_PHASES and self.phase in self.HOLD_HIT_PHASES:
            # 残した手順の中立止めの間に衝撃: 見ていない間に後ろへ当たったとして、後退を続けずすぐ前へ(rear_impact と同じ)。
            # 中立止めの間は step が呼ばれず、impact_g は処理したフレームの直近 0.15 秒の IMU しか見ない。back_t(最後に後ろへ
            # 動いていた時刻)も中立止めの間は進まないので、見張りの「0.3 秒の内」の決まりでも見えないことがある。見張りが
            # 見つけたとき(上の rear_impact)はそちらが先。中立止めの前に手順が始まっていなかった(phase0 = None)ときは見ない
            # (このフレームで始めた手順の衝撃は、前に当たった前の物かもしれない)
            self.guard_event = ('rear_impact_hold', self.phase, round(self.backed_s, 3))
            self._go('escape', now)
            return self._escape(now)
        el = now - self.t_phase
        if self.phase == 'brake':                       # 前進の直後の後退側は必ずブレーキ(捨てる1回目)
            if el < self.BRAKE_S:
                return (self.center, self.n + rev), ASSIST_RECOVERY
            self._go('wait', now)
            el = 0.0
        if self.phase == 'wait':                        # 前へ転がっている間は後退しないので、止まるのを待つ
            if erpm < STOPPED_ERPM:
                if el < self.WAIT_STOP_S:
                    return (self.center, self.n + rev), ASSIST_RECOVERY
                self._go('brake', now)
                return (self.center, self.n + rev), ASSIST_RECOVERY
            self._go('gap', now)
            el = 0.0
        if self.phase == 'gap':
            gap_s = self.RETRY_GAP_S if self.retries else self.GAP_S
            if el < gap_s:
                return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
            self._go('reverse', now)
            el = 0.0
        if self.phase == 'reverse':
            if impact_g >= self.IMPACT_G and el > 0.1:     # 後ろにぶつかった: すぐ前へ
                self._go('escape', now)
                return self._escape(now)
            if erpm > BACKING_ERPM:
                self.backed_s += dt
                if phase0 == 'reverse':         # fix7 s4-b: 'reverse' に入ったフレームのサンプルは後退側を出す前(やり直しの決めに使わない)
                    self.seen_rev_s += dt
            if back_guard and self.skip_gap:
                # fix7 r2 s4-c: 'gap' を飛ばして入った 'reverse' は、続けて SURE_BACK_S 後ろへ動いたら「本当に下がっている」
                # ('reverse' に入ったフレームは除く。途切れたら・中立止めのあとは数え直す)
                if phase0 == 'reverse' and erpm > BACKING_ERPM:
                    if self.run_t0 is None:
                        self.run_t0 = now
                    elif now - self.run_t0 >= SURE_BACK_S:
                        self.back_sure = True
                else:
                    self.run_t0 = None
                # まだ本当に下がっているを見ていなくて、このフレームで後ろへ動いていなければ下がっていない(跳ね・たわみの戻りだけ。
                # 前は 'reverse' の中の跳ね 1 つで early_none もやり直しも止まった。このフレームで後ろへ動いていれば次のフレームで決める)
                none_yet = not self.back_sure and erpm <= BACKING_ERPM
            else:
                # 下がり始めない: ESCが後退に入っていない(t_check は段を変えたとき t_phase と同じ。残した中立止めのあとは明けたフレーム)。
                # 見るのは見た後退だけ(fix6 r2 s3-c: 見ていない間の後退の見積もり est_s は除く。est_s = 0 なら今までと同じ)。
                # 見張り ON は 'reverse' の中で見た後退だけ(fix7 s4-b: 'gap' の惰性の数え・back_early で入ったフレーム・'reverse' の前の
                # 中立止めのサンプルは除く。前は 1 サンプルの跳ねで backed_s > 0 になり、ESC がブレーキのままでもやり直さなかった)
                none_yet = (self.seen_rev_s == 0.0) if back_guard else (self.backed_s - self.est_s == 0.0)
            if none_yet and now - self.t_check > CHECK_S:
                if back_guard and self.skip_gap:
                    # fix7 s4-b: 'gap' を飛ばして入った 'reverse'(back_early・back_hold)で下がらない: 見た後ろへの動きは跳ね・たわみの
                    # 戻りだった(ESC はブレーキのまま)。やり直し(+50 µs)にはせず、ふつうの 'gap' から(下がった時間・見積もりも捨てる)
                    self.guard_event = ('early_none', 'reverse', round(self.backed_s, 3))
                    self.skip_gap = False
                    self.run_t0, self.back_sure = None, False
                    self.backed_s = self.est_s = self.seen_rev_s = 0.0
                    self._go('gap', now)
                    return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
                if self.retries < self.MAX_RETRIES:
                    self.retries += 1
                    self.backed_s = self.est_s = 0.0       # 後退側を出しても下がらない: 見積もりの後退も無かった(fix6 r2)
                    self.seen_rev_s = 0.0
                    self.skip_gap = False
                    self._go('gap', now)
                    return (rev_steer, self.n + GAP_OFFSET_US), ASSIST_RECOVERY
                self._go('escape', now)                    # やり直しても下がらない: 前進で抜けてみる
                return self._escape(now)
            if self.backed_s < rev_s and el < self.MAX_REV_TOTAL_S:
                return (rev_steer, self.n + rev), ASSIST_RECOVERY
            self._go('coast', now)
            el = 0.0
        if self.phase == 'coast':                       # 中立で惰性(人は後退をすぐ離している)
            if el < self.COAST_S and erpm > 1000:
                return (rev_steer, self.n), ASSIST_RECOVERY
            self._go('escape', now)
        if self.phase == 'escape':
            return self._escape(now)
        return None, ASSIST_NONE

    def _escape(self, now):
        """止まる前に曲がっていた向きへハンドルを切って前進(後退から前進はすぐ切り替わる)→ 終わったらモデルへ"""
        if now - self.t_phase < self.ESCAPE_S:
            steer = self.center + self.turn * self.ESCAPE_STEER_FRAC * self.range
            return (steer, self.n - self.ESCAPE_US), ASSIST_RECOVERY
        self.phase = None
        self.slow_since = None
        self.impact_t, self.impact_moving = None, False
        self.cooldown_until = now + self.COOLDOWN_S
        return None, ASSIST_NONE


class EscReverse(object):
    """リカバリーをOFFにしたとき、モデル自身の後退の指令を ESC が受け付けるように出す"""
    DEADBAND_US = 40.0       # 中立から これ以上 大きい = 後退側(ブレーキ)
    BRAKE_S = 0.2
    GAP_S = 0.15
    RETRY_GAP_S = 0.3
    MAX_RETRIES = 2
    REV_MAX_S = 1.0          # 実際に下がり続けてよい長さ(その後 REST_S は中立)
    REST_S = 0.5

    def __init__(self, neutral):
        self.n = float(neutral)
        self.reset()

    def reset(self):
        self.state = 'idle'          # idle / brake / wait / gap / reverse / rest
        self.t_state = 0.0
        self.retries = 0
        self.backing_s = 0.0
        self.t_last = None

    def _go(self, state, now):
        self.state, self.t_state = state, now

    def step(self, thr, erpm, now, rev_us=DEFAULT_REV_US):
        """戻り値 (出す値, 補助したか)"""
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.2)
        self.t_last = now
        if self.state == 'rest' and now - self.t_state < self.REST_S:
            return (self.n, True) if thr > self.n + self.DEADBAND_US else (thr, False)
        if thr <= self.n + self.DEADBAND_US:            # 前進・中立: そのまま(手順は解除)
            self.reset()
            self.t_last = now
            return thr, False
        if self.state == 'rest':
            self.reset()
            self.t_last = now
        want = min(thr, self.n + rev_us)
        el = now - self.t_state
        if self.state == 'idle':
            self._go('brake', now)
            el = 0.0
        if self.state == 'brake':
            if el < self.BRAKE_S or erpm < STOPPED_ERPM:   # まだ前へ転がっている間はブレーキのまま
                return thr, False
            self._go('gap', now)
            el = 0.0
        if self.state == 'gap':
            if el < (self.RETRY_GAP_S if self.retries else self.GAP_S):
                return self.n + GAP_OFFSET_US, True
            self._go('reverse', now)
            el = 0.0
        if erpm > BACKING_ERPM:
            self.backing_s += dt
            if self.backing_s > self.REV_MAX_S:
                self._go('rest', now)
                return self.n, True
        elif self.backing_s == 0.0 and el > CHECK_S and self.retries < self.MAX_RETRIES:
            self.retries += 1
            self._go('gap', now)
            return self.n + GAP_OFFSET_US, True
        return want, want != thr


class EscTrack(object):
    """PCA9685 に書けたスロットルから、ESC が後退で走らせていたかもしれない区間を見積もる(2026-10-27 fix6 safety r1。
    drive が rec_hold_keep = 1・rec_back_guard = 1 の日だけ使う。残した中立止めのあと、車両の値が来なかった間の数えにだけ)。
    決まりはこのESCの後退(モジュールの説明の始め。10/10 のログ 116 回分)のとおり:
    ・前進(中立 − FWD_US より前)を書いたら「下がれる」が消える(前進の直後の後退側はブレーキ)
    ・ブレーキ(下がれないときの後退側)のあと、中立の帯が READY_NEUTRAL_S 続いたら下がれる(人は約 0.12 秒・閉じたループの
      偽物の ESC は 0.1 秒を 10 ms 刻みで数える。多めに数える側で 0.08 秒)
    ・下がれるときに後退側(中立 + REV_US より後ろ)を書き始めたら、DELAY_S あとから後退側が終わるまで後退で走らせている
      (下がり始めは後退側から約 0.08 秒あと。10/10 のログは 100 Hz・偽物の ESC は 10 ms 刻みで 0.07 秒から走らせることが
      あるので、多めに数える側で 0.06 秒)。後退で走らせたあとの中立のあとも下がれる
    ・分からないとき(走行の始め・人側の間)は下がれるとする。前へ転がっている間は後退しない決まりは見ない
    どれも多めに数える側(見ていない間に下がったとして、そのあとの後退を短くする側)。本物の ESC は 89% がこの形"""
    FWD_US = 20.0            # 中立からこれより前 = 前進(中立の帯 約 1478〜1538 µs の前の端)
    REV_US = EscReverse.DEADBAND_US      # これより後ろ = 後退側(ブレーキか後退)
    READY_NEUTRAL_S = 0.08
    DELAY_S = 0.06
    MAX_RUNS = 8

    def __init__(self, neutral):
        self.n = float(neutral)
        self.runs = collections.deque(maxlen=self.MAX_RUNS)     # [後退で走らせ始めた時刻 ns, 終わった時刻 ns(None = 続いている)]
        self.side = None             # 最後に書けた値の側('f' 前進 / 'n' 中立の帯 / 'r' 後退側 / None 分からない)
        self.ready = True
        self.braked = False
        self.neu_t0 = None

    def unknown(self, t_ns):
        """人側の間など、ESC に届いた値が分からない(下がれるとする)"""
        self._end(t_ns)
        self.side, self.ready, self.braked, self.neu_t0 = None, True, False, None

    def _end(self, t_ns):
        if self.runs and self.runs[-1][1] is None:
            self.runs[-1][1] = max(int(t_ns), self.runs[-1][0])

    def write(self, t_ns, thr):
        """t_ns に thr [µs] を書けた(次に書くまで続く)"""
        t_ns, thr = int(t_ns), float(thr)
        side = 'f' if thr < self.n - self.FWD_US else ('r' if thr > self.n + self.REV_US else 'n')
        if (self.side == 'n' and self.braked and self.neu_t0 is not None
                and t_ns - self.neu_t0 >= self.READY_NEUTRAL_S * 1e9):
            self.ready, self.braked = True, False                # ブレーキのあとの中立が続いた: 下がれる
        if side == 'f':
            self._end(t_ns)
            self.ready = self.braked = False
            self.neu_t0 = None
        elif side == 'r':
            self.neu_t0 = None
            if self.side != 'r':                                 # 後退側の始め
                if self.ready:
                    self.runs.append([t_ns + int(self.DELAY_S * 1e9), None])
                else:
                    self.braked = True
        else:
            self._end(t_ns)
            if self.side != 'n':
                self.neu_t0 = t_ns
        self.side = side

    def rev_s(self, a_ns, b_ns):
        """[a_ns, b_ns] のうち、後退で走らせていたかもしれない秒"""
        s = 0
        for t0, t1 in self.runs:
            lo, hi = max(t0, int(a_ns)), min(int(b_ns) if t1 is None else t1, int(b_ns))
            if hi > lo:
                s += hi - lo
        return s / 1e9


class EscMayReverse(EscTrack):
    """EscTrack と同じ決まりで、もっと多めに「ESC が後退で走らせていたかもしれない」とみる(2026-10-27 fix7 safety s4-a。drive が
    rec_back_guard = 1 の日に、書けた値を入れて Recovery の esc_rev に使う。EscTrack(中立止めの見積もり)とは別に持つ)。
    ・ブレーキのあと中立を 1 回でも書いたら下がれる(人は中立 0.02 秒で後退に入れたこともある。10/10 のログ)
    ・下がれるときの後退側は、書いたときから後退とみる(遅れ 0)
    下がれないとみるのは、前進のあとブレーキ側だけを書いている間(このESCではブレーキ。前進の直後の後退側は人の 29 回すべて・
    AI の 26 回のうち跳ね返りの 2 回のほか、後退しなかった)と、そのあとの中立の間(まだ後退側を書いていない)だけ"""
    READY_NEUTRAL_S = 0.0
    DELAY_S = 0.0


class StartKick(object):
    """発進の補助(2026-10-23 ユーザーの依頼「AIモードにしたら、すぐ走り出すように」)。
    CH3 を AI 側にした直後、止まっていてモデルが前へ進まない(中立付近)とき、弱い前進(中立 − KICK_US)を出して走り出させる。
    - 一度 MOVE_ERPM より速く前へ動いたら終わり(その AI の区間ではもう出さない。CH3 を人側へ戻すと次の AI 側で 1 回)
    - MAX_S 出しても動かなければやめる。前が近い(blocked)・モデルが後退かブレーキを出しているときは出さない
    - モデルがもっと強く前へ出しているときは、モデルのまま(補助は前進を弱めない)"""
    KICK_US = 80.0           # 中立から前へ(約 1.2 m/s の定常。人の発進は 1379〜1423 µs)
    MOVE_ERPM = -800         # これより速く前へ動いたら発進できた(eRPM は負 = 前進)
    MAX_S = 2.0

    def __init__(self, neutral_us):
        self.n = float(neutral_us)
        self.reset()

    def reset(self):
        self.done = False
        self.t0 = None

    def step(self, thr, erpm, now, blocked=False):
        """戻り値: 出すスロットル [µs]、補助しないなら None"""
        if self.done:
            return None
        if erpm <= self.MOVE_ERPM:
            self.done = True
            return None
        if self.t0 is None:
            self.t0 = now
        if now - self.t0 > self.MAX_S:
            self.done = True
            return None
        if blocked or thr > self.n + EscReverse.DEADBAND_US:
            return None
        want = self.n - self.KICK_US
        return want if thr > want else None


# ---------------------------------------------------------------- 発進の前進(2026-10-26。既定 OFF)
# 終わりの理由(AI_CMD の launch_end。0 = このフレームでは終わっていない)。1〜9 は protocol_v3 14 章の ①〜⑨。
# 10 は仕様に無い足し: 走行中に launch_boost を 1 → 0 にした(build_launch.md)
LAUNCH_END_NAMES = {0: '', 1: '速さに届いた', 2: '最大の時間', 3: 'ヨー 60 deg/s', 4: '舵 200 µs(1 m/s 以上)',
                    5: '衝撃 2.5G', 6: '前が近い・ToF ガード', 7: 'CH3 を人側', 8: '中立で止めた・走行の停止',
                    9: '車両の値が古い・後ろへ動いた', 10: 'OFF にした'}
# 始めなかった理由(記録の出来事と画面だけ。AI_CMD には書かない)
LAUNCH_SKIP_NAMES = {'off': 'OFF', 'no_switch': 'AI 側にしたのを見ていない(走行の始めから AI 側)',
                     'hold': '中立で止めたあと', 'model': 'モデルの出力が数でない', 'moving': '止まっていない(|eRPM| ≥ 400)',
                     'veh': '車両の値が古い', 'tof': 'ToF が無い・古い・ガード 0', 'ttc_n': 'TTC が 5 フレームそろわない',
                     'near': '前が近い(TTC の中央値 < 3.3 秒)', 'end': '始めのフレームで終わりの条件'}


class LaunchIn(NamedTuple):
    """発進の前進の 1 フレームの入力(drive.py が launch_boost = 1 のときだけ作る。OFF は None)"""
    us: float = 250.0                  # launch_us: 出す前進の下限 [µs](中立から。ゲインは掛けない)
    mps: float = 2.5                   # launch_mps: 0.1 秒先の速さの見込みがこれに届いたら終わり [m/s]
    max_s: float = 1.2                 # launch_max_s [秒]
    v: float = float('nan')            # 速さ [m/s](速さの天井と同じ: 車両の値 3 つの中央値)
    a: float = float('nan')            # 加速 [m/s²](速さの天井と同じ: 0.1 秒前の v との差)
    gz: float = float('nan')           # ヨー [deg/s](IMU 5 サンプルの平均、左 +)
    steer_rel: float = float('nan')    # 車に出す舵 − 中心 − trim [µs]
    ttc: float = float('nan')          # このフレームの ToF ガードの TTC [秒]
    ttc5: Tuple[float, ...] = ()       # 直近のフレームの TTC(このフレームを含む。中立で止めたら空にする)
    tof_ok: bool = False               # ToF ガードがある(tof_guard ≥ 1)・ToF の値が 0.1 秒以内
    veh_ok: bool = False               # 車両の値が 0.1 秒以内
    impact_g: float = 0.0              # 水平の衝撃 [G](直近 0.15 秒の最大)
    guard_changed: bool = False        # ToF ガードがこのフレームの出力を変えた


class LaunchBoost(object):
    """発進の前進(protocol_v3 14 章)。AI 側にした最初のフレームで、止まっていて前が空いていれば始め、
    スロットル = min(モデル, 中立 − us) を出す(前進だけ。止まっている間のモデルのブレーキは無視。舵はモデルのまま)。
    終わったら(どの理由でも)同じ AI の区間ではもう出さない。CH3 を人側へ戻すと、次の AI 側で 1 回"""
    US_RANGE = (150.0, 300.0)          # drive.PARAM_LIMITS と同じ(念のためここでも切る)
    MPS_RANGE = (2.0, 3.0)
    MAX_S_RANGE = (0.5, 2.0)
    START_ERPM = 400.0                 # |eRPM| < これ = 止まっている
    TTC_START_S = 3.3                  # 直近 5 フレームの TTC の中央値がこれ以上(止まっていれば前 1.0 m 以上空き)
    TTC_N = 5
    END_YAW_DPS = 60.0                 # ③
    END_STEER_US = 200.0               # ④(END_STEER_MPS 以上のとき)
    END_STEER_MPS = 1.0
    END_IMPACT_G = 2.5                 # ⑤
    END_TTC_S = 0.8                    # ⑥ v ≥ END_TTC_MPS で TTC < これが END_TTC_N フレーム続いた
    END_TTC_N = 2
    END_TTC_MPS = 0.3
    END_BACK_ERPM = 200.0              # ⑨ eRPM > これ = 後ろへ動いた(eRPM は負 = 前進)
    PRED_S, ACCEL_MAX = 0.10, 3.0      # ① 0.1 秒先の見込み v + clip(a, 0, 3) × 0.1(safety_caps.CeilParams と同じ)
    US_DEFAULT, MPS_DEFAULT, MAX_S_DEFAULT = 250.0, 2.5, 1.2   # 数でない値のとき(drive.DEFAULT_PARAMS・LaunchIn と同じ)

    def __init__(self, neutral_us):
        self.n = float(neutral_us)
        self.count = 0                 # 始めた回数(stats の launch_boosts)
        self.armed = False             # この走行で人側のフレームを見た(「AI 側にした」と分かる)
        self.last_end = 0              # 最後に終わった理由
        self.last_skip = ''            # 最後に始めなかった理由(LAUNCH_SKIP_NAMES の鍵)
        self.event = None              # このフレームの出来事 (種類, 情報)。drive.py が記録の出来事に書く
        self._section()

    def _section(self):
        self.decided = False           # この AI の区間で、始めるかを決めた
        self.active = False
        self.used = False              # この AI の区間で出した(終わったあとも StartKick を出さない)
        self.skipped = False           # この AI の区間で「出さない」と決めた(ON のときだけ。launch_no_kick で StartKick も出さない)
        self.held = False              # 人側のフレームのあとに中立で止めた
        self.t0 = None
        self.cfg = None                # 始めたときの (us, mps, max_s)
        self.near_n = 0
        self.frames = 0

    @staticmethod
    def _clip(x, rng, default):
        """範囲に切る。数でない値(NaN・±inf・None・文字)は既定値(2026-10-26 fix2: min(max(nan, lo), hi) は nan を通した。
        drive.py の入口(_clean_params)も ±inf を「数値ではありません」として断る)"""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(v):
            return float(default)
        return min(max(v, rng[0]), rng[1])

    def _end(self, code, now=None):
        self.active = False
        self.last_end = int(code)
        self.event = ('end', {'code': int(code), 'reason': LAUNCH_END_NAMES.get(int(code), ''), 'frames': self.frames,
                              's': None if now is None or self.t0 is None else round(now - self.t0, 3)})
        return int(code)

    def human(self):
        """CH3 が人側(または dry run)のフレーム。戻り値: このフレームで終わった理由(⑦。出していなければ 0)"""
        code = self._end(7) if self.active else 0
        self._section()
        self.armed = True
        return code

    def hold(self):
        """中立で止めた(⑧)。戻り値: 終わった理由(出していなければ 0)。止めたあとの最初の AI 側では始めない"""
        self.held = True
        return self._end(8) if self.active else 0

    def stop(self):
        """走行の停止(⑧)"""
        return self._end(8) if self.active else 0

    def _why_not(self, thr, erpm, inp):
        if inp is None:
            return 'off'
        if not self.armed:
            return 'no_switch'
        if self.held:
            return 'hold'
        if not math.isfinite(thr):
            return 'model'
        if not abs(float(erpm)) < self.START_ERPM:
            return 'moving'
        if not inp.veh_ok:
            return 'veh'
        if not inp.tof_ok:
            return 'tof'
        xs = [float(x) for x in inp.ttc5[-self.TTC_N:]]
        if len(xs) < self.TTC_N or not all(math.isfinite(x) for x in xs):
            return 'ttc_n'
        if not float(np.median(xs)) >= self.TTC_START_S:
            return 'near'
        return ''

    def _end_reason(self, erpm, now, inp):
        """終わりの条件(安全の順に見る)。0 = まだ続ける"""
        v = float(inp.v)
        if not inp.veh_ok or not math.isfinite(v) or float(erpm) > self.END_BACK_ERPM:
            return 9
        moving = v >= self.END_TTC_MPS
        self.near_n = self.near_n + 1 if (moving and math.isfinite(inp.ttc) and inp.ttc < self.END_TTC_S) else 0
        if inp.guard_changed or not inp.tof_ok or not math.isfinite(inp.ttc) or self.near_n >= self.END_TTC_N:
            return 6
        if float(inp.impact_g) >= self.END_IMPACT_G:
            return 5
        if math.isfinite(inp.gz) and abs(float(inp.gz)) >= self.END_YAW_DPS:
            return 3
        if v >= self.END_STEER_MPS and math.isfinite(inp.steer_rel) and abs(float(inp.steer_rel)) >= self.END_STEER_US:
            return 4
        a = float(inp.a)
        a_c = min(max(a, 0.0), self.ACCEL_MAX) if math.isfinite(a) else 0.0
        if v + a_c * self.PRED_S >= self.cfg[1]:
            return 1
        if now - self.t0 >= self.cfg[2]:
            return 2
        return 0

    def step(self, thr, erpm, now, inp):
        """AI 側のフレーム。thr = モデルのスロットル(ゲイン・ガードのあと)、inp = LaunchIn か None(OFF)。
        戻り値 (出すスロットル [µs] か None(モデルのまま), このフレームで終わった理由 0〜10)"""
        if not self.decided:
            self.decided = True
            why = self._why_not(thr, erpm, inp)
            if not why:
                self.cfg = (self._clip(inp.us, self.US_RANGE, self.US_DEFAULT),
                            self._clip(inp.mps, self.MPS_RANGE, self.MPS_DEFAULT),
                            self._clip(inp.max_s, self.MAX_S_RANGE, self.MAX_S_DEFAULT))
                self.t0, self.near_n, self.frames = now, 0, 0
                code = self._end_reason(erpm, now, inp)
                if code:
                    why = 'end'
                    self.skipped = True
                    self.last_skip = why
                    self.event = ('skip', {'why': why, 'reason': LAUNCH_SKIP_NAMES[why], 'code': code,
                                           'code_reason': LAUNCH_END_NAMES[code]})
                    return None, 0
                self.active = self.used = True
                self.count += 1
                self.event = ('start', {'us': self.cfg[0], 'mps': self.cfg[1], 'max_s': self.cfg[2],
                                        'erpm': float(erpm), 'ttc_med': round(float(np.median(inp.ttc5[-self.TTC_N:])), 3)})
            else:
                if inp is not None:
                    self.skipped = True
                    self.last_skip = why
                    self.event = ('skip', {'why': why, 'reason': LAUNCH_SKIP_NAMES[why]})
                return None, 0
        if not self.active:
            return None, 0
        if inp is None:                 # 走行中に OFF にした
            return None, self._end(10, now)
        code = self._end_reason(erpm, now, inp)
        if code:
            return None, self._end(code, now)
        want = self.n - self.cfg[0]
        self.frames += 1
        return (thr if (math.isfinite(thr) and thr < want) else want), 0     # 前進だけ(thr が nan・inf なら want)


class Assist(object):
    """走行ループから使う窓口。人が操縦中・出力しない確認中は何もしない"""

    def __init__(self, ref):
        self.ref = (float(ref[0]), float(ref[1]))
        self.esc = EscReverse(self.ref[1])
        self.rec = Recovery(self.ref)
        self.kick = StartKick(self.ref[1])
        self.boost = LaunchBoost(self.ref[1])
        self.esc_count = 0
        self.kick_count = 0
        self._esc_active = False
        self._kick_active = False
        self.launch_on = False          # このフレームで発進の前進を出した(AI_CMD の launch_on)
        self.launch_end = 0             # このフレームで発進の前進が終わった理由(AI_CMD の launch_end)
        self._no_kick = False           # この AI の区間は StartKick を出さない(launch_no_kick で「出さない」と決めた区間。fix3 r1-rs1-2)

    def reset(self, keep_recovery=False):
        """中立で止めたあとなど。発進の補助は戻さない(CH3 を人側へ戻したときだけ、次の AI 側で 1 回)。
        keep_recovery = True(drive の rec_hold_keep = 1 で、AI 側のままの中立止め。2026-10-26 fix3 safety1-1)は、
        リカバリーを戻さない(中止・途中の段・armed・衝撃・あきらめの数をそのまま)。後退の受け付けは戻す。
        既定 False は今までと同じ(リカバリーも全部戻す)"""
        self.esc.reset()
        if not keep_recovery:
            self.rec.reset()
        self._esc_active = False

    def launch_hold(self):
        """中立で止めたフレーム(drive.py の hold が記録の前に呼ぶ)。戻り値: 発進の前進が終わった理由(⑧。無ければ 0)"""
        self.launch_on = False
        self.launch_end = self.boost.hold()
        return self.launch_end

    def step(self, model_us, erpm, select, now, esc_on=True, rec_on=True, impact_g=0.0,
             rev_us=DEFAULT_REV_US, rev_s=DEFAULT_REV_S, kick_on=False, blocked=False, launch=None, rec_guard=False,
             launch_no_kick=False, hold_g=0.0, hold_since=None, hold_back_s=0.0, hold_unseen_s=0.0, rec_keep=False,
             hold_rev_s=0.0, esc_rev=None):
        """戻り値 (出す値 (steer, throttle), 補助の種類)。blocked = 前が近い(発進の補助を出さない)。
        launch = 発進の前進の入力 LaunchIn(launch_boost = 1 のとき。None = OFF で今までと同じ)。
        rec_guard = リカバリーの後ろの見張り(drive の rec_back_guard。False = 今までと同じ)。
        launch_no_kick = 発進の前進が「出さない」と決めた区間では StartKick も出さない(drive の launch_no_kick。
        False = 今までと同じ。launch が None(OFF)なら決めないので効かない)。一度「出さない」になった区間は、途中で
        launch_no_kick を 0 にしても出さない(CH3 を人側にするまで。2026-10-26 fix3 r1-rs1-2: 前は 1 → 0 にした最初のフレームで
        StartKick の 2 秒の窓が新しく開き、リカバリーの中止・後退の途中を上書きして前進 80 µs を 2 秒出した)。
        hold_g = Recovery.step の同じ名前(rec_hold_keep の中立止めの間の衝撃。既定 0 = 今までと同じ)。
        hold_since・hold_back_s = Recovery.step の同じ名前(rec_hold_keep の中立止めのあと。既定 None・0 = 今までと同じ。
        2026-10-27 fix3 r2-r2-rs-1・rs-2)。hold_unseen_s = Recovery.step の同じ名前(2026-10-27 fix5 s3。既定 0 = 今までと同じ)。
        hold_rev_s = Recovery.step の同じ名前(2026-10-27 fix6 s3-a。既定 0 = 今までと同じ)。
        esc_rev = Recovery.step の同じ名前(drive の EscMayReverse。2026-10-27 fix7 s4-a。既定 None = 今までと同じ)。
        発進の補助・発進の前進がこのフレームを出したときは Recovery を呼ばない(今までと同じ。
        補助は AI の区間の始めで、前へ動けば終わる。Recovery はその区間の始めに戻してあり、動くまで手順を始めない)。
        rec_keep = drive の rec_hold_keep と rec_back_guard がどちらも 1(中立止めでリカバリーを残す日。既定 False = 今までと同じ)。
        True なら、発進の補助・発進の前進が出したフレームでも Recovery の「前に見たフレームの時刻」(t_last)をこのフレームにする
        (2026-10-27 fix5 s2: 前は区間の始めから None のままで、発進の前進の途中でカメラが止まり、その間に壁に当たっても、明けた
        フレームで「見ていない時間」が分からず、長い中立止めのあとの中止(_after_hold)が効かなかった)"""
        self.boost.event = None
        self.launch_on = False
        if not select:
            self.launch_end = self.boost.human()
            self.reset()
            self.kick.reset()
            self._kick_active = False
            self._no_kick = False
            return model_us, ASSIST_NONE
        lb, self.launch_end = self.boost.step(model_us[1], erpm, now, launch)
        if launch_no_kick and self.boost.event is not None and self.boost.event[0] == 'skip':
            self.boost.event[1]['no_kick'] = True       # 記録の出来事: この区間は StartKick も出さない
        if launch_no_kick and self.boost.skipped:
            self._no_kick = True        # この区間の終わり(CH3 を人側)まで残す。0 のままなら立たない(今までと同じ)
        if lb is not None:              # 発進の前進(StartKick の場所。リカバリー・後退の受け付けより先)
            self.launch_on = True
            self._kick_active = False
            if abs(erpm) >= self.rec.ARM_ERPM:      # 出している間に動いたことはリカバリーに残す(終わったあとの衝突に備える)
                self.rec.armed = True
            if rec_keep:                # このフレームは見た(fix5 s2: 次の中立止めの「見ていない時間」をここから数える)
                self.rec.t_last = now
            return (model_us[0], float(lb)), ASSIST_LAUNCH
        # 発進の前進を出した区間では、終わったあとも StartKick を出さない。launch_no_kick なら「出さない」と決めた区間でも出さない
        if kick_on and not self.boost.used and not self._no_kick:
            k = self.kick.step(model_us[1], erpm, now, blocked)
            if k is not None:
                if not self._kick_active:
                    self.kick_count += 1
                self._kick_active = True
                if rec_keep:            # 発進の前進と同じ(fix5 s2)
                    self.rec.t_last = now
                return (model_us[0], float(k)), ASSIST_START
        self._kick_active = False
        if rec_on:
            o, k = self.rec.step(model_us, erpm, now, impact_g, rev_us, rev_s, back_guard=rec_guard, hold_g=hold_g,
                                 hold_since=hold_since, hold_back_s=hold_back_s, hold_unseen_s=hold_unseen_s,
                                 hold_rev_s=hold_rev_s, esc_rev=esc_rev)
            if o is not None:
                self.esc.reset()
                return (float(o[0]), float(o[1])), k
            # 止まっている間のモデルの後退は中立に(下がるのはリカバリーに任せる。前へ走っている間はブレーキとしてそのまま)
            if model_us[1] > self.ref[1] + EscReverse.DEADBAND_US and erpm >= STOPPED_ERPM:
                return (model_us[0], self.ref[1]), ASSIST_ESC
            return model_us, ASSIST_NONE
        if self.rec.active:
            self.rec.reset()
        if esc_on:
            thr, changed = self.esc.step(model_us[1], erpm, now, rev_us)
            if changed:
                if not self._esc_active:
                    self.esc_count += 1
                self._esc_active = True
                return (model_us[0], float(thr)), ASSIST_ESC
            self._esc_active = False
        return model_us, ASSIST_NONE

    @property
    def recoveries(self):
        return self.rec.count
