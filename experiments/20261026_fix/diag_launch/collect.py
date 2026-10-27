"""原因 C の診断(1): 止まってから AI で走り出した場面を全部集める(10/23・10/24 の AI 走行 50 本)。2026-10-26

読むだけ。data/raw・data/labels は変えない。

発進 = AI の区間(labels の ai が続くところ)の最初のフレームで |eRPM| < 400(0.2 m/s 未満)。
  補助(assist = 4)が出た発進・出なかった発進の両方。
最初の R1 の起点 = label_events.checked_anchors(偽の起点を外したもの。label_log と同じ作り方)。
走りながらの同じ区間 = 同じ走行の「続いた周」(起点 → 次の起点がずっと AI・5〜16 秒・向きの変化 300〜420°)の、
  起点の d m 手前 → 起点(d = 発進から最初の起点までの車輪の距離)。同じ走行に周が無ければ、同じモデル・同じ速さのゲイン・
  同じ回の周を使う(ref = 'same_setting')。

出すもの(out/launches.json):
  発進ごと: 起点までの時間・失う時間・動き出し(0.2 / 0.4 m/s)・補助の長さ・モデルが出した前進・舵・加速(車輪と IMU)・
  空転の目安(車輪の速さの増え ÷ IMU の加速の積分)・距離の格子(0.1 m)の上の発進と周の比べ(時間・速さ・指令・舵)。
  周の区間(参考): 同じ走行の周の、同じ場所の値。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I collect.py
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I collect.py --day 20261103      # 11/3(2026-10-26 fix2)
        → out/launches_20261103.json(--out FILE で変えられる)。--day は何回でも・コンマ区切りでも

ほかの日(--day。2026-10-26 fix2・係 tools で足した。10/23・10/24 を読むとき(引数なし)は今までとビット単位で同じ):
  記録 = その日の AI 走行(log.json が読めて drive があり、dry run でない。0 バイトの log.json は飛ばす)。
  ラベルが data/labels に無い記録は label_events.label_log でその場で作る(書き出さない)。0 バイトの npz は load_log が飛ばす。
  発進ごとに足す欄(10/23・10/24 には足さない):
    launch_param(log.json の launch_boost)・launch_recorded(AI_CMD に launch_on の欄がある)・
    launch_frames(assist = 5 のフレーム)・launch_on_frames・launch_on_s(AI 側にしてから最後に出したフレームまで)・
    launch_end_code・launch_end_reason・launch_end_s(AI_CMD の launch_end の最初の 0 でない値)・
    launch_events(events.jsonl の launch_boost: [時刻, start/skip/end, 理由])・
    launch_state(off / on = 出た / on_not_started = launch_boost 1 なのに出ない → 手順書 7 章: ON として数えない)・
    settings(log.json の天井・左・自動・発進の前進の値・ゲイン)・回(説明欄の block → bench_candidates.json の run_id)・
    collisions_first_lap(最初の周の衝突 [時刻, G, 場所])。最初の周 = 発進 → 1 周目の線(2026-10-26 fix3、反証 1 回目 tools:F9:
      tools/standing_3lap.py の 1 周目と同じ。線 = A2 から d_pre 戻ったところ、d_pre = 動いた所(0.2 m/s)→ A1 の車輪の距離)。
      前は発進 → A2 で、2 周目の HS・LH の当たりも「最初の周」に入った。first_lap_end_s = その終わり・first_lap_a2_s = A2・
      first_lap_end = 'line' / 'ai_end'(A2 が無い・線の前に AI の区間が終わった: AI の区間の終わりまで。前と同じ)・
      first_lap_full = 線まで AI 側(前は「A2 が AI の区間の中」)。
  同じ設定の周(ref = 'same_setting')の鍵に、天井・左・自動合わせ・目標を足す(10/23・10/24 では欄が無いので同じ)。
    左・目標は、天井・自動合わせが ON のときだけ鍵に入れる(2026-10-26 fix3。summarize の組と同じ。前は OFF でも入れた)。
  まとめ(summary): 回・設定ごとの発進の前進の ON/OFF の比べ(protocol_v3 14 章・手順書 6 章: ON の「AI 側 → 最初の R1 の起点」の中央値 ≤
  OFF − 0.15 秒・ON の最初の周の衝突 0・空転の比 ≤ 1.3・LH の入口の速さ ON − OFF ≤ +0.3 m/s)。失う時間 loss_s の中央値も並べる。
  2026-10-26 critic: 自動合わせ ON の本は、組と同じ設定の周の鍵のゲインを log.json の throttle_gain にする(フレームの
  ゲインは電圧で本ごとに違い、OFF と ON が別の組になって比べが出なかった。反証 1 回目 R1)。ON の組に launch_us の違う本が
  混ざったら注意の行を出す(tools:F4)。自動合わせ OFF の本・10/23・10/24 の読み方は変えていない。
  2026-10-26 fix3(tools:F4 の残り): ON に launch_us の違う本が混ざったら、比べは「その組の最後の ON の launch_us」の ON の本と
  OFF の本で出す(protocol_v3 14 章: 空転の比 > 1.3 → 残りの ON を 200 µs に。API の値は再起動まで残るので、回 4 はその強さで
  走る)。最初の周の衝突だけは、強さによらず全部の ON で数える(14 章の「ON の最初の周で当たる → ON をやめる」)。混ざって
  いないときは今までと同じ(出力も同じ)。
  2026-10-26 fix3 直す 1 回目(2 組目。ほかの日だけ。10/23・10/24 を読むとき(引数なし)は今までとビット単位で同じ):
  - r1-rec3-1  R1 の起点に届かなかった発進(当たって人側・15 秒以内に起点なし)も、発進 → AI の区間の終わり + 0.2 秒(その前に
    起点があれば起点まで)の衝突と場所を collisions_first_lap に入れる(first_lap_end = 'no_r1')。前は行から抜け、まとめにも
    出なかったので、ON の発進が R1 の前に 6G で当たっても「最初の周の衝突 0 → 使ってよい」と出た。まとめは組ごとに
    not_reached(状態ごとの本数・名前・理由・衝突)を出し、ON に届かなかった発進があれば checks に on_all_reached_r1 = False
    (→ 満たさない)、その衝突があれば first_lap_collisions_0 = False。届いた ON が無くても、OFF があれば「満たさない」の行を出す。
  - r1-rec3-3  走行中に変えた本(tools/standing_3lap.midrun_changes と同じ見方: events.jsonl の params・log.json の
    params_end・AI_CMD の天井・自動合わせが途中から OFF・発進の前進の終わり 10・フレームのゲイン/トリムが途中で変わった)は、
    行に midrun を付け、まとめの比べから外して「途中で変えた N 本(名前)」の行を出す(手順書 7 章: ON の組にも OFF の組にも
    入れない)。ただし途中で変えた ON の発進の最初の周の衝突は「最初の周の衝突 0」に数える(衝突は止める条件)。その本の周は、最初に変えた時刻より前に終わった周だけ、ほかの本の「同じ設定の周」に使う(時刻が分からなければ使わない)。
  - r1-rec3-5  設定(settings)に rec_back_guard・rec_hold_keep・launch_no_kick を足す(log.json に欄があるときだけ)。組の中で
    ON/OFF が混ざったら注意の行。
  - --skip LOG[@T](何回でも): まとめから外す(車のせいでない理由で R1 の前に止めた本など。外した名前を出す)。
  どれも、その場面が無ければ出力(行・まとめ・JSON)は前と同じ(足す欄は、あるときだけ)。
  2026-10-27 fix3 直す 2 回目・1 組目(反証 2 回目 r2-rec2-1。ほかの日だけ):
  - まとめの組の鍵にはゲイン(自動合わせ OFF はフレームのゲイン)が入る。手順書 4 章 3 は毎本ゲインを電圧の表で入れ直すので、
    回 3 の 8 本でパックの電圧が下がると組が分かれ、「ON − OFF」の行が出ない・1 本対 1 本の答えが別々に出た(6 章の
    「4 本対 4 本の中央値」にならない)。
  - --ignore-gain: まとめの組の鍵からゲインを外す(同じ回・同じモデル・同じスイッチの本を 1 組に)。組の行に「g0.90〜0.95
    (ゲインをまとめた)」とゲインの幅を出す(JSON の組は gain = None・gain_range・gain_ignored)。発進ごとの行・同じ設定の周の
    鍵(失う時間の元)は変えない。
  - --ignore-gain なしで、同じ回・同じモデル・同じスイッチなのにゲインの違う組が 2 つ以上あり、その中に発進の前進 ON の本が
    あれば、それぞれの組に「注意: ゲインが本ごとに違うので組が分かれた(0.90・0.92)…--ignore-gain で読む」の行
    (JSON の組に gain_split)。その場面が無ければ出力は前と同じ。
  2026-10-27 fix3 直す 2 回目・2 組目(反証 2 回目 r2-rec2-2・3・5。ほかの日だけ。10/23・10/24 を読むとき(引数なし)は
  今までとビット単位で同じ):
  - r2-rec2-3  最初の周の終わりが 'ai_end'(線の前に AI の区間が終わった)の発進は、AI の区間の終わり + FLIP_TAIL_S(0.5 秒。
    その前に線があれば線)まで衝突を数える。前は AI の最後のフレームまでで、人が当たる直前に人側にした ON の衝突(惰性で当たる)
    を数えず「使ってよい」と出た。R1 に届かなかった発進('no_r1')の足しも 0.2 → 0.5 秒にそろえた(本物の 10/23・10/24 の 50 本で、
    AI の区間の終わりから 0.5 秒以内の人側の衝突は 7 件。4G 以上の 4 件は 0.16〜0.35 秒あと、うち 8G 以上 2 件は 0.34・0.35 秒)。'line' の決め方(線が AI の区間の終わり + 0.2 秒以内)は
    前のまま。'ai_end' の ON の発進があれば、組に「最初の周を AI の区間の終わり + 0.5 秒まで数えた ON」の行(線の前に人側
    on_before_line / A2 が無く線が分からない on_no_a2。名前だけ。比べは変えない)。A2 が無い('ai_end' の多く)のは、記録が A2 の
    前で終わった・当たって止まった・人側で R1 を通らなかった本で、そのときも AI の区間の終わり + 0.5 秒まで数える。
  - r2-rec2-2  LH の入口の速さ(v_launch_at_lh)は、走りながらの周(同じ記録か同じ設定の周)が無いと出ず、回 3 を全部
    「止まって 1 周」で走ると「LH — m/s → 満たさない」になった。発進のフレームの場所(loc_code の LH = 6)で決めた LH の入口の
    速さを全部の発進で持ち(周の無い発進は行に lh_self として出す)、比べる ON・OFF の発進のどれかに周の LH が無ければ、
    ON・OFF とも発進のフレームの場所の値で比べる(組の compare に lh_src = 'launch_loc'。行の LH は「(発進の場所で)」)。
    それでも出せなければ理由の行。全部の発進に周の LH があるときは今までと同じ。
  - r2-rec2-5  読めなかった記録(npz が無い・壊れた: 電源断・jetformer-web の落ち。前は「skip」の 1 行だけで比べから消えた)を、
    組のまとめ(unreadable)と JSON(unreadable)に、名前・理由・発進の前進の ON/OFF(log.json の launch_boost と
    bench_candidates.json の計画 expect.launch_boost のどちらかが 1 なら ON)で出す。読めない ON があれば、その組は
    「満たさない」(checks に on_all_readable = False)。記録のシーンが欠けていて(load_log の problems)、最初の周の間の
    IMU が途中で切れた・抜けた発進(first_lap_gap)も同じ。--skip 記録名 で外せる(動画・紙で確かめた本)。
  2026-10-27 fix3 直す 3 回目・1 組目(反証 3 回目 r3-r3-rec-1。ほかの日だけ。その場面が無ければ出力は前と同じ):
  - log.json が読めない記録(0 バイト・壊れた・無い。電源断の本物の形 = 10/23 の 20261023-204825 は log.json・scene.json・npz が
    全部 0 バイト)は、logs() が人の収集・dry run と同じく飛ばしていたので、r2-rec2-5 の「読めない記録」にも入らず、回 3 の ON の本が
    落ちても「使ってよい」と出た(skip の行も出なかった)。unreadable_meta_logs() で拾い、読めない記録(unreadable。meta = False・
    launch_state = 'unknown')として出す。回・ON/OFF が分からないので、その日の発進の前進 ON のある組(出た・出ず・届かなかった・
    途中で変えた・読めない ON)は全部「満たさない」(checks の on_all_readable = False。組に unreadable_unknown)。名前と、同じ日の
    前と次の読める AI の記録(回の番号)を出す。--skip 記録名 で外せる。logs() の返す一覧は変えない
  2026-10-27 fix3 直す 3 回目・2 組目(反証 3 回目 r3-r3-rec-3・rec-4。ほかの日だけ。その場面が無ければ出力は前と同じ):
  - r3-r3-rec-3  手順書 7 章の「その本は数えない(memo に「試し」)」(確認の文・走った値が計画と違った本)を読まず、試しの本が回 3 の
    比べに入って答えが変わった(4 本対 4 本の中央値に 1 本混ざるだけで「満たさない」→「使ってよい」)。説明欄(log.json と走り終わりの
    メモ = scene.json を section_scorecard.effective_description でまとめたもの)の memo(と key=値 の外の文)に人が「試し」と書いた
    本(tools/standing_3lap.trial_memo)は、行に trial(memo の文)を付け、途中で変えた本と同じに比べから外し「試しの N 本(名前・
    memo)」の行を出す(ON の組にも OFF の組にも入れない)。ただし試しの ON の発進の最初の周の衝突は「最初の周の衝突 0」に数える
    (衝突は止める条件)。走行開始でサーバーが足す「計画と違うので試し(元 cand=…): …」(web/bench.plan_check。11/3 はゲインが
    1.00 でない本の全部に付く)は試しに数えない。--skip でも外せる(前と同じ)
  - r3-r3-rec-4  空転の比は IMU の積分が 0.2 m/s 以下だと None で、まとめの空転max と条件から黙って抜けた(いちばんひどい空転の
    ON が「使ってよい」になった)。ON の発進の比が出ないとき、車輪の増えが 0.5 m/s より大きければ「いちばんひどい空転」= 比 > 1.3
    とみなし(slip_extreme。checks の slip = False)、ほかの理由(窓が短い・IMU が抜けた・車輪も IMU もほとんど増えない)なら
    「空転の比が出ない」(slip_unmeasured。checks に slip_all_measured = False)。どちらも比べに使う ON(混ざったら最後の強さ)で見る。
    名前の行を出す。比が出る発進だけのときは前と同じ
  2026-10-27 fix5(反証 4 回目 c1・c3。ほかの日だけ。その場面が無ければ出力は前と同じ。10/23・10/24 を読むとき(引数なし)は今までと
  ビット単位で同じ):
  - c3  まとめの組の鍵には天井・左・自動合わせ・目標が入るので、回 3 の途中で起こし直して左が 4.2 → 4.1 に戻る(RESET_ON_BOOT)と、
    回 3 が 2 本対 2 本の 2 組に分かれ、片方が「使ってよい」・片方が「満たさない」と出た(注意はゲインの gain_split だけ)。手順書 6 章の
    「同じ回の ON 4 本対 OFF 4 本」と違う行で回 4 の ON が決まりえた。今は、同じ回・同じモデルで比べに入る本(届いた・届かなかった・
    読めない。途中で変えた・試しの本は入れない)のある組が 2 つ以上の形(天井・左・自動・目標)に分かれ、どれかに発進の前進 ON の本が
    あれば、どの組にも form_split(形の文)と「注意: … N 組に分かれた」の行を付け、比べは「満たさない」(checks の one_form = False。
    安全側 = 回 4 は発進の前進 OFF)。計画と違う形の本を「試し」か --skip で外せば 1 つの形に戻る
  - c1  シーンが欠けた本(load_log の problems: npz が無い・0 バイト・読めない)を、標準出力に 1 本 1 行「記録が欠けている本 …」で出す
    (前は JSON の problems だけで、最初の周に掛からない欠け = 落ちる前の 9G などは標準出力に出なかった)。比べは変えない
  2026-10-27 fix6(反証 6-1 回目 w1-tools-r6-1-tools-1。ほかの日だけ。その場面が無ければ出力は前と同じ。10/23・10/24 を読むとき
  (引数なし)は今までとビット単位で同じ):
  - AI 側にしたシーンごと書けずに落ちた本(電源断・jetformer-web の落ち。recorder はシーンを 20 秒ためて終わりに npz を書くので、
    人側で置いて待った 1 つ目のシーンだけが残る。10/24 の本物も、AI 側にしたのが 20 秒より後の本がふつうにある)は、読めても AI の
    フレームが 1 つも無いので、main が黙って飛ばしていた(skip の行・読めない記録・記録が欠けている本のどれにも出ず、回 3 の ON の本が
    最初の周に当たっても「使ってよい」と出た)。今は、AI のフレームが無く、フレームの無い欠けたシーン(lost_scenes: その間は人側・
    AI 側が分からない)のある本を「skip … AI 側のフレームが無い …」と読めない記録(unreadable。lost_scenes)に入れる。発進の前進の
    ON/OFF は前と同じ(log.json の launch_boost・計画)に、events.jsonl の発進の前進の start(drive.py がすぐ書く。jetformer-web の
    落ちでは残る。電源断では消えうる = 下の fix6-2)も ON の手がかりにする(launch_starts)。ON なら、その組は「満たさない」
    (on_all_readable = False)。--skip 記録名 で外せる
  - 読めた本でも、欠けたシーンの間に events.jsonl の発進の前進の start がある(読めた発進のあと、2 回目の発進のシーンごと落ちた)
    なら、その発進の記録が無いので、同じく読めない記録(ON)に入れる。--first-only で、読めた発進のあとの start だけなら入れない
    (読めた本の 2 回目以降の発進と同じ)。発進の前進 OFF の本の、欠けたシーンの発進は記録に手がかりが無い(数えられない)
  - 「記録が欠けている本 …」の行を、AI のフレームが無い本にも出す(読めない記録に入れた・欠けたシーンにもフレームがあり AI 側に
    していない)。JSON の problems にも入れる。読めた本で上の start がある本は、行の後ろに 1 文
  2026-10-27 fix6-2(反証 6-2 回目 w2-tools-r62-1。ほかの日だけ。その場面が無ければ出力は前と同じ。10/23・10/24 を読むとき(引数なし)は
  今までとビット単位で同じ):
  - 上の「欠けたシーンの間の start」は events.jsonl が残る前提だった。recorder.add_event は足すだけで fsync しないので、電源断では
    落ちる前の約 30 秒(/proc/sys/vm/dirty_expire_centisecs = 3000)に書いた行が消える(本物の 10/10・10/23・10/24 の落ちた本 3 本は
    どれも電源断の形: 0 バイトの jpg・npz・scene.json。10/24 の 20261024-162305 は落ちる前 30.7 秒の中に書いたものが消えた)。2 回目の
    発進は最初の周(約 8〜10 秒)で落ちると、その start の行もほぼ必ず消え、ON の 2 回目の発進が黙って比べから消えて「使ってよい」と出た
    (start の行が 1 行残った本だけ「満たさない」)。今は、発進の前進 ON の読めた本で、最後の読めるフレームのあとにフレームの無い欠けた
    シーン(lost_tail: 電源断・jetformer-web の落ちの形。落ちれば必ず記録の終わりが欠ける)があれば、start が無くても、その本を読めない
    記録(ON。lost_scenes・lost_after_s = 最後の読めるフレームの時刻・launch_on_from = ON の手がかり)に入れる(→ 満たさない)。ON の
    手がかり(launch_on_from)= log.json の launch_boost・計画(前と同じ)・events.jsonl の発進の前進の出来事(start・skip・end。
    drive.py は ON のときだけ書く)・走行中に launch_boost を 1 にした(midrun)。--first-only は、その本に読めた発進があれば入れない
    (欠けた区間の発進は 2 回目以降。上の決まりと同じ)。欠けたシーンの間の start で入れた本は前と同じ(二重には入れない)。真ん中だけ
    欠けた本(あとのシーンが読める = 電源断の形でない)・発進の前進 OFF の本は前と同じ。AI のフレームが無い本の ON/OFF にも同じ手がかり
    を使う(前は log.json・計画・start だけ)。「記録が欠けている本 …」の行の後ろに 1 文。--skip 記録名 で外せる(本ごと)
  2026-10-27 fix7(反証 7-1 回目 w1-tools-r7-1-tools-1。ほかの日だけ。その場面が無ければ出力は前と同じ。10/23・10/24 を読むとき(引数なし)は
  今までとビット単位で同じ):
  - 両方 ON の日(rec_back_guard = rec_hold_keep = 1。11/3 は全部の回)は、リカバリー中止(assist 3。中立で人を待つ)が CH3 を人側にする
    まで解けない。中止で止まった車を、その場で人側 → AI 側に戻して走り直すと(手順書 7 章の中立止めの行・approvals 3 の「人が CH3 で戻す」
    のとおり)、新しい AI の区間ができて発進の行が 1 つ増え、R1 の手前(コースの途中)から出たその発進が ON/OFF の起点までの中央値に入った
    (作り物の回 3 で −0.125 秒「満たさない」→ −0.165 秒「使ってよい」)。今は、同じ本の中で一つ前の AI の区間がリカバリー中止で終わった
    発進(tools/standing_3lap.abort_resumes。区間の最後のフレームは中立止め・AI_CMD が人側のフレームを除いて見る。走り直しで 0.5 m も
    進まずにまた人側にしたら次の区間も)に after_abort(中止の始め・中止で終わった区間の終わり・人側の秒)を付け、R1 に届いた走り直しは
    途中で変えた本・試しの本と同じに比べ(起点まで・失う・動く・空転・LH・2 回目以降の数)から外し、組に abort_resume_excluded(名前・
    起点まで・その最初の周の衝突)と「リカバリー中止のあとの走り直し N 本 …」の行を出す。手で HS に置き直しても車輪の距離は 0 なので、
    記録からは置き直したか分からない(一律に外す = 安全側)。ただし ON の走り直しの最初の周の衝突は「最初の周の衝突 0」に数える
    (abort_resume_on_collisions。衝突は止める条件)。R1 に届かなかった走り直しは前と同じ「R1 に届かなかった発進」(ON なら on_all_reached_r1
    = False で満たさない・その衝突も数える。比べの中央値にはもとから入らない)で、行の後ろに「うちリカバリー中止のあとの走り直し N」
    (not_reached の after_abort)。発進ごとの行(launches)と標準出力の行にも印。中止の無い本は前と同じ
  2026-10-27 fix7 r2(反証 7-2 回目 w2-tools-r7-2-tools-1・2。ほかの日だけ。その場面が無ければ出力は前と同じ。10/23・10/24 を読むとき
  (引数なし)は今までとビット単位で同じ):
  - tools-1  手順書 7 章「中立で停止中 → 車に近づく前に CH3 を人側」のとおり中立止めの途中で人側にすると、リカバリー中止(assist 3)は
    1 フレームも書かれず(明けた最初のフレームでしか決まらない)、その場から戻した発進に印が付かなかった(作り物の回 3 で −0.165 秒
    「使ってよい」のまま)。今は standing_3lap.abort_resumes に車輪の eRPM も渡し、中立止め(0.5 秒以上)・カメラの途切れ(フレームの
    無い中立止め)のまま止まって終わった区間のあとの発進にも同じ after_abort(kind = 'hold' / 'stall'。中止は 'abort')を付ける。扱いは
    fix7 r1 と同じ。名前は fix7 r1 の「リカバリー中止のあとの走り直し」のまま(手順書が引く言葉)。組の行・発進の行に止まった形
    (中止 X 秒 = r1 と同じ文 / 中立止め X 秒から Y 秒 / カメラの途切れ X 秒から Y 秒)。組の行の説明の文に、中立止め・途切れのあとも
    同じこと・空転と最初の周の欠けは数えることを足した(行の頭「リカバリー中止のあとの走り直し N 本(」は同じ)
  - tools-2  fix7 r1 は走り直しの ON(出た)の発進を、起点まで・失う・動く・LH の中央値と一緒に、空転と最初の周の記録の欠けの決まり
    からも外していた(退行。car_check の「発進の空転の比 … → ⑤ は空転の条件を満たさない」と食い違った)。今は、R1 に届いた走り直しの
    ON(比べに使う強さ。launch_us が混ざった組は最後の強さの走り直しだけ)の空転(比 > 1.3・比が出ないのに車輪が増えた = いちばんひどい
    空転)を組の abort_resume_slip に入れて checks の slip = False、比が出ない走り直しを abort_resume_slip_unmeasured に入れて
    slip_all_measured = False、最初の周の記録が欠けた走り直し(first_lap_gap)を first_lap_gap_on に足して on_all_readable = False
    (衝突の abort_resume_on_collisions と同じ考え)。起点まで・失う・動く・LH の中央値からは今まで通り外す。abort_resume_excluded の
    行に、ON の走り直しの空転の比と最初の周の欠けを出す
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import label_events as le  # noqa: E402
try:
    from jetformer.assist import LAUNCH_END_NAMES, ASSIST_LAUNCH  # noqa: E402
except Exception:       # noqa: BLE001  発進の前進の無いコード
    LAUNCH_END_NAMES, ASSIST_LAUNCH = {}, 5

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
RAW = Path('/home/jetson/jetformer/data/raw')
LAB = Path('/home/jetson/jetformer/data/labels')
KV = 0.5e-3
STOP_ERPM = 400.0
MOVE1_ERPM, MOVE2_ERPM = 400.0, 800.0     # 動き出し: 0.2 m/s・0.4 m/s(StartKick.MOVE_ERPM と同じ 800)
DX = 0.1                                  # 距離の格子 [m]
MAX_TO_R1_S = 15.0
DEFAULT_DAYS = ('20261023', '20261024')   # 引数なしで読む日(今までと同じ)
BENCH = Path('/home/jetson/jetformer/data/bench_candidates.json')
SETTING_KEYS = ('throttle_gain', 'steer_gain', 'speed_ceiling', 'ceil_all_mps', 'ceil_left_mps', 'speed_auto', 'auto_target_mps',
                'launch_boost', 'launch_us', 'launch_mps', 'launch_max_s', 'zone_caps', 'start_kick')
# 試験の選択肢(2026-10-26 fix3 直す 1 回目 r1-rec3-5)。log.json の params に欄があるときだけ設定に入れる(欄の無い記録の行は前と同じ)
OPT_KEYS = ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick')
OPT_NAMES = {'rec_back_guard': '後ろの見張り', 'rec_hold_keep': 'リカバリーを残す', 'launch_no_kick': '見送りで補助なし'}
# 人側にした直後の衝突も「最初の周」に数える長さ: AI の区間の終わりのあとこれだけ(2026-10-27 fix3 r2-rec2-3。R1 に届いた発進の
# 'ai_end' と、届かなかった発進の 'no_r1'(r1-rec3-1。前は 0.2 秒)で同じ。本物の 10/23・10/24 の 50 本で、AI の区間の終わりから
# 0.5 秒以内の人側の衝突は 7 件。4G 以上の 4 件は 0.16〜0.35 秒あと(8G 以上 2 件は 0.34・0.35 秒)。fix3/fix_r2_2.md)
FLIP_TAIL_S = 0.5
NO_R1_TAIL_S = FLIP_TAIL_S
LINE_TOL_S = 0.2             # 線が AI の区間の終わりのこれだけあとまでなら「線まで AI 側」('line'。前のまま)
LOC_LH = 6                   # label_events.LOC_NAMES の LH(フレームの loc_code)
GAP_MAX_S = 0.5              # 欠けた記録(シーンの npz が無い)で、最初の周の間に IMU がこれより長く抜けたら first_lap_gap(r2-rec2-5)
LOST_HEAD_S, LOST_TAIL_S = 0.1, 0.005    # 欠けたシーンの間の発進の前進の start の幅(in_lost。2026-10-27 fix6)
CRIT = {'to_r1_gain_s': 0.15, 'slip_max': 1.3, 'lh_dv_max': 0.3}    # 発進の前進を使ってよいか(protocol_v3 14 章)
SLIP_IMU_MIN = 0.2           # IMU の前の加速の積分がこれ以下なら比を出さない(前からの値。standing_3lap.launch_slip と同じ)
SLIP_EXTREME_DV_WHEEL = 0.5  # 比が出ないのに車輪の速さがこれより増えた = いちばんひどい空転(> 1.3 とみなす。2026-10-27 r3-r3-rec-4)
MODEL = {
    'clean_driving_20261011-nocmd-s3-20261011-024626': 'B',
    'bench_20261022-slow-sec-s1': 'SLOW-SEC1',
    'bench_20261021-ens-p2': 'ENS',
    'bench_20261021-aug-s0': 'AUG',
    'clean_driving_20261011-act-s0-nocmd-stride3-e30': 'ACT',
    'bench_20261021-wgt-s0': 'WGT',
}


def logs(days=DEFAULT_DAYS):
    if tuple(days) == DEFAULT_DAYS:            # 今までと同じ(ラベルのある記録だけ)
        return sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name[:8] in ('20261023', '20261024')
                      and (LAB / (p.name + '.npz')).exists())
    out = []
    for p in sorted(RAW.iterdir()):
        if not p.is_dir() or len(p.name) != 15 or p.name[:8] not in days:
            continue
        meta = read_meta(p.name)
        dr = (meta or {}).get('drive') or {}
        if not dr or dr.get('dry_run'):
            continue                           # 読めない log.json(0 バイト)・人の収集・dry run
        out.append(p.name)
    return out


def read_meta(log):
    p = RAW / log / 'log.json'
    try:
        if p.stat().st_size == 0:
            return None
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def unreadable_meta_logs(days):
    """ほかの日(--day)の、log.json が読めない記録 [(名前, 理由)](2026-10-27 fix3 直す 3 回目 r3-r3-rec-1)。logs() は人の収集・
    dry run と見分けられないので飛ばす。電源断の本物の形(10/23 の 20261023-204825: log.json・scene.json・npz が全部 0 バイト)"""
    out = []
    for p in sorted(RAW.iterdir()):
        if not p.is_dir() or len(p.name) != 15 or p.name[:8] not in days:
            continue
        f = p / 'log.json'
        try:
            if f.stat().st_size == 0:
                out.append((p.name, 'log.json が 0 バイト'))
                continue
            json.loads(f.read_text(encoding='utf-8'))
        except FileNotFoundError:
            out.append((p.name, 'log.json が無い'))
        except OSError as e:
            out.append((p.name, 'log.json が読めない(%s)' % e.__class__.__name__))
        except ValueError:
            out.append((p.name, 'log.json が壊れている'))
    return out


def bench_blocks():
    """{日付: {block: run_id}} と {run: 候補の id}(bench_candidates.json。読めなければ空)"""
    try:
        b = json.loads(BENCH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}, {}
    days = {d.get('date'): {bl.get('block'): bl.get('run_id') for bl in d.get('blocks') or []} for d in b.get('days') or []}
    names = {c.get('run'): c.get('id') for c in b.get('candidates') or [] if c.get('run')}
    return days, names


def bench_plan_launch():
    """{日付: {block: 計画の発進の前進(expect.launch_boost。0 / 1 / None)}}(bench_candidates.json。読めなければ空)。
    読めなかった記録の ON/OFF を決めるため(2026-10-27 fix3 r2-rec2-5)"""
    try:
        b = json.loads(BENCH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return {d.get('date'): {bl.get('block'): (bl.get('expect') or {}).get('launch_boost') for bl in d.get('blocks') or []}
            for d in b.get('days') or []}


def unreadable_info(log, err, blocks, names, plans):
    """読めなかった記録(r2-rec2-5)の名前・理由・回・モデル・設定(log.json の drive.params)・発進の前進の ON/OFF。
    launch_state は log.json の launch_boost か計画(説明欄の block → expect.launch_boost)のどちらかが 1 なら 'on'"""
    meta = read_meta(log) or {}
    dr = meta.get('drive') or {}
    p = dr.get('params') or {}
    desc = meta.get('description') or ''
    run = dr.get('run') or ''
    day = '%s-%s-%s' % (log[:4], log[4:6], log[6:8])
    try:
        plan = (plans or {}).get(day, {}).get(int(desc_kv(desc, 'block') or 0))
    except ValueError:
        plan = None
    S = dict({k: p.get(k) for k in SETTING_KEYS}, **{k: p[k] for k in OPT_KEYS if k in p})
    try:
        g = round(float(S.get('throttle_gain')), 3)
    except (TypeError, ValueError):
        g = None
    return {'log': log, 'why': str(err), 'no': ai_no(desc), 'session': session_new(log, desc, blocks or {}),
            'model': (names or {}).get(run) or MODEL.get(run, run[:24]), 'gain': g, 'settings': S,
            'launch_param': S.get('launch_boost'), 'plan_launch': plan,
            'launch_state': 'on' if (_on(S.get('launch_boost')) or _on(plan)) else 'off'}


def unreadable_meta_info(log, why, readable, blocks):
    """log.json が読めない記録(r3-r3-rec-1)。回・モデル・設定・発進の前進の ON/OFF は分からない(meta = False・
    launch_state = 'unknown')。near = 同じ日の、前と次の読める AI の記録(名前(回の番号))。紙・動画とどの本かを照らす手がかり"""
    day = '%s-%s-%s' % (log[:4], log[4:6], log[6:8])

    def tag(x):
        if x is None:
            return None
        desc = (read_meta(x) or {}).get('description') or ''
        try:
            rid = (blocks or {}).get(day, {}).get(int(desc_kv(desc, 'block') or 0))
        except ValueError:
            rid = None
        return '%s(%s)' % (x, rid or 'block なし')
    same = sorted(x for x in readable if x[:8] == log[:8])
    prev = [x for x in same if x < log]
    nxt = [x for x in same if x > log]
    return {'log': log, 'why': why, 'no': None, 'session': None, 'model': None, 'gain': None, 'settings': {},
            'launch_param': None, 'plan_launch': None, 'launch_state': 'unknown', 'meta': False,
            'near': [tag(prev[-1] if prev else None), tag(nxt[0] if nxt else None)]}


def desc_kv(desc, k):
    for x in (desc or '').split(' memo=')[0].split():
        if x.startswith(k + '='):
            return x[len(k) + 1:]
    return None


def session_new(log, desc, blocks):
    """ほかの日の回: 説明欄の block → bench_candidates.json の run_id の頭(11/3 の「1-2」→ 回1)。分からなければ ?"""
    day = '%s-%s-%s' % (log[:4], log[4:6], log[6:8])
    md = '%s/%s' % (log[4:6], log[6:8])
    try:
        rid = blocks.get(day, {}).get(int(desc_kv(desc, 'block') or 0))
    except ValueError:
        rid = None
    return '%s 回%s' % (md, rid.split('-')[0]) if rid else '%s ?' % md


def launch_events(log, T0):
    """events.jsonl の launch_boost の出来事 [(時刻 [秒], start/skip/end, 理由)]"""
    out = []
    try:
        lines = (RAW / log / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        # 時刻は cam_epoch_us(AI_CMD と同じ UNIX µs。2026-10-26 fix2 から drive.py が書く)。無ければ cam_t_us・t_us を、
        # UNIX µs のときだけ(drive.py の cam_t_us は単調時計で、記録の時刻と比べられない。前もどの区間にも入らなかった)
        tu = e.get('cam_epoch_us')
        if tu is None:
            tu = e.get('cam_t_us', e.get('t_us'))
            if tu is not None and int(tu) < 10 ** 15:
                tu = None
        if e.get('kind') != 'launch_boost' or tu is None:
            continue
        out.append((round((int(tu) - T0) / 1e6, 3), str(e.get('event') or ''), str(e.get('reason') or e.get('code_reason') or '')))
    return out


def session_of(log, no):
    if log.startswith('20261023'):
        if no is None:
            return '10/23 前'
        return '10/23 前' if no <= 7 else ('10/23 公式' if no <= 15 else '10/23 後')
    if no is None:
        return '10/24 ?'
    return '10/24 1回目' if no <= 4 else ('10/24 2回目' if no <= 10 else '10/24 3回目')


def ai_no(desc):
    for x in (desc or '').split():
        if x.startswith('ai_no='):
            try:
                return int(x[6:])
            except ValueError:
                return None
    return None


def first_reach(vt, Df, target, lo_t, hi_t):
    """lo_t〜hi_t の間で前進の距離 Df が target に初めて届いた時刻(線形に補う。tools/standing_3lap.py の _first_reach と同じ)"""
    m = (vt >= lo_t) & (vt <= hi_t)
    if not m.any():
        return None
    tt, DD = vt[m], Df[m]
    if DD[-1] < target:
        return None
    k = int(np.searchsorted(DD, target, side='left'))
    if k <= 0:
        return float(tt[0])
    x0, x1 = DD[k - 1], DD[k]
    fr = (target - x0) / (x1 - x0) if x1 > x0 else 0.0
    return float(tt[k - 1] + fr * (tt[k] - tt[k - 1]))


def first_time(t, mask):
    i = np.nonzero(mask)[0]
    return float(t[i[0]]) if len(i) else None


def smooth(x, n):
    if len(x) < n:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, 'same')


def load(log, extended=False, blocks=None, names=None):
    """extended = ほかの日(--day): ラベルが無ければその場で作り、発進の前進の欄・設定・回を足す"""
    if not extended or ((LAB / (log + '.json')).exists() and (LAB / (log + '.npz')).exists()):
        lab = json.loads((LAB / (log + '.json')).read_text())
        z = dict(np.load(LAB / (log + '.npz')))
        lab_src = 'file'
    else:
        lab, z = le.label_log(log, RAW)
        if z is None:
            raise ValueError(lab.get('error') or 'ラベルを作れません')
        lab_src = 'computed'
    L = le.load_log(log, RAW)
    T0 = int(lab['t0_us'])
    V, I = L['VEHICLE'], L['IMU']
    vt = (V['timestamps_us'].astype(np.int64) - T0) / 1e6
    verpm = V['erpm'].astype(np.float64)
    it = (I['timestamps_us'].astype(np.int64) - T0) / 1e6
    ie = np.interp(it, vt, verpm)
    gz = I['gyro'][:, 2].astype(np.float64)
    still = (np.abs(ie) < 50) & (np.abs(gz) < 5)
    hg, bias = le.horizontal_g(I['accel'], still)
    hd, gzu = le.heading_deg(it, gz, still)
    impacts = le.detect_collisions(it, hg)
    hits = [(round(float(it[e['i']]), 3), e['g']) for e in impacts if not e['suspect']]
    anchors, gm, gzs, rej = le.checked_anchors(it, hd, gzu, vt, verpm, z['t_s'], z['ai'].astype(bool), hits)
    D, Df = le.cumulative_distance(vt, verpm)
    ay = I['accel'][:, 1].astype(np.float64) - bias[1]      # 前 + [m/s²](止まっているときのずれを引く)
    d = {'log': log, 'lab': lab, 'z': z, 'meta': L['meta'], 'vt': vt, 'erpm': verpm, 'v': -verpm * KV, 'Df': Df,
         'it': it, 'gz': gzu, 'hd': hd, 'ay': ay, 'hg': hg, 'hits': hits, 'anchors': np.asarray(anchors, float),
         'ref': (float(lab['ref_us']['steer_center']), float(lab['ref_us']['throttle_neutral']))}
    if extended:
        A = L['AI_CMD']
        lon = lend = None
        if A is not None and 'launch_on' in A and 't_us' in z:
            ts = np.asarray(A['timestamps_us'], np.int64)
            fu = np.asarray(z['t_us'], np.int64)
            k = np.clip(np.searchsorted(ts, fu), 0, len(ts) - 1)
            ok = ts[k] == fu
            lon = np.where(ok, np.asarray(A['launch_on'], bool)[k], False)
            lend = np.where(ok, np.asarray(A['launch_end'], np.int64)[k], 0) if 'launch_end' in A else np.zeros(len(fu), np.int64)
        p = lab.get('params_at_start') or {}
        d.update(extended=True, labels_src=lab_src, launch_on=lon, launch_end=lend, launch_ev=launch_events(log, T0),
                 settings=dict({k: p.get(k) for k in SETTING_KEYS}, **{k: p[k] for k in OPT_KEYS if k in p}),
                 gm=np.asarray(gm, float), gzs=gzs,
                 session=session_new(log, lab.get('description'), blocks or {}),
                 model_name=(names or {}).get(lab.get('run') or '') or MODEL.get(lab.get('run') or '', (lab.get('run') or '')[:24]),
                 problems=list(L.get('problems') or []),
                 # r2-rec2-5: シーンの npz が無い・読めない(電源断・落ちで最後のシーンが書かれなかったなど)
                 data_missing=any(('ありません' in x or '読めません' in x) for x in (L.get('problems') or [])),
                 scenes=list(L.get('scenes') or []),      # 2026-10-27 fix6: lost_scenes が使う(JSON には書かない)
                 midrun=midrun_of(log, L, lab, z, T0, lend),
                 trial=trial_of(log, lab))
    return d


def lost_scenes(d):
    """2026-10-27 fix6(反証 6-1 回目 w1-tools-r6-1-tools-1。ほかの日だけ): シーンが欠けた本(load_log の problems)の、フレーム
    (AI_CMD。AI_CMD の無い本はカメラ)が 1 つも無い欠けたシーン [(シーンの番号 4 桁, 前のフレームの時刻, 次のフレームの時刻)]
    (記録の頭からの秒。前・次のシーンにフレームが無ければ None)。フレームの無いシーンは人側・AI 側が分からない(AI 側にした
    シーンごと書けずに落ちた形)。欠けていても AI_CMD のフレームがあるシーン(VEHICLE だけ無いなど)は AI_CMD の select で分かるので
    入れない。ラベルにフレームのシーンの欄(scene)が無ければ、欠けたシーンは全部「最後のフレームのあと」とみなす"""
    log, z = d['log'], d['z']
    names = []
    for x in d.get('problems') or []:
        s = x.split('/')[0]
        if s not in names:
            names.append(s)
    if not names:
        return []
    t = np.asarray(z['t_s'], float)
    if 'scene' not in z or not d.get('scenes'):
        return [(s[len(log) + 1:], float(t[-1]) if len(t) else None, None) for s in names]
    fs = np.asarray(z['scene'], np.int64)
    idx = {s: i for i, s in enumerate(d['scenes'])}
    out = []
    for s in names:
        i = idx.get(s)
        if i is None or bool((fs == i).any()):
            continue
        lo, hi = t[fs < i], t[fs > i]
        out.append((s[len(log) + 1:], float(lo.max()) if len(lo) else None, float(hi.min()) if len(hi) else None))
    return out


def in_lost(te, ls):
    """時刻 te [秒] が欠けたシーン(lost_scenes)の間か。前のフレームの LOST_HEAD_S 前より後(そこからの読める AI の区間は 0.1 秒
    未満で、launch_rows が発進の行にしない)・次のフレームの LOST_TAIL_S 前より前(launch_events の時刻は 1 ms に丸めるので、次の
    シーンの最初のフレームの start = 読める発進を入れない。フレームの間は 30 fps で約 33 ms)"""
    return any((lo is None or te > lo - LOST_HEAD_S) and (hi is None or te < hi - LOST_TAIL_S) for _, lo, hi in ls)


def launch_starts(d, ls=None):
    """events.jsonl の発進の前進の start(出した)の時刻 [秒](2026-10-27 fix6)。ls(lost_scenes)を渡せば、欠けたシーンの間
    (in_lost)の start だけ。drive.py は発進の前進が ON のときだけ、AI 側にした最初のフレームで出せば start(見送れば skip。
    見送った発進は読めても ON に数えない = on_not_started)を書き、書いたときにファイルへ足す(recorder.add_event)ので、その
    シーンの npz が書けずに jetformer-web が落ちても残る。ただし fsync しないので、電源断では落ちる前の約 30 秒に書いた行が消える
    (2026-10-27 fix6-2。lost_tail・launch_on_from で拾う)。時刻の無い出来事は見ない(launch_events と同じ)"""
    return [te for te, ev, _ in d.get('launch_ev') or [] if ev == 'start' and (ls is None or in_lost(te, ls))]


def lost_tail(ls):
    """lost_scenes のうち、最後の読めるフレームのあと(あとのシーンにフレームが無い)の欠けたシーン [(番号 4 桁, 最後の読める
    フレームの時刻)](2026-10-27 fix6-2)。電源断・jetformer-web の落ちは記録の終わりを欠く(recorder はフレームの無いシーンを
    ふつうに止めたときは消すので、終わりの欠けは落ちた形)。真ん中だけの欠け(あとのシーンが読める)は入れない"""
    return [(s, lo) for s, lo, hi in ls if hi is None]


def launch_event_kinds(log):
    """events.jsonl の発進の前進の出来事(kind = launch_boost)の event の並び(時刻の無い行も。読めない行・NUL の行は飛ばす)"""
    try:
        lines = (RAW / log / 'events.jsonl').read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get('kind') == 'launch_boost':
            out.append(str(e.get('event') or ''))
    return out


def launch_on_from(d, u):
    """発進の前進 ON の手がかり [文](2026-10-27 fix6-2。[] = 手がかりなし = OFF)。log.json の launch_boost・計画(unreadable_info の
    launch_param・plan_launch。前と同じ)・events.jsonl の発進の前進の出来事(start・skip・end。drive.py は発進の前進が ON のとき
    だけ書く)・走行中に launch_boost を 1 にした(midrun: events.jsonl の params・log.json の params_end)"""
    out = []
    if _on(u.get('launch_param')):
        out.append('log.json')
    if _on(u.get('plan_launch')):
        out.append('計画')
    ev = launch_event_kinds(d['log'])
    if ev:
        out.append('events.jsonl の発進の前進の出来事(%s)' % '・'.join(sorted(set(ev), key=ev.index)))
    mid = [m for m in d.get('midrun') or [] if m.get('key') == 'launch_boost' and _on(m.get('to'))]
    if mid:
        out.append('走行中に launch_boost を 1 に%s' % ('(%.1f 秒)' % mid[0]['t_s'] if mid[0].get('t_s') is not None else ''))
    return out


def starts_text(ts):
    """start の時刻の文(25.00 秒・45.00 秒)"""
    return '・'.join('%.2f 秒' % x for x in ts)


def trial_of(log, lab):
    """人が説明欄に「試し」と書いた本なら memo の文、でなければ None(2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-3。ほかの日
    だけ)。説明欄 = log.json(ラベルの description)と走り終わりのメモ(scene.json)をまとめたもの。自動の「計画と違うので試し(元
    cand=…)」は数えない(tools/standing_3lap.trial_memo)"""
    import section_scorecard as ss  # noqa: E402  ほかの日だけ読む
    import standing_3lap as s3  # noqa: E402
    desc, _ = ss.effective_description(log, lab.get('description') or '', RAW)
    return s3.trial_memo(desc)


def first_lap_gap(d, a, b):
    """欠けた記録(data_missing)で、最初の周 a〜b 秒の間に IMU が途切れた(b より前に終わった・GAP_MAX_S より長く抜けた)なら
    その文、でなければ None(r2-rec2-5。欠けていない記録は見ない = 前と同じ)"""
    if not d.get('data_missing'):
        return None
    it = d['it']
    if not len(it) or float(it[-1]) < b - 0.05:
        return '記録が %.2f 秒で終わった(最初の周の終わり %.2f 秒の前。シーンが欠けている)' % (float(it[-1]) if len(it) else 0.0, b)
    m = (it >= a) & (it <= b)
    tt = it[m]
    if len(tt) < 2 or float(np.diff(tt).max()) > GAP_MAX_S or float(tt[0]) - a > GAP_MAX_S:
        return 'IMU が最初の周の間で抜けている(シーンが欠けている)'
    return None


def midrun_of(log, L, lab, z, T0, lend):
    """走行中に変えた設定 [{key, from, to, t_s, src}](2026-10-26 fix3 直す 1 回目 r1-rec3-3)。tools/standing_3lap.midrun_changes を
    そのまま使う(③ standing_3lap・④ by_params と同じ見方): events.jsonl の params・log.json の drive.params_end・AI_CMD の
    天井・自動合わせが途中から OFF(section_scorecard.speed_ctrl_summary)・発進の前進の終わり 10・AI のフレームのゲイン・トリム
    が途中で変わった。ほかの日(--day)だけ。無ければ []"""
    import section_scorecard as ss  # noqa: E402  ほかの日だけ読む(10/23・10/24 の道は前のまま)
    import standing_3lap as s3  # noqa: E402
    t = np.asarray(z['t_s'], np.float64)
    ai = np.asarray(z['ai'], bool) & ~np.asarray(z['dry_run'], bool) if 'dry_run' in z else np.asarray(z['ai'], bool)
    params = lab.get('params_at_start') or {}
    spc = ss.speed_ctrl_summary(L.get('AI_CMD'), T0, t, ai, params)
    standing = []                  # 発進の前進の終わり 10(OFF にした): standing_3lap は発進の行の end_code で見る。同じ時刻を渡す
    if lend is not None:
        k = np.nonzero(ai & (np.asarray(lend) == 10))[0]
        if len(k):
            standing = [{'launch': {'end_code': 10, 'end_s': 0.0}, 't_ai': float(t[k[0]])}]
    meta = L.get('meta') or {'drive': {'params': params}}
    return s3.midrun_changes(meta, {'speed_ctrl': spc}, z, ai, standing, RAW, log, T0)


def midrun_text(mid):
    """走行中に変えた設定の短い文(standing_3lap.midrun_text と同じ)"""
    import standing_3lap as s3  # noqa: E402
    return s3.midrun_text(mid)


def _on(x):
    try:
        return int(float(x)) == 1
    except (TypeError, ValueError):
        return False


def gain_key(S, frame_gain):
    """組の鍵のゲイン。自動合わせ ON の本はフレームのゲインが電圧で本ごとに変わる(0.965・0.972…)ので、
    log.json の throttle_gain(スライダーの値)で組にする(2026-10-26 critic、反証 1 回目 R1・tools:F1: 回 3 を
    天井 + 自動合わせで走ると OFF と ON が別の組になり、比べが出なかった)。OFF の本は今まで通りフレームのゲイン"""
    S = S or {}
    if _on(S.get('speed_auto')):
        try:
            return round(float(S.get('throttle_gain')), 3)
        except (TypeError, ValueError):
            pass
    return frame_gain


def extended_launch(d, seg, t0, t1, ts):
    """ほかの日の発進の前進の欄(AI_CMD の launch_on・launch_end・assist = 5、events.jsonl)"""
    z = d['z']
    S = d['settings']
    a_seg = z['assist'][seg]
    row = {'session_key': d['session'], 'settings': S, 'launch_param': S.get('launch_boost'),
           'launch_recorded': d['launch_on'] is not None, 'launch_frames': int(np.sum(a_seg == ASSIST_LAUNCH)),
           'launch_on_frames': None, 'launch_on_s': None, 'launch_end_code': None, 'launch_end_reason': None,
           'launch_end_s': None, 'labels_src': d['labels_src'],
           'launch_events': [list(e) for e in d['launch_ev'] if t0 - 0.2 <= e[0] <= t1]}
    if d['launch_on'] is not None:
        on = d['launch_on'][seg]
        end = d['launch_end'][seg]
        row['launch_on_frames'] = int(on.sum())
        if on.any():
            row['launch_on_s'] = round(float(ts[on][-1] - t0) + 1 / 30.0, 3)
        k = np.nonzero(end > 0)[0]
        if len(k):
            c = int(end[k[0]])
            row.update(launch_end_code=c, launch_end_reason=LAUNCH_END_NAMES.get(c, str(c)), launch_end_s=round(float(ts[k[0]] - t0), 3))
    started = bool(row['launch_on_frames']) or any(e[1] == 'start' for e in row['launch_events'])
    row['launch_state'] = 'off' if not _on(S.get('launch_boost')) else ('on' if started else 'on_not_started')
    return row


def ai_continuous(z, a, b):
    t, ai = z['t_s'], z['ai'].astype(bool)
    m = (t >= a - 0.05) & (t <= b + 0.05)
    if not m.any():
        return False
    tt = t[m]
    return bool(ai[m].all() and (np.diff(tt).max(initial=0) <= 0.2) and tt[0] <= a + 0.1 and tt[-1] >= b - 0.1)


def laps_of(d):
    """続いた周: 起点 a → 次の起点 b(ずっと AI・5〜16 秒・向きの変化 300〜420°)"""
    A, out = d['anchors'], []
    for a, b in zip(A[:-1], A[1:]):
        if not (5.0 <= b - a <= 16.0):
            continue
        dpsi = float(np.interp(b, d['it'], d['hd']) - np.interp(a, d['it'], d['hd']))
        if not (300 <= dpsi <= 420):
            continue
        if not ai_continuous(d['z'], a, b):
            continue
        out.append((float(a), float(b)))
    return out


def frame_vals(d, tq):
    """フレーム(30 Hz)の値を時刻 tq に(直前のフレーム)"""
    z = d['z']
    t = z['t_s']
    k = np.clip(np.searchsorted(t, tq, side='right') - 1, 0, len(t) - 1)
    c, n = d['ref']
    return {'f_out': n - z['out'][k, 1], 'f_raw': n - z['raw'][k, 1], 'f_model': n - z['model_out'][k, 1],
            's_out': z['out'][k, 0] - c, 'assist': z['assist'][k], 'loc': z['loc_code'][k], 'gain': z['gains'][k, 1]}


def lap_profile(d, a, b, dmax):
    """周の終わりの起点 b から手前 dmax m までを、起点までの距離 r の格子に: 起点までの時間・速さ・指令・舵・加速"""
    vt, Df = d['vt'], d['Df']
    m = (vt >= a) & (vt <= b)
    tt, DD = vt[m], Df[m]
    Db = float(np.interp(b, vt, Df))
    r_all = Db - DD                                # 起点までの距離(減っていく)
    r = np.arange(0.0, dmax + 1e-9, DX)
    ok = r <= r_all[0]
    # r → 時刻(r は単調に減る。止まったところは同じ値なので、最後に通った時刻)
    rr, ti = r_all[::-1], tt[::-1]
    t_at = np.interp(r, rr, ti)
    t_at[~ok] = np.nan
    fv = frame_vals(d, np.nan_to_num(t_at, nan=a))
    v = np.interp(np.nan_to_num(t_at, nan=a), vt, d['v'])
    ay = np.interp(np.nan_to_num(t_at, nan=a), d['it'], smooth(d['ay'], 9))
    nanify = lambda x: np.where(ok, x, np.nan)   # noqa: E731
    return {'r': r, 'tau': nanify(b - t_at), 'v': nanify(v), 'f_out': nanify(fv['f_out']), 'f_raw': nanify(fv['f_raw']),
            's_out': nanify(fv['s_out']), 'ay': nanify(ay), 'loc': np.where(ok, fv['loc'], -1)}


def last_lh_start(loc):
    """発進の格子の場所(loc_code)で、R1 の前の最後の LH(LOC_LH)の続きの始めの番号。LH が無ければ None(r2-rec2-2)"""
    is_lh = np.asarray(loc) == LOC_LH
    ks = np.nonzero(is_lh)[0]
    if not len(ks):
        return None
    k = int(ks[-1])
    while k > 0 and is_lh[k - 1]:
        k -= 1
    return k


def slip_window(d, t_a, t_b):
    """車輪の速さの増え と IMU の前の加速の積分 の比(1 より大きい = 車輪が先に回った = 空転の疑い)"""
    vt, it = d['vt'], d['it']
    if t_b - t_a < 0.15:
        return None
    dv_w = float(np.interp(t_b, vt, d['v']) - np.interp(t_a, vt, d['v']))
    m = (it >= t_a) & (it <= t_b)
    if m.sum() < 5:
        return None
    dv_i = float(np.trapz(d['ay'][m], it[m]))
    return {'t_a': round(t_a, 3), 't_b': round(t_b, 3), 'dv_wheel': round(dv_w, 3), 'dv_imu': round(dv_i, 3),
            'ratio': round(dv_w / dv_i, 3) if dv_i > SLIP_IMU_MIN else None}


def slip_state(sl):
    """空転の比の読み(2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-4。tools/standing_3lap.slip_kind と同じ決め方):
    'ok' ≤ 1.3 / 'high' > 1.3 / 'extreme' 比が出ない(IMU の積分 ≤ 0.2 m/s)のに車輪が 0.5 m/s より増えた(> 1.3 とみなす)/
    'unknown' 比が出ない(slip が無い = 窓が短い・IMU が抜けた、か車輪も IMU もほとんど増えない)"""
    if not sl:
        return 'unknown'
    r = sl.get('ratio')
    if r is not None:
        return 'high' if r > CRIT['slip_max'] else 'ok'
    dw, di = sl.get('dv_wheel'), sl.get('dv_imu')
    if dw is not None and di is not None and dw > SLIP_EXTREME_DV_WHEEL and di <= SLIP_IMU_MIN:
        return 'extreme'
    return 'unknown'


# tools/standing_3lap.RESUME_NAME と同じ(fix7 r1 の名前のまま = 手順書 7 章・8.1 が引く言葉。r2 で中立止め・途切れのあとも入る)
RESUME_NAME = 'リカバリー中止のあとの走り直し'


def abort_text(ab):
    """after_abort の止まった形の文(2026-10-27 fix7 r2。standing_3lap.abort_text と同じ形。秒は印の値のまま):
    「中止 X 秒」(fix7 r1 の文のまま)/「中立止め X 秒から Y 秒」/「カメラの途切れ X 秒から Y 秒」(kind の無い印は中止)"""
    kind = ab.get('kind') or 'abort'
    if kind == 'hold':
        return '中立止め %s 秒から %s 秒' % (ab.get('abort_s'), ab.get('hold_s'))
    if kind == 'stall':
        return 'カメラの途切れ %s 秒から %s 秒' % (ab.get('abort_s'), ab.get('gap_s'))
    return '中止 %s 秒' % ab.get('abort_s')


def slip_text(sl):
    """発進の空転の比の文(2026-10-27 fix7 r2。走り直しの ON の行に出す)"""
    k = slip_state(sl)
    if k in ('ok', 'high'):
        return '空転の比 %.2f%s' % (sl['ratio'], ' > %.1f' % CRIT['slip_max'] if k == 'high' else '')
    if k == 'extreme':
        return '空転の比が出ない(車輪 +%.2f・IMU +%.2f m/s = いちばんひどい空転とみなす > %.1f)' % (
            sl['dv_wheel'], sl['dv_imu'], CRIT['slip_max'])
    return '空転の比が出ない(%s)' % ('車輪 +%.2f・IMU +%.2f m/s。どちらもほとんど増えない' % (sl['dv_wheel'], sl['dv_imu'])
                                    if sl else '窓が短い(LH・R1 が近い)か IMU が抜けた')


def launch_rows(d, laps_ref, ref_kind):
    z, lab = d['z'], d['lab']
    t, ai = z['t_s'], z['ai'].astype(bool)
    vt, it = d['vt'], d['it']
    c, n = d['ref']
    no = ai_no(lab.get('description'))
    run = lab.get('run') or ''
    model = MODEL.get(run, run[:24]) if not d.get('extended') else d['model_name']
    rows = []
    ab = {}
    if d.get('extended'):              # 2026-10-27 fix7: リカバリー中止のあとの走り直し(ほかの日だけ。中止の無い本は {})
        import standing_3lap as s3  # noqa: E402  ほかの日だけ読む(10/23・10/24 の道は前のまま)
        # 2026-10-27 fix7 r2: 車輪の eRPM も渡す(中立止め・カメラの途切れのまま止まって終わった区間のあとも)
        ab = s3.abort_resumes(t, ai, z.get('assist'), z.get('hold'), z.get('ai_cmd_select'), vt, d['Df'], d['erpm'])
    for i0, i1 in le.runs(ai):
        t0, t1 = float(t[i0]), float(t[i1])
        e0 = float(np.interp(t0, vt, d['erpm']))
        if abs(e0) >= STOP_ERPM or t1 - t0 < 0.1:
            continue
        seg = slice(i0, i1 + 1)
        a_seg = z['assist'][seg]
        kick = np.nonzero(a_seg == 4)[0]
        esc0 = np.nonzero(a_seg[:15] == 1)[0]             # 最初の 0.5 秒の「止まっている間のブレーキ → 中立」
        g = float(np.median(z['gains'][seg, 1]))
        mv = (vt >= t0) & (vt <= t1)
        t_m1 = first_time(vt[mv], -d['erpm'][mv] >= MOVE1_ERPM)
        t_m2 = first_time(vt[mv], -d['erpm'][mv] >= MOVE2_ERPM)
        # モデルが前へ出すまで(ゲイン後の model_out で 80 µs 以上・raw で 80 µs 以上)
        fm = n - z['model_out'][seg, 1]
        fr = n - z['raw'][seg, 1]
        ts = t[seg]
        t_model80 = first_time(ts, fm >= 80)
        t_model40 = first_time(ts, fm >= 40)
        row = {'log': d['log'], 'no': no, 'session': session_of(d['log'], no) if not d.get('extended') else d['session'],
               'model': model, 'gain': round(g, 3),
               'start_kick_param': (lab.get('params_at_start') or {}).get('start_kick'),
               't_ai': round(t0, 3), 'erpm_at_ai': round(e0, 0), 'ai_seg_s': round(t1 - t0, 2),
               'kick': bool(len(kick)), 'kick_frames': int(len(kick)), 'kick_s': round(len(kick) / 30.0, 3),
               'kick_end_s': round(float(ts[kick[-1]] - t0) + 1 / 30.0, 3) if len(kick) else None,
               'esc_neutral_frames_0_5s': int(len(esc0)),
               'move_0p2_s': round(t_m1 - t0, 3) if t_m1 is not None else None,
               'move_0p4_s': round(t_m2 - t0, 3) if t_m2 is not None else None,
               'model80_s': round(t_model80 - t0, 3) if t_model80 is not None else None,
               'model40_s': round(t_model40 - t0, 3) if t_model40 is not None else None,
               'f_model_first5': [round(float(x), 0) for x in fm[:5]],
               'f_raw_first5': [round(float(x), 0) for x in fr[:5]],
               # その記録の何回目の発進か・それより前の衝突(2026-10-26 fix2 R2-rec-3。人が止めて戻したあとの発進を見分ける)
               'start_no': len(rows) + 1, 'hits_before': sum(1 for h in d.get('hits') or [] if h[0] < t0)}
        # 最初の 1.0 秒・1.5 秒の指令(車に出した前進・モデルの前進(ゲイン後)・ゲイン前)
        for w in (0.5, 1.0, 1.5):
            mm = (ts - t0) < w
            row['f_out_mean_%.1fs' % w] = round(float(np.mean(n - z['out'][seg, 1][mm])), 1)
            row['f_model_mean_%.1fs' % w] = round(float(np.nanmean(fm[mm])), 1)
            row['s_out_absmean_%.1fs' % w] = round(float(np.nanmean(np.abs(z['out'][seg, 0][mm] - c))), 1)
        if d.get('extended'):
            row.update(extended_launch(d, seg, t0, t1, ts))
            if d.get('midrun'):        # 走行中に変えた本(r1-rec3-3。まとめの比べに入れない)
                row['midrun'] = d['midrun']
            if d.get('trial'):         # 人が memo に「試し」と書いた本(r3-r3-rec-3。まとめの比べに入れない。あるときだけ)
                row['trial'] = d['trial']
            if i0 in ab:               # 2026-10-27 fix7: リカバリー中止のあとの走り直し(まとめの比べに入れない。あるときだけ)
                row['after_abort'] = ab[i0]
        nxt = [a for a in d['anchors'] if t0 < a <= t1]
        if not nxt or nxt[0] - t0 > MAX_TO_R1_S:
            row['to_r1_s'] = None
            row['why_no_r1'] = 'AI の区間が短い' if t1 - t0 < 3 else '15 秒以内に起点なし'
            if d.get('extended'):      # r1-rec3-1: R1 の前の衝突(当たって人側にした発進)を落とさない
                te = min(t1 + NO_R1_TAIL_S, float(nxt[0])) if nxt else t1 + NO_R1_TAIL_S
                fl = [h for h in d['hits'] if t0 <= h[0] < te]
                locs = le.locate([h[0] for h in fl], d['anchors'], d['gm'], d['it'], d['hd'], d['gzs']) if fl else []
                row['collisions_first_lap'] = [[h[0], h[1], le.LOC_DETAIL[int(k)]] for h, k in zip(fl, locs)]
                row['first_lap_end_s'] = round(te, 3)
                row['first_lap_end'] = 'no_r1'
                row['first_lap_full'] = False
                gap = first_lap_gap(d, t0, te)          # r2-rec2-5(欠けた記録だけ。あるときだけ)
                if gap:
                    row['first_lap_gap'] = gap
            rows.append(row)
            continue
        a1 = float(nxt[0])
        D0 = float(np.interp(t0, vt, d['Df']))
        dist = float(np.interp(a1, vt, d['Df']) - D0)
        col = [h for h in d['hits'] if t0 <= h[0] < a1]
        rec = bool(np.any(z['assist'][seg][(ts < a1)] == 2))
        row.update({'to_r1_s': round(a1 - t0, 3), 'dist_m': round(dist, 2), 'collisions': col, 'recovery': rec,
                    'dpsi_deg': round(float(np.interp(a1, it, d['hd']) - np.interp(t0, it, d['hd'])), 1)})
        if d.get('extended'):          # 最初の周(発進 → 1 周目の線。線までに AI の区間が終われば、その終わり)の衝突と場所
            # 2026-10-26 fix3(tools:F9): tools/standing_3lap.py の 1 周目と同じ。A2 = A1 の次の起点(5〜16 秒あと。standing_3lap の
            # 起点のつながりと同じ)、線 = A2 から d_pre(動いた所 → A1 の車輪の距離)戻ったところ。前は A2(AI の区間の中)まで
            nx = [float(a) for a in d['anchors'] if a > a1]
            a2 = nx[0] if nx and 5.0 <= nx[0] - a1 <= 16.0 else None
            # 2026-10-27 fix3 r2-rec2-3: 線の前に AI の区間が終わったら、終わり + FLIP_TAIL_S(その前に線があれば線)まで。人が当たる
            # 直前に人側にした衝突(惰性で当たる)も数える(届かなかった発進の 'no_r1' と同じ足し。前は AI の最後のフレームまで)
            te, kind = t1 + FLIP_TAIL_S, 'ai_end'
            if a2 is not None:
                ts_ = t_m1 if t_m1 is not None else t0
                d_pre = dist - (float(np.interp(ts_, vt, d['Df'])) - D0)
                tl = first_reach(vt, d['Df'], float(np.interp(a2, vt, d['Df'])) - d_pre, ts_, a2)
                if tl is not None and tl <= t1 + LINE_TOL_S:
                    te, kind = tl, 'line'
                elif tl is not None and tl < te:      # 線が AI の区間の終わりの 0.2〜0.5 秒あと: 最初の周は線で終わる
                    te = tl
            fl = [h for h in d['hits'] if t0 <= h[0] < te]
            locs = le.locate([h[0] for h in fl], d['anchors'], d['gm'], d['it'], d['hd'], d['gzs']) if fl else []
            row['collisions_first_lap'] = [[h[0], h[1], le.LOC_DETAIL[int(k)]] for h, k in zip(fl, locs)]
            row['first_lap_end_s'] = round(te, 3)
            row['first_lap_end'] = kind
            row['first_lap_a2_s'] = round(a2, 3) if a2 is not None else None
            row['first_lap_full'] = kind == 'line'       # 1 周目の線まで AI 側(前は「A2 が AI の区間の中」)
            gap = first_lap_gap(d, t0, te)              # r2-rec2-5(欠けた記録だけ。あるときだけ)
            if gap:
                row['first_lap_gap'] = gap
        # 発進の距離の格子(発進から x m)
        m = (vt >= t0) & (vt <= a1)
        tt, DD = vt[m], d['Df'][m] - D0
        x = np.arange(0.0, dist + 1e-9, DX)
        # x に初めて届いた時刻(DD は単調に増える)
        idx = np.searchsorted(DD, x, side='left')
        idx = np.clip(idx, 1, len(DD) - 1)
        x0, x1 = DD[idx - 1], DD[idx]
        frac = np.where(x1 > x0, (x - x0) / np.maximum(x1 - x0, 1e-9), 0.0)
        tL = tt[idx - 1] + frac * (tt[idx] - tt[idx - 1])
        tL[0] = t0 if t_m1 is None else first_time(tt, DD > 0.005) or t0
        fvL = frame_vals(d, tL)
        prof = {'x': x, 't': tL - t0, 'v': np.interp(tL, vt, d['v']), 'f_out': fvL['f_out'], 'f_raw': fvL['f_raw'],
                'f_model': fvL['f_model'], 's_out': fvL['s_out'], 'assist': fvL['assist'],
                'ay': np.interp(tL, it, smooth(d['ay'], 9)), 'loc': fvL['loc']}
        lh_self = None
        if d.get('extended'):          # r2-rec2-2: 発進のフレームの場所(loc_code の LH)で決めた LH の入口(周が無くても出る)
            # R1 の前の最後の LH の続き(R1 に入る直前の左ヘアピン)の始め。起点より前のフレームの場所は向きから決めるので、置いた
            # ところ・SW の出口が一瞬 LH になることがある(本物の 20261024-160847 は 0〜0.16 m が LH)→ 最初の LH は使わない
            k_s = last_lh_start(prof['loc'])
            if k_s is not None:
                lh_self = {'x_m': round(float(x[k_s]), 2), 'v_mps': round(float(prof['v'][k_s]), 2),
                           't_s': round(float(prof['t'][k_s]), 3)}
            row['_lh_self'] = lh_self   # 中だけの欄(JSON には書かない。まとめが使う)
        # 走りながら(周の終わりの起点の dist m 手前 → 起点)
        P = []
        for (la, lb, dd) in laps_ref:
            p = lap_profile(dd, la, lb, dist)
            if np.isfinite(p['tau'][-1 if len(p['tau']) else 0]):
                P.append(p)
        row['fly_n'] = len(P)
        row['fly_ref'] = ref_kind
        if d.get('extended') and not P:      # 周の無い発進(r2-rec2-2。前は LH の欄が無かった)
            row['lh_self'] = lh_self
        if P:
            # 発進の x = 起点までの距離 r = dist − x
            r_of_x = dist - x
            def at(p, key):
                return np.interp(r_of_x, p['r'], p[key])
            tauF = np.nanmedian(np.stack([at(p, 'tau') for p in P]), axis=0)
            T_fly = float(np.nanmedian([p['tau'][-1] for p in P]))     # 起点の dist m 手前 → 起点
            tF = T_fly - tauF                                           # 周の、発進の場所から x m までの時間
            F = {k: np.nanmedian(np.stack([at(p, k) for p in P]), axis=0) for k in ('v', 'f_out', 'f_raw', 's_out', 'ay')}
            locF = np.stack([np.round(at(p, 'loc')) for p in P])
            loc_mode = np.array([np.bincount(col_.astype(int)[col_ >= 0], minlength=7).argmax() if (col_ >= 0).any() else -1
                                 for col_ in locF.T])
            loss_x = prof['t'] - tF
            row.update({'fly_s': round(T_fly, 3), 'loss_s': round(a1 - t0 - T_fly, 3),
                        'start_loc': le.LOC_NAMES[int(loc_mode[0])] if loc_mode[0] >= 0 else None})
            # LH の入口(周の場所で LH が始まる x)
            lh = np.nonzero(loc_mode == 6)[0]
            x_lh = float(x[lh[0]]) if len(lh) else None
            row['x_lh_m'] = round(x_lh, 2) if x_lh is not None else None
            # 失う時間の内訳: 動き出すまで(x = 0)・直線(x_lh まで)・LH〜起点
            k_lh = int(lh[0]) if len(lh) else len(x) - 1
            row['loss_parts'] = {'dead_s': round(float(loss_x[0]), 3),
                                 'straight_s': round(float(loss_x[k_lh] - loss_x[0]), 3),
                                 'lh_to_r1_s': round(float((a1 - t0 - T_fly) - loss_x[k_lh]), 3)}
            # 周の速さに追いつくところ(発進の速さ ≥ 周の速さ − 0.1 m/s)
            catch = np.nonzero(prof['v'] >= F['v'] - 0.1)[0]
            catch = catch[catch > 2] if len(catch) else catch
            row['catch_x_m'] = round(float(x[catch[0]]), 2) if len(catch) else None
            row['v_fly_at_start'] = round(float(F['v'][0]), 2)
            row['v_fly_at_lh'] = round(float(F['v'][k_lh]), 2)
            row['v_launch_at_lh'] = round(float(prof['v'][k_lh]), 2)
            row['t_launch_to_lh'] = round(float(prof['t'][k_lh]), 3)
            row['t_fly_to_lh'] = round(float(tF[k_lh]), 3)
            # 直線の指令の比べ(発進 vs 周、同じ場所。x が 0.3 m〜x_lh)
            sm = (x >= 0.3) & (x <= (x_lh if x_lh is not None else dist))
            if sm.any():
                row['straight_f_out_launch'] = round(float(np.nanmean(prof['f_out'][sm])), 1)
                row['straight_f_out_fly'] = round(float(np.nanmean(F['f_out'][sm])), 1)
                row['straight_f_raw_launch'] = round(float(np.nanmean(prof['f_raw'][sm])), 1)
                row['straight_f_raw_fly'] = round(float(np.nanmean(F['f_raw'][sm])), 1)
                row['straight_s_abs_launch'] = round(float(np.nanmean(np.abs(prof['s_out'][sm]))), 1)
                row['straight_s_abs_fly'] = round(float(np.nanmean(np.abs(F['s_out'][sm]))), 1)
                row['straight_ay_launch'] = round(float(np.nanmean(prof['ay'][sm])), 2)
                row['straight_v_launch'] = round(float(np.nanmean(prof['v'][sm])), 2)
                row['straight_v_fly'] = round(float(np.nanmean(F['v'][sm])), 2)
            lm = x >= (x_lh if x_lh is not None else dist)
            if lm.any():
                row['lh_f_out_launch'] = round(float(np.nanmean(prof['f_out'][lm])), 1)
                row['lh_f_out_fly'] = round(float(np.nanmean(F['f_out'][lm])), 1)
                row['lh_v_launch'] = round(float(np.nanmean(prof['v'][lm])), 2)
                row['lh_v_fly'] = round(float(np.nanmean(F['v'][lm])), 2)
            row['grid'] = {'x': x.round(2).tolist(), 't_launch': prof['t'].round(3).tolist(), 't_fly': tF.round(3).tolist(),
                           'v_launch': prof['v'].round(3).tolist(), 'v_fly': F['v'].round(3).tolist(),
                           'f_out_launch': np.round(prof['f_out'], 1).tolist(), 'f_out_fly': np.round(F['f_out'], 1).tolist(),
                           'f_model_launch': np.round(prof['f_model'], 1).tolist(),
                           'f_raw_launch': np.round(prof['f_raw'], 1).tolist(), 'f_raw_fly': np.round(F['f_raw'], 1).tolist(),
                           's_out_launch': np.round(prof['s_out'], 1).tolist(), 's_out_fly': np.round(F['s_out'], 1).tolist(),
                           'ay_launch': np.round(prof['ay'], 2).tolist(), 'ay_fly': np.round(F['ay'], 2).tolist(),
                           'loc_fly': loc_mode.tolist(), 'assist_launch': prof['assist'].astype(int).tolist()}
        # 時間の格子(発進から 0〜3 秒、1/30 秒): 速さ・加速(車輪・IMU)・指令
        tg = t0 + np.arange(0, min(3.0, a1 - t0), 1 / 30.0)
        fvT = frame_vals(d, tg)
        vw = np.interp(tg, vt, d['v'])
        vt_s = smooth(d['v'], 9)
        aw = np.gradient(vt_s, vt)
        row['tgrid'] = {'t': (tg - t0).round(3).tolist(), 'v': vw.round(3).tolist(),
                        'a_wheel': np.interp(tg, vt, aw).round(2).tolist(),
                        'a_imu': np.interp(tg, it, smooth(d['ay'], 9)).round(2).tolist(),
                        'f_out': np.round(fvT['f_out'], 1).tolist(), 'f_model': np.round(fvT['f_model'], 1).tolist(),
                        'f_raw': np.round(fvT['f_raw'], 1).tolist(), 's_out': np.round(fvT['s_out'], 1).tolist(),
                        'assist': fvT['assist'].astype(int).tolist(),
                        'gz': np.interp(tg, it, smooth(d['gz'], 9)).round(1).tolist()}
        # 空転の目安: 動き出し(0.2 m/s)から 0.6 秒、または LH の手前まで
        if t_m1 is not None:
            tb = min(t_m1 + 0.6, t0 + (row.get('t_launch_to_lh') or 99), a1)
            row['slip'] = slip_window(d, t_m1, tb)
        rows.append(row)
    return rows


def flying_slip(d, laps):
    """参考: 周の中の加速(ヘアピンの出口など)で、車輪の速さの増え と IMU の加速の積分を比べる"""
    out = []
    vt = d['vt']
    vs = smooth(d['v'], 9)
    aw = np.gradient(vs, vt)
    for a, b in laps:
        m = (vt >= a) & (vt <= b)
        idx = np.nonzero(m)[0]
        if not len(idx):
            continue
        acc = aw[idx] > 1.5
        for i0, i1 in le.runs(acc):
            ta, tb = float(vt[idx[i0]]), float(vt[idx[i1]])
            if tb - ta < 0.25:
                continue
            s = slip_window(d, ta, tb)
            if s and s['ratio'] is not None:
                fv = frame_vals(d, np.array([ta, (ta + tb) / 2]))
                s.update({'v_a': round(float(np.interp(ta, vt, d['v'])), 2), 'v_b': round(float(np.interp(tb, vt, d['v'])), 2),
                          'f_out_mid': round(float(fv['f_out'][1]), 1), 's_out_mid': round(float(fv['s_out'][1]), 1)})
                out.append(s)
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def summarize(rows, first_only=False, ignore_gain=False, unreadable=None):
    """ほかの日のまとめ: 回と設定(モデル・ゲイン・天井・左・自動・目標)ごとに、発進の前進の OFF / ON(出た)/ ON(出ず)を
    比べる(protocol_v3 14 章の比べは同じ回の中の OFF・ON の交互。回は説明欄の block から。分からなければ「?」でまとまる)。
    first_only = その記録の 1 回目の発進だけ(2026-10-26 fix2 R2-rec-3: 人が止めて戻したあとの発進は置いた場所がばらばら)。
    どちらでも、2 回目以降の発進の数を restarts に書く。ignore_gain = 組の鍵からゲインを外す(2026-10-27 fix3 r2-rec2-1。
    --ignore-gain。回 3 で電圧の表どおりにゲインを入れ直した本を 1 組にする)。
    unreadable = 読めなかった記録(unreadable_info。2026-10-27 fix3 r2-rec2-5)。鍵が同じ組に、無ければゲインのほかが同じ組
    (全部)に、それも無ければ自分の組に入れる。読めない ON があれば、その組は満たさない"""
    def base(r):
        S = r.get('settings') or {}
        return (r['session'], r['model'], None if ignore_gain else gain_key(S, r['gain']), int(_on(S.get('speed_ceiling'))),
                S.get('ceil_left_mps') if _on(S.get('speed_ceiling')) else None,
                int(_on(S.get('speed_auto'))), S.get('auto_target_mps') if _on(S.get('speed_auto')) else None)
    groups, nreach, mids, trials, aborts = {}, {}, {}, {}, {}
    for r in rows:
        if first_only and (r.get('start_no') or 1) != 1:
            continue
        if r.get('midrun'):            # 2026-10-26 fix3 直す 1 回目 r1-rec3-3: 走行中に変えた本は比べに入れない(名前だけ出す)
            mids.setdefault(base(r), []).append(r)
            continue
        if r.get('trial'):             # 2026-10-27 r3-r3-rec-3: 人が memo に「試し」と書いた本も比べに入れない(名前だけ出す)
            trials.setdefault(base(r), []).append(r)
            continue
        if r.get('after_abort') and r.get('to_r1_s') is not None:
            # 2026-10-27 fix7: R1 に届いたリカバリー中止のあとの走り直しも比べに入れない(名前だけ出す)。届かなかった走り直しは前と
            # 同じに下の「R1 に届かなかった発進」(ON なら満たさない。比べの中央値にはもとから入らない)
            aborts.setdefault(base(r), []).append(r)
            continue
        if r.get('to_r1_s') is None:
            if 'launch_state' in r:    # r1-rec3-1: ほかの日の R1 に届かなかった発進(前は黙って外した)
                nreach.setdefault(base(r), {}).setdefault(r['launch_state'], []).append(r)
            continue
        groups.setdefault(base(r), {}).setdefault(r.get('launch_state', 'off'), []).append(r)

    def agg_nr(rs):
        """R1 に届かなかった発進(r1-rec3-1): 本数・名前・理由・AI の区間・その衝突(発進 → AI の区間の終わり + NO_R1_TAIL_S。
        2026-10-27 r2-rec2-3 から 0.5 秒。前は 0.2 秒)"""
        out_ = {'n': len(rs), 'logs': ['%s@%.2f' % (r['log'], r['t_ai']) for r in rs], 'no': [r.get('no') for r in rs],
                'why': [r.get('why_no_r1') for r in rs], 'ai_seg_s': [r.get('ai_seg_s') for r in rs],
                'collisions': sum(len(r.get('collisions_first_lap') or []) for r in rs),
                'hits': [[r['log']] + h for r in rs for h in r.get('collisions_first_lap') or []]}
        ab_ = ['%s@%.2f' % (r['log'], r['t_ai']) for r in rs if r.get('after_abort')]
        if ab_:                        # 2026-10-27 fix7: うちリカバリー中止のあとの走り直し(あるときだけ)
            out_['after_abort'] = ab_
        return out_

    def opts_of(r):
        S = r.get('settings') or {}
        return tuple(k for k in OPT_KEYS if _on(S.get(k)))

    def agg(st_, rs):
        slips = [(r.get('slip') or {}).get('ratio') for r in rs]
        out_ = {'n': len(rs), 'logs': ['%s@%.2f' % (r['log'], r['t_ai']) for r in rs], 'no': [r['no'] for r in rs],
                'to_r1_median_s': _med([r['to_r1_s'] for r in rs]), 'loss_median_s': _med([r.get('loss_s') for r in rs]),
                'move_0p2_median_s': _med([r.get('move_0p2_s') for r in rs]),
                'slip_max': max([x for x in slips if x is not None], default=None),
                'v_lh_median_mps': _med([r.get('v_launch_at_lh') for r in rs]),
                'collisions_to_r1': sum(len(r.get('collisions') or []) for r in rs),
                'collisions_first_lap': sum(len(r.get('collisions_first_lap') or []) for r in rs),
                'first_lap_hits': [[r['log']] + h for r in rs for h in r.get('collisions_first_lap') or []],
                'launch_on_s_median': _med([r.get('launch_on_s') for r in rs]),
                'restarts': sum(1 for r in rs if (r.get('start_no') or 1) >= 2),
                'launch_end_reasons': sorted({r.get('launch_end_reason') or '' for r in rs} - {''}),
                # 2026-10-26 critic(tools:F4): 途中で launch_us を 200 に下げた ON(手順書 7 章)が 250 と同じ組に入るので、値を出す
                'launch_us': sorted({float((r.get('settings') or {}).get('launch_us')) for r in rs
                                     if st_ != 'off' and (r.get('settings') or {}).get('launch_us') is not None}),
                'frame_gain': [min(r['gain'] for r in rs), max(r['gain'] for r in rs)]}
        if st_ == 'on':                # 2026-10-27 r3-r3-rec-4: 比が出ない ON の発進(あるときだけ。無ければ前と同じ)
            ext = [r for r in rs if slip_state(r.get('slip')) == 'extreme']
            unk = [r for r in rs if slip_state(r.get('slip')) == 'unknown']
            if ext:
                out_['slip_extreme'] = ['%s@%.2f(車輪 +%.2f・IMU +%.2f m/s)' % (r['log'], r['t_ai'], r['slip']['dv_wheel'],
                                                                             r['slip']['dv_imu']) for r in ext]
            if unk:
                out_['slip_unmeasured'] = ['%s@%.2f(%s)' % (r['log'], r['t_ai'], (
                    '車輪 +%.2f・IMU +%.2f m/s。どちらもほとんど増えない' % (r['slip']['dv_wheel'], r['slip']['dv_imu']) if r.get('slip')
                    else '窓が短い(LH・R1 が近い)か IMU が抜けた')) for r in unk]
        return out_
    # r2-rec2-5: 読めなかった記録を組に入れる(鍵が同じ組 → ゲインのほかが同じ組の全部 → 自分の組)
    ur = {}
    keys_all = set(groups) | set(nreach) | set(mids) | set(trials) | set(aborts)
    unk_all = [u for u in unreadable or [] if u.get('meta') is False]    # r3-r3-rec-1: log.json も読めない(回・ON/OFF 不明)
    for u in unreadable or []:
        if u.get('meta') is False:
            continue
        ku = base({'session': u['session'], 'model': u['model'], 'gain': u['gain'], 'settings': u['settings']})
        tg = [ku] if ku in keys_all else (sorted((k for k in keys_all if k[:2] + k[3:] == ku[:2] + ku[3:]), key=str) or [ku])
        for k in tg:
            ur.setdefault(k, []).append(u)
    # r3-r3-rec-1: log.json の読めない記録は、その日の発進の前進 ON のある組(出た・出ず・届かなかった・途中で変えた・読めない ON)の
    # 全部に入れる(どの回のどの本か分からない。入れた組は満たさない)。ON のある組が無ければどの組にも入れない(比べの行は出ない)
    unk = {}
    if unk_all:
        def has_on(k):
            rs = ([r for rs_ in groups.get(k, {}).values() for r in rs_] + [r for rs_ in nreach.get(k, {}).values() for r in rs_]
                  + list(mids.get(k, [])) + list(trials.get(k, [])) + list(aborts.get(k, [])))
            return (any(r.get('launch_state') in ('on', 'on_not_started') for r in rs)
                    or any(u['launch_state'] == 'on' for u in ur.get(k, [])))
        for u in unk_all:
            md = '%s/%s ' % (u['log'][4:6], u['log'][6:8])
            for k in sorted(keys_all | set(ur), key=str):
                if str(k[0]).startswith(md) and has_on(k):
                    unk.setdefault(k, []).append(u)

    def lh_self_v(r):
        """発進のフレームの場所で決めた LH の入口の速さ(r2-rec2-2。無ければ None)"""
        return (r.get('_lh_self') or r.get('lh_self') or {}).get('v_mps')
    out, sib, forms = [], {}, {}
    for k in sorted(set(groups) | set(nreach) | set(mids) | set(trials) | set(aborts) | set(ur), key=str):     # 届かなかった・途中で変えた・試し・中止のあと・読めない本だけの組も出す
        g = {'session': k[0], 'model': k[1], 'gain': k[2], 'speed_ceiling': k[3], 'ceil_left_mps': k[4], 'speed_auto': k[5],
             'auto_target_mps': k[6], 'by_launch': {}}
        for st_, rs in sorted(groups.get(k, {}).items()):
            g['by_launch'][st_] = agg(st_, rs)
        # 次の欄は、あるときだけ(無い = なし。前の出力と同じ形のまま)
        if k in nreach:
            g['not_reached'] = {st_: agg_nr(rs) for st_, rs in sorted(nreach[k].items())}
        if k in mids:
            g['midrun_excluded'] = ['%s@%.2f(%s)' % (r['log'], r['t_ai'], midrun_text(r['midrun'])) for r in mids[k]]
            # 途中で変えた ON(出た)の発進の最初の周の衝突は、比べの「最初の周の衝突 0」には数える(衝突は止める条件。強さ・設定に
            # よらず全部の ON。tools:F4 の残りと同じ考え)
            mc = sum(len(r.get('collisions_first_lap') or []) for r in mids[k] if r.get('launch_state') == 'on')
            if mc:
                g['midrun_on_collisions'] = mc
        if k in trials:                # r3-r3-rec-3: 人が memo に「試し」と書いた本(名前と memo。あるときだけ)
            g['trial_excluded'] = ['%s@%.2f(%s・%s)' % (r['log'], r['t_ai'], r.get('launch_state', 'off'), r['trial'][:40])
                                   for r in trials[k]]
            # 試しの ON(出た)の発進の最初の周の衝突も「最初の周の衝突 0」に数える(衝突は止める条件。途中で変えた本と同じ考え)
            tc = sum(len(r.get('collisions_first_lap') or []) for r in trials[k] if r.get('launch_state') == 'on')
            if tc:
                g['trial_on_collisions'] = tc
        if k in aborts:                # 2026-10-27 fix7: R1 に届いたリカバリー中止のあとの走り直し(名前・中止・人側の秒・起点まで・衝突)
            # fix7 r2: 止まった形(中止・中立止め・カメラの途切れ)と、ON の走り直しの空転の比・最初の周の記録の欠け(比べの条件に数える)
            g['abort_resume_excluded'] = ['%s@%.2f(%s・%s・人側 %s 秒・起点まで %.3f 秒%s%s%s)' % (
                r['log'], r['t_ai'], r.get('launch_state', 'off'), abort_text(r['after_abort']), r['after_abort'].get('human_s'),
                r['to_r1_s'], '・' + slip_text(r.get('slip')) if r.get('launch_state') == 'on' else '',
                '・最初の周の記録が欠けた(%s)' % r['first_lap_gap'] if r.get('launch_state') == 'on' and r.get('first_lap_gap') else '',
                '・最初の周の衝突 %s' % '・'.join('%.2f 秒 %.1fG %s' % tuple(h) for h in r['collisions_first_lap'])
                if r.get('collisions_first_lap') else '') for r in aborts[k]]
            # 走り直しの ON(出た)の発進の最初の周の衝突も「最初の周の衝突 0」に数える(衝突は止める条件。途中で変えた本・試しと同じ考え)
            ac = sum(len(r.get('collisions_first_lap') or []) for r in aborts[k] if r.get('launch_state') == 'on')
            if ac:
                g['abort_resume_on_collisions'] = ac
        if k in ur:                    # r2-rec2-5: 読めなかった記録(名前・理由・発進の前進の ON/OFF)
            us_ = ur[k]
            g['unreadable'] = {'n': len(us_), 'logs': [u['log'] for u in us_], 'no': [u.get('no') for u in us_],
                               'why': [u['why'] for u in us_], 'launch_state': [u['launch_state'] for u in us_],
                               'launch_param': [u.get('launch_param') for u in us_], 'plan_launch': [u.get('plan_launch') for u in us_],
                               'on': sum(1 for u in us_ if u['launch_state'] == 'on')}
            if any(u.get('launch_starts') for u in us_):     # 2026-10-27 fix6: events.jsonl の発進の前進の start(あるときだけ)
                g['unreadable']['launch_starts'] = [u.get('launch_starts') for u in us_]
            if any(u.get('launch_on_from') for u in us_):    # 2026-10-27 fix6-2: ON の手がかり(あるときだけ)
                g['unreadable']['launch_on_from'] = [u.get('launch_on_from') for u in us_]
        gap_on = [r for r in groups.get(k, {}).get('on', []) + nreach.get(k, {}).get('on', []) if r.get('first_lap_gap')]
        # 2026-10-27 fix7 r2(w2-tools-r7-2-tools-2): R1 に届いた走り直しの ON(出た)の発進の最初の周の記録の欠けも数える(fix7 r1 は
        # 比べから外したときに一緒に落としていた。欠けた最初の周の衝突は記録で確かめられない = 衝突と同じく止める条件の側)
        gap_on += [r for r in aborts.get(k, []) if r.get('launch_state') == 'on' and r.get('first_lap_gap')]
        if gap_on:                     # r2-rec2-5: 最初の周の間で記録が欠けた ON の発進
            g['first_lap_gap_on'] = ['%s@%.2f(%s%s)' % (r['log'], r['t_ai'], r['first_lap_gap'],
                                                        '。' + RESUME_NAME if r.get('after_abort') else '') for r in gap_on]
        # r2-rec2-3: 最初の周を AI の区間の終わり + 0.5 秒まで数えた ON(名前だけ。比べは変えない)。線の前に人側にした本と、A2 が
        # 無い(記録が A2 の前で終わった・当たって止まった・人側で R1 を通らなかった)ので線が分からない本
        ae = [r for r in groups.get(k, {}).get('on', []) if r.get('first_lap_end') == 'ai_end']
        if ae:
            g['on_before_line'] = ['%s@%.2f' % (r['log'], r['t_ai']) for r in ae if r.get('first_lap_a2_s') is not None]
            g['on_no_a2'] = ['%s@%.2f' % (r['log'], r['t_ai']) for r in ae if r.get('first_lap_a2_s') is None]
        if k in unk:                   # r3-r3-rec-1: log.json の読めない記録(回・ON/OFF 不明。あるときだけ)
            g['unreadable_unknown'] = {'n': len(unk[k]), 'logs': [u['log'] for u in unk[k]], 'why': [u['why'] for u in unk[k]],
                                       'near': [u['near'] for u in unk[k]]}
        bad_read = bool((g.get('unreadable') or {}).get('on') or gap_on or k in unk)
        allr = [r for rs in list(groups.get(k, {}).values()) + list(nreach.get(k, {}).values()) for r in rs]
        if len({opts_of(r) for r in allr}) > 1:          # r1-rec3-5: 試験の選択肢の ON/OFF が混ざった(再起動で戻ったなど)
            g['mixed_options'] = {OPT_NAMES[o]: {'on': sum(1 for r in allr if o in opts_of(r)),
                                                 'off': sum(1 for r in allr if o not in opts_of(r))}
                                  for o in OPT_KEYS if len({o in opts_of(r) for r in allr}) > 1}
        on, off = g['by_launch'].get('on'), g['by_launch'].get('off')
        on_nr = (g.get('not_reached') or {}).get('on')        # R1 に届かなかった ON(出た)の発進
        on_use = on                    # 比べに使う ON(混ざっていなければ全部の ON。今までと同じ)
        rows_on_use = groups.get(k, {}).get('on', [])
        if on and len(on['launch_us']) > 1:
            g['mixed_launch_us'] = on['launch_us']
            # 2026-10-26 fix3(tools:F4 の残り): 最後の ON の強さで比べる(protocol_v3 14 章「残りの ON を 200 µs に」。回 4 はこの強さ)
            us_of = lambda r: (r.get('settings') or {}).get('launch_us')      # noqa: E731
            rows_on = [r for r in groups[k]['on'] if us_of(r) is not None]
            last = max(rows_on, key=lambda r: (r['log'], r['t_ai']))
            use = [r for r in rows_on if float(us_of(r)) == float(us_of(last))]
            on_use = agg('on', use)
            rows_on_use = use
            g['launch_us_used'] = float(us_of(last))
            g['on_used'] = on_use
        if k in aborts:
            # 2026-10-27 fix7 r2(w2-tools-r7-2-tools-2): R1 に届いた走り直しの ON(出た)の発進の空転も、比べに使う ON と同じに見る
            # (起点まで・失う・動く・LH の中央値からは外したまま)。空転は置いた場所によらず車輪で起きた事実(car_check の「発進の空転の比」
            # と同じ)。launch_us が混ざった組は、比べに使う強さ(最後の ON の強さ)の走り直しだけ(r3-r3-rec-4・tools:F4 の残りと同じ)
            ab_on = [r for r in aborts[k] if r.get('launch_state') == 'on']
            if g.get('launch_us_used') is not None:
                ab_on = [r for r in ab_on if (r.get('settings') or {}).get('launch_us') is not None
                         and float(r['settings']['launch_us']) == float(g['launch_us_used'])]
            ab_hi = [r for r in ab_on if slip_state(r.get('slip')) in ('high', 'extreme')]
            ab_un = [r for r in ab_on if slip_state(r.get('slip')) == 'unknown']
            if ab_hi:                  # 比 > 1.3・いちばんひどい空転 → checks の slip = False
                g['abort_resume_slip'] = ['%s@%.2f(%s)' % (r['log'], r['t_ai'], slip_text(r.get('slip'))) for r in ab_hi]
            if ab_un:                  # 比が出ない → checks の slip_all_measured = False
                g['abort_resume_slip_unmeasured'] = ['%s@%.2f(%s)' % (r['log'], r['t_ai'], slip_text(r.get('slip')))
                                                     for r in ab_un]
        if on and off:
            d_to = None if on_use['to_r1_median_s'] is None or off['to_r1_median_s'] is None else round(on_use['to_r1_median_s'] - off['to_r1_median_s'], 3)
            d_loss = None if on_use['loss_median_s'] is None or off['loss_median_s'] is None else round(on_use['loss_median_s'] - off['loss_median_s'], 3)
            d_v = None if on_use['v_lh_median_mps'] is None or off['v_lh_median_mps'] is None else round(on_use['v_lh_median_mps'] - off['v_lh_median_mps'], 3)
            lh_src = None
            rows_off = groups[k]['off']
            if any(r.get('v_launch_at_lh') is None for r in list(rows_on_use) + list(rows_off)):
                # r2-rec2-2: 周の LH が無い発進がある → ON・OFF とも発進のフレームの場所の LH で比べる(同じ決め方どうし)
                lh_src = 'launch_loc'
                v_on_s, v_off_s = _med([lh_self_v(r) for r in rows_on_use]), _med([lh_self_v(r) for r in rows_off])
                d_v = None if v_on_s is None or v_off_s is None else round(v_on_s - v_off_s, 3)
                for st_, rs in groups[k].items():
                    g['by_launch'][st_]['v_lh_self_median_mps'] = _med([lh_self_v(r) for r in rs])
                if g.get('on_used'):
                    g['on_used']['v_lh_self_median_mps'] = v_on_s
            chk = {'to_r1': d_to is not None and d_to <= -CRIT['to_r1_gain_s'],
                   # 衝突は強さによらず全部の ON。R1 に届かなかった ON の発進の衝突も(r1-rec3-1)
                   'first_lap_collisions_0': (on['collisions_first_lap'] == 0 and not (on_nr and on_nr['collisions'])
                                              and not g.get('midrun_on_collisions') and not g.get('trial_on_collisions')
                                              and not g.get('abort_resume_on_collisions')),
                   # r3-r3-rec-4: いちばんひどい空転(比が出ない。> 1.3 とみなす)の ON があれば満たさない
                   # fix7 r2: 走り直しの ON(比べに使う強さ)の空転も(abort_resume_slip)
                   'slip': (on_use['slip_max'] is not None and on_use['slip_max'] <= CRIT['slip_max']
                            and not on_use.get('slip_extreme') and not g.get('abort_resume_slip')),
                   'lh_speed': d_v is not None and d_v <= CRIT['lh_dv_max']}
            if on_nr:                  # r1-rec3-1: 届かなかった ON があれば満たさない(最初の周を確かめられない。名前を出す)
                chk['on_all_reached_r1'] = False
            if bad_read:               # r2-rec2-5: 読めない(欠けた)ON があれば満たさない(最初の周を確かめられない)
                chk['on_all_readable'] = False
            # r3-r3-rec-4: 空転の比が出ない ON があれば満たさない(確かめられない)。fix7 r2: 走り直しの ON も
            if on_use.get('slip_unmeasured') or g.get('abort_resume_slip_unmeasured'):
                chk['slip_all_measured'] = False
            g['compare'] = {'d_to_r1_s': d_to, 'd_loss_s': d_loss, 'd_v_lh_mps': d_v, 'checks': chk, 'all_ok': all(chk.values())}
            if on_use is not on:
                g['compare'].update(launch_us_used=g['launch_us_used'], n_on_used=on_use['n'])
            if lh_src:                 # r2-rec2-2(あるときだけ)
                g['compare']['lh_src'] = lh_src
        elif (on_nr or bad_read) and off:   # r1-rec3-1・r2-rec2-5: 届いた・読めた ON が無い → 比べられない = 満たさない
            ck = {'to_r1': False, 'first_lap_collisions_0': not ((on_nr or {}).get('collisions') or g.get('midrun_on_collisions')
                                                                 or g.get('trial_on_collisions')
                                                                 or g.get('abort_resume_on_collisions')),
                  'slip': False, 'lh_speed': False}
            if on_nr:
                ck['on_all_reached_r1'] = False
            if bad_read:
                ck['on_all_readable'] = False
            g['compare'] = {'d_to_r1_s': None, 'd_loss_s': None, 'd_v_lh_mps': None, 'checks': ck, 'all_ok': False}
        rs_all = allr + list(mids.get(k, [])) + list(trials.get(k, [])) + list(aborts.get(k, []))
        if ignore_gain:                # r2-rec2-1: まとめたゲインの幅(組の鍵のゲイン。自動合わせ ON はスライダーの値)
            gk = [gain_key(r.get('settings') or {}, r['gain']) for r in rs_all]
            g['gain_ignored'] = True
            g['gain_range'] = [min(gk), max(gk)] if gk else None
        out.append(g)
        # ゲインのほかが同じ組(回・モデル・スイッチ)と、発進の前進 ON の本(出た・出ず。届かなかった・途中で変えた・読めないも)があるか
        sib.setdefault(k[:2] + k[3:], []).append((g, any(r.get('launch_state') in ('on', 'on_not_started') for r in rs_all)
                                                  or bool((g.get('unreadable') or {}).get('on'))))
        # 2026-10-27 fix5 c3: 比べに入る本(届いた・届かなかった・読めない。途中で変えた・試しは外れているので入れない)のある組を、
        # 回・モデルごとに。発進の前進 ON の本(出た・出ず・読めない ON)があるか
        if allr or k in ur:
            forms.setdefault(k[:2], []).append((g, any(r.get('launch_state') in ('on', 'on_not_started') for r in allr)
                                                or bool((g.get('unreadable') or {}).get('on'))))
    if not ignore_gain:                # r2-rec2-1: 同じ回・同じモデル・同じスイッチでゲインだけ違う組(発進の前進 ON の本があるとき)
        for gs in sib.values():
            if len(gs) >= 2 and any(on for _, on in gs):
                gains = sorted({g['gain'] for g, _ in gs if g['gain'] is not None})     # (None = 読めない本だけの組でゲイン不明)
                for g, _ in gs:
                    g['gain_split'] = gains
    # 2026-10-27 fix5(反証 4 回目 c3): 同じ回・同じモデルの本が、天井・左・自動合わせ・目標の違いで 2 つ以上の形に分かれ、どれかの組に
    # 発進の前進 ON の本があれば、その回・モデルのどの組にも注意(form_split)を付け、比べ(compare)は「満たさない」(checks の
    # one_form = False)。回 3 の途中で起こし直して左 4.2 → 4.1 に戻ったのに入れ直し忘れた日など。前は 2 本対 2 本の 2 行に分かれ、
    # 手順書 6 章の「同じ回の ON 4 本対 OFF 4 本」でない行で回 4 の ON が決まりえた(片方が「使ってよい」)。安全側(回 4 は発進の前進
    # OFF)。計画と違う形の本は走り終わりのメモに「試し」(外れる)か --skip で外せば 1 つの形に戻る。ゲインの違い(gain_split)は別
    for gs in forms.values():
        fs = sorted({form_of(g) for g, _ in gs}, key=str)
        if len(fs) >= 2 and any(on for _, on in gs):
            for g, _ in gs:
                g['form_split'] = [form_text(f) for f in fs]
                if 'compare' in g:
                    g['compare']['checks']['one_form'] = False
                    g['compare']['all_ok'] = False
    return out


def form_of(g):
    """組の走りの形(天井・左・自動合わせ・目標。c3)"""
    return (g['speed_ceiling'], g['ceil_left_mps'], g['speed_auto'], g['auto_target_mps'])


def form_text(f):
    """形の文(組の行と同じ書き方: 天井1(4.2) 自動0)"""
    return '天井%s%s 自動%s%s' % (f[0], '(%s)' % f[1] if f[0] else '', f[2], '(%s)' % f[3] if f[2] else '')


def summary_lines(summary):
    lines = ['', '発進の前進の比べ(設定ごと。ON = 出た・ON_not_started = launch_boost 1 なのに出ない → ON として数えない)']
    f = lambda x: '—' if x is None else ('%.3f' % x if isinstance(x, float) else str(x))      # noqa: E731
    def line(st_, b):
        return '  %-15s n=%d(2 回目以降の発進 %d) 起点まで %s 失う %s 動く %s 空転max %s LH %s m/s 衝突(起点まで %d・最初の周 %d) 出した %s %s' % (
            st_, b['n'], b.get('restarts', 0), f(b['to_r1_median_s']), f(b['loss_median_s']), f(b['move_0p2_median_s']), f(b['slip_max']),
            f(b['v_lh_median_mps']) + ('(発進の場所で %s)' % f(b['v_lh_self_median_mps']) if 'v_lh_self_median_mps' in b else ''),
            b['collisions_to_r1'], b['collisions_first_lap'], f(b['launch_on_s_median']),
            '/'.join(b['launch_end_reasons']))
    for g in summary:
        gtxt = ('g%.2f' % g['gain'] if g['gain'] is not None else
                'g%s(ゲインをまとめた --ignore-gain)' % ('〜'.join('%.2f' % x for x in sorted(set(g.get('gain_range') or []))) or '?')
                if g.get('gain_ignored') else 'g?(読めない本だけ。ゲイン不明)')
        lines.append('%s %s %s 天井%s%s 自動%s%s' % (g['session'], g['model'], gtxt, g['speed_ceiling'],
                                                '(%s)' % g['ceil_left_mps'] if g['speed_ceiling'] else '',
                                                g['speed_auto'], '(%s)' % g['auto_target_mps'] if g['speed_auto'] else ''))
        for st_, b in g['by_launch'].items():
            lines.append(line(st_, b))
        if g.get('on_used'):
            lines.append(line('on(%g µs だけ)' % g['launch_us_used'], g['on_used']))
        if 'compare' in g:
            c = g['compare']
            lines.append('  ON − OFF: 起点まで %s 秒(≤ −%.2f)・失う %s 秒・LH %s m/s(≤ +%.1f)・空転 ≤ %.1f・最初の周の衝突 0 → %s %s' % (
                f(c['d_to_r1_s']), CRIT['to_r1_gain_s'], f(c['d_loss_s']),
                f(c['d_v_lh_mps']) + ('(発進のフレームの場所で)' if c.get('lh_src') == 'launch_loc' else ''), CRIT['lh_dv_max'],
                CRIT['slip_max'], '使ってよい' if c['all_ok'] else '満たさない', c['checks']))
            if c.get('lh_src') == 'launch_loc':      # r2-rec2-2(あるときだけ)
                lines.append('  LH の入口の速さは、周の無い発進があるので ON・OFF とも発進のフレームの場所(LH)で比べた%s'
                             % ('' if c['d_v_lh_mps'] is not None else
                                '。それでも出せない(発進のフレームに LH が無い)= 満たさない。回 3 のどれか 1 本を 2 周(A2 まで AI 側)'
                                'にすると周と比べられる'))
        if g.get('mixed_launch_us'):
            lines.append('  注意: ON に launch_us の違う本が混ざっている(%s µs)。比べは最後の ON の強さ %g µs の ON %d 本と OFF'
                         '(回 4 はこの強さのまま走る。protocol_v3 14 章「残りの ON を 200 µs に」)。最初の周の衝突は全部の ON(%d 本)で数える'
                         % ('・'.join('%g' % x for x in g['mixed_launch_us']), g['launch_us_used'], g['on_used']['n'],
                            g['by_launch']['on']['n']))
        # 2026-10-26 fix3 直す 1 回目 r1-rec3-1・3・5(あるときだけ)
        for st_, b in (g.get('not_reached') or {}).items():
            hits = '・'.join('%s %.2f 秒 %.1fG %s' % (h[0][-6:], h[1], h[2], h[3]) for h in b['hits']) or 'なし'
            lines.append('  R1 に届かなかった発進(%s)%d 本(比べの数に入れない): %s。理由 %s。その衝突 %d(%s)%s' % (
                st_, b['n'], '・'.join(b['logs']), '/'.join(sorted({w or '?' for w in b['why']})), b['collisions'], hits,
                # 2026-10-27 fix7(あるときだけ)。r2: 名前を変えた(中立止め・カメラの途切れのあとも)
                '。うち%s %d(%s)' % (RESUME_NAME, len(b['after_abort']), '・'.join(b['after_abort']))     # fix7 r1 と同じ文
                if b.get('after_abort') else ''))
        if (g.get('not_reached') or {}).get('on'):
            lines.append('  → ON の発進が R1 の前で終わった本がある = 満たさない(最初の周を確かめられない)。当たった・人が危なくて'
                         '止めた → 回 4 は OFF(手順書 7 章)。車のせいでない理由(人・物がコースに入った・試し)で止めた本だけ、'
                         '--skip 記録名 で外して読み直せる(外した名前が出る)')
        if g.get('unreadable'):        # r2-rec2-5(あるときだけ)
            u = g['unreadable']
            yn = lambda x: '?' if x is None else ('ON' if _on(x) else 'OFF')      # noqa: E731
            # fix6-2: ON の手がかりのうち log.json・計画のほか(発進の前進の出来事・走行中に ON)を後ろに(あるときだけ)
            other = lambda src: [x for x in src or [] if x not in ('log.json', '計画')]      # noqa: E731
            lines.append('  読めない記録 %d 本(比べの数に入れない): %s' % (u['n'], '・'.join(
                '%s(%s。発進の前進 %s: log.json %s・計画 %s%s%s)' % (lg, w, 'ON' if st_ == 'on' else 'OFF', yn(lp), yn(pl),
                                                                '・events.jsonl の start %s' % starts_text(ss) if ss else '',
                                                                '・%s' % '・'.join(other(src)) if other(src) else '')
                for lg, w, st_, lp, pl, ss, src in zip(u['logs'], u['why'], u['launch_state'], u['launch_param'], u['plan_launch'],
                                                       u.get('launch_starts') or [None] * u['n'],
                                                       u.get('launch_on_from') or [None] * u['n']))))
        if g.get('first_lap_gap_on'):
            lines.append('  最初の周の記録が欠けた発進(on)%d 本: %s' % (len(g['first_lap_gap_on']), '・'.join(g['first_lap_gap_on'])))
        if (g.get('unreadable') or {}).get('on') or g.get('first_lap_gap_on'):
            lines.append('  → 読めない(欠けた)ON の本がある = 満たさない(電源断・jetformer-web の落ちなどで npz が書かれなかった。'
                         '最初の周に当たったかを確かめられない)。動画・紙で確かめ、当たっていない・車のせいでないと分かった本だけ、'
                         'main session が --skip 記録名 で外して読み直せる(外した名前が出る)')
        if g.get('unreadable_unknown'):     # r3-r3-rec-1(あるときだけ)
            u = g['unreadable_unknown']
            lines.append('  読めない記録(log.json も読めない。回・発進の前進の ON/OFF が分からない)%d 本: %s' % (u['n'], '・'.join(
                '%s(%s。前の記録 %s・次の記録 %s)' % (lg, w, nr[0] or 'なし', nr[1] or 'なし')
                for lg, w, nr in zip(u['logs'], u['why'], u['near']))))
            lines.append('  → この日の発進の前進 ON のある組は全部「満たさない」(電源断などで log.json・npz が書かれなかった本は、どの回の'
                         'どの本か・ON だったか・最初の周に当たったかを記録で確かめられない)。紙・動画でどの本か確かめ、ON の本でない・'
                         '当たっていない・車のせいでないと分かったら、main session が --skip 記録名 で外して読み直せる(外した名前が出る)')
        if g.get('on_before_line') or g.get('on_no_a2'):
            lines.append('  最初の周を AI の区間の終わり + %.1f 秒まで数えた ON %d 本(人側にした直後の衝突も数える): 線の前に人側 %s / '
                         'A2 が無く線が分からない %s' % (FLIP_TAIL_S, len(g['on_before_line']) + len(g['on_no_a2']),
                                                  '・'.join(g['on_before_line']) or 'なし', '・'.join(g['on_no_a2']) or 'なし'))
        if g.get('midrun_excluded'):
            lines.append('  途中で変えた %d 本(比べに入れない。ON の組にも OFF の組にも入れない。手順書 7 章): %s%s' % (
                len(g['midrun_excluded']), '・'.join(g['midrun_excluded']),
                '。うち ON の発進の最初の周の衝突 %d(比べの「最初の周の衝突 0」には数える)' % g['midrun_on_collisions']
                if g.get('midrun_on_collisions') else ''))
        if g.get('trial_excluded'):    # r3-r3-rec-3(あるときだけ)
            lines.append('  試しの %d 本(説明欄の memo に「試し」。比べに入れない。ON の組にも OFF の組にも入れない。手順書 7 章): %s%s' % (
                len(g['trial_excluded']), '・'.join(g['trial_excluded']),
                '。うち ON の発進の最初の周の衝突 %d(比べの「最初の周の衝突 0」には数える)' % g['trial_on_collisions']
                if g.get('trial_on_collisions') else ''))
        if g.get('abort_resume_excluded'):     # 2026-10-27 fix7(あるときだけ)。r2: 中立止め・カメラの途切れのあとも・空転と欠けは数える
            lines.append('  %s %d 本(同じ本の一つ前の AI の区間がリカバリー中止で終わり、CH3 を人側 → AI 側にした発進。中立で停止中'
                         '(中立止め 0.5 秒以上・カメラの途切れ)のまま人側にして止まって終わった区間のあとも同じ。止まった所から出たかもしれない'
                         '(置き直しても車輪の距離は 0 なので記録から分からない)ので、比べ(起点まで・失う・動く・LH)に入れない。ON の組にも'
                         ' OFF の組にも入れない。ON の発進の最初の周の衝突・空転・最初の周の記録の欠けは比べの条件に数える。手順書 7 章): %s%s' % (
                             RESUME_NAME, len(g['abort_resume_excluded']), '・'.join(g['abort_resume_excluded']),
                             '。うち ON の発進の最初の周の衝突 %d(比べの「最初の周の衝突 0」には数える)' % g['abort_resume_on_collisions']
                             if g.get('abort_resume_on_collisions') else ''))
        if g.get('abort_resume_slip'):         # 2026-10-27 fix7 r2(あるときだけ)
            lines.append('  → %sの ON の発進が空転した %d 本(比 > %.1f か、比が出ないのに車輪が増えた = いちばんひどい空転): %s → 空転の'
                         '条件は満たさない(checks の slip。空転は置いた場所によらず車輪で起きた事実。car_check の「発進の空転の比」と同じ)'
                         % (RESUME_NAME, len(g['abort_resume_slip']), CRIT['slip_max'], '・'.join(g['abort_resume_slip'])))
        if g.get('abort_resume_slip_unmeasured'):     # 2026-10-27 fix7 r2(あるときだけ)
            lines.append('  → %sの ON の発進で空転の比が出ない %d 本: %s → 空転を確かめられない = 満たさない(checks の slip_all_measured)。'
                         '動画で空転を見る' % (RESUME_NAME, len(g['abort_resume_slip_unmeasured']),
                                         '・'.join(g['abort_resume_slip_unmeasured'])))
        sl_src = g.get('on_used') or g['by_launch'].get('on') or {}      # r3-r3-rec-4: 比べに使う ON(あるときだけ)
        if sl_src.get('slip_extreme'):
            lines.append('  空転がいちばんひどい ON の発進 %d 本(車輪が %.1f m/s より増えたのに IMU の積分が %.1f m/s 以下で比が出ない = '
                         '比 > %.1f とみなす → 空転の条件は満たさない): %s' % (
                             len(sl_src['slip_extreme']), SLIP_EXTREME_DV_WHEEL, SLIP_IMU_MIN, CRIT['slip_max'],
                             '・'.join(sl_src['slip_extreme'])))
        if sl_src.get('slip_unmeasured'):
            lines.append('  空転の比が出ない ON の発進 %d 本: %s → 空転を確かめられない = 満たさない(checks の slip_all_measured)。'
                         '動画で空転を見る。決めは main session(--skip 記録名 で外すと、起点までの時間の比べからも外れる)' % (
                             len(sl_src['slip_unmeasured']), '・'.join(sl_src['slip_unmeasured'])))
        if g.get('mixed_options'):
            lines.append('  注意: 試験の選択肢の ON/OFF が混ざっている(%s)。再起動で OFF に戻った本が無いか ② runs_1103 で見る'
                         % '・'.join('%s ON %d・OFF %d' % (k, v['on'], v['off']) for k, v in g['mixed_options'].items()))
        if g.get('form_split'):        # 2026-10-27 fix5 c3(あるときだけ)
            lines.append('  注意: 同じ回・同じモデルの本が、天井・左・自動合わせ・目標の違いで %d 組に分かれた(%s)。手順書 6 章の'
                         '「同じ回の ON 4 本対 OFF 4 本」(本番の形 1 つ)でないので、この回のどの組も「満たさない」(checks の one_form)'
                         '→ 回 4 は発進の前進 OFF。起こし直しで左・天井・自動合わせが既定に戻った本が無いか ② runs_1103 で見る。'
                         '計画と違う形の本は走り終わりのメモに「試し」(4 章 9)か、main session が --skip 記録名 で外して読み直す'
                         % (len(g['form_split']), '・'.join(g['form_split'])))
        if g.get('gain_split'):        # r2-rec2-1(あるときだけ)
            lines.append('  注意: ゲインが本ごとに違うので組が分かれた(%s)。手順書 6 章の「同じ回の ON 4 本対 OFF 4 本」は '
                         '--ignore-gain で 1 組にして読む(電圧の表どおりの入れ直しなら同じ形)' % '・'.join('%.2f' % x for x in g['gain_split']))
        if g['speed_auto'] and g['by_launch']:
            lo, hi = min(b['frame_gain'][0] for b in g['by_launch'].values()), max(b['frame_gain'][1] for b in g['by_launch'].values())
            lines.append('  (自動合わせ ON: 組はスライダーのゲイン %s。走ったフレームのゲインは %.3f〜%.3f)' % (
                '%.2f' % g['gain'] if g['gain'] is not None else '〜'.join('%.2f' % x for x in sorted(set(g.get('gain_range') or []))),
                lo, hi))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description='止まってから AI で走り出した場面を集める')
    ap.add_argument('--day', action='append', default=None, help='YYYYMMDD(何回でも・コンマ区切りでも。既定 20261023・20261024)')
    ap.add_argument('--out', default=None, help='出力の JSON(既定 out/launches.json。--day のときは out/launches_<日>.json)')
    ap.add_argument('--first-only', action='store_true',
                    help='まとめ(summary)を、その記録の 1 回目の発進だけで作る(人が止めて戻したあとの発進を入れない)')
    ap.add_argument('--ignore-gain', action='store_true',
                    help='まとめ(summary)の組の鍵からゲインを外す(同じ回・同じモデル・同じスイッチの本を 1 組に。回 3 で電圧の表どおりに '
                         'ゲインを入れ直した本の ON/OFF の比べ。2026-10-27 fix3 r2-rec2-1)。発進ごとの行は変えない')
    ap.add_argument('--skip', action='append', default=None, metavar='LOG[@T]',
                    help='まとめ(summary)から外す記録(何回でも。LOG@T なら AI 側にした時刻 T 秒の発進だけ。2026-10-26 fix3 '
                         '直す 1 回目: 車のせいでない理由で R1 の前に止めた本など。当たった本は外さない)。発進ごとの行は変えない')
    a = ap.parse_args(argv)
    first_only = bool(a.first_only)              # (a はこのあと周の a, b で上書きされる)
    a_ignore_gain = bool(a.ignore_gain)
    skip = [x.strip() for v in (a.skip or []) for x in v.split(',') if x.strip()]
    days = DEFAULT_DAYS if not a.day else tuple(sorted({x.strip().replace('-', '') for v in a.day for x in v.split(',') if x.strip()}))
    extended = days != DEFAULT_DAYS
    out_path = Path(a.out) if a.out else (OUT / 'launches.json' if not extended else OUT / ('launches_%s.json' % '_'.join(days)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks, names = bench_blocks() if extended else ({}, {})
    plans = bench_plan_launch() if extended else {}
    data, laps, unreadable = {}, {}, []
    noai = {}                          # 2026-10-27 fix6: AI 側のフレームが無く、シーンが欠けた本 {名前: (d, 欠けたシーン, 読めない記録)}
    day_logs = logs(days)
    for log in day_logs:
        try:
            d = load(log, extended, blocks, names) if extended else load(log)
        except Exception as e:      # noqa: BLE001
            print('skip', log, e)
            if extended:               # r2-rec2-5: 読めなかった記録もまとめに出す(前は上の 1 行だけで比べから消えた)
                unreadable.append(unreadable_info(log, e, blocks, names, plans))
            continue
        if not d['z']['ai'].any():
            if extended and d.get('data_missing'):      # 2026-10-27 fix6(あるときだけ。無ければ前と同じ)
                # フレームの無い欠けたシーンがある = AI 側にしたシーンごと書けずに落ちたかもしれない本(電源断・jetformer-web の落ち)。
                # 読めない記録(r2-rec2-5)に入れる。前は黙って飛ばし、回 3 の ON の本が落ちても比べから消えて「使ってよい」と出た
                ls = lost_scenes(d)
                u = None
                if ls:
                    st = launch_starts(d)
                    why = ('AI 側のフレームが無い(シーン %s にフレームが無い = npz が無い・読めない。AI 側にしたシーンごと書けずに'
                           '落ちたかもしれない)' % '・'.join(s for s, _, _ in ls))
                    print('skip', log, why + ('。events.jsonl に発進の前進の start %s' % starts_text(st) if st else ''))
                    u = unreadable_info(log, why, blocks, names, plans)
                    u['lost_scenes'] = [s for s, _, _ in ls]
                    if st:             # 発進の前進を出した出来事がある = ON(log.json・計画が OFF でも。途中で ON にした本)
                        u['launch_starts'] = st
                        u['launch_state'] = 'on'
                    elif u['launch_state'] != 'on':
                        # 2026-10-27 fix6-2: start が無くても(電源断で消えうる)、ほかの ON の手がかり(発進の前進の出来事・走行中に
                        # launch_boost を 1 に)があれば ON(あるときだけ。無ければ前と同じ)
                        src = launch_on_from(d, u)
                        if src:
                            u['launch_state'] = 'on'
                            u['launch_on_from'] = src
                    unreadable.append(u)
                noai[log] = (d, ls, u)
            continue
        data[log] = d
        laps[log] = laps_of(d)
    unknown = []
    if extended:                       # r3-r3-rec-1: log.json の読めない記録(前は logs() が黙って飛ばした。無ければ前と同じ)
        for log, why in unreadable_meta_logs(days):
            print('skip', log, '%s(読めない記録。回・発進の前進の ON/OFF が分からない)' % why)
            unknown.append(unreadable_meta_info(log, why, day_logs, blocks))
        if unknown:
            unreadable = sorted(unreadable + unknown, key=lambda u: u['log'])
    # 同じ設定(モデル・ゲイン・回)の周
    def key(d):
        lab = d['lab']
        no = ai_no(lab.get('description'))
        g = float(np.median(d['z']['gains'][d['z']['ai'].astype(bool), 1]))
        if d.get('extended'):          # ほかの日: 回と、周を変える設定(天井・左・自動合わせ・目標)も鍵に
            S = d['settings']
            return (d['model_name'], gain_key(S, round(g, 2)), d['session'], _on(S.get('speed_ceiling')),
                    S.get('ceil_left_mps') if _on(S.get('speed_ceiling')) else None,          # 2026-10-26 fix3: ON のときだけ
                    _on(S.get('speed_auto')), S.get('auto_target_mps') if _on(S.get('speed_auto')) else None)
        return (MODEL.get(lab.get('run') or '', lab.get('run')), round(g, 2), session_of(d['log'], no))
    by_key = {}
    for log, d in data.items():
        by_key.setdefault(key(d), []).append(log)

    def ref_laps(o):
        """ほかの本の「同じ設定の周」に使う周。走行中に変えた本(r1-rec3-3。ほかの日だけ)は、最初に変えた時刻より前に終わった周だけ
        (時刻の分からない変え方 = params_end だけ、なら使わない)。変えていない本は全部(今までと同じ)"""
        mid = data[o].get('midrun')
        if not mid:
            return laps[o]
        ts = [m.get('t_s') for m in mid]
        if any(x is None for x in ts):
            return []
        return [(a, b) for a, b in laps[o] if b <= min(ts)]
    rows, fly_slip, lap_rows = [], [], []
    for log, d in data.items():
        if laps[log]:
            ref = [(a, b, d) for a, b in laps[log]]
            kind = 'same_log'
        else:
            ref = [(a, b, data[o]) for o in by_key[key(d)] for a, b in ref_laps(o)]
            kind = 'same_setting' if ref else 'none'
        rows += launch_rows(d, ref, kind)
        fly_slip += [dict(s, log=log) for s in flying_slip(d, laps[log])]
        for a, b in laps[log]:
            lap_rows.append({'log': log, 'a': round(a, 3), 'b': round(b, 3), 'lap_s': round(b - a, 3)})
    lost_ev = {}                       # 2026-10-27 fix6: 読めた本の、欠けたシーンの間の発進の前進の start {名前: (時刻, 2 回目以降か)}
    if extended:
        for log, d in data.items():
            if not d.get('data_missing'):
                continue
            ls = lost_scenes(d)
            st = launch_starts(d, ls)
            if not st:
                continue
            t_read = [r['t_ai'] for r in rows if r['log'] == log]
            later = bool(first_only and t_read and min(t_read) < min(st))
            lost_ev[log] = (st, later)
            if later:                  # --first-only: 読めた発進のあとの発進(2 回目以降)は、読めても比べに入れない(summarize と同じ)
                continue
            sc = [s for s, lo, hi in ls if any(in_lost(te, [(s, lo, hi)]) for te in st)]
            u = unreadable_info(log, 'シーン %s にフレームが無い(npz が無い・読めない)間の、発進の前進を出した発進の記録が無い'
                                '(AI 側にしたシーンごと書けずに落ちた)' % '・'.join(sc), blocks, names, plans)
            u.update(lost_scenes=sc, launch_starts=st, launch_state='on')
            unreadable.append(u)
    # 2026-10-27 fix6-2(反証 6-2 回目 w2-tools-r62-1): 電源断では events.jsonl の start も消える(fsync しない。落ちる前の約 30 秒)。
    # 発進の前進 ON の読めた本で、最後の読めるフレームのあとにフレームの無い欠けたシーン(落ちた形)があれば、start が無くても、その間の
    # 発進(2 回目以降)の記録が無いかもしれないので読めない記録(ON)に入れる。上の start で入れた本は入れない。--first-only で読めた
    # 発進がある本は入れない(欠けた区間の発進は 2 回目以降)。発進の前進 OFF の本・真ん中だけ欠けた本は前と同じ
    lost_tl = {}                       # {名前: (欠けたシーン, 最後の読めるフレームの時刻, 2 回目以降か, ON の手がかり)}
    if extended:
        for log, d in data.items():
            if not d.get('data_missing') or log in lost_ev:
                continue
            tl = lost_tail(lost_scenes(d))
            if not tl:
                continue
            u = unreadable_info(log, '', blocks, names, plans)
            src = launch_on_from(d, u)
            if not src:                # 発進の前進 OFF の本: 欠けた区間の発進は記録に手がかりが無い(前と同じ。動画・紙で)
                continue
            sc = [s for s, _ in tl]
            t_last = tl[0][1]
            t_read = [r['t_ai'] for r in rows if r['log'] == log]
            later = bool(first_only and t_read and (t_last is None or min(t_read) <= t_last))
            lost_tl[log] = (sc, t_last, later, src)
            if later:                  # --first-only: 読めた発進のあとの発進(2 回目以降)は比べに入れない(上の決まりと同じ)
                continue
            u['why'] = ('シーン %s(最後の読めるフレーム %sのあと)にフレームが無い(電源断・落ちの形。npz が無い・0 バイト・読めない)。'
                        '発進の前進 ON の本なので、その間の発進(2 回目以降)の記録が無いかもしれない(電源断では events.jsonl の start も'
                        '消える)' % ('・'.join(sc), '%.2f 秒' % t_last if t_last is not None else '?'))
            u.update(lost_scenes=sc, lost_after_s=round(t_last, 3) if t_last is not None else None, launch_state='on',
                     launch_on_from=src)
            unreadable.append(u)
    if extended and (any(not later for _, later in lost_ev.values()) or any(not v[2] for v in lost_tl.values())):
        unreadable = sorted(unreadable, key=lambda u: u['log'])
    res = {'created_by': 'experiments/20261026_fix/diag_launch/collect.py', 'launches': rows,
           'flying_accel_slip': fly_slip, 'laps': lap_rows}
    skipped = []
    if extended:
        rows_sum, ur_sum = rows, unreadable
        if skip:                       # --skip(あるときだけ。無ければ前と同じ)
            hit = lambda r: any(x == r['log'] or x == '%s@%.2f' % (r['log'], r['t_ai']) for x in skip)      # noqa: E731
            skipped = ['%s@%.2f' % (r['log'], r['t_ai']) for r in rows if hit(r)]
            rows_sum = [r for r in rows if not hit(r)]
            # 読めない記録(r2-rec2-5)は記録の名前で外す(LOG@T の LOG でも)
            hit_u = lambda u: any(x.split('@')[0] == u['log'] for x in skip)      # noqa: E731
            skipped += ['%s(読めない記録)' % u['log'] for u in unreadable if hit_u(u)]
            ur_sum = [u for u in unreadable if not hit_u(u)]
        # problems: 読めた本と、AI 側のフレームが無くシーンが欠けた本(fix6。あるときだけ)の、記録の並び(無ければ前と同じ)
        books = [(lg, data[lg] if lg in data else noai[lg][0]) for lg in day_logs if lg in data or lg in noai]
        res.update(days=list(days), summary=summarize(rows_sum, first_only=first_only, ignore_gain=a_ignore_gain, unreadable=ur_sum),
                   summary_first_only=first_only, problems={lg: d['problems'] for lg, d in books if d.get('problems')})
        if unreadable:                 # r2-rec2-5(あるときだけ。無ければ前と同じ)
            res['unreadable'] = unreadable
        # 中だけの欄(頭が _。r2-rec2-2 の _lh_self)は JSON に書かない(欄の並びはそのまま)
        res['launches'] = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
        if skip:
            res.update(summary_skip=skip, summary_skipped=skipped)
        if a_ignore_gain:              # --ignore-gain(あるときだけ。無ければ前と同じ)
            res.update(summary_ignore_gain=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('logs', len(data), 'launches', len(rows), 'with R1', sum(r.get('to_r1_s') is not None for r in rows),
          'laps', len(lap_rows), 'flying accel windows', len(fly_slip))
    for r in rows:
        print('%s No.%-3s %-10s g%.2f kick=%d(%3dfr) mv%-6s m80=%-6s toR1=%-6s d=%-5s loss=%-6s parts=%s fly=%s/%s %s' % (
            r['log'], r['no'], r['model'], r['gain'], r['kick'], r['kick_frames'], r['move_0p4_s'], r['model80_s'],
            r.get('to_r1_s'), r.get('dist_m'), r.get('loss_s'), r.get('loss_parts'), r.get('fly_n'), r.get('fly_ref'),
            r.get('start_loc')) + (' 中止のあとの走り直し(%s・人側 %s 秒。%s)' % (      # 2026-10-27 fix7(あるときだけ)。r2: 止まった形の文
                abort_text(r['after_abort']), r['after_abort'].get('human_s'),
                '比べに入れない' if r.get('to_r1_s') is not None else 'R1 に届かなかった発進に数える') if r.get('after_abort') else ''))
    if extended:
        print('\n'.join(summary_lines(res['summary'])))
        unk_sum = [u for u in ur_sum if u.get('meta') is False]
        if unk_sum:                    # r3-r3-rec-1(あるときだけ): 組の外にも 1 行(ON のある組が無い日も名前が見えるように)
            for u in unk_sum:
                gs = ['%s %s' % (g['session'], g['model']) for g in res['summary']
                      if u['log'] in (g.get('unreadable_unknown') or {}).get('logs', [])]
                print('読めない記録(log.json も読めない)%s: %s。前の記録 %s・次の記録 %s。%s' % (
                    u['log'], u['why'], u['near'][0] or 'なし', u['near'][1] or 'なし',
                    ('発進の前進 ON のある組 %d 組(%s)を「満たさない」にした' % (len(gs), '・'.join(gs))) if gs else
                    '発進の前進 ON のある組は無い(どの組にも入れていない)'))
        # 2026-10-27 fix5 c1(あるときだけ): シーンが欠けた本(npz が無い・0 バイト・読めない)を 1 行ずつ。fix6: AI 側のフレームが
        # 無い本(比べの data に入らない)も出す。読めた本で欠けたシーンの間に発進の前進の start がある本は、後ろに 1 文(無ければ前と同じ)
        for lg in sorted(set(data) | set(noai)):
            d_ = data[lg] if lg in data else noai[lg][0]
            pr = d_.get('problems') or []
            if not pr:
                continue
            sc_ = []
            for x in pr:
                s_ = x.split('/')[0][len(lg) + 1:]
                if s_ not in sc_:
                    sc_.append(s_)
            S = d_.get('settings') or {}
            on_ = _on(S.get('launch_boost'))
            if lg in data:
                mid = '欠けた区間の衝突・8G 以上は記録に無い(最初の周に掛かれば「最初の周の記録が欠けた発進」で満たさない)'
                if lg in lost_ev:
                    st, later = lost_ev[lg]
                    mid += ('。欠けたシーンの間に発進の前進の start %s = その発進の記録が無い。%s' % (
                        starts_text(st), '読めた発進のあと(2 回目以降)なので --first-only の比べには入れない' if later else
                        '「読めない記録」に入れた(ON なので満たさない。--skip 記録名 で外せる)'))
                elif lg in lost_tl:    # 2026-10-27 fix6-2(あるときだけ。無ければ前と同じ文)
                    sc, t_last, later, _ = lost_tl[lg]
                    on_ = True
                    mid += ('。最後の読めるフレーム %sのあとのシーン %s にフレームが無い(落ちた形)= 発進の前進 ON の本なので、その間の'
                            '発進(2 回目以降)の記録が無いかもしれない(電源断では events.jsonl の start も消える)。%s' % (
                                '%.2f 秒' % t_last if t_last is not None else '?', '・'.join(sc),
                                '読めた発進のあとなので --first-only の比べには入れない' if later else
                                '「読めない記録」に入れた(ON なので満たさない。動画・紙で、その間に発進が無かった・当たっていないと'
                                '確かめたら --skip 記録名 で外せる。外すと読めた発進も比べから外れる)'))
            elif noai[lg][2] is not None:
                ls, u = noai[lg][1], noai[lg][2]
                on_ = u['launch_state'] == 'on'
                mid = ('AI 側のフレームが無い(シーン %s にフレームが無い = AI 側にしたシーンごと書けずに落ちたかもしれない)。「読めない'
                       '記録」に入れた(%s)' % ('・'.join(s for s, _, _ in ls),
                                         'ON なので満たさない。--skip 記録名 で外せる' if on_ else 'OFF なので比べは変えない'))
            else:
                mid = ('AI 側のフレームが無い(欠けたシーンにもフレームがあり、どこも AI 側にしていない = 発進なし。比べには入らない)。'
                       '欠けた区間の衝突・8G 以上は記録に無い')
            print('記録が欠けている本 %s(%s・発進の前進 %s): シーン %s の npz が無い・読めない(%d 件)。%s→ 動画・紙で(手順書 7 章「いつでも」)'
                  % (lg, d_.get('session') or '?', 'ON' if on_ else 'OFF', '・'.join(sc_), len(pr), mid))
        if skip:
            print('まとめから外した発進(--skip %s): %s' % (','.join(skip), '・'.join(skipped) or '(当たる記録なし)'))
        print('出力', out_path)


if __name__ == '__main__':
    main()
