# 昇格の段の報告 — 第 4 回で合格した 3 モードを JetFormer に入れた(sim_rl_modes_20261013 / r3)

- 作成 2026-10-15 04:15。段の役目: modes の作業(wf_7e500cf6-ccd)が終わったあと(journal に label 'review' の result があることを確かめた)、
  第 4 回(r4・r4b)の学習の段と確認の段で **合格した 3 つのモード(絶対壁にぶつからない・離脱復活・経路いろいろ)** を、統合の段が作った道
  (`bundle/<mode>/` → `rl/runtime` → `web/rl` → `jetformer/sim/static/rl` と登録簿 `data/sim_rl/modes/<mode>/`)で JetFormer に入れ、
  JetFormer 本体の学習除外の残り(離脱復活の指摘)を直し、テスト一式・ブラウザ(http://192.168.55.1:8080)の確認・見本の記録・ばらつきの物差しまで行う。
- 依頼(ユーザー、2026-10-13 21:20):「シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども
  作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意」、
  2026-10-14 08:30「作業を引き続き実施」(崩れた 2 つの優先 1 のモードと経路いろいろの第 3 回)。
- 守ったこと: 実車・PCA9685・I2C・GPU に触れていない。RL の学習はしていない。Firefox と pytest 一式は `jf-heavy --mem 2500M --lock big`、Node は `jf-heavy --lock none` + nice 15。
  ベストラップの作業のファイルは変えていない。git にコミットしていない。共有のファイルを変える前に `experiments/20261013_sim_rl_modes/r3/promote/before/` に写した
  (sha256 は `before/sha256_before.txt`、変えたあとは `r3/promote/sha256_after.txt`)。jetformer-web の再起動は待機中を API で確かめてから 2 回。

---

## 0. 結論

**3 つとも昇格した。JetFormer のシミュレーターで選べる強化学習モードは 5 つになり、全部「合格」**(障害物よけだけは出していない)。

| モード | 状態(登録簿) | 第 4 回の試験(1 回だけ) | 確認 r4(学習のコードを使わない反証) | ブラウザ(Firefox、8080) | 見本の記録(5 分 = 15 シーン) |
|---|---|---|---|---|---|
| 絶対壁にぶつからない `no_wall` | **合格**(前は不合格(参考)) | 3,038 回で衝突 0・最小すき間 8.10 cm(標準)/ 8.16 cm(物理のずれ)・停止から 1 周 9.75 秒 | 合格(新しい組 5,749 回で衝突 0) | F-ID 5 本 差 ≤ 5.6e-16・走行 OK | 15/15 合格・最小すき間 11.6 cm・シールド 1 フレーム/シーン |
| 離脱復活 `recover` | **合格**(前は不合格(参考)) | 乱しつき 100 回で 99 走り切り・成功 0.993・キック 4 組 ≥ 0.98・停止から 1 周 8.85 秒 | 合格(新しい開始 200/200・Firefox 100/100) | F-ID 4 本 差 ≤ 5.6e-16・走行 OK | 15/15 合格・ゲートの外し 0(前の版 6/15)・学習から外す 18% |
| 経路いろいろ `line` | **合格**(前は出していない) | 542 回で衝突 0・止まり 0・最小 4.13 cm・左右 34 cm・イン/アウト 16 cm | 合格(新しい組 + 攻め 2,077 回で衝突 0) | F-ID 2 本 差 ≤ 5.3e-16・走行 OK | 15/15 合格・8 つの線の形・線のばらつき 9.97 cm(教師の 28.5 倍) |
| 最速・速さいろいろ | 合格(そのまま) | — | — | F-ID 3 本(変わっていないことの確かめ)・走行 OK | 記録の段の見本をそのまま使う |

- **走行中の仕組みを 1 つにまとめた**(3 つの候補は別々の shield.mjs・mode_runtime.mjs で試験されていた)。まとめた版と候補の版を同じ開始・種・物理で走らせると、
  **閉ループ 139 回すべてで全フレームの操作・info.jf・教師の metadata・乱しの出来事がビット単位で同じ**、保存した F-ID 14 本も差 0(2 章)。
- **JetFormer 本体の学習除外の残りを直した**(展開したシーンから作り直すと正解が 1 フレームずれる・ジャンル 8 の無い Jetson で外した区間が広がりすぎる)。
  実機のシーン 138 は前後でビット単位で同じ、展開したシミュレーターのシーン 222 は正解(future_actions)だけ直る。**pytest 一式 679 passed・3 skipped**(5 章)。
- **見本 45 シーンを記録の段と同じ道で記録**(03:03〜03:59、外れ 0)し、3 つのデータセット `sim-rl-{no_wall,recover,line}_20261015-0302` に取り込んだ。
  全部、開ループの再生(姿勢の差 ≤ 8.8e-14 m)・教師の同一性・r4 の決まり・ユーザーの検査と同じ検査・記録そのものの数え直しに合格(7 章)。
- **ばらつき**: 5 モード(最速・速さは記録の段の見本)を合わせると、前の教師のデータに比べ **線のばらつき 22.7 倍(7.96 cm)・状態の網羅 9.5 倍**。
  記録の段と同じ 4 モードの組み合わせ(no_wall・recover を r4 に)では 18.3 倍・7.6 倍(前は 15.0 倍・7.5 倍)。経路いろいろ 1 つで 28.5 倍(8 章)。
- 残る注意(11 章): 絶対壁にぶつからないは今も「1 本の線」(線のばらつき 1.2 倍)で、最小すき間 8.1 cm はシールドの仕組みで守っている(方策だけでは 4.5 cm)。
  離脱復活は余裕が小さく(停止から 1 周 8.85 秒・基準 9.0、最小 3.5 cm・基準 3 cm)、すき間 5 cm 未満の開始とゲート 7 の右の開始で止まる。
  経路いろいろは方策だけでは角の一部でまだ崩れ、シールドが欠かせない。どれも記録の前の確かめが替え、画面の注意に書いた。

---

## 1. 始める前に確かめたこと・読んだもの

- `wf_7e500cf6-ccd/journal.jsonl` の最後は label 'review' の started → result(02:29 に確認)。modes の作業の 4 つの報告(integrate.md・record.md・review.md・README.md)と
  r3 の計画(plan.md 5 章「昇格の段」)、確認 r4 の 3 つの報告の申し送り(check_no_wall_r4.md 9・10 章、check_recover_r4.md 12 章、check_line_r4.md 8 章)を読んだ。
- 前の試み(この段の途中で電源が落ちたもの)は無かった(r3/promote・screens・promote.md が無い)。最初から行った。
- 見つけたときの状態: jetformer-web 動作中(電源断の再起動 16:36 のあと)、ブリッジ **停止中**・モデルなし、学習・走行・収集・ジョブなし、記録のプランは 2 つとも finished。
  modes の作業の成果は main の 2d928b4 にコミット済み(この段はコミットしない)。

## 2. 走行中の仕組みを 1 つにまとめた(`rl/runtime` = `web/rl` = `jetformer/sim/static/rl`)

| ファイル | 前(共有) | 候補 no_wall r4 | 候補 recover r4 | 候補 line r4b | まとめた版 |
|---|---|---|---|---|---|
| mode_runtime.mjs | 50de50e7 | 7f8ee13d | 0758cb1b | 2fa7146d | **f59b1955**(3 つの差分を足し合わせた) |
| shield.mjs | d682a6b5(v3) | 8ece9be4(v5) | 29c9d111(v4-proto) | 130ed5b2(v5 + v5b) | **130ed5b2**(no_wall・line) |
| shield_v4p.mjs | — | — | — | — | **29c9d111**(recover の shield.mjs をそのまま) |
| perturb_v2.mjs・recoverable_table.mjs | — | — | bb7e58f7・4ac6b682 | — | 同じ |
| ほか 14 ファイル | 共通 | 同じ | 同じ | 同じ | 同じ |

- 130ed5b2 は 8ece9be4 に「select 'v5b'(合格する候補が無いときはブレーキの候補だけから選ぶ)」を足しただけで、既定の select は 'v5'(確認 line r4 の申し送り 8.1 のとおり共有はこれ)。
- 離脱復活は試験に使った jf-shield/v4-proto(すくみをほどくあり、3.5 cm)を `shield_v4p.mjs` から使い、値は `RECOVER_R4.shield` で固定した。mode.json の runtime.shield_params は説明だけ。
  乱しは v2(立て直せる表 v2、`recoverable_table.mjs` の sha256 8735365c…)。
- 経路いろいろは、ページの teacher.js が登録簿の mode.json の `runtime`(shield: true・shield_params の select 'v5b'・4 cm)を `modeRuntime` で渡して初めてシールドが入る
  (候補の差分 `teacher.js.diff`・`sim-rl.js.diff` を当てた)。no_wall は mode.json の値 = shield.mjs の既定。
- **同じ動きの確かめ**(`r3/promote/equiv_runtime.mjs`、結果 `r3/promote/equiv/`): 試験の組から選んだ開始(no_wall は物理のずれ・摩擦円・遅延・低摩擦の範囲外も含む 58、
  recover は乱しあり・乱しなし 29、line は 7 つの組 40、変わっていない fastest・speed・obstacle 12)を、候補の runtime(line はページと同じく modeRuntime を渡す)と
  まとめた版(ページと同じく modeRuntime を渡す。no_wall は渡さない道も 1/4)で元の sim-core.js に走らせた:

  | モード | 回 | 全フレーム同じ | metadata 同じ | 乱しの出来事同じ |
  |---|---|---|---|---|
  | no_wall | 58 | 58 | 58 | 58 |
  | recover | 29 | 29 | 29 | 29 |
  | line | 40 | 40 | 40 | 40 |
  | fastest・speed・obstacle(前の共有の版と比べ) | 12 | 12 | 12 | 12 |

  F-ID(保存した snapshot の列を与える)14 本(no_wall 5・recover 4・line 2・fastest 1・speed 2)も操作・info.jf の差 0・metadata 同じ。
- `rl/make_web_bundle.mjs` で `web/rl` を作り直した(19 ファイル、manifest 5e6cf478…)。候補の runtime/manifest.json が古い問題(確認の 3 つの申し送り)はこれで消えた。
  `integrate/publish_registry.py` が `jetformer/sim/static/rl/` に写した(sha256 を照合)。RUNTIME_VERSION は変えていない(F-ID の info.jf.p が変わるため)。

## 3. bundle と登録簿

- `bundle/{no_wall,recover,line}/` を候補の一式で置き換え・新しく置いた(`r3/promote/write_bundles.py`。前の一式は `before/bundle/`)。中身は policy・eval・F-ID・シールドの値
  (no_wall `shield_config.json`、recover `shield.json`・`perturbation.json`・`recoverable_table_v2.json`、line `shield.json`)・`sha256.json`。
- **status.json = accepted「合格」**。r4 の試験の値(`test`)・判定の決まり(`judged_by`)・確認 r4 の結果(`check`、報告・数字)・注意(`caveats_ja`、確認の段の数字のまま)・
  置き換えた版(`replaces`)・走行中の仕組み(`runtime`、同じ動きの確かめの数字)を書いた。recover は確認 r4 の申し送りどおり注意 (a)〜(d) の数字を入れた。
- mode.json は候補のまま + 確認の段の申し送り(`overrides_from_promote` に鍵の名前):
  - no_wall: `dataset_card`(取り込むデータセットのカードに写る。r4 の数字・方策だけで 4.5 cm・物理のずれの隅 7.0 cm・記録の分布の外で止まる)・`ui_note_ja`。
  - recover: `ui_note_ja` を「開始のすき間 **5 cm** より小さい開始と **ゲート 7 の右寄り** の開始で止まることがある」に(確認 r4 の申し送り 12.4)、`dataset_card` に確認 r4・余裕・「強」固定の値。
  - line: `dataset_card`(線の形・r4b の数字・シールドなしで崩れる角・右寄り/アウトの追従)・`ui_note_ja`。
- 登録簿: `integrate/publish_registry.py` の VERDICTS とカードを r4 にして(前の版は `before/publish_registry.py`)走らせた。bundle の status の r4 の中身を登録簿の status にも写す。
  line の `eval.json` は回ごとの行を除いた(0.85 MB → 15 KB、`eval_full` に元の sha256)。カードの「ブラウザの同一性」は 6 章の F-ID の結果。
- 失敗した第 3 回の候補 `r3/candidates/{no_wall,recover,line}_r3/status.json` を確認 r3 の結果で not_accepted に正した(line_r3 は学習の段の accepted のままだった)。

## 4. 記録のシーンの組(`data/sim_rl/modes/<mode>/scenes.json`)

`r3/promote/make_scene_pool_r4.mjs`(統合の段の make_scene_pool.mjs の写し。まとめた runtime + mode.json の runtime を modeRuntime で渡す = ページと同じ)。
決まり = r4 の追補の記録の前の確かめ(止まり = フレーム 30 以降に 0.2 m/s 未満が 30 フレーム以上(教師に渡した状態と sim のステップ後の両方)・シールドの作動の割合・後ろへの動き 0・
ゲートの外し 0(離脱復活も r4 から)・離脱復活は 1 周以上進む)+ 統合の段の決まり(衝突 0・有効な周 ≥ 1・すき間)。開始・種は統合の段と同じ作り方(Halton、種 700000 + 1000·m + 番号)。

| モード | 組 | 候補 | 合格 | 落ちた理由 | 組の最小すき間 | 時間 |
|---|---|---|---|---|---|---|
| no_wall | 90 | 91 | 91 | — | 10.6 cm | 133 秒 |
| recover | 90 | 92 | 91 | 乱しのあとの衝突 1 | 2.0 cm(すき間の下限は決まりに無い) | 99 秒 |
| line | 90 | 91 | 91 | — | 8.0 cm | 115 秒 |

- 経路いろいろの線の形は、modes.json の family_plan_15 の割合(15 本ごとに 左 2・右 2・中央 1・イン 2・アウト 2・レイト 2・アーリー 1・ランダム 3)で順に回した。
  連続した 15 本ならどこから取っても同じ割合になる(最初は種から決めた形で作ったが、プランの窓に左寄り・中央・アーリーが入らないので記録の前に作り直した。deviations.md)。

## 5. JetFormer 本体

| ファイル | 変えたこと |
|---|---|
| `jetformer/dataset.py` | 展開したシーン(操作がカメラのフレームと 1 対 1)は正解(未来の操作)をフレームの番号で引く(`_frame_indexed`。確認の段の `dataset_frame_index_v2.patch`) |
| `jetformer/scene_import.py` | 展開したシーンの「未設定のまま残す区間」を、学習除外で外した観測(mode 0 かつ future_valid = False)だけにする(ジャンル 8 の無い Jetson でもシーン全体が未設定にならない。`scene_import_unset_segments.patch`) |
| `jetformer/sim/static/rl/teacher.js`・`static/sim-rl.js` | mode.json の runtime を modeRuntime として渡す(line の差分)。欄に `ui_note_ja` の注意・線の形・シールドのフレーム数(シールドのあるモード全部)を出す |
| `jetformer/sim/rl_modes.py`・`web/static/app.js` | `ui_note` を画面へ。前の版の方策のデータセット(カードに `rl_superseded`)を「集めたデータ」の数から分け、「(ほかに前の版の方策で記録した N データセット)」と出す |
| `tests/test_sim_modes.py` | ジャンルの表を一時フォルダーに書く(本物の data/modes.json に頼らない)+ ジャンル 8 なしの試験(`test_sim_modes_hermetic.patch`) |
| `tests/test_sim_label_frame_index.py`・`tests/test_sim_label_mask_e2e.py`(新) | 33,333/33,334 µs のシーンで future_actions[k] = vehicle[k+1..k+10]・取り込み → 展開 → 作り直し → ジャンルの付け直し → 作り直しで乱した操作が正解に入らない |
| `tests/test_sim_modes_r4.py`(新、4 件) | teacher.js が runtime を渡す・モードごとのシールド(line v5b / なし、no_wall v5、recover v4-proto と RECOVER_R4・乱し v2)・配るファイルと sha256・ui_note・前の版のデータセットの数え方 |
| `README.md` | 強化学習モードの表(5 つ合格)・第 4 回・組の決まり・replay_check_r4・正解をフレームの番号で引くこと・試験 |

- 確認の段の 2 つの指摘(「シーンからの作り直しで正解がずれる」「シーンタブのジャンルの付け直しで外した区間が消える」)のうち、後者は統合の段がすでに直していた(`_segments_keeping_mask`)。
  この段は前者と、後者の残り(ジャンル 8 の無い Jetson)を直した。
- **pytest 一式**(jf-heavy --lock big): 差分を当てたあと **678 passed・3 skipped**、最後の小さな変更(前の版のデータセットの数え方・試験 1 件)のあとにもう一度 **679 passed・3 skipped**
  (`r3/promote/pytest_full.log`・`pytest_full_2.log`。確認の段の写しで落ちた SIGINT の試験は本物のリポジトリでは通った)。
- **本物の data/raw で前後の比べ**(`r3/rc4/promote/real_scene_equal_rc4.py`、結果 `r3/promote/real_scene_equal_promote.json`): 360 シーンのうち実機の 138 はビット単位で同じ、
  展開したシミュレーターのシーン 222 は future_actions だけが変わる(1 シーン 400 観測、future_valid の数は同じ)。

## 6. ブラウザ(http://192.168.55.1:8080、DISPLAY :0 の snap Firefox 157.0.1、geckodriver 4448)

ページは非ループバックの http(`isSecureContext` = false・`crypto.subtle` なし)で動いた。結果 `r3/promote/browser/result.json`・`browser_fid.json`、画面 `r3/screens/`。

- **F-ID(ページの teacher.js = 登録簿の mode.json の runtime を渡す道)**: 14 本すべて合格。同じことを Node でも teacher.js を通して行い 14/14 で差 0・metadata 同じ。

  | モード | 列 | 舵の差(最大) | 速さの差(最大) | info.jf の違い | metadata | シールドのフレーム | 1 フレームの時間(平均 / 95% / 最大、33 ms 超) |
  |---|---|---|---|---|---|---|---|
  | no_wall | fid_std / slow / corner | ≤ 3.9e-16 | ≤ 2.7e-15 | 0 | 同じ | 1 / 388 / 278 | 3.2 / 8 / 15、2.6 / 5 / 11、4.6 / 8 / 10 ms(0) |
  | no_wall | fid_escape / fid_escape_rev(後ろへ 12 フレーム) | ≤ 5.6e-16 | ≤ 2.7e-15 | 0 | 同じ | 12 / 74 | 3.4 / 9 / 14 ms、3.0 / 7 / **115 ms(1 フレーム)** |
  | recover | FP_P11・FP_P24・RP_P10 | ≤ 5.6e-16 | ≤ 2.5e-15 | 0 | 同じ | 6・13・5 | 1.4〜7.8 / 4〜5 / 13・**301・244 ms(0・19・8 フレーム)** |
  | recover | RP_P27(止まりの開始。すくみをほどくが 91 フレーム) | 1.1e-16 | 7.5e-16 | 0 | 同じ | 144 | **80.8 / 118 / 135 ms(413 フレーム)** |
  | line | fid_std_left08 / g30r120 のシールド | ≤ 5.3e-16 | ≤ 2.7e-15 | 0 | 同じ | 0 / 3 | 1.8 / 3 / 16、2.1 / 4 / 10 ms(0) |
  | fastest・speed | 3 本 | ≤ 5.0e-16 | ≤ 8.2e-15 | 0 | 同じ | — | 0.3〜0.4 / 1 / ≤ 5 ms |

  1 フレームの時間は Firefox の performance.now(1 ms 刻み)。no_wall の抜け出しの計画のフレームは Firefox で 115 ms(Node の混んだ CPU では最大 0.8 秒)。
  離脱復活は乱しの表を引くフレーム・すくみをほどくフレームが 0.2〜0.3 秒かかることがあり、止まりの開始(P27)では 1 フレーム約 0.12 秒が続く(ページは約 8 fps に落ちる。
  記録の前の確かめがその開始を替えるので記録には出ない。記録はシミュレーターの固定の刻みなので、遅くなるのは時間だけ)。
- **各モードの走行(U1)**(`?rl=<key>`・実行速度「高速」・sim の約 10 秒。`screens/drive_<key>_{window,panel,camera}.png`):

  | `?rl=` | フレーム | sim 秒 | 衝突 | 周 | 走行中の欄 |
  |---|---|---|---|---|---|
  | no_wall | 314 | 10.47 | 0 | 1 | 最小すき間 13.2 cm・シールド 1 フレーム |
  | recover | 313 | 10.43 | 0 | 1 | 乱し 2 回・学習から外す操作 42 フレーム・最小 9.2 cm |
  | line | 306 | 10.20 | 0 | 1 | 線 中央(c0 −0.17)・速さの倍率 93%・最小 11.6 cm(記録のあとにもう一度: 312 フレーム・衝突 0) |
  | fastest / speed | 315 / 311 | 10.50 / 10.37 | 0 | 1 | 最小 6.3 cm / 倍率 87.5%・10.0 cm |
  | rc(モード未選択) | 312 | 10.40 | 0 | 1 | RL の教師は作られていない(元の RC加速教師) |

  欄には mode.json の注意(`注意: …`)が出る(`screens/drive_no_wall_panel.png`・`drive_recover_panel.png`)。
- **カード(U3)**(`screens/promote_tab_card.png`・`promote_tab_card_{no_wall,recover,line}.png`、記録のあと `*_after.png`): 5 モードが「合格」・障害物よけは「出していません」。
  no_wall「試験 3038 回で衝突 0・最小すき間 8.1 cm(標準の物理。シールドのしきい値)・8.2 cm(物理のずれ)。停止から 1 周 9.75 秒」、recover「乱した 100 回のうち 99 回が 20 秒を走り切った…」、
  line「試験 542 回で衝突 0・止まり 0・最小すき間 4.1 cm。左右の線の差 34 cm・イン/アウト 16 cm・線のばらつき 9.9 cm」。記録のあとは「集めたデータ 1 データセット・15 シーン・5.0 分
  (ほかに前の版の方策で記録した 2 データセット)」。

## 7. 見本の記録(プラン `20261015-030258-d27688`)

- 記録の段と同じ道(`r3/promote/record_plan_r3.py` = `record/record_plan.py` の置き場所だけ r3 に): http://192.168.55.1:8080 → シミュレーター タブ → カード「強化学習モード」で
  no_wall・recover・line に 5 分ずつ(ほかは 0)→ 検証の割合 0.2・種 自動(= 2)→「取り込んだら、元のシーンを消す」を外す →「プランを作って開く」→ 新しいタブで「開始」1 回。
  記録のためにブリッジを起動した(モデルなし。換算は収集器の既定 1441 / 1496 µs)。**03:03〜03:59(56 分、1 シーン約 75 秒)、45/45 シーン・外れ 0・替え 0**。
  画面 `screens/record_*.png`。結果 `r3/promote/record/record_result_20261015-030258-d27688.json`。
- 取り込み(ページの実行係が自動): ジャンル 8・タグ(mode.json の tags)・学習除外・カードの `rl`・`rl_notes`(mode.json の dataset_card)・シーンタブへの展開。

| データセット | シーン(train / val) | 学習に使えるフレーム | タグ | 教師の印(context.teacher) |
|---|---|---|---|---|
| `sim-rl-no_wall_20261015-0302` | 15(12 / 3) | 8,850 / 9,000 | simulator・sim-rl・sim-rl-no_wall・sim-safe | jf-rl-no_wall-v1・e632d96b…・jf-shield/v5(select v5・8.1 cm) |
| `sim-rl-recover_20261015-0302` | 15(12 / 3) | **7,227** / 9,000(乱した操作の前後 1,623 フレームを除外) | 〃・sim-rl-recover・sim-recover | jf-rl-recover-v1・68e321e4…・v4-proto 3.5 cm・乱し v2(表 8735365c…) |
| `sim-rl-line_20261015-0302` | 15(12 / 3) | 8,850 / 9,000 | 〃・sim-rl-line・sim-line | jf-rl-line-v1・5c197a2b…・jf-shield/v5(select v5b・4 cm)・線 v2 |

- **確かめ(全部合格)**(`r3/promote/after_record_r3.sh`。結果 `r3/promote/record/{replay,verify,check_recorded}/<mode>.json`):

  | 確かめ | no_wall | recover | line |
  |---|---|---|---|
  | 開ループの再生(元の sim-core.js、姿勢の差) | ≤ 6.4e-14 m | ≤ 5.8e-14 m | ≤ 8.8e-14 m |
  | 同じ状態での教師の操作(舵 / 速さ)・info.jf の違い | 1.3e-13 / 5.4e-13・0 | 2.4e-13 / 9.1e-13・0 | 2.6e-13 / 5.3e-12・0 |
  | r4 の決まり(`replay_check_r4.mjs`): 衝突 0・有効な周・止まり・すき間・シールドの割合・後ろへ・ゲートの外し・進み | 15/15(周 2 × 15・最小 **11.6 cm**・止まり 0・シールド ≤ 0.17%・後ろへ 0・外し 0) | 15/15(周 2 × 15・最小 **3.7 cm**・外し **0**・進み ≥ 56.2 m) | 15/15(周 2 × 15・最小 **10.1 cm**・シールド ≤ 0.5%・後ろへ 0・外し 0) |
  | ユーザーの検査と同じ検査(`record/verify_mode_dataset.py`: センサーの時刻・RGB/ToF 600 枚とも違う・IMU・PWM・未来ラベル・学習除外の数え直し・カード・シーンタブ) | 15/15 | 15/15(除外 1,623 = カード) | 15/15 |
  | 記録そのものからの数え直し(`check_recorded_scenes_rc4.mjs`: 止まり・進み・ゲートの外し、ゲートの数えが evaluation-report と同じ) | 15/15 | 15/15 | 15/15 |
  | 記録の前の予想(Node の閉ループ)と分かれる最初のフレーム | 372〜599(8 シーンは 600 フレーム全部同じ) | 31〜134 | 34〜136 |

- 離脱復活の乱し: 1 シーン 3〜5 回(平均 4.1)・学習から外す割合 平均 18%、立て直し(protocol definitions.recovered、乱しなしの線は登録簿の r4 の方策で作り直したもの)
  62 回のうち成功 **60(96.8%)**・立て直しの時間 中央値 0 秒・95% 点 0.92 秒(前の版の見本は 63/66 = 95%、ゲートの外し 2 シーン → 今回 0)。
- 経路いろいろの 15 シーンは、左寄り 2・右寄り 2・中央 1・イン 2・アウト 2・レイト 2・アーリー 1・ランダム 3(計画どおり。フレーム 0 の info.jf.p.line で確かめた)。
- 確かめのあと、元のシーンの ZIP 45 個を JetFormer の API(`DELETE /api/sim/datasets/<id>`)で消した(`r3/promote/record/deleted_sources.json`。記録の段と同じ)。ブリッジは止めた。

## 8. ばらつき(`r3/variation.json`、記録の段と同じコード `record/variation.mjs`)

前の教師のデータ(`jetformer-simulator-id8-20min`、60 シーン。VM1 7.6 µs・VM3 0.35 cm・VM7 279)を同じ式で計算した値を 1 とした比。最速・速さいろいろは記録の段の見本(1125)。

| 群 | VM2 位置でそろえた舵 | VM3 線のばらつき | VM4 速さ | VM7 状態の網羅 | 開始ゲート | 1 周目 | シーンの最小すき間 |
|---|---|---|---|---|---|---|---|
| 前の教師 | 4.2 µs(1) | 0.35 cm(1) | 0.012 m/s | 279(1) | 1 | 8.71〜8.76 秒 | 3.2 cm |
| 最速 | 41.5 µs(9.9) | 1.82 cm(5.2) | 0.050 | 768(2.8) | 15 | 7.91〜8.62 | 1.4 cm |
| 絶対壁にぶつからない(r4) | 10.3 µs(2.5) | 0.41 cm(**1.2**) | 0.016 | 299(**1.1**) | 15 | 9.51〜9.88 | **11.6 cm** |
| 離脱復活(r4) | 64.1 µs(15.3) | 4.11 cm(11.8) | 0.102 | 1,466(5.3) | 15 | 8.70〜9.55 | 3.7 cm |
| 速さいろいろ | 50.2 µs(12.0) | 2.17 cm(6.2) | **0.466** | 1,250(4.5) | 15 | 8.14〜12.68 | 8.4 cm |
| **経路いろいろ(r4b)** | 37.3 µs(8.9) | **9.97 cm(28.5)** | 0.100 | 1,656(6.0) | 15 | 8.54〜9.54 | 10.1 cm |
| **5 モード合わせ**(各 15 シーン / VM7 は各 1,800 フレーム) | 66.2 µs(15.8) | **7.96 cm(22.7)** | 0.320 | **2,635(9.5)** | 24 | 7.91〜12.68 | 1.4 cm |
| 4 モード(記録の段と同じ組み合わせ、no_wall・recover を r4 に) | 66.2 µs(15.8) | 6.41 cm(18.3) | 0.364 | 2,100(7.6) | 24 | — | — |
| (参考)記録の段の 4 モード(no_wall r2・recover r1) | 63.4 µs(15.2) | 5.26 cm(15.0) | 0.371 | 2,090(7.5) | 22 | — | — |

- protocol の判定(記録の段と同じ式): 全体 VM3 0.0796 m ≥ 0.039・VM7 9.5 倍 ≥ 3 → **合格**。離脱復活 出来事 4.1/シーン ≥ 3.5・外す割合 0.18(0.10〜0.40)・VM7 5.3 倍 ≥ 2 → 合格。
  速さいろいろ(記録の段の見本)VM4 0.466 ≥ 0.4・1 周目の幅 4.54 秒 ≥ 3 → 合格。
- 1 つ抜いた 5 モード合わせ: 経路いろいろを抜くと VM3 6.41 cm・VM7 2,100、離脱復活を抜くと 7.05 cm・2,275、ほかを抜いても 8.0〜8.7 cm・2,305〜2,768。**経路いろいろがいちばん効く**。
- 絶対壁にぶつからない(r4)は r2 と同じく、どこから始めても 3 m ほどで 1 本の線に乗る(線 1.2 倍・網羅 1.1 倍)。安全な線の見本で、ばらつきにはほとんど効かない。
- 図: `r3/fig/promote_record_trajectories.png`(上から見た軌跡)・`promote_record_variation_summary.png`・`promote_record_ey_by_s.png`・`promote_record_speed_hist.png`・`promote_record_clearance_hist.png`。

## 9. 前の版の方策で記録したデータセット

- `sim-rl-no_wall_20261014-{1025,1125}`(r2、シールド v3)・`sim-rl-recover_20261014-{1025,1125}`(r1、乱し v1)は作り直さず、カードに `rl_superseded` を足した
  (置き換えた方策・離脱復活はゲートを外したシーンの一覧 **1025 は 2/3・1125 は 6/15**。記録の段・README の「2/15」は誤りで、確認 r4 の数え直しが正しい)。データ・ラベルは変えていない。
  前のカードは `r3/promote/before/dataset_cards/`。画面の「集めたデータ」では前の版として分けて出る。

## 10. 決まり・計画の外のこと

`deviations.md` の末尾「昇格の段(r3/promote、2026-10-15)」に 11 項目(シールドを 2 つのファイルにした・metadata の書き方・line の eval.json を軽くした・組の作り方と線の形の割り当て・
登録簿を 2 回書いた・本体の小さな足し・前のデータセットのカード・第 3 回の候補の状態・再起動 2 回とブリッジ・元のシーンを残して確かめてから消した・離脱復活のカードの見出し)。
同じ所に、確認の段の申し送りどおり no_wall r4・line r4b の学習の段の逸脱の文書(`no_wall_r4/deviations_r4.md`・`line_r4b/deviations_line_r4b.md`)を写した。
ユーザー向けのまとめ `README.md` の先頭に、第 4 回で 3 モードが合格して差し替えたことの追記を入れた(本文は第 2 回までのまま)。

## 11. 残る注意・課題(数字つき)

1. **絶対壁にぶつからない**: 標準の物理の最小すき間はシールドのしきい値 8.1 cm に張り付く(方策だけでは 4.54 cm)。物理のずれの隅では 7.0 cm、舵の速さ 0.8 で柱へ向いた開始で止まりの小さな島。
   記録の分布(±0.2 m・±15°・開始のすき間 12 cm 以上)の外の開始では 200 回中 6 回止まる(衝突 0)= 画面の注意に書いた。見本は線のばらつき 1.2 倍で、ばらつきには効かない。
2. **離脱復活**: 余裕が小さい(停止から 1 周 8.85 秒・基準 9.0、乱しなしの最小 3.50 cm・基準 3 cm)。開始のすき間 3〜5 cm の開始は 9/60 で止まり、ゲート 7 の右寄りの開始でも止まる(r4 で新しく出た)。
   止まる開始では 1 フレーム約 0.12 秒かかり、ページが約 8 fps に落ちる。乱しの強さを「強」に固定すると立て直しの成功 0.957。半分以上の乱しは線の帯から出ない(C2 を高く見せる)。
3. **経路いろいろ**: 方策だけ(シールドなし)では、ゲート 32 の左寄りの端など角の一部でまだ崩れる(確認 r4 の攻めの組で 23/360 衝突。シールドありでは 0・最小 4.0 cm)。
   右寄り・アウトの線は追従の 95% 点が 15〜18 cm。ゲート 11 の開始で黒シートの横断が数えられないことがある(記録の前の確かめが替える)。
4. ブラウザと Node は閉ループで分かれる(今回の記録: 離脱復活・経路いろいろは 31〜136 フレーム目から、no_wall は 8 シーンが最後まで同じ)。記録はそのまま「方策の走り」だが、
   前もって確かめたシーンが記録で外れることはある(今回 45 シーン中 0)。取り込んだら `r3/promote/replay_check_r4.mjs` で確かめる(README に書いた)。
5. **このデータで JetFormer を学習して効くか**(シミュレーターの閉ループ・実機の検証の誤差)はまだ試していない(この作業の範囲の外)。次は 5 モードの見本 + 前の教師で学習し、
   まず離脱復活・経路いろいろのデータで「ずれたときの戻し方」「いろいろな線」が入るかを見る。
6. 空き容量は約 9.4 GB(この段の記録で残したのは データセット 3 つ 0.73 GB・シーンタブ 0.13 GB。pytest 一式が /tmp に残した一時フォルダー 3 GB は消した)。

## 12. ファイル・やり直し方

- コード `experiments/20261013_sim_rl_modes/r3/promote/`: `rt_merged/`(まとめた runtime。共有の rl/runtime に写したもの)・`equiv_runtime.mjs`・`make_equiv_cases.py`・`run_equiv.sh`・
  `write_bundles.py`・`episode_r4.mjs`・`make_scene_pool_r4.mjs`・`run_pools_r4.sh`・`replay_check_r4.mjs`・`browser_promote.py`・`record_plan_r3.py`・`after_record_r3.sh`・`chain_after_record.sh`・
  `make_variation_json_r3.py`・`record_figs_r3.py`・`note_old_datasets.py`・`before/`(変える前の写し)。共有で変えたもの: `rl/runtime/`・`web/rl/`・`bundle/{no_wall,recover,line}/`・
  `integrate/publish_registry.py`・`integrate/pools/pool_{no_wall,recover,line}.json`(報告の側)。
- 結果 `jetformer_reports/sim_rl_modes_20261013/r3/promote/`: `equiv/`・`pools/`・`browser/`(result.json・result_after.json・run.log)・`browser_fid.json`・`record/`(replay・verify・
  check_recorded・variation_raw_5modes.json・variation_raw_4modes.json・figdata_5modes.json・nominal_line_recover_r4.json・deleted_sources.json・record_plan.log)・`pytest_full*.log`・
  `real_scene_equal_promote.json`・`publish_{1,2}.json`・`progress.log`・`sha256_after.txt`。まとめ `r3/variation.json`、画面 `r3/screens/`、図 `r3/fig/promote_record_*.png`。

```bash
cd ~/jetformer/experiments/20261013_sim_rl_modes
H="/home/jetson/bin/jf-heavy --mem 1500M --lock none --need-mb 600 -- nice -n 15"
python3 -I r3/promote/make_equiv_cases.py > cases.json && $H node r3/promote/equiv_runtime.mjs --cases cases.json --out equiv.json   # 同じ動きの確かめ
node rl/make_web_bundle.mjs && python3 -I r3/promote/write_bundles.py && bash r3/promote/run_pools_r4.sh                            # web/rl・bundle・シーンの組
python3 integrate/publish_registry.py --browser-fid ~/jetformer_reports/sim_rl_modes_20261013/r3/promote/browser_fid.json            # 登録簿・static/rl
/home/jetson/bin/jf-heavy --mem 2500M --lock big -- python -u r3/promote/browser_promote.py                                           # Firefox(8080)
/home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 1800 -- python -u r3/promote/record_plan_r3.py --minutes no_wall=5,recover=5,line=5 --keep-sources
bash r3/promote/after_record_r3.sh <プランの ID>                                                                                     # 確かめ・ばらつき・図
```

## 13. 状態(段の終わり)

- jetformer-web: 03:02 に再起動した版(この段のコード)で動作中。ブリッジ: **停止中**・モデルなし(見つけたときと同じ)。ブリッジの作業場所のシーンの ZIP は 0 個。
- 起動したプロセス(Node の比べ・シーンの組・確かめ、Firefox・geckodriver、pytest)はすべて終わっている。Firefox のプロファイル(2 つ、計 268 MB)は消した。
- git にはコミットしていない(JetFormer 本体・試験・README・experiments の変更は作業ツリーにある)。
