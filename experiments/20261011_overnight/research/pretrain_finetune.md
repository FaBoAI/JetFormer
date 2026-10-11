# JetFormer:大規模事前学習 → コース別ファインチューニングの設計(調査と段階計画)

作成: 2026-10-11(夜間調査)。対象: `~/jetformer` の JetFormer v0(`jetformer/model.py`)。
目的: 「大規模事前学習 → コース別ファインチューニング → Offline強化学習」の流れのうち、**前の2段**を具体的に設計する。
最後の Offline RL へのつなぎ方も11章に書く。

**この文書について**

- 2022〜2026年の論文・公式資料を読んでまとめた。出典は [番号] で示し、URL は末尾にまとめた。
- **実験はしていない**。GPUで学習ジョブが動いていたため、torch と CUDA は使っていない。
- 遅延の数値は文献の値。JetPack 7.2 / TensorRT 10.16 の自分の環境では測っていない。採用する前に必ず実測すること(10章のStage 1)。
- 文中の「期待効果」は**仮説**である。文献の数値は分野(実車・シミュレータ・マニピュレーション)が違うので、そのまま当てはまらない。
  段階ごとに「合格の基準」を決めてあり、実測して判断する作りにした。
- **改訂(2026-10-11 06時台、Jetson の電源断から復帰した後)**
  - 主な引用を一次資料で確認し直した。誤りを3件直した: [10] の題名と結果、[42]・[49] の URL。結果は末尾の「検証メモ」にまとめた。
  - 初版の後に出た実測を1章に足した: clean_driving の学習結果(操作値の入力あり/なし)と、新しい AI 走行 009/010/101 の集計。
    これらは結論の 7・8 を裏づける材料なので、0章の 9 にまとめた。

---

## 0. 結論(先に要点)

1. **「大規模」の中身は、公開の運転データではなく次の3つにする。**
   - (a) 基盤モデル(DINOv2/v3 など)の画像特徴
   - (b) 自分の全ログ(人・AI・停止・衝突を含む)での自己教師あり学習
   - (c) コースの配置を変えて集める「複数コース」の走行データ

   nuScenes・Waymo・YouTube の車載動画は、カメラ高さ(約1.5m級 対 0.12m)・速度・環境が違いすぎる。
   方策の事前学習には使わない(2章)。
   小型車の公開データで最も近いのは GNM 系の CoryHall(RCカー、屋内廊下、2時間)[1][4]。
   それでも行動の表し方と環境が違うので、使うとしても画像エンコーダの補助まで。
2. **画像バックボーンは、まず「強い特徴をそのまま入れる」実験から始める。**
   - 候補は DINOv2 ViT-S/14(21M、Apache-2.0)または DINOv3 ViT-S/16(21M、独自ライセンス)[39][41]。
   - Orin Nano(MAXN_SUPER、TensorRT 10.3)での実測例: ViT-S/16 は FP16 で 2.36ms、ResNet-50 は 1.86ms(224px)[46]。
   - 今のモデル全体は 1.9ms(自前の実測)。差し替えても推論は +2〜4ms 程度と見込む(推定)。30fps の範囲に収まる。
   - ただし ViT は FP16 で精度が落ちる報告がある(ViT-S/16 で 4〜7ポイント)[46][49]。FP32 との差の確認は必須。
   - 遅延を一切増やしたくない場合は、DINOv2-B を教師にして今の ResNet-18 に特徴を蒸留する(Theia・RADIO の方式)[43][44]。
3. **自前データの自己教師あり学習では、次の5つの目的を1つのエンコーダに同時に学習させる。**
   - 画像のマスク復元、または基盤モデルの特徴の蒸留
   - 画像から今の ToF 距離を当てる
   - 画像と操作から 0.3〜0.7秒先の ToF を当てる(ViDAR 型)
   - 逆ダイナミクス(前後のフレームから操作・ヨーレートを当てる)
   - 潜在空間での次状態予測(LAW 型)

   補助の出力はエクスポートのときに外すので、**走行時の計算は増えない**。
   「未来の ToF」の予測は、R1(入口が速いと当たる)の弱点に直接効く学習信号になる。
   潜在空間の次状態予測は、そのまま Offline RL の世界モデルにも使える [21][22][33][36]。
4. **方策の事前学習は「複数のコース配置 × ジャンル × ペース」で行う。**
   - データ量の効き方は、1つの環境での周回数より**環境の数**で決まる(べき乗則)。
     環境あたり約50デモで頭打ちになる、という報告がある [59]。
   - 最初の目標: 8配置 × 約25周のきれいな周 + 立て直し、評価用に2配置を学習から外す。
   - ジャンルは今と同じ分岐ヘッド(CIL)にする。加えて FiLM での条件付けと、連続値の「ペース」を入れる [51][52][53]。
5. **コース別ファインチューニングは段階を踏む。**
   - 順序は「頭だけ(線形プローブ)→ LoRA(融合Transformer)→ 必要なら前段の層だけ解凍」[55][56][57][58]。
   - LoRA は ONNX に出す前に重みへ統合できるので、**TensorRT の推論速度は変わらない**。
   - 学習は Jetson 上で 7〜20分で済む見込み(推定、12章)。
6. **計算資源は次のように分ける。**
   - 事前学習(自己教師あり・複数コース)はクラウドGPUで行う。1回あたり A100/H100 で数GPU時間、5〜30ドル程度(推定)。
   - Jetson は、収集・推論・コース別の軽い微調整・評価に限る。Jetson での事前学習は、1M枚規模だと2.5日以上かかる(12章)。
     8GB・スワップなしでメモリも足りないので現実的でない。
7. **評価は走行(閉ループ)の指標を主にする。** 予測誤差(µs)だけでは走りの良し悪しは分からない [65][66][67]。
   オフラインの指標は、次の2つを足して相関を高める。
   - AI走行中に人が取り返した区間で作る「介入区間の検証セット」(9章)
   - 当たりやすい場所(R1/DE/GM)ごとの誤差

   事前学習が本当に役立っているかは、**学習に使っていないコース配置への適応曲線**で測る(0/5/10/20周での走行指標)。
8. **事前学習では直らないこと**もはっきりさせておく。
   - 「今の操作値・速度」をまねる近道(ego status のショートカット・copycat)[63][64]
   - 人の衝突をまねること

   これはデータの選び方、入力のドロップアウト、DAgger 型のデータで対処する。
9. **初版の後の実測が、上の判断を裏づけた(1章の表)。**
   - clean_driving の学習誤差は 13.0µs、検証誤差は 35.7µs で、差が約2.7倍ある。
     **データが足りず、過学習している**。事前学習と正則化が効く余地がある。
   - 今の操作値を入力から外すと、検証誤差は 36.3µs から 49.0µs に増えた(同じ25エポックで比較、+35%)。
     入力なし版の誤差の約1/4(12.7µs)を、今の操作値から得ている。
     その一部は正当な情報(操作は急には変わらない)だが、近道(まね)の分も含まれうる。開ループの誤差だけでは区別できない。
     **開ループの µs だけで選ぶと近道を使うモデルが勝つ**ので、2つを閉ループで比べることが Stage 0 の最初の実験になる。
   - 同じ first_trial モデルでゲインを上げた走行(009/010/101)は、衝突が 2.86回/周(95%信頼区間 1.77〜4.38)だった。
     006〜008 は 1.24〜1.40回/周。きれいな周は0周だった。**ゲインは学習の代わりにならない**。
     ただし AI の走行は 99秒(7.3周相当)と短く、原因がゲインか速度かは切り分けていない。
   - ToF の「TTC<0.5秒」は、21回の衝突のうち20回を中央値0.46秒前に捉えた。一方、誤警報が毎分17.5回ある。
     **0.33〜0.67秒先の ToF を当てる補助タスクは、予測できる信号がある時間幅にちょうど合っている**。
     画像・速度・ステアを見て学習すれば、ルールより誤警報を減らせるかもしれない(仮説)。

---

## 1. 前提:いまの JetFormer とデータ(実測・集計)

| 項目 | 値 | 出所 |
|---|---|---|
| モデル | ResNet-18(ImageNet)+ ToF 小CNN + IMU/操作値/ジャンルのトークン → Transformer 4層・256次元 → ジャンル別ヘッドで 10フレーム先まで | `model.py` |
| パラメータ数 | **約14.1M**(ResNet-18 部分 11.2M、それ以外 2.9M)。`model.py` の構成から計算した。「約10M」ではない | 計算 |
| 推論 | TensorRT FP16 で 1.9ms(モデルのみ)、転送込みで 5.7ms。撮像から出力まで平均 28ms(最大 35ms) | README / model_design.md |
| 学習速度(Orin Nano) | 約170〜200枚/秒。clean_driving の 1/3 間引き(学習 5,912サンプル)で 1エポック約34秒、30エポックで 1,020秒 | README / result.json |
| 生ログ | 23ログ、カメラ 80,025枚 = **44.4分** | `data/raw` を集計 |
| その内訳(車両データ 100Hz で集計) | 人が走らせている 23.5分、人の操縦で停止中 7.7分、**AI 12.3分**。1周9.5秒で割ると、人 約150周分・AI 約78周分(停止や非周回を含む概算) | 同上(VEHICLE.npz 2件は壊れていて除外) |
| 最新の学習データ | `clean_driving_20261011`: 52シーン 27,502枚(15.3分)。高速走行 10,181枚、低速走行 17,321枚 | dataset_card.json |
| 予測の誤差(検証) | first_trial: 51µs。**clean_driving(30エポック完了): 35.7µs(高速 40.7 / 低速 32.3)、学習誤差 13.0µs**。<br>**操作値の入力なし版: 49.0µs(高速 61 / 低速 41)**。電源断で26エポック目に止まり、25エポック目の最良値。同じ25エポックの入力あり版は 36.3µs | `data/runs/clean_driving_20261011-*/result.json` |
| 走行の弱点 | 衝突 1.2回/周(人は 0.75)。91% が R1・DE・GM の3か所。R1 は入口が 6200eRPM(約3.1m/s)以上だと 8回中6回当たる | ai_drive_best_practices.md |
| 新しい AI 走行(009/010/101、first_trial、ゲインを上げた設定) | AI の走行 99秒・7.3周相当で、衝突 21回 = **2.86回/周**(95%信頼区間 1.77〜4.38)。006〜008 は 1.24〜1.40回/周。<br>きれいな周 0、AIだけの周 1(10.15秒)、人の交代 15回。<br>衝突の場所: DE・DE2 11回、LH(黒カーテン)4回、GM 3回、駐車枠の板 2回、R1 1回。<br>eRPM の p90 は 7,000〜8,200(約3.5〜4.1m/s)。006〜008 は 6,000〜6,500。<br>ステアの反転 2.0〜3.3回/秒、ハンドルが限界にいる割合 11〜15%。<br>ToF の TTC<0.5秒: 衝突21回中20回を中央値0.46秒前に捉えた。誤警報は毎分17.5回 | `overnight_20261011/runs/summary.json` |
| 既知の近道 | 現在の操作値・自車速度をまねる。速度ゲインを上げても、モデルが自分の速度を見て指令を下げ、打ち消してしまう | 同上 |

ここから言えること。

- **1つのコースのラップタイムは、すでに人とほぼ同じ**。残っている課題は衝突と、まれな状態(壁へ向いた、速すぎる)。
  事前学習で特に期待するのは次の3つ。
  - (1) まれな状態・照明の変化(光のゲート、黒いカーテンのトンネル)に強い特徴
  - (2) 先の見通し(未来の ToF・ヨーレート)を内部に持つこと
  - (3) **新しいコース配置へ少ない周回で適応できること**(講習の会場ごとにコースが変わる場合に特に効く)
- 速度は eRPM 1あたり約 0.5e-3 m/s。6200eRPM は約 3.1m/s で、**遅延が 10ms 増えると約 3cm 余分に進む**。バックボーンを大きくするときは、この量で判断する。
- 学習誤差 13.0µs に対して検証誤差 35.7µs。学習サンプルは間引き後 5,912 しかなく、**分散(データ不足)が主な誤差**と見られる。
  この状態は、事前学習した特徴・補助タスク・拡張が最も効きやすい(3.2 の反証にも注意)。
- 操作値の入力を外すと誤差が +35% になる。モデルは「今の操作値」を強い手がかりにしている。
  上の表の「既知の近道」(速度ゲインを打ち消す)や、新しい走行でゲインを上げても良くならなかったことと矛盾しない。
  ただし、これは開ループの誤差なので、走りが悪くなるとは限らない(9章)。

---

## 2. 公開データとシミュレータ:小さなRCカーに使えるか

| 候補 | 中身・規模 | 車・カメラ | 使い道の判断 |
|---|---|---|---|
| **GNM の混合データ**(RECON 25h、SCAND 約9h、CoryHall 2h など。計60〜70h、6種のロボット)[1] | 画像 + 自己位置(ウェイポイント)。Open X-Embodiment / TFDS で入手可(`berkeley_gnm_cory_hall` など)[4][5] | CoryHall は**RCカー**、屋内の廊下、1.2m/s、2時間(GNM の表)[1]。ほかは Jackal・Spot・ATV など。**TFDS 版の CoryHall は画像が 64×85 画素**で、7,331件・1.39GiB しかない [4] | 画像エンコーダの補助としてなら可。行動はウェイポイントで、PWM とは違う。GNM 自身は、多様なデータを混ぜると単一データより強く、劣化にも頑健になると報告している [1]。ViNT(100時間超・8種のロボットで事前学習)は、CARLA の新しい課題で、ゼロショットの成功率 40% → 1時間未満の追加データで 80% になった [2]。後継の NoMaD は、目標なしの探索まで1つの拡散方策で扱う [3]。**屋内の白・赤の板のコースとは見た目が大きく違う**。64×85 画素では 224×224 の入力の事前学習にも粗すぎる |
| Donkey Car コミュニティ | 公式は「10〜20周のきれいな走行(5千〜2万枚)で止める」「失敗はその場で消すか `donkey tubclean` で消す」と案内している [7] | 1/16 車、低い広角カメラ | **まとまった公開データは見つからなかった**(2回検索して確認。Hugging Face・Kaggle にステア・スロットル付きの tub はなく、Roboflow に物体検出用の 692枚などがあるだけ [84])。各自の tub(データ一式)がばらばらにあるだけ。数値の目安としてだけ使う |
| Donkey シミュレータ(gym-donkeycar、Unity)[8] | コースを自動生成できる。実車のデジタルツインの研究がある [9]。実機の Donkey 型の車で、**世界モデル型の強化学習(Dreamer)が5分未満の経験で全コースを走れるようになった**報告がある [10] | 小型車 | 事前学習のデータ源にはなる。ただし ToF・IMU・ESC の特性が実機と違う。「仮想と実機の試験の差は大きい」という報告がある [9]。[10] は、小さな車でも世界モデル(3.1 の潜在の次状態予測)が少ないデータで役立つ例でもある |
| AWS DeepRacer | シミュレータでの強化学習が前提(グレースケールの単眼カメラ1枚)[11] | 1/18 車 | **AWS のサービスは 2025-12-15 に終わり**、自分のアカウントで動かすオープンソース版に移った [12]。sim-to-real の知見(RandConv で見た目をランダム化)[14] は参考になる |
| F1TENTH / RoboRacer | ほぼ 2D LiDAR が前提(TinyLidarNet など)[16]。RL で人に勝った報告もある [15] | 1/10 車 | **カメラのデータはほとんどない**。手法(ドメインランダム化・アクチュエータのモデル化)だけ参考にする |
| Duckietown のログ | 実機ログとシミュレータを混ぜた模倣学習の例がある [17] | 小型車、カメラ高さが近い | 環境が「テープの車線」で違う。優先度は低い |
| nuScenes / Waymo / OpenDV-YouTube(1,747時間)/ comma / **Yaak L2D**(教習車60台・独30都市・3年分。ステア・アクセル・ブレーキ付き。LeRobot 形式 [83]) | 規模は大きい [18][83]。YouTube 動画での方策の事前学習(ACO・PPGeo)は実車の運転で効いている [19][20] | 乗用車、カメラ高さ 約1.5m級、公道 | **方策には使わない**。視点の高さ・速度・交通の規則・FOV が違う。画像特徴としての価値は、DINOv2 などの基盤モデルで十分に代わりがきく。OpenDV は YouTube の利用条件にも注意がいる [18]。L2D は LeRobot 形式で読み込みやすいが、10Hz・乗用車・公道で、ずれは同じ。ライセンスは配布ページで確認すること(ブログには記載なし)|
| CARLA | 乗用車の都市シミュレータ | — | RCカーのコースには合わない |
| **自前コースの 3D ガウシアン・スプラッティング(3DGS)複製** | 1本の動画からシミュレータを作り、ナビゲーション方策を学習。実世界で成功率が上がった(Vid2Sim)[72]。3DGS の中での大規模 RL で衝突率が 1/3 になった(RAD)[73] | 自分の車・コース | **中長期の有力候補**。新しい視点の画像を作る拡張と、強化学習の環境の両方に使える。作るのに手間がかかり、ToF・IMU を真似る部分も要る |

まとめ。

- 公開データで「すぐ方策が良くなる」ものは、小型の屋内コースには**ほぼない**。
- 価値があるのは次の3つ。
  - (a) 基盤モデルの重み(DINOv2 は約1.4億枚、DINOv3 は約17億枚の画像で事前学習)[39][41]
  - (b) 多様なロボットのナビゲーションデータ(画像エンコーダの補助)
  - (c) 自分のコースの複製シミュレータ(後の段階)
- 公開データを混ぜるときの注意。
  - 線形プローブ(9章)で効いているかを確かめる。
  - 割合は小さく(10〜20%以下)始める。
  - 「自前データだけ」のモデルと必ず比べる。

---

## 3. 自前の「ラベルなし」データでの自己教師あり事前学習

### 3.1 何が使えるか

自前データは、カメラ・ToF・IMU・eRPM・操作値(人またはAI)が**時刻を揃えて**すべて記録されている。
これは公開の動画にない強み。人が操縦していない区間も、AIの出力と車の動きが残っているので、自己教師あり学習の材料になる。
対象は、AI走行 12.3分、停止中、衝突の前後、リカバリー。

| 目的 | 学習するもの | 根拠 | 自分たちへの効き方(仮説) | 費用・注意 |
|---|---|---|---|---|
| **マスク画像の復元(MAE / VideoMAE)** | 画像の一部を隠し、元に戻す | MAE [24]、VideoMAE [25]、運動制御向けの MVP [27]。エゴセントリック動画で学んだ VC-1 は、**分野に合わせた追加学習で大きく伸びた** [29] | 自分のコースの見た目に強くなる | ViT が前提。クラウドで数GPU時間 |
| **基盤モデルの特徴の蒸留** | 自前の画像で、DINOv2-B などの教師と同じ特徴を出す小さな生徒を学習 | AM-RADIO(複数の教師を1つに)[43]、Theia(CLIP・DINOv2・ViT を蒸留し、ロボット学習で教師より良い)[44] | **走行時の計算はそのままで**、基盤モデルの特徴が得られる | 教師の計算をキャッシュするか、学習中に毎回計算する |
| **画像 → ToF(今)** | 画像から 42×54 の距離を当てる(無効ゾーンはマスク) | 複数のモダリティを隠して復元する MultiMAE [26] | 画像に奥行きの感覚を持たせる。ToF が欠けたときにも頑健になる | ToF の解像度は低いが、無料で手に入るラベル |
| **未来の ToF の予測** | 画像・IMU・操作から +0.33秒/+0.67秒先の ToF を当てる | 過去の画像から未来の点群を当てる ViDAR は、計画の衝突率を約15%下げた [21] | **R1 の「入口の速さ」問題に直結**。今の ToF 予測器(TTC)を学習で内部に持たせる形 | 安い。ToF のフレーム落ちに注意 |
| **逆ダイナミクス(IDM)** | (o_t, o_t+k) から、その間の操作・ヨーレート・速度変化を当てる | 逆ダイナミクスは「スクラッチより一貫して良い唯一の目的」だった。行動のクローン(BC)は文脈が観測できないと崩れる [33]。VPT・ACO は IDM で動画に疑似ラベルを付けた [34][19] | 操作に関係する特徴(コースの曲がり具合など)を学ぶ。AIの区間も使える | ESC の「停止→後退→中立→後退」は PWM と動きの対応が崩れる。**目標は PWM でなく、ヨーレートと eRPM の変化**にする |
| **潜在空間での次状態予測(JEPA / LAW 型)** | z_t と操作列から z_t+k を当てる。画像そのものは作らない | LAW は E2E 運転を自己教師で強化した(nuScenes・NAVSIM・CARLA)[22]。V-JEPA 2-AC は約62時間のロボット動画で行動条件付きの世界モデルを作った [35]。DINO-WM は DINOv2 の特徴の上で世界モデルを作った [36] | **Offline RL・モデル予測の土台**(11章)になる | 崩壊(すべて同じ出力)を防ぐ工夫が要る(EMA の教師、分散の正則化) |
| 次フレームの画像生成(拡散モデル)| 動画そのものを予測する | GenAD [18]、comma の世界モデルで方策を学習 [23] | 将来の「学習済みシミュレータ」の候補 | **重い**。今の規模ではやらない |

### 3.2 正直な反証:事前学習は常に効くわけではない

- 「データ拡張 + 浅い CNN をスクラッチで学習」が、凍結した事前学習の表現(MVP [27]・R3M [28] など)と**同じくらい強い**という報告がある。
  原因は事前学習データと制御の課題とのずれで、微調整すると差が縮む [30]。
- 運転の分野では逆向きの報告もある。CARLA で、DINO の自己蒸留で事前学習した画像エンコーダは、ImageNet の分類で事前学習したものより効率が良かった [45]。
- 事前学習の効果は、データの「量」より「画像の分布」で決まる。ImageNet も意外に強い [31]。
- 汎用の表現で、すべての課題に勝つものはない [29]。
- 分布の外での頑健さは、ViT が「物体を切り分けられるか」で予測できる。ImageNet の精度では予測できない [32]。
  照明が変わる区間(光のゲート、トンネル)がある自分たちのコースでは、ここを測る価値がある。

→ どの段階でも、**「スクラッチ + 拡張」と「ImageNet の ResNet-18(今のもの)」を必ずベースラインに置く**。
自己教師あり学習を足して走行の指標が良くならなければ、採用しない。

### 3.3 推奨する組み合わせ(Stage 2 で使う)

1つのエンコーダ(画像)と ToF・IMU のエンコーダに、補助ヘッドを付けて同時に学習する。損失は次の和にする。

```
L = L_feat(DINOv2教師の特徴の蒸留 または MAE)
  + λ1·L_tof_now(画像→ToF)
  + λ2·L_tof_future(+10/+20フレームのToF)
  + λ3·L_idm(ヨーレート・ΔeRPM・ステア)
  + λ4·L_latent(潜在の次状態)
```

- 補助ヘッドは学習のときだけ使う。`export_onnx` では今の出力 `actions` だけを出す。
- データは衝突やリカバリーも含めて**全部**使う。行動のクローン(BC)の教師にするかどうかとは分けて考える。
- 学習・検証の分割はシーン単位(今の splits.json と同じ)。**日付・コース配置ごとに分けた検証**も作る。

---

## 4. 画像バックボーン:Orin Nano で 30fps に収まるもの

| バックボーン | パラメータ | 計算量(224px)| Orin Nano の実測(文献) | ライセンス・注意 |
|---|---|---|---|---|
| ResNet-18(今) | 11.2M | 約1.8 GMACs | JetFormer 全体で 1.9ms(自前、TRT 10.16 FP16) | 基準 |
| ResNet-50 | 25.6M | 約4.1 GMACs | 1.86ms(FP16、GPU のみ)[46] | — |
| EfficientNet-B0 | 5.3M | 約0.39 GMACs | 2.11ms [46] | — |
| MobileNetV4 [50] | (型の記載なし) | — | 1.43ms [46] | — |
| **ViT-S/16**(DINOv3 ViT-S/16 と同じ形) | 21M [41] | 約4.6 GMACs | **2.36ms(FP16)**。FP32 で作り直すと 5.92ms [46]。**FP16 で精度が 4〜7ポイント落ちた** [46] | DINOv3 は独自ライセンス(DINOv3 License)[41] |
| **DINOv2 ViT-S/14(+レジスタ)** | 21M | 約6 GMACs(256トークン、推定) | 未計測。ViT-S/16 から見て **3〜4ms と推定** | Apache-2.0。レジスタ付きは特徴マップのノイズが少ない [39][40] |
| DINOv2 ViT-B/14 | 86M | 約23 GMACs | Orin Nano Super で「126」(単位・条件の記載なし。FPS なら約8ms)[47] | **教師として使う**。走行用には重い |
| DINOv3 ConvNeXt-Tiny | 29M [41] | 約4.5 GMACs | 未計測 | CNN なので FP16 の扱いが楽そう(推定) |
| EfficientViT-B1 | 9.1M | 約0.52 GMACs | Orin Nano での分類の値は見つからず。Jetson AI Lab の手順は EfficientViT-L2-SAM 用で、JetPack 5/6 が前提 [42] | 軽い。基盤モデルの特徴ではない |
| Theia-T / Theia-S(DeiT-T/S 基盤) | 約6M / 22M(DeiT-T/S の値。論文の表は Theia-B 86M のみ) | 約1.3 / 4.6 GMACs | 未計測 | CLIP・DINOv2・ViT の蒸留。CortexBench で Theia-B 79.8 点、MVP 77.4、R3M 76.5、VC-1 69.6 [44] |
| E-RADIO | — | — | 「CLIP・DINOv2 より 6〜10倍速い」(作者の主張)[43] | NVIDIA のライセンスを確認する |
| DINOv2-L / DINOv3-L | 300M | — | 224px で 27ms / 44ms(分割・混合精度・FP32)[46] | 走行には使えない |

文献の測定条件: JetPack 6.2・TensorRT 10.3・MAXN_SUPER・GPU の計算時間だけ(前処理を除く)[46]。
自分の環境は JetPack 7.2・TensorRT 10.16 なので、**同じ値になる保証はない**。

### 4.1 遅延の予算

| 項目 | 今 | 上限の目安 |
|---|---|---|
| モデル推論(TRT FP16) | 1.9ms | **6ms 以下**(+4ms = 3.1m/s で約1.2cm) |
| 転送込みの推論 | 5.7ms | 10ms 以下 |
| 撮像から出力まで | 平均 28ms、最大 35ms | 平均 33ms 以下(1フレーム以内) |

### 4.2 FP16 の注意

- 実測の例 [46]: すべて FP16 で作った ViT-S/16 のエンジンは、224px でサーバーの FP32 より正解率が 5.56ポイント低かった(30シードの平均)。
  CNN の差は 0.1ポイント以内だった。同じモデルを FP32 で作り直すと、サーバーとの一致は 99.95% に戻った(ただし 2.36ms → 5.92ms)。
  論文は「**壊れた FP16 エンジンのほうが速く動くことがある**」と注意している。速さだけで選ばず、配備したエンジンで精度を確かめる。
- ViT は LayerNorm の分散の計算で FP16 があふれやすい(FP16 の最大値は 65,504)[49]。
- 対策は3つ。
  - 該当する層を FP32 に固定する(TensorRT の層精度)。融合で設定が効かないことがある [49]。
  - BF16 で学習して、値の範囲を小さく保つ。
  - CNN(ConvNeXt・ResNet)を選ぶ。
- 今の JetFormer は、TRT と PyTorch(FP32)の差が最大 0.0007 だった。**新しいバックボーンでも同じ確認(差が 0.005 以下 ≒ 2.5µs 以下)を合格の条件にする**。

### 4.3 推奨

- **A案(早い)**: DINOv2 ViT-S/14-reg を凍結して使う。
  - 16×16 のパッチを 8×8 に平均プーリングし、64トークンにして融合に入れる。
  - 融合とヘッドだけを学習する(線形プローブ)。その後、最後の2〜4ブロックを小さい学習率(1e-5)で解凍する(LP-FT)[58]。
- **B案(遅延ゼロ)**: DINOv2-B(教師)の特徴を ResNet-18(生徒)へ蒸留する。
  - 自前の全フレームを使い、クラウドで行う。
  - 今の推論グラフはそのまま使える。
- 両方を作って、**走行の指標で決める**。

---

## 5. 複数コース・複数ジャンルの条件付け

今のモデルには、ジャンル ID のトークンとジャンル別ヘッド(CIL の分岐)がある。
これは条件付き模倣学習の定番で、まだ必要 [51]。そのうえで次を足す。

| 追加するもの | 方法 | 理由・根拠 |
|---|---|---|
| ジャンルの FiLM | ジャンルの埋め込みから、融合 Transformer の各層の LayerNorm の倍率とずれを作る | RT-1 は指示で画像エンコーダを FiLM で条件付けた [52][53]。トークン1個より効きが強い |
| **連続値の「ペース」** | 周ごとのラップタイムを、その配置での順位(0〜1)にして入力する。走行時はスライダーで指定する | 速度ゲインはモデルに打ち消される(1章)。**モデル自身に速く走る指示を学ばせる**。目標の成績を条件にする手法(Decision Transformer・RvS)と同じ考え方で、Offline RL への橋になる [80][81] |
| コース ID は**入れない**(または 50% で落とす) | 配置に依存しない方策を学ぶ。配置ごとの違いはファインチューニング(LoRA)で吸収する | 新しい配置で動かない ID に頼らせない。CrossFormer・Octo は、ロボットごとのトークン・頭で多様なデータをまとめている [6][54] |
| 車に依存しない補助出力 | 未来のヨーレート(gyro z)と速度(eRPM)を、操作値と一緒に予測する(学習のときだけ) | GNM は行動をロボットの速度で正規化し、別のロボットへ移した [1]。サーボ・ESC の個体差やトリムに強くなる |
| 行動のまとまりの出力は今のまま | 10フレーム先までの操作列を一度に出す形を、事前学習でも保つ | ACT・Diffusion Policy の「行動のまとまり(action chunk)」と同じ考えで、模倣学習で効果が確かめられている [75][76] |
| 操作値の入力のドロップアウト | 学習のとき、今の操作値・速度を 30〜50% の確率で 0 にする | copycat・ego status の近道を弱める [63][64]。すでに `--no-command-input` があるので、その中間を作る |
| 駐車(ジャンル1〜3)| 共通のエンコーダに、新しいヘッドと LoRA を足して少ないデモで学ぶ | TAIL は LoRA が、全体の 1% のパラメータで最良になり、前の課題を忘れにくいと報告している [56] |

---

## 6. ファインチューニングの方法(コース別)

### 6.1 選択肢

| 方法 | 学習するパラメータ | 長所 | 短所・根拠 |
|---|---|---|---|
| 頭だけ(線形プローブ)| ジャンル別ヘッドと trunk(約0.1M) | 速い。忘れない | 見た目が大きく変わると足りない |
| **LoRA(r=8〜16)** | 融合 Transformer の注意・MLP に低ランクの差分(約50〜100k) | **1% 程度のパラメータで全体の微調整に並ぶか上回り、忘れにくい** [55][56]。ONNX の前に統合できるので TRT の速度は同じ | ランクと対象の層は調整がいる |
| FiLM・LayerNorm・バイアスだけ | 数千〜数万 | とても小さい | 効果は限られる |
| 一部の層だけ解凍(surgical)| 例: 画像エンコーダの前段 | 画像の崩れのような変化には、**前の層だけ調整するのが最良**だった [57] | どの層を解凍するかは変化の種類で変わる |
| 全体の微調整 | 全部 | 同じ分布では最も伸びる | 少ないデータで事前学習の特徴が崩れ、**分布の外で弱くなる**。線形プローブ → 全体(LP-FT)が対策 [58] |

### 6.2 推奨の手順(1コースあたり)

1. 事前学習済みの方策(Stage 3)をそのまま新しい配置で走らせ、**ゼロショットの指標**を記録する。
2. 新しい配置で、きれいな周を 20〜40周(約3〜6分)と、当たりやすい場所ごとの立て直しを 10〜15回集める。
3. 頭だけを学習する → LoRA(融合、r=8)を足す → 照明・床が大きく違うなら、画像エンコーダの前段の LayerNorm や最初のステージを解凍する。
   - Jetson 上で学習する。エンコーダを凍結すれば、メモリは今の学習(約3GB)より少なく済む見込み。
4. AI 走行を 1〜2回(各4分以上、収集 ON)行う。
   - プロポの CH3 で人が取り返した区間を、HG-DAgger として足す [60][61]。
   - 介入の直前・直後のサンプルに重みを付ける(Sirius の重み付き BC)[62]。
5. コースごとの差分(LoRA と頭)をファイルで保存する。走行用のエンジンは、統合してから作る。

---

## 7. データ量の目安

| 出所 | 目安 |
|---|---|
| Donkey Car 公式 | 1コースで、きれいな走行 10〜20周(5千〜2万枚)[7] |
| 自分たち | first_trial(12,648枚、約7分)のモデルで、ラップタイムは人と同じくらいになった。**衝突は人の 1.6倍**(ai_run_003〜008)。ゲインを上げた 009/010/101 では 2.86回/周。clean_driving(27,502枚、約15分)は学習を終えた(検証 35.7µs、学習 13.0µs)。走りはまだ評価していない |
| 模倣学習のスケーリング則 [59] | 汎化は**環境と物体の組の数**に対してべき乗則で伸びる。1つの環境で増やしても頭打ちになる。推奨は「32組 × 各50デモ」(1,600デモ)。8/16/32組では、それぞれ 400/800/1,600デモで頭打ちになった(マニピュレーション。成功率は約90%)。4人で1日の午後に集められる量 |
| GNM / ViNT | 事前学習は GNM が 60時間・6種のロボット、ViNT が 100時間超・8種 [1][2]。ViNT は CARLA(シミュレータ)の新しい課題で、1時間未満の追加データで成功率 40% → 80% になった(論文の表3で確認)[2] |
| ACO | YouTube 120時間超から、1秒に1枚で 130万枚 [19] |

自分たちへの当てはめ(**仮説**。1周 = 約285フレーム)。

| 段階 | 集める量 | フレーム数 |
|---|---|---|
| 今のコースの改善 | 指針どおり、R1・DE をきれいに抜ける周を約30周と、立て直しを各10〜15回 | 約1万 |
| 複数コースの事前学習・1期 | 8配置 × 25周 + 立て直し。うち2配置は評価用に学習から外す | 約6万(約35〜45分) |
| 複数コースの事前学習・2期 | 16配置。左回りと右回りの両方を数えると、板の並べ替え8回で16配置になる | 約12万 |
| 自己教師あり学習 | 上のすべてと、AI走行・停止・衝突。日付・照明・電池の状態を散らす | 目標 30万〜100万(3〜10時間) |
| コース別ファインチューニング | 20〜40周と立て直し | 6千〜1.5万 |

**注意**

- 30fps の連続フレームはほとんど同じなので、「枚数」より「周回の数・状態の種類」で数える。今の 1/3 の間引きは妥当。
- 衝突の1秒前から2秒後は、行動のクローン(BC)には使わない(今の `mask_impacts`)。自己教師あり学習には使う。

---

## 8. データ拡張と sim-to-real

| 拡張 | 方法 | 注意 |
|---|---|---|
| 明るさ・ガンマ・色・ノイズ・JPEG・ぼかし | 今の color jitter を広げる。動きのぼかし・露出の変化(トンネル・ゲート)を足す | 照明の変化への頑健さを、明るさ別の検証で測る |
| 小さなランダムシフト(±4px 程度) | DrQ-v2 などで効果が大きい [71] | 大きく動かすとステアの意味が変わる |
| **拡張した画像で出力を揃える正則化(I-RAS 型)** | 同じフレームの元の画像と拡張した画像(明るさ・ぼかし・小さなシフト)で、出力する操作列の差を罰する。隣り合うフレームの出力の差も罰する | DeepRacer の小さな車で、カクカクしたステアを減らすために使われた。この手法の車は 2022 年の DeepRacer 決勝で優勝した [14]。新しい走行では、ステアの反転が 2.0〜3.3回/秒、ハンドルが限界にいる割合が 11〜15% だった(1章)。**学習のときだけの損失なので、走行時の計算は増えない**。強くかけすぎると、急なカーブで切り遅れる。[14] は速度と報酬に応じて重みを下げている |
| **ヨー回転の視点合成と、ステアの補正ラベル** | 純粋な回転なら、画像の変換はホモグラフィ K·R·K⁻¹ で**奥行きなしに正確**に作れる。ToF もゾーンの列をずらして合わせる。ラベルは「2秒で元の線に戻る」ように補正する | DAVE-2 は左右のカメラと視点変換で、ずれから戻る例を作った [68]。ChauffeurNet は走りの軌跡を乱して悪い例を合成した [69]。**DE(緑マットの出口で約15度右向き)の弱点に直結**。補正ラベルの作り方が間違うと逆効果なので、人の立て直しのデータと比べて調整する |
| 横の平行移動の視点合成 | 奥行きが必要。ToF(54×42)は粗いので、単眼の奥行き推定(Depth Anything V2 [74])で補う | 欠けた部分の埋め方で偽物っぽくなる。3DGS の複製が根本的な対策 [72] |
| 左右反転 | 自己教師あり学習だけに使う。ステア・gyro z・accel x の符号も反転する | **左回りのコースの BC には使わない**(右回りの配置を集めるほうがよい) |
| センサーの欠落・ノイズ | ToF のフレーム落ち、IMU のバイアス、操作値の入力のドロップアウト | 走行で起きる「データなし」に強くする |
| 時間のずれ | 入力と出力の対応を ±1フレームずらす | 遅延のばらつき(28〜35ms)に強くする |
| シミュレータのドメインランダム化 | 見た目・摩擦・遅延をランダムにする [70]。RandConv [14] | ランダム化すると**車が遅くなる**傾向がある(完走率と引き換え)[13]。最終の方策は実機データで仕上げる |

---

## 9. 評価のしかた

### 9.1 オフライン(学習の後すぐ)

1. **予測誤差(µs)**: 今の `tools/evaluate.py`。教師あり入力と自己入力の2条件で、1〜10フレーム先を見る。
2. **介入区間の検証セット(新設)**
   - AI 走行(ai_run_003〜010・101 など)の中から、人が CH3 で取り返した区間と、人の立て直し区間を集める。
   - Codevilla らは「前向きの1台のカメラの、上手な運転だけのデータ」では予測誤差と走りの相関がとても低いと示した。
     一方で、横のカメラ(=軌道から外れた状態)での誤差は、走りとの相関が高かった [65]。
   - 自分たちの「取り返した区間」は、この「軌道から外れた状態 + 専門家のラベル」にあたる。
3. **場所別の誤差**: R1・DE・GM の手前1.5秒の誤差と、R1 の入口の予測速度(eRPM 換算)。
4. **線形プローブ**(事前学習の良し悪しを安く測る)
   - 凍結した特徴から、+3フレーム先のステア、TTC<0.5秒(ToF から作る)、コース上の位置(`laps.py` の周回の進み具合)を当てる。
   - 照明が変わる区間(ゲート・トンネル)だけでも測る [32]。
5. **ペースの効き**: ペースの入力を 0.2→0.8 に変えたとき、予測するスロットルと eRPM が単調に増えるか。
6. **不確かさの指標**: 種(シード)を変えた 3〜5個のモデル、または MC ドロップアウトで、出力のばらつきを測る。
   Aich らは、こうした「モデルの不確かさ」に基づくオフライン指標が、従来のオフライン指標より閉ループとの相関を 13% 以上高めたと報告している。実車ではさらに差が大きかった [66]。
   自分たちでは、R1・DE・GM の手前での不確かさが、衝突の起きやすさと関係するかを調べる。
7. **近道の検出**: 操作値の入力あり/なしの組(35.7µs 対 49.0µs、1章)を、開ループと閉ループの両方で比べる。
   開ループで勝ったほうが閉ループで負けるなら、開ループの µs の指標は近道に引きずられている。その場合は、自己入力の評価と、介入区間の検証セットを主に使う。

### 9.2 閉ループ(実車、主な判断基準)

指針(ai_drive_best_practices.md)の指標と手順をそのまま使う。

- 指標: AIだけの周のラップ中央値、衝突/周(水平 2.5G 以上)、交代/周、スタック/10周、ハンドルが限界にいる割合。
- 1回の試行は AI 側で4分以上(約24周)。これで衝突の3倍の差が見える。2倍の差には約43周が要る。
- A→B→A の順で走る。

**事前学習の評価で追加するもの**

- 学習から外した配置での**適応曲線**: 0/5/10/20周の微調整データで、それぞれ AI 走行を 4分行う。
  完走できた周の割合、衝突/周、ラップ中央値を比べる。
  比較の相手は「事前学習なし(ImageNet の ResNet-18 + その配置のデータだけ)」。
- 照明の条件(ゲートの ON/OFF、昼と夜)を変えた走行。

**注意**: NAVSIM のような「反応しないシミュレーション」の指標は、実車の運転では開ループの変位誤差より走りと相関が高い [67]。
自分たちでも、ToF からの TTC を使った「予測した軌道の衝突の危険度」を、オフラインの指標として試す価値がある。

---

## 10. 推奨する段階的計画

```
Stage 0 評価の土台 ──► Stage 1 基盤モデルの特徴 ──► Stage 2 自前データの自己教師あり学習
                                                            │
       Stage 5 Offline RL(別調査)◄── Stage 4 コース別 FT ◄── Stage 3 複数コース・ジャンル・ペースの方策事前学習
```

### Stage 0:評価の土台(1〜2日、Jetson のみ、費用なし)

- **データ**
  - 既存のログから「介入区間の検証セット」を作る(9.1)。
  - 今後の収集に、**コース配置 ID・照明の条件・電池の電圧・日付**を記録する欄を足す(後の分割と分析に必須)。
- **モデル**: 変更しない。ベースラインを固定する。
  - (a) 今の v0(clean_driving、35.7µs。**学習済み**)
  - (a') v0 で操作値の入力なし(49.0µs、25エポック。**学習済み**だが電源断で途中まで)
  - (b) v0 + 操作値のドロップアウト 0.3(a と a' の中間)
  - (c) スクラッチ + 拡張
  - 最初の実験は、学習済みの (a) と (a') の閉ループ比較。ゲインは 1.0 に戻して固定する。
    009/010/101 のようにゲインを上げた状態では、衝突が多すぎて比べられない(2.86回/周)。
- **合格の基準**: 閉ループの評価の手順(9.2)が、1回のバッテリーで回せること。

### Stage 1:基盤モデルの特徴を入れる(1〜2週間)

- **データ**: 今のもの(clean_driving_20261011 など)だけ。
- **モデル**
  - A案: DINOv2 ViT-S/14-reg を凍結し、LP-FT で学習する。
  - B案: DINOv2-B の特徴を ResNet-18 へ蒸留する。
  - Orin Nano で TRT 10.16 のベンチマークを取る(ResNet-18 / DINOv2-S / DINOv3-S / ConvNeXt-T / EfficientViT-B1)。
- **計算**
  - A案の線形プローブの段階は、エンコーダを凍結するので Jetson でもできる。
  - B案の蒸留はクラウド(数GPU時間)。
- **期待(仮説)**: 照明の変化と、軌道から外れた状態での予測の改善。開ループの µs 誤差の改善は小さいかもしれない(1章で、すでに約40µs)。
- **合格の基準**(すべて満たす)
  - モデルの推論が 6ms 以下
  - FP16 と FP32 の差が 0.005 以下
  - 介入区間の検証セットで誤差が 10% 以上下がる
  - 閉ループの衝突/周が悪くならない(各24周以上)
- **リスク**: ViT の FP16 であふれる。Jetson で解凍して学習するとメモリ不足になる(→ 凍結 + 特徴のキャッシュで回避)。

### Stage 2:自前データの自己教師あり事前学習(2〜4週間)

- **データ**: すべてのログ。今は 44分。目標は 3〜10時間。
  - 普段の収集を毎回 ON にし、AI 走行も必ず記録する。
  - 日付・照明・電池の状態を散らす。
- **モデル**: 3.3 の多目的の損失。補助ヘッドは学習のときだけ使う。
- **計算**: クラウド(A100/H100 1枚で 1回 数時間、5〜30ドル程度)。12章を参照。
  データは今の tar エクスポートか、Hugging Face(非公開)で渡す。
- **期待(仮説)**
  - 文献: PPGeo は、データがとても少ないとき 2%〜100%以上の改善 [20]。ViDAR は計画の衝突率を約15%下げた [21]。
  - 自分たちの目標: Stage 1 に比べて、R1 の入口の予測速度が人に近づき、R1 の衝突が減ること。
- **合格の基準**
  - 線形プローブ(TTC・ステア)が Stage 1 より良い
  - 閉ループで衝突/周が 20% 以上下がる(43周以上で比べる)
- **リスク**
  - 自分たちのデータは1つのコースに偏っていて、表現も偏る。→ Stage 3 のデータと一緒に回す。
  - スクラッチ + 拡張と差が出ない可能性 [30]。そのときは採用しない。

### Stage 3:複数コース・ジャンル・ペースの方策事前学習(1〜3か月、収集が主)

- **データ**
  - 白・赤の板を並べ替えて配置を作る。1期は 8配置、2期は 16配置。
  - 1配置あたり約25周のきれいな周 + 立て直し(当たりやすい場所あたり 5〜10回)。
  - 左回り・右回り、芝・PTFE シートの有無、ゲート・トンネルの有無を散らす。
  - **評価用に2配置を学習から外す**。
  - ジャンルは高速・低速の両方で集め、周ごとのペースを記録する。
- **モデル**: FiLM、ペースの入力、コース ID なし、車に依存しない補助出力、操作値のドロップアウト(5章)。
  画像エンコーダは Stage 2 のものから始める。
- **計算**: クラウド(1回 1〜4 GPU時間)。
- **期待(仮説)**: 外した配置で、ゼロショットで一定の割合を完走する。少ない周の微調整で、今のコースと同じ水準に届く。
  - ViNT は、ゼロショット 40% → 1時間未満の微調整で 80% だった(CARLA の課題)[2]。
  - スケーリング則は、配置の数に対してべき乗則 [59]。
- **合格の基準**: 外した配置で「10周の微調整」のモデルが、「その配置のデータだけ(40周)でスクラッチ学習」したモデルと比べて、衝突/周が同じかそれより少ない。
- **リスク**
  - 板の並べ替えに手間がかかる(配置図と写真を残し、再現できるようにする)。
  - 配置が似すぎると多様性が足りない。

### Stage 4:コース別ファインチューニング(各コース 1日)

- 6.2 の手順を使う。Jetson 上で、凍結したエンコーダ + LoRA なら 7〜20分の学習(推定、12章)。
- **合格の基準(今のコース)**
  - 衝突/周 ≤ 0.75(人と同じ)
  - AIだけの周のラップ中央値 ≤ 人の中央値(9.8秒)
  - 交代/周 → 0 に近づく
- **リスク**: 少ないデータで過学習する(→ 検証は別の日・別の電池の状態の走行で行う)。人の衝突をまねる(→ 衝突の前後を除外する)。

---

## 11. Offline RL へのつなぎ方(次の段階のための設計の約束)

ファインチューニングの後に Offline RL を足すとき、ここまでの設計で次のものが揃う。

1. **凍結した(またはゆっくり更新する)エンコーダ**: 強化学習の価値関数と方策が、同じ特徴を使える。
   視覚の強化学習では、拡張(ランダムシフト)が効く [71]。
2. **報酬の材料は、すでにログにある**
   - 周回の進み具合(`laps.py`)
   - 衝突(IMU 2.5G、ToF の TTC)
   - 人の交代(SELECT)、リカバリー(AI_CMD の assist)
   - ハンドルの張り付き
3. **行動の記録**: AI_CMD の `model_out`(補助の前の出力)、`out`、`assist` が残っている。データを集めた方策と、実際に出した値を区別できる。
4. **世界モデル**: Stage 2 の潜在空間での次状態予測を、モデルベースの手法の土台にできる。
   - 例: Dreamer・TD-MPC 系 [37][38]、DINO-WM 型の計画 [36]
5. **手法の候補**
   - データの中の行動だけで学ぶ IQL [77]
   - 保守的な CQL [78]
   - 後で実機で追加学習するなら Cal-QL [79]
   - いちばん簡単な第一歩は、**ペース・報酬で条件付けた BC(RvS)**[81]。Stage 3 のペースの入力がそのまま使える。
6. 長期的には、3DGS でコースを複製し、その中で強化学習する。RAD は模倣学習を正則化として使い、衝突率を 1/3 にした [73]。

---

## 12. 計算量とコストの見積もり(推定)

前提

- Orin Nano Super の性能は、INT8 で 67 TOPS(疎行列の値)、メモリ帯域 102GB/s [48]。
- 学習の計算量は、順伝播の約3倍。
- Jetson の実効性能は、今の学習の実測から **約2.2〜2.5 TFLOPS** とした。
  clean_driving は 5,912サンプル × 30エポック = 17.7万サンプルを 1,020秒で学習した(約174サンプル/秒 × 約12.6 GFLOP ≒ 2.2 TFLOPS)。
  下の表の Jetson の時間は 2.5 TFLOPS で計算したので、実際は約1.15倍長くなる。
- クラウドの実効性能は、A100 で約100 TFLOPS、H100 で約250 TFLOPS と仮定した(混合精度、利用率は約30%)。
- データの読み込み・JPEG の展開が律速すると、時間は 2〜3倍になる。

| 作業 | 1サンプルの学習計算量 | サンプル数 | Jetson | クラウド 1枚 |
|---|---|---|---|---|
| 今の BC(27.5k枚、1/3、30エポック)| 約12.6 GFLOP | 17.7万(実測。衝突の前後を除いた後) | **17分(実測 1,020秒)** | 1分未満 |
| MAE ViT-S(100万枚 × 50エポック) | 約11 GFLOP | 5,000万 | 約2.5日 → **不可**(メモリも足りない) | 約0.6〜1.5時間(+ 読み込み) |
| DINOv2-B → ResNet-18 の蒸留(教師を毎回計算、100万枚 × 30エポック)| 約57 GFLOP | 3,000万 | 約8日 → 不可 | 約2〜5時間 |
| 複数コースの BC(ViT-S/14 バックボーン、50万枚 × 30エポック) | 約38 GFLOP | 1,500万 | 約2.6日 → 不可 | 約0.6〜1.6時間 |
| コース別の微調整(エンコーダ凍結 + LoRA、3万枚の 1/3 × 20エポック) | 約5 GFLOP(ResNet-18級)〜約14 GFLOP(ViT-S/14) | 約20万 | **約7〜18分**(凍結した特徴をキャッシュすれば、さらに短い) | — |

- クラウドの価格の目安(2026年、二次情報、変動あり): A100 は 1時間あたり約2〜3ドル、H100 は約3〜4ドル [82]。
  事前学習1回は 5〜30ドル、実験一式(10〜20回)は 100〜500ドル程度。
  - 参考: Theia は H100 で約150 GPU時間かけて、ImageNet 120万枚で蒸留した [44]。
- データ量: 実測(ログ 20261010-200916)で、JPEG は1枚約9KB、ToF・IMU・車両を含めて1フレーム約13KB。100万枚で約13GB。

---

## 13. リスクと未確認のこと

| リスク | 影響 | 対策 |
|---|---|---|
| 公開データの見た目のずれ(カメラ高さ 約10倍、屋外、速度)| 負の転移 | 方策には使わない。エンコーダの補助としても、線形プローブで確認してから混ぜる |
| ViT の FP16 であふれる [46][49] | 走行中の出力がおかしくなる | FP32 との差を確認する。LayerNorm を FP32 にするか、CNN 系を選ぶ |
| 遅延が増える | 3.1m/s では 10ms で 3cm | 推論 6ms 以下を条件にする。蒸留(B案)なら増えない |
| Jetson のメモリ(8GB、スワップなし) | 大きなバックボーンの学習で本体ごと再起動する | Jetson ではエンコーダを凍結し、特徴をキャッシュする。事前学習はクラウドで行う |
| Jetson の電源断(**実際に起きた**: 2026-10-11 未明、操作値の入力なし版の学習が26エポック目で止まった) | 長い学習が途中で失われる | 1エポックごとにチェックポイントを保存し、そこから再開できるようにする。数時間を超える学習はクラウドで行う |
| 近道(操作値・速度のまね)[63][64] | 開ループの誤差は小さいのに走りが悪い | 入力のドロップアウト、自己入力での評価、閉ループの評価 |
| 少ないデータの微調整で特徴が崩れる [58] | 新しい照明・配置に弱くなる | 線形プローブ → LoRA → 一部の解凍、の順。全体の微調整は比べるためだけに使う |
| ランダム化で遅くなる [13] | ラップタイムが悪くなる | シミュレータは事前学習に限る。仕上げは実機データで行う |
| ライセンス | DINOv3 は独自ライセンス [41]。OpenDV は YouTube の条件 [18] | 配布・商用の前に確認する。DINOv2 は Apache-2.0 |
| 自己教師あり学習が効かない [30][31] | 時間を無駄にする | 各段階の「合格の基準」で止める |

**未確認のこと**

- DINOv2-S/14、ConvNeXt-T、EfficientViT-B1 の、自分の環境での遅延。
- カメラの取付位置(`config.CALIBRATION` の 0.12m は未実測)と、画角(FOV)。
- 009/010/101 で衝突が増えた原因。ゲインを上げたこと・速度(eRPM p90 が約1,000〜1,800高い)・人が早めに交代したことが重なっていて、切り分けていない。
- 操作値の入力なし版の閉ループの成績(開ループでは +35% 悪い)。

---

## 14. GPU が空いたらすぐやること(短い順)

1. Orin Nano で、候補のバックボーンの TensorRT FP16 ベンチマークを取る。`tools/benchmark_model.py` の流れで、約1時間。
2. ai_run_003〜010・101 から、人が取り返した区間を抜き出して、介入区間の検証セットを作る。
3. clean_driving_20261011 で、次のものをオフラインと閉ループで比べる。
   - (a) 今の v0(**学習済み**、35.7µs)
   - (a') 操作値の入力なし(**学習済み**、49.0µs)。(a) との閉ループ比較は、走行の日にすぐできる。ゲインは 1.0 に固定する
   - (b) v0 + 操作値のドロップアウト 0.3
   - (c) DINOv2-S を凍結した版(線形プローブ)
4. 収集の画面に「コース配置 ID・照明」の欄を足すかどうかを決める(Stage 3 の前提)。

---

## 検証メモ(2026-10-11 改訂時)

一次資料(arXiv の本文・公式ページ)で確かめた主な数値と、直したところ。

| 対象 | 結果 |
|---|---|
| [46] Orin Nano の遅延 | **確認**。表の値は 224px・FP16 で MobileNetV4 1.426ms、EfficientNet-B0 2.113ms、ResNet50 1.858ms、ViT-S/16 2.359ms(30回の中央値)。ViT-S/16 を FP32 で作り直すと 5.92ms。DINOv2-L 27.0ms、DINOv3-L 44.0ms。条件は JetPack 6.2・TensorRT 10.3.0・MAXN_SUPER・GPU の計算時間だけ。ViT-S/16 の FP16 の低下は 4〜7ポイント(224px で 5.56) |
| [2] ViNT | **確認**(初版では二次情報としていた)。CARLA でゼロショット 40% → 1時間未満のデータで 80%(表3) |
| [59] スケーリング則 | **確認**。32組 × 50デモ。8/16/32組で 400/800/1,600デモで頭打ち。成功率は約90% |
| [33] 逆ダイナミクス | **確認**。序論に「スクラッチより一貫して良い唯一の目的」とある(要旨にはない) |
| [44] Theia | **確認**。H100 で約150 GPU時間(表5 では152)、ImageNet 120万枚を50エポック |
| [7][13][21][30][45][66] | 文中の主張と一致することを確認 |
| [12] DeepRacer | **確認**。2025-12-15 にコンソールのサービスが終わり、オープンソースの DeepRacer on AWS に移った |
| [10] | **修正**。題名を直した。結果は「約10分」ではなく「5分未満」。手法は SAC+VAE ではなく Dreamer(世界モデル型) |
| [42][49] | **修正**。URL が 404 だったので、つながる URL に直した |
| [4] CoryHall | **追記**。TFDS 版は 64×85 画素・7,331件・1.39GiB |
| [14] | 本文で RandConv を sim-to-real に使っていることを確認。I-RAS の正則化を8章に足した |

確かめていないもの: [47] の「126」の単位(ページに書いていない)、[82] の価格(二次情報で変わりやすい)、[83] の時間数(ブログの表では空欄だった)。

---

## 参考文献

1. GNM: A General Navigation Model to Drive Any Robot(Shah et al., 2022/ICRA 2023)https://arxiv.org/abs/2210.03370
2. ViNT: A Foundation Model for Visual Navigation(CoRL 2023)https://arxiv.org/abs/2306.14846
3. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration(2023)https://arxiv.org/abs/2310.07896
4. TFDS: berkeley_gnm_cory_hall https://www.tensorflow.org/datasets/catalog/berkeley_gnm_cory_hall
5. Open X-Embodiment: Robotic Learning Datasets and RT-X Models(2023)https://arxiv.org/abs/2310.08864
6. Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation(CrossFormer, CoRL 2024)https://arxiv.org/abs/2408.11812
7. Donkey Car docs: Train an autopilot with Keras https://docs.donkeycar.com/guide/deep_learning/train_autopilot/
8. gym-donkeycar https://github.com/tawnkramer/gym-donkeycar
9. Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems(IEEE TSE 2022)https://arxiv.org/abs/2112.11255
10. Viitala, Boney et al., Learning to Drive (L2D) as a Low-Cost Benchmark for Real-World Reinforcement Learning(2020。Dreamer で5分未満)https://arxiv.org/abs/2008.00715
11. DeepRacer: Educational Autonomous Racing Platform for Experimentation with Sim2Real RL(2019)https://arxiv.org/abs/1911.01562
12. AWS DeepRacer FAQ(サービス終了 2025-12-15)https://aws.amazon.com/es/deepracer/faqs/ / DeepRacer on AWS(後継のソリューション)https://builder.aws.com/content/2otsNvwo08hywC4Ovpr7mi1ySyI/deepracer-is-back-first-impressions-of-deepracer-on-aws
13. Sim-To-Real Transfer for Miniature Autonomous Car Racing(2020)https://arxiv.org/abs/2011.05617
14. Image-based Regularization for Action Smoothness in Autonomous Miniature Racing Car with Deep RL(2023)https://arxiv.org/abs/2307.08230
15. On learning racing policies with reinforcement learning(IROS 2025)https://arxiv.org/abs/2504.02420
16. F1TENTH の概観・ベンチマーク(2024)https://arxiv.org/abs/2402.18558 / TinyLidarNet(2024)https://arxiv.org/abs/2410.07447
17. Imitation Learning Approach for AI Driving Olympics Trained on Real-world and Simulation Data Simultaneously(2020)https://arxiv.org/abs/2007.03514
18. GenAD: Generalized Predictive Model for Autonomous Driving(OpenDV、CVPR 2024)https://arxiv.org/abs/2403.09630
19. Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining(ACO、ECCV 2022)https://arxiv.org/abs/2204.02393
20. Policy Pre-training for Autonomous Driving via Self-supervised Geometric Modeling(PPGeo、ICLR 2023)https://arxiv.org/abs/2301.01006
21. Visual Point Cloud Forecasting enables Scalable Autonomous Driving(ViDAR、CVPR 2024)https://arxiv.org/abs/2312.17655
22. Enhancing End-to-End Autonomous Driving with Latent World Model(LAW、ICLR 2025)https://arxiv.org/abs/2406.08481
23. Learning to Drive from a World Model(comma.ai、CVPRW 2025)https://arxiv.org/abs/2504.19077
24. Masked Autoencoders Are Scalable Vision Learners(MAE)https://arxiv.org/abs/2111.06377
25. VideoMAE(2022)https://arxiv.org/abs/2203.12602
26. MultiMAE: Multi-modal Multi-task Masked Autoencoders(ECCV 2022)https://arxiv.org/abs/2204.01678
27. Masked Visual Pre-training for Motor Control(MVP、2022)https://arxiv.org/abs/2203.06173
28. R3M: A Universal Visual Representation for Robot Manipulation(2022)https://arxiv.org/abs/2203.12601
29. Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?(VC-1、2023)https://arxiv.org/abs/2303.18240
30. On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline(ICML 2023)https://arxiv.org/abs/2212.05749
31. An Unbiased Look at Datasets for Visuo-Motor Pre-Training(CoRL 2023)https://arxiv.org/abs/2310.09289
32. What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?(2023)https://arxiv.org/abs/2312.12444
33. Inverse Dynamics Pretraining Learns Good Representations for Multitask Imitation(NeurIPS 2023)https://arxiv.org/abs/2305.16985
34. Video PreTraining (VPT)(2022)https://arxiv.org/abs/2206.11795
35. V-JEPA 2(2025)https://arxiv.org/abs/2506.09985
36. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning(2024)https://arxiv.org/abs/2411.04983
37. DreamerV3(2023)https://arxiv.org/abs/2301.04104
38. TD-MPC2(2023)https://arxiv.org/abs/2310.16828
39. DINOv2(2023)https://arxiv.org/abs/2304.07193
40. Vision Transformers Need Registers(2023)https://arxiv.org/abs/2309.16588
41. DINOv3(2025)https://arxiv.org/abs/2508.10104 / モデルカード(サイズ・ライセンス)https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
42. EfficientViT(2022/ICCV 2023)https://arxiv.org/abs/2205.14756 / Jetson AI Lab の手順(EfficientViT-L2-SAM、JetPack 5/6)https://jetson-ai-lab.com/vit/tutorial_efficientvit.html
43. AM-RADIO(CVPR 2024)https://arxiv.org/abs/2312.06709 / https://github.com/NVlabs/RADIO
44. Theia: Distilling Diverse Vision Foundation Models for Robot Learning(CoRL 2024)https://arxiv.org/abs/2407.20179
45. DINO Pre-training for Vision-based End-to-end Autonomous Driving(2024)https://arxiv.org/abs/2407.10803
46. Jetson Orin Nano での入力解像度と遅延の実測(ResNet50 / ViT-S/16 / DINOv2-L など、TensorRT 10.3、2026)https://arxiv.org/abs/2609.14247(表は https://arxiv.org/html/2609.14247v1)
47. Jetson AI Lab benchmarks(Orin Nano Super)https://www.jetson-ai-lab.com/archive/benchmarks.html
48. Jetson Orin Nano Super(67 TOPS、102GB/s)https://www.infoq.com/news/2024/12/nvidia-jetson-ai-supercomputer / https://www.phoronix.com/news/NVIDIA-Jetson-Orin-Nano-Super
49. TensorRT 10.x Accuracy Considerations(FP16 の上限 65,504、ICastLayer・層の精度の指定)https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html(10.16.0 の URL は 2026-10-11 時点で 404)/ LayerNorm の FP16 のあふれ https://github.com/pytorch/TensorRT/issues/2730 / LayerNorm を FP32 に指定しても融合で効かなかった例(利用者の報告)https://forums.developer.nvidia.com/t/layernorm-fp16-overflows-trt-8-5-6-on-xavier-nx/345971
50. MobileNetV4(2024)https://arxiv.org/abs/2404.10518
51. End-to-end Driving via Conditional Imitation Learning(CIL)https://arxiv.org/abs/1710.02410
52. FiLM(2017)https://arxiv.org/abs/1709.07871
53. RT-1(2022)https://arxiv.org/abs/2212.06817
54. Octo(2024)https://arxiv.org/abs/2405.12213
55. LoRA(2021)https://arxiv.org/abs/2106.09685
56. TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models(ICLR 2024)https://arxiv.org/abs/2310.05905
57. Surgical Fine-Tuning Improves Adaptation to Distribution Shifts(ICLR 2023)https://arxiv.org/abs/2210.11466
58. Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution(LP-FT、ICLR 2022)https://arxiv.org/abs/2202.10054
59. Data Scaling Laws in Imitation Learning for Robotic Manipulation(ICLR 2025)https://arxiv.org/abs/2410.18647
60. DAgger(Ross et al., 2011)https://arxiv.org/abs/1011.0686
61. HG-DAgger: Interactive Imitation Learning with Human Experts(ICRA 2019)https://arxiv.org/abs/1810.02890
62. Robot Learning on the Job(Sirius、RSS 2023)https://arxiv.org/abs/2211.08416
63. Fighting Copycat Agents in Behavioral Cloning from Observation Histories(NeurIPS 2020)https://arxiv.org/abs/2010.14876
64. Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?(CVPR 2024)https://arxiv.org/abs/2312.03031
65. On Offline Evaluation of Vision-based Driving Models(ECCV 2018)https://arxiv.org/abs/1809.04843
66. Scalable Offline Metrics for Autonomous Driving(2025)https://arxiv.org/abs/2510.08571
67. NAVSIM(NeurIPS 2024)https://arxiv.org/abs/2406.15349
68. End to End Learning for Self-Driving Cars(DAVE-2、2016)https://arxiv.org/abs/1604.07316
69. ChauffeurNet(2018)https://arxiv.org/abs/1812.03079
70. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World(2017)https://arxiv.org/abs/1703.06907
71. Mastering Visual Continuous Control: Improved Data-Augmented RL(DrQ-v2、2021)https://arxiv.org/abs/2107.09645
72. Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation(2025)https://arxiv.org/abs/2501.06693
73. RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning(NeurIPS 2025)https://arxiv.org/abs/2502.13144
74. Depth Anything V2(2024)https://arxiv.org/abs/2406.09414
75. ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware(2023)https://arxiv.org/abs/2304.13705
76. Diffusion Policy(2023)https://arxiv.org/abs/2303.04137
77. Offline RL with Implicit Q-Learning(IQL)https://arxiv.org/abs/2110.06169
78. Conservative Q-Learning(CQL)https://arxiv.org/abs/2006.04779
79. Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning(2023)https://arxiv.org/abs/2303.05479
80. Decision Transformer(2021)https://arxiv.org/abs/2106.01345
81. RvS: What is Essential for Offline RL via Supervised Learning?(2021)https://arxiv.org/abs/2112.10751
82. GPU の価格(二次情報、2026)https://deploybase.ai/articles/lambda-cloud-gpu-pricing / https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/
83. Yaak, LeRobot goes to driving school(L2D データセット、2025)https://www.yaak.ai/blog/lerobot-goes-to-driving-school / https://huggingface.co/datasets/yaak-ai/L2D
84. Roboflow Universe の Donkey Car の画像(物体検出用、692枚)https://universe.roboflow.com/jimmyaiot/donkey-car
