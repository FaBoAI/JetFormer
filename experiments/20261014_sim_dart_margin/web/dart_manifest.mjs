// 準備の段の record/install_dart.py が書いたもの(手で変えない)。データセットごとの先生・乱し・塩と、web/ のファイルの sha256
export const DART_MANIFEST = Object.freeze({
 "schema": "jf-sim-dart-browser-manifest/v1",
 "protocol_sha256": "c423c30f3a01ea705e07ad0ceae1175ff78c08dada85b803151e207323680aae",
 "datasets": {
  "D1": {
   "kind": "fast",
   "noise": "N_fast",
   "salt": 53671,
   "teacher_id": "rl-bestlap-ppo-dart-v1",
   "collection_name": "rl-dart-fast-10min",
   "policy_sha256": "7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee",
   "weights_sha256": "2babaf5c63561eeee9e4c25ddd2af7fa7ac5b6a0d52648528328018f0dafbcb7",
   "planned_single_lap_s": 8.093,
   "note": "T_fast(ベストラップの RL 方策)+ DART の乱し N_fast"
  },
  "D2": {
   "kind": "margin",
   "noise": null,
   "salt": null,
   "teacher_id": "rl-line-margin-c03-v1",
   "collection_name": "rl-margin-line-10min",
   "policy_sha256": "1ae6be0793914094dbabfbdc325905ab8af01f12ea3bc7b2638b05c6bd0167ae",
   "weights_sha256": "6e46797ed0cbceec78a75d89c91c7b6bfc5d98e4b31c1d59df081d60aac618e9",
   "planned_single_lap_s": 8.924,
   "note": "T_margin(経路いろいろ line r2、c0 = +0.3・c_v = 1.0・線 v2)乱しなし"
  },
  "D3": {
   "kind": "margin",
   "noise": "N_margin",
   "salt": 54183,
   "teacher_id": "rl-line-margin-c03-dart-v1",
   "collection_name": "rl-margin-dart-10min",
   "policy_sha256": "1ae6be0793914094dbabfbdc325905ab8af01f12ea3bc7b2638b05c6bd0167ae",
   "weights_sha256": "6e46797ed0cbceec78a75d89c91c7b6bfc5d98e4b31c1d59df081d60aac618e9",
   "planned_single_lap_s": 8.924,
   "note": "T_margin + DART の乱し N_margin"
  }
 },
 "files": {
  "dart-teacher.js": "7bab6bdf193fd54ab92de597eb4d0a8e0002f56f51a24e82a1c87c36903f6d51",
  "dart_driver.mjs": "32bfa72b586336964c1a7c707b31647ede2e9f11086cbb8e333021382d543e1a",
  "dart_noise.mjs": "372ca64ee0778c2ce5db3407a5189e1ed298dc25e61e7d1465927113205e887b",
  "entry-D1.js": "f9540380f11cd303fad5a5f7129d9d163bec1d5e2e763e785efa67e9ade02091",
  "entry-D2.js": "dd997d26df2f94eeae199b7ea87e4d2fbc8779c1154792b135adbe4e7ac6c456",
  "entry-D3.js": "77d5f262e52fcc2f8c5b33a152e286b412d5fd7cf2f1991b2bc9cd9d16380b06",
  "gate_miss.mjs": "bcd6da01502b23c93138f6d54f779a645d7c225b40ecb1ffa53f8ede1c44afbe",
  "plan_D1.mjs": "1ec16bc113b931d25267cf77b0de30e82c400bd5338633d9be75797b938cb705",
  "plan_D2.mjs": "38685c09fbfb1c900a58866a22fc746bfac6d5820e9280ded842a0f9823d8dd8",
  "plan_D3.mjs": "3d877d0a19c55506b7fb99854ea146b74e016e9df36b625b761900ed039e3d1c",
  "rt/catalogue.mjs": "e6ff082462c7cb4cd22886a5c7d23fa4661d713f4257207a9baba23168197ef9",
  "rt/features.mjs": "3cf941237cdce719ba963036f3f43cec67998b6c07e0295ef3a6d17b8ba38790",
  "rt/layouts.mjs": "a3d7051a563d7fc648a541753c3f71ab35c214b759da664ef8e2359fb69cebc4",
  "rt/line_profiles.mjs": "f7b4d330c0032dae73f7fc7d67f92e4f3933f9e2d539eddebe89c8338742d848",
  "rt/line_v2_table.json": "39e1b1c427b1246fcfe4008761b175d500ec83b2c926eee1d46b7005ad2378da",
  "rt/line_v2_table.mjs": "2016c7dda6607ea20cf57be6d2b76006ecbf2c911fde21e004d18e777433ae9e",
  "rt/mode_runtime.mjs": "50de50e7d490f954d0ff13a7de8b20dfb27f15601d6e6a18755e9763f7638f0d",
  "rt/obs_modes.mjs": "e72f638942f286a250175b5039c868eb759d5f97e17507c1e28106c4dfb49a69",
  "rt/perturb.mjs": "5bb84250e840dd18af7f7b839d3e92c45a4b74e13ab516f7b0ec09203742bed5",
  "rt/policy_forward.mjs": "f16f9f50a5b95065a5ab36a28419625f4e6df61f404fa213c7313569a3efa3e0",
  "rt/reference_loop.json": "0bd89546358b73fc0fb7201934151097d6e2dbd87521fb22a9b24596101f97e4",
  "rt/reference_loop.mjs": "446d4bbc506d88e9f9be048aaaebab59916e4179ca8192aeb49c06ad864cb901",
  "rt/rng.mjs": "abbf0c60f7f082ba12a3cde053a91ff83f3680b87ce7058b6361b1e772cff23b",
  "rt/room_table.json": "693358617d50462319c811f652f6aa093a443c34bec3a2e4bcda2ce7fb4b8d4a",
  "rt/room_table.mjs": "91e0dac4840f081c1355be809ddb5783811e4d9191e8705e2d14531791a67663",
  "rt/shield.mjs": "d682a6b5996102772298579d229189eae57f9434c159f9e7b6e8b754205e4c76",
  "rt/weights_fast.mjs": "2babaf5c63561eeee9e4c25ddd2af7fa7ac5b6a0d52648528328018f0dafbcb7",
  "rt/weights_margin.mjs": "6e46797ed0cbceec78a75d89c91c7b6bfc5d98e4b31c1d59df081d60aac618e9",
  "teacher_core.mjs": "abe903adc4bf41c26cfbae50348ef3e98e423931a5bf7f831d96d5e661d3c507"
 }
});
