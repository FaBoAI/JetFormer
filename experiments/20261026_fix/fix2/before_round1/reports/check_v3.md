# 段 0 の 8: 机上の台本(desk_v3・eval_v3)の確かめ

- 作った時刻: 2026-10-26 14:32:16(3.0 秒)
- protocol_v3.json `97c677d62467a421`・protocol_numbers_v3.json `5fad1a67ce17a251`
- desk_v3.py `3221880bf09dba6957eaee6c93016651dfdb8a0540ef4215b1a67a51c63f014c`・eval_v3.py `91044fa2f2735bc0d1cf7d9e7896c42d7f8a65782d03c9d6e95a9d6ca0c48f57`
- 許す差: ±0.01 µs・±0.0001
- 答え: **全部同じ**(外れ 0 個)

## 1. 確かめの表

| 部分 | 個数 | 外れ | 最大の差 [µs] | 最大の差 [割合] |
|---|---|---|---|---|
| 1 5 本の H1・介入 | 40 | 0 | 0.000000 | 0.0000000 |
| 1 5 本の平均・sd・閾値 | 54 | 0 | 0.000049 | 0.0000000 |
| 2 新しい欄(5 本) | 35 | 0 | 0.000000 | 0.0000000 |
| 2 新しい欄(v2 の 22 本) | 155 | 0 | 0.000000 | 0.0000000 |
| 2 5 本の平均・sd | 20 | 0 | 0.000050 | 0.0000484 |
| 3 物差し(保存した予測) | 109 | 0 | 0.004987 | 0.0000495 |
| 3 物差し 対 GPU の probe_check | 16 | 0 | 0.004558 | 0.0000441 |
| 3 low10: G1 の保存した予測 対 GPU の probe_check | 4 | 0 | 0.000000 | 0.0000454 |
| 3 5 本の平均・sd | 6 | 0 | 0.001098 | 0.0000183 |
| 4 v2 の 22 本の介入 | 89 | 0 | 0.000496 | 0.0000000 |
| 8 閾値の数え直し | 26 | 0 | 0.000351 | 0.0000445 |
| v2 の desk_v2.json と同じ | 298 | 0 | 0.000005 | 0.0000050 |
| v2 の eval_v2.json と同じ | 270 | 0 | 0.000050 | 0.0000489 |
| 自己試験(protocol の数え方) | 29 | 0 | 0.000000 | 0.0000000 |
| 自己試験(全部の門で v2 の 19 本が落ちる) | 19 | 0 | 0.000000 | 0.0000000 |

## 2. 門(gates_v3)を 27 本に当てた

| 学習 | 新しい門 | 全部の門 | 落ちた門 | 未の門 |
|---|---|---|---|---|
| 5 本: A_s0 | × | × | brake・probe_low10 | — |
| 5 本: A_s1 | 未 | × | brake | probe_low10 |
| 5 本: v0-s2 | 未 | 未 | — | engine_ok・sanity_ok・inference・brake・probe_low10 |
| 5 本: v0-s3 | 未 | 未 | — | engine_ok・sanity_ok・inference・brake・probe_low10 |
| 5 本: B | ○ | ○ | — | — |
| bfast_20261025-e15-s0 | × | × | brake・probe_low10 | — |
| bfast_20261025-e15-s1 | 未 | × | brake | probe_low10 |
| bfast_20261025-e15-s2 | 未 | × | brake | probe_low10 |
| v2_20261025-v2h-aug2-e60-s0 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-aug2-e60-s1 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st23・probe_stand_brake_side・probe_low10 | — |
| v2_20261025-v2h-aug2-e60-s2 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-aug2-e60-s3 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-e100-s0 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-e30-s0 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side・probe_low10 | — |
| v2_20261025-v2h-e30-s1 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side・probe_low10 | — |
| v2_20261025-v2h-e30-s2 | × | × | brake・straight・straight_1023・standstill_back・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-e30-s3 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-e60-s0 | × | × | brake・straight・straight_1010・straight_1023・probe_st23 | probe_low10 |
| v2_20261025-v2h-e60-s1 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-e60-s2 | × | × | brake・straight・straight_1010・straight_1023・standstill_back・probe_st10・probe_st23・probe_stand_brake_side・probe_low10 | — |
| v2_20261025-v2h-e60-s3 | × | × | brake・straight・straight_1023・standstill_back・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2h-msf-aug2-e60-s0 | × | × | inference・brake・straight・straight_1010・probe_st10 | probe_low10 |
| v2_20261025-v2h-msf-aug2-e60-s1 | × | × | brake・straight・straight_1010・straight_1023・probe_st10・probe_st23 | probe_low10 |
| v2_20261025-v2m-e30-s0 | × | × | brake・straight・straight_1023・standstill_back・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2m-e30-s1 | × | × | brake・straight・straight_1023・standstill_back・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2m-e30-s2 | × | × | brake・straight・straight_1023・standstill_back・probe_st23・probe_stand_brake_side | probe_low10 |
| v2_20261025-v2m-e30-s3 | × | × | brake・straight・straight_1023・standstill_back・probe_st23・probe_stand_brake_side | probe_low10 |

## 3. 足りない値(GPU で予測し直すのは学習のあと)

| 学習 | 欄 | 理由 | 出し方 |
|---|---|---|---|
| 5 本: A_s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| 5 本: A_s1 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| 5 本: A_s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| 5 本: v0-s2 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| 5 本: v0-s2 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| 5 本: v0-s2 | sanity_ok | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s2 | trt_p99_ms | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s2 | brake_lowspeed_frac | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s2 | engine_ok | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s3 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| 5 本: v0-s3 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| 5 本: v0-s3 | sanity_ok | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s3 | trt_p99_ms | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s3 | brake_lowspeed_frac | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: v0-s3 | engine_ok | エンジン・TensorRT の仕事が無い | エンジンを作り desk_v3.py trt(GPU) |
| 5 本: B | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| bfast_20261025-e15-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| bfast_20261025-e15-s1 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| bfast_20261025-e15-s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| bfast_20261025-e15-s2 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| bfast_20261025-e15-s2 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-aug2-e60-s0 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-aug2-e60-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-aug2-e60-s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-aug2-e60-s2 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-aug2-e60-s2 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-aug2-e60-s3 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-aug2-e60-s3 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e100-s0 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-e100-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e30-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e30-s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e30-s2 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-e30-s2 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e30-s3 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-e30-s3 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e60-s0 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-e60-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e60-s1 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-e60-s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e60-s2 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-e60-s3 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-e60-s3 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-msf-aug2-e60-s0 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-msf-aug2-e60-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2h-msf-aug2-e60-s1 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2h-msf-aug2-e60-s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2m-e30-s0 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2m-e30-s0 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2m-e30-s1 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2m-e30-s1 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2m-e30-s2 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2m-e30-s2 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
| v2_20261025-v2m-e30-s3 | probe の low10_brake_frac | 10/10 の低速は C3 の集合に無く、保存した FP32 の予測も無い | probe_gates.py run --ckpt best(GPU。学習のあと) |
| v2_20261025-v2m-e30-s3 | probe_last(last.pth の物差し) | C3 の保存した予測は best だけ | probe_gates.py run --ckpt last(GPU。門には使わない) |
