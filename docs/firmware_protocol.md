# JetFormer基板との通信仕様(Jetson側の受け方)

ファームの正式な仕様はファーム側のREADMEが正です。ここでは、JetFormerがそれをどう読んでいるかをまとめます。

- コントローラ基板(XIAO ESP32-S3):`~/jetformer_board/firmware/leracer_ctrl/README.md`(ファーム v2)
- ToF(ESP32-P4 + VL53L9):`~/jetformer_board/firmware/csi_ranging/`(`tools/l9proto.py`)

## 車両データ:I2C bus 7 / 0x28、スナップショット(ファーム v2)

`jetformer/sensors/vehicle.py`。100Hzでレジスタ0x20から28バイトを読む(`read_i2c_block_data(0x28, 0x20, 28)`)。
基板はマスタが読みに来た瞬間にSCLをストレッチして、その時点の値を詰めて返す。

`struct "<BBHIHHHBBiiHBB"`:magic 0xA5 / version 1 / seq / t_us / CH1 / CH2 / CH3 / flags / hall_state /
erpm / hall_count / hall_errors / 予約 / CRC-8(多項式0x07、初期値0、byte 0..26)

| 項目 | 使い方 |
|---|---|
| CH1 / CH2 / CH3 [µs] | ステアリング / スロットル / モード切替。保存は `VEHICLE.npz` の `pwm[N,3]` |
| flags bit0-2 | CH1-3を受信中(100ms以内にパルスあり)→ `valid[N,3]`。偽の値は学習に使わない |
| flags bit3 SELECT | 1 = AI走行側。**人が操縦していないので学習の教師から除外**(`select`) |
| flags bit4 | モーター回転中(`moving`) |
| erpm | 電気角rpm(符号=回転方向)。モーター軸rpm = erpm / 極対数。モデルには erpm/20000 で入力 |
| hall_count / hall_errors | ホールのステップ累積(電気角1回転=6)/ 不正遷移の累積。品質確認用 |
| seq / t_us | 前回と同じseqは読み捨て。基板の時刻は `board_t_us` として保存 |

読み取りの扱い:
- 開始時に5回まで読み直してスナップショット形式を確認する。応答しなければ「ファームv2を書き込んでください」で止める
- CRC不一致・形式違いは3回まで読み直す(回数は画面の「読み直し」に出る)。3回続けて失敗したら再接続
- 2026-10-10 の確認:100Hz・500回連続で CRC エラー 0、seq は毎回 +1、基板時刻の間隔 10.00ms

### ファームへの改善提案:CH3の信号断でSELECTを人側へ

現在のファーム(leracer.c の main ループ)は、SELECTを「最後に測ったCH3のパルス幅」だけで決めている。
送信機の電源断・圏外で受信機がパルスを止めると、SELECTはAI側のまま残り、プロポで取り戻せない
(flags の受信中ビットだけが落ちる)。Jetson側はこの状態(SELECT=1かつCH3が受信なし)を検出して中立を出すが、
基板側でも次のようにするのが確実:

```c
int64_t now = esp_timer_get_time();
if (s_rc[2].fall == 0 || now - s_rc[2].fall > RC_TIMEOUT_US) {
    pwm3 = 0;                 /* CH3が来ていない → 人側(SELECT=0、LED青) */
}
```

あわせて、受信機のフェイルセーフでCH3を人側(1500µs未満)にしておくこと。

### 旧ファーム v1(`JETFORMER_VEHICLE_PROTOCOL=legacy`)

レジスタ0x00〜0x02を4バイトずつ読む方式。v1はTXバッファに古い値が溜まるため、レジスタを切り替えても
前のチャネルの値が出てくることがあり、**学習データには使えない**。互換のため読み出しだけ残してあり、
画面には「PWM:v1(学習不可)」と出る。データセットを作るときは自動的に除外される。

## ToF:USB CDC `/dev/ttyACM0`(csi_ranging ファーム)

`jetformer/sensors/tof.py`。解析はファーム公式の `l9proto.py` をそのまま取り込んだ
`jetformer/sensors/l9proto.py` を使う(自前実装しない)。

- フレーム:`L9FR`(生)/ `L9FE`(バイトスタッフィング:0xFF,0x0F,0x16,0x14,0x7D → 0x7D, 値^0x20)
- 中身:seq u32 | width u16 | height u16 | ゾーン u16 × (w×h) | checksum u32(= seq + **ペイロードの全バイトの和**)
- 54×42 ゾーン、約32fps。フレームの間に `#ST rx=.. center=.. drops=..` のステータス行が混ざる

ゾーン値(`jetformer/tof_format.py`):

| ビット | 意味 |
|---|---|
| 下位13bit | 距離 [mm]。0x1FFF = 範囲外(無効) |
| 上位3bit | 信頼度 0〜7 |

収集データ(`TOF_FRONT.npz`、データセットの16bit PNG、`aligned` の `tof`)は**生の値のまま**保存し、
使う側で `tof_format.distance_mm()` / `confidence()` を通して解釈する。
