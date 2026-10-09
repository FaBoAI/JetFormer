# JetFormer基板 ファームウェア通信仕様(修正提案)

## 1. 現行ファーム(main/leracer.c)のI2C読み出しの問題

現行仕様:I2Cスレーブ 0x28、レジスタIDを1バイトwrite → int32(LE)を4バイトread。
ファームは「最後に選択されたレジスタの値」をTX FIFOに詰め続ける。

Jetson実機(2026-10-10)で確認した現象:

| 操作 | 読めたバイト列 | 解釈 |
|---|---|---|
| 0x00(PWM1)選択 → 24バイト読み | `… a2 05 00 00 a2 05 00 00` | 1442µs(正) |
| 続けて 0x10(モーター%)選択 → 24バイト読み | `05 00 00 a5 05 00 00 a5 …` | **PWM1の値が3バイトずれて出続け、モーター%が出てこない** |
| 4レジスタを順に8バイトずつ読む(15周) | PWM1 = 1444, 1499, 975, 768, … | **他チャネルの値が混入** |
| FIFOが空のとき | `00 00 00 …` | 本物の値0と区別できない |

原因:レジスタ切り替え時に、前のレジスタの値がFIFO(リングバッファ)に大量に残るため。
**Jetson側の工夫では各チャネルを正しく読み分けられない**(読み捨て量を増やすと今度はFIFOが空になりゴミを読む)。

## 2. 推奨:USBストリームに車両パケット `JFVH` を追加する

ToF(`L9FR`)を送っているUSB CDCのストリームに、同じ書き込み関数で以下のパケットを
**100Hz程度**で混ぜて送る。JetsonのJetFormerはすでに受信・検証に対応済み
(`config.VEHICLE['protocol'] = 'usb'` に切り替えるだけ)。

利点:
- ToFと操作値が**同じ基板の時計**で刻まれる(センサーフュージョンの時刻同期が正確になる)
- I2Cバス7(PCA9685・IMU・OLEDと共用)の帯域を使わない
- マジック+連番+チェックサムで、取りこぼし・破損を確実に検出できる

### パケット形式(リトルエンディアン、28バイト、パディングなし)

| オフセット | 型 | 内容 |
|---|---|---|
| 0 | char[4] | `'J' 'F' 'V' 'H'` |
| 4 | u32 | seq(パケット毎に+1) |
| 8 | u32 | t_us(`esp_timer_get_time()` の下位32bit) |
| 12 | u16 | PWM1 パルス幅 [µs](GPIO1) |
| 14 | u16 | PWM2 パルス幅 [µs](GPIO2) |
| 16 | u16 | PWM3 パルス幅 [µs](GPIO3) |
| 18 | u16 | モーター1 パルス幅 [µs](GPIO43、生値) |
| 20 | u16 | モーター1 回転 [%](0〜100) |
| 22 | u16 | flags: bit0〜2 = PWM1〜3の信号あり、bit3 = モーター1の信号あり(直近100ms以内にエッジがあれば1) |
| 24 | u32 | checksum = (seq + t_us + オフセット12〜22の各u16の和) & 0xFFFFFFFF |

`L9FR` と同じく「連番+データの和」のチェックサムです。

### 実装例(ESP-IDF)

```c
typedef struct __attribute__((packed)) {
    char     magic[4];      // "JFVH"
    uint32_t seq;
    uint32_t t_us;
    uint16_t pwm_us[3];
    uint16_t m1_us;
    uint16_t m1_pct;
    uint16_t flags;
    uint32_t checksum;
} jf_vehicle_packet_t;      // sizeof == 28

static void send_vehicle_packet(void)
{
    static uint32_t seq = 0;
    jf_vehicle_packet_t p;
    memcpy(p.magic, "JFVH", 4);
    p.seq    = seq++;
    p.t_us   = (uint32_t)esp_timer_get_time();
    p.pwm_us[0] = pwm_width_us[0];          // 既存のPWM計測値
    p.pwm_us[1] = pwm_width_us[1];
    p.pwm_us[2] = pwm_width_us[2];
    p.m1_us  = motor_width_us[0];
    p.m1_pct = motor_pct(motor_width_us[0]); // 既存の換算(10000µs=0%, 2600µs=100%)
    p.flags  = signal_valid_bits();          // 直近100ms以内にエッジがあったチャネル
    p.checksum = p.seq + p.t_us + p.pwm_us[0] + p.pwm_us[1] + p.pwm_us[2]
               + p.m1_us + p.m1_pct + p.flags;
    usb_stream_write(&p, sizeof(p));         // L9FRを送っているのと同じ関数で
}
// 10ms周期のタスク/タイマから send_vehicle_packet() を呼ぶ
```

注意:`L9FR` フレームの途中に割り込まないこと(フレーム単位で書き込むか、同じ送信キューに積む)。

## 3. 代替:I2Cのまま直す場合(スナップショットフレーム)

USBを使えない場合は、I2Cで「全値を1つのフレームにまとめて詰め続ける」方式にする。
レジスタ切り替えをやめるので、FIFOに古いデータが残っても壊れない。
Jetson側は `protocol = 'frame'` で対応済み(48バイト読み、チェックサムの合う最新フレームを採用)。

| オフセット | 型 | 内容 |
|---|---|---|
| 0 | char[2] | `'J' 'F'` |
| 2 | u16 | seq |
| 4 | u32 | t_ms(基板時刻) |
| 8 | u16 ×3 | PWM1〜3 [µs] |
| 14 | u16 | モーター1 [µs] |
| 16 | u16 | モーター1 [%] |
| 18 | u16 | flags(上と同じ) |
| 20 | u16 | 予約(0) |
| 22 | u16 | checksum = 先頭22バイトの和 & 0xFFFF |

ファームは値が更新されるたびに(または空きができるたびに)この24バイトをTX FIFOへ詰める。

## 4. 確認したい点

- ToF距離(`L9FR`)の単位(現在は mm と仮定)
- PWM1/PWM2/PWM3 のどれがステアリング/スロットルか(現在は PWM1=ステアリング、PWM2=スロットルと仮定)
- `#ST rx= center= drops=` の `drops` の意味(USB側が読んでいない間の破棄数?)
