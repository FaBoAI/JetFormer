# 出典: JetFormer基板 firmware/csi_ranging/tools/l9proto.py(ファーム公式のパーサ。手を加えずに取り込む)
"""VL53L9 USB CDC フレームの受信パーサ(l9viz / l9pc / l9web / jetracer 共通)

2つの形式を受け付ける(リトルエンディアン):
  L9FR(旧): [magic 'L9FR'][seq u32][w u16][h u16][zones w*h*u16][checksum u32]
  L9FE(新): magic 'L9FE' の後ろの全バイトをエスケープしたもの。
            0xFF, 0x0F, 0x16, 0x14, 0x7D は「0x7D, 値^0x20」の2バイトで送られる。
  checksum = seq + ゾーン全バイトの加算和 (u32 wrap)。L9FE はエスケープ前の値で計算する。

L9FE にした理由: macOS の Chrome(Web Serial)は tty を PARMRK 付き・IEXTEN 残しで開く。
その結果、カーネルの高速経路が無効になって受信した 0xFF が「FF FF」に膨らみ、受信キュー
(約1KB)が溢れて中身ごと捨てられる(IEXTEN が残っていれば 0x0F / 0x16 も削除される)。
問題になる値をストリームに出さないことで、ブラウザからでも欠落なく受信できる。

feed() には届いた分だけを細かく渡してよい。L9FE のデコードは途中経過を持ち越すので、
細切れに渡しても 1 フレームあたりの処理量は変わらない。
"""
import struct

MAGIC_RAW = b"L9FR"
MAGIC_ESC = b"L9FE"
ESC = 0x7D
_ESC_B = bytes((ESC,))
_HDR = struct.Struct("<IHH")  # seq, w, h(magic の直後)


class FrameParser:
    """バイト列を少しずつ与えると、検証済みフレーム (seq, w, h, payload) を返す"""

    def __init__(self):
        self.buf = bytearray()
        self.frames = 0
        self.crc_errors = 0
        self._esc = None   # 先頭の L9FE フレームのデコード途中経過 [j, dec, want, dims]

    def feed(self, data):
        out = []
        buf = self.buf
        buf += data
        while True:
            if self._esc is None:
                i = _find_magic(buf)
                if i < 0:
                    if len(buf) > 3:
                        del buf[:-3]
                    break
                if i > 0:
                    del buf[:i]
                if buf[:4] == MAGIC_ESC:
                    self._esc = [4, bytearray(), _HDR.size, None]
            if self._esc is not None:
                body = _decode_escaped(buf, self._esc)
            else:
                body = _decode_raw(buf)
            if body is None:
                break  # フレームの途中まで → 続きを待つ
            self._esc = None
            consumed, frame = body
            if frame is None:
                del buf[:4]  # 壊れている → magic の直後から探し直す(自称長は信用しない)
                continue
            seq, w, h, payload, crc = frame
            if (seq + sum(payload)) & 0xFFFFFFFF != crc:
                self.crc_errors += 1
                del buf[:4]
                continue
            del buf[:consumed]
            self.frames += 1
            out.append((seq, w, h, payload))
        return out


def _find_magic(buf):
    a = buf.find(MAGIC_RAW)
    b = buf.find(MAGIC_ESC)
    if a < 0:
        return b
    if b < 0:
        return a
    return min(a, b)


def _valid_dims(w, h):
    return 0 < w <= 256 and 0 < h <= 256


def _decode_raw(buf):
    if len(buf) < 4 + _HDR.size:
        return None
    seq, w, h = _HDR.unpack_from(buf, 4)
    if not _valid_dims(w, h):
        return 4, None
    n = w * h * 2
    need = 4 + _HDR.size + n + 4
    if len(buf) < need:
        return None
    payload = bytes(buf[4 + _HDR.size:4 + _HDR.size + n])
    (crc,) = struct.unpack_from("<I", buf, need - 4)
    return need, (seq, w, h, payload, crc)


def _decode_escaped(buf, st):
    """st = [j(次に読む buf 位置), dec(デコード済み), want(必要な dec 長), dims]。
    足りなければ st を更新して None、揃えば (消費バイト数, frame or None)"""
    j, dec, want, dims = st
    end = len(buf)
    while True:
        while len(dec) < want:
            stop = min(end, j + (want - len(dec)))
            k = buf.find(_ESC_B, j, stop)
            if k < 0:
                dec += buf[j:stop]
                j = stop
                if len(dec) < want:
                    st[0] = j
                    return None
            else:
                dec += buf[j:k]
                if k + 1 >= end:
                    st[0] = k
                    return None
                dec.append(buf[k + 1] ^ 0x20)
                j = k + 2
        if dims is not None:
            break
        seq, w, h = _HDR.unpack_from(dec, 0)
        if not _valid_dims(w, h):
            return 4, None
        dims = (w, h)
        want = _HDR.size + w * h * 2 + 4
        st[2], st[3] = want, dims
    seq, w, h = _HDR.unpack_from(dec, 0)
    n = w * h * 2
    payload = bytes(dec[_HDR.size:_HDR.size + n])
    (crc,) = struct.unpack_from("<I", dec, _HDR.size + n)
    return j, (seq, w, h, payload, crc)


def encode_escaped(seq, w, h, payload):
    """テスト用: ファームと同じ規則で L9FE パケットを組み立てる"""
    raw = _HDR.pack(seq, w, h) + payload
    crc = (seq + sum(payload)) & 0xFFFFFFFF
    raw += struct.pack("<I", crc)
    out = bytearray(MAGIC_ESC)
    for b in raw:
        if b in (0xFF, 0x0F, 0x16, 0x14, ESC):
            out += bytes((ESC, b ^ 0x20))
        else:
            out.append(b)
    return bytes(out)
