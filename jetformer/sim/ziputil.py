"""ZIP と JSON を安全に読むための共通の確認(シミュレーター本体の導入・コースのアップロードで使う)"""

import json
import stat
import zipfile

JUNK_PREFIXES = ('__MACOSX/',)
JUNK_NAMES = ('.DS_Store', 'Thumbs.db')


class ZipCheckError(ValueError):
    """ZIP の形が受け付けられない(日本語の理由)"""


def is_junk(name):
    base = name.rstrip('/').rsplit('/', 1)[-1]
    return name.startswith(JUNK_PREFIXES) or base in JUNK_NAMES or base.startswith('._')


def check_path(name, allow_dir=False):
    """ZIP の中のパスを確かめて要素に分ける。絶対パス・..・\\・:・NUL・空の要素は拒否する"""
    if not isinstance(name, str) or not name:
        raise ZipCheckError('空のパスがあります')
    if '\x00' in name or '\\' in name or ':' in name:
        raise ZipCheckError('使えない文字(\\ : NUL)を含むパスがあります: %r' % name)
    if name.startswith('/'):
        raise ZipCheckError('絶対パスが入っています: %s' % name)
    body = name[:-1] if (allow_dir and name.endswith('/')) else name
    parts = body.split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise ZipCheckError('「..」や空の要素を含むパスがあります: %s' % name)
    return parts


def entry_type(info):
    """'file' か 'dir'。シンボリックリンク・特殊ファイル・暗号化・未対応の圧縮方式は拒否する"""
    if info.flag_bits & 0x1:
        raise ZipCheckError('暗号化されたファイルは受け付けません: %s' % info.filename)
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ZipCheckError('未対応の圧縮方式です(stored と deflate だけ): %s' % info.filename)
    mode = (info.external_attr >> 16) & 0o170000
    if info.filename.endswith('/'):
        if mode not in (0, stat.S_IFDIR):
            raise ZipCheckError('フォルダーの種類が不正です: %s' % info.filename)
        return 'dir'
    if mode not in (0, stat.S_IFREG):
        raise ZipCheckError('シンボリックリンクや特殊ファイルは受け付けません: %s' % info.filename)
    return 'file'


def scan(zf, max_entries, max_ratio=None, ratio_min_bytes=1 << 20):
    """中央ディレクトリを確かめて [(info, parts, kind)] を返す(ゴミは除く)。
    重複(大文字小文字を無視)・件数・圧縮率もここで見る"""
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise ZipCheckError('ファイルが多すぎます(%d 件。%d 件まで)' % (len(infos), max_entries))
    seen = set()
    out = []
    for info in infos:
        if is_junk(info.filename):
            continue
        kind = entry_type(info)
        parts = check_path(info.filename, allow_dir=(kind == 'dir'))
        key = '/'.join(parts).casefold()
        if key in seen:
            raise ZipCheckError('同じ名前のファイルが重複しています(大文字小文字の違いを含む): %s' % info.filename)
        seen.add(key)
        if (max_ratio and kind == 'file' and info.file_size >= ratio_min_bytes
                and info.file_size > max_ratio * max(info.compress_size, 1)):
            raise ZipCheckError('圧縮率が高すぎるファイルがあります(%s)' % info.filename)
        out.append((info, parts, kind))
    return out


def _reject_constant(name):
    raise ValueError('NaN・Infinity は使えません')


def _depth(v, limit, d=1):
    if d > limit:
        raise ValueError('JSON の入れ子が深すぎます(%d 段まで)' % limit)
    if isinstance(v, dict):
        for x in v.values():
            _depth(x, limit, d + 1)
    elif isinstance(v, list):
        for x in v:
            _depth(x, limit, d + 1)


def strict_json(data, what, max_depth=None):
    """NaN・Infinity を拒否して JSON を読む(入れ子の深さも制限できる)。だめなら ValueError(日本語)"""
    try:
        text = data.decode('utf-8') if isinstance(data, (bytes, bytearray)) else data
        value = json.loads(text, parse_constant=_reject_constant)
    except RecursionError:
        raise ValueError('%s: JSON の入れ子が深すぎます' % what)
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError('%s を JSON として読めません: %s' % (what, e))
    if max_depth:
        try:
            _depth(value, max_depth)
        except ValueError as e:
            raise ValueError('%s: %s' % (what, e))
    return value


def read_limited(zf, info, limit, what):
    """ZIP の 1 ファイルを上限つきで読む(宣言の大きさだけを信じない)"""
    if info.file_size > limit:
        raise ZipCheckError('%s が大きすぎます(%d バイト。%d バイトまで)' % (what, info.file_size, limit))
    with zf.open(info) as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ZipCheckError('%s が大きすぎます' % what)
    return data
