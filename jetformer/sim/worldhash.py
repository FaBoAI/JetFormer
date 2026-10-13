"""シミュレーターのコースの world hash(sim-core.js の normalizedWorld + FNV-1a 32bit)を Python で計算する

sim-core.js は makeWorld(buildCourse().physics) を normalizedWorld() で正規化し、JSON.stringify した文字列に
FNV-1a 32bit をかけて report().world_hash にする。ここでは同じ正規化と、JavaScript と同じ数値の文字列化を行う
(JS のコードは実行しない)。確認済み: 旧 v2 = 547d7e80、Default(v3)= 0a464384。

FNV 32bit は衝突し得るので、同じコースかどうかの判定には正規化した JSON の sha256(canonical_sha256)を使う。
"""

import hashlib
import math

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def _finite(v):
    """JS の typeof v === 'number' && Number.isFinite(v)(Python の bool は数ではない)"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(v):
    """JSON の数値を JS と同じ double にする"""
    return float(v)


def _clamp_number(value, fallback, lo, hi):
    """sim-core の number(value, fallback, lo, hi): 有限ならはさみ込み、そうでなければ既定値"""
    if _finite(value):
        x = _num(value)
        return max(lo, min(x, hi)) if not (lo <= x <= hi) else value
    return fallback


def _point(v):
    if isinstance(v, list) and len(v) >= 2 and _finite(v[0]) and _finite(v[1]):
        return [v[0], v[1]]
    return None


def _dist(a, b):
    return math.hypot(_num(b[0]) - _num(a[0]), _num(b[1]) - _num(a[1]))


def _wrap(angle):
    a = _num(angle)
    return math.atan2(math.sin(a), math.cos(a))


def _js_string_of(v):
    """String(v)(コースの kind や id に使う。ここに来るのは文字列か数か真偽値だけ)"""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return js_number(v)
    raise ValueError('文字列にできない値です: %r' % (v,))


def _truthy(v):
    """JS の真偽(0・''・null・false・NaN は偽)"""
    if v is None or v is False:
        return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v == v and v != 0
    if isinstance(v, str):
        return v != ''
    return True


def normalized_world(inp):
    """sim-core.js の normalizedWorld() を移したもの(キーの順序も同じ)"""
    if not isinstance(inp, dict):
        inp = {}

    segments = []
    for i, s in enumerate(inp.get('segments') if isinstance(inp.get('segments'), list) else []):
        if not isinstance(s, dict):
            continue
        a, b = _point(s.get('a')), _point(s.get('b'))
        if not (a and b):
            continue
        seg = {'id': 'segment-%d' % i, 'a': a, 'b': b, 'radius': _clamp_number(s.get('radius'), .01, 0, 2),
               'height': _clamp_number(s.get('height'), .1, 0, 10),
               'kind': _js_string_of(s.get('kind')) if _truthy(s.get('kind')) else 'rail'}
        if _num(seg['height']) > 0:
            segments.append(seg)

    disks = []
    for i, s in enumerate(inp.get('disks') if isinstance(inp.get('disks'), list) else []):
        if not isinstance(s, dict) or not _finite(s.get('x')) or not _finite(s.get('z')):
            continue
        ob = {'id': 'disk-%d' % i, 'x': s['x'], 'z': s['z'], 'radius': _clamp_number(s.get('radius'), .02, 0, 2),
              'height': _clamp_number(s.get('height'), .1, 0, 10),
              'kind': _js_string_of(s.get('kind')) if _truthy(s.get('kind')) else 'post'}
        if ob['kind'] == 'post-foot':
            ob.update({'shape': 'rectangle', 'w': .16, 'd': .15})
        if _num(ob['height']) > 0:
            disks.append(ob)

    surfaces = []
    for i, s in enumerate(inp.get('surfaces') if isinstance(inp.get('surfaces'), list) else []):
        if (isinstance(s, dict) and _finite(s.get('x')) and _finite(s.get('z')) and _finite(s.get('w')) and _finite(s.get('d'))
                and _num(s['w']) > 0 and _num(s['d']) > 0):
            surfaces.append({'id': _js_string_of(s.get('id')) if _truthy(s.get('id')) else 'surface-%d' % i,
                             'x': s['x'], 'z': s['z'], 'w': s['w'], 'd': s['d'],
                             'mu': _clamp_number(s.get('mu'), .8, .01, 3)})

    pts = [p for p in (_point(v) for v in (inp.get('route') if isinstance(inp.get('route'), list) else [])) if p]
    route = [p for i, p in enumerate(pts) if not i or _dist(p, pts[i - 1]) > 1e-8]
    if len(route) > 1 and _dist(route[0], route[-1]) < 1e-8:
        route.pop()

    sp = inp.get('spawn') if isinstance(inp.get('spawn'), dict) else {}
    r0x = route[0][0] if route and _truthy(route[0][0]) else 0
    r0z = route[0][1] if route and _truthy(route[0][1]) else 0
    spawn = {'x': _clamp_number(sp.get('x'), r0x, -1e6, 1e6), 'z': _clamp_number(sp.get('z'), r0z, -1e6, 1e6),
             'yaw': _wrap(_clamp_number(sp.get('yaw'), 0, -1e6, 1e6))}

    bi = inp.get('bounds') if _truthy(inp.get('bounds')) and isinstance(inp.get('bounds'), dict) else {}
    bounds = {'minX': _clamp_number(bi.get('minX'), -20, -1e6, 1e6), 'maxX': _clamp_number(bi.get('maxX'), 20, -1e6, 1e6),
              'minZ': _clamp_number(bi.get('minZ'), -20, -1e6, 1e6), 'maxZ': _clamp_number(bi.get('maxZ'), 20, -1e6, 1e6)}
    if _num(bounds['maxX']) <= _num(bounds['minX']):
        bounds['minX'], bounds['maxX'] = -20, 20
    if _num(bounds['maxZ']) <= _num(bounds['minZ']):
        bounds['minZ'], bounds['maxZ'] = -20, 20

    crossings = []
    for i, g in enumerate(inp.get('requiredCrossings') if isinstance(inp.get('requiredCrossings'), list) else []):
        if not isinstance(g, dict) or not _finite(g.get('minX')) or not _finite(g.get('maxX')) or not _finite(g.get('z')):
            continue
        if _num(g['minX']) >= _num(g['maxX']):
            continue
        d = g.get('direction')
        crossings.append({'id': _js_string_of(g.get('id')) if _truthy(g.get('id')) else 'crossing-%d' % i,
                          'surface_id': _js_string_of(g.get('surface_id')) if _truthy(g.get('surface_id')) else '',
                          'minX': g['minX'], 'maxX': g['maxX'], 'z': g['z'],
                          'direction': 1 if (_finite(d) and _num(d) == 1) else -1})

    return {'segments': segments, 'disks': disks, 'surfaces': surfaces, 'bounds': bounds, 'route': route,
            'spawn': spawn, 'requiredCrossings': crossings}


def js_number(v):
    """JavaScript の Number.prototype.toString(JSON.stringify と同じ)。-0 は '0'、非有限は 'null'"""
    if isinstance(v, bool):
        raise TypeError('真偽値は数ではありません')
    x = float(v)
    if not math.isfinite(x):
        return 'null'
    if x == 0:
        return '0'
    sign = '-' if x < 0 else ''
    r = repr(abs(x))                      # 最短で元に戻る桁(JS と同じ桁を選ぶ)
    mant, _, exp = r.partition('e')
    e10 = int(exp) if exp else 0
    ip, _, fp = mant.partition('.')
    digits = (ip + fp).lstrip('0')
    e10 -= len(fp)
    stripped = digits.rstrip('0')
    e10 += len(digits) - len(stripped)
    digits = stripped
    k = len(digits)
    n = k + e10                           # 値 = 0.digits × 10^n
    if k <= n <= 21:
        s = digits + '0' * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + '.' + digits[n:]
    elif -6 < n <= 0:
        s = '0.' + '0' * (-n) + digits
    else:
        e = n - 1
        es = ('+' if e >= 0 else '-') + str(abs(e))
        s = (digits if k == 1 else digits[0] + '.' + digits[1:]) + 'e' + es
    return sign + s


_ESC = {'"': '\\"', '\\': '\\\\', '\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}


def _js_quote(s):
    out = ['"']
    for ch in s:
        c = ord(ch)
        if ch in _ESC:
            out.append(_ESC[ch])
        elif c < 0x20 or 0xD800 <= c <= 0xDFFF:
            out.append('\\u%04x' % c)
        else:
            out.append(ch)
    out.append('"')
    return ''.join(out)


def js_stringify(v):
    """JSON.stringify(v)(空白なし。dict はキーの順序のまま)"""
    if v is None:
        return 'null'
    if v is True:
        return 'true'
    if v is False:
        return 'false'
    if isinstance(v, (int, float)):
        return js_number(v)
    if isinstance(v, str):
        return _js_quote(v)
    if isinstance(v, (list, tuple)):
        return '[' + ','.join(js_stringify(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ','.join(_js_quote(str(k)) + ':' + js_stringify(x) for k, x in v.items()) + '}'
    raise TypeError('JSON にできない値です: %r' % (v,))


def fnv1a32(text):
    """sim-core の hash(): 文字列のコードポイントごとに charCodeAt(0)(UTF-16 の先頭の単位)をかける"""
    h = FNV_OFFSET
    for ch in text:
        c = ord(ch)
        if c > 0xFFFF:
            c = 0xD800 + ((c - 0x10000) >> 10)
        h = ((h ^ c) * FNV_PRIME) & 0xFFFFFFFF
    return '%08x' % h


def world_hash(world):
    """makeWorld の出力(physics-world.json)の world hash(開始ゲート 0・ずれ 0 の正準 hash)"""
    return fnv1a32(js_stringify(normalized_world(world)))


def canonical_sha256(world):
    """正規化した JSON 文字列の sha256(FNV の衝突に頼らずに同じコースかを判定する)"""
    return hashlib.sha256(js_stringify(normalized_world(world)).encode('utf-8')).hexdigest()


def describe(world):
    """{world_hash, canonical_sha256, stats}"""
    nw = normalized_world(world)
    text = js_stringify(nw)
    return {'world_hash': fnv1a32(text), 'canonical_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'stats': {'segments': len(nw['segments']), 'disks': len(nw['disks']), 'surfaces': len(nw['surfaces']),
                      'gates': len(nw['route']), 'crossings': len(nw['requiredCrossings'])}}
