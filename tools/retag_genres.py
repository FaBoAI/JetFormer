#!/usr/bin/env python3
"""ジャンルの付け替え(2026-11-07 のユーザーの決め)。記録のジャンルを次の表にそろえる(番号も振り直した):

    0 未設定 / 1〜3 駐車(1)(2)(3) / 4 練習コース 人間 / 5 練習コース AI / 6 予選コース 人間 / 7 予選コース AI / 8 シミュレーター

    python tools/retag_genres.py modes  --backup DIR    data/modes.json を上の表にする(前の表は DIR へ写す)
    python tools/retag_genres.py plan   [--day YYYYMMDD [--course 予選コース|練習コース]]   付け替えの表を出すだけ(何も書かない)
    python tools/retag_genres.py apply  --backup DIR [--day ...]   付け替える(書く前に元の log.json・scene.json を DIR へ写す)
    python tools/retag_genres.py undo   --backup DIR    apply・modes の前に戻す(そのあとで変わっていないファイルだけ)

決まり
- コース: 11/6・11/8(予選の会場・車体オレンジ)は予選コース、それ以外の日は練習コース(QUALIFYING_DAYS。--day のときは --course で変えられる)
- 運転: 記録の log.json の drive がある = AI、無い = 人間(記録ごと。AI の記録の中で人が手で戻した区間も AI)。
  log.json が壊れている記録は、シーンの scene.json の drive で見る。drive の dry_run(モーターに出さない確かめ)も AI
  (V2 のデータでも AI の記録として扱っていた)
- 付け替えるのは走りのジャンル(4〜7 と、11/7 の昼に仮に使った 10〜13)の区間だけ。コース × 運転の番号にそろえる。
  駐車(1〜3)・未設定(0)・シミュレーター(8)の区間はそのまま。区間の境目も変えない
- ただし 11/6 の記録は、駐車の印でも予選コースにする(ALL_QUALIFYING_DAYS。ユーザーの決め 11/7「11/6 の走行は全て予選コース」)
- AI 走行の記録には、走ったときのモデルの番号(L-A は 4 = 前の高速走行)が付く。apply で、その日の AI の記録を 5 か 7 にそろえる
- 最初の番号は mode_id_before(scene.json は mode_segments_before も)に残す(一度だけ。AI の記録では走ったときのモデルの番号)
- 走行中・収集中は書かない(http://localhost:8080/api/status で確かめる。--force で飛ばす)
- 前のモデル(V2・L-A など)は、自分の学んだ表(model_config の modes)で走るので、走行タブの番号と名前(4 高速走行・5 低速走行)は
  この表と別(jetformer/web/static/app.js の走行タブはモデルの表から作る)
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jetformer import config  # noqa: E402

TABLE = [
    {'id': 0, 'name': '未設定', 'description': 'ジャンル未設定(学習には使わない)'},
    {'id': 1, 'name': '駐車(1)', 'description': ''},
    {'id': 2, 'name': '駐車(2)', 'description': ''},
    {'id': 3, 'name': '駐車(3)', 'description': ''},
    {'id': 4, 'name': '練習コース 人間', 'description': '練習コース(10/10〜11/3 の会場)で人が運転した記録'},
    {'id': 5, 'name': '練習コース AI', 'description': '練習コースで AI が運転した記録(AI の記録の中で人が手で戻した区間も含む)'},
    {'id': 6, 'name': '予選コース 人間', 'description': '予選コース(11/6・11/8 の会場。車体オレンジ)で人が運転した記録'},
    {'id': 7, 'name': '予選コース AI', 'description': '予選コースで AI が運転した記録(AI の記録の中で人が手で戻した区間も含む)。'
                                                 'AI 走行の記録には走ったモデルの番号(L-A は 4)が付くので、あとで tools/retag_genres.py apply で付け替える'},
    {'id': 8, 'name': 'シミュレーター', 'description': ''},
]
KEEP_IDS = {0, 1, 2, 3, 8}             # 前の表から名前ごと引き継ぐ(名前が違えば止める)
DRIVING = {4, 5, 6, 7, 10, 11, 12, 13}  # 付け替える区間のジャンル(10〜13 は 11/7 の昼に仮に使った番号)
PARKING = {1, 2, 3}
QUALIFYING_DAYS = {'20261106', '20261108'}
ALL_QUALIFYING_DAYS = {'20261106'}      # この日は駐車の印の記録も予選コースにする
NEW_ID = {('練習コース', '人間'): 4, ('練習コース', 'AI'): 5, ('予選コース', '人間'): 6, ('予選コース', 'AI'): 7}
MANIFEST = 'manifest.json'


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    """dataset._write_json と同じ書き方(一時ファイル → fsync → 置き換え・indent 1・日本語はそのまま)"""
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- ジャンルの表
def merged_modes(modes):
    """今の表 → 新しい表(TABLE)。何度通しても同じ。0〜3・8 の名前が違う・9 から先に画面で足したジャンルがあれば止める"""
    have = {m['id']: m for m in modes}
    for i in sorted(KEEP_IDS):
        if i in have and have[i]['name'] != TABLE[i]['name']:
            raise SystemExit('ID %d が「%s」です(「%s」のはず。付け替えの前提と違うので止めます)' % (i, have[i]['name'], TABLE[i]['name']))
    known = {m['name'] for m in TABLE} | {'高速走行', '低速走行', '低速走行・良', '高速走行・良', 'AI・きれいな速い周'}
    extra = sorted('%d %s' % (m['id'], m['name']) for m in modes if m['id'] > 8 and m['name'] not in known)
    if extra:
        raise SystemExit('画面から足したジャンル %s があります(消さないので、先にどうするか決めてください)' % extra)
    out = []
    for t in TABLE:
        if t['id'] in KEEP_IDS and t['id'] in have:
            m = {k: v for k, v in have[t['id']].items() if k != 'retired'}     # 説明は前の物を残す
            out.append(m)
        else:
            out.append(dict(t))
    return out


def names_of(modes):
    return {m['id']: m['name'] for m in modes}


# ---------------------------------------------------------------- 付け替えの表
def segs_of(sj):
    segs = sj.get('mode_segments')
    if isinstance(segs, list) and segs and all(isinstance(g, dict) and isinstance(g.get('mode_id'), int) for g in segs):
        return segs
    return None


def plan(raw_dir, day=None, course=None):
    """data/raw → 付け替えの表(書かない)。{'logs': [...], 'scenes': [...], 'skipped': [...]}"""
    raw_dir = Path(raw_dir)
    logs, scenes, skipped = [], [], []
    for d in sorted(os.listdir(raw_dir)):
        p = raw_dir / d
        if not p.is_dir():
            continue
        if d.startswith('ds-'):
            skipped.append({'log': d, 'why': 'シミュレーターの記録(そのまま)'})
            continue
        dday = d[:8]
        if day and dday != day:
            continue
        crs = course if (day and course) else ('予選コース' if dday in QUALIFYING_DAYS else '練習コース')
        olds = DRIVING | (PARKING if dday in ALL_QUALIFYING_DAYS else set())
        lj = read_json(p / 'log.json')
        log_ok = isinstance(lj, dict)
        log_ai = bool(log_ok and isinstance(lj.get('drive'), dict))
        scene_ai = set()
        for s in sorted(x for x in os.listdir(p) if (p / x).is_dir() and x.startswith(d + '-')):
            sj = read_json(p / s / 'scene.json')
            if not isinstance(sj, dict):
                skipped.append({'scene': s, 'why': 'scene.json が無い・読めない(そのまま)'})
                continue
            segs = segs_of(sj)
            if segs is None:
                skipped.append({'scene': s, 'why': 'ジャンルの区間が無い・形が違う(そのまま)'})
                continue
            ai = log_ai if log_ok else bool(sj.get('drive'))
            scene_ai.add(bool(sj.get('drive')))
            who = 'AI' if ai else '人間'
            new = NEW_ID[(crs, who)]
            ids = [g['mode_id'] for g in segs]
            main = sj.get('mode_id')
            change = any(i in olds and i != new for i in ids) or (main in olds and main != new)
            row = {'scene': s, 'log': d, 'day': dday, 'course': crs, 'who': who, 'old': main, 'old_segments': ids,
                   'new': new if (change or main in olds) else main, 'action': 'retag' if change else 'keep',
                   'olds': sorted(olds), 'duration_s': float(sj.get('duration_s') or 0.0)}
            if not change:
                row['why'] = ('そろっている' if main == new else '駐車' if main in PARKING else '未設定' if main == 0 else
                              'シミュレーター' if main == 8 else 'ジャンル %s' % main)
            scenes.append(row)
        if not log_ok:
            skipped.append({'log': d, 'why': 'log.json が無い・読めない(log.json はそのまま。シーンは scene.json の drive で分けた)'})
            continue
        if scene_ai and scene_ai != {log_ai}:
            skipped.append({'log': d, 'why': '注意: log.json と scene.json の drive が合わない(log.json の側で分けた)'})
        old = lj.get('mode_id')
        new = NEW_ID[(crs, 'AI' if log_ai else '人間')]
        change = old in olds and old != new
        logs.append({'log': d, 'day': dday, 'course': crs, 'who': 'AI' if log_ai else '人間', 'old': old,
                     'new': new if (change or old in olds) else old, 'action': 'retag' if change else 'keep',
                     'olds': sorted(olds), 'run': (lj.get('drive') or {}).get('run') if log_ai else None})
    return {'made_at': now(), 'raw': str(raw_dir), 'day': day, 'course': course, 'logs': logs, 'scenes': scenes,
            'skipped': skipped}


def summary(pl, names):
    """(前 → 後) ごとのシーンの数と分"""
    agg = {}
    for r in pl['scenes']:
        k = (r['action'], r['old'], r['new'])
        a = agg.setdefault(k, [0, 0.0])
        a[0] += 1
        a[1] += r['duration_s'] / 60.0
    lines = []
    for (act, old, new), (n, mins) in sorted(agg.items(), key=lambda x: (x[0][0] != 'retag', str(x[0][2]), str(x[0][1]))):
        lines.append('%-6s %2s → %2s %-10s %4d シーン %6.1f 分' % (
            '付け替え' if act == 'retag' else 'そのまま', old, new, names.get(new, '?'), n, mins))
    nl = sum(1 for r in pl['logs'] if r['action'] == 'retag')
    lines.append('記録(log.json): 付け替え %d・そのまま %d・飛ばした %d' % (
        nl, len(pl['logs']) - nl, sum(1 for s in pl['skipped'] if 'log' in s)))
    return lines


# ---------------------------------------------------------------- 書く
def busy(force):
    if force:
        return None
    try:
        with urllib.request.urlopen('http://localhost:8080/api/status', timeout=3) as r:
            st = json.load(r)
    except Exception:
        return None                    # 管理画面が動いていない(試験など)
    if (st.get('recorder') or {}).get('recording') or (st.get('session') or {}).get('active') or st.get('driving'):
        return '収集中か走行中です(終わってから流してください)'
    return None


class Writer(object):
    """元のファイルを backup/data/<data からの相対の場所> へ写してから書く。manifest に元と後の sha256 を残す"""

    def __init__(self, backup, data_dir):
        self.backup = Path(backup)
        self.data_dir = Path(data_dir)
        self.mpath = self.backup / MANIFEST
        self.man = read_json(self.mpath) or {'made_at': now(), 'files': {}}

    def write(self, path, obj, note):
        path = Path(path)
        rel = str(path.relative_to(self.data_dir))
        ent = self.man['files'].get(rel)
        if ent is None:
            dst = self.backup / 'data' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(path, dst)
            if sha(dst) != sha(path):
                raise SystemExit('写しが元と合いません: %s' % rel)
            ent = {'before_sha': sha(path), 'backup': str(dst)}
        write_json(path, obj)
        ent.update(after_sha=sha(path), note=note, at=now())
        self.man['files'][rel] = ent
        self.save()

    def save(self):
        self.backup.mkdir(parents=True, exist_ok=True)
        write_json(self.mpath, self.man)


def apply(pl, names, w, retag_note):
    n_scene = n_log = 0
    raw = Path(pl['raw'])
    for r in pl['scenes']:
        if r['action'] != 'retag':
            continue
        sp = raw / r['log'] / r['scene'] / 'scene.json'
        sj = read_json(sp)
        segs = segs_of(sj)
        if sj.get('mode_id') != r['old'] or [g['mode_id'] for g in segs] != r['old_segments']:
            raise SystemExit('%s が表を作ったあとで変わりました(もう一度 plan から)' % r['scene'])
        sj.setdefault('mode_id_before', sj.get('mode_id'))
        sj.setdefault('mode_segments_before', [dict(g) for g in segs])
        new, olds = r['new'], set(r['olds'])
        sj['mode_segments'] = [{k: (new if k == 'mode_id' and v in olds else v) for k, v in g.items() if k != 'mode_name'}
                               for g in segs]
        if sj.get('mode_id') in olds:
            sj['mode_id'] = new
        sj.pop('mode_name', None)             # 名前は読むときに付ける(dataset.update_scene と同じ)
        sj.pop('mode_mixed', None)
        sj['mode_retag'] = retag_note
        w.write(sp, sj, 'scene %s → %s' % (r['old'], new))
        n_scene += 1
    for r in pl['logs']:
        if r['action'] != 'retag':
            continue
        lp = raw / r['log'] / 'log.json'
        lj = read_json(lp)
        if not isinstance(lj, dict) or lj.get('mode_id') != r['old']:
            raise SystemExit('%s/log.json が表を作ったあとで変わりました(もう一度 plan から)' % r['log'])
        lj.setdefault('mode_id_before', lj.get('mode_id'))
        lj.setdefault('mode_name_before', lj.get('mode_name'))
        lj['mode_id'] = r['new']
        lj['mode_name'] = names[r['new']]
        lj['mode_retag'] = retag_note
        w.write(lp, lj, 'log %s → %s' % (r['old'], r['new']))
        n_log += 1
    return n_scene, n_log


def ids_in_raw(raw_dir):
    """data/raw の scene.json・log.json に出てくるジャンルの番号の集まり"""
    out = set()
    for dp, _dn, fn in os.walk(raw_dir):
        for f in fn:
            if f not in ('scene.json', 'log.json'):
                continue
            j = read_json(os.path.join(dp, f))
            if not isinstance(j, dict):
                continue
            if isinstance(j.get('mode_id'), int):
                out.add(j['mode_id'])
            for g in j.get('mode_segments') or []:
                if isinstance(g, dict) and isinstance(g.get('mode_id'), int):
                    out.add(g['mode_id'])
    return out


def undo(backup, data_dir, force=False):
    man = read_json(Path(backup) / MANIFEST)
    if not man:
        raise SystemExit('%s に %s がありません' % (backup, MANIFEST))
    done, kept = 0, []
    items = sorted(man['files'].items(), key=lambda kv: kv[0] == 'modes.json')      # ジャンルの表は最後に戻す
    for rel, ent in items:
        p = Path(data_dir) / rel
        if not p.exists() or sha(p) != ent.get('after_sha'):
            kept.append(rel)
            continue
        if rel == 'modes.json' and not force:
            ids = {m['id'] for m in (read_json(ent['backup']) or [])}
            stray = sorted(ids_in_raw(Path(data_dir) / 'raw') - ids)
            if stray:
                kept.append('%s(記録がまだジャンル %s を使っているので戻さない。先に記録を戻すか --force)' % (rel, stray))
                continue
        tmp = str(p) + '.undo.tmp'
        shutil.copy2(ent['backup'], tmp)
        if sha(tmp) != ent['before_sha']:
            os.unlink(tmp)
            raise SystemExit('写しが壊れています: %s' % ent['backup'])
        os.replace(tmp, p)
        done += 1
    return done, kept


# ---------------------------------------------------------------- 入口
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('modes', 'plan', 'apply', 'undo'))
    ap.add_argument('--day', help='この日(YYYYMMDD)の記録だけ')
    ap.add_argument('--course', choices=('予選コース', '練習コース'), help='--day の日のコース(既定は QUALIFYING_DAYS で決める)')
    ap.add_argument('--backup', help='元のファイルの写しと manifest.json を置く場所(modes・apply・undo で要る)')
    ap.add_argument('--out', help='plan の表(JSON)の書き先')
    ap.add_argument('--data', default=str(config.DATA_DIR), help='data の場所(試験だけ)')
    ap.add_argument('--force', action='store_true', help='収集中・走行中の確かめ・undo の確かめを飛ばす(試験だけ)')
    a = ap.parse_args(argv)
    data = Path(a.data)
    mpath = data / 'modes.json'
    if a.cmd in ('modes', 'apply', 'undo') and not a.backup:
        ap.error('--backup が要ります')
    if a.course and not a.day:
        ap.error('--course は --day と一緒に使います')
    if a.cmd == 'undo':
        done, kept = undo(a.backup, data, a.force)
        print('戻した %d ファイル' % done)
        for k in kept:
            print('  戻さなかった: %s' % k)
        return 0
    cur = read_json(mpath)
    if not isinstance(cur, list):
        raise SystemExit('%s が読めません' % mpath)
    new_modes = merged_modes(cur)
    names = names_of(new_modes)
    if a.cmd == 'modes':
        why = busy(a.force)
        if why:
            raise SystemExit(why)
        if new_modes == cur:
            print('ジャンルの表はそろっています(何もしません)')
            return 0
        Writer(a.backup, data).write(mpath, new_modes, 'modes.json: 0〜8 の表(4〜7 を練習・予選コース × 人間・AI に振り直し、9〜13 を消した)')
        for m in new_modes:
            print('%2d %s' % (m['id'], m['name']))
        return 0
    pl = plan(data / 'raw', a.day, a.course)
    for line in summary(pl, names):
        print(line)
    for s in pl['skipped']:
        print('  飛ばした: %s %s' % (s.get('log') or s.get('scene'), s['why']))
    if a.out:
        write_json(a.out, pl)
    if a.cmd == 'plan':
        return 0
    if names_of(cur) != names:
        raise SystemExit('先に modes でジャンルの表をそろえてください(apply は data/modes.json を書きません)')
    why = busy(a.force)
    if why:
        raise SystemExit(why)
    w = Writer(a.backup, data)
    note = {'at': now(), 'by': 'tools/retag_genres.py',
            'rule': '4 練習コース 人間・5 練習コース AI・6 予選コース 人間・7 予選コース AI。11/6・11/8 = 予選コース、drive あり = AI'}
    n_scene, n_log = apply(pl, names, w, note)
    print('付け替えた: シーン %d・記録 %d(写しと manifest: %s)' % (n_scene, n_log, a.backup))
    return 0


if __name__ == '__main__':
    sys.exit(main())
