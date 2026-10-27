"""直す 6-1 回目・係 tools: 本物の記録(読むだけ)で、AI の走行(log.json に drive・dry run でない)のうち
(1) シーンが欠けている(load_log の problems)本、(2) AI 側のフレームが無い本、(3) 欠けたシーンにフレームが無い本、
(4) events.jsonl の発進の前進の start/skip がフレームの外にある本 を数える。data/raw・labels は読むだけ。書くのは out/ だけ"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF))
import label_events as le  # noqa: E402

RAW = JF / 'data' / 'raw'
LAB = JF / 'data' / 'labels'
OUT = Path(__file__).resolve().parents[1] / 'out' / 'scan_real_noai.json'


def meta_of(p):
    try:
        if (p / 'log.json').stat().st_size == 0:
            return None
        return json.loads((p / 'log.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def main():
    res = []
    for p in sorted(RAW.iterdir()):
        if not p.is_dir() or len(p.name) != 15:
            continue
        m = meta_of(p)
        dr = (m or {}).get('drive') or {}
        if not dr or dr.get('dry_run'):
            continue
        log = p.name
        L = le.load_log(log, RAW)
        pr = L['problems']
        if (LAB / (log + '.npz')).exists():
            z = dict(np.load(LAB / (log + '.npz')))
            src = 'file'
        else:
            _, z = le.label_log(log, RAW)
            src = 'computed'
        if z is None:
            res.append({'log': log, 'problems': len(pr), 'labels': 'none'})
            continue
        ai_any = bool(np.asarray(z['ai'], bool).any())
        sc_pr = sorted({x.split('/')[0] for x in pr})
        sc_idx = {s: i for i, s in enumerate(L['scenes'])}
        fs = set(np.asarray(z['scene']).tolist()) if 'scene' in z else None
        noframe = [s for s in sc_pr if fs is not None and sc_idx.get(s) not in fs]
        ev = []
        try:
            for line in (p / 'events.jsonl').read_text(encoding='utf-8').splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('kind') == 'launch_boost':
                    ev.append((e.get('event'), e.get('cam_epoch_us')))
        except OSError:
            pass
        ev_out = []
        if 't_us' in z and len(z['t_us']):
            fu = np.asarray(z['t_us'], np.int64)
            for kind, tu in ev:
                if tu is None:
                    continue
                dmin = float(np.min(np.abs(fu - int(tu)))) / 1e6
                if dmin > 0.5:
                    ev_out.append((kind, dmin))
        res.append({'log': log, 'labels': src, 'ai_any': ai_any, 'problems': len(pr), 'problem_scenes': sc_pr,
                    'problem_scenes_without_frames': noframe, 'launch_events': len(ev), 'launch_events_outside_frames': ev_out,
                    'launch_boost': (dr.get('params') or {}).get('launch_boost')})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('drive books', len(res))
    print('with problems', [r['log'] for r in res if r.get('problems')])
    print('no AI frames', [r['log'] for r in res if r.get('ai_any') is False])
    print('no AI frames AND problems', [r['log'] for r in res if r.get('ai_any') is False and r.get('problems')])
    print('problem scenes without frames', [(r['log'], r['problem_scenes_without_frames']) for r in res
                                            if r.get('problem_scenes_without_frames')])
    print('launch events', [(r['log'], r['launch_events']) for r in res if r.get('launch_events')])
    print('launch events outside frames', [(r['log'], r['launch_events_outside_frames']) for r in res
                                           if r.get('launch_events_outside_frames')])
    print('labels none', [r['log'] for r in res if r.get('labels') == 'none'])


if __name__ == '__main__':
    main()
