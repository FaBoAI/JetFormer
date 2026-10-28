"""exp11 の「ON が OFF より悪い」印のある組を分ける。乱れ(跳ね・止まり・欠け・人側・段差・当たりのあとの止まり)を抜いた同じ場面
(base: 車・モデル・ESC・後ろの物・坂・跳ね返りはそのまま)の ON / OFF も流して比べる。
  A  ON は乱れなしの OFF と同じかよい(= OFF がその乱れで得をした: 中立止めでリカバリーが消え、モデルの前進が後退を早く止めた・
     そもそもリカバリーが始まらなかった など。ON は決めた手順のまま)
  B  10/23 の形(ON の出来事に back_early。壁の手前でブレーキ側 → ESC が後退を待つ)
  C  坂 + 残すの中止(hold_long)のあと中立で後ろへ転がった
  T  フレームの位相だけの違い(前の壁の押し: 当たりのフレームの eRPM で止まりの待ちが 0.5 / 0.8 秒に分かれる など)。印が前の壁の
     押し・前の当たりの速さだけで、後ろの当たり・0 m の壁押しの印が無いもの
  Z  後退に入らない ESC の 0 m・壁押しの数の違い(リカバリーの回数の違い。1 回あたりは同じ)
  D  そのほか(手で見る)
    python -I -B classify11.py <kind> [out.json]
"""
import collections
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('RS8_TAG', 'cls_' + '_'.join(sys.argv[1:2]))
sys.path.insert(0, str(HERE))
import h11  # noqa: E402
import exp11  # noqa: E402

PERTURB = ('glitch', 'drops', 'veh_skip', 'tof_skip', 'imu_skip', 'spikes', 'hit_drop', 'human')


def base_of(sd):
    return {k: v for k, v in sd.items() if k not in PERTURB}


def main(kind, out_name=None):
    rows = []
    for f in sorted(glob.glob(str(HERE / 'out' / ('exp11_%s_*.json' % kind)))):
        rows += json.loads(Path(f).read_text())['rows']
    flagged = [r for r in rows if [w for w in r['worse'] if not w.startswith('giveup')]]
    res = []
    cnt = collections.Counter()
    try:
        for r in flagged:
            sd = r['sd']
            ws = [w for w in r['worse'] if not w.startswith('giveup')]
            on = r['on']
            guards = [g[0] for g in on['guards']]
            cat = None
            if 'back_early' in guards and any(w in ws for w in ('rear_more', 'rear_faster', 'front_faster')):
                cat = 'B'
            elif sd.get('slope') and 'hold_long' in guards and abs(sd['slope']) > sd.get('coast', 1.5):
                cat = 'C'
            elif set(ws) <= {'zero_push', 'front_push_more'} and sd.get('esc') == 'norev':
                cat = 'Z'
            b = base_of(sd)
            p = exp11.params_of(sd)
            mb_on = h11.metrics(h11.run('now', h11.ON(p), exp11.build(b), wrap=False), sd.get('ai_from', 1.0))
            mb_off = h11.metrics(h11.run('now', h11.OFF(p), exp11.build(b), wrap=False), sd.get('ai_from', 1.0))
            better_than_base_off = (on['rear_n'] <= mb_off['rear_n'] and on['rear_max_v'] <= mb_off['rear_max_v'] + 0.1
                                    and on['zero_push'] <= mb_off['zero_push'])
            if cat is None and better_than_base_off and any(w in ws for w in ('rear_more', 'rear_faster', 'zero_push',
                                                                              'rear_push_more')):
                cat = 'A'
            if cat is None and set(ws) <= {'front_push_more', 'front_faster'}:
                cat = 'T'
            if cat is None:
                cat = 'D'
            cnt[cat] += 1
            row = {'cat': cat, 'worse': ws, 'sd': sd, 'on': {k: on[k] for k in ('rear', 'front_push_s', 'zero_push', 'recoveries',
                                                                                  'giveup_t')},
                   'off': {k: r['off'][k] for k in ('rear', 'front_push_s', 'zero_push', 'recoveries', 'giveup_t')},
                   'guards_on': on['guards'][:8],
                   'base_on': {k: mb_on[k] for k in ('rear', 'front_push_s', 'zero_push', 'recoveries')},
                   'base_off': {k: mb_off[k] for k in ('rear', 'front_push_s', 'zero_push', 'recoveries')}}
            res.append(row)
            print(cat, ws, sd.get('i'), sd['form'], sd['model'], sd['esc'], sd.get('rear'), 'on', on['rear'][:2],
                  'off', r['off']['rear'][:2], 'base_on', mb_on['rear'][:2], 'base_off', mb_off['rear'][:2], flush=True)
    finally:
        h11.cleanup()
    print('flagged', len(flagged), 'of', len(rows), dict(cnt))
    (HERE / 'out' / (out_name or ('classify11_%s.json' % kind))).write_text(
        json.dumps({'n_rows': len(rows), 'n_flagged': len(flagged), 'cats': dict(cnt), 'rows': res}, ensure_ascii=False,
                   default=str))


if __name__ == '__main__':
    main(*sys.argv[1:3])
