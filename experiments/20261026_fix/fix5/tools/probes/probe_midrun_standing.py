"""反証 4 回目(記録と夜の集計): 走行中に設定を変えた本(手順書 7 章「走行中に OFF にした本は数えない」)を ③ standing_3lap は
数えるか。比べ: 人が memo に「試し」と書いた本(r3-r3-rec-3 で「数えない(試し)」)。

作り物の 11/3 回 4 の本(止まって 3 周・衝突なし・3 周が 25 秒以下)を 3 本:
  A ふつう / B 走行中に天井を OFF(events.jsonl の params と log.json の params_end。新しいサーバーの形)/ C memo に「試し」
③ の counted・best_standing・表の「確か」、④ by_params の 1 章・2 章を見る。本物の data/raw は読まない。作り物は消す。
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
from test_standing_3lap import write_standing_log  # noqa: E402
from test_lap_anchor import T0_US  # noqa: E402
import standing_3lap as s3  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = HERE / 'tmp_midrun'
OUT = HERE / 'out'
PRM = {'throttle_gain': 0.92, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0}
DESC = ('ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB battery=#__ V_before=__ '
        'V_after=__ check=__ official=0 memo=計画と違うので試し(元 cand=B): throttle_gain 0.92(基準 1.0)')


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    OUT.mkdir(exist_ok=True)
    raw = TMP / 'raw'
    raw.mkdir(parents=True)
    logs = {'A': '20261103-150000', 'B': '20261103-150500', 'C': '20261103-151000'}
    for i, (k, log) in enumerate(logs.items()):
        d, _, _ = write_standing_log(raw, log, dur=40.0, t_ai=1.0, t_move=1.3, ramp_s=0.5, params=PRM, run='runB',
                                     description=DESC % (20 + i, 20 + i))
        if k == 'B':          # 12 秒に天井を OFF(drive.set_params の events.jsonl と、止めたときの log.json の params_end)
            (d / 'events.jsonl').write_text(json.dumps({'kind': 'params', 'changed': {'speed_ceiling': [1, 0]},
                                                        'epoch_us': T0_US + 12_000_000}) + '\n', encoding='utf-8')
            j = json.loads((d / 'log.json').read_text(encoding='utf-8'))
            j['drive']['params_end'] = dict(PRM, speed_ceiling=0)
            (d / 'log.json').write_text(json.dumps(j), encoding='utf-8')
        if k == 'C':          # 走り終わりのメモに「試し」(POST /api/bench/after と同じ scene.json)
            from jetformer.web import bench as wb
            desc = DESC % (22, 22)
            new = wb.after_description(desc, 'C', '8.30', '7.95', 'その他', '試し(天井を入れ忘れた)')
            for sc in sorted(p for p in d.iterdir() if p.is_dir()):
                (sc / 'scene.json').write_text(json.dumps({'description': new}, ensure_ascii=False), encoding='utf-8')
    res = s3.run(list(logs.values()), raw, TMP / 'nolabels', None)
    md = s3.to_markdown(res)
    by = {r['log']: r for r in res}
    out = {}
    for k, log in logs.items():
        r = by[log]
        st = [s for s in r['standing'] if s.get('to_r1_s') is not None]
        b = s3.best_standing(r)
        out[k] = {'log': log, 'settings_text': r['settings_text'], 'midrun': [m['key'] for m in r.get('midrun') or []],
                  'trial': r.get('trial'), 'from_move_s': [s.get('from_move_s') for s in st],
                  'valid': [s.get('valid') for s in st], 'counted': [s.get('counted') for s in st],
                  'best_standing_s': b['from_move_s'] if b else None,
                  'table_row': [x for x in md.splitlines() if log in x and x.startswith('| ')][0]}
    qmd, runs, groups = qp.to_markdown(res, '20261103', None)
    out['by_params_ch1'] = {x['log']: {'settings': x['settings'], 'standing_best_s': x['standing_best_s'], 'flags': x['flags']}
                            for x in runs}
    out['by_params_ch2'] = [{'text': g['text'], 'runs': g['runs'], 'standing_n': g['standing_n'],
                             'standing_clean_le25': g['standing_clean_le25']} for g in groups]
    (OUT / 'probe_midrun_standing.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'probe_midrun_standing.md').write_text(md + '\n\n' + qmd, encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
