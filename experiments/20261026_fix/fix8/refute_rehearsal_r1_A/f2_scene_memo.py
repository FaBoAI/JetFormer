"""指摘 rehearsal-after-memo の別の道(データのタブのシーンの説明 = scene.json)を夜の道具が「試し」と読むか(打ち消しの票 A)。
作り物の data/raw(作業場所の tmp_f2/)に 1 本だけ作り、tools/section_scorecard.effective_description と
tools/standing_3lap.trial_memo を流す。本物の data/raw は読まない・書かない。"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import section_scorecard as ss  # noqa: E402
import standing_3lap as s3  # noqa: E402

raw = HERE / 'tmp_f2' / 'raw'
if raw.exists():
    shutil.rmtree(raw)
log = '20261103-092600'
base = ('bench=real_bench_20261023 day=2026-11-03 block=8 cand=B model=m battery=#A V_before=8.31 V_after=__ check=合格 '
        'memo= AI走行 No.9')
for i, d in enumerate([base, base]):
    p = raw / log / ('%s_%02d' % (log, i))
    p.mkdir(parents=True)
    (p / 'scene.json').write_text(json.dumps({'description': d}, ensure_ascii=False), encoding='utf-8')
eff0, _ = ss.effective_description(log, base, raw)
print('直す前: trial_memo =', s3.trial_memo(eff0))
# データのタブでシーン 1 つの説明の memo= の後ろに「試し(…)」を足して保存した形(PATCH /api/scenes/<名前> は scene.json の description を書く)
sj = raw / log / ('%s_01' % log) / 'scene.json'
sj.write_text(json.dumps({'description': base.replace('memo=', 'memo=試し(起こし直しで左 4.1 のまま)')}, ensure_ascii=False),
              encoding='utf-8')
eff1, info = ss.effective_description(log, base, raw)
print('シーン 1 つに memo=試し(…): trial_memo =', s3.trial_memo(eff1), '/ edited_scenes =', info['edited_scenes'])
shutil.rmtree(HERE / 'tmp_f2')
