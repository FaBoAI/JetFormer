"""fix8 係 safety 直す 8-1 回目: 前の案の係(fix8/proposal/ の pA・pB・pC。11/3 のあとの案)の「本物の記録 235 本で今と出す値が違う
フレームは 0」(proposal_safety_after1103.md 4.5)を、1 本の記録の場面をつないで照らす(読むだけ。案の写しは読み込むだけ)。
見張り ON(esc_rev あり)と OFF。流し方は tests/test_rec_fix8_safety_r1.py の (A)。
    python -B prev_props12.py → out/prev_props12.json
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_rec_fix8_safety_r1 as T  # noqa: E402

PSHA = {'pA': '82e2f8a584c66fe292d2122800dcacb928d93c24aba78ca53faa5331915528d3',
        'pB': '05d78b8aeaf09a64364e284191ce19259b3b0baee4d449674056f3fe6228e5b6',
        'pC': '7ac97688a93713704762dc29f3ff64d17dc2ec4b758332ec4200ffd2240585b8'}


def load(name):
    fp = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'proposal' / name / 'assist.py'
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == PSHA[name], name
    spec = importlib.util.spec_from_loader('jetformer._pp12_%s' % name, loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(compile(fp.read_text(encoding='utf-8'), str(fp), 'exec'), m.__dict__)
    return m


def main():
    mods = {'now': T.A}
    spec = {'now_on': ('now', True, True), 'now_off': ('now', False, False)}
    for p in PSHA:
        mods[p] = load(p)
        spec[p + '_on'] = (p, True, True)
        spec[p + '_off'] = (p, False, False)
    res = {p: {'diff_on': [], 'diff_off': 0} for p in PSHA}
    frames = units = 0
    for log in T.raw_logs():
        cat, meta, _ = T.load_unit(log, T.scene_dirs(log))
        if cat is None:
            continue
        r = T.assist_replay(cat, meta, mods=mods, spec=spec)
        t0 = T.log_t0_us(log) or int(r['t_us'][0])
        frames += len(r['t_us'])
        units += 1
        for p in PSHA:
            res[p]['diff_on'] += [{'log': log, 's': round((int(r['t_us'][i]) - t0) / 1e6, 3)}
                                  for i in T.diff_idx(r, p + '_on', 'now_on')]
            res[p]['diff_off'] += len(T.diff_idx(r, p + '_off', 'now_off'))
    out = {'units': units, 'frames': frames, 'props': res}
    print(json.dumps(out, ensure_ascii=False))
    (HERE / 'out' / 'prev_props12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
