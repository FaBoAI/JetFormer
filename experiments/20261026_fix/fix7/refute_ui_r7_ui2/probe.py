"""反証 w1-ui-r7-ui-2(試験の台本の click・slide が長いタイマーを捨てる)を自分で確かめる(読むだけ。車・jetformer-web に触れない)。

もとの台本 tests/fixtures/app_sim_fix6_r2.js と、runShortTimers の 1 行だけ直した写し sim_keep.js(この場所。自分で作った)で、
本物の app.js(今の版・直す前の写し tests/fixtures/fix6_ui_r2_before/app.js)を node で流して比べる。

  python probe.py <一時の場所>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_ui_fix6_r2 as T  # noqa: E402

tmp = Path(sys.argv[1])
ORIG = ROOT / 'tests' / 'fixtures' / 'app_sim_fix6_r2.js'
KEEP = HERE / 'sim_keep.js'


def get_cands():                      # tests/test_ui_fix6.py の cands(pytest の fixture)と同じ
    from jetformer.web import bench
    cc = bench.load_candidates()
    days = [i for i, d in enumerate(cc.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    return dict(cc, today='2026-11-03'), days[0]


c, di = get_cands()


def run(sim, steps, app=None, name='x', server=None):
    T.SIM = sim
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    return T.run_sim(d, c, steps, app=app, server=server)


def gets(res):
    return [[x for x in v if x[1].startswith(T.ST)] for v in T.vals(res, 'calls')]


def show(label, res, toasts):
    print('  ', label)
    for x in res:
        if x['op'] in ('ntimers', 'eval', 'read', 'el', 'click', 'slide', 'tap', 'timers', 'tick'):
            print('     ', json.dumps(x, ensure_ascii=False)[:260])
        if x['op'] == 'calls':
            print('      calls:', json.dumps([[m, u.split('?')[0], s] for m, u, s in x['value']], ensure_ascii=False)[:400])


AFTER = [['ntimers'], ['eval', 'driveTimer !== null'], ['api', {'ceil_left_mps': 4.2}], ['tick'], ['tick'], ['ntimers'], ['calls'],
         ['read', 'st_params'], ['el', '#dr-ceil-spec', 'text']]
HEAD = T.BASE + [['tab', 'drive'], ['tick'], ['ntimers'], ['calls']]
SC = {
    'S1 click(#btn-dr-tg-up)': HEAD + [['click', '#btn-dr-tg-up']] + AFTER,
    'S2 slide(#dr-trim 10)': HEAD + [['slide', '#dr-trim', 10]] + AFTER,
    'S3 tap + timers(台本の中の、長いタイマーを戻す道)': HEAD + [['tap', '#btn-dr-tg-up'], ['timers']] + AFTER,
}
print('== 1. 同じ操作を、もとの台本と sim_keep.js で(今の app.js)')
for k, steps in SC.items():
    print(k)
    for lab, sim in (('もと', ORIG), ('keep', KEEP)):
        res, to = run(sim, steps, name=k[:2] + lab, server=T.PRE_B)
        st = [x['value'] for x in res if x['op'] == 'read'][-1]
        print('   [%s] click/slide の後のタイマー=%s  driveTimer!==null=%s  tick 2 回の後のタイマー=%s  その間の status GET=%d  画面の左=%s  天井の横の文=%s'
              % (lab, T.vals(res, 'ntimers')[1], T.vals(res, 'eval')[0], T.vals(res, 'ntimers')[2], len(gets(res)[-1]),
                 st.get('ceil_left_mps'), (T.vals(res, 'el')[0] or '')[:60]))

print('\n== 2. test_ui_fix6_r2 の NORMAL(停止ボタンの click のある流れ)を、今の app.js で、2 つの台本で')
tail = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
for name, steps0 in T.NORMAL.items():
    steps = [s if s[0] != 'pick' else ['pick', di, T.blk(c, di, 10)] for s in steps0]
    r1, _ = run(ORIG, steps + tail, name='n_o', server=T.PRE_A)
    r2, _ = run(KEEP, steps + tail, name='n_k', server=T.PRE_A)
    print(name)
    print('   ntimers もと=%s' % T.vals(r1, 'ntimers'))
    print('   ntimers keep=%s' % T.vals(r2, 'ntimers'))
    print('   status GET の数 もと=%s keep=%s' % ([len(g) for g in gets(r1)], [len(g) for g in gets(r2)]))
    print('   calls が同じ=%s ntimers が同じ=%s' % (T.vals(r1, 'calls') == T.vals(r2, 'calls'), T.vals(r1, 'ntimers') == T.vals(r2, 'ntimers')))

print('\n== 3. sim_keep.js で、test_normal_paths_exactly_as_before と同じ比べ(今の app.js と直す前の写し)')
for name, steps0 in T.NORMAL.items():
    steps = [s if s[0] != 'pick' else ['pick', di, T.blk(c, di, 10)] for s in steps0]
    new, tn = run(KEEP, steps + tail, name='c_n', server=T.PRE_A)
    old, to = run(KEEP, steps + tail, app=T.BEFORE / 'app.js', name='c_o', server=T.PRE_A)
    ok = (T.vals(new, 'calls') == T.vals(old, 'calls') and T.vals(new, 'ntimers') == T.vals(old, 'ntimers') and tn == to
          and T.tagged(new) == T.tagged(old) and T.vals(new, 'posts') == T.vals(old, 'posts')
          and [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in T.starts(new)]
          == [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in T.starts(old)])
    print('   %s: 今の版と直す前が同じ=%s' % (name, ok))
