"""反証 8-2 回目・係 ui: 11/3 の朝の流れ(画面の読み直し → 基準の設定 → 見張り → 残す → ブロック → ▶。両方 ON の日)が、8-1 の
app.js の直し(走行タブに戻った直後の断り)のあとも通るか。人は断られたら 1 秒待って押し直す(sim_r2.js の human)。
本物の app.js・index.html(node の vm)。読むだけ。出力 out/p3_morning.txt"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import run, blk, DI, HERE, RESET, confirm_lines  # noqa: E402

lines = []
KEYS = ('throttle_gain', 'steer_gain', 'lookahead', 'tof_guard', 'start_kick', 'speed_ceiling', 'speed_auto', 'launch_boost', 'steer_trim_us')


def log(*a):
    s = ' '.join(str(x) for x in a)
    lines.append(s)
    print(s)


def H(op, page='A'):
    return ['@', page, ['human', op]]


BASE = [H(['baseline']), H(['hcheck', '#dr-rec-back-guard', True]), H(['hcheck', '#dr-rec-hold-keep', True])]
BLOCK1 = [H(['pick', DI, blk(1)])]
START = [H(['hstart'])]


def case(name, steps, pages=('A',), want=(1, 1)):
    out = run(steps, pages=list(pages), name='m_%d.json' % len(lines))
    log('## ' + name)
    hs = [s for s in out['steps'] if s.get('op') == 'human']
    for h in hs:
        n_ref = sum(1 for t in h['tries'] if t['refused'])
        log('  %s %-8s 押した %d 回(断られた %d)%s' % (h['page'], h['what'], len(h['tries']), n_ref,
                                                 ('・最後のトースト ' + str(h['last'].get('toasts', [])[-1:])) if h['what'] != 'hcheck' else ''))
    st = [h['last'] for h in hs if h['what'] == 'hstart'][-1]
    sent = st.get('sent') or {}
    sv = st.get('server_at_start') or {}
    c = confirm_lines(st.get('confirm'))
    ok = st.get('status') == 200 and (sv.get('rec_back_guard'), sv.get('rec_hold_keep')) == want \
        and ('リカバリーの後ろの見張り: ON' in (st.get('confirm') or '')) == (want[0] == 1) \
        and ('中立止めのあともリカバリーを残す: ON' in (st.get('confirm') or '')) == (want[1] == 1) \
        and 'block=1 ' in (st.get('desc') or '') + ' '
    log('  ▶ 返事 %s・確認の文: %s' % (st.get('status'), ' / '.join(x[:34] for x in c)))
    log('  送った値: %s' % json.dumps({k: sent.get(k) for k in KEYS}))
    log('  説明欄: %s' % (st.get('desc') or '')[:90])
    log('  サーバーが走った値: 見張り %s・残す %s・左 %s → %s' % (sv.get('rec_back_guard'), sv.get('rec_hold_keep'), sv.get('ceil_left_mps'),
                                                     ('通った(両方 ON・確認の文と同じ・ブロック 1)' if want == (1, 1) else '期待どおり(見張り・残す OFF で走り、確認の文にも 2 行が無い = 4 章 6 で気づける)') if ok else '!! 通らない'))
    return out, ok


res = []
# V1 手順書どおり(読み直す → 走行タブ → すぐ基準の設定 …)。一覧はすぐ返る
res.append(case('V1 読み直し → 走行タブ → 基準の設定 → 見張り → 残す → ブロック 1 → ▶(一覧はすぐ返る)',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']]] + BASE + BLOCK1 + START))
# V2 走行タブを開くたびに一覧の GET が遅い(1 本目の押しのあとで返る)
res.append(case('V2 同じ流れで、走行タブを開いたときの一覧の GET が遅い(基準の設定の 1 回目のあとに返る)',
                [['reload', 'A'], ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['@', 'A', ['baseline']],
                 ['hold', '/api/drive/models', False], ['release', '/api/drive/models']] + BASE + BLOCK1 + START))
# V3 読み直し → 走行タブ(版の印)→ ライブ(センサーの開始・向きの点検)→ 走行タブに戻ってすぐ基準の設定(一覧が遅い)
res.append(case('V3 走行タブ(版の印)→ ライブ(センサー)→ 戻ってすぐ基準の設定(一覧が遅い 1 回)→ …',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']], ['tickall'], ['@', 'A', ['tab', 'live']], ['@', 'A', ['tickle', 1000]],
                 ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['@', 'A', ['human', ['baseline']]],
                 ['hold', '/api/drive/models', False], ['release', '/api/drive/models'], ['@', 'A', ['tickle', 1000]], H(['baseline']),
                 H(['hcheck', '#dr-rec-back-guard', True]), H(['hcheck', '#dr-rec-hold-keep', True])] + BLOCK1 + START))
# V4 朝の dry run(モーターに出力しない ON → ▶ → ■ → 外す)のあと
res.append(case('V4 dry run(ON → ▶ → ■ → 外す)→ 見張り → 残す → ブロック 1 → ▶',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']], H(['baseline']), ['@', 'A', ['eval', "document.querySelector('#dr-dry').checked = true; 1"]],
                 H(['hstart']), ['@', 'A', ['tickle', 250]], ['@', 'A', ['hstop']], ['@', 'A', ['tickle', 1000]],
                 ['@', 'A', ['eval', "document.querySelector('#dr-dry').checked = false; 1"]],
                 H(['hcheck', '#dr-rec-back-guard', True]), H(['hcheck', '#dr-rec-hold-keep', True])] + BLOCK1 + START))
# V5 見張り・残すのあと、ブロックの前に起き直し(7 章: 入れ直す)
res.append(case('V5 見張り・残すのあと起き直し → 知らせ → 入れ直す → ブロック 1 → ▶',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']]] + BASE + [['down_all', True], ['@', 'A', ['tickle', 1000]],
                 ['reboot', 2000.0, RESET], ['down_all', False], ['@', 'A', ['tickle', 1000]],
                 H(['hcheck', '#dr-rec-back-guard', True]), H(['hcheck', '#dr-rec-hold-keep', True])] + BLOCK1 + START))
# V5b 起き直しに気づかず(入れ直さずに)▶ → 確認の文に 2 行が無い(4 章 6 で気づく)= 正しい知らせ
res.append(case('V5b(比べ)起き直しのあと入れ直さずにブロック 1 → ▶(確認の文に 2 行が無いのが正しい)',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']]] + BASE + [['down_all', True], ['@', 'A', ['tickle', 1000]],
                 ['reboot', 2000.0, RESET], ['down_all', False], ['@', 'A', ['tickle', 1000]]] + BLOCK1 + START, want=(0, 0)))
# V6 スイッチが使えない日の API(curl)をライブのタブの間に → 戻ってブロック 1 → ▶
res.append(case('V6 ライブの間に curl で見張り・残す(手順書 3 章の API)→ 戻ってすぐブロック 1 → ▶',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']], H(['baseline']), ['@', 'A', ['tab', 'live']], ['@', 'A', ['tickle', 1000]],
                 ['api', {'rec_back_guard': 1, 'rec_hold_keep': 1}], ['@', 'A', ['tab', 'drive']]] + BLOCK1 + START))
# V7 ブロックのあと ▶ の前にライブ(カメラを見る)→ 戻ってすぐ ▶(一覧が遅い)。入れた値(pending)が残るか
res.append(case('V7 ブロック 1 のあとライブ → 戻ってすぐ ▶(一覧が遅い 1 回)',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']]] + BASE + BLOCK1 + [['@', 'A', ['tab', 'live']], ['@', 'A', ['tickle', 1000]],
                 ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['@', 'A', ['human', ['hstart']]],
                 ['hold', '/api/drive/models', False], ['release', '/api/drive/models'], ['@', 'A', ['tickle', 1000]]] + START))
# V8 2 画面: PC は走行タブを開いたまま、スマホで朝の流れ。PC はライブ → 戻る
res.append(case('V8 2 画面(PC は走行タブ・スマホ A で朝の流れ。途中で PC がライブ → 戻る)',
                [['reload', 'A'], ['@', 'B', ['tab', 'drive']], ['@', 'A', ['tab', 'drive']]] + BASE + [['@', 'B', ['tab', 'live']],
                 ['tickall'], ['@', 'B', ['tab', 'drive']]] + BLOCK1 + START, pages=('A', 'B')))
# V9 2 画面: PC で見張り・残す、スマホはライブにいて戻ってすぐブロック 1 → ▶(一覧が遅い)
res.append(case('V9 2 画面(PC B で見張り・残す、スマホ A はライブ → 戻ってすぐブロック 1 → ▶。一覧が遅い 1 回)',
                [['reload', 'A'], ['@', 'A', ['tab', 'drive']], ['@', 'B', ['tab', 'drive']], ['tickall'], H(['baseline']),
                 ['@', 'A', ['tab', 'live']], H(['hcheck', '#dr-rec-back-guard', True], 'B'), H(['hcheck', '#dr-rec-hold-keep', True], 'B'),
                 ['hold', '/api/drive/models', True], ['@', 'A', ['tab', 'drive']], ['@', 'A', ['human', ['pick', DI, blk(1)]]],
                 ['hold', '/api/drive/models', False], ['release', '/api/drive/models'], ['@', 'A', ['tickle', 1000]]] + BLOCK1 + START,
                pages=('A', 'B')))
log('')
log('通った %d / %d' % (sum(ok for _, ok in res), len(res)))
(HERE / 'out' / 'p3_morning.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
