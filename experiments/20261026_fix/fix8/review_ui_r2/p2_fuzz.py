"""反証 8-2 回目・係 ui: でたらめな朝(1 画面と 2 画面)で、8-1 の app.js の直しのあとに次の 3 つが崩れないか(読むだけ。sim_r2.js)。
  I1 走り出した ▶ の確認の文の「見張り」「残す」の行と天井の左が、サーバーが走った値と同じ
  I2 断りが解けない形が無い: 終わりに通信を全部戻し、走行を止め、(a) 走行タブのまま 1 秒 × 3 / (b) ライブ → 走行タブに戻って 1 秒 × 2
     のあと、人が ▶ を押せて(disabled でない)「読み込んでいます」「つながっていません」で断られない
  I3 表示の問い合わせの輪が 1 本(終わりの pollDrive のタイマーが 1 つ以下)
操作: タブの行き来(素早い行き来も)・一覧の GET を止める/返す(古い順・新しい順)/失敗・status を止める/途切れ・5 秒(tick)・起き直し・
curl・スイッチ・基準の設定・ブロック・▶・■・ほかの画面の走行。出力 out/p2_fuzz.txt"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import run, blk, DI, HERE, RESET, confirm_lines  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 82
APPX = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '-' else None     # 比べる app.js(既定は本物の今の版)
WAIT = len(sys.argv) > 4 and sys.argv[4] == 'wait'   # wait: ▶ の前に その画面の 1 秒を進める(人が押すまでの間。表示の輪の 1 秒の遅れを除く)
REF = ('サーバーの設定を読み込んでいます', 'サーバーにつながっていません')


def gen(rng, two):
    pages = ['A', 'B'] if two else ['A']
    st = [['@', p, ['tab', 'drive']] for p in pages] + [['tickall'], ['tickall']]
    boot = [1000.0]
    for _ in range(rng.randint(14, 30)):
        p = rng.choice(pages)
        r = rng.random()
        if r < 0.16:
            st += [['@', p, ['tab', rng.choice(['live', 'live', 'scenes'])]], ['@', p, ['tickle', rng.choice([250, 1000])]], ['@', p, ['tab', 'drive']]]
        elif r < 0.22:
            st += [['@', p, ['tab', 'live']], ['@', p, ['tab', 'drive']]] * rng.randint(2, 3)
        elif r < 0.30:
            st.append(['hold', '/api/drive/models', rng.random() < 0.6])
        elif r < 0.35:
            st.append([rng.choice(['release', 'release1', 'releaseN']), '/api/drive/models'])
        elif r < 0.40:
            st.append(['hold', '/api/drive/status', rng.random() < 0.5])
        elif r < 0.44:
            st.append([rng.choice(['release', 'release1', 'releaseN']), '/api/drive/status'])
        elif r < 0.47:
            st.append(['models_fail', rng.random() < 0.5])
        elif r < 0.50:
            st.append(['status_fail_next', rng.randint(1, 2)])
        elif r < 0.53:
            boot[0] += 1000.0
            st += [['down_all', True], ['@', p, ['tickle', 1000]], ['reboot', boot[0], RESET], ['down_all', False]]
        elif r < 0.58:
            st.append(['api', rng.choice([{'rec_back_guard': 0}, {'rec_back_guard': 1, 'rec_hold_keep': 1}, {'ceil_left_mps': 4.2}, {'ceil_left_mps': 4.1}])])
        elif r < 0.66:
            st.append(['@', p, ['hcheck', rng.choice(['#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-speed-ceiling']), rng.random() < 0.7]])
        elif r < 0.70:
            st.append(['@', p, ['baseline']])
        elif r < 0.76:
            st.append(['@', p, ['pick', DI, blk(rng.randint(1, 6))]])
        elif r < 0.86:
            st += ([['@', p, ['tickle', 1000]]] if WAIT else []) + [['@', p, ['hstart']]]
        elif r < 0.91:
            st.append(['@', p, ['hstop']])
        elif r < 0.96:
            st.append(['@', p, ['tickle', rng.choice([250, 1000])]])
        else:
            st.append(['@', p, ['tick']])
    # 終わり: 通信を全部戻し、走行を止め、時間を進める
    st += [['hold', '/api/drive/models', False], ['hold', '/api/drive/status', False], ['release', '/api/drive/models'],
           ['release', '/api/drive/status'], ['models_fail', False], ['down_all', False], ['set_running', False]]
    for p in pages:
        st += [['@', p, ['tickle', 1000]]]
    end_a = [['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['dstate']],
             ['@', 'A', ['tlist']], ['@', 'A', ['hstart']]]
    end_b = [['@', 'A', ['hstop']], ['set_running', False], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tab', 'live']],
             ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tab', 'drive']], ['@', 'A', ['tickle', 1000]], ['@', 'A', ['tickle', 1000]],
             ['@', 'A', ['dstate']], ['@', 'A', ['tlist']], ['@', 'A', ['hstart']]]
    return pages, st + [['@', 'A', ['tab', 'drive']]] + end_a + end_b


def main():
    rng = random.Random(SEED)
    lines, bad = [], []
    n_start = n_ref = 0
    for i in range(N):
        two = i % 2 == 1
        pages, steps = gen(rng, two)
        out = run(steps, pages=pages, name='fz_%d.json' % i, **({'app': APPX} if APPX else {}))
        ss = out['steps']
        starts = [s for s in ss if s.get('op') == 'start' and s.get('sent') is not None]
        # I1
        for s in starts:
            if s.get('status') != 200:
                continue
            n_start += 1
            c = s.get('confirm') or ''
            sv = s.get('server_at_start') or {}
            g = 'リカバリーの後ろの見張り: ON' in c
            h = '中立止めのあともリカバリーを残す: ON' in c
            if g != (sv.get('rec_back_guard') == 1) or h != (sv.get('rec_hold_keep') == 1):
                bad.append('%d I1 見張り/残す 確認 %s/%s 走った %s/%s' % (i, g, h, sv.get('rec_back_guard'), sv.get('rec_hold_keep')))
            if +sv.get('speed_ceiling', 0) == 1:
                want = '左旋回中 %s m/s' % ('%g' % sv['ceil_left_mps'])
                if want not in c.replace('4.10', '4.1').replace('4.20', '4.2'):
                    cl = [x for x in c.split('\n') if '天井' in x]
                    bad.append('%d I1 天井の左 確認 %s 走った %s' % (i, cl, sv['ceil_left_mps']))
        for s in [x for x in ss if x.get('op') == 'start' and x.get('sent') is not None and x.get('pre')]:
            pr = s['pre']
            if pr.get('rg') is not None and pr['idle'] and pr['rg'] != pr['g']:
                bad.append('%d I4 戻ったあとに出した問い合わせを読まずに ▶ が送った(読んだ番号 %s・戻った番号 %s)' % (i, pr['rg'], pr['g']))
        n_ref += sum(1 for s in ss if s.get('op') == 'start' and s.get('sent') is None and not s.get('disabled')
                     and any(t.startswith(REF) for t in s.get('new_toasts', []) + []))
        # I2・I3(終わりの 2 つの ▶)
        ends = [s for s in ss if s.get('op') == 'start' and s.get('page') == 'A'][-2:]
        dst = [s for s in ss if s.get('op') == 'dstate' and s.get('page') == 'A'][-2:]
        tls = [s for s in ss if s.get('op') == 'tlist' and s.get('page') == 'A'][-2:]
        for tag, e, d, t in zip(('(a) 走行タブのまま', '(b) 戻ったあと'), ends, dst, tls):
            npoll = sum(1 for x in t['value'] if x[0] == 'pollDrive')
            refused = e.get('sent') is None
            why = 'disabled' if e.get('disabled') else (out['toasts']['A'][-1][:40] if refused and out['toasts']['A'] else '')
            if refused or npoll > 1:
                bad.append('%d I2/I3 %s: ▶ %s(%s)・pollDrive のタイマー %d・state %s' % (i, tag, '断られた' if refused else '走った', why, npoll, json.dumps(d['value'], ensure_ascii=False)))
        lines.append('%2d %s 操作 %3d・走った ▶ %d' % (i, '2 画面' if two else '1 画面', len(steps), len(starts)))
    head = ['app.js: %s(%s)・▶ の前の 1 秒: %s' % (APPX or 'jetformer/web/static/app.js(今の版)', __import__('hashlib').sha256((APPX or Path('/home/jetson/jetformer/jetformer/web/static/app.js')).read_bytes()).hexdigest()[:12], 'あり' if WAIT else 'なし'),
            'でたらめな朝 %d 本(種 %d。1 画面と 2 画面を交互)・走り出した ▶ %d 本・「読み込んでいます」「つながっていません」で断った ▶ %d 回' % (N, SEED, n_start, n_ref),
            '崩れ: %d 件(I1 %d・I2 %d・I3 %d・I4 %d)' % (len(bad), sum(' I1 ' in b for b in bad), sum('断られた' in b for b in bad), sum('タイマー' in b and not 'タイマー 1・' in b and not 'タイマー 0・' in b for b in bad), sum(' I4 ' in b for b in bad))] + ['  ' + b for b in bad]
    txt = '\n'.join(head + [''] + lines) + '\n'
    (HERE / 'out' / ('p2_fuzz_%d_%s_%s.txt' % (SEED, ('instr' if 'instr' in str(APPX) else 'before') if APPX else 'now', 'wait' if WAIT else 'raw'))).write_text(txt, encoding='utf-8')
    print('\n'.join(head))


if __name__ == '__main__':
    main()
