"""反証 r2-rec2-4: 本物の 2 本で car_check の「衝突」の行(読むだけ。JSON は書かない)"""
import sys
sys.path.insert(0, '/home/jetson/jetformer/tools')
import car_check as cc  # noqa: E402
for log in sys.argv[1:]:
    r = cc.check_log(log)
    txt = cc.to_text(r)
    print('=====', log)
    for l in txt.split('\n'):
        if '衝突' in l:
            print(l)
    print('label 全部の 8G 以上:', [(c['t_s'], c['g'], c['ai'], c['loc']) for c in
                                   cc.ss._labels(log, cc.config.DATA_DIR / 'labels', cc.config.RAW_DIR)[0]['events']['collisions'] if c['g'] >= 8])
