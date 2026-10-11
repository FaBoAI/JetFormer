from ld import *
from regime import gains_tl
for log, segs in [('20261010-172910', [(14, 48.5, '1.8/1.2'), (52.7, 67.4, '1.45/1.4'), (68.3, 79.7, '1.45/1.6'), (80.4, 94.9, '1.45/1.8'), (95.2, 102.9, '1.25/1.8'), (103, 112.7, '1.25/2.0'), (113.1, 172, '1.45/2.0')]),
                  ('20261010-173445', [(22.5, 78.4, '1.4/2.0'), (79.1, 229.2, '1.6/2.0'), (236.7, 284, '1.0/1.0')])]:
    d = load(log)
    print(log)
    for a, b, lab in segs:
        m = (d.at >= a) & (d.at < b) & d.sel
        mv = (d.vt >= a) & (d.vt < b) & d.vsel
        rs = d.raw[m, 0] - 1443; rt = d.raw[m, 1] - 1498; os_ = d.out[m, 0] - 1443; ot = d.out[m, 1] - 1498
        print('  %-9s AI %5.1fs raw steer |med| %3.0f p95|.| %3.0f  out sat %3.0f%% | raw thr med %+4.0f out thr med %+4.0f | eRPM med %4.0f p90 %4.0f' % (
            lab, m.sum() / 30, np.median(np.abs(rs)), np.percentile(np.abs(rs), 95), 100 * np.mean(np.abs(os_) >= 440), np.median(rt), np.median(ot), np.median(-d.erpm[mv]), np.percentile(-d.erpm[mv], 90)))
for log in ['20261010-152405', '20261010-152707', '20261010-150424']:
    d = D(log); mv = np.abs(d.erpm) > 300
    print(log, 'human steer |med| %3.0f p95 %3.0f sat %3.0f%% thr med %+4.0f eRPM med %4.0f p90 %4.0f' % (np.median(np.abs(d.pwm[mv, 0] - 1443)), np.percentile(np.abs(d.pwm[mv, 0] - 1443), 95), 100*np.mean(np.abs(d.pwm[mv,0]-1443) >= 440), np.median(d.pwm[mv, 1] - 1498), np.median(-d.erpm[mv]), np.percentile(-d.erpm[mv], 90)))
