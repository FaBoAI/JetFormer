"""反証 7-1 回目・safety: 壁を押していた車が、リカバリーの 'brake'(後退側 = 前進のあとなのでこのESCはブレーキ)に変わった
ときの少しの戻り(バンパー・タイヤのたわみ。本物の 10/10 のリカバリーの 'brake' で eRPM 295 = 0.15 m/s が出た。
fix2/safety2/out/scan_recovery_brake_phase.json)で、rec_back_guard の back_early が 'gap'(中立)を飛ばすか。
場面 rev(前の壁 2.0 m・モデルは前進 120 µs で押し続ける)。scene['spring'] = (戻る速さ m/s, 秒, 押していた秒の下限)。
ON(両方 1)と OFF(両方 0)を同じ場面で。今の版と fix5 のあとの写し(f6b)。出力 out/exp_spring.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import first_rec, scene  # noqa: E402

N = rv7.N


def summ(res):
    sim, rec, pca, ph = res
    m = first_rec(res)
    m['starts'] = [t for t, p0, p1, *_ in ph if p0 is None and p1 == 'brake']
    m['phases'] = []
    last = None
    for t, p0, p1, *_r in ph:
        if p1 != last:
            m['phases'].append((t, p1))
            last = p1
    m['phases'] = m['phases'][:24]
    m['min_x'] = round(min(x for _, x, *_ in sim.log), 3)
    m['rev_on_t'] = next((t for t, r, c in sim.car.revlog if r), None)
    m['gap_writes'] = sum(1 for t, s, th in pca.w if abs(th - (N + 10.0)) < 1e-6)
    m['max_erpm_brake'] = None
    return m


def main():
    rv7.check_now()
    p = rv7.base_params()
    out = {}
    try:
        for spring in (None, (0.10, 0.03, 0.2), (0.14, 0.03, 0.2), (0.16, 0.02, 0.2), (0.16, 0.03, 0.2), (0.16, 0.05, 0.2),
                       (0.2, 0.03, 0.2), (0.25, 0.04, 0.2)):
            for cam0 in (0.15, 0.16, 0.17, 0.18):
                sc = scene('rev', dur=12.0)
                sc['cam0'] = cam0
                if spring is not None:
                    sc['spring'] = spring
                row = {}
                for nm, mod, pp in (('on', 'now', rv7.ON(p)), ('off', 'now', rv7.OFF(p)), ('on_f6b', 'f6b', rv7.ON(p))):
                    row[nm] = summ(rv7.run(mod, pp, sc))
                key = '%s cam0=%.2f' % (spring, cam0)
                out[key] = row
                print(key, {k: (m['back'], m['esc_push'], m['front_push'], m['giveup'], len(m['starts']), m['guards'][:2])
                            for k, m in row.items()}, flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / 'exp_spring.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
