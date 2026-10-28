"""打ち消しの票 A(反証 8-2 rehearsal-r2-vbefore-screen-vs-server)。同じ説明欄を本物の speed_auto.v_before_of と
本物の drive._SpeedAuto(走行ループの電圧の源の決め)に渡し、基板の電圧が無い(ファームなし)ときの電圧の源と m_ff を出す。
説明欄はサーバーで ai_no= が頭に付く(bench.with_ai_no)形にもしてみる。装置は開かない(_SpeedAuto は計算だけ)。
使い方: python vsrc_server.py <説明欄の JSON 配列のファイル>"""
import json
import sys

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import drive, speed_auto          # noqa: E402
from jetformer.web import bench as bench_web     # noqa: E402

SRC = {speed_auto.SRC_NONE: 'なし', speed_auto.SRC_BOARD: '基板', speed_auto.SRC_DESC: '説明欄'}
descs = json.load(open(sys.argv[1], encoding='utf-8'))
params = dict(drive.DEFAULT_PARAMS)
params.update(speed_ceiling=1, speed_auto=1, auto_target_mps=4.27, auto_vref_v=8.35)
for d in descs:
    d_srv = bench_web.with_ai_no(d, 7)            # サーバーが走行開始で足す形
    sa = drive._SpeedAuto(params, d_srv)
    g = sa.gain(1_000_000_000, 1.00, 0.0, [])     # 止まっている・基板の電圧のサンプルなし
    print(json.dumps({'desc': d, 'v_before_of': speed_auto.v_before_of(d), 'get_field': bench_web.get_field(d, 'V_before'),
                      'SpeedAuto.v_desc': sa.v_desc, 'src': SRC.get(sa.src, sa.src), 'v_used': sa.v_used,
                      'm_ff': sa.m_ff, 'gain': g}, ensure_ascii=False, default=str))
